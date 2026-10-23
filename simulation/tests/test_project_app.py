"""Exercise the local builder's request boundary without opening a network port."""
from email.message import Message
from io import BytesIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import threading
from types import SimpleNamespace
import unittest

from lidar_project.project_app import ProjectApp, ProjectHandler


class FakeWorkflow:
    def __init__(self, *, gate=None, fail=None, outputs=()):
        self.event_callback = None
        self.gate = gate
        self.fail = fail
        self.outputs = list(outputs)
        self.entered = threading.Event()
        self.calls = []

    def task_catalog(self):
        return {'tasks': [{'id': 'render_validation', 'label': 'Scientific figures'}]}

    def status(self):
        return {'tasks': [{'id': 'render_validation', 'outputs': self.outputs}]}

    def run(self, task, *, fresh=False):
        self.calls.append((task, fresh))
        self.entered.set()
        if self.gate is not None and not self.gate.wait(3):
            raise RuntimeError('Test did not release the workflow')
        if self.fail:
            raise RuntimeError(self.fail)
        self.event_callback({'type': 'stage_reused', 'task': 'saved_data', 'label': 'Saved data'})
        self.event_callback({'type': 'stage_completed', 'task': task, 'label': 'Figures ready'})


class InMemoryHandler(ProjectHandler):
    """Use the real request methods and file streaming with in-memory I/O."""
    def __init__(self, app, path, *, headers=None, body=b''):
        self.server = SimpleNamespace(app=app, server_port=8766)
        self.path = path
        self.headers = Message()
        for key, value in (headers or {}).items():
            self.headers[key] = value
        self.rfile = BytesIO(body)
        self.wfile = BytesIO()
        self.response = None
        self.response_headers = {}

    def send_response(self, code, message=None):
        self.response = code

    def send_header(self, key, value):
        self.response_headers[key] = value

    def end_headers(self):
        pass

    def json(self):
        return json.loads(self.wfile.getvalue())


class ProjectAppTests(unittest.TestCase):
    def setUp(self):
        temporary = TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()/'project'
        self.root.mkdir()

    def app(self, **kwargs):
        workflow = FakeWorkflow(**kwargs)
        return ProjectApp(self.root, workflow), workflow

    def get(self, app, path, *, host='127.0.0.1:8766'):
        handler = InMemoryHandler(app, path, headers={'Host': host} if host else {})
        handler.do_GET()
        return handler

    def post(self, app, body, *, headers=None, path='/api/run'):
        payload = body if isinstance(body, bytes) else json.dumps(body).encode()
        request_headers = {'Host': '127.0.0.1:8766', 'Origin': 'http://127.0.0.1:8766',
                           'Content-Type': 'application/json', 'Content-Length': str(len(payload)),
                           'X-Project-Token': app.token}
        request_headers.update(headers or {})
        handler = InMemoryHandler(app, path, headers=request_headers, body=payload)
        handler.do_POST()
        return handler

    def join(self, app):
        app.worker.join(3)
        self.assertFalse(app.worker.is_alive())

    def test_only_catalogued_tasks_or_all_are_executable(self):
        app, workflow = self.app()
        for task in ('unknown', '../../tools/export_report.py', 'render_validation; arbitrary-command'):
            with self.subTest(task=task), self.assertRaises(ValueError):
                app.start(task)
        self.assertEqual(workflow.calls, [])
        self.assertIsNone(app.job)
        app.start('all', fresh=True)
        self.join(app)
        self.assertEqual(workflow.calls, [('all', True)])
        self.assertEqual(app.snapshot_job()['state'], 'complete')
        self.assertEqual(app.snapshot_job()['completed'], 2)

    def test_running_job_rejects_a_second_build(self):
        gate = threading.Event()
        app, workflow = self.app(gate=gate)
        try:
            app.start('render_validation')
            self.assertTrue(workflow.entered.wait(2))
            response = self.post(app, {'task': 'all'})
            self.assertEqual(response.response, 409)
            self.assertIn('already running', response.json()['error'])
            self.assertTrue(app.snapshot_job()['running'])
            self.assertEqual(workflow.calls, [('render_validation', False)])
        finally:
            gate.set()
            self.join(app)

    def test_failed_jobs_are_visible_and_a_later_build_can_recover(self):
        app, workflow = self.app(fail='Required saved study is incomplete')
        app.start('render_validation')
        self.join(app)
        status = self.get(app, '/api/status').json()
        self.assertFalse(status['running'])
        self.assertEqual(status['job']['state'], 'failed')
        self.assertIn('incomplete', status['job']['error'])
        self.assertIn('incomplete', (self.root/'build/app_error.log').read_text())
        workflow.fail = None
        app.start('render_validation')
        self.join(app)
        self.assertEqual(app.snapshot_job()['state'], 'complete')
        self.assertIsNone(app.snapshot_job()['error'])

    def test_host_origin_and_token_are_required_for_builds(self):
        for headers in (
                {'Host': 'example.com:8766'}, {'Host': '127.0.0.1.example.com:8766'},
                {'Origin': 'https://unrelated.example'}, {'Origin': 'null'},
                {'X-Project-Token': ''}, {'X-Project-Token': 'wrong-token'}):
            app, workflow = self.app()
            with self.subTest(headers=headers):
                response = self.post(app, {'task': 'all'}, headers=headers)
                self.assertEqual(response.response, 403)
                self.assertEqual(workflow.calls, [])
        app, _ = self.app()
        for host in ('example.com:8766', '127.0.0.1.example.com:8766', None):
            with self.subTest(host=host):
                self.assertEqual(self.get(app, '/api/status', host=host).response, 403)
        response = self.post(app, {'task': 'render_validation'},
                             headers={'Host': 'localhost:8766', 'Origin': 'http://localhost:8766'})
        self.assertEqual(response.response, 202)
        self.join(app)

    def test_request_body_cannot_supply_a_command_or_non_boolean_rebuild(self):
        invalid = [({'task': 'all', 'command': 'arbitrary'}, 400),
                   ({'task': 'all', 'fresh': 'false'}, 400),
                   ({'task': 5}, 400), (['all'], 400),
                   (b'{malformed', 400), ({'task': 'unlisted'}, 400)]
        for body, code in invalid:
            app, workflow = self.app()
            with self.subTest(body=body):
                response = self.post(app, body)
                self.assertEqual(response.response, code)
                self.assertEqual(workflow.calls, [])
        app, _ = self.app()
        self.assertEqual(self.post(app, {'task': 'all'}, headers={
            'Content-Type': 'text/plain'}).response, 415)
        self.assertEqual(self.post(app, {'task': 'all'}, headers={
            'Content-Length': '4097'}).response, 400)

    def test_artifact_urls_exclude_code_and_external_paths(self):
        outputs = [{'path': 'outputs/reports/report.pdf'}, {'path': 'data/study/summary.json'},
                   {'path': 'tools/private.py'}, {'path': 'outputs/../tools/private.py'},
                   {'path': str(self.root.parent/'outside.txt')}]
        app, _ = self.app(outputs=outputs)
        exposed = app.status()['deliverables']
        self.assertEqual(exposed[0]['url'], '/files/outputs/reports/report.pdf')
        self.assertEqual(exposed[1]['url'], '/files/data/study/summary.json')
        self.assertTrue(all('url' not in item for item in exposed[2:]))

    def test_generated_videos_are_linked_and_served(self):
        relative = 'simulalted_environments/environment_conditions.mp4'
        path = self.root/relative
        path.parent.mkdir()
        path.write_bytes(b'recorded-video-fixture')
        app, _ = self.app(outputs=[{'path': relative}])
        self.assertEqual(app.status()['deliverables'][0]['url'], '/files/'+relative)
        for route in ('/'+relative, '/files/'+relative):
            with self.subTest(route=route):
                result = self.get(app, route)
                self.assertEqual(result.response, 200)
                self.assertEqual(result.response_headers['Content-Type'], 'video/mp4')
                self.assertEqual(result.wfile.getvalue(), b'recorded-video-fixture')

    def test_static_serving_contains_decoded_paths_and_symlinks(self):
        for relative, content in (('index.html', '<title>Project</title>'),
                                  ('assets/main.js', 'project code'),
                                  ('outputs/reports/result.txt', 'generated result'),
                                  ('data/study/summary.json', '{"complete": true}'),
                                  ('tools/private.py', 'private source')):
            path = self.root/relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        outside = self.root.parent/'outside.txt'
        outside.write_text('outside project')
        (self.root/'outputs/external.txt').symlink_to(outside)
        (self.root/'outputs/internal-code.py').symlink_to(self.root/'tools/private.py')
        app, _ = self.app()
        for route in ('/', '/assets/main.js', '/files/outputs/reports/result.txt',
                      '/outputs/reports/result.txt',
                      '/files/data/study/summary.json'):
            with self.subTest(route=route):
                self.assertEqual(self.get(app, route).response, 200)
        for route in ('/files/tools/private.py', '/files/outputs/../tools/private.py',
                      '/files/outputs/%2e%2e/tools/private.py',
                      '/files/outputs/../../outside.txt', '/files/outputs/external.txt',
                      '/files/outputs/internal-code.py', '/assets/../tools/private.py'):
            with self.subTest(route=route):
                self.assertEqual(self.get(app, route).response, 403)
        for route in ('/outputs/../tools/private.py', '/outputs/%2e%2e/tools/private.py',
                      '/outputs/external.txt', '/outputs/internal-code.py'):
            with self.subTest(route=route):
                self.assertEqual(self.get(app, route).response, 403)
        self.assertEqual(self.get(app, '/files/outputs/not-generated.pdf').response, 404)
        self.assertEqual(self.get(app, '/').response_headers['X-Content-Type-Options'], 'nosniff')
        self.assertIn("frame-ancestors 'none'", self.get(app, '/').response_headers[
            'Content-Security-Policy'])


if __name__ == '__main__':
    unittest.main()
