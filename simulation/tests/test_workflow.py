"""Exercise reproducible orchestration without rerunning expensive studies."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
import zipfile

from lidar_project.workflow import REPORT_CASE_IDS, VIDEO_CASE_IDS, TASKS, Task, Workflow, WorkflowError


class DeliverableWorkflow(unittest.TestCase):
    def dependency_fixture(self, root, *, superset=False):
        """Retained raw-file references for a fresh scope or a frozen superset."""
        directory = Path(root) / "data/navigation_dependencies"
        directory.mkdir(parents=True)
        def file(relative, value="{}"):
            path = directory / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix == ".npz":
                with zipfile.ZipFile(path, "w") as archive:
                    archive.writestr("raw.npy", b"retained archive fixture")
            else:
                path.write_text(value)
        ids = list(REPORT_CASE_IDS) + ([f"legacy_{i}" for i in range(11)] if superset else [])
        cases, runs, replays = [], [], []
        for case_id in ids:
            for seed in (1, 2, 3):
                run, replay = f"runs/{case_id}/seed_{seed}", f"replays/{case_id}/seed_{seed}"
                for name in ("trajectory.json", "trajectory.npz", "mapping.json", "mapping.npz"):
                    file(f"{run}/{name}")
                for name in ("replay.json", "traces.csv"):
                    file(f"{replay}/{name}")
                runs.append(dict(case_id=case_id, seed=seed, run_directory=run))
                replays.append(dict(case_id=case_id, seed=seed, replay_directory=replay))
            cases.append(dict(id=case_id, representative_run=f"runs/{case_id}/seed_1",
                              representative_replay=f"replays/{case_id}/seed_1"))
        summary = dict(cases=cases, seeds=[1, 2, 3], runs=runs, replays=replays)
        file("summary.json", json.dumps(summary))
        for name in ("manifest.json", "runs.csv", "replay.csv"):
            file(name)
        for seed in (1, 2, 3):
            file(f"reference_poses_seed_{seed}.npz")
        return directory, summary

    def fixture(self, root, *, fail=False):
        root = Path(root)
        (root / "worker.py").write_text(
            "from pathlib import Path\n"
            "import json,sys\n"
            "p=Path(sys.argv[1]);p.parent.mkdir(parents=True,exist_ok=True)\n"
            "p.write_text(json.dumps({'created':True}))\n"
            + ("raise SystemExit(7)\n" if fail else "print('complete')\n"), encoding="utf-8")
        source = root / "scientific_input.py"
        source.write_text("value = 1\n", encoding="utf-8")
        sample = Task("sample", "Example data", "data", (),
                      ("worker.py", "data with spaces/result.json"),
                      ("data with spaces/result.json",), ("scientific_input.py",), ("data with spaces",))
        view = Task("view", "Example media", "media", ("sample",),
                    ("worker.py", "outputs/view.json"), ("outputs/view.json",), ("worker.py",))
        return Workflow(root, python=sys.executable, tasks=(sample, view))

    def test_runs_from_any_cwd_and_reuses_unchanged_stages(self):
        with tempfile.TemporaryDirectory(prefix="lidar project ") as root:
            workflow = self.fixture(root)
            events = []
            workflow.event_callback = events.append
            first = workflow.run("view")
            self.assertEqual([r["task"] for r in first["stages"]], ["sample", "view"])
            path = Path(root) / "data with spaces/result.json"
            original_mtime = path.stat().st_mtime_ns
            second = workflow.run("all")
            self.assertTrue(all(r["reused"] for r in second["stages"]))
            self.assertEqual(path.stat().st_mtime_ns, original_mtime)
            self.assertTrue(any(e["type"] == "log" and e["line"] == "complete" for e in events))
            self.assertTrue(all(t["verified"] for t in workflow.status()))

    def test_changed_numerical_sources_require_explicit_fresh_and_keep_backup(self):
        with tempfile.TemporaryDirectory() as root:
            workflow = self.fixture(root)
            workflow.run("sample")
            (Path(root) / "scientific_input.py").write_text("value = 2\n", encoding="utf-8")
            with self.assertRaisesRegex(WorkflowError, "fresh rebuild"):
                workflow.run("sample")
            result = workflow.run("sample", fresh=True)
            self.assertFalse(result["stages"][0]["reused"])
            backups = list((Path(root) / "build/backups").rglob("result.json"))
            self.assertEqual(len(backups), 1)
            self.assertEqual(json.loads(backups[0].read_text()), {"created": True})

    def test_failed_fresh_generation_restores_previous_data_and_blocks_dependent(self):
        with tempfile.TemporaryDirectory() as root:
            workflow = self.fixture(root)
            workflow.run("sample")
            original = (Path(root) / "data with spaces/result.json").read_bytes()
            failing = self.fixture(root, fail=True)
            with self.assertRaisesRegex(WorkflowError, "exit 7"):
                failing.run("view", fresh=True)
            self.assertEqual((Path(root) / "data with spaces/result.json").read_bytes(), original)
            self.assertFalse((Path(root) / "outputs/view.json").exists())
            self.assertTrue(list((Path(root) / "build/failed").rglob("result.json")))

    def test_existing_saved_data_adoption_checks_archives_and_records_provenance(self):
        with tempfile.TemporaryDirectory() as root:
            workflow = self.fixture(root)
            directory = Path(root) / "data with spaces"
            directory.mkdir()
            (directory / "result.json").write_text('{"created":true}')
            with zipfile.ZipFile(directory / "raw.npz", "w") as archive:
                archive.writestr("sample.npy", b"kept raw measurements")
            result = workflow.run("sample")
            self.assertTrue(result["stages"][0]["reused"])
            manifest = json.loads(workflow.manifest_path.read_text())
            self.assertIn("original source revision unavailable", manifest["tasks"]["sample"]["provenance"])
            (directory / "raw.npz").write_bytes(b"truncated")
            with self.assertRaisesRegex(WorkflowError, "failed validation"):
                workflow.run("sample")

    def test_saved_data_edit_is_not_silently_adopted_again(self):
        with tempfile.TemporaryDirectory() as root:
            workflow = self.fixture(root)
            workflow.run("sample")
            (Path(root) / "data with spaces/result.json").write_text('{"created":false}')
            self.assertEqual(workflow.status()[0]["status"], "stale")
            with self.assertRaisesRegex(WorkflowError, "changed after"):
                workflow.run("sample")

    def test_existing_presentation_without_source_record_is_regenerated(self):
        with tempfile.TemporaryDirectory() as root:
            workflow = self.fixture(root)
            output = Path(root) / "outputs/view.json"
            output.parent.mkdir()
            output.write_text('{"stale":true}')
            result = workflow.run("view")
            self.assertFalse(result["stages"][-1]["reused"])
            self.assertEqual(json.loads(output.read_text()), {"created": True})

    def test_unknown_tasks_and_paths_cannot_become_shell_commands(self):
        with tempfile.TemporaryDirectory() as root:
            workflow = self.fixture(root)
            with self.assertRaisesRegex(WorkflowError, "Unknown"):
                workflow.run("sample; arbitrary command")
            with self.assertRaisesRegex(WorkflowError, "fresh must"):
                workflow.run("sample", fresh="yes")
            bad = Task("bad", "Bad path", "data", (), (), ("../outside.txt",))
            unsafe = Workflow(root, tasks=(bad,))
            with self.assertRaisesRegex(WorkflowError, "escapes"):
                unsafe.task_catalog()

    def test_active_catalog_is_only_the_single_scene_report(self):
        self.assertEqual([task.id for task in TASKS], ["evaluation", "dependencies", "sampling_reflectance", "threshold",
                         "weather_figures", "report_docx", "report_pdf", "report_viewer", "video_inputs",
                         "environment_video", "sensor_state_video", "factorial_video", "package"])
        task = next(task for task in TASKS if task.id == "dependencies")
        self.assertEqual(task.args[task.args.index("--cases") + 1:], REPORT_CASE_IDS)
        figures = next(task for task in TASKS if task.id == "weather_figures")
        # Eight embedded report images plus the separate full-size weather plot.
        self.assertEqual(sum(path.endswith(".png") for path in figures.outputs), 9)
        self.assertIn("outputs/sensor_dependencies/weather_measurements_compact.png", figures.outputs)
        videos = next(task for task in TASKS if task.id == "video_inputs")
        self.assertEqual(videos.args[videos.args.index("--cases") + 1:], VIDEO_CASE_IDS)
        self.assertNotEqual(videos.data_roots, task.data_roots)
        self.assertTrue(all("--use-saved" in task.args for task in TASKS
                            if task.id in ("environment_video", "sensor_state_video", "factorial_video")))

    def test_dependency_audit_accepts_fresh_scope_or_frozen_superset(self):
        for superset in (False, True):
            with self.subTest(superset=superset), tempfile.TemporaryDirectory() as root:
                directory, summary = self.dependency_fixture(root, superset=superset)
                workflow = Workflow(root)
                workflow._validate(workflow.tasks["dependencies"])
                summary["replays"].pop(0)
                (directory / "summary.json").write_text(json.dumps(summary))
                with self.assertRaisesRegex(WorkflowError, f"{3*len(REPORT_CASE_IDS)} distinct replays"):
                    workflow._validate(workflow.tasks["dependencies"])

    def test_dependency_audit_rejects_missing_seed_and_raw_run_file(self):
        with tempfile.TemporaryDirectory() as root:
            directory, summary = self.dependency_fixture(root, superset=True)
            workflow = Workflow(root)
            summary["seeds"] = [1, 2]
            (directory / "summary.json").write_text(json.dumps(summary))
            with self.assertRaisesRegex(WorkflowError, "seeds 1, 2, 3"):
                workflow._validate(workflow.tasks["dependencies"])
            summary["seeds"] = [1, 2, 3]
            (directory / "summary.json").write_text(json.dumps(summary))
            (directory / "runs/fog/seed_3/trajectory.npz").unlink()
            with self.assertRaisesRegex(WorkflowError, "Incomplete saved study"):
                workflow._validate(workflow.tasks["dependencies"])


if __name__ == "__main__":
    unittest.main()
