"""Loopback-only report reader and optional fixed deliverable workflows."""
from __future__ import annotations

import argparse
from collections import deque
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import html
import json
import mimetypes
from pathlib import Path
import secrets
import re
import threading
import traceback
from urllib.parse import quote, unquote, urlsplit
import webbrowser


def guide_html(source):
    """Render the maintained guide's small Markdown subset without dependencies."""
    def inline(text):
        text = html.escape(text)
        text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
        return re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    blocks, paragraph, code, rows = [], [], None, []
    def flush():
        if paragraph:
            blocks.append("<p>"+inline(" ".join(paragraph))+"</p>")
            paragraph.clear()
        if rows:
            header, *body = rows
            body = [row for row in body if not all(re.fullmatch(r"[:\- ]+", cell) for cell in row)]
            blocks.append("<div class='table'><table><thead><tr>"+"".join("<th>"+inline(cell)+"</th>" for cell in header)+"</tr></thead><tbody>"+
                          "".join("<tr>"+"".join("<td>"+inline(cell)+"</td>" for cell in row)+"</tr>" for row in body)+"</tbody></table></div>")
            rows.clear()
    for line in source.splitlines():
        if line.startswith("```"):
            flush()
            if code is None:
                code = []
            else:
                blocks.append("<pre><code>"+html.escape("\n".join(code))+"</code></pre>")
                code = None
        elif code is not None:
            code.append(line)
        elif line.startswith("|"):
            if paragraph:
                flush()
            rows.append([cell.strip() for cell in line.strip("|").split("|")])
        elif not line.strip():
            flush()
        elif re.match(r"^#{1,3} ", line):
            flush()
            level = len(line)-len(line.lstrip("#"))
            blocks.append(f"<h{level}>"+inline(line[level+1:])+f"</h{level}>")
        else:
            if rows or re.match(r"^\d+\. ", line):
                flush()
            paragraph.append(line.strip())
    flush()
    return ("<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<title>LiDAR project setup guide</title><style>body{font:16px/1.65 system-ui,sans-serif;color:#192c3c;background:#f5f6f6;margin:0}main{max-width:980px;margin:auto;padding:30px 24px 70px}h1{font:36px Georgia,serif}h2{margin-top:36px;font-size:24px}a{color:#163d59}p{max-width:85ch}pre{background:white;padding:18px;overflow:auto;border:1px solid #dce1e4}code{font-size:13px;background:#edf3f6;padding:2px 4px}pre code{background:none;padding:0}.table{overflow:auto}table{border-collapse:collapse;width:100%;background:white}th,td{text-align:left;border-bottom:1px solid #dce1e4;padding:9px;font-size:14px}</style>"
            "<main><a href='/'>Return to the visual report</a>"+"".join(blocks)+"</main></html>")


class ProjectApp:
    def __init__(self, root: Path, workflow):
        self.root = Path(root).resolve()
        self.workflow = workflow
        self.token = secrets.token_urlsafe(32)
        self.lock = threading.RLock()
        self.job = None
        self.logs = deque(maxlen=2000)
        self.worker = None

    def _event(self, event):
        with self.lock:
            message = event.get("line") or event.get("message") or event.get("label") or event.get("type", "")
            self.logs.append(str(message))
            if self.job:
                if event.get("task"):
                    self.job["current_task"] = event["task"]
                if event.get("label"):
                    self.job["label"] = event["label"]
                if event.get("type") in ("stage_completed", "stage_reused"):
                    self.job["completed"] += 1
                    self.job["progress"]["completed"] = self.job["completed"]

    def snapshot_job(self):
        with self.lock:
            if not self.job:
                return {"running": False, "state": "idle", "log": []}
            return {**self.job, "log": list(self.logs)}

    def start(self, task, fresh=False):
        catalog = self.workflow.task_catalog()
        if isinstance(catalog, dict):
            catalog = catalog.get("tasks", [])
        if task != "all" and task not in {row["id"] for row in catalog}:
            raise ValueError("Choose a listed deliverable to generate")
        dependencies = {row["id"]: row.get("dependencies", []) for row in catalog}
        planned = set()
        def include(name):
            if name in planned:
                return
            planned.add(name)
            for dependency in dependencies.get(name, []):
                include(dependency)
        for name in (dependencies if task == "all" else [task]):
            include(name)
        with self.lock:
            if self.job and self.job["running"]:
                raise RuntimeError("A build is already running. Wait for it to finish.")
            self.logs.clear()
            self.job = {"id": secrets.token_hex(6), "running": True, "state": "running",
                        "task": task, "fresh": fresh, "current_task": None,
                        "label": "Preparing deliverables", "completed": 0,
                        "progress": {"completed": 0, "total": len(planned)},
                        "error": None, "started_at": datetime.now(timezone.utc).isoformat()}
            self.worker = threading.Thread(target=self._run, args=(task, fresh), daemon=True)
            self.worker.start()
            return self.snapshot_job()

    def _run(self, task, fresh):
        try:
            self.workflow.event_callback = self._event
            self.workflow.run(task, fresh=fresh)
        except Exception as exc:
            with self.lock:
                self.logs.append(str(exc))
                self.job.update(running=False, state="failed", error=str(exc), label="Build needs attention")
            diagnostic = self.root / "build" / "app_error.log"
            diagnostic.parent.mkdir(parents=True, exist_ok=True)
            diagnostic.write_text(traceback.format_exc(), encoding="utf-8")
        else:
            with self.lock:
                self.job.update(running=False, state="complete", label="Deliverables are ready")

    def status(self):
        tasks = self.workflow.status()
        if isinstance(tasks, dict):
            tasks = tasks.get("tasks", [])
        exposed = []
        for task in tasks:
            outputs = []
            for output in task.get("outputs", []):
                item = dict(output)
                candidate = Path(item.get("path", ""))
                if not candidate.is_absolute():
                    candidate = self.root / candidate
                try:
                    relative = candidate.resolve().relative_to(self.root)
                    if relative.parts[0] in ("outputs", "data", "simulalted_environments"):
                        item["url"] = "/files/" + quote(relative.as_posix(), safe="/")
                except (ValueError, IndexError):
                    pass
                outputs.append(item)
            exposed.append({**task, "outputs": outputs})
        job = self.snapshot_job()
        return {"project": "3D LiDAR midterm project", "running": job["running"],
                "job": job, "tasks": exposed, "csrf_token": self.token,
                "deliverables": [output for row in exposed for output in row["outputs"]]}


class ProjectServer(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, address, app):
        self.app = app
        super().__init__(address, ProjectHandler)


class ProjectHandler(BaseHTTPRequestHandler):
    server_version = "LiDARProject/1.0"

    def log_message(self, format, *args):
        # The workflow log is shown in the HTML interface; ordinary polling is quiet.
        pass

    def _host_allowed(self):
        host = self.headers.get("Host", "")
        return host in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}

    def _headers(self, content_type, size=None, dashboard=False, visual_report=False):
        self.send_header("Content-Type", content_type)
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("Cache-Control", "no-store")
        if size is not None:
            self.send_header("Content-Length", str(size))
        if dashboard:
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")
        elif visual_report:
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval'; style-src 'self' 'unsafe-inline'; img-src 'self' data: blob:; worker-src blob:; connect-src 'none'; font-src 'self' data:; object-src 'none'; base-uri 'none'; frame-ancestors 'none'")

    def _json(self, value, code=200):
        data = json.dumps(value, allow_nan=False).encode("utf-8")
        self.send_response(code)
        self._headers("application/json; charset=utf-8", len(data))
        self.end_headers()
        self.wfile.write(data)

    def _file(self, path, dashboard=False, visual_report=False):
        if not path.is_file():
            return self._json({"error": "This file has not been generated yet."}, 404)
        mime = mimetypes.guess_type(str(path))[0] or "application/octet-stream"
        self.send_response(200)
        self._headers(mime + ("; charset=utf-8" if mime.startswith("text/") else ""), path.stat().st_size, dashboard, visual_report)
        if path.suffix.lower() in (".zip", ".docx", ".npz"):
            self.send_header("Content-Disposition", f'attachment; filename="{path.name}"')
        self.end_headers()
        try:
            with path.open("rb") as stream:
                while block := stream.read(1024 * 1024):
                    self.wfile.write(block)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        if not self._host_allowed():
            return self._json({"error": "Use the local project address."}, 403)
        path = unquote(urlsplit(self.path).path)
        if "\x00" in path:
            return self._json({"error": "Invalid project path."}, 400)
        if path in ("/", "/index.html"):
            return self._file(self.server.app.root / "index.html", visual_report=True)
        if path in ("/builder", "/builder.html"):
            return self._file(self.server.app.root / "builder.html", dashboard=True)
        if path == "/README.md":
            return self._file(self.server.app.root / "README.md")
        if path == "/guide":
            data = guide_html((self.server.app.root / "README.md").read_text(encoding="utf-8")).encode("utf-8")
            self.send_response(200)
            self._headers("text/html; charset=utf-8", len(data), dashboard=True)
            self.end_headers()
            self.wfile.write(data)
            return
        if path == "/api/status":
            try:
                return self._json(self.server.app.status())
            except Exception as exc:
                return self._json({"error": str(exc)}, 500)
        if path == "/api/job":
            return self._json(self.server.app.snapshot_job())
        if path.startswith(("/files/", "/outputs/", "/simulalted_environments/")):
            relative = Path(path.removeprefix("/files/") if path.startswith("/files/") else path.lstrip("/"))
            if not relative.parts or relative.parts[0] not in ("data", "outputs", "simulalted_environments"):
                return self._json({"error": "Choose a generated project file."}, 403)
            candidate = (self.server.app.root / relative).resolve()
            try:
                candidate.relative_to((self.server.app.root / relative.parts[0]).resolve())
            except ValueError:
                return self._json({"error": "File is outside this project."}, 403)
            return self._file(candidate)
        if path.startswith("/assets/"):
            assets = (self.server.app.root / "assets").resolve()
            candidate = (self.server.app.root / path.lstrip("/")).resolve()
            try:
                candidate.relative_to(assets)
            except ValueError:
                return self._json({"error": "Unknown asset."}, 403)
            return self._file(candidate)
        return self._json({"error": "Unknown project address."}, 404)

    def do_POST(self):
        if not self._host_allowed():
            return self._json({"error": "Use the local project address."}, 403)
        expected_origins = {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}
        if self.headers.get("Origin") not in (None, *expected_origins):
            return self._json({"error": "Builds must be started from the project page."}, 403)
        if not secrets.compare_digest(self.headers.get("X-Project-Token", ""), self.server.app.token):
            return self._json({"error": "Refresh the project page before starting a build."}, 403)
        if urlsplit(self.path).path != "/api/run":
            return self._json({"error": "Unknown project action."}, 404)
        if self.headers.get("Content-Type", "").split(";")[0] != "application/json":
            return self._json({"error": "Expected a JSON build request."}, 415)
        try:
            size = int(self.headers.get("Content-Length", "0"))
            if not 0 < size <= 4096:
                raise ValueError("Invalid build request size")
            body = json.loads(self.rfile.read(size))
            if not isinstance(body, dict) or set(body) - {"task", "fresh"}:
                raise ValueError("Choose a listed task and rebuild option")
            task, fresh = body.get("task"), body.get("fresh", False)
            if not isinstance(task, str) or not isinstance(fresh, bool):
                raise ValueError("Invalid task or rebuild option")
            return self._json({"job": self.server.app.start(task, fresh)}, 202)
        except RuntimeError as exc:
            return self._json({"error": str(exc)}, 409)
        except (ValueError, TypeError, json.JSONDecodeError) as exc:
            return self._json({"error": str(exc)}, 400)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Open the local LiDAR visual report")
    parser.add_argument("--port", type=int, default=8766, help="Local port; 0 selects an available port")
    parser.add_argument("--no-browser", action="store_true", help="Start without opening a browser")
    args = parser.parse_args(argv)
    if not 0 <= args.port <= 65535:
        parser.error("Port must be between 0 and 65535")
    from .workflow import Workflow
    root = Path(__file__).resolve().parents[1]
    app = ProjectApp(root, Workflow(project_root=root))
    try:
        server = ProjectServer(("127.0.0.1", args.port), app)
    except OSError as exc:
        parser.exit(1, f"Cannot open the local page: {exc}. Try --port 0.\n")
    url = f"http://127.0.0.1:{server.server_port}/"
    print(f"LiDAR visual report: {url}\nKeep this window open while viewing the local report.", flush=True)
    if not args.no_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nProject page stopped.")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
