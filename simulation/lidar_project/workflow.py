"""Allowlisted, reproducible deliverable workflow used by the local dashboard.

Numerical generation and presentation remain separate CLI subprocesses. Existing
studies are audited before first reuse; later runs compare content fingerprints.
An imported study records that its original source revision was not recorded.
Changing numerical code never silently triggers an expensive replacement run.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from typing import Callable
import zipfile

from .paths import PROJECT_ROOT


class WorkflowError(RuntimeError):
    """A requested generation stage could not produce verified outputs."""


@dataclass(frozen=True)
class Task:
    id: str
    label: str
    kind: str
    dependencies: tuple[str, ...]
    args: tuple[str, ...]
    outputs: tuple[str, ...]
    sources: tuple[str, ...] = ()
    data_roots: tuple[str, ...] = ()
    description: str = ""


REPORT_CASE_IDS = (
    "baseline", "reflectance_dark", "reflectance_bright", "target_yaw_45",
    "target_yaw_75", "target_short", "target_tall", "target_near", "target_far",
    "rain", "fog", "noise_high",
)

VIDEO_CASE_IDS = (
    "baseline", "fog", "rain",
)


def _tasks() -> tuple[Task, ...]:
    """The current report: target properties and weather in one fixed scene.

    Every retained stage generates report evidence or its presentation.
    """
    sensor = ("lidar_project/model", "lidar_project/api.py", "lidar_project/config.py",
              "lidar_project/data.py", "configs/sensor.jsonc")
    motion = sensor + ("lidar_project/trajectory", "lidar_project/mapping",
                       "lidar_project/navigation/control.py", "configs/trajectory.jsonc")
    figures = tuple("outputs/all_case_report/" + name + extension
                    for name in ("target_characteristics", "weather_measurements",
                                 "weather_trajectories", "weather_profiles", "weather_maps")
                    for extension in (".png", ".pdf"))
    return (
        Task("evaluation", "Controlled geometry and response checks", "data", (),
             ("run.py", "evaluate", "--data-dir", "data/evaluation"),
             ("data/evaluation/evaluation.json", "data/evaluation/geometry/summary.json",
              "data/evaluation/geometry/frustum_geometry.npz", "data/evaluation/geometry/cases.csv",
              "data/evaluation/response/summary.json", "data/evaluation/response/trials.npz",
              "data/evaluation/response/laws.csv", "data/evaluation/response/detection.csv"),
             sensor + ("lidar_project/experiments/geometry.py", "lidar_project/experiments/response.py",
                       "lidar_project/experiments/runner.py"),
             ("data/evaluation",), "Verify analytical assumptions; includes 54,000 controlled response trials."),
        Task("dependencies", "Target and weather measurements in one scene", "data", (),
             ("run.py", "evaluate-navigation-dependencies", "--config", "configs/trajectory.jsonc",
              "--data-dir", "data/navigation_dependencies", "--cases", *REPORT_CASE_IDS),
             ("data/navigation_dependencies/summary.json", "data/navigation_dependencies/manifest.json",
              "data/navigation_dependencies/runs.csv", "data/navigation_dependencies/replay.csv",
              "data/navigation_dependencies/reference_poses_seed_1.npz",
              "data/navigation_dependencies/reference_poses_seed_2.npz",
              "data/navigation_dependencies/reference_poses_seed_3.npz"),
             motion + ("lidar_project/experiments/navigation_dependencies.py", "lidar_project/experiments/environments.py",
                       "lidar_project/experiments/trajectory.py"),
             ("data/navigation_dependencies",),
             "Nine target settings, three weather conditions and a receiver-noise check share a baseline: 12 unique conditions, three seeds each."),
        Task("sampling_reflectance", "Controlled sampling and target-reflectance study", "data", ("dependencies",),
             ("tools/evaluate_sampling_reflectance.py", "--source-dir", "data/navigation_dependencies",
              "--output-dir", "data/sampling_reflectance"),
             ("data/sampling_reflectance/summary.json", "data/sampling_reflectance/manifest.json",
              "data/sampling_reflectance/reference_poses_seed_1.npz",
              "data/sampling_reflectance/reference_poses_seed_2.npz",
              "data/sampling_reflectance/reference_poses_seed_3.npz"),
             motion + ("tools/evaluate_sampling_reflectance.py", "lidar_project/experiments/sampling_reflectance.py",
                       "lidar_project/experiments/navigation_dependencies.py", "lidar_project/experiments/environments.py",
                       "lidar_project/experiments/trajectory.py"), ("data/sampling_reflectance",),
             "Four sampling/reflectance combinations at identical replay poses; fixed mounting angles and field of view."),
        Task("threshold", "Near-threshold detection calibration", "data", ("dependencies",),
             ("run.py", "evaluate-navigation-threshold", "--source-dir", "data/navigation_dependencies",
              "--data-dir", "data/navigation_threshold", "--trials", "1000"),
             ("data/navigation_threshold/summary.json", "data/navigation_threshold/design.json", "data/navigation_threshold/settings.csv"),
             sensor + ("lidar_project/experiments/navigation_threshold.py",), ("data/navigation_threshold",),
             "33,000 independent sensor trials at three frozen viewpoints in the primary scene."),
        Task("weather_figures", "Sensor, target, weather and 3D visibility figures", "media", ("dependencies", "threshold", "sampling_reflectance"),
             ("tools/render_report_figures.py", "--data-dir", "data", "--output-dir", "outputs"),
             figures + ("outputs/weather_report/weather_fov_los.png", "outputs/weather_report/weather_fov_los.pdf",
                        "outputs/weather_report/fov_los.json", "outputs/all_case_report/manifest.json",
                        "outputs/navigation_validation/report_threshold_detection.png", "outputs/report_figures_manifest.json",
                        "outputs/sensor_dependencies/sensor_dependencies.png", "outputs/sensor_dependencies/sensor_dependencies.pdf",
                        "outputs/sensor_dependencies/weather_measurements_compact.png", "outputs/sensor_dependencies/weather_measurements_compact.pdf",
                        "outputs/sensor_dependencies/tables.csv", "outputs/sensor_dependencies/manifest.json"),
             ("tools/render_report_figures.py", "lidar_project/visualization/all_case_report.py",
              "lidar_project/visualization/weather_coverage.py", "lidar_project/visualization/navigation_validation.py",
              "lidar_project/visualization/sensor_dependencies.py"),
             description="Eight report figures from retained measurements; no navigation or sensor simulation."),
        Task("report_docx", "Editable course report", "report", ("evaluation", "weather_figures"),
             ("tools/export_report.py", "--output-dir", "outputs/reports", "--format", "docx", "--use-saved-figures"),
             ("outputs/reports/Midterm_Project_Report.docx",), ("tools/build_report.py", "tools/export_report.py")),
        Task("report_pdf", "PDF course report", "report", ("report_docx",),
             ("tools/export_report.py", "--output-dir", "outputs/reports", "--format", "pdf"),
             ("outputs/reports/Midterm_Project_Report.pdf", "outputs/reports/report_metadata.json"),
             ("tools/export_report.py",)),
        Task("report_viewer", "Offline interactive report", "media", ("report_pdf", "weather_figures", "environment_video", "sensor_state_video", "factorial_video"),
             ("tools/build_visual_report.py", "--output", "index.html"), ("index.html",),
             ("tools/build_visual_report.py", "lidar_project/visualization/report_data.py",
              "lidar_project/visualization/report_math.py"),
             description="All dependency results, equations, fixed-scene weather playback and supporting videos."),
        Task("video_inputs", "Recorded environment comparison", "data", (),
             ("run.py", "evaluate-navigation-dependencies", "--config", "configs/trajectory.jsonc",
              "--data-dir", "data/video_inputs/all", "--cases", *VIDEO_CASE_IDS),
             ("data/video_inputs/all/summary.json", "data/video_inputs/all/manifest.json",
              "data/video_inputs/all/runs.csv", "data/video_inputs/all/replay.csv"),
             motion + ("lidar_project/experiments/navigation_dependencies.py", "lidar_project/experiments/environments.py",
                       "lidar_project/experiments/trajectory.py"), ("data/video_inputs/all",),
             "Normal, Fog and Rain, three seeds each; separate inputs preserve report evidence."),
        Task("environment_video", "Normal, Fog and Rain video", "media", ("video_inputs",),
             ("tools/generate_environment_video.py", "--data-dir", "data/video_inputs/all", "--use-saved"),
             ("simulalted_environments/environment_conditions.mp4", "simulalted_environments/environment_conditions.json"),
             ("tools/generate_environment_video.py", "lidar_project/visualization/simulation_video.py"),
             description="Synchronized 3D scans and growing maps in identical obstacle geometry."),
        Task("sensor_state_video", "Fixed-angle best and worst measurement conditions", "media", ("sampling_reflectance",),
             ("tools/generate_sensor_state_video.py", "--data-dir", "data/sampling_reflectance", "--use-saved"),
             ("simulalted_environments/sensor_state_best_vs_worst.mp4", "simulalted_environments/sensor_state_best_vs_worst.json"),
             ("tools/generate_sensor_state_video.py", "lidar_project/visualization/simulation_video.py"),
             description="Dense/bright and sparse/weak target sensing at identical viewpoints and mounting angles."),
        Task("factorial_video", "Separate sampling and target-feature effects", "media", ("sampling_reflectance",),
             ("tools/generate_target_features_video.py", "--data-dir", "data/sampling_reflectance", "--use-saved"),
             ("simulalted_environments/sampling_reflectance_factorial.mp4", "simulalted_environments/sampling_reflectance_factorial.json"),
             ("tools/generate_target_features_video.py", "lidar_project/visualization/simulation_video.py"),
             description="All four combinations isolate sampling and reflectance while holding sensor state fixed."),
        Task("package", "Submission archive", "package", ("report_pdf", "report_viewer", "weather_figures", "environment_video", "sensor_state_video", "factorial_video"),
             ("tools/package_project.py", "--output", "outputs/submission/Midterm_Project_Submission.zip"),
             ("outputs/submission/Midterm_Project_Submission.zip",),
             ("tools/package_project.py", "README.md", "index.html", "start.py", "pyproject.toml", ".gitattributes", "tests", "configs", "lidar_project", "simulalted_environments")),
    )

TASKS = _tasks()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class Workflow:
    """Execute only declared stages, with array arguments and no shell evaluation."""

    def __init__(self, project_root=None, *, python=None,
                 event_callback: Callable[[dict], None] | None = None, tasks=None):
        self.project_root = Path(project_root or PROJECT_ROOT).expanduser().resolve()
        self.python = str(Path(python or sys.executable).resolve())
        self.event_callback = event_callback or (lambda event: None)
        self.tasks = {task.id: task for task in (TASKS if tasks is None else tasks)}
        self.build_dir = self.project_root / "build"
        # The narrowed target/weather report has its own adoption record. Keep
        # the former multi-study workflow's provenance unchanged on disk.
        self.manifest_path = self.build_dir / "report_manifest.json"
        self._lock = threading.Lock()
        self._manifest = {"format_version": 1, "tasks": {}}
        if self.manifest_path.exists():
            try:
                value = _read(self.manifest_path)
                if value.get("format_version") == 1:
                    self._manifest = value
            except (OSError, ValueError):
                pass

    def _path(self, relative) -> Path:
        path = (self.project_root / relative).resolve()
        if not path.is_relative_to(self.project_root):
            raise WorkflowError(f"Path escapes the project folder: {relative}")
        return path

    def task_catalog(self) -> list[dict]:
        return [dict(id=t.id, label=t.label, kind=t.kind, dependencies=list(t.dependencies),
                     description=t.description,
                     outputs=[dict(path=p, label=Path(p).name, exists=self._path(p).is_file()) for p in t.outputs])
                for t in self.tasks.values()]

    def status(self) -> list[dict]:
        """Cheap dashboard state; a run performs the complete archive/content audit."""
        rows = self.task_catalog()
        for row in rows:
            record = self._manifest.get("tasks", {}).get(row["id"])
            present = all(item["exists"] for item in row["outputs"])
            row["status"] = "ready" if present else "missing"
            row["verified"] = bool(record and record.get("status") == "complete" and present)
            row["updated_utc"] = record.get("finished_utc") if record else None
            if record and present:
                for item in record.get("outputs", []):
                    path = self._path(item["path"])
                    if not path.exists() or path.stat().st_size != item["bytes"] or path.stat().st_mtime_ns != item["mtime_ns"]:
                        row["status"], row["verified"] = "stale", False
                        break
                try:
                    fingerprint, _ = self._input_fingerprint(self.tasks[row["id"]])
                    if fingerprint != record.get("input_fingerprint"):
                        row["status"], row["verified"] = "stale", False
                except WorkflowError:
                    row["status"], row["verified"] = "stale", False
        return rows

    def _event(self, event_type, task, **kwargs):
        self.event_callback(dict(type=event_type, task=task.id, label=task.label, time_utc=_utc(), **kwargs))

    def _write_manifest(self):
        self.build_dir.mkdir(parents=True, exist_ok=True)
        self._manifest.update(project=str(self.project_root), python=self.python, updated_utc=_utc(),
                              interpretation="Model consistency and simulated task outcomes; no physical calibration claim.")
        temporary = self.manifest_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(self._manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(self.manifest_path)

    def _files(self, relative_paths) -> list[Path]:
        files = set()
        for relative in relative_paths:
            path = self._path(relative)
            if path.is_dir():
                files.update(p for p in path.rglob("*") if p.is_file() and "__pycache__" not in p.parts
                             and p.suffix not in {".pyc", ".tmp"})
            elif path.is_file():
                files.add(path)
        return sorted(files)

    def _inventory(self, paths, previous=()) -> list[dict]:
        previous = {item["path"]: item for item in previous}
        rows = []
        for path in self._files(paths):
            relative = path.relative_to(self.project_root).as_posix()
            stat = path.stat()
            old = previous.get(relative)
            digest = old["sha256"] if old and old["bytes"] == stat.st_size and old["mtime_ns"] == stat.st_mtime_ns else _sha(path)
            rows.append(dict(path=relative, bytes=stat.st_size, mtime_ns=stat.st_mtime_ns, sha256=digest))
        return rows

    def _input_fingerprint(self, task):
        sources = task.sources + (("lidar_project/visualization/default_theme.json",
                                   "lidar_project/visualization/theme.py",
                                   "lidar_project/visualization/templates",
                                   "lidar_project/visualization/static") if task.kind == "media" else ())
        inventory = self._inventory(sources)
        dependencies = {}
        for name in task.dependencies:
            record = self._manifest["tasks"].get(name)
            if not record or record.get("status") != "complete":
                raise WorkflowError(f"Dependency {name} did not complete")
            dependencies[name] = record["output_fingerprint"]
        payload = dict(command=[self.python, *task.args], outputs=list(task.outputs),
                       source_files=[(i["path"], i["sha256"]) for i in inventory], dependencies=dependencies)
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        return digest, payload

    def _owned_files(self, task):
        paths = task.data_roots if task.kind == "data" else task.outputs
        return tuple(p.relative_to(self.project_root).as_posix() for p in self._files(paths))

    def _validate(self, task):
        missing = [p for p in task.outputs if not self._path(p).is_file() or self._path(p).stat().st_size == 0]
        if missing:
            raise WorkflowError("Missing required output: " + ", ".join(missing))
        # Audit the raw archives, not only convenient summary files. CRC failures
        # or a truncated JSON block prevent adopting a study as a valid cache.
        audit_paths = list(task.data_roots or task.outputs)
        for path in self._files(audit_paths):
            if path.suffix in {".json", ".jsonc"} and path.suffix != ".jsonc":
                _read(path)
            elif path.suffix in {".npz", ".docx", ".zip"}:
                with zipfile.ZipFile(path) as archive:
                    bad = archive.testzip()
                    if bad is not None or not archive.namelist():
                        raise WorkflowError(f"Invalid archive: {path.name}")
        data = self.project_root / "data"
        if task.id == "dependencies":
            root = data / "navigation_dependencies"
            value = _read(root / "summary.json")
            required = set(REPORT_CASE_IDS)
            cases = {row["id"]: row for row in value.get("cases", [])}
            if not required.issubset(cases) or set(value.get("seeds", [])) != {1, 2, 3}:
                raise WorkflowError(f"Report study requires all {len(REPORT_CASE_IDS)} conditions and seeds 1, 2, 3")
            expected_pairs = {(case_id, seed) for case_id in required for seed in (1, 2, 3)}
            for key in ("runs", "replays"):
                rows = [row for row in value.get(key, []) if row["case_id"] in required]
                pairs = [(row["case_id"], row["seed"]) for row in rows]
                if set(pairs) != expected_pairs or len(pairs) != len(expected_pairs):
                    raise WorkflowError(f"Report study is incomplete: require {len(expected_pairs)} distinct {key}")
                for row in rows:
                    if key == "runs":
                        self._require_run(root, row["run_directory"])
                    else:
                        self._require_files(root / row["replay_directory"], ("replay.json", "traces.csv"))
            # A frozen 22-condition study is a valid superset. Its additional
            # conditions never enter the current report's target/weather plots.
            for case_id in REPORT_CASE_IDS:
                row = cases[case_id]
                self._require_run(root, row["representative_run"], mapping=True)
                self._require_files(root / row["representative_replay"], ("replay.json", "traces.csv"))
        elif task.id == "sampling_reflectance":
            root = data / "sampling_reflectance"
            value = _read(root / "summary.json")
            required = {"dense_bright", "dense_dark", "sparse_bright", "sparse_dark"}
            if {c["id"] for c in value.get("cases", [])} != required or set(value.get("seeds", [])) != {1, 2, 3}:
                raise WorkflowError("Sampling/reflectance study requires all four combinations and seeds 1, 2, 3")
            expected = {(case_id, seed) for case_id in required for seed in (1, 2, 3)}
            for key in ("runs", "replays"):
                rows = value.get(key, [])
                pairs = [(row["case_id"], row["seed"]) for row in rows]
                if set(pairs) != expected or len(pairs) != len(expected):
                    raise WorkflowError(f"Sampling/reflectance study requires twelve distinct {key}")
                for row in rows:
                    relative = row["run_directory" if key == "runs" else "replay_directory"]
                    self._require_run(root, relative, mapping=True)
                    if key == "replays":
                        self._require_files(root / relative, ("scan_targets.npz", "traces.csv", "diagnostics.json"))
        elif task.id == "threshold":
            root = data / "navigation_threshold"
            value, design = _read(root / "summary.json"), _read(root / "design.json")
            if value.get("trials_per_setting") != 1000 or len(value.get("settings", [])) != 33 or value["aggregate"]["trials"] != 33000:
                raise WorkflowError("Threshold study is incomplete; require 33 settings and 33,000 trials")
            for row in value["settings"]:
                self._require_files(root / "trials", (f"setting_{row['setting_id']:02d}.npz",))
            source = data / "navigation_dependencies"
            if design.get("source_manifest_sha256") != _sha(source / "manifest.json") or design.get("source_pose_sha256") != _sha(source / "reference_poses_seed_1.npz"):
                raise WorkflowError("Threshold design does not match the saved navigation viewpoints")
        elif task.id == "evaluation":
            value = _read(data / "evaluation/response/summary.json")
            if value["detection"]["independent_trials"] != 54000:
                raise WorkflowError("Controlled response trials are incomplete")

    def _require_files(self, directory, names):
        directory = directory.resolve()
        if not directory.is_relative_to(self.project_root):
            raise WorkflowError("Saved study references a file outside the project")
        for name in names:
            path = directory / name
            if not path.is_file() or not path.stat().st_size:
                raise WorkflowError(f"Incomplete saved study: {path.relative_to(self.project_root)}")

    def _require_run(self, root, relative, mapping=False):
        self._require_files(root / relative, ("trajectory.json", "trajectory.npz"))
        if mapping:
            self._require_files(root / relative, ("mapping.json", "mapping.npz"))

    def _backup(self, task, run_id):
        moved = []
        for relative in task.data_roots:
            source = self._path(relative)
            if source.exists():
                destination = self.build_dir / "backups" / run_id / task.id / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(destination))
                moved.append((source, destination))
        return moved

    def _restore(self, moved, run_id, task):
        for source, backup in moved:
            if source.exists():
                failed = self.build_dir / "failed" / run_id / task.id / source.relative_to(self.project_root)
                failed.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source), str(failed))
            source.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(backup), str(source))

    def _execute(self, task, log_path):
        environment = os.environ.copy()
        environment.update(PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
        environment.setdefault("MPLCONFIGDIR", str(self.build_dir / "matplotlib"))
        command = [self.python, *task.args]
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, cwd=self.project_root, env=environment,
                                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, encoding="utf-8", errors="replace", shell=False)
            assert process.stdout is not None
            with process.stdout:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    self._event("log", task, line=line.rstrip(), message=line.rstrip())
            code = process.wait()
        if code:
            raise WorkflowError(f"{task.label} failed (exit {code}); see {log_path.relative_to(self.project_root)}")

    def run(self, task_id="all", fresh=False) -> dict:
        if not isinstance(task_id, str) or (task_id != "all" and task_id not in self.tasks):
            raise WorkflowError("Unknown workflow task")
        if not isinstance(fresh, bool):
            raise WorkflowError("fresh must be true or false")
        if not self._lock.acquire(blocking=False):
            raise WorkflowError("A workflow is already running")
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
        started = time.monotonic()
        completed = set()
        visiting = set()
        results = []
        logs = self.build_dir / "logs" / run_id
        logs.mkdir(parents=True, exist_ok=True)
        try:
            def visit(name):
                if name in completed:
                    return
                if name in visiting:
                    raise WorkflowError("Workflow dependencies contain a cycle")
                visiting.add(name)
                task = self.tasks[name]
                for dependency in task.dependencies:
                    visit(dependency)
                visiting.remove(name)
                fingerprint, inputs = self._input_fingerprint(task)
                record = self._manifest["tasks"].get(name, {})
                present = all(self._path(path).is_file() for path in task.outputs)
                # Saved measurements can be adopted after the full design/raw
                # archive audit. Presentations need a recorded source match;
                # their first build regenerates rather than blessing an old view.
                should_reuse = (present and not (fresh and task.kind == "data")
                                and (task.kind == "data" or bool(record)))
                if should_reuse:
                    try:
                        self._validate(task)
                    except (OSError, ValueError, zipfile.BadZipFile, WorkflowError) as exc:
                        if task.kind == "data":
                            raise WorkflowError(f"Saved {task.label} failed validation: {exc}. Choose a fresh rebuild to replace it; prior data will be archived.") from exc
                        should_reuse = False
                if should_reuse and record and record.get("input_fingerprint") != fingerprint:
                    if task.kind == "data":
                        raise WorkflowError(f"Sources or upstream data for {task.label} changed. Choose a fresh rebuild to regenerate the numerical study.")
                    should_reuse = False
                old_inventory = record.get("outputs", [])
                owned = self._owned_files(task)
                inventory = self._inventory(owned, old_inventory) if should_reuse else []
                if should_reuse and record and [(p["path"], p["sha256"]) for p in inventory] != [(p["path"], p["sha256"]) for p in old_inventory]:
                    if task.kind == "data":
                        raise WorkflowError(f"Saved data for {task.label} changed after its recorded build. Choose a fresh rebuild after reviewing the changes.")
                    should_reuse = False
                moved = []
                self._event("stage_reused" if should_reuse else "stage_started", task,
                            message="Validated saved outputs" if should_reuse else "Generating required outputs")
                if not should_reuse:
                    if task.kind == "data":
                        moved = self._backup(task, run_id)
                    try:
                        self._execute(task, logs / (name + ".log"))
                        self._validate(task)
                    except Exception as exc:
                        self._restore(moved, run_id, task)
                        self._event("stage_failed", task, message=str(exc))
                        raise
                    inventory = self._inventory(self._owned_files(task))
                output_fingerprint = hashlib.sha256(json.dumps([(i["path"], i["sha256"]) for i in inventory]).encode()).hexdigest()
                provenance = record.get("provenance") if should_reuse and record else (
                    "imported_saved_outputs: original source revision unavailable; raw files and design audited at adoption"
                    if should_reuse else "generated_by_recorded_command")
                self._manifest["tasks"][name] = dict(status="complete", label=task.label, kind=task.kind,
                    input_fingerprint=fingerprint, inputs=inputs, output_fingerprint=output_fingerprint,
                    outputs=inventory, finished_utc=_utc(), provenance=provenance,
                    reused=should_reuse, backups=[str(b.relative_to(self.project_root)) for _, b in moved])
                self._write_manifest()
                completed.add(name)
                results.append(dict(task=name, reused=should_reuse, output_fingerprint=output_fingerprint))
                if not should_reuse:
                    self._event("stage_completed", task, message="Outputs generated and verified")

            targets = tuple(self.tasks) if task_id == "all" else (task_id,)
            for target in targets:
                visit(target)
            result = dict(task=task_id, fresh=fresh, run_id=run_id, status="complete", stages=results,
                          elapsed_s=time.monotonic() - started, manifest=str(self.manifest_path))
            self._manifest["last_run"] = result
            self._write_manifest()
            return result
        except Exception as exc:
            self._manifest["last_run"] = dict(task=task_id, fresh=fresh, run_id=run_id,
                                              status="failed", error=str(exc), stages=results)
            self._write_manifest()
            raise
        finally:
            self._lock.release()
