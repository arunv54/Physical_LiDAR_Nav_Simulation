"""Run controlled studies without importing visualization libraries."""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import platform
import time

import numpy as np


def run_evaluation(directory, part="all"):
    """Save measurements, independent references, and reproducibility metadata."""
    if part not in {"all", "geometry", "response"}:
        raise ValueError("part must be all, geometry, or response")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "evaluation.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    manifest = {
        "format_version": 1,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "studies": dict(previous.get("studies", {})),
        "interpretation": "Consistency of a virtual sensor with analytical model assumptions; not physical sensor calibration.",
    }
    if part in {"all", "geometry"}:
        from .geometry import run_geometry
        start = time.perf_counter()
        summary = run_geometry(directory / "geometry")
        manifest["studies"]["geometry"] = {"summary": summary, "elapsed_s": time.perf_counter() - start}
    if part in {"all", "response"}:
        from .response import run_response
        start = time.perf_counter()
        summary = run_response(directory / "response")
        manifest["studies"]["response"] = {"summary": summary, "elapsed_s": time.perf_counter() - start}
    manifest["finished_utc"] = datetime.now(timezone.utc).isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    return manifest
