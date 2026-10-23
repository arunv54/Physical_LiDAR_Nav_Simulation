"""Portable trajectory snapshots with full raw returns in numeric format v2.

Version 1 archives remain readable as XY-observation histories; missing raw 3D
measurements are explicitly marked unavailable rather than reconstructed.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np


def _json(value):
    if isinstance(value, dict):
        return {str(k): _json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_json(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def save_result(result, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {}
    for prefix in ("states", "controls"):
        arrays.update({f"{prefix}_{key}": np.asarray(value) for key, value in result[prefix].items()})
    reference_metadata = {}
    for key, value in result["reference"].items():
        if isinstance(value, np.ndarray):
            arrays["reference_" + key] = value
        else:
            reference_metadata[key] = value
    offsets = np.zeros(len(result["observations"]) + 1, dtype=np.int64)
    for i, points in enumerate(result["observations"]):
        offsets[i + 1] = offsets[i] + len(points)
    arrays["observation_offsets"] = offsets
    arrays["observation_xy"] = np.vstack(result["observations"]) if result["observations"] else np.empty((0, 2))
    scans = result.get("scans", [])
    scans_available = bool(result.get("scans_available", False))
    if scans_available and len(scans) != len(result["controls"]["t"]):
        raise ValueError("Recorded trajectory scans must align with the K control snapshots")
    if not scans_available and scans:
        raise ValueError("Unavailable trajectory scans cannot contain invented measurements")
    scan_offsets = np.zeros(len(scans) + 1, dtype=np.int64)
    xyz, intensity = [], []
    for i, scan in enumerate(scans):
        points, strength = np.asarray(scan["xyz_world"]), np.asarray(scan["intensity"])
        if points.ndim != 2 or points.shape[1] != 3 or strength.shape != (len(points),):
            raise ValueError("Each raw scan requires xyz_world (N,3) and intensity (N,)")
        if len(points) != int(result["controls"]["n_returns"][i]):
            raise ValueError("Raw scan point count differs from its recorded return count")
        scan_offsets[i + 1] = scan_offsets[i] + len(points)
        xyz.append(points)
        intensity.append(strength)
    arrays["scan_offsets"] = scan_offsets
    arrays["scan_xyz_world"] = np.vstack(xyz) if xyz else np.empty((0, 3))
    arrays["scan_intensity"] = np.concatenate(intensity) if intensity else np.empty(0)
    if any(value.dtype.hasobject for value in arrays.values()):
        raise TypeError("Trajectory archives must contain numeric arrays only")
    np.savez_compressed(directory / "trajectory.npz", **arrays)
    metadata = {key: copy.deepcopy(result[key]) for key in
                ("name", "status", "seed", "history_recorded", "config", "sensor", "world_objects", "assumptions")}
    metadata.update(format_version=2, scans_available=scans_available, reference_metadata=reference_metadata,
                    state_fields=list(result["states"]), control_fields=list(result["controls"]),
                    reference_fields=[key for key, value in result["reference"].items() if isinstance(value, np.ndarray)])
    (directory / "trajectory.json").write_text(json.dumps(_json(metadata), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return directory


def load_result(directory):
    directory = Path(directory)
    result = json.loads((directory / "trajectory.json").read_text(encoding="utf-8"))
    version = result.pop("format_version")
    if version not in (1, 2):
        raise ValueError("Unsupported trajectory archive version")
    with np.load(directory / "trajectory.npz", allow_pickle=False) as archive:
        result["states"] = {key: archive["states_" + key].copy() for key in result.pop("state_fields")}
        result["controls"] = {key: archive["controls_" + key].copy() for key in result.pop("control_fields")}
        result["reference"] = result.pop("reference_metadata")
        result["reference"].update({key: archive["reference_" + key].copy() for key in result.pop("reference_fields")})
        offsets = archive["observation_offsets"]
        result["observations"] = [archive["observation_xy"][offsets[i]:offsets[i + 1]].copy() for i in range(len(offsets) - 1)]
        result["scans_available"] = version >= 2 and bool(result.get("scans_available", False))
        result["scans"] = []
        if result["scans_available"]:
            offsets = archive["scan_offsets"]
            xyz, intensity = archive["scan_xyz_world"], archive["scan_intensity"]
            count = len(result["controls"]["t"])
            if (not np.issubdtype(offsets.dtype, np.integer) or len(offsets) != count + 1
                    or offsets[0] != 0 or np.any(np.diff(offsets) < 0)
                    or offsets[-1] != len(xyz) or xyz.ndim != 2 or xyz.shape[1] != 3
                    or intensity.shape != (len(xyz),)):
                raise ValueError("Raw scan archive has inconsistent snapshot/point offsets")
            for i in range(count):
                start, stop = offsets[i:i + 2]
                if stop - start != int(result["controls"]["n_returns"][i]):
                    raise ValueError("Raw scan archive return counts disagree with control history")
                result["scans"].append(dict(xyz_world=xyz[start:stop].copy(), intensity=intensity[start:stop].copy()))
    return result
