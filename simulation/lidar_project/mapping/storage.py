"""Pickle-free persistence for measured-surface maps."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ARRAY_FIELDS = (
    "map_xyz", "map_intensity", "map_first_seen", "map_voxel_index",
    "ground_first_seen", "obstacle_first_seen",
)


def _validate(mapping):
    meta = mapping["metadata"]
    if meta.get("format_version") != 1:
        raise ValueError("Unsupported measured-map format version")
    size = len(mapping["map_xyz"])
    if mapping["map_xyz"].shape != (size, 3) or mapping["map_voxel_index"].shape != (size, 3):
        raise ValueError("Map positions and voxel indices must have shape(N,3)")
    if mapping["map_intensity"].shape != (size,) or mapping["map_first_seen"].shape != (size,):
        raise ValueError("Map intensity and first-seen arrays must have length N")
    shape = tuple(meta["grid_shape"])
    for key in ("ground_first_seen", "obstacle_first_seen"):
        if mapping[key].shape != shape:
            raise ValueError("First-seen raster shape does not match map metadata")
    for key in ("map_first_seen", "ground_first_seen", "obstacle_first_seen"):
        values = mapping[key]
        if not np.issubdtype(values.dtype, np.integer):
            raise ValueError("First-seen indices must have integer dtype")
        if np.any(values < (-1 if key != "map_first_seen" else 0)) or np.any(values >= meta["steps"]):
            raise ValueError("First-seen indices lie outside the recorded scan history")
    if any(np.asarray(mapping[key]).dtype.hasobject for key in ARRAY_FIELDS):
        raise TypeError("Object arrays are not permitted in measured-map archives")


def save_map(mapping, directory):
    """Save mapping.npz and mapping.json, with no rendering or simulation."""
    _validate(mapping)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(directory / "mapping.npz",
                        **{key: np.asarray(mapping[key]) for key in ARRAY_FIELDS})
    (directory / "mapping.json").write_text(
        json.dumps(mapping["metadata"], indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return directory


def load_map(directory):
    """Load exact saved arrays; object deserialization is never permitted."""
    directory = Path(directory)
    metadata = json.loads((directory / "mapping.json").read_text(encoding="utf-8"))
    with np.load(directory / "mapping.npz", allow_pickle=False) as archive:
        result = {key: archive[key].copy() for key in ARRAY_FIELDS}
    result["metadata"] = metadata
    _validate(result)
    return result
