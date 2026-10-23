"""Cumulative measured-surface mapping using already recorded world points.

Robot pose is assumed known exactly by the upstream simulator. This module
does not estimate pose, ray-trace free space, consult scene truth, or infer the
interiors of objects. All endpoint classes are observations, not traversability.
"""
from __future__ import annotations

import math
import numbers

import numpy as np


def _positive(value, name):
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")
    return value


def _limits(value, count, name):
    array = np.asarray(value, dtype=float)
    if array.shape != (count,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain {count} finite values")
    if count == 4 and not (array[0] < array[1] and array[2] < array[3]):
        raise ValueError("extent must be [xmin,xmax,ymin,ymax] with increasing bounds")
    if count == 2 and not array[0] < array[1]:
        raise ValueError("z_limits_m must have increasing bounds")
    return array


def build_map(result, voxel_size_m=0.12, resolution_m=0.2, extent=None,
              ground_threshold_m=0.12, z_limits_m=None):
    """Accumulate full recorded LiDAR endpoints, with no future-data averaging.

    ROI and optional z limits are lower-inclusive, upper-exclusive. Raster cells
    form a regular grid enclosing the requested ROI; metadata records both
    extents when the requested bounds are not multiples of the resolution.
    Voxel keys are floor(world_xyz/voxel_size_m), including negative coordinates.
    """
    if not result.get("scans_available", False):
        raise ValueError("Full recorded 3D scans are unavailable. Legacy or unrecorded trajectories cannot reconstruct a 3D measured map.")
    scans = result.get("scans", [])
    expected_steps = len(result.get("controls", {}).get("t", scans))
    if len(scans) != expected_steps:
        raise ValueError("Expected one recorded scan per control step, with no additional terminal scan")
    voxel_size_m = _positive(voxel_size_m, "voxel_size_m")
    resolution_m = _positive(resolution_m, "resolution_m")
    ground_threshold_m = float(ground_threshold_m)
    if not math.isfinite(ground_threshold_m):
        raise ValueError("ground_threshold_m must be finite")
    if extent is None:
        try:
            extent = result["config"]["planner"]["extent"]
        except KeyError as error:
            raise ValueError("Provide an explicit extent when the configuration has no planner extent") from error
    roi = _limits(extent, 4, "extent")
    z_limits = None if z_limits_m is None else _limits(z_limits_m, 2, "z_limits_m")
    nx = max(1, int(math.ceil((roi[1] - roi[0]) / resolution_m - 1e-12)))
    ny = max(1, int(math.ceil((roi[3] - roi[2]) / resolution_m - 1e-12)))
    raster_extent = [float(roi[0]), float(roi[0] + nx * resolution_m),
                     float(roi[2]), float(roi[2] + ny * resolution_m)]
    ground_first = np.full(ny * nx, -1, dtype=np.int64)
    obstacle_first = np.full(ny * nx, -1, dtype=np.int64)
    seen = set()
    point_chunks, intensity_chunks, step_chunks, voxel_chunks = [], [], [], []
    raw_count = finite_count = retained_count = nonfinite_intensity_count = 0
    for step, scan in enumerate(scans):
        points = np.asarray(scan["xyz_world"], dtype=float)
        intensity = np.asarray(scan["intensity"], dtype=float)
        if points.ndim != 2 or points.shape[1] != 3 or intensity.shape != (len(points),):
            raise ValueError("Each scan requires xyz_world shape(N,3) and intensity shape(N,)")
        raw_count += len(points)
        finite = np.isfinite(points).all(axis=1)
        finite_count += int(finite.sum())
        keep = (finite & (points[:, 0] >= roi[0]) & (points[:, 0] < roi[1])
                & (points[:, 1] >= roi[2]) & (points[:, 1] < roi[3]))
        if z_limits is not None:
            keep &= (points[:, 2] >= z_limits[0]) & (points[:, 2] < z_limits[1])
        points, intensity = points[keep], intensity[keep]
        retained_count += len(points)
        nonfinite_intensity_count += int((~np.isfinite(intensity)).sum())
        if not len(points):
            continue

        # Classify every retained measurement BEFORE 3D voxel reduction.
        columns = np.floor((points[:, 0] - roi[0]) / resolution_m).astype(np.int64)
        rows = np.floor((points[:, 1] - roi[2]) / resolution_m).astype(np.int64)
        cells = np.clip(rows, 0, ny - 1) * nx + np.clip(columns, 0, nx - 1)
        low = points[:, 2] <= ground_threshold_m
        for first_seen, selected in ((ground_first, low), (obstacle_first, ~low)):
            selected_cells = np.unique(cells[selected])
            unseen = first_seen[selected_cells] < 0
            first_seen[selected_cells[unseen]] = step

        fractional = np.floor(points / voxel_size_m)
        if np.any(np.abs(fractional) >= np.iinfo(np.int64).max):
            raise ValueError("Voxel indices exceed the supported integer range")
        voxels = fractional.astype(np.int64)
        _, first_indices = np.unique(voxels, axis=0, return_index=True)
        new_indices = []
        for index in np.sort(first_indices):
            key = tuple(voxels[index])
            if key not in seen:
                seen.add(key)
                new_indices.append(index)
        if new_indices:
            chosen = np.asarray(new_indices, dtype=np.int64)
            point_chunks.append(points[chosen].copy())
            intensity_chunks.append(intensity[chosen].copy())
            voxel_chunks.append(voxels[chosen].copy())
            step_chunks.append(np.full(len(chosen), step, dtype=np.int64))

    xyz = np.vstack(point_chunks) if point_chunks else np.empty((0, 3), dtype=float)
    intensity = np.concatenate(intensity_chunks) if intensity_chunks else np.empty(0, dtype=float)
    first_seen = np.concatenate(step_chunks) if step_chunks else np.empty(0, dtype=np.int64)
    voxel_index = np.vstack(voxel_chunks) if voxel_chunks else np.empty((0, 3), dtype=np.int64)
    observed = (ground_first >= 0) | (obstacle_first >= 0)
    metadata = dict(
        format_version=1, map_type="cumulative measured-surface endpoints",
        coordinate_frame="world", pose_assumption="World points use the simulator's known exact pose; no SLAM or pose estimation.",
        voxel_size_m=voxel_size_m, resolution_m=resolution_m,
        roi_extent_m=roi.tolist(), extent_m=raster_extent, grid_shape=[ny, nx],
        ground_threshold_m=ground_threshold_m,
        z_limits_m=None if z_limits is None else z_limits.tolist(), steps=len(scans),
        raw_return_count=raw_count, finite_return_count=finite_count,
        retained_measurement_count=retained_count, map_point_count=len(xyz),
        nonfinite_intensity_count=nonfinite_intensity_count,
        observed_cell_count=int(observed.sum()), roi_cell_count=nx * ny,
        observed_cell_fraction=float(observed.mean()),
        observed_cell_area_m2=float(observed.sum() * resolution_m ** 2),
        map_z_range_m=[float(xyz[:, 2].min()), float(xyz[:, 2].max())] if len(xyz) else None,
        semantics={
            "voxel": "Stable first observed measured representative; later observations never change its position or intensity.",
            "step": "Scan/control index k; no extra final-state scan. Negative one is empty; indices beyond the last scan hold the final map.",
            "raster": {"0": "Unknown: no measured endpoint in this cell.",
                       "1": "Observed low/ground-like endpoint at z<=threshold.",
                       "2": "Observed elevated-surface endpoint at z>threshold."},
            "precedence": "Elevated-surface evidence takes precedence only from its own first observed step onward; later ground observations do not erase it.",
            "limits": "No rays mark free or traversable space. No unseen interiors or true obstacle coverage are inferred. All measured heights, including above robot body height, are retained unless an explicit z crop is requested.",
            "area": "Observed endpoint cells divided by regular raster ROI cells; this is not traversable area or true surface coverage. For non-grid-aligned crops, the enclosing regular raster extent is recorded separately.",
        })
    return dict(map_xyz=xyz, map_intensity=intensity, map_first_seen=first_seen,
                map_voxel_index=voxel_index, ground_first_seen=ground_first.reshape(ny, nx),
                obstacle_first_seen=obstacle_first.reshape(ny, nx), metadata=metadata)


def _step(mapping, step):
    count = int(mapping["metadata"]["steps"])
    if step is None:
        return count - 1
    if not isinstance(step, numbers.Integral) or isinstance(step, (bool, np.bool_)):
        raise TypeError("step must be an integer scan index or None")
    if step < -1:
        raise ValueError("step must be -1 or a nonnegative scan index")
    return min(int(step), count - 1)


def cloud_at_step(mapping, step=None):
    """Return only stable voxel representatives observed by the selected scan."""
    visible = mapping["map_first_seen"] <= _step(mapping, step)
    return dict(xyz_world=mapping["map_xyz"][visible],
                intensity=mapping["map_intensity"][visible],
                first_seen_step=mapping["map_first_seen"][visible],
                voxel_index=mapping["map_voxel_index"][visible])


def observed_grid(mapping, step=None):
    """Reconstruct endpoint evidence: unknown=0, low=1, elevated surface=2."""
    selected = _step(mapping, step)
    low, high = mapping["ground_first_seen"], mapping["obstacle_first_seen"]
    result = np.zeros(low.shape, dtype=np.uint8)
    result[(low >= 0) & (low <= selected)] = 1
    result[(high >= 0) & (high <= selected)] = 2
    return result
