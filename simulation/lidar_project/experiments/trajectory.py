"""Failure-aware metrics for the report's recorded navigation trajectories."""
from __future__ import annotations

import numpy as np


def point_polyline_distances(points, route):
    """Euclidean distances to a geometric polyline, independent of its timing."""
    points = np.asarray(points, dtype=float).reshape(-1, 2)
    route = np.asarray(route, dtype=float).reshape(-1, 2)
    if not len(route):
        raise ValueError("A reference route must contain at least one point")
    if len(route) == 1:
        return np.linalg.norm(points - route[0], axis=1)
    delta = np.diff(route, axis=0)
    squared = np.sum(delta * delta, axis=1)
    relative = points[:, None, :] - route[None, :-1, :]
    fraction = np.sum(relative * delta[None, :, :], axis=2) / np.where(squared > 0, squared, 1)
    fraction = np.clip(fraction, 0, 1)
    residual = relative - fraction[:, :, None] * delta[None, :, :]
    return np.sqrt(np.min(np.sum(residual * residual, axis=2), axis=1))



def tracking_metrics(times, positions, reference):
    """State-sampled errors, including the final state; no time weighting."""
    times = np.asarray(times, dtype=float)
    positions = np.asarray(positions, dtype=float)
    if len(times) != len(positions) or not len(times):
        raise ValueError("Trajectory times and positions must contain the same nonzero number of states")
    reference_xy = np.asarray(reference["xy"], dtype=float)
    reference_t = np.asarray(reference["t"], dtype=float)
    if len(reference_xy) != len(reference_t) or not len(reference_t):
        raise ValueError("Reference time and position arrays must have equal nonzero length")
    if np.any(np.diff(reference_t) < 0):
        raise ValueError("Reference timestamps must be nondecreasing")
    spatial_error = point_polyline_distances(positions, reference_xy)
    scheduled = np.column_stack([np.interp(times, reference_t, reference_xy[:, axis]) for axis in range(2)])
    time_error = np.linalg.norm(positions - scheduled, axis=1)
    return dict(
        spatial_route_rmse_m=float(np.sqrt(np.mean(spatial_error ** 2))),
        spatial_route_max_m=float(spatial_error.max()),
        time_reference_rmse_m=float(np.sqrt(np.mean(time_error ** 2))),
        time_reference_max_m=float(time_error.max()),
    )



def evaluate_result(result, case_id, factor, seed, lidar_enabled=True):
    """Summarize a run without dropping its terminal state or failed outcome."""
    from ..trajectory import swept_surface_gap
    from .. import api

    states, controls = result["states"], result["controls"]
    times, positions = np.asarray(states["t"]), np.asarray(states["xy"])
    omega = np.asarray(controls["omega"], dtype=float)
    return_counts = np.asarray(controls["n_returns"], dtype=float)
    if len(times) != len(omega) + 1:
        raise ValueError("A run must contain K controls and K+1 states, including its true terminal state")
    if len(return_counts) != len(omega):
        raise ValueError("Each control step must have one recorded return count")
    if np.any(np.diff(times) <= 0):
        raise ValueError("Recorded trajectory state timestamps must increase")
    active = result["config"]
    radius, height = active["robot"]["radius_m"], active["robot"]["height_m"]
    objects = result["world_objects"]
    if objects and isinstance(objects[0], dict):
        objects = api.build_scene({"objects": objects})
    gaps = [swept_surface_gap(positions[0], positions[0], objects, radius, height)]
    gaps.extend(swept_surface_gap(a, b, objects, radius, height) for a, b in zip(positions[:-1], positions[1:]))
    minimum_gap = float(min(gaps))
    status = str(result["status"])
    metric = dict(
        case_id=case_id, factor=factor, seed=int(seed), status=status,
        goal_completed=int(status == "goal"), collision=int(status == "collision"),
        timeout=int(status == "timeout"), state_count=len(times), control_steps=len(omega),
        elapsed_s=float(times[-1] - times[0]),
        path_length_m=float(np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()),
        swept_min_surface_gap_m=minimum_gap,
        initial_surface_gap_m=float(gaps[0]),
        overlap_observed=int(minimum_gap < 0.0),
        final_goal_distance_m=float(np.linalg.norm(positions[-1] - np.asarray(active["robot"]["goal"]))),
    )
    metric.update(tracking_metrics(times, positions, result["reference"]))
    metric.update(
        angular_rate_total_variation_rad_s=float(np.abs(np.diff(omega)).sum()),
        angular_rate_step_change_mean_rad_s=float(np.abs(np.diff(omega)).mean()) if len(omega) > 1 else 0.0,
        maximum_abs_commanded_angular_rate_rad_s=float(np.abs(omega).max()) if len(omega) else 0.0,
        mean_returns=float(return_counts.mean()) if len(return_counts) else 0.0,
        mean_returns_per_control_step=float(return_counts.mean()) if len(return_counts) else 0.0,
        mean_obstacle_points_per_control_step=float(np.mean(controls["n_obstacle_points"])) if len(omega) and "n_obstacle_points" in controls else 0.0,
        scans_performed=len(omega) if lidar_enabled else 0,
        lidar_enabled=int(lidar_enabled),
    )
    if "swept_surface_gap" in controls and len(omega):
        recorded = np.asarray(controls["swept_surface_gap"], dtype=float)
        independently_recomputed = np.asarray(gaps[1:])
        metric["swept_gap_recomputation_max_error_m"] = float(np.max(np.abs(recorded - independently_recomputed)))
    else:
        metric["swept_gap_recomputation_max_error_m"] = 0.0
    return metric
