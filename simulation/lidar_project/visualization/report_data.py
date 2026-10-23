"""Compact presentation data, assembled exclusively from saved study results.

This adapter never imports the sensor, planner, controller or map builder. The
interactive report replays recorded poses and first-seen measured endpoints;
it does not create measurements or rerun an experiment when a view changes.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _source(path, project):
    path = Path(path)
    return {"path": path.relative_to(project).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "bytes": path.stat().st_size}


def _display_array(array, decimals=5):
    """Round display coordinates only; all reported numeric metrics stay exact."""
    return np.round(np.asarray(array, float), decimals).tolist()


def load_sensor_dependency_data(project):
    """Expose the fixed-angle factorial evidence without running experiments."""
    project = Path(project).resolve()
    path = project / 'data/sampling_reflectance/summary.json'
    study = _read(path)
    required = {'dense_bright', 'dense_dark', 'sparse_bright', 'sparse_dark'}
    if {row['id'] for row in study['cases']} != required:
        raise ValueError('The visual report requires all four sampling/reflectance controls')
    if set(study['seeds']) != {1, 2, 3} or len(study['replays']) != 12:
        raise ValueError('The sampling/reflectance comparison requires twelve matched-pose replays')
    primary = _read(project / 'data/navigation_dependencies/summary.json')
    noise = {row['id']: row for row in primary['cases'] if row['id'] in ('baseline', 'noise_high')}
    if len(noise) != 2:
        raise ValueError('The visual report requires the receiver-noise comparison')
    return dict(cases=study['cases'], settings=study['settings'],
                controlled_invariants=study['controlled_invariants'],
                factorial_contrasts=study['factorial_contrasts'],
                limitations=study['limitations'], noise=noise,
                source=_source(path, project))


def _mapping_payload(directory, *, max_points=16_000):
    """Read measured representatives and their original first-observation steps.

    Sampling is deterministic in stored voxel order, without modifying or
    inventing any endpoint. Full map metrics are computed from all saved voxels
    and saved first-seen endpoint cells, independently of display sampling.
    """
    directory = Path(directory)
    metadata = _read(directory / "mapping.json")
    trajectory = _read(directory / "trajectory.json")
    with np.load(directory / "mapping.npz", allow_pickle=False) as source:
        xyz = source["map_xyz"]
        first = source["map_first_seen"]
        ground = source["ground_first_seen"]
        obstacle = source["obstacle_first_seen"]
    with np.load(directory / "trajectory.npz", allow_pickle=False) as source:
        state_t, control_t = source["states_t"], source["controls_t"]
        xy, theta = source["states_xy"], source["states_theta"]
        reference = source["reference_xy"]
        command_v, command_omega = source["controls_v"], source["controls_omega"]
        returns = source["controls_n_returns"]
        controller_points = source["controls_n_obstacle_points"]
        surface_gap = source["controls_swept_surface_gap"]

    if max_points < 1:
        raise ValueError("max_points must be positive")
    if len(state_t) != len(control_t) + 1:
        raise ValueError("Recorded mapping needs one initial/final state and each control state")
    if len(first) != len(xyz) or (len(first) and (first.min() < 0 or first.max() >= len(control_t))):
        raise ValueError("Measured endpoints have invalid first-observation indices")
    if not np.allclose(control_t, state_t[:-1], atol=1e-10, rtol=0):
        raise ValueError("Stored scan/control times do not align with recorded poses")

    indices = (np.arange(len(xyz)) if len(xyz) <= max_points else
               np.linspace(0, len(xyz) - 1, max_points, dtype=np.int64))
    rounded = _display_array(xyz[indices])
    points = [point + [int(step)] for point, step in zip(rounded, first[indices])]
    curve = []
    for step, time_s in enumerate(control_t):
        cells = (((ground >= 0) & (ground <= step)) |
                 ((obstacle >= 0) & (obstacle <= step)))
        observed = int(np.count_nonzero(cells))
        curve.append({"step": step, "time_s": float(time_s),
                      "points": int(np.count_nonzero(first <= step)),
                      "observed_cells": observed,
                      "observed_fraction": observed / cells.size,
                      "returns": int(returns[step]),
                      "controller_points": int(controller_points[step])})

    dt = np.diff(state_t)
    expected_xy = xy[:-1] + dt[:, None] * command_v[:, None] * np.column_stack((np.cos(theta[:-1]), np.sin(theta[:-1])))
    expected_theta = theta[:-1] + dt * command_omega
    angle_residual = np.arctan2(np.sin(theta[1:] - expected_theta), np.cos(theta[1:] - expected_theta))
    robot = trajectory["config"]["robot"]
    motion_check = {
        "control_intervals": len(control_t),
        "max_recorded_euler_position_residual_m": float(np.max(np.linalg.norm(xy[1:] - expected_xy, axis=1), initial=0.)),
        "max_recorded_euler_heading_residual_rad": float(np.max(np.abs(angle_residual), initial=0.)),
        "speed_limit_mps": float(robot["v_max"]),
        "max_abs_commanded_speed_mps": float(np.max(np.abs(command_v), initial=0.)),
        "angular_speed_limit_rad_s": float(np.radians(robot["omega_max_deg_s"])),
        "max_abs_commanded_angular_speed_rad_s": float(np.max(np.abs(command_omega), initial=0.)),
        "scope": "Consistency of the displayed playback runs' saved states and commands only; no controller or trajectory rerun.",
    }
    return {
        "label": {"baseline": "Clear air", "rain": "Rain 50 mm/h", "fog": "Fog extinction 0.6 /m"}.get(trajectory["name"], trajectory["name"]),
        "seed": trajectory["seed"], "status": trajectory["status"],
        "times": state_t.tolist(), "scan_times": control_t.tolist(),
        "trajectory": _display_array(np.column_stack((xy, np.zeros(len(xy))))),
        "theta_rad": theta.tolist(),
        "reference": _display_array(np.column_stack((reference, np.zeros(len(reference))))),
        "points": points, "full_map_points": len(xyz), "display_map_points": len(points),
        "full_map_curve": curve,
        "map_step_at_state": [min(step, len(control_t)-1) for step in range(len(state_t))],
        "extent_m": metadata["extent_m"], "voxel_size_m": metadata["voxel_size_m"],
        "resolution_m": metadata["resolution_m"], "stats": metadata,
        "robot_radius_m": robot["radius_m"], "robot_height_m": robot["height_m"],
        "mount_height_m": trajectory["config"]["mount_height_m"],
        "sensor": trajectory["sensor"], "world_objects": trajectory["world_objects"],
        "controls": {"v_mps": command_v.tolist(), "omega_rad_s": command_omega.tolist(),
                     "swept_surface_gap_m": surface_gap.tolist()},
        "motion_check": motion_check,
        "display_sampling": "Deterministic evenly spaced original voxel indices; coordinates rounded to 5 decimals for display. Counts and curves use all saved measurements.",
        "playback_semantics": "At state k, display only measured voxels with first_step <= map_step_at_state[k]. The terminal state has no extra scan and holds the last map. Never interpolate endpoints or scan times.",
    }




def load_weather_report_data(project, *, max_map_points=16_000):
    """Present target replays and fixed-scene atmospheric comparisons.

    This intentionally omits the separate layout, mounting, FOV and sampling
    comparisons. Every profile and trajectory below is copied from retained
    weather runs or matched-pose replays, never generated by the presentation.
    """
    project = Path(project).resolve()
    folder = project / 'data/navigation_dependencies'
    primary = _read(folder / 'summary.json')
    ids = ('baseline', 'rain', 'fog')
    labels = {'baseline': 'Clear air', 'rain': 'Rain 50 mm/h', 'fog': 'Fog κ = 0.6 m⁻¹'}
    target_ids = ('baseline', 'reflectance_dark', 'reflectance_bright', 'target_yaw_45',
                  'target_yaw_75', 'target_short', 'target_tall', 'target_near', 'target_far')
    targets = [{key: row[key] for key in ('id', 'label', 'factor', 'description', 'overrides', 'replay')}
               for case_id in target_ids for row in primary['cases'] if row['id'] == case_id]
    cases = [dict(row, label=labels[row['id']]) for row in primary['cases'] if row['id'] in ids]
    cases.sort(key=lambda row: ids.index(row['id']))
    files = {'primary': folder / 'summary.json',
             'threshold': project / 'data/navigation_threshold/summary.json',
             'threshold_design': project / 'data/navigation_threshold/design.json',
             'geometry': project / 'data/evaluation/geometry/summary.json',
             'response': project / 'data/evaluation/response/summary.json'}
    files['fov'] = project / 'outputs/weather_report/fov_los.json'
    scene = _read(folder / 'runs/baseline/seed_1/trajectory.json')
    mapping, runs, replays = {}, [], []
    for case_id in ids:
        for seed in primary['seeds']:
            directory = folder / 'runs' / case_id / f'seed_{seed}'
            summary = next(row for row in primary['runs'] if row['case_id'] == case_id and row['seed'] == seed)
            metadata = _read(directory / 'trajectory.json')
            if metadata['world_objects'] != scene['world_objects']:
                raise ValueError('Weather comparison requires the same obstacle and target geometry in every run')
            for key, value in scene['sensor'].items():
                if key not in ('air_extinction_per_m', 'rain_mm_per_h') and metadata['sensor'][key] != value:
                    raise ValueError('Weather comparison unexpectedly changes a non-atmospheric sensor parameter')
            with np.load(directory / 'trajectory.npz', allow_pickle=False) as saved:
                t = saved['states_t'].copy(); xy = saved['states_xy'].copy()
                control_t = saved['controls_t'].copy()
                run = {'case_id': case_id, 'label': labels[case_id], 'seed': seed,
                       'summary': summary, 'times': t.tolist(), 'control_times': control_t.tolist(),
                       'xy': _display_array(xy), 'reference_xy': _display_array(saved['reference_xy']),
                       'route_error_m': saved['states_tracking_error'].tolist(),
                       'returns': saved['controls_n_returns'].tolist(),
                       'controller_points': saved['controls_n_obstacle_points'].tolist(),
                       'swept_gap_m': saved['controls_swept_surface_gap'].tolist(),
                       'speed_mps': saved['controls_v'].tolist(),
                       'angular_speed_rad_s': saved['controls_omega'].tolist(),
                       'path_m': np.r_[0., np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))].tolist()}
            with np.load(directory / 'mapping.npz', allow_pickle=False) as saved:
                ground, obstacle = saved['ground_first_seen'], saved['obstacle_first_seen']
                run['observed_fraction'] = [float(np.count_nonzero(
                    ((ground >= 0) & (ground <= step)) | ((obstacle >= 0) & (obstacle <= step)))) / ground.size
                    for step in range(len(control_t))]
            runs.append(run)
            for name in ('trajectory.json', 'trajectory.npz', 'mapping.npz'):
                files[f'{case_id}_{seed}_{name}'] = directory / name
            replay_path = folder / 'replays' / case_id / f'seed_{seed}' / 'traces.csv'
            with replay_path.open(newline='', encoding='utf-8') as stream:
                traces = list(csv.DictReader(stream))
            def values(key):
                return [float(row[key]) if row[key] not in ('', 'nan', 'NaN') else None for row in traces]
            replays.append({'case_id': case_id, 'label': labels[case_id], 'seed': seed,
                            'times': values('t'), 'range_rmse_m': values('range_rmse_m'),
                            'predicted_range_rmse_m': values('predicted_range_rmse_m'),
                            'eligible_power_w': values('mean_eligible_power_w'),
                            'detection_probability': values('observed_detection_probability'),
                            'detected_returns': values('detected_returns'),
                            'eligible_echoes': values('eligible_echoes'),
                            'all_returns': values('all_detected_returns')})
            files[f'{case_id}_{seed}_replay'] = replay_path
            if seed == 1:
                mapping[case_id] = _mapping_payload(directory, max_points=max_map_points)
                mapping[case_id]['source_run'] = directory.relative_to(project).as_posix()
                files[f'{case_id}_mapping_metadata'] = directory / 'mapping.json'
    result = {
        'schema_version': 2,
        'primary': {'cases': cases, 'settings': primary['settings'], 'definitions': primary['definitions'],
                    'limitations': [primary['limitations'][index] for index in (0, 1, 2, 3, 4, 8)],
                    'seeds': primary['seeds'],
                    'runs_count': len(runs), 'replays_count': len(replays), 'common_viewpoint': []},
        'weather': {'runs': runs, 'replays': replays, 'case_ids': list(ids),
                    'scene_objects': scene['world_objects'], 'sensor': scene['sensor'],
                    'robot': scene['config']['robot'], 'extent_m': primary['settings']['extent'],
                    'semantics': 'One fixed obstacle scene, target, sensor mount, scan pattern, controller and planning reference. Only atmospheric attenuation changes. Three seeds per weather setting; no independent scene replication.'},
        'targets': {'cases': targets, 'case_ids': list(target_ids),
                    'semantics': 'Nine target settings are compared at the same 271 saved baseline poses. Target diffuse weight, yaw, height or location changes individually. These are measurement comparisons; no target-change navigation outcome is used.'},
        'threshold': _read(files['threshold']), 'fov': _read(files['fov']), 'mapping': mapping,
        'verification': {'geometry': _read(files['geometry']), 'response': _read(files['response']),
                         'recorded_motion': {case: item['motion_check'] for case, item in mapping.items()}},
        'provenance': {'files': [_source(path, project) for path in files.values()],
                       'semantics': 'Presentation of saved target measurement comparisons and fixed-scene weather results. FOV scene geometry is explanatory; map points are measured endpoints. No scans or experiments run in this page.',
                       'display_coordinate_decimals': 5, 'map_point_limit_per_run': max_map_points}}
    json.dumps(result, allow_nan=False)
    return result
