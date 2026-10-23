"""Paired sampling-density / diffuse-reflectance experiment at fixed mounting.

The 2 x 2 design keeps target geometry, weather, receiver noise, full-circle FOV
and every mounting angle fixed. Matched replay uses one saved baseline pose
sequence per seed; closed-loop navigation is scored separately. Raw replay and
navigation clouds are retained for media without resampling or invented points.
"""
from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .. import api
from ..mapping import build_map, save_map
from ..trajectory import generate_reference, load_result, run_trajectory, save_result
from ..trajectory.configuration import effective_config
from ..trajectory.engine import mounted_sensor_pose
from .navigation_dependencies import (
    TARGET_LABEL, _csv, _json, _jsonable, _scene, aggregate_navigation,
    apply_case, pool_diagnostics, scan_diagnostics, study_config,
)
from .trajectory import evaluate_result

DENSE_AZIMUTHS = 360
SPARSE_AZIMUTHS = 90
BRIGHT_REFLECTANCE = .85
DARK_REFLECTANCE = .00005


def factorial_cases():
    """All four cells, including intermediate controls, rather than endpoints only."""
    return [dict(id=f"{density}_{material}", label=f"{density.capitalize()} / {material}",
                 factor="sampling_reflectance", sampling=density, reflectance=material,
                 azimuth_samples=count, target_diffuse_reflectance=rho,
                 description=f"{count} azimuths x 8 channels; study-target diffuse coefficient {rho:g}; fixed mounting.",
                 comparison_baseline="dense_bright",
                 overrides={"sensor": {"measurements_per_cycle": count},
                            "target": {"material": {"rho_d": rho, "rho_s": 0.}}})
            for density, count in (("dense", DENSE_AZIMUTHS), ("sparse", SPARSE_AZIMUTHS))
            for material, rho in (("bright", BRIGHT_REFLECTANCE), ("dark", DARK_REFLECTANCE))]


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _geometry(objects):
    cleaned = copy.deepcopy(objects)
    for obj in cleaned:
        if obj.get("label") == TARGET_LABEL:
            obj.get("material", {}).pop("rho_d", None)
    return cleaned


def controlled_invariants(configs):
    """Reject accidentally confounded experiments before any scans are generated."""
    first = configs[0]
    fixed_sensor = None
    for active in configs:
        effective, _ = effective_config(active, None)
        cfg = asdict(api.build_sensor(effective["_effective_sensor"]))
        cfg.pop("measurements_per_cycle")
        if fixed_sensor is None:
            fixed_sensor = cfg
        elif cfg != fixed_sensor:
            raise ValueError("Only azimuth sample count may change in sensor configuration")
        for key in ("robot", "controller", "planner", "mount_height_m", "sensor_mount"):
            if active[key] != first[key]:
                raise ValueError(f"Factorial comparison changed controlled field {key}")
        if _geometry(active["world"]["known_objects"]) != _geometry(first["world"]["known_objects"]):
            raise ValueError("Known scene geometry/material changed")
        if _geometry(active["world"]["unexpected_objects"]) != _geometry(first["world"]["unexpected_objects"]):
            raise ValueError("Unexpected scene changed beyond target reflectance")
    return dict(mount_height_m=first["mount_height_m"], sensor_mount=first["sensor_mount"],
                sensor_except_azimuth_count=fixed_sensor,
                world_except_target_reflectance=first["world"],
                robot=first["robot"], controller=first["controller"], planner=first["planner"])


def _pooled(rows):
    pooled = pool_diagnostics(rows)
    n = pooled["pose_count"]
    pooled.update(mean_returns_per_scan=pooled["all_detected_returns"] / n,
                  mean_target_returns_per_scan=pooled["detected_returns"] / n,
                  mean_target_eligible_per_scan=pooled["eligible_echoes"] / n,
                  detected_target_per_emitted_beam=pooled["detected_returns"] / pooled["total_beams"])
    return pooled


def distance_bins(rows):
    """Pool whole snapshots by their mean eligible-target distance, not a per-ray bin."""
    edges = (.5, 1., 2., 3., 5., 10., 200.)
    result = []
    for low, high in zip(edges[:-1], edges[1:]):
        selected = [row for row in rows if row["mean_eligible_range_m"] is not None
                    and low <= row["mean_eligible_range_m"] < high]
        if selected:
            result.append(dict(mean_pose_distance_min_m=low, mean_pose_distance_max_m=high,
                               **_pooled(selected)))
    return result


def _replay_recording(config, case, seed, source, destination):
    """Record every counterfactual scan at the immutable baseline control poses."""
    effective, _ = effective_config(config, None)
    cfg, scene = api.build_sensor(effective["_effective_sensor"]), _scene(effective)
    sensor = api.LidarSensor(cfg, seed=seed)
    result = copy.deepcopy(source)
    result.update(name=case["id"], status="replay", seed=seed, config=effective,
                  sensor=asdict(cfg), world_objects=[copy.deepcopy(obj) for obj in source["world_objects"]],
                  observations=[], scans=[], history_recorded=True, scans_available=True)
    target_object = next(obj for obj in result["world_objects"] if obj["label"] == TARGET_LABEL)
    target_object["material"] = dict(rho_d=case["target_diffuse_reflectance"], rho_s=0.)
    result["assumptions"] = ["Counterfactual measurements at saved baseline poses; no controls are executed by this replay.",
                             "Every case of the same seed has exactly the same K+1 robot states and K scan times.",
                             "No additional scan occurs at the final state; cumulative mapping uses saved scan chronology only.",
                             "Target labels are used for evaluation/colour only, never by the navigation controller."]
    diagnostics, target_masks = [], []
    for step, time_s in enumerate(source["controls"]["t"]):
        xy, theta = source["states"]["xy"][step], source["states"]["theta"][step]
        sensor.pose = mounted_sensor_pose(effective, xy, theta)
        scan = sensor.scan(scene)
        diagnostics.append(scan_diagnostics(scan, sensor.pose, scene, cfg, effective, step, time_s, xy, theta))
        mask = ((scan.xyz_world[:, 2] >= effective["controller"]["z_ground_m"])
                & (scan.xyz_world[:, 2] <= effective["robot"]["height_m"]))
        result["observations"].append(scan.xyz_world[mask, :2].copy())
        result["scans"].append(dict(xyz_world=scan.xyz_world.copy(), intensity=scan.intensity.copy()))
        result["controls"]["n_returns"][step] = scan.n
        result["controls"]["n_obstacle_points"][step] = int(mask.sum())
        target_masks.append(scan.label_id == scene.labels.index(TARGET_LABEL))
    save_result(result, destination)
    offsets = np.r_[0, np.cumsum([len(mask) for mask in target_masks])]
    np.savez_compressed(destination / "scan_targets.npz", scan_offsets=offsets,
                        target_mask=np.concatenate(target_masks).astype(bool))
    mapping = build_map(result, voxel_size_m=.12, resolution_m=.2, extent=config["planner"]["extent"])
    save_map(mapping, destination)
    _csv(destination / "diagnostics.csv", diagnostics)
    _csv(destination / "traces.csv", diagnostics)
    metrics = _pooled(diagnostics)
    metrics.update(map_voxels=len(mapping["map_xyz"]), observed_fraction=mapping["metadata"]["observed_cell_fraction"])
    _json(destination / "diagnostics.json", metrics)
    _json(destination / "replay.json", dict(case_id=case["id"], seed=seed, metrics=metrics,
                                           poses=f"../../../reference_poses_seed_{seed}.npz",
                                           target_label=TARGET_LABEL, distance_bins=distance_bins(diagnostics)))
    return metrics, diagnostics


def factorial_contrasts(replays):
    """Paired descriptive differences, with availability and conditional P_D separate."""
    metrics = ("mean_returns_per_scan", "mean_target_returns_per_scan", "mean_target_eligible_per_scan",
               "observed_detection_probability", "expected_detection_probability", "mean_eligible_power_w",
               "range_rmse_m", "observed_fraction")
    pairs = (("sampling_at_bright", "dense_bright", "sparse_bright"),
             ("sampling_at_dark", "dense_dark", "sparse_dark"),
             ("reflectance_at_dense", "dense_bright", "dense_dark"),
             ("reflectance_at_sparse", "sparse_bright", "sparse_dark"))
    by_seed = {}
    for row in replays:
        by_seed.setdefault(row["seed"], {})[row["case_id"]] = row
    results = []
    for name, high, low in pairs:
        seed_rows = []
        for seed, selected in sorted(by_seed.items()):
            row = dict(seed=seed)
            for metric in metrics:
                a, b = selected[high][metric], selected[low][metric]
                row[metric] = None if a is None or b is None else a - b
            seed_rows.append(row)
        means = {metric: float(np.mean([r[metric] for r in seed_rows if r[metric] is not None]))
                 if any(r[metric] is not None for r in seed_rows) else None for metric in metrics}
        results.append(dict(name=name, minuend=high, subtrahend=low, paired_by_seed=seed_rows,
                            mean_paired_difference=means))
    return results


def run_study(directory, source_directory="data/navigation_dependencies", seeds=(1, 2, 3), config=None):
    """Generate all numerical inputs needed by the comparison figures and video."""
    destination = Path(directory).expanduser().resolve()
    source_directory = Path(source_directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide nonempty, distinct seed labels")
    cases, baseline = factorial_cases(), study_config(config)
    active_configs = [apply_case(baseline, case) for case in cases]
    invariants = controlled_invariants(active_configs)
    source_results, source_records = {}, []
    reference = generate_reference(baseline)
    for seed in seeds:
        source = source_directory / "runs" / "baseline" / f"seed_{seed}"
        if not (source / "trajectory.npz").exists():
            source = destination / "sources" / "baseline" / f"seed_{seed}"
            save_result(run_trajectory(baseline, "baseline", seed=seed, reference=reference), source)
        saved = load_result(source)
        expected = [dict(type="plane", label="ground", point=[0, 0, 0], normal=[0, 0, 1], material="asphalt")]
        expected += baseline["world"]["known_objects"] + baseline["world"]["unexpected_objects"]
        if saved["world_objects"] != expected:
            raise ValueError("Baseline source scene differs from the controlled factorial scene")
        if not np.array_equal(saved["reference"]["xy"], reference["xy"]):
            raise ValueError("Baseline source does not share the controlled planning reference")
        source_results[seed] = saved
        source_records.append(dict(seed=seed, directory=str(source),
                                   trajectory_json_sha256=_hash(source / "trajectory.json"),
                                   trajectory_npz_sha256=_hash(source / "trajectory.npz")))
        np.savez_compressed(destination / f"reference_poses_seed_{seed}.npz",
                            t=saved["states"]["t"][:-1], xy=saved["states"]["xy"][:-1],
                            theta=saved["states"]["theta"][:-1],
                            reference_xy=reference["xy"], reference_t=reference["t"])
    settings = dict(azimuth_samples=[DENSE_AZIMUTHS, SPARSE_AZIMUTHS], channels=8,
                    emitted_beams_per_scan=[DENSE_AZIMUTHS * 8, SPARSE_AZIMUTHS * 8],
                    diffuse_reflectance=[BRIGHT_REFLECTANCE, DARK_REFLECTANCE],
                    reflectance_power_ratio=BRIGHT_REFLECTANCE / DARK_REFLECTANCE,
                    mount_height_m=baseline["mount_height_m"], sensor_mount=baseline["sensor_mount"],
                    best_case="dense_bright", worst_case="sparse_dark",
                    pose_source="Canonical same-seed baseline control-snapshot poses; identical across all four replay cases.")
    definitions = dict(
        best_worst="Favourable/adverse endpoints for measurement availability within this factorial design, not a globally optimal sensor or a navigation success ranking.",
        target_opportunities="Emitted rays intersecting the isolated study-target geometry before LOS/range tests.",
        eligible_echoes="Nearest opaque hit is the target and its range satisfies the inclusive sensor range gate.",
        detected_returns="Observed target detections; denominator for conditional target P_D is eligible_echoes, not all emitted rays.",
        mean_eligible_power_w="Analytical noiseless power over every eligible target echo, avoiding threshold-selection bias.",
        mean_power_w="Measured power conditional on target detection; this average is selection biased near threshold.",
        range_rmse_m="Measured-minus-true target range RMS conditional on detected target returns; undefined when none detected.",
        distance_bins="Whole snapshots grouped by mean eligible target range; sums use all eligible/detected target echoes in those snapshots, not an exact per-ray range histogram.",
        observed_fraction="Common ROI XY cells with measured endpoints; neither map accuracy nor inferred free/traversable space.",
        sampling="Measurements_per_cycle controls angular sampling density (1 versus 4 degrees), not emitted optical power.",
        sensor_state="Known robot position/yaw vary along the same fixed replay trajectory; mount orientation and height are constant.",
        paired_seed="Same seed labels and poses are paired, but changed beam counts change RNG consumption; per-ray noise is not forced identical.")
    limitations = ["Diffuse coefficients and receiver optics are illustrative virtual stress settings, not calibrated real paints or a specific LiDAR.",
                   "The dark coefficient 5e-5 is deliberately selected to cross the model's receiver threshold over the recorded approach/departure.",
                   "Only the designated study_target material changes; other objects, geometry, atmosphere, noise, mount angles and FOV stay fixed.",
                   "A high density scan emits four times as many rays; target intersections need not scale by exactly four because of angular alignment.",
                   "Matched replay isolates measurement causes; closed-loop trajectories can subsequently diverge because measurements drive the controller.",
                   "Recorded poses are exact and scans are ideal snapshots; no SLAM, within-sweep motion, atmospheric backscatter or moving targets.",
                   "Three paired seeds are descriptive rather than a population reliability estimate."]
    _json(destination / "manifest.json", dict(study="sampling_reflectance", cases=cases, seeds=seeds,
                                             settings=settings, definitions=definitions, limitations=limitations,
                                             controlled_invariants=invariants, sources=source_records))
    runs, replays, completed = [], [], []
    start = time.perf_counter()
    for case, active in zip(cases, active_configs):
        effective, _ = effective_config(active, None)
        cfg, scene = api.build_sensor(effective["_effective_sensor"]), _scene(effective)
        nav_rows, replay_traces = [], []
        for seed in seeds:
            diagnostics = []
            def observe(*, scan, pose, time, robot_xy, robot_theta):
                diagnostics.append(scan_diagnostics(scan, pose, scene, cfg, effective,
                                                    len(diagnostics), time, robot_xy, robot_theta))
            result = run_trajectory(active, case["id"], seed=seed, reference=reference, on_scan=observe)
            relative = Path("runs") / case["id"] / f"seed_{seed}"
            save_result(result, destination / relative)
            mapping = build_map(result, voxel_size_m=.12, resolution_m=.2, extent=baseline["planner"]["extent"])
            save_map(mapping, destination / relative)
            _csv(destination / relative / "diagnostics.csv", diagnostics)
            _json(destination / relative / "diagnostics.json", _pooled(diagnostics))
            nav = evaluate_result(result, case["id"], case["factor"], seed, True)
            nav.update(run_directory=relative.as_posix(), observed_fraction=mapping["metadata"]["observed_cell_fraction"],
                       map_voxels=len(mapping["map_xyz"]))
            nav_rows.append(nav)
            runs.append(nav)
            relative_replay = Path("replays") / case["id"] / f"seed_{seed}"
            replay, traces = _replay_recording(active, case, seed, source_results[seed], destination / relative_replay)
            replay.update(case_id=case["id"], seed=seed, replay_directory=relative_replay.as_posix())
            replays.append(replay)
            replay_traces.extend(traces)
            print(f"sampling/reflectance: {case['id']} seed {seed}: navigation {nav['status']}; "
                  f"matched target {replay['detected_returns']}/{replay['eligible_echoes']}, "
                  f"P_D={replay['observed_detection_probability']:.4f}", flush=True)
        pooled = _pooled(replay_traces)
        replay_selected = [row for row in replays if row["case_id"] == case["id"]]
        pooled.update(mean_observed_fraction=float(np.mean([r["observed_fraction"] for r in replay_selected])),
                      mean_map_voxels=float(np.mean([r["map_voxels"] for r in replay_selected])))
        completed.append({**case, "aggregate": aggregate_navigation(nav_rows), "replay": pooled,
                          "distance_bins": distance_bins(replay_traces),
                          "representative_run": f"runs/{case['id']}/seed_{seeds[0]}",
                          "representative_replay": f"replays/{case['id']}/seed_{seeds[0]}"})
        _csv(destination / "runs.csv", runs)
        _csv(destination / "replay.csv", replays)
    summary = dict(format_version=1, study="sampling_reflectance", seeds=list(seeds), settings=settings,
                   definitions=definitions, limitations=limitations, sources=source_records,
                   controlled_invariants=invariants, cases=completed, runs=runs, replays=replays,
                   factorial_contrasts=factorial_contrasts(replays), elapsed_wall_time_s=time.perf_counter() - start)
    _json(destination / "summary.json", summary)
    return _jsonable(summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/sampling_reflectance")
    parser.add_argument("--source-dir", default="data/navigation_dependencies")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    args = parser.parse_args(argv)
    run_study(args.output_dir, args.source_dir, seeds=args.seeds)


if __name__ == "__main__":
    main()
