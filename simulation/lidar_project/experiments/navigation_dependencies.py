"""Controlled measurement replay and closed-loop navigation dependency study.

The numerical experiment writes portable data only. A shared known-map route
and frozen baseline poses separate changes in measurements from feedback-driven
changes in robot motion. Geometry labels are used for scoring, never control.
"""
from __future__ import annotations

import copy
import csv
import json
import math
from pathlib import Path
import time

import numpy as np

from .. import api
from ..mapping import build_map, save_map
from ..trajectory import load_config, generate_reference, run_trajectory, save_result
from ..trajectory.configuration import effective_config
from ..trajectory.engine import mounted_sensor_pose
from .environments import fixed_environment
from .trajectory import evaluate_result


TARGET_LABEL = "study_target"


def dependency_cases():
    """One-factor cases; front-facing yaw has its own matched FOV baseline."""
    def case(id, label, factor, description, overrides=None, comparison="baseline"):
        return dict(id=id, label=label, factor=factor, description=description,
                    overrides=overrides or {}, comparison_baseline=comparison)
    return [
        case("baseline", "Baseline", "baseline", "Cluttered scene; 0.6 m mount, level full-circle 8-channel scan, pure diffuse study target."),
        case("mount_low", "Low mount: 0.3 m", "sensor_position", "Mount height changes to 0.3 m.", {"mount_height_m": .3}),
        case("mount_high", "High mount: 1.6 m", "sensor_position", "Mount height changes to 1.6 m.", {"mount_height_m": 1.6}),
        case("pitch_up", "Pitch up: 15 degrees", "sensor_orientation", "Mount pitch -15 degrees; positive pitch is downward.", {"sensor_mount": {"pitch_deg": -15.}}),
        case("pitch_down", "Pitch down: 15 degrees", "sensor_orientation", "Mount pitch +15 degrees.", {"sensor_mount": {"pitch_deg": 15.}}),
        case("yaw_full", "Full FOV, yaw 60 degrees", "sensor_orientation", "Negative control: a full circle rotated by an integer azimuth step preserves physical sampling directions.", {"sensor_mount": {"yaw_deg": 60.}}),
        case("fov_front", "Front FOV: 120 degrees", "field_of_view", "Horizontal FOV -60 to +60 degrees; fixed one-degree angular spacing.", {"sensor": {"hfov_start_deg": -60., "hfov_end_deg": 60.}}),
        case("fov_side", "120-degree FOV, yaw 90 degrees", "sensor_orientation", "Yaw offset +90 degrees with the same partial FOV as fov_front.", {"sensor": {"hfov_start_deg": -60., "hfov_end_deg": 60.}, "sensor_mount": {"yaw_deg": 90.}}, "fov_front"),
        case("reflectance_dark", "Diffuse reflectance 0.02", "target_reflectance", "Only the study target's diffuse reflectance changes; specular weight stays zero.", {"target": {"material": {"rho_d": .02, "rho_s": 0.}}}),
        case("reflectance_bright", "Diffuse reflectance 0.85", "target_reflectance", "Only the study target's diffuse reflectance changes; specular weight stays zero.", {"target": {"material": {"rho_d": .85, "rho_s": 0.}}}),
        case("target_yaw_45", "Target yaw: 45 degrees", "target_orientation", "Rotate the target box 45 degrees; incidence, projected area and footprint change together.", {"target": {"yaw_deg": 45.}}),
        case("target_yaw_75", "Target yaw: 75 degrees", "target_orientation", "Rotate the target box 75 degrees; this is not an isolated cosine-law test.", {"target": {"yaw_deg": 75.}}),
        case("target_short", "Target height: 0.3 m", "target_size", "Reduce target height while preserving its XY footprint and ground contact.", {"target": {"center": [11., 1.1, .15], "size": [.8, 1.6, .3]}}),
        case("target_tall", "Target height: 2.0 m", "target_size", "Increase target height while preserving its XY footprint and ground contact.", {"target": {"center": [11., 1.1, 1.], "size": [.8, 1.6, 2.]}}),
        case("target_near", "Target x: 9 m", "target_position", "Translate the held-out target along x; range, occlusion and task geometry can change.", {"target": {"center": [9., 1.1, .6]}}),
        case("target_far", "Target x: 13 m", "target_position", "Translate the held-out target along x; inspect fixed-pose measurement traces for range effects.", {"target": {"center": [13., 1.1, .6]}}),
        case("rain", "Rain: 50 mm/h", "environment", "Illustrative atmospheric attenuation from the existing rain extinction law; no rain backscatter.", {"sensor": {"rain_mm_per_h": 50.}}),
        case("fog", "Extinction: 0.6 /m", "environment", "Illustrative extinction coefficient 0.6 per metre; no fog backscatter.", {"sensor": {"air_extinction_per_m": .6}}),
        case("noise_high", "Receiver noise: 10 times", "measurement_reliability", "Receiver RMS optical noise increases from 4e-9 to 4e-8 W; threshold SNR remains fixed.", {"sensor": {"noise_rms_w": 4e-8}}),
        case("sampling_sparse", "Azimuth samples: 90", "sampling", "Four-degree instead of one-degree azimuth spacing; full-circle FOV remains fixed.", {"sensor": {"measurements_per_cycle": 90}}),
    ]


def study_config(config=None):
    original = load_config(config) if config is None or isinstance(config, (str, Path)) else copy.deepcopy(config)
    active = fixed_environment(original)
    for obj in active["world"]["unexpected_objects"]:
        if obj["label"] == "unexpected_barrel":
            obj.clear()
            obj.update(type="box", label=TARGET_LABEL, center=[11., 1.1, .6],
                       size=[.8, 1.6, 1.2], yaw_deg=0., material={"rho_d": .3, "rho_s": 0.})
    active["sensor_mount"] = dict(offset_xy_m=[0., 0.], roll_deg=0., pitch_deg=0., yaw_deg=0.)
    active["scenarios"] = [{"name": "baseline"}]
    return active


def apply_case(baseline, case):
    """Apply explicit study overrides without changing the shared planning map."""
    active, changes = copy.deepcopy(baseline), case["overrides"]
    active.setdefault("sensor_overrides", {}).update(changes.get("sensor", {}))
    active.setdefault("sensor_mount", {}).update(changes.get("sensor_mount", {}))
    for key in ("mount_height_m", "lidar_enabled"):
        if key in changes:
            active[key] = changes[key]
    target = next(obj for obj in active["world"]["unexpected_objects"] if obj["label"] == TARGET_LABEL)
    target.update(copy.deepcopy(changes.get("target", {})))
    active["scenarios"] = [{"name": case["id"]}]
    return active


def _scene(config):
    ground = dict(type="plane", label="ground", point=[0, 0, 0], normal=[0, 0, 1], material="asphalt")
    return api.build_scene({"objects": [ground] + copy.deepcopy(config["world"]["known_objects"])
                            + copy.deepcopy(config["world"].get("unexpected_objects", []))})


def emitted_rays(cfg, pose, full_circle=False):
    """Independently reconstruct nominal pencil-ray directions, without RNG."""
    if cfg.n_beam_samples != 1:
        raise ValueError("Dependency opportunity audit requires one sub-ray per beam")
    azimuths = (cfg.hfov_start_deg + 360. / cfg.measurements_per_cycle * np.arange(cfg.measurements_per_cycle)
                if full_circle else cfg.azimuths_deg())
    az, el = np.meshgrid(np.radians(azimuths), np.radians(cfg.elevations_deg()), indexing="ij")
    local = np.column_stack((np.cos(el).ravel() * np.cos(az).ravel(),
                             np.cos(el).ravel() * np.sin(az).ravel(), np.sin(el).ravel()))
    rotation = (api.rot_z(np.radians(pose.yaw_deg)) @ api.rot_y(np.radians(pose.pitch_deg))
                @ api.rot_x(np.radians(pose.roll_deg)))
    direction = local @ rotation.T
    origin = np.broadcast_to(np.asarray(pose.position, float), direction.shape)
    if np.any(np.asarray(pose.velocity)) or pose.yaw_rate_deg_s:
        raise ValueError("Dependency replay audits snapshot scans only")
    return origin, direction


def opportunity_audit(cfg, pose, scene, target_label=TARGET_LABEL):
    """Count geometry, LOS and range eligibility before inspecting detections."""
    target_id = scene.labels.index(target_label)
    target = scene.objects[target_id]
    origin, direction = emitted_rays(cfg, pose)
    alone_range, _ = target.intersect(origin, direction)
    ranges, nearest_id, normal = scene.cast(origin, direction)
    opportunity = np.isfinite(alone_range)
    visible = opportunity & (nearest_id == target_id)
    eligible = visible & (ranges >= cfg.min_range_m) & (ranges <= cfg.max_range_m)
    full_origin, full_direction = emitted_rays(cfg, pose, full_circle=True)
    full_range, _ = target.intersect(full_origin, full_direction)
    distance = ranges[eligible]
    cosine = np.clip(-np.sum(normal[eligible] * direction[eligible], axis=1), 0, 1)
    material = target.material
    specular = np.clip(2 * cosine ** 2 - 1, 0, 1) ** material.spec_exp
    brdf = material.rho_d / np.pi + material.rho_s * (material.spec_exp + 2) / (2 * np.pi) * specular
    # Independent expanded aperture/transmission expression, not sensor.received_power.
    coefficient = cfg.peak_power_w * (np.pi * cfg.aperture_diam_m ** 2 / 4) * cfg.optical_efficiency
    extinction = cfg.air_extinction_per_m + (cfg.rain_a * cfg.rain_mm_per_h ** cfg.rain_b if cfg.rain_mm_per_h > 0 else 0.)
    power = coefficient * brdf * cosine * np.exp(-2 * extinction * distance) / distance ** 2
    if cfg.noise_enabled:
        probability = np.fromiter((.5 * math.erfc((cfg.snr_threshold - p / cfg.noise_rms_w) / math.sqrt(2)) for p in power), float, len(power))
    else:
        probability = (power >= cfg.snr_threshold * cfg.noise_rms_w).astype(float)
    return dict(total_beams=len(direction), target_opportunities=int(opportunity.sum()),
                full_fov_opportunities=int(np.isfinite(full_range).sum()),
                visible_target_echoes=int(visible.sum()),
                blocked_target_echoes=int((opportunity & ~visible).sum()),
                eligible_echoes=int(eligible.sum()), expected_detections=float(probability.sum()),
                expected_detection_variance=float(np.sum(probability * (1 - probability))),
                eligible_range_sum_m=float(distance.sum()),
                eligible_power_sum_w=float(power.sum()),
                eligible_incidence_sum_deg=float(np.degrees(np.arccos(cosine)).sum()))


def _ratio(numerator, denominator):
    return float(numerator / denominator) if denominator else None


def _finish_metrics(row):
    n, eligible = row["detected_returns"], row["eligible_echoes"]
    row.update(observed_detection_probability=_ratio(n, eligible),
               expected_detection_probability=_ratio(row["expected_detections"], eligible),
               mean_eligible_power_w=_ratio(row["eligible_power_sum_w"], eligible),
               mean_eligible_range_m=_ratio(row["eligible_range_sum_m"], eligible),
               mean_eligible_incidence_deg=_ratio(row["eligible_incidence_sum_deg"], eligible),
               mean_power_w=_ratio(row["power_sum_w"], n), mean_snr=_ratio(row["snr_sum"], n),
               mean_target_range_m=_ratio(row["range_sum_m"], n),
               mean_incidence_deg=_ratio(row["incidence_sum_deg"], n),
               range_bias_m=_ratio(row["range_error_sum_m"], n),
               range_rmse_m=math.sqrt(row["range_error_sq_sum_m2"] / n) if n else None,
               predicted_range_rmse_m=math.sqrt(row["predicted_variance_sum_m2"] / n) if n else None,
               normalized_rmse=math.sqrt(row["normalized_error_sq_sum"] / n) if n else None)
    return row


SUM_FIELDS = ("total_beams", "target_opportunities", "full_fov_opportunities", "visible_target_echoes",
              "blocked_target_echoes", "eligible_echoes", "detected_returns", "expected_detections",
              "expected_detection_variance",
              "eligible_range_sum_m", "eligible_power_sum_w", "eligible_incidence_sum_deg",
              "target_controller_points", "all_detected_returns", "power_sum_w", "snr_sum",
              "range_sum_m", "incidence_sum_deg", "range_error_sum_m", "range_error_sq_sum_m2",
              "predicted_variance_sum_m2", "normalized_error_sq_sum")


def scan_diagnostics(scan, pose, scene, sensor_cfg, config, step=0, time_s=0.,
                     robot_xy=None, robot_theta=0.):
    """Read an already-produced cloud; scene truth enters the scoring side only."""
    row = opportunity_audit(sensor_cfg, pose, scene)
    target_id = scene.labels.index(TARGET_LABEL)
    target_mask = scan.label_id == target_id
    n = int(target_mask.sum())
    values = {name: np.asarray(getattr(scan, name))[target_mask] for name in
              ("range_m", "range_true_m", "power_w", "snr", "incidence_deg")}
    errors = values["range_m"] - values["range_true_m"]
    variance = (sensor_cfg.range_floor_sigma_m ** 2 + (.5 * 299792458.0 * sensor_cfg.pulse_rise_ns * 1e-9) ** 2
                / (2 * np.maximum(values["snr"], 1e-9)))
    if not sensor_cfg.noise_enabled:
        variance = np.zeros(n)
    z = scan.xyz_world[target_mask, 2]
    relevant = (z >= config["controller"]["z_ground_m"]) & (z <= config["robot"]["height_m"])
    xy = np.asarray(robot_xy if robot_xy is not None else pose.position[:2])
    row.update(step=int(step), t=float(time_s), robot_x_m=float(xy[0]), robot_y_m=float(xy[1]),
               robot_theta_rad=float(robot_theta), sensor_enabled=1, detected_returns=n,
               target_controller_points=int(relevant.sum()), all_detected_returns=int(scan.n),
               power_sum_w=float(values["power_w"].sum()), snr_sum=float(values["snr"].sum()),
               range_sum_m=float(values["range_true_m"].sum()), incidence_sum_deg=float(values["incidence_deg"].sum()),
               range_error_sum_m=float(errors.sum()), range_error_sq_sum_m2=float((errors ** 2).sum()),
               predicted_variance_sum_m2=float(variance.sum()),
               normalized_error_sq_sum=float(np.sum(errors ** 2 / np.maximum(variance, 1e-300))),
               target_range_m=float(np.median(values["range_true_m"])) if n else None,
               power_median_w=float(np.median(values["power_w"])) if n else None,
               snr_median=float(np.median(values["snr"])) if n else None,
               incidence_median_deg=float(np.median(values["incidence_deg"])) if n else None)
    if n > row["eligible_echoes"]:
        raise AssertionError("Detected target returns exceed independently eligible target rays")
    return _finish_metrics(row)


def disabled_diagnostics(step, time_s, xy, theta):
    row = {key: 0 for key in SUM_FIELDS}
    row.update(step=int(step), t=float(time_s), robot_x_m=float(xy[0]), robot_y_m=float(xy[1]),
               robot_theta_rad=float(theta), sensor_enabled=0, target_range_m=None,
               power_median_w=None, snr_median=None, incidence_median_deg=None)
    return _finish_metrics(row)


def pool_diagnostics(rows):
    result = {key: sum(row[key] for row in rows) for key in SUM_FIELDS}
    result.update(snapshots=len(rows), pose_count=len(rows),
                  sensor_enabled=int(any(row["sensor_enabled"] for row in rows)))
    return _finish_metrics(result)


def aggregate_navigation(rows):
    """Retain failures; means of prefixes are explicitly termination summaries."""
    return dict(runs=len(rows), goals=sum(r["goal_completed"] for r in rows),
                collisions=sum(r["collision"] for r in rows), timeouts=sum(r["timeout"] for r in rows),
                mean_elapsed_s=float(np.mean([r["elapsed_s"] for r in rows])),
                mean_path_length_m=float(np.mean([r["path_length_m"] for r in rows])),
                worst_surface_gap_m=min(r["swept_min_surface_gap_m"] for r in rows),
                mean_observed_fraction=float(np.mean([r["observed_fraction"] for r in rows])),
                mean_spatial_route_rmse_m=float(np.mean([r["spatial_route_rmse_m"] for r in rows])),
                mean_returns_per_control_step=float(np.mean([r["mean_returns_per_control_step"] for r in rows])))


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _json(path, value):
    path.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _csv(path, rows):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _replay(config, case, seed, baseline_result, reference, destination):
    effective, _ = effective_config(config, None)
    cfg, scene = api.build_sensor(effective["_effective_sensor"]), _scene(effective)
    sensor = api.LidarSensor(cfg, seed=seed)
    poses = baseline_result["states"]
    n = len(baseline_result["controls"]["t"])
    selected = {0, n // 2, n - 1}
    traces, snapshots = [], []
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "snapshots").mkdir(exist_ok=True)
    for step in range(n):
        t, xy, theta = poses["t"][step], poses["xy"][step], poses["theta"][step]
        sensor.pose = mounted_sensor_pose(effective, xy, theta)
        if effective.get("lidar_enabled", True):
            pc = sensor.scan(scene)
            row = scan_diagnostics(pc, sensor.pose, scene, cfg, effective, step, t, xy, theta)
            if step in selected:
                relative = f"snapshots/step_{step:04d}.npz"
                pc.save_npz(destination / relative)
                snapshots.append(dict(step=step, t=float(t), path=relative))
        else:
            row = disabled_diagnostics(step, t, xy, theta)
        traces.append(row)
    _csv(destination / "traces.csv", traces)
    pooled = pool_diagnostics(traces)
    pooled.update(case_id=case["id"], seed=seed)
    _json(destination / "replay.json", dict(
        case_id=case["id"], seed=seed, snapshots=snapshots, metrics=pooled,
        poses=f"../../../reference_poses_seed_{seed}.npz", target_label=TARGET_LABEL,
        assumptions=["Same baseline control-snapshot poses for every case; no feedback or trajectory changes in replay.",
                     "Snapshot point clouds are immutable saved measurements at first, middle and last replay poses."]))
    return pooled, traces


def run_study(directory, config=None, seeds=(1, 2, 3), case_ids=None):
    """Save controlled replays, closed-loop runs, and endpoint maps for all cases."""
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    seeds = tuple(int(seed) for seed in seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide nonempty, distinct seed labels")
    cases = dependency_cases()
    if case_ids:
        requested = set(case_ids)
        unknown = requested - {case["id"] for case in cases}
        if unknown:
            raise ValueError(f"Unknown dependency cases: {sorted(unknown)}")
        requested.add("baseline")
        if "fov_side" in requested:
            requested.add("fov_front")
        cases = [case for case in cases if case["id"] in requested]
    baseline = study_config(config)
    reference = generate_reference(baseline)
    settings = dict(channels=8, azimuths=360, subrays=1, dt_s=baseline["robot"]["dt"],
                    time_budget_s=baseline["robot"]["dt"] * baseline["robot"]["max_steps"],
                    start=baseline["robot"]["start"], goal=baseline["robot"]["goal"],
                    voxel_size_m=.12, grid_resolution_m=.2, extent=baseline["planner"]["extent"],
                    target_label=TARGET_LABEL, pose_source="same-seed baseline closed-loop control snapshots")
    definitions = dict(
        replay="Measurements at identical saved baseline poses; scene/sensor changes cannot alter the replay route.",
        closed_loop="All cases use one shared incomplete-map reference; the sensed controller may choose different motions.",
        target_opportunities="Emitted nominal pencil rays intersecting the target alone, before range gates or other opaque objects.",
        full_fov_opportunities="Counterfactual full-circle nominal ray grid at the same pose and mounting orientation; diagnostic only.",
        visible_target_echoes="Target opportunities whose nearest opaque scene hit is the target, before the range gate.",
        eligible_echoes="Visible nearest target hits inside inclusive min/max range gates; denominator for target-conditioned detection probability.",
        detected_returns="Detected returns labeled as the study target; excludes ground and all other objects.",
        expected_detections="Sum of independent analytical Gaussian threshold probabilities over eligible pencil-ray target echoes.",
        mean_power_w="Mean measured power conditional on detected target returns; threshold selection can bias this above the unconditional mean.",
        mean_eligible_power_w="Mean analytical noiseless power over all eligible target beams; excludes detector selection bias.",
        range_rmse_m="Pooled measured-minus-noise-free echo range RMS, conditional on detected study-target returns.",
        predicted_range_rmse_m="Square root of mean assumed range variance evaluated at each detected return's measured SNR.",
        observed_fraction="Fraction of common ROI XY cells containing measured endpoints; neither map accuracy nor traversable free space.",
        failure_prefix="All failed runs remain included; elapsed time, path and mapping describe the terminated prefix.",
        pairing="Seed labels and baseline pose sequences are paired. Changing beam count/geometry can change random consumption, so errors are not identical per ray.")
    limitations = [
        "Three seed labels are descriptive, not a population reliability estimate.",
        "Known exact poses, ideal snapshot scans, static objects; no SLAM, rolling motion, latency or acceleration model.",
        "Illustrative optical/material/weather coefficients; no physical-device calibration or real-weather validation.",
        "Rain and fog add attenuation only; atmospheric backscatter and false weather returns are omitted.",
        "Target yaw changes incidence, projected area and collision footprint together; target translation changes range and scene geometry together.",
        "An added LOS panel is itself a real visible obstacle; closed-loop effects cannot be attributed solely to target occlusion.",
        "Disabling LiDAR removes emissions; detection probabilities are undefined rather than zero when no eligible echoes exist.",
        "Full-circle yaw is a geometric negative control; ray ordering may change noise assignment despite the same physical sampling directions.",
        "Reference poses follow the baseline trajectory, including its finite endpoint; replay results measure that shared exposure, not every possible viewpoint.",
    ]
    _json(directory / "manifest.json", dict(study="navigation_dependencies", cases=cases, baseline_config=baseline,
                                             seeds=seeds, settings=settings, definitions=definitions, limitations=limitations))
    baseline_results, nav_rows, replay_rows, output_cases = {}, [], [], []
    started = time.perf_counter()
    for case in cases:
        active = apply_case(baseline, case)
        effective, _ = effective_config(active, None)
        scene, sensor_cfg = _scene(effective), api.build_sensor(effective["_effective_sensor"])
        selected_nav, selected_replay, all_replay_traces = [], [], []
        for seed in seeds:
            diagnostics = []
            def observe(*, scan, pose, time, robot_xy, robot_theta):
                diagnostics.append(scan_diagnostics(scan, pose, scene, sensor_cfg, effective,
                                                    len(diagnostics), time, robot_xy, robot_theta))
            result = run_trajectory(active, case["id"], seed=seed, reference=reference, on_scan=observe)
            if not effective.get("lidar_enabled", True):
                diagnostics = [disabled_diagnostics(i, t, result["states"]["xy"][i], result["states"]["theta"][i])
                               for i, t in enumerate(result["controls"]["t"])]
            relative_run = Path("runs") / case["id"] / f"seed_{seed}"
            run_directory = directory / relative_run
            save_result(result, run_directory)
            mapping = build_map(result, voxel_size_m=.12, resolution_m=.2, extent=baseline["planner"]["extent"])
            save_map(mapping, run_directory)
            _csv(run_directory / "diagnostics.csv", diagnostics)
            _json(run_directory / "diagnostics.json", pool_diagnostics(diagnostics))
            row = evaluate_result(result, case["id"], case["factor"], seed, effective.get("lidar_enabled", True))
            observed = (mapping["ground_first_seen"] >= 0) | (mapping["obstacle_first_seen"] >= 0)
            row.update(run_directory=relative_run.as_posix(), observed_fraction=float(observed.mean()),
                       map_voxels=len(mapping["map_xyz"]))
            selected_nav.append(row)
            nav_rows.append(row)
            if case["id"] == "baseline":
                baseline_results[seed] = result
                states = result["states"]
                np.savez_compressed(directory / f"reference_poses_seed_{seed}.npz", t=states["t"][:-1],
                                    xy=states["xy"][:-1], theta=states["theta"][:-1],
                                    reference_xy=reference["xy"], reference_t=reference["t"])
            relative_replay = Path("replays") / case["id"] / f"seed_{seed}"
            replay, replay_traces = _replay(active, case, seed, baseline_results[seed], reference,
                                             directory / relative_replay)
            replay["replay_directory"] = relative_replay.as_posix()
            selected_replay.append(replay)
            replay_rows.append(replay)
            all_replay_traces.extend(replay_traces)
            print(f"navigation dependencies: {case['id']} seed {seed}: {row['status']}, "
                  f"{row['elapsed_s']:.1f} s, gap {row['swept_min_surface_gap_m']:.3f} m; "
                  f"replay target detections {replay['detected_returns']}/{replay['eligible_echoes']}", flush=True)
            _csv(directory / "runs.csv", nav_rows)
            _csv(directory / "replay.csv", replay_rows)
        output_cases.append({**case, "aggregate": aggregate_navigation(selected_nav),
                             "replay": pool_diagnostics(all_replay_traces),
                             "representative_run": selected_nav[0]["run_directory"],
                             "representative_replay": selected_replay[0]["replay_directory"]})
        summary = dict(format_version=1, study="navigation_dependencies", seeds=list(seeds),
                       settings=settings, definitions=definitions, limitations=limitations,
                       cases=output_cases, runs=nav_rows, replays=replay_rows,
                       elapsed_wall_time_s=time.perf_counter() - started)
        _json(directory / "summary.json", summary)
    return _jsonable(summary)
