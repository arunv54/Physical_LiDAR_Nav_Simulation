"""Independent threshold checks using real rays and frozen navigation viewpoints.

This experiment changes attenuation, not geometry or receiver settings. Every
trial calls the complete one-beam LiDAR scanner against the original opaque
navigation scene. The closed-loop path is not rerun by this measurement test.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np

from .. import api
from ..paths import DATA_DIR
from ..trajectory.configuration import effective_config
from ..trajectory.engine import mounted_sensor_pose
from .navigation_dependencies import TARGET_LABEL, _csv, _json, _scene, emitted_rays


MEAN_SNRS = (6., 7., 8., 9., 9.5, 10., 10.5, 11., 12., 13., 14.)


def expanded_power(cfg, distance_m, cosine, rho_d, extinction_per_m=0.):
    """Expanded diffuse aperture equation, independent of received_power()."""
    aperture_area = math.pi * cfg.aperture_diam_m ** 2 / 4
    return (cfg.peak_power_w * aperture_area * cfg.optical_efficiency
            * rho_d / math.pi * cosine / distance_m ** 2
            * math.exp(-2 * extinction_per_m * distance_m))


def wilson_interval(detections, trials, z=1.959963984540054):
    """Pointwise binomial interval, including zero and all-success samples."""
    if trials < 1 or not 0 <= detections <= trials:
        raise ValueError("Binomial counts require 0 <= detections <= trials and trials > 0")
    p = detections / trials
    den = 1 + z * z / trials
    middle = (p + z * z / (2 * trials)) / den
    half = z / den * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials ** 2))
    return max(0., middle - half), min(1., middle + half)


def summarize_trials(detected, range_error_m, measured_snr, cfg, expected_pd):
    """Keep missed trials in the denominator; range scores condition on returns."""
    detected = np.asarray(detected, dtype=bool)
    error, snr = np.asarray(range_error_m, float), np.asarray(measured_snr, float)
    if len(detected) < 1 or error.shape != detected.shape or snr.shape != detected.shape:
        raise ValueError("Trial arrays must have equal, nonempty shapes")
    if np.any(~np.isfinite(error[detected])) or np.any(~np.isfinite(snr[detected])):
        raise ValueError("Detected returns require finite range errors and measured SNR")
    n, k = len(detected), int(detected.sum())
    low, high = wilson_interval(k, n)
    timing = (.5 * 299_792_458. * cfg.pulse_rise_ns * 1e-9) ** 2
    variance = cfg.range_floor_sigma_m ** 2 + timing / (2 * snr[detected])
    errors = error[detected]
    p = float(expected_pd)
    row = dict(trials=n, detections=k, misses=n-k, expected_pd=p, observed_pd=k/n,
               wilson_low=low, wilson_high=high,
               model_in_pointwise_wilson=bool(low <= p <= high),
               detection_residual=k/n-p,
               binomial_standard_error=math.sqrt(p*(1-p)/n),
               standardized_count_residual=(k-n*p)/math.sqrt(n*p*(1-p)) if 0 < p < 1 else None,
               brier_score=float(np.mean((detected.astype(float)-p)**2)),
               expected_brier_score=p*(1-p),
               range_bias_m=float(errors.mean()) if k else None,
               range_rmse_m=float(np.sqrt(np.mean(errors**2))) if k else None,
               predicted_conditional_range_rmse_m=float(np.sqrt(variance.mean())) if k else None,
               normalized_range_rmse=float(np.sqrt(np.mean(errors**2/variance))) if k else None,
               range_floor_sigma_m=float(cfg.range_floor_sigma_m))
    return row


def pool_settings(rows):
    """Pool Bernoulli calibration with counts, without treating misses as ranges."""
    if not rows:
        raise ValueError("At least one setting is required")
    trials = sum(row['trials'] for row in rows)
    returns = sum(row['detections'] for row in rows)
    rmse_sq = sum(row['trials']*row['detection_residual']**2 for row in rows)/trials
    eligible_range = [row for row in rows if row['detections']]
    return dict(settings=len(rows), trials=trials, detections=returns,
                calibration_rmse=math.sqrt(rmse_sq),
                max_abs_detection_residual=max(abs(row['detection_residual']) for row in rows),
                pointwise_wilson_contains_model=sum(row['model_in_pointwise_wilson'] for row in rows),
                mean_brier_score=sum(row['trials']*row['brier_score'] for row in rows)/trials,
                expected_mean_brier_score=sum(row['trials']*row['expected_brier_score'] for row in rows)/trials,
                conditional_range_rmse_m=math.sqrt(sum(row['detections']*row['range_rmse_m']**2 for row in eligible_range)/returns) if returns else None,
                predicted_conditional_range_rmse_m=math.sqrt(sum(row['detections']*row['predicted_conditional_range_rmse_m']**2 for row in eligible_range)/returns) if returns else None,
                normalized_range_rmse=math.sqrt(sum(row['detections']*row['normalized_range_rmse']**2 for row in eligible_range)/returns) if returns else None)


def prepare_design(source_dir=DATA_DIR / 'navigation_dependencies', mean_snrs=MEAN_SNRS):
    """Freeze three approach viewpoints and one original target ray per viewpoint."""
    source = Path(source_dir).expanduser().resolve()
    source_manifest = source / 'manifest.json'
    config = json.loads(source_manifest.read_text(encoding='utf-8'))['baseline_config']
    effective, _ = effective_config(config, None)
    sensor_cfg = api.build_sensor(effective['_effective_sensor'])
    if sensor_cfg.n_beam_samples != 1 or not sensor_cfg.noise_enabled:
        raise ValueError('Threshold study requires baseline pencil rays and enabled noise')
    scene = _scene(effective)
    target_id = scene.labels.index(TARGET_LABEL)
    target = scene.objects[target_id]
    if target.material.rho_s != 0 or target.material.rho_d <= 0:
        raise ValueError('Threshold study target must be positive pure diffuse')
    target_spec = next(obj for obj in config['world']['unexpected_objects'] if obj['label'] == TARGET_LABEL)
    center = np.asarray(target_spec['center'], float)
    size = np.asarray(target_spec['size'], float)
    source_poses = source / 'reference_poses_seed_1.npz'
    with np.load(source_poses) as poses:
        t, xy, theta = (np.asarray(poses[key]) for key in ('t', 'xy', 'theta'))
    candidates = []
    for step in range(len(t)):
        # Approach-only viewpoints avoid confusing receding and approaching rays.
        if xy[step, 0] >= center[0] - size[0]/2:
            continue
        pose = mounted_sensor_pose(effective, xy[step], theta[step])
        origins, directions = emitted_rays(sensor_cfg, pose)
        ranges, labels, normals = scene.cast(origins, directions)
        eligible = (labels == target_id) & (ranges >= sensor_cfg.min_range_m) & (ranges <= sensor_cfg.max_range_m)
        ids = np.flatnonzero(eligible)
        if not len(ids):
            continue
        hits = origins[ids] + ranges[ids, None]*directions[ids]
        ray_id = int(ids[np.argmin(np.sum(((hits-center)/size)**2, axis=1))])
        n_channels = sensor_cfg.n_channels
        az_id, channel = divmod(ray_id, n_channels)
        cosine = float(np.clip(-normals[ray_id]@directions[ray_id], 0., 1.))
        candidates.append(dict(step=step, time_s=float(t[step]), robot_xy=xy[step].tolist(),
                               robot_theta_rad=float(theta[step]), sensor_pose=asdict(pose),
                               ray_id=ray_id, azimuth_index=az_id, channel_index=channel,
                               azimuth_deg=float(sensor_cfg.azimuths_deg()[az_id]),
                               elevation_deg=float(sensor_cfg.elevations_deg()[channel]),
                               origin=origins[ray_id].tolist(), direction=directions[ray_id].tolist(),
                               target_point=hits[np.flatnonzero(ids == ray_id)[0]].tolist(),
                               range_m=float(ranges[ray_id]), incidence_cosine=cosine,
                               incidence_deg=math.degrees(math.acos(cosine)), eligible_target_rays=int(len(ids))))
    if len(candidates) < 3:
        raise ValueError('Saved baseline route does not contain three eligible target viewpoints')
    selected = []
    for pose_id, anchor in enumerate((8., 5., 2.)):
        choices = [c for c in candidates if c['step'] not in {s['step'] for s in selected}]
        choice = min(choices, key=lambda c: (abs(c['range_m']-anchor), c['step'])).copy()
        choice.update(pose_id=pose_id, label=f"View {pose_id+1}: {choice['range_m']:.2f} m")
        selected.append(choice)
    settings = []
    for view in selected:
        ray_cfg = api.replace(sensor_cfg, n_channels=1, vfov_lower_deg=view['elevation_deg'],
                              vfov_upper_deg=view['elevation_deg'], measurements_per_cycle=1,
                              hfov_start_deg=view['azimuth_deg'], hfov_end_deg=view['azimuth_deg']+.001,
                              rain_mm_per_h=0.)
        pose = api.SensorPose(**view['sensor_pose'])
        origin, direction = emitted_rays(ray_cfg, pose)
        if not np.allclose(direction[0], view['direction'], atol=1e-14):
            raise AssertionError('One-beam extraction changed the original navigation ray')
        distance, labels, _ = scene.cast(origin, direction)
        if labels[0] != target_id or not np.isclose(distance[0], view['range_m'], atol=1e-12):
            raise AssertionError('Extracted ray does not hit original nearest opaque target')
        unattenuated = expanded_power(ray_cfg, view['range_m'], view['incidence_cosine'], target.material.rho_d)
        for mean_snr in mean_snrs:
            if not math.isfinite(mean_snr) or mean_snr <= 0:
                raise ValueError('Mean SNR must be positive and finite')
            kappa = math.log(unattenuated/(ray_cfg.noise_rms_w*mean_snr))/(2*view['range_m'])
            if kappa < 0:
                raise ValueError('Requested SNR would require negative extinction')
            cfg = api.replace(ray_cfg, air_extinction_per_m=kappa)
            predicted_power = expanded_power(cfg, view['range_m'], view['incidence_cosine'], target.material.rho_d, kappa)
            predicted_snr = predicted_power/cfg.noise_rms_w
            expected_pd = .5*math.erfc((cfg.snr_threshold-predicted_snr)/math.sqrt(2.))
            settings.append(dict(setting_id=len(settings), pose_id=view['pose_id'], pose_label=view['label'],
                                 mean_snr=float(predicted_snr), requested_mean_snr=float(mean_snr),
                                 expected_pd=expected_pd, predicted_power_w=predicted_power,
                                 extinction_per_m=kappa, range_m=view['range_m'],
                                 incidence_deg=view['incidence_deg'], sensor_config=asdict(cfg)))
    design = dict(format_version=1, study='navigation_threshold', source_dir=str(source),
                  source_pose_file=str(source_poses), source_pose_sha256=hashlib.sha256(source_poses.read_bytes()).hexdigest(),
                  source_manifest_sha256=hashlib.sha256(source_manifest.read_bytes()).hexdigest(),
                  target_label=TARGET_LABEL, target=target_spec, source_config=config,
                  views=selected, settings=settings,
                  assumptions=[
                      'Three original eligible rays at frozen seed-1 approach poses; no navigation/controller reruns.',
                      'Only air extinction changes within each view; pure diffuse target, receiver noise and threshold remain fixed.',
                      'Extinction is solved from the independent optical equation to span the detector transition; values are illustrative, not calibrated weather.',
                      'Every trial calls LidarSensor.scan against the complete original nearest-opaque-hit scene.',
                      'Detection is conditioned on exactly one eligible target echo per trial, including all misses in the denominator.',
                      'Gaussian threshold probabilities and range uncertainty are assumed model consistency checks, not physical-device validation.',
                      'Wilson intervals are pointwise; simultaneous coverage of all settings is not asserted.',
                      'Power/SNR/range arrays are NaN on misses because the scanner provides no recorded return for those trials.',
                  ])
    return design, scene


def run_study(directory, source_dir=DATA_DIR / 'navigation_dependencies', trials_per_setting=1000):
    """Run independent single-ray scanner trials; save design before drawing noise."""
    if isinstance(trials_per_setting, bool) or int(trials_per_setting) != trials_per_setting or trials_per_setting < 1:
        raise ValueError('trials_per_setting must be a positive integer')
    trials_per_setting = int(trials_per_setting)
    destination = Path(directory).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    (destination/'trials').mkdir(exist_ok=True)
    design, scene = prepare_design(source_dir)
    stride = max(10000, trials_per_setting)
    design.update(trials_per_setting=trials_per_setting,
                  seed_scheme=f'7000000 + setting_id*{stride} + trial_index; disjoint seed labels for every trial')
    _json(destination/'design.json', design)
    target_id = scene.labels.index(TARGET_LABEL)
    rows = []
    started = time.perf_counter()
    for setting in design['settings']:
        cfg = api.LidarConfig(**setting['sensor_config'])
        view = design['views'][setting['pose_id']]
        pose = api.SensorPose(**view['sensor_pose'])
        seed = 7_000_000 + setting['setting_id']*stride + np.arange(trials_per_setting, dtype=np.int64)
        detected = np.zeros(trials_per_setting, bool)
        labels = np.full(trials_per_setting, -1, np.int32)
        measured_power, measured_snr, errors, true_range, sigma = [np.full(trials_per_setting, np.nan) for _ in range(5)]
        for trial in range(trials_per_setting):
            cloud = api.LidarSensor(cfg, pose, seed=int(seed[trial])).scan(scene)
            if cloud.n_beams != 1 or cloud.n > 1:
                raise AssertionError('Single eligible ray must produce zero or one return')
            if cloud.n:
                if cloud.label_id[0] != target_id:
                    raise AssertionError('Trial returned a non-target object despite fixed nearest-hit eligibility')
                if not np.isclose(cloud.range_true_m[0], view['range_m'], atol=1e-11):
                    raise AssertionError('Trial true range changed from prepared geometry')
                detected[trial], labels[trial] = True, target_id
                measured_power[trial], measured_snr[trial] = cloud.power_w[0], cloud.snr[0]
                errors[trial] = cloud.range_m[0]-cloud.range_true_m[0]
                true_range[trial] = cloud.range_true_m[0]
                sigma[trial] = math.sqrt(cfg.range_floor_sigma_m**2 + (.5*299_792_458.*cfg.pulse_rise_ns*1e-9)**2/(2*cloud.snr[0]))
        relative = f"trials/setting_{setting['setting_id']:02d}.npz"
        np.savez_compressed(destination/relative, seed=seed, detected=detected, label_id=labels,
                            measured_power_w=measured_power, measured_snr=measured_snr,
                            range_error_m=errors, true_range_m=true_range, predicted_sigma_m=sigma)
        row = summarize_trials(detected, errors, measured_snr, cfg, setting['expected_pd'])
        row.update(setting_id=setting['setting_id'], pose_id=setting['pose_id'], pose_label=setting['pose_label'],
                   mean_snr=setting['mean_snr'], extinction_per_m=setting['extinction_per_m'],
                   range_m=view['range_m'], incidence_deg=view['incidence_deg'], trials_file=relative)
        rows.append(row)
        print(f"navigation threshold: view {view['pose_id']+1}, mean SNR {setting['mean_snr']:.1f}: "
              f"{row['detections']}/{row['trials']}, model {row['expected_pd']:.4f}", flush=True)
    summary = dict(format_version=1, study='navigation_threshold', design_file='design.json',
                   trials_per_setting=trials_per_setting, views=design['views'], settings=rows,
                   aggregate=pool_settings(rows), assumptions=design['assumptions'],
                   elapsed_wall_time_s=time.perf_counter()-started)
    _json(destination/'summary.json', summary)
    _csv(destination/'settings.csv', rows)
    return summary
