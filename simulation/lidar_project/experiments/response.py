"""Controlled target, atmospheric-loss, detection and uncertainty experiments.

These experiments verify the implemented assumptions against independently
evaluated analytical expressions. They do not calibrate a physical LiDAR or
establish real material reflectance or weather-response accuracy.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import math
from pathlib import Path
import time

import numpy as np

from .. import api
from ..paths import DEFAULT_SENSOR_CONFIG

SEED_BASE = 20261017000000
TRIALS_PER_BIN = 1000
RHO_VALUES = (0.05, 0.10)
RATIOS = np.linspace(0.9, 1.1, 9)
WEATHER = (
    ("clear", {"air_extinction_per_m": 2e-4, "rain_mm_per_h": 0.0}),
    ("heavy_rain", {"air_extinction_per_m": 2e-4, "rain_mm_per_h": 50.0}),
    ("dense_fog", {"air_extinction_per_m": 0.6, "rain_mm_per_h": 0.0}),
)


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, np.ndarray)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    return value


def _write_json(path, value):
    path.write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path, rows):
    if not rows:
        raise ValueError("An experiment table cannot be empty")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _extinction(cfg):
    # Evaluated from the specified inputs, rather than the sensor's derived property.
    return cfg.air_extinction_per_m + cfg.rain_a * cfg.rain_mm_per_h ** cfg.rain_b


def reference_power(cfg, rho, range_m, incidence_deg=0.0):
    """Independent diffuse, beam-filling, normal/oblique planar link calculation."""
    angle = math.radians(incidence_deg)
    # Aperture area pi*D^2/4 cancels the diffuse BRDF's factor pi.
    coefficient = cfg.peak_power_w * cfg.optical_efficiency * cfg.aperture_diam_m ** 2 * rho / 4.0
    return coefficient * max(math.cos(angle), 0.0) * math.exp(-2 * _extinction(cfg) * range_m) / range_m ** 2


def reference_r50(cfg, rho):
    """Bisect the independent link expression at mean echo/noise = threshold."""
    target = cfg.snr_threshold * cfg.noise_rms_w
    lo, hi = cfg.min_range_m, cfg.max_range_m
    if not reference_power(cfg, rho, lo) > target > reference_power(cfg, rho, hi):
        raise ValueError("The controlled single-echo 50% point is outside the sensing gate")
    for _ in range(90):
        mid = (lo + hi) / 2
        if reference_power(cfg, rho, mid) > target:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def wilson_interval(successes, trials):
    """Two-sided 95% Wilson score interval for a binomial proportion."""
    z = 1.959963984540054
    p = successes / trials
    denominator = 1 + z * z / trials
    center = (p + z * z / (2 * trials)) / denominator
    half = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / denominator
    return (0.0 if successes == 0 else max(0.0, center - half),
            1.0 if successes == trials else min(1.0, center + half))


def _pencil_config(**overrides):
    cfg = api.LidarConfig(
        n_channels=1, vfov_lower_deg=0.0, vfov_upper_deg=0.0,
        hfov_start_deg=0.0, hfov_end_deg=0.01, measurements_per_cycle=1,
        n_beam_samples=1)
    return api.replace(cfg, **overrides)


def _plane(range_m, rho, incidence_deg=0.0):
    theta = math.radians(incidence_deg)
    return api.Scene([api.Plane(
        (range_m, 0.0, 0.0), (-math.cos(theta), math.sin(theta), 0.0),
        api.Material(f"illustrative_diffuse_{rho:g}", rho), "plane")])


def _law_row(case, weather, cfg, rho, range_m, incidence_deg):
    pc = api.LidarSensor(cfg, seed=0).scan(_plane(range_m, rho, incidence_deg))
    if pc.n_beams != 1 or pc.n != 1:
        raise RuntimeError("A power-law test must retain exactly its eligible on-axis pencil-beam echo")
    predicted = reference_power(cfg, rho, range_m, incidence_deg)
    measured = float(pc.power_w[0])
    return dict(case=case, weather=weather, rho=rho, range_m=range_m,
                incidence_deg=incidence_deg, extinction_per_m=_extinction(cfg),
                predicted_power_w=predicted, measured_power_w=measured,
                relative_error=(measured - predicted) / predicted,
                measured_range_m=float(pc.range_m[0]))


def _deterministic_laws(directory):
    # Retain weak echoes solely to inspect their optical power, rather than the detector.
    cfg = _pencil_config(noise_enabled=False, snr_threshold=1e-30)
    rows = []
    for rho in (0.05, 0.30, 0.85):
        for distance in (5, 10, 15, 20, 30, 40, 60, 80, 100, 125, 150):
            rows.append(_law_row("distance", "clear", cfg, rho, float(distance), 0.0))
    for angle in (0, 15, 30, 45, 60, 75):
        rows.append(_law_row("incidence", "clear", cfg, 0.30, 10.0, float(angle)))
    for weather, overrides in WEATHER:
        current = api.replace(cfg, **overrides)
        for distance in (2, 3, 4, 5, 6, 8, 10, 15, 20, 25, 30):
            rows.append(_law_row("extinction", weather, current, 0.30, float(distance), 0.0))
    _write_csv(directory / "laws.csv", rows)
    return rows, cfg


def _monte_carlo(directory):
    cfg = _pencil_config(noise_enabled=True)
    cases = [(weather, overrides, rho) for weather, overrides in WEATHER for rho in RHO_VALUES]
    shape = (len(cases), len(RATIOS), TRIALS_PER_BIN)
    seeds = np.empty(shape, dtype=np.int64)
    detected = np.zeros(shape, dtype=bool)
    errors = np.full(shape, np.nan)
    echo_errors = np.full(shape, np.nan)
    measured_snr = np.full(shape, np.nan)
    measured_power = np.full(shape, np.nan)
    predicted_sigma = np.full(shape, np.nan)
    normalized = np.full(shape, np.nan)
    ranges = np.empty(shape[:2])
    expected_probabilities = np.empty(shape[:2])
    detection_rows, uncertainty_rows, r50_rows = [], [], []
    pose = api.SensorPose(position=(0.0, 0.0, 0.0))
    for case_id, (weather, overrides, rho) in enumerate(cases):
        current = api.replace(cfg, **overrides)
        r50 = reference_r50(current, rho)
        r50_rows.append(dict(case_id=case_id, weather=weather, rho=rho,
                             extinction_per_m=_extinction(current), r50_m=r50))
        for bin_id, ratio in enumerate(RATIOS):
            distance = float(ratio * r50)
            ranges[case_id, bin_id] = distance
            if not current.min_range_m <= distance <= current.max_range_m:
                raise RuntimeError("Monte Carlo reference distance is outside the range gate")
            scene = _plane(distance, rho)
            power = reference_power(current, rho, distance)
            snr = power / current.noise_rms_w
            probability = 0.5 * math.erfc((current.snr_threshold - snr) / math.sqrt(2))
            expected_probabilities[case_id, bin_id] = probability
            for trial_id in range(TRIALS_PER_BIN):
                seed = SEED_BASE + case_id * 1000000 + bin_id * 10000 + trial_id
                seeds[case_id, bin_id, trial_id] = seed
                pc = api.LidarSensor(current, pose, seed=seed).scan(scene)
                if pc.n_beams != 1:
                    raise RuntimeError("Each trial must have precisely one eligible, normal-incidence beam")
                if not pc.n:
                    continue
                detected[case_id, bin_id, trial_id] = True
                error = float(pc.range_m[0] - distance)
                echo_error = float(pc.range_m[0] - pc.range_true_m[0])
                amplitude_snr = float(pc.power_w[0] / current.noise_rms_w)
                sigma = math.sqrt(current.range_floor_sigma_m ** 2 +
                                  (299792458.0 * current.pulse_rise_ns * 1e-9 / 2) ** 2 /
                                  (2 * amplitude_snr))
                errors[case_id, bin_id, trial_id] = error
                echo_errors[case_id, bin_id, trial_id] = echo_error
                measured_snr[case_id, bin_id, trial_id] = amplitude_snr
                measured_power[case_id, bin_id, trial_id] = pc.power_w[0]
                predicted_sigma[case_id, bin_id, trial_id] = sigma
                normalized[case_id, bin_id, trial_id] = echo_error / sigma
            success = int(detected[case_id, bin_id].sum())
            low, high = wilson_interval(success, TRIALS_PER_BIN)
            observed = success / TRIALS_PER_BIN
            detection_rows.append(dict(
                case_id=case_id, weather=weather, rho=rho, extinction_per_m=_extinction(current),
                range_m=distance, r_over_r50=float(ratio), r50_m=r50,
                trials=TRIALS_PER_BIN, detections=success, predicted_power_w=power,
                expected_snr=snr, predicted_p=probability, observed_p=observed,
                ci_low=low, ci_high=high, probability_residual=observed - probability))
            mask = detected[case_id, bin_id]
            err = errors[case_id, bin_id, mask]
            sig = predicted_sigma[case_id, bin_id, mask]
            standardized = normalized[case_id, bin_id, mask]
            rms = float(np.sqrt(np.mean(err ** 2))) if success else np.nan
            sigma_rms = float(np.sqrt(np.mean(sig ** 2))) if success else np.nan
            uncertainty_rows.append(dict(
                case_id=case_id, weather=weather, rho=rho, range_m=distance,
                r_over_r50=float(ratio), expected_snr=snr, detections=success,
                bias_m=float(np.mean(err)) if success else np.nan, rmse_m=rms,
                predicted_sigma_rms_m=sigma_rms,
                expected_sigma_at_mean_snr_m=math.sqrt(
                    current.range_floor_sigma_m ** 2 +
                    (299792458.0 * current.pulse_rise_ns * 1e-9 / 2) ** 2 / (2 * snr)),
                normalized_mean=float(np.mean(standardized)) if success else np.nan,
                normalized_std=float(np.std(standardized, ddof=1)) if success > 1 else np.nan,
                rmse_to_sigma_ratio=rms / sigma_rms if success else np.nan))
        print(f"response: completed {weather}, diffuse weight {rho:g}", flush=True)
    _write_csv(directory / "detection.csv", detection_rows)
    _write_csv(directory / "uncertainty.csv", uncertainty_rows)
    np.savez_compressed(
        directory / "trials.npz", case_weather=np.asarray([c[0] for c in cases]),
        case_rho=np.asarray([c[2] for c in cases]), ratios=RATIOS, range_m=ranges,
        predicted_p=expected_probabilities, seeds=seeds, detected=detected,
        range_error_m=errors, range_error_to_echo_m=echo_errors, measured_snr=measured_snr,
        measured_power_w=measured_power, predicted_sigma_m=predicted_sigma, normalized_error=normalized)
    return detection_rows, uncertainty_rows, r50_rows, normalized, errors, predicted_sigma, cfg


def _fixed_scene_weather(directory):
    raw = api.load_config_file(DEFAULT_SENSOR_CONFIG)
    cfg, pose, scene, _, _ = api.build_all(raw)
    cfg = api.replace(cfg, n_channels=16, measurements_per_cycle=720, n_beam_samples=3,
                      vfov_lower_deg=-15.0, vfov_upper_deg=15.0)
    seeds = (4101, 4102, 4103)
    labels = list(scene.labels)
    ground_ids = {i for i, name in enumerate(labels) if name.lower() == "ground"}
    rows = []
    scan_directory = directory / "fixed_scene"
    scan_directory.mkdir(exist_ok=True)
    for weather, overrides in WEATHER:
        current = api.replace(cfg, **overrides)
        for seed in seeds:
            pc = api.LidarSensor(current, pose, seed=seed).scan(scene)
            pc.save_npz(str(scan_directory / f"{weather}_seed{seed}.npz"))
            counts = [int(np.count_nonzero(pc.label_id == i)) for i in range(len(labels))]
            nonground = sum(count for i, count in enumerate(counts) if i not in ground_ids)
            row = dict(weather=weather, seed=seed, extinction_per_m=_extinction(current),
                       n_beams=pc.n_beams, n_returns=pc.n, nonground_returns=nonground,
                       raw_return_yield=pc.n / pc.n_beams)
            row.update({f"returns_{name}": count for name, count in zip(labels, counts)})
            rows.append(row)
    _write_csv(directory / "fixed_scene_weather.csv", rows)
    return rows, dict(base_configuration=raw, sensor=dataclasses.asdict(cfg),
                      pose=dataclasses.asdict(pose), seeds=seeds,
                      note="Same static scene and pose; only atmospheric loss and independent noise seeds change.")


def run_response(directory):
    """Run and save controlled response verification; return a JSON-safe summary."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    laws, law_cfg = _deterministic_laws(directory)
    detection, uncertainty, r50, normalized, errors, sigmas, mc_cfg = _monte_carlo(directory)
    scene_rows, fixed_manifest = _fixed_scene_weather(directory)
    manifest = dict(
        experiment="Controlled 3D pulsed-LiDAR response verification",
        seed_base=SEED_BASE,
        deterministic=dict(
            sensor=dataclasses.asdict(law_cfg),
            note="K=1, one on-axis beam, noise disabled; threshold 1e-30 retains weak returns solely for optical-power inspection.",
            distance_m=[5, 10, 15, 20, 30, 40, 60, 80, 100, 125, 150],
            rho_values=[0.05, 0.30, 0.85], incidence_deg=[0, 15, 30, 45, 60, 75]),
        monte_carlo=dict(
            sensor=dataclasses.asdict(mc_cfg), rho_values=RHO_VALUES, ratios=RATIOS,
            trials_per_bin=TRIALS_PER_BIN, snr_threshold=mc_cfg.snr_threshold,
            range_gate_m=[mc_cfg.min_range_m, mc_cfg.max_range_m],
            seed_scheme="seed_base + case_id*1000000 + bin_id*10000 + trial_id; all trial seeds unique",
            detection_reference="Phi(P_reference/noise_rms - snr_threshold); one eligible echo per trial",
            uncertainty_reference="sigma^2 = floor^2 + (c*t_rise/2)^2/(2*measured_power/noise_rms), conditional on detection"),
        fixed_scene=fixed_manifest, weather=[dict(name=name, overrides=values) for name, values in WEATHER],
        limitations=[
            "Verification of an implemented model against its stated equations, not physical validation.",
            "Diffuse weights are illustrative model inputs, not calibrated real material reflectances.",
            "Weather changes attenuation only; there are no droplet returns, fog backscatter or refraction.",
            "Detection conditions on one known, visible, beam-filling target inside sensing gates.",
            "Range uncertainty conditions on detected noisy power, so the detected-power distribution is truncated.",
            "Fixed-scene raw return yield is not target-conditioned detection probability.",
            "Wilson intervals are pointwise 95% intervals, not simultaneous coverage of all bins.",
        ])
    _write_json(directory / "config_manifest.json", manifest)
    valid = np.isfinite(normalized)
    pooled_z = normalized[valid]
    detected_errors = errors[valid]
    detected_sigmas = sigmas[valid]
    within = sum(row["ci_low"] - 1e-14 <= row["predicted_p"] <= row["ci_high"] + 1e-14 for row in detection)
    weather_summary = []
    for weather, _ in WEATHER:
        rows = [r for r in scene_rows if r["weather"] == weather]
        weather_summary.append(dict(
            weather=weather, seeds=len(rows), n_beams=rows[0]["n_beams"],
            mean_returns=float(np.mean([r["n_returns"] for r in rows])),
            min_returns=min(r["n_returns"] for r in rows), max_returns=max(r["n_returns"] for r in rows),
            mean_nonground_returns=float(np.mean([r["nonground_returns"] for r in rows])),
            mean_raw_return_yield=float(np.mean([r["raw_return_yield"] for r in rows]))))
    summary = dict(
        experiment="response", output_directory=str(directory),
        radiometry=dict(rows=len(laws), max_abs_relative_power_error=max(abs(r["relative_error"]) for r in laws),
                        max_abs_range_error_m=max(abs(r["measured_range_m"] - r["range_m"]) for r in laws)),
        detection=dict(cases=len(r50), bins=len(detection), independent_trials=len(detection) * TRIALS_PER_BIN,
                       max_abs_probability_residual=max(abs(r["probability_residual"]) for r in detection),
                       analytic_probability_within_wilson95=within, r50=r50),
        uncertainty=dict(
            detected_trials=int(valid.sum()), pooled_normalized_mean=float(np.mean(pooled_z)),
            pooled_normalized_std=float(np.std(pooled_z, ddof=1)),
            pooled_range_bias_m=float(np.mean(detected_errors)),
            pooled_range_rmse_m=float(np.sqrt(np.mean(detected_errors ** 2))),
            pooled_predicted_sigma_rms_m=float(np.sqrt(np.mean(detected_sigmas ** 2))),
            normalized_fraction_within_1_96=float(np.mean(np.abs(pooled_z) <= 1.959963984540054))),
        fixed_scene_weather=weather_summary, elapsed_seconds=time.perf_counter() - started,
        interpretation="Implemented-assumption verification only; no physical calibration or measured-device validation claimed.")
    summary = _jsonable(summary)
    _write_json(directory / "summary.json", summary)
    return summary
