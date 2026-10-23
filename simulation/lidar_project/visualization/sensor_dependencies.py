"""Render pose, sampling and target-response evidence from saved studies.

This module never generates measurements. The factorial replay uses the same
poses and fixed mounting angles; only azimuth sample count and the designated
target's diffuse coefficient vary. Pose height has its own analytic check.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

CASES = ("dense_bright", "dense_dark", "sparse_bright", "sparse_dark")
COLORS = ("#2676A5", "#C87637", "#589BBE", "#D9A079")
LABELS = ("360 bright", "360 weak", "90 bright", "90 weak")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _case_rows(summary):
    cases = summary["cases"]
    indexed = cases if isinstance(cases, dict) else {c["id"]: c for c in cases}
    return [indexed[name] for name in CASES]


def _bins(directory, summary, case_id):
    """Pool poses by mean eligible range; this is not per-return range binning."""
    edges = np.array([.5, 1, 2, 3, 5, 10, 200.])
    counts = np.zeros((len(edges)-1, 3), float)
    replays = [r for r in summary["replays"] if r["case_id"] == case_id]
    for replay in replays:
        with (directory / replay["replay_directory"] / "traces.csv").open(newline="") as stream:
            for row in csv.DictReader(stream):
                eligible = int(row["eligible_echoes"])
                if not eligible:
                    continue
                mean_range = float(row["eligible_range_sum_m"]) / eligible
                index = np.searchsorted(edges, mean_range, side="right") - 1
                if 0 <= index < len(counts):
                    counts[index] += (eligible, int(row["detected_returns"]), float(row["expected_detections"]))
    present = counts[:, 0] > 0
    centers = .5 * (edges[:-1] + edges[1:])
    return centers[present], counts[present, 1] / counts[present, 0], counts[present, 2] / counts[present, 0], counts[present, 0].astype(int)


def _compact_weather(data_root, output):
    """Retain full-width readable labels in the report's short weather panel."""
    import matplotlib.pyplot as plt
    from .all_case_report import _weather_replays, _measurement_profiles, CASES as weather_cases, COLORS as weather_colors
    from matplotlib.lines import Line2D

    replays, sources = _weather_replays(data_root / "navigation_dependencies")
    profiles = _measurement_profiles(replays)
    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 7,
                         "axes.labelsize": 7, "axes.titlesize": 7.4,
                         "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "figure.facecolor": "white"}):
        fig, axes = plt.subplots(1, 3, figsize=(6.5, 1.65))
        fig.subplots_adjust(left=.07, right=.99, bottom=.23, top=.74, wspace=.4)
        for case in weather_cases:
            rows = [r for r in profiles if r["case_id"] == case]
            times = [r["time_s"] for r in rows]
            for ax, key, scale in zip(axes, ("mean_target_returns", "mean_eligible_power_w", "pooled_cumulative_target_rms_mm"), (1, 1e6, 1)):
                ax.plot(times, [np.nan if r[key] is None else r[key]*scale for r in rows], lw=.9, color=weather_colors[case])
        for ax, title, label in zip(axes, ("Target detections", "Received power", "Range accuracy"), ("Returns / pose", "Echo power (µW)", "Cumulative RMS (mm)")):
            ax.set(title=title, xlabel="Time (s)", ylabel=label, xlim=(0, 18.2), xticks=(0, 6, 12, 18))
            ax.grid(axis="y", color="#DFE5E8", lw=.4)
            ax.tick_params(length=2, pad=1)
        axes[1].set_yscale("log")
        axes[2].set_ylim(bottom=0)
        handles = [Line2D([0], [0], color=weather_colors[c], lw=1.2, label=label) for c, label in zip(weather_cases, ("Normal", "Rain", "Fog"))]
        fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=7, bbox_to_anchor=(.52, 1.02))
        fig.savefig(output / "weather_measurements_compact.png", dpi=280)
        fig.savefig(output / "weather_measurements_compact.pdf")
        plt.close(fig)
    return sources


def render_sensor_dependencies(data_root, output_dir):
    """Write the report figure, exact metrics CSV and a source-hashed manifest."""
    import matplotlib.pyplot as plt

    data_root, output = Path(data_root).resolve(), Path(output_dir).resolve()
    directory = data_root / "sampling_reflectance"
    summary = _read(directory / "summary.json")
    cases = _case_rows(summary)
    with (data_root / "evaluation/geometry/geometry_pose.csv").open(newline="") as stream:
        height = [r for r in csv.DictReader(stream) if r["experiment"] == "ground_height"]
    if not height:
        raise ValueError("Saved height check is missing")
    navigation = _read(data_root / "navigation_dependencies/summary.json")
    nav_cases = {c["id"]: c for c in navigation["cases"]}
    if "noise_high" not in nav_cases:
        raise ValueError("Generate noise_high for the report's receiver-noise comparison")
    output.mkdir(parents=True, exist_ok=True)
    style = {"font.family": "DejaVu Sans", "font.size": 9.2,
             "axes.titlesize": 10, "axes.labelsize": 9.2,
             "xtick.labelsize": 8.2, "ytick.labelsize": 8.2,
             "axes.edgecolor": "#B8C2C9", "axes.spines.top": False,
             "axes.spines.right": False, "pdf.fonttype": 42,
             "figure.facecolor": "white", "savefig.facecolor": "white"}
    with plt.rc_context(style):
        fig, axes = plt.subplots(2, 2, figsize=(7.8, 3.7), layout="constrained")
        ax = axes[0, 0]
        heights = [float(r["height_m"]) for r in height]
        predicted = [float(r["predicted_range_m"]) for r in height]
        measured = [float(r["measured_range_m"]) for r in height]
        ax.plot(heights, predicted, color="#344B5B", lw=1.4, label="Analytic")
        ax.scatter(heights, measured, s=24, facecolors="white", edgecolors=COLORS[0], zorder=3, label="Simulation")
        ax.set(xlabel="Mount height (m)", ylabel="Ground range (m)", title="(a) Sensor position   fixed angles")
        ax.legend(frameon=False, fontsize=7.4, loc="upper left")
        ax = axes[0, 1]
        for density, marker, tint in (("dense", "o", COLORS[0]), ("sparse", "s", COLORS[2])):
            pair = [next(c for c in cases if c["id"] == density + "_" + level) for level in ("dark", "bright")]
            rho = [float(c["target_diffuse_reflectance"]) for c in pair]
            powers = [c["replay"]["mean_eligible_power_w"] * 1e6 for c in pair]
            ax.loglog(rho, powers, marker=marker, ms=4, color=tint, label="M = " + ("360" if density == "dense" else "90"))
        ax.set(xlabel="Target diffuse coefficient", ylabel="Eligible power (µW)", title="(b) Target response   matched poses")
        ax.legend(frameon=False, fontsize=7.4, loc="upper left")
        ax = axes[1, 0]
        range_data = {}
        for c, color, label in zip(cases, COLORS, LABELS):
            x, observed, expected, count = _bins(directory, summary, c["id"])
            range_data[c["id"]] = dict(mean_pose_range_bin_centers_m=x.tolist(), observed_probability=observed.tolist(), expected_probability=expected.tolist(), eligible_echoes=count.tolist())
            ax.plot(x, expected*100, color=color, lw=1.0, linestyle="--", alpha=.8)
            ax.scatter(x, observed*100, color=color, s=17, marker="o" if c["id"].startswith("dense") else "x", label=label)
        ax.set(xlabel="Mean eligible range at pose (m)", ylabel="Detected / eligible (%)", ylim=(-5, 108), xlim=(.3, 10), title="(c) Position and target   detection")
        ax.legend(frameon=False, fontsize=6.9, ncol=2, loc="center right")
        ax = axes[1, 1]
        x = np.arange(4)
        values = [1000*c["replay"]["range_rmse_m"] for c in cases]
        ax.bar(x, values, color=COLORS, width=.65)
        ax.plot(x, [1000*c["replay"]["predicted_range_rmse_m"] for c in cases], "k_", ms=12, mew=1.5, label="Predicted RMS")
        for index, c in enumerate(cases):
            ax.text(index, values[index] + .5, f"{c['replay']['detected_returns']:,}", ha="center", fontsize=7.1)
        ax.set(xticks=x, xticklabels=LABELS, ylabel="Target range RMS (mm)", ylim=(0, max(values)*1.4), title="(d) Sampling and target   range error")
        ax.tick_params(axis="x", labelsize=7.4)
        ax.legend(frameon=False, fontsize=7.1, loc="upper left")
        for ax in axes.flat:
            ax.grid(axis="y", color="#D9E0E5", lw=.5, alpha=.8)
            ax.set_axisbelow(True)
        fig.savefig(output / "sensor_dependencies.png", dpi=280)
        fig.savefig(output / "sensor_dependencies.pdf")
        plt.close(fig)
    weather_sources = _compact_weather(data_root, output)
    metrics = []
    for c in cases:
        r, a = c["replay"], c["aggregate"]
        metrics.append(dict(case_id=c["id"], total_emitted_beams=r["total_beams"], eligible_target_echoes=r["eligible_echoes"], detected_target_returns=r["detected_returns"], observed_detection_probability=r["observed_detection_probability"], expected_detection_probability=r["expected_detection_probability"], mean_eligible_power_w=r["mean_eligible_power_w"], target_range_rmse_m=r["range_rmse_m"], predicted_target_range_rmse_m=r["predicted_range_rmse_m"], mean_points_per_scan=a["mean_returns_per_control_step"], mean_observed_endpoint_fraction=a["mean_observed_fraction"]))
    with (output / "tables.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    sources = [directory / "summary.json", data_root / "evaluation/geometry/geometry_pose.csv", data_root / "evaluation/geometry/summary.json", data_root / "navigation_dependencies/summary.json"]
    sources.extend(directory / r["replay_directory"] / "traces.csv" for r in summary["replays"])
    sources.extend(weather_sources)
    manifest = dict(schema_version=1, prepared_from_saved_results=True, figure="sensor_dependencies.png", cases=list(CASES), metrics=metrics, range_conditioning=range_data, height_check=dict(heights_m=heights, predicted_ranges_m=predicted, measured_ranges_m=measured, maximum_absolute_error_m=float(np.max(np.abs(np.asarray(measured)-predicted)))), receiver_noise_comparison={c: dict(mean_points_per_scan=nav_cases[c]["aggregate"]["mean_returns_per_control_step"], target_range_rmse_m=nav_cases[c]["replay"]["range_rmse_m"]) for c in ("baseline", "noise_high")}, sources=[dict(path=str(path.relative_to(data_root)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sources], limitations=["Pose bins group snapshots by their mean eligible range, not individual returns; incidence and exposure also vary along the route.", "Bright/weak pairs share geometry at each sample density; different sampling grids can intersect different points.", "Range RMS is conditional on detection, and counts above RMS bars are actual detected-return denominators.", "The very weak diffuse coefficient is an illustrative virtual stress setting, not a calibrated physical material."])
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/sensor_dependencies"))
    arguments = parser.parse_args()
    print(json.dumps(render_sensor_dependencies(arguments.data_dir, arguments.output_dir), indent=2))
