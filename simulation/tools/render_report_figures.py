"""Render the report's pose, sensor configuration, target and weather figures.

This stage never reruns sensor scans or navigation. The coverage preparation
audits geometry at an existing pose in the unchanged primary scene.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))


def _project_path(value) -> Path:
    path = Path(value).expanduser()
    return (path if path.is_absolute() else PROJECT / path).resolve()


def render_report_figures(data_dir=PROJECT / "data", output_dir=PROJECT / "outputs") -> dict:
    from lidar_project.visualization.all_case_report import render_weather_report
    from lidar_project.visualization.weather_coverage import prepare_weather_coverage, render_weather_coverage
    from lidar_project.visualization.navigation_validation import _threshold
    from lidar_project.visualization.sensor_dependencies import render_sensor_dependencies
    from lidar_project.visualization import theme
    import matplotlib.pyplot as plt

    data, output = _project_path(data_dir), _project_path(output_dir)
    dependency = data / "navigation_dependencies"
    summary_path = data / "navigation_threshold/summary.json"
    if not (dependency / "summary.json").is_file() or not summary_path.is_file():
        raise FileNotFoundError("Generate the saved navigation and threshold studies before rendering report figures")
    weather = render_weather_report(dependency, output / "all_case_report")
    sensor_dependencies = render_sensor_dependencies(data, output / "sensor_dependencies")
    coverage = prepare_weather_coverage(dependency, output / "weather_report")
    render_weather_coverage(coverage, output / "weather_report")
    threshold_output = output / "navigation_validation/report_threshold_detection.png"
    threshold_output.parent.mkdir(parents=True, exist_ok=True)
    threshold_summary = json.loads(summary_path.read_text(encoding="utf-8"))
    with plt.rc_context({**theme.MATPLOTLIB, "font.size": 8.5}):
        threshold = _threshold(threshold_summary, threshold_output)
    pngs = [output / "all_case_report" / (name + ".png") for name in (
        "target_characteristics", "weather_trajectories", "weather_profiles", "weather_maps")]
    pngs.extend((output / "weather_report/weather_fov_los.png", threshold_output,
                 output / "sensor_dependencies/sensor_dependencies.png",
                 output / "sensor_dependencies/weather_measurements_compact.png"))
    sources = [dependency / "summary.json", dependency / "manifest.json", summary_path,
               data / "navigation_threshold/design.json", data / "sampling_reflectance/summary.json"]
    manifest = dict(
        schema_version=1,
        scope="Sensor pose and noise checks; four sampling/reflectance combinations; nine target settings and three fixed-scene weather conditions.",
        prepared_from_saved_results=True,
        figure_count=len(pngs),
        figures=[dict(path=path.relative_to(output).as_posix(), bytes=path.stat().st_size,
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in pngs],
        weather_manifest="all_case_report/manifest.json",
        coverage_geometry="weather_report/fov_los.json",
        weather_runs=weather["runs"],
        target_settings=weather["target_characteristics"]["conditions"],
        sensor_dependencies_manifest="sensor_dependencies/manifest.json",
        sampling_reflectance_settings=len(sensor_dependencies["cases"]),
        threshold=threshold,
        sources=[dict(path=path.relative_to(data).as_posix(),
                      sha256=hashlib.sha256(path.read_bytes()).hexdigest()) for path in sources],
    )
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "report_figures_manifest.json"
    destination.write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return dict(output_dir=str(output), manifest=str(destination), figures=len(pngs),
                weather_runs=weather["runs"], target_settings=manifest["target_settings"],
                threshold_trials=threshold["trials"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT / "data")
    parser.add_argument("--output-dir", type=Path, default=PROJECT / "outputs")
    args = parser.parse_args()
    print(json.dumps(render_report_figures(args.data_dir, args.output_dir), indent=2))


if __name__ == "__main__":
    main()
