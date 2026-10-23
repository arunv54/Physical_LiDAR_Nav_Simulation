# LiDAR midterm project

Physics-based 3D LiDAR simulation, fixed-scene measurement evaluations, and the
course report. Numerical simulation and presentation code are separate.

**Project team:** Arunkumar Venkatesan, Xuyi Lian, and Apurva Singh.

## Read the report

[Open the live visual report](https://xuyilian.github.io/MAE5810-LiDAR-Midterm/),
or open `index.html` directly. The page works offline and presents saved
measurements, equations, 3D FOV/LOS geometry, trajectories, and growing maps.
The editable report and PDF are in `outputs/reports/`.
The native LaTeX source and its compile guide are in [`latex_report/`](latex_report/).

For a local browser address, run with Python 3.10 or newer:

```sh
python3 start.py
```

On macOS, `Start Project.command` launches the same page. Keep its terminal
window open. Use `--port 0` if the default port is occupied, or `--no-browser`
to open the printed address yourself. The server listens at `127.0.0.1`.

## Install generation dependencies

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[visualization,reports]"
```

Video export requires FFmpeg on your executable search path. On macOS:

```sh
brew install ffmpeg
```

PDF export requires LibreOffice and Times New Roman. A managed bundled renderer
is used when available. Otherwise, set the executable path before starting:

```sh
export LIDAR_SOFFICE="/Applications/LibreOffice.app/Contents/MacOS/soffice"
```

Word generation works separately if PDF export is unavailable. PDF export
checks the course page limit before replacing the published file.

## Generate deliverables

Start the local server and open `/builder` at its printed address. Choose
**Generate all deliverables**, or select an individual stage. Leave
**Recompute numerical studies** unchecked to reuse verified measurements;
missing inputs are generated automatically. A fresh numerical build archives
previous owned data in ignored `build/backups/`.

The GitHub checkout includes source, reports, media, summaries, and raw threshold
trials. Large scan and navigation histories regenerate from the supplied
configurations and fixed seeds. The offline HTML and saved videos play without
these raw arrays. Build records and logs remain in ignored `build/`.

The report evaluates sensor pose, sampling density, receiver noise, target
characteristics, and Normal/Rain/Fog navigation in one fixed obstacle setup.
Target and weather comparisons share a baseline; weather comparisons hold the
scene, robot, sensor, controller, and planned route fixed. Mapping uses the known simulated pose.

To reproduce numerical evidence directly:

```sh
python -m lidar_project evaluate
python -m lidar_project evaluate-navigation-dependencies --seeds 1 2 3 --cases baseline reflectance_dark reflectance_bright target_yaw_45 target_yaw_75 target_short target_tall target_near target_far rain fog noise_high
python -m lidar_project evaluate-navigation-threshold --trials 1000
python tools/evaluate_sampling_reflectance.py
```

Render and export the saved report evidence:

```sh
python tools/render_report_figures.py
python tools/export_report.py --use-saved-figures
python tools/build_visual_report.py
```

These presentation commands do not rerun the sensor or navigation. To export
only one document format, use `tools/export_report.py --format docx` or
`--format pdf`.

## Simulation videos

```sh
python tools/generate_environment_video.py
python tools/generate_sensor_state_video.py
python tools/generate_target_features_video.py
```

The generators save MP4 videos and source manifests in
`simulalted_environments/` (the requested folder spelling):

| Video | Comparison |
| --- | --- |
| `environment_conditions.mp4` | Normal, Fog and Rain navigation |
| `sensor_state_best_vs_worst.mp4` | Dense/bright versus sparse/weak target measurements |
| `sampling_reflectance_factorial.mp4` | All four sampling/reflectance combinations |

The fixed-angle comparisons use 360 or 90 azimuth samples and target diffuse
coefficients 0.85 or 0.00005. All mounting angles, FOV, obstacle geometry,
atmosphere and receiver noise stay fixed. Matched-pose replay uses exactly the
same sensor position and heading at each time; the four-case comparison isolates
sampling and target effects. The low diffuse coefficient is an illustrative
weak-return stress setting, not a calibrated material.

Use `--use-saved` to render without simulation, `--regenerate` to rebuild the
relevant numerical inputs, or `--data-dir PATH` to choose another saved study.
The controlled study is stored in `data/sampling_reflectance/`; missing baseline
histories can be regenerated there without replacing report evidence.

The videos display measured scans and accumulating maps alongside faint true
obstacle geometry. Pink points identify detected target returns. The measurement
replays follow recorded baseline poses; separately labelled closed-loop summaries
describe navigation outcomes. Future scans never appear early, and terminal
states create no extra measurement. Weather models attenuation only; physical
calibration, atmospheric backscatter and SLAM remain outside this model.

## Files

| Deliverable | Location |
| --- | --- |
| Offline interactive report | `index.html` |
| Editable report and course PDF | `outputs/reports/` |
| Target and weather figures, trajectories, profiles, maps | `outputs/all_case_report/` |
| Sensor pose and sampling/reflectance figures and tables | `outputs/sensor_dependencies/` |
| Fixed-scene 3D FOV/LOS geometry and figure | `outputs/weather_report/` |
| Threshold detection figure | `outputs/navigation_validation/report_threshold_detection.png` |
| Figure provenance | `outputs/report_figures_manifest.json` |
| Environment and controlled dependency videos | `simulalted_environments/` |
| Submission archive and inventory | `outputs/submission/` |

```text
configs/                         Sensor and trajectory inputs
lidar_project/
  model/                         Geometry, materials, LiDAR measurement physics
  config.py, data.py, api.py      Validated inputs and point-cloud storage
  trajectory/                    Route planning, tracking, recorded runs
  navigation/control.py          Steering geometry used by trajectory tracking
  mapping/                       Measured-point accumulation and storage
  experiments/                   Model checks, fixed-scene studies, controlled factorial study
  visualization/                 Report figures, HTML assets, video renderer
  cli.py                         Numerical generation commands
  workflow.py, project_app.py     Deliverables builder and local server
tools/                           Report, video, and submission generators
tests/                           Numerical, storage, report, and workflow checks
data/                            Saved evidence and regenerated raw histories
outputs/                         Report figures and documents
simulalted_environments/          Generated videos and source manifests
build/                           Ignored caches, logs, and verification artifacts
```

Appearance changes belong in `lidar_project/visualization/`. The HTML report
uses `templates/report.html` and `static/report*`; video appearance is in
`simulation_video.py`. Neither renderer changes the sensor model.

## Verify and package

```sh
python -m unittest discover -s tests -v
python tools/package_project.py
```

To prepare the self-contained LaTeX submission, stage its assets, compile
`latex_report/main.tex` with Tectonic or pdfLaTeX, and package it:

```sh
python tools/package_latex_report.py --stage-only
cd latex_report
tectonic main.tex
cd ..
python tools/package_latex_report.py
```

This creates `outputs/submission/Midterm_Project_LaTeX.zip`, including the native
LaTeX PDF, figures, videos, offline visual report, and simulation source/evidence.

The submission ZIP contains source, configurations, tests, the guide, reports,
figures, videos, CSV/JSON evidence, and threshold trials. Evidence is stored
under `evidence/data/` in the archive. Large navigation histories, caches, and
local logs are omitted; the documented generators rebuild the raw data.
`python run.py` provides the same numerical commands as `python -m lidar_project`.
