# 3D LiDAR Simulation and Midterm Project

A physics-based simulation of a spinning 3D LiDAR sensor, with experiments on geometry, surface response, detection, range noise, weather attenuation, endpoint mapping, and closed-loop navigation.

## Repository layout

- `simulation/` — Python source, sensor and scene configuration, tests, and the local interactive project app.
- `simulated_environments/` — rendered environment and sensor comparison videos, with JSON sidecar manifests. This is a standalone media section of the repository.
- `figures/` — plots and diagrams used in the report.
- `main.tex` and `main.pdf` — editable LaTeX report and compiled copy.
- `visual_report.html` — offline interactive report with saved measurements and visualizations.
- `manifest.json` — package inventory and evidence map.
- `LiDAR_Project_Data_And_Outputs.zip` — saved numerical data and generated report outputs. Extract it at the repository root to restore `simulation/data/` and `simulation/outputs/`.

## Run the simulation project

Python 3.10 or newer is required. From the repository root:

```sh
cd simulation
python -m venv .venv
```

Activate the virtual environment, then install the project dependencies and start the local app:

```sh
python -m pip install -e ".[visualization,reports]"
python start.py
```

Open the local address printed by the app. To restore the saved data and report outputs first, run `python -m zipfile -e LiDAR_Project_Data_And_Outputs.zip .` from the repository root. The simulation README contains additional instructions for evaluating experiments, rebuilding figures and reports, and exporting videos. Video export requires FFmpeg.

## Rebuild the report

The LaTeX source uses relative paths to `figures/`. Compile `main.tex` from the repository root with Tectonic or a standard LaTeX installation:

```sh
tectonic main.tex
```

The report describes simulator consistency under its configured assumptions; it does not claim physical-device calibration or general navigation reliability.