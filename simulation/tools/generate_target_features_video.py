#!/usr/bin/env python3
"""Generate a four-panel matched-pose sampling / reflectance factorial video.

Run: python tools/generate_target_features_video.py
Columns change only target diffuse reflectance; rows change only azimuth sample
count. Numerical results are generated if missing, independently of rendering.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar_project.visualization.simulation_video import video_main

if __name__ == "__main__":
    video_main("factorial")
