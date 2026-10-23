#!/usr/bin/env python3
"""Generate fixed-angle dense/bright versus sparse/dark measurement video.

Run from the repository: python tools/generate_sensor_state_video.py
The two configured endpoints change azimuth sampling and only the study target's
diffuse reflectance. Mount angles, weather, FOV and all replay poses are fixed.
Use generate_target_features_video.py for four cases that isolate each factor.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar_project.visualization.simulation_video import video_main

if __name__ == "__main__":
    video_main("sensor")
