#!/usr/bin/env python3
"""Generate one synchronized Normal / Fog / Rain simulation video.

Run from the repository: python tools/generate_environment_video.py
Missing raw archives are regenerated in data/video_inputs/environments, without
changing saved report studies. Use --help for size, seed and encoder options.
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar_project.visualization.simulation_video import video_main

if __name__ == "__main__":
    video_main("environment")
