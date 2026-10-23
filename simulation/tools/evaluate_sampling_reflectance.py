#!/usr/bin/env python3
"""Generate fixed-angle sampling-density / target-reflectance numerical study."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lidar_project.experiments.sampling_reflectance import main

if __name__ == "__main__":
    main()
