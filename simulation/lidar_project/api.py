"""Stable numerical facade shared by navigation, commands, and renderers.

Importing this module never imports visualization, a GUI, or a command parser.
"""
from dataclasses import replace

from .config import (
    ConfigError, apply_overrides, build_all, build_pose,
    build_scene, build_sensor, load_config_file,
)
from .data import PointCloud, load_npz
from .model.geometry import (
    EPS, Primitive, Plane, Sphere, OrientedBox, VerticalCylinder, Scene,
    rot_x, rot_y, rot_z, flu_to_ned,
)
from .model.materials import MATERIALS, Material
from .model.sensor import (
    C_LIGHT, LidarConfig, SensorPose, LidarSensor, beam_pattern, received_power,
    max_range_estimate,
)

__all__ = [
    "C_LIGHT", "EPS", "ConfigError", "Material", "MATERIALS",
    "Primitive", "Plane", "Sphere", "OrientedBox", "VerticalCylinder", "Scene",
    "LidarConfig", "SensorPose", "LidarSensor", "PointCloud", "load_npz",
    "rot_x", "rot_y", "rot_z", "flu_to_ned", "beam_pattern", "received_power",
    "max_range_estimate", "load_config_file",
    "apply_overrides", "build_all", "build_sensor", "build_pose", "build_scene",
    "replace",
]
