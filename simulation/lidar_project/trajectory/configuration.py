"""Portable input loading for the separate reference-tracking experiment."""
from __future__ import annotations

import copy
import math
from pathlib import Path

from .. import api
from ..paths import CONFIG_DIR


def load_config(path=None):
    path = Path(path or CONFIG_DIR / "trajectory.jsonc").expanduser().resolve()
    config = api.load_config_file(path)
    if "_sensor_base_snapshot" not in config:
        base = Path(config.get("base_config", "sensor.jsonc"))
        if not base.is_absolute():
            base = path.parent / base
        config["_sensor_base_snapshot"] = copy.deepcopy(api.load_config_file(base).get("sensor", {}))
    config["_source"] = str(path)
    return config


def effective_config(config, scenario):
    result = copy.deepcopy(config)
    if isinstance(scenario, str):
        matches = [s for s in result.get("scenarios", []) if s["name"] == scenario]
        if not matches:
            raise ValueError(f"Unknown trajectory scenario {scenario!r}")
        scenario = matches[0]
    scenario = scenario or {"name": "baseline"}
    for key in ("robot", "controller"):
        result[key].update(scenario.get(key, {}))
    if "mount_height_m" in scenario:
        result["mount_height_m"] = scenario["mount_height_m"]
    mount = {"offset_xy_m": [0.0, 0.0], "roll_deg": 0.0, "pitch_deg": 0.0, "yaw_deg": 0.0}
    for source in (result.get("sensor_mount", {}), scenario.get("sensor_mount", {})):
        if not isinstance(source, dict):
            raise ValueError("sensor_mount must be an object")
        unknown = set(source) - set(mount)
        if unknown:
            raise ValueError(f"Unknown sensor_mount fields: {sorted(unknown)}")
        mount.update(source)
    try:
        offset = mount["offset_xy_m"]
        if not isinstance(offset, (list, tuple)) or len(offset) != 2:
            raise ValueError("sensor_mount.offset_xy_m must contain two finite numbers")
        mount["offset_xy_m"] = [float(value) for value in offset]
        for field in ("roll_deg", "pitch_deg", "yaw_deg"):
            mount[field] = float(mount[field])
        values = mount["offset_xy_m"] + [mount[field] for field in ("roll_deg", "pitch_deg", "yaw_deg")]
        height = float(result["mount_height_m"])
        if not all(math.isfinite(value) for value in values) or not math.isfinite(height) or height < 0:
            raise ValueError("Sensor mounting fields must be finite, with nonnegative mount_height_m")
    except (TypeError, OverflowError) as error:
        raise ValueError("Sensor mounting fields must be finite numbers") from error
    result["sensor_mount"] = mount
    if "lidar_enabled" in scenario:
        result["lidar_enabled"] = scenario["lidar_enabled"]
    sensor = {**result.get("_sensor_base_snapshot", {}), **result.get("sensor_overrides", {}), **scenario.get("sensor", {})}
    result["_effective_sensor"] = sensor
    return result, str(scenario.get("name", "baseline"))
