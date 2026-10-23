"""The single fixed obstacle environment used by report and video experiments."""
from __future__ import annotations

import copy


def _box(label, x, y, sx, sy, height, material="concrete", yaw=0):
    return dict(type="box", label=label, center=[x, y, height / 2],
                size=[sx, sy, height], yaw_deg=yaw, material=material)


def _cylinder(label, x, y, radius, height, material="wood"):
    return dict(type="cylinder", label=label, base=[x, y, 0], radius=radius,
                height=height, material=material)


def fixed_environment(config):
    """Preserve one planning map and its three held-out obstacles across cases."""
    active = copy.deepcopy(config)
    active.setdefault("sensor_overrides", {}).update(
        n_channels=8, measurements_per_cycle=360, n_beam_samples=1,
        air_extinction_per_m=2e-4, rain_mm_per_h=0.0)
    active["mount_height_m"] = 0.6
    active["robot"].update(start=[0.0, 0.0], goal=[18.0, 0.0], v_max=1.2,
                           max_steps=180)
    active["planner"]["extent"] = [-2.0, 21.0, -6.0, 7.0]
    active["controller"]["safety_margin_m"] = 0.25
    active["lidar_enabled"] = True
    active["world"] = dict(known_objects=[
        _box("known_wall", 6, -.8, 2, 3.8, 1.2),
        _box("crate_south", 10, -2.3, 1.6, 1.4, 1.4, "wood"),
        _box("stack_east", 15, -.8, 1.1, 2.2, 1.6),
        _box("pallet_north", 14.5, 4.3, 2.2, 1.0, .6, "wood", -15),
        _cylinder("pillar_west", 3, 3.5, .3, 2.6),
        _cylinder("pillar_midnorth", 8.8, 4.1, .4, 2.2),
        _cylinder("pillar_eastnorth", 17.2, 3.4, .3, 2.5),
        _box("rack_southwest", 4, -4, 2.8, 1, 2.0),
        dict(type="sphere", label="metal_sphere", center=[10, 5, .6],
             radius=.6, material="aluminium"),
    ], unexpected_objects=[
        _cylinder("unexpected_barrel", 11, 1.1, .8, 1.2, "concrete"),
        _box("unexpected_bin", 13.2, 2.9, .7, .8, .75, "car_paint_black"),
        _cylinder("unexpected_low_barrel", 8.7, -.7, .4, .9),
    ])
    active["scenarios"] = [{"name": "cluttered"}]
    return active
