"""Discrete unicycle reference tracking driven only by detected obstacle returns."""
from __future__ import annotations

import copy
from dataclasses import asdict
import math

import numpy as np

from .. import api
from ..navigation.control import free_path_lengths, wrap
from .configuration import effective_config
from .planning import generate_reference, interpolate_reference, project_reference, swept_surface_gap


def mounted_sensor_pose(config, position, theta):
    """Compose a rigid mount with a planar robot pose (theta is in radians).

    XY offsets are in the robot body frame; height is absolute above the flat
    ground. Orientation is Rz(robot yaw + mount yaw) Ry(pitch) Rx(roll), matching
    SensorPose's convention. Configurations without sensor_mount retain the
    original co-located, level sensor pose exactly.
    """
    mount = config.get("sensor_mount", {})
    offset_x, offset_y = mount.get("offset_xy_m", (0.0, 0.0))
    yaw = float(np.degrees(theta))
    mount_yaw = mount.get("yaw_deg", 0.0)
    if mount_yaw:
        yaw += mount_yaw
    if offset_x or offset_y:
        c, s = math.cos(theta), math.sin(theta)
        sensor_xy = (position[0] + c * offset_x - s * offset_y,
                     position[1] + s * offset_x + c * offset_y)
    else:
        sensor_xy = position
    return api.SensorPose(position=(*sensor_xy, config["mount_height_m"]),
                          yaw_deg=yaw, roll_deg=mount.get("roll_deg", 0.0),
                          pitch_deg=mount.get("pitch_deg", 0.0))


def run_trajectory(config, scenario=None, seed=1, record_history=True, reference=None, on_scan=None):
    """Run one trajectory; ground truth is used for route creation/scoring only.

    Each update takes an ideal snapshot scan at the known true pose, chooses a
    control from obstacle return coordinates, then integrates one forward-Euler
    segment. States include both the initial and final pose. No acceleration,
    localization uncertainty, scan latency, or moving obstacles are modeled.

    Optional on_scan receives keyword arguments scan, pose, time, robot_xy,
    robot_theta immediately after each enabled sensor scan and before choosing
    controls. Pose and robot_xy are copies; scan must be treated as read-only.
    This observer consumes no additional sensor random numbers and its results
    are not retained. Disabled LiDAR produces no callback invocation.
    """
    if on_scan is not None and not callable(on_scan):
        raise TypeError("on_scan must be callable or None")
    effective, name = effective_config(config, scenario)
    robot, control = effective["robot"], effective["controller"]
    dt = float(robot["dt"])
    if dt <= 0 or robot["max_steps"] < 1 or robot["v_max"] <= 0 or robot["omega_max_deg_s"] <= 0:
        raise ValueError("Robot dt, max_steps, speed and turn limit must be positive")
    if control["d_slow_m"] <= control["d_stop_m"] or control["n_headings"] < 4:
        raise ValueError("Require d_slow_m>d_stop_m and at least four heading candidates")
    reference = copy.deepcopy(reference if reference is not None else generate_reference(effective))
    ground = {"type": "plane", "label": "ground", "point": [0, 0, 0], "normal": [0, 0, 1], "material": "asphalt"}
    objects = [ground] + copy.deepcopy(effective["world"]["known_objects"]) + copy.deepcopy(effective["world"].get("unexpected_objects", []))
    scene = api.build_scene({"objects": objects})
    sensor_cfg = api.build_sensor(effective["_effective_sensor"])
    sensor = api.LidarSensor(sensor_cfg, seed=int(seed))
    position = np.asarray(robot["start"], float)
    goal = np.asarray(robot["goal"], float)
    theta = float(np.radians(robot["start_heading_deg"]))
    headings = np.linspace(-np.pi, np.pi, int(control["n_headings"]), endpoint=False)
    inflation = robot["radius_m"] + control["safety_margin_m"]
    omega_max = np.radians(robot["omega_max_deg_s"])
    progress = 0.0
    state_rows, command_rows, observations, scans = [], [], [], []

    def state(time, v, omega):
        nonlocal progress
        progress, _, _ = project_reference(position, reference, progress)
        _, tracking, _ = project_reference(position, reference)
        nominal = np.array([np.interp(time, reference["t"], reference["xy"][:, axis]) for axis in range(2)])
        return dict(t=time, xy=position.copy(), theta=theta, v=v, omega=omega,
                    tracking_error=tracking, time_tracking_error=float(np.linalg.norm(position - nominal)),
                    progress_s=progress, surface_gap=swept_surface_gap(position, position, scene, robot["radius_m"], robot["height_m"]))

    state_rows.append(state(0.0, 0.0, 0.0))
    status = "goal" if np.linalg.norm(goal - position) <= robot["goal_tolerance_m"] else "timeout"
    if state_rows[0]["surface_gap"] <= 0:
        status = "collision"
    for k in range(int(robot["max_steps"])) if status == "timeout" else []:
        sensor.pose = mounted_sensor_pose(effective, position, theta)
        if effective.get("lidar_enabled", True):
            pc = sensor.scan(scene)
            if on_scan is not None:
                on_scan(scan=pc, pose=copy.deepcopy(sensor.pose), time=k * dt,
                        robot_xy=position.copy(), robot_theta=theta)
            mask = (pc.xyz_world[:, 2] >= control["z_ground_m"]) & (pc.xyz_world[:, 2] <= robot["height_m"])
            obstacle_points = pc.xyz_world[mask, :2]
            returns = pc.n
        else:
            obstacle_points = np.empty((0, 2))
            returns = 0
        target = interpolate_reference(reference, progress + control["route_lookahead_m"])
        psi_target = math.atan2(target[1] - position[1], target[0] - position[0])
        target_free = float(free_path_lengths(obstacle_points, position, np.array([psi_target]), inflation, control["lookahead_m"])[0])
        if target_free < np.linalg.norm(target - position) + control["d_stop_m"]:
            target = interpolate_reference(reference, progress + control.get("avoidance_lookahead_m", control["route_lookahead_m"]))
            psi_target = math.atan2(target[1] - position[1], target[0] - position[0])
        candidates = np.unique(np.r_[headings, psi_target, theta])
        free = free_path_lengths(obstacle_points, position, candidates, inflation, control["lookahead_m"])
        cost = (control["w_tracking"] * abs(wrap(candidates - psi_target))
                + control["w_turn"] * abs(wrap(candidates - theta))
                - control["w_free"] * free / control["lookahead_m"])
        feasible = free >= control["d_stop_m"] + 0.15
        best = int(np.argmin(np.where(feasible, cost, np.inf))) if feasible.any() else int(np.argmax(free))
        selected = float(candidates[best])
        heading_error = float(wrap(selected - theta))
        omega = float(np.clip(control["k_omega"] * heading_error, -omega_max, omega_max))
        forward_free = float(free_path_lengths(obstacle_points, position, np.array([theta]), inflation, control["lookahead_m"])[0])
        scale = float(np.clip((forward_free - control["d_stop_m"]) / (control["d_slow_m"] - control["d_stop_m"]), 0, 1))
        v = robot["v_max"] * max(math.cos(heading_error), 0.0) * scale
        v = float(min(v, max(forward_free - control["d_stop_m"], 0.0) / dt, np.linalg.norm(goal - position) / dt))
        next_position = position + dt * v * np.array([math.cos(theta), math.sin(theta)])
        gap = swept_surface_gap(position, next_position, scene, robot["radius_m"], robot["height_m"])
        perceived = float(np.linalg.norm(obstacle_points - position, axis=1).min() - robot["radius_m"]) if len(obstacle_points) else math.inf
        command_rows.append(dict(t=k * dt, v=v, omega=omega, target_xy=target.copy(), selected_heading=selected,
                                 n_returns=returns, n_obstacle_points=len(obstacle_points), perceived_clearance=perceived,
                                 swept_surface_gap=gap, route_progress_s=progress))
        if record_history:
            observations.append(obstacle_points.copy())
            # Retain the already computed measurement, including ground. This
            # adds no sensor calls and consumes no additional random numbers.
            if effective.get("lidar_enabled", True):
                scans.append(dict(xyz_world=pc.xyz_world.copy(), intensity=pc.intensity.copy()))
            else:
                scans.append(dict(xyz_world=np.empty((0, 3)), intensity=np.empty(0)))
        position = next_position
        theta = float(wrap(theta + omega * dt))
        state_rows.append(state((k + 1) * dt, v, omega))
        if gap <= 0:
            status = "collision"
            break
        if np.linalg.norm(goal - position) <= robot["goal_tolerance_m"]:
            status = "goal"
            break
    state_arrays = {key: np.asarray([row[key] for row in state_rows]) for key in state_rows[0]}
    command_keys = ("t", "v", "omega", "target_xy", "selected_heading", "n_returns", "n_obstacle_points", "perceived_clearance", "swept_surface_gap", "route_progress_s")
    commands = {key: np.asarray([row[key] for row in command_rows], dtype=float).reshape((-1, 2) if key == "target_xy" else (-1,)) for key in command_keys}
    return dict(name=name, status=status, seed=int(seed), reference=reference, states=state_arrays,
                controls=commands, observations=observations, history_recorded=bool(record_history),
                scans=scans, scans_available=bool(record_history),
                config=effective, sensor=asdict(sensor_cfg), world_objects=objects,
                assumptions=["Known exact robot pose; snapshot scans at each control state, with within-sweep motion disabled.",
                             "Incomplete known map generates the reference; withheld obstacles enter only LiDAR sensing and scoring.",
                             "Local heading/speed control uses obstacle return coordinates, never ground-truth obstacle geometry or labels.",
                             "Reference time is nominal cruise timing and does not guarantee turn-rate feasibility at corners.",
                             "States contain initial/final poses; state v/omega store the command applied during the preceding interval.",
                             "Recorded raw scans contain all detected world-coordinate returns and intensity at K control snapshots; the final state has no additional scan.",
                             "Swept gap is path-to-filled-footprint distance minus disc radius; negative values indicate overlap, not penetration depth."])
