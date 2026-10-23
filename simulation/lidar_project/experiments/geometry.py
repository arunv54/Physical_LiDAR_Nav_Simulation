"""Controlled geometric experiments with independent analytical references.

All validation cases use one pencil ray per firing and disable random noise.
Finite-beam scans are saved separately as illustrations and are not scored
against a pencil-ray reference.  No plotting library is imported here.
"""
from __future__ import annotations

import csv
from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np

from .. import api


_EPS = 1e-6
_MATERIAL = {"name": "reference_diffuse", "rho_d": 0.7, "rho_s": 0.0, "spec_exp": 50.0}
_ASSUMPTIONS = [
    "Analytical comparisons use opaque planes, spheres, or axis-aligned boxes; one central ray per firing; and no random noise.",
    "Independent references use closed-form intersections and an explicit Z-Y-X rotation matrix, rather than the model's intersection/rotation helpers.",
    "Nominal FOV membership describes point inclusion in angular/range limits, independently of occlusion, discrete sampling, and optical detection.",
    "Eligible beams have a nearest positive opaque intersection inside the configured range gate. They are distinct from returns passing the optical threshold.",
    "Expected deterministic detections additionally apply the diffuse optical-power law and the unchanged receiver threshold.",
    "Surface-point fractions use deterministic samples of the complete target surface, including its back side; these fractions are not visible-surface fractions.",
    "Full-360 yaw comparisons use level sensors and yaw increments that are integer multiples of the azimuth step; equality does not imply arbitrary finite grids are invariant to every yaw offset.",
    "The 3-subray examples illustrate finite beam width and are excluded from pencil-ray error/count agreement metrics.",
    "These experiments assess numerical consistency with analytical idealizations, not agreement with real LiDAR measurements.",
]


def _rotation(pose):
    """Explicit active Z-Y-X rotation, independently expanded from its matrix."""
    yaw, pitch, roll = np.radians([pose.yaw_deg, pose.pitch_deg, pose.roll_deg])
    cy, sy, cp, sp, cr, sr = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch), np.cos(roll), np.sin(roll)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ])


def _beam_grid(cfg, pose):
    """Reconstruct configured samples without calling the sensor sampler."""
    step = 360.0 / cfg.measurements_per_cycle
    span = cfg.hfov_end_deg - cfg.hfov_start_deg
    n_az = cfg.measurements_per_cycle if span >= 360.0 - 1e-9 else int(np.floor(span / step + 1e-9)) + 1
    az = cfg.hfov_start_deg + step * np.arange(n_az)
    el = (np.array([cfg.vfov_upper_deg]) if cfg.n_channels == 1 else
          np.linspace(cfg.vfov_lower_deg, cfg.vfov_upper_deg, cfg.n_channels))
    aa, ee = np.meshgrid(np.radians(az), np.radians(el), indexing="ij")
    local = np.column_stack((np.cos(ee).ravel() * np.cos(aa).ravel(),
                             np.cos(ee).ravel() * np.sin(aa).ravel(), np.sin(ee).ravel()))
    directions = local @ _rotation(pose).T
    origins = np.broadcast_to(np.asarray(pose.position, float), directions.shape).copy()
    return origins, directions, np.degrees(aa).ravel(), np.degrees(ee).ravel(), len(el), len(az)


def nominal_fov_mask(points, cfg, pose):
    """Inclusive continuous angular/range support, independently of LOS/detection.

    Positive pitch follows the actual active Z-Y-X convention: sensor +x tilts
    toward world -z.  Tolerances address floating-point representations of exact
    mathematical boundaries, not a beam-width expansion.
    """
    points = np.asarray(points, float).reshape(-1, 3)
    local = (points - np.asarray(pose.position)) @ _rotation(pose)
    ranges = np.linalg.norm(local, axis=1)
    az = np.arctan2(local[:, 1], local[:, 0])
    el = np.arctan2(local[:, 2], np.hypot(local[:, 0], local[:, 1]))
    start = np.radians(cfg.hfov_start_deg)
    span = np.radians(cfg.hfov_end_deg - cfg.hfov_start_deg)
    delta = np.mod(az - start, 2 * np.pi)
    angular_tol = 1e-12
    horizontal = (np.ones(len(points), bool) if span >= 2 * np.pi - 1e-9 else
                  ((delta <= span + angular_tol) | (2 * np.pi - delta <= angular_tol)))
    return (horizontal & (el >= np.radians(cfg.vfov_lower_deg) - angular_tol)
            & (el <= np.radians(cfg.vfov_upper_deg) + angular_tol)
            & (ranges >= cfg.min_range_m - 1e-9) & (ranges <= cfg.max_range_m + 1e-9)
            & np.all(np.isfinite(local), axis=1))


def _plane_intersection(spec, origins, directions):
    normal = np.asarray(spec["normal"], float)
    normal /= np.linalg.norm(normal)
    numerator = (np.asarray(spec["point"]) - origins) @ normal
    denominator = directions @ normal
    ranges = np.full(len(origins), np.inf)
    usable = np.abs(denominator) > 1e-9
    np.divide(numerator, denominator, out=ranges, where=usable)
    ranges[~usable | (ranges <= _EPS)] = np.inf
    normals = np.broadcast_to(normal, origins.shape).copy()
    return ranges, normals


def _sphere_intersection(spec, origins, directions):
    relative = origins - np.asarray(spec["center"])
    projected = np.sum(relative * directions, axis=1)
    discriminant = projected ** 2 - (np.sum(relative ** 2, axis=1) - spec["radius"] ** 2)
    root = np.sqrt(np.maximum(discriminant, 0))
    near, far = -projected - root, -projected + root
    ranges = np.where(near > _EPS, near, far)
    ranges[(discriminant < 0) | (ranges <= _EPS)] = np.inf
    points = origins + directions * np.where(np.isfinite(ranges), ranges, 0)[:, None]
    normals = (points - np.asarray(spec["center"])) / spec["radius"]
    return ranges, normals


def _box_intersection(spec, origins, directions):
    if float(spec.get("yaw_deg", 0)) != 0:
        raise ValueError("The independent box reference is axis aligned")
    lower = np.asarray(spec["center"]) - np.asarray(spec["size"]) / 2
    upper = np.asarray(spec["center"]) + np.asarray(spec["size"]) / 2
    near = np.full_like(origins, -np.inf)
    far = np.full_like(origins, np.inf)
    for axis in range(3):
        parallel = np.abs(directions[:, axis]) < 1e-12
        outside = (origins[:, axis] < lower[axis]) | (origins[:, axis] > upper[axis])
        near[parallel & outside, axis] = np.inf
        far[parallel & outside, axis] = -np.inf
        moving = ~parallel
        a = (lower[axis] - origins[moving, axis]) / directions[moving, axis]
        b = (upper[axis] - origins[moving, axis]) / directions[moving, axis]
        near[moving, axis], far[moving, axis] = np.minimum(a, b), np.maximum(a, b)
    enter, exit = near.max(axis=1), far.min(axis=1)
    inside = enter <= _EPS
    ranges = np.where(inside, exit, enter)
    ranges[(exit < enter) | (exit <= _EPS)] = np.inf
    face_axis = np.where(inside, far.argmin(axis=1), near.argmax(axis=1))
    normals = np.zeros_like(origins)
    row = np.arange(len(origins))
    normals[row, face_axis] = -np.sign(directions[row, face_axis])
    return ranges, normals


def _reference(scene_spec, origins, directions):
    count = len(origins)
    nearest = np.full(count, np.inf)
    object_id = np.full(count, -1, int)
    normals = np.zeros_like(origins)
    individual = []
    solvers = {"plane": _plane_intersection, "sphere": _sphere_intersection, "box": _box_intersection}
    for i, spec in enumerate(scene_spec["objects"]):
        ranges, normal = solvers[spec["type"]](spec, origins, directions)
        individual.append(ranges)
        closer = ranges < nearest
        nearest[closer], object_id[closer], normals[closer] = ranges[closer], i, normal[closer]
    return nearest, object_id, normals, individual


def _expected_detection(cfg, scene_spec, ranges, object_id, normals, directions, eligible):
    power = np.zeros(len(ranges))
    for i, spec in enumerate(scene_spec["objects"]):
        use = eligible & (object_id == i)
        material = spec["material"]
        if material["rho_s"] != 0:
            raise ValueError("Independent detection comparison assumes a purely diffuse surface")
        incidence = np.abs(np.sum(normals[use] * directions[use], axis=1))
        aperture = np.pi * (cfg.aperture_diam_m / 2) ** 2
        extinction = cfg.air_extinction_per_m + (cfg.rain_a * cfg.rain_mm_per_h ** cfg.rain_b
                                               if cfg.rain_mm_per_h > 0 else 0.0)
        power[use] = (cfg.peak_power_w * aperture * cfg.optical_efficiency * material["rho_d"] / np.pi
                      * incidence / ranges[use] ** 2 * np.exp(-2 * extinction * ranges[use]))
    return eligible & (power / cfg.noise_rms_w >= cfg.snr_threshold), power


def _box_corners(spec):
    if spec is None:
        return np.empty((0, 3))
    signs = np.array([[x, y, z] for z in (-1, 1) for y in (-1, 1) for x in (-1, 1)])
    return np.asarray(spec["center"]) + signs * np.asarray(spec["size"]) / 2


def _surface_samples(spec):
    if spec is None:
        return np.empty((0, 3))
    if spec["type"] == "sphere":
        i = np.arange(4096)
        z = 1 - 2 * (i + 0.5) / len(i)
        phi = i * np.pi * (3 - np.sqrt(5))
        radius = np.sqrt(1 - z * z)
        return np.asarray(spec["center"]) + spec["radius"] * np.column_stack((radius * np.cos(phi), radius * np.sin(phi), z))
    if spec["type"] == "box":
        q = np.linspace(-1, 1, 21)
        a, b = np.meshgrid(q, q)
        faces = []
        for axis in range(3):
            free = [j for j in range(3) if j != axis]
            for sign in (-1, 1):
                face = np.zeros((a.size, 3))
                face[:, axis] = sign
                face[:, free[0]], face[:, free[1]] = a.ravel(), b.ravel()
                faces.append(face)
        return np.asarray(spec["center"]) + np.vstack(faces) * np.asarray(spec["size"]) / 2
    return np.empty((0, 3))


def _write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _save_frusta(directory, entries):
    vertices, faces, origins, names = [], [], [], []
    vertex_offsets, face_offsets = [0], [0]
    minimum, configured_maximum = [], []
    for name, cfg, pose in entries:
        # The displayed sector is explicitly clipped to 15 m, not a claim about
        # maximum detection range. Angular near/far boundaries are curved.
        aa = np.linspace(np.radians(cfg.hfov_start_deg), np.radians(cfg.hfov_end_deg), 33)
        ee = np.linspace(np.radians(cfg.vfov_lower_deg), np.radians(cfg.vfov_upper_deg), 13)
        az, el = np.meshgrid(aa, ee)
        rays = np.stack((np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)), -1)
        rays = rays @ _rotation(pose).T
        origin = np.asarray(pose.position)
        shape = np.stack((origin + cfg.min_range_m * rays, origin + 15.0 * rays))
        v = shape.reshape(-1, 3)
        nr, nc = len(ee), len(aa)
        local_faces = []
        def quad(a, b, c, d):
            local_faces.extend([[a, b, c], [a, c, d]])
        for shell in range(2):
            offset = shell * nr * nc
            for r in range(nr - 1):
                for c in range(nc - 1):
                    a = offset + r * nc + c
                    quad(a, a + 1, a + nc + 1, a + nc)
        offset = nr * nc
        for r in range(nr - 1):
            for c in (0, nc - 1):
                a = r * nc + c
                quad(a, a + nc, offset + a + nc, offset + a)
        for c in range(nc - 1):
            for r in (0, nr - 1):
                a = r * nc + c
                quad(a, a + 1, offset + a + 1, offset + a)
        vertices.append(v)
        faces.append(np.asarray(local_faces, int))
        origins.append(origin)
        names.append(name)
        vertex_offsets.append(vertex_offsets[-1] + len(v))
        face_offsets.append(face_offsets[-1] + len(local_faces))
        minimum.append(cfg.min_range_m)
        configured_maximum.append(cfg.max_range_m)
    np.savez_compressed(directory / "frustum_geometry.npz", vertices=np.vstack(vertices), faces=np.vstack(faces),
                        vertex_offsets=vertex_offsets, face_offsets=face_offsets, case_names=np.array(names),
                        sensor_origins=origins, display_range_m=15.0, minimum_range_m=minimum,
                        configured_max_range_m=configured_maximum)


def run_geometry(directory) -> dict:
    """Run controlled comparisons and save numerical evidence, without plots."""
    directory = Path(directory)
    (directory / "scans").mkdir(parents=True, exist_ok=True)
    (directory / "references").mkdir(parents=True, exist_ok=True)
    manifest = {"schema_version": 1, "study": "geometry", "assumptions": _ASSUMPTIONS, "cases": {}}
    rows, pose_rows, fov_rows, los_rows, frusta = [], [], [], [], []
    base = api.LidarConfig(n_channels=16, vfov_lower_deg=-12, vfov_upper_deg=12,
                           measurements_per_cycle=720, n_beam_samples=1, noise_enabled=False)
    target_sphere = {"type": "sphere", "label": "target", "center": [10.0, 0.0, 1.8],
                     "radius": 1.0, "material": dict(_MATERIAL)}

    def case(name, group, cfg, pose, objects, variable="", value=0.0, kind="analytic", target_index=None,
             occluder_index=None):
        scene_spec = {"materials": {}, "objects": objects}
        scene = api.build_scene(scene_spec)
        cloud = api.LidarSensor(cfg, pose, seed=0).scan(scene)
        cloud.save_npz(directory / "scans" / f"{name}.npz")
        origins, directions, az, el, n_ch, n_az = _beam_grid(cfg, pose)
        ranges, indices, normals, individual = _reference(scene_spec, origins, directions)
        eligible = np.isfinite(ranges) & (ranges >= cfg.min_range_m) & (ranges <= cfg.max_range_m)
        model_ranges, model_indices, _ = scene.cast(origins, directions)
        model_eligible = (model_indices >= 0) & (model_ranges >= cfg.min_range_m) & (model_ranges <= cfg.max_range_m)
        expected, power = _expected_detection(cfg, scene_spec, ranges, indices, normals, directions, eligible)
        actual_ranges = cloud.range_image.T.ravel()
        actual = np.isfinite(actual_ranges)
        compare = actual & expected
        residual = actual_ranges[compare] - ranges[compare]
        target = objects[target_index] if target_index is not None else None
        target_points = _surface_samples(target)
        surface_mask = nominal_fov_mask(target_points, cfg, pose)
        center_inside = (int(nominal_fov_mask([target["center"]], cfg, pose)[0]) if target else None)
        target_ranges = individual[target_index] if target_index is not None else np.full(len(ranges), np.inf)
        potential_target = (np.isfinite(target_ranges) & (target_ranges >= cfg.min_range_m)
                            & (target_ranges <= cfg.max_range_m))
        blocked = potential_target & (indices != target_index) if target is not None else np.zeros(len(ranges), bool)
        row = dict(case=name, group=group, kind=kind, variable=variable, value=float(value),
                   height_m=float(pose.position[2]), x_m=float(pose.position[0]), y_m=float(pose.position[1]),
                   yaw_deg=float(pose.yaw_deg), pitch_deg=float(pose.pitch_deg), roll_deg=float(pose.roll_deg),
                   hfov_start_deg=cfg.hfov_start_deg, hfov_end_deg=cfg.hfov_end_deg,
                   vfov_lower_deg=cfg.vfov_lower_deg, vfov_upper_deg=cfg.vfov_upper_deg,
                   n_beams=cloud.n_beams, eligible_reference=int(eligible.sum()), eligible_model=int(model_eligible.sum()),
                   eligible_mismatch=int(np.count_nonzero(eligible != model_eligible)) if kind == "analytic" else None,
                   expected_returns=int(expected.sum()) if kind == "analytic" else None, retained_returns=cloud.n,
                   detection_mismatch=int(np.count_nonzero(expected != actual)) if kind == "analytic" else None,
                   range_max_abs_error_m=float(np.max(np.abs(residual))) if kind == "analytic" and residual.size else None,
                   range_rmse_m=float(np.sqrt(np.mean(residual ** 2))) if kind == "analytic" and residual.size else None,
                   reference_range_m=float(ranges[0]) if len(ranges) == 1 and np.isfinite(ranges[0]) else None,
                   measured_range_m=float(actual_ranges[0]) if len(actual_ranges) == 1 and actual[0] else None,
                   target_center_in_fov=center_inside,
                   target_surface_fraction_in_fov=float(surface_mask.mean()) if target_points.size else None,
                   target_eligible_reference=int(np.count_nonzero(eligible & (indices == target_index))) if target else None,
                   target_eligible_model=int(np.count_nonzero(model_eligible & (model_indices == target_index))) if target else None,
                   target_returns=int(np.count_nonzero(cloud.label_id == target_index)) if target else None,
                   blocked_target_beams=int(blocked.sum()))
        rows.append(row)
        np.savez_compressed(directory / "references" / f"{name}.npz", sensor_origin=np.asarray(pose.position),
                            origins=origins, directions_world=directions, azimuth_deg=az, elevation_deg=el,
                            reference_range_m=ranges, reference_object_id=indices, eligible_reference=eligible,
                            model_nearest_range_m=model_ranges, model_object_id=model_indices, eligible_model=model_eligible,
                            expected_detection=expected, expected_power_w=power, actual_detection=actual,
                            potential_target_range_m=target_ranges, blocked_target=blocked,
                            target_surface_points=target_points, target_surface_fov=surface_mask,
                            target_box_corners=_box_corners(target) if target and target["type"] == "box" else np.empty((0, 3)),
                            occluder_box_corners=_box_corners(objects[occluder_index]) if occluder_index is not None else np.empty((0, 3)))
        manifest["cases"][name] = dict(kind=kind, group=group, variable=variable, value=float(value),
                                       config=asdict(cfg), pose=asdict(pose), scene=scene_spec,
                                       artifacts={"scan": f"scans/{name}.npz", "reference": f"references/{name}.npz"})
        return row

    def pose_case(name, group, cfg, pose, objects, variable, value):
        row = case(name, group, cfg, pose, objects, variable, value)
        pose_rows.append(dict(case=name, experiment=group, height_m=row["height_m"], pitch_deg=row["pitch_deg"],
                              yaw_deg=row["yaw_deg"], x_m=row["x_m"], y_m=row["y_m"],
                              predicted_range_m=row["reference_range_m"], measured_range_m=row["measured_range_m"],
                              max_abs_error_m=row["range_max_abs_error_m"]))

    pencil = replace(base, n_channels=1, vfov_lower_deg=-15, vfov_upper_deg=-15,
                     hfov_start_deg=0, hfov_end_deg=0.1)
    ground = {"type": "plane", "label": "ground", "point": [0, 0, 0], "normal": [0, 0, 1], "material": dict(_MATERIAL)}
    for i, height in enumerate((0.6, 1.2, 1.8, 2.4, 3.0)):
        pose_case(f"ground_height_{i}", "ground_height", pencil, api.SensorPose(position=(0, 0, height)),
                  [ground], "height_m", height)
    for i, pitch in enumerate((-10, -5, 0, 5, 10)):
        pose_case(f"ground_pitch_{i}", "ground_pitch", pencil, api.SensorPose(position=(0, 0, 1.8), pitch_deg=pitch),
                  [ground], "pitch_deg", pitch)
    wall = {"type": "plane", "label": "reference_wall", "point": [12, 0, 0], "normal": [-1, 0, 0], "material": dict(_MATERIAL)}
    wall_cfg = replace(pencil, vfov_lower_deg=-4, vfov_upper_deg=-4, hfov_start_deg=5, hfov_end_deg=5.1)
    for i, x in enumerate((-2, 0, 2)):
        pose_case(f"plane_position_{i}", "plane_position", wall_cfg, api.SensorPose(position=(x, 1, 1.8)),
                  [wall], "x_m", x)
    for i, yaw in enumerate((-15, 0, 15)):
        pose_case(f"plane_yaw_{i}", "plane_yaw", wall_cfg, api.SensorPose(position=(0, 1, 1.8), yaw_deg=yaw),
                  [wall], "yaw_deg", yaw)
    for i, pitch in enumerate((-10, 0, 10)):
        pose_case(f"plane_pitch_{i}", "plane_pitch", wall_cfg, api.SensorPose(position=(0, 1, 1.8), pitch_deg=pitch),
                  [wall], "pitch_deg", pitch)

    for mode in ("partial", "full360"):
        fov_cfg = replace(base, hfov_start_deg=-40 if mode == "partial" else -180,
                          hfov_end_deg=40 if mode == "partial" else 180)
        for axis, angles in (("yaw", (0, 20, 30, 35, 40, 45, 50, 60, 90, 180)),
                             ("pitch", (-20, -15, -12, -10, 0, 10, 12, 15, 20))):
            for i, angle in enumerate(angles):
                pose = api.SensorPose(position=(0, 0, 1.8), **{axis + "_deg": angle})
                name = f"fov_{mode}_{axis}_{i}"
                row = case(name, "fov_" + axis, fov_cfg, pose, [target_sphere], axis + "_deg", angle, target_index=0)
                fov_rows.append(dict(case=name, horizontal_mode=mode, motion_axis=axis, angle_deg=angle,
                                     nominal_center_inside=row["target_center_in_fov"],
                                     nominal_surface_fraction=row["target_surface_fraction_in_fov"],
                                     reference_eligible=row["eligible_reference"], model_eligible=row["eligible_model"],
                                     predicted_count=row["expected_returns"], measured_count=row["retained_returns"]))
                if mode == "partial" and ((axis == "yaw" and angle in (0, 60)) or (axis == "pitch" and angle == 10)):
                    frusta.append((name, fov_cfg, pose))
    position_cfg = replace(base, hfov_start_deg=-40, hfov_end_deg=40)
    for i, (x, y) in enumerate(((-2, 0), (0, 0), (2, 0), (0, -2), (0, 2))):
        case(f"fixed_scene_position_{i}", "fixed_scene_position", position_cfg,
             api.SensorPose(position=(x, y, 1.8)), [target_sphere], "position_index", i, target_index=0)

    target_box = {"type": "box", "label": "target", "center": [10, 0, 1.8], "size": [0.2, 2, 2],
                  "yaw_deg": 0, "material": dict(_MATERIAL)}
    los_cfg = replace(base, hfov_start_deg=-15, hfov_end_deg=15, vfov_lower_deg=-10, vfov_upper_deg=10)
    los_pose = api.SensorPose(position=(0, 0, 1.8))
    blockers = [("los_clear", None),
                ("los_opaque", {"type": "box", "label": "occluder", "center": [5, 0, 1.8],
                                "size": [0.2, 3, 3], "yaw_deg": 0, "material": dict(_MATERIAL)}),
                ("los_below_min", {"type": "box", "label": "near_occluder", "center": [0.15, 0, 1.8],
                                   "size": [0.1, 1, 1], "yaw_deg": 0, "material": dict(_MATERIAL)})]
    for name, blocker in blockers:
        objects = [target_box] + ([blocker] if blocker else [])
        row = case(name, "los", los_cfg, los_pose, objects, "occluder", len(objects) - 1,
                   target_index=0, occluder_index=1 if blocker else None)
        los_rows.append(dict(case=name, occluder="none" if blocker is None else blocker["label"],
                             target_returns=row["target_returns"], reference_target_eligible=row["target_eligible_reference"],
                             model_target_eligible=row["target_eligible_model"], blocked_target_beams=row["blocked_target_beams"],
                             nominal_center_inside=row["target_center_in_fov"]))

    for name, cfg, pose, objects, target_index, blocker_index in (
        ("example_partial_yaw0", replace(position_cfg, n_beam_samples=3), api.SensorPose(position=(0, 0, 1.8)), [target_sphere], 0, None),
        ("example_partial_yaw60", replace(position_cfg, n_beam_samples=3), api.SensorPose(position=(0, 0, 1.8), yaw_deg=60), [target_sphere], 0, None),
        ("example_los_clear", replace(los_cfg, n_beam_samples=3), los_pose, [target_box], 0, None),
        ("example_los_opaque", replace(los_cfg, n_beam_samples=3), los_pose, [target_box, blockers[1][1]], 0, 1),
    ):
        case(name, "finite_beam_example", cfg, pose, objects, kind="illustration",
             target_index=target_index, occluder_index=blocker_index)

    _write_csv(directory / "cases.csv", rows, list(rows[0]))
    _write_csv(directory / "geometry_pose.csv", pose_rows, list(pose_rows[0]))
    _write_csv(directory / "fov.csv", fov_rows, list(fov_rows[0]))
    _write_csv(directory / "los.csv", los_rows, list(los_rows[0]))
    _save_frusta(directory, frusta)
    manifest["frustum"] = {"artifact": "frustum_geometry.npz", "display_range_m": 15.0,
                            "note": "Faces index each case's local vertex slice; near/far shell sectors have curved angular boundaries."}
    _write_json(directory / "manifest.json", manifest)
    analytic = [r for r in rows if r["kind"] == "analytic"]
    yaw_counts = {mode: {str(r["angle_deg"]): r["measured_count"] for r in fov_rows
                         if r["horizontal_mode"] == mode and r["motion_axis"] == "yaw"}
                  for mode in ("partial", "full360")}
    height_rows = [r for r in pose_rows if r["experiment"] == "ground_height"]
    height_slope = float(np.polyfit([r["height_m"] for r in height_rows], [r["measured_range_m"] for r in height_rows], 1)[0])
    error_values = [r["range_max_abs_error_m"] for r in analytic if r["range_max_abs_error_m"] is not None]
    summary = {
        "schema_version": 1, "study": "geometry", "case_count": len(rows), "analytic_cases": len(analytic),
        "illustration_cases": len(rows) - len(analytic),
        "eligibility_mismatch_beams": sum(r["eligible_mismatch"] for r in analytic),
        "deterministic_detection_mismatch_beams": sum(r["detection_mismatch"] for r in analytic),
        "max_abs_range_error_m": max(error_values, default=0.0),
        "ground_height_measured_slope": height_slope,
        "ground_height_analytic_slope": float(1 / np.sin(np.radians(15))),
        "ground_pitch_ranges_m": {str(r["pitch_deg"]): r["measured_range_m"] for r in pose_rows if r["experiment"] == "ground_pitch"},
        "yaw_return_counts": yaw_counts,
        "full360_yaw_count_span": max(yaw_counts["full360"].values()) - min(yaw_counts["full360"].values()),
        "los_target_returns": {r["case"]: r["target_returns"] for r in los_rows},
        "los_blocked_target_beams": {r["case"]: r["blocked_target_beams"] for r in los_rows},
        "below_min_occluder_blocks_target": next(r["target_returns"] for r in los_rows if r["case"] == "los_below_min") == 0,
        "assumptions": _ASSUMPTIONS,
    }
    _write_json(directory / "summary.json", summary)
    return summary
