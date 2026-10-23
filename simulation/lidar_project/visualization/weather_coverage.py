"""Prepare and draw coverage of the unchanged weather-comparison scene.

This deterministic geometry audit uses an existing baseline pose and the full
saved 360-degree, eight-channel beam grid. It never calls the noisy scanner,
planner, controller, or map builder, and it does not modify any scene object.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from .. import api
from ..experiments.navigation_dependencies import (
    TARGET_LABEL, _scene, emitted_rays, opportunity_audit,
)
from ..trajectory.engine import mounted_sensor_pose
from ..trajectory.storage import load_result


def _box_geometry(obj, role):
    local = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                      [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], float)
    vertices = local * obj.h @ obj.R.T + obj.c
    faces = [[0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7]]
    return dict(label=obj.label if hasattr(obj, "label") else role, role=role,
                vertices=vertices.tolist(), faces=faces,
                label_at=(obj.c + [0,0,obj.h[2]+.12]).tolist())



def _subset(indices, limit):
    indices = np.asarray(indices, dtype=int)
    return indices[np.unique(np.linspace(0, len(indices)-1, min(limit, len(indices))).astype(int))] if len(indices) else indices



def _geometry(obj, role):
    """Tessellate only existing primitives; meshes never affect ray queries."""
    if hasattr(obj, "R") and hasattr(obj, "h"):
        return _box_geometry(obj, role)
    if hasattr(obj, "hgt"):
        n = 32
        angle = np.arange(n) * 2 * np.pi / n
        ring = np.column_stack((np.cos(angle), np.sin(angle))) * obj.r
        vertices = np.vstack((np.column_stack((ring, np.zeros(n))),
                              np.column_stack((ring, np.full(n, obj.hgt))))) + obj.base
        faces = [list(range(n - 1, -1, -1)), list(range(n, 2*n))]
        faces += [[i, (i+1) % n, (i+1) % n+n, i+n] for i in range(n)]
    elif hasattr(obj, "r") and hasattr(obj, "c"):
        longitude = np.linspace(0, 2*np.pi, 25)
        latitude = np.linspace(-np.pi/2, np.pi/2, 13)
        lo, la = np.meshgrid(longitude, latitude)
        vertices = np.column_stack((np.cos(la).ravel()*np.cos(lo).ravel(),
                                    np.cos(la).ravel()*np.sin(lo).ravel(),
                                    np.sin(la).ravel())) * obj.r + obj.c
        faces = []
        for j in range(len(latitude)-1):
            for i in range(len(longitude)-1):
                a = j*len(longitude)+i
                faces.append([a, a+1, a+len(longitude)+1, a+len(longitude)])
    else:
        raise ValueError(f"Unsupported display primitive: {type(obj).__name__}")
    return dict(label=obj.label, role=role, vertices=vertices.tolist(), faces=faces,
                label_at=(vertices.mean(axis=0)+[0, 0, .7]).tolist())


def prepare_weather_coverage(dependency_dir, output_dir):
    """Audit a naturally partly occluded target without changing the scene."""
    directory, output = Path(dependency_dir).resolve(), Path(output_dir).resolve()
    run = directory / "runs" / "baseline" / "seed_1"
    result = load_result(run)
    cfg = api.build_sensor(result["sensor"])
    if (cfg.n_channels != 8 or cfg.measurements_per_cycle != 360
            or cfg.hfov_start_deg != -180 or cfg.hfov_end_deg != 180):
        raise ValueError("Expected the unchanged full-circle primary baseline")
    scene = _scene(result["config"])
    tid = scene.labels.index(TARGET_LABEL)
    target = scene.objects[tid]
    candidates = []
    for step in range(len(result["controls"]["t"])):
        pose = mounted_sensor_pose(result["config"], result["states"]["xy"][step],
                                   result["states"]["theta"][step])
        origins, directions = emitted_rays(cfg, pose)
        target_range, _ = target.intersect(origins, directions)
        nearest_range, labels, _ = scene.cast(origins, directions)
        nominal = np.isfinite(target_range)
        visible = nominal & (labels == tid)
        blocked = nominal & ~visible
        if visible.any() and blocked.any():
            balance = min(int(visible.sum()), int(blocked.sum()))
            distance = float(np.linalg.norm(np.asarray(pose.position)-target.c))
            candidates.append((balance, -distance, step, pose))
    if not candidates:
        raise ValueError("No saved pose naturally has visible and occluded target rays")
    _, _, step, pose = max(candidates, key=lambda row: row[:2])
    origins, directions = emitted_rays(cfg, pose)
    target_range, _ = target.intersect(origins, directions)
    nearest_range, nearest_ids, _ = scene.cast(origins, directions)
    nominal = np.isfinite(target_range)
    visible = nominal & (nearest_ids == tid)
    blocked = nominal & ~visible
    selected = np.r_[_subset(np.flatnonzero(visible), 6),
                     _subset(np.flatnonzero(blocked), 6)]
    rotation = (api.rot_z(np.radians(pose.yaw_deg))
                @ api.rot_y(np.radians(pose.pitch_deg))
                @ api.rot_x(np.radians(pose.roll_deg)))
    local = directions @ rotation
    rays = [dict(
        beam_index=int(i), target_point=(origins[i]+target_range[i]*directions[i]).tolist(),
        nearest_point=(origins[i]+nearest_range[i]*directions[i]).tolist(),
        visible=bool(visible[i]), nearest_label=scene.labels[int(nearest_ids[i])],
        azimuth_deg=float(np.degrees(np.arctan2(local[i, 1], local[i, 0]))),
        elevation_deg=float(np.degrees(np.arcsin(local[i, 2]))),
        target_range_m=float(target_range[i]), nearest_range_m=float(nearest_range[i]),
        range_eligible=bool(cfg.min_range_m <= nearest_range[i] <= cfg.max_range_m),
    ) for i in selected]
    blocker_ids = sorted(set(int(i) for i in nearest_ids[blocked]))
    objects = [_geometry(target, "target")]
    objects += [_geometry(scene.objects[i], "occluder") for i in blocker_ids]
    # Full-scene display meshes are retained separately. The paper view focuses
    # on the target and its actual blockers, while all objects enter the cast.
    all_meshes = [_geometry(obj, "target" if i == tid else "context")
                  for i, obj in enumerate(scene.objects) if i != 0]
    counts = opportunity_audit(cfg, pose, scene)
    origin = np.asarray(pose.position, float)
    payload = dict(
        schema_version=1,
        source=dict(run="runs/baseline/seed_1", seed=1, step=int(step),
                    time_s=float(result["states"]["t"][step]),
                    config="Unchanged primary weather-comparison baseline: 12 obstacles plus ground, full 360-degree coverage",
                    trajectory_sha256=hashlib.sha256((run/"trajectory.json").read_bytes()).hexdigest(),
                    pose_selection="Maximum balance of naturally visible and blocked target rays; distance breaks ties; no noisy detections used"),
        sensor=dict(position=origin.tolist(), rotation_world_from_sensor=rotation.tolist(),
                    yaw_deg=float(pose.yaw_deg), pitch_deg=float(pose.pitch_deg), roll_deg=float(pose.roll_deg)),
        hfov_deg=[-180., 180.], vfov_deg=[float(min(cfg.elevations_deg())), float(max(cfg.elevations_deg()))],
        min_range_m=float(cfg.min_range_m), max_range_m=float(cfg.max_range_m), display_range_m=4.,
        panels=[dict(id="fixed_scene", label="Unchanged obstacle scene", objects=objects,
                     rays=rays, audit=counts, displayed_visible_rays=int(visible[selected].sum()),
                     displayed_blocked_rays=int(blocked[selected].sum()))],
        scene_objects=all_meshes, scene_object_specs=result["world_objects"],
        scene_obstacle_count=len(scene.objects)-1,
        displayed_object_labels=[item["label"] for item in objects],
        blocking_labels={scene.labels[i]: int(np.sum(blocked & (nearest_ids == i))) for i in blocker_ids},
        view_limits=[[float(origin[0]-4.3), float(max(target.c[0]+target.h[0]+.6, origin[0]+4.3))],
                     [float(origin[1]-4.3), float(origin[1]+4.3)], [-.9, 2.1]],
        view_elevation_deg=25., view_azimuth_deg=-64.,
        interpretation="Deterministic nearest-surface audit at an actual baseline pose, not a measured point cloud. All 12 original obstacles and ground enter the ray query; only the target and its true blocker meshes are drawn for clarity. The full 360-degree angular envelope is truncated to a 4 m display radius, whereas target rays continue to their physical endpoints. No FOV window, sensor hardware, obstacle, target, or material is changed. Weather attenuation changes power, not geometric FOV/LOS at identical poses.",
    )
    output.mkdir(parents=True, exist_ok=True)
    path = output/"fov_los.json"
    path.write_text(json.dumps(payload, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return payload


def _sector_lines(position, rotation, horizontal, vertical, radius):
    """Display-only geometry of the stated angular FOV; no visibility queries."""
    def world(azimuth, elevation):
        a, e = np.broadcast_arrays(np.radians(azimuth), np.radians(elevation))
        local = np.stack((np.cos(e)*np.cos(a), np.cos(e)*np.sin(a), np.sin(e)), axis=-1)
        return position + radius * local @ rotation.T
    azimuth = np.linspace(*horizontal, 101)
    elevation = np.linspace(*vertical, 41)
    lines = [world(azimuth, vertical[0]), world(azimuth, vertical[1]),
             world(horizontal[0], elevation), world(horizontal[1], elevation)]
    for a in horizontal:
        for e in vertical:
            lines.append(np.stack((position, world(a, e))))
    return lines, world


def render_weather_coverage(data, output_dir):
    """Draw saved audit geometry without performing additional ray queries."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(6.5, 3.0))
    ax = fig.add_subplot(111, projection="3d")
    fig.subplots_adjust(left=0, right=1, bottom=.16, top=.88)
    origin = np.asarray(data["sensor"]["position"])
    rotation = np.asarray(data["sensor"]["rotation_world_from_sensor"])
    boundary, world = _sector_lines(origin, rotation, data["hfov_deg"], data["vfov_deg"], data["display_range_m"])
    for line in boundary[:2]:
        ax.plot(*line.T, color="#72A1BC", lw=.75, alpha=.8)
    for elevation in data["vfov_deg"]:
        edge = world(np.linspace(-180, 180, 73), elevation)
        polygons = [[origin, edge[i], edge[i+1]] for i in range(len(edge)-1)]
        ax.add_collection3d(Poly3DCollection(polygons, facecolor="#83B6D1", alpha=.07, edgecolor="none"))
    panel = data["panels"][0]
    for item in panel["objects"]:
        vertices = np.asarray(item["vertices"])
        ax.add_collection3d(Poly3DCollection([vertices[f] for f in item["faces"]],
            facecolor="#DFBA79" if item["role"] == "target" else "#97A2AA",
            edgecolor="#53636D", linewidth=.6, alpha=.44))
        center = vertices.mean(axis=0)
        if item["role"] == "target":
            ax.text(*(center+[-.25, -.75, 1.05]), "Study target", fontsize=8, ha="center")
        else:
            ax.text(*(center+[0., -2., .9]), "Existing wall", fontsize=8, ha="center",
                    bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=.7))
    for ray in panel["rays"]:
        target, nearest = np.asarray(ray["target_point"]), np.asarray(ray["nearest_point"])
        color = "#126FA6" if ray["visible"] else "#BE424B"
        ax.plot(*np.stack((origin, target if ray["visible"] else nearest)).T, color=color, lw=.9)
        if not ray["visible"]:
            ax.plot(*np.stack((nearest, target)).T, color=color, lw=.9, ls=(0, (3, 2)))
            ax.scatter(*nearest, color=color, marker="x", s=10, linewidth=.6, depthshade=False)
    ax.scatter(*origin, color="#172B36", s=22, depthshade=False)
    ax.text(*(origin+[-.4, .1, .3]), "LiDAR", fontsize=8, ha="center")
    limits = np.asarray(data["view_limits"])
    ax.set(xlim=limits[0], ylim=limits[1], zlim=limits[2])
    ax.set_box_aspect(np.diff(limits, axis=1).ravel(), zoom=1.18)
    ax.view_init(elev=data["view_elevation_deg"], azim=data["view_azimuth_deg"])
    ax.set_xlabel("World x (m)", fontsize=8, labelpad=0)
    ax.set_ylabel("World y (m)", fontsize=8, labelpad=-2)
    ax.set_zlabel("z (m)", fontsize=8, labelpad=-4)
    ax.set_xticks((0, 4, 8, 12)); ax.set_yticks((-2, 2, 6)); ax.set_zticks((0, 2))
    ax.tick_params(labelsize=7.5, pad=0)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis._axinfo["grid"]["color"] = "#DFE6EB"
        axis._axinfo["grid"]["linewidth"] = .4
    audit = panel["audit"]
    fig.text(.5, .96, f"Unchanged 360° scan at {data['source']['time_s']:.1f} s: "
             f"{audit['visible_target_echoes']} visible + {audit['blocked_target_echoes']} occluded target rays",
             fontsize=9, ha="center", va="top")
    fig.text(.5, .89, f"{audit['total_beams']:,} emitted rays audited · 12 original obstacles + ground · "
             "4 m envelope shown; target rays extend farther", fontsize=7.5, ha="center", va="top")
    fig.legend(handles=[Patch(facecolor="#D9EAF4", edgecolor="#72A1BC", label="Full-circle FOV"),
                        Line2D([], [], color="#126FA6", label="Visible target ray"),
                        Line2D([], [], color="#BE424B", label="Existing-wall hit"),
                        Line2D([], [], color="#BE424B", ls="--", label="Occluded continuation")],
               loc="lower center", bbox_to_anchor=(.5, .005), ncol=4, fontsize=7.2,
               frameon=False, columnspacing=1, handlelength=1.7)
    paths = {}
    for extension in ("png", "pdf"):
        path = output/f"weather_fov_los.{extension}"
        fig.savefig(path, dpi=300, facecolor="white")
        paths[extension] = str(path)
    plt.close(fig)
    return paths


def main():
    project = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=project/"data/navigation_dependencies")
    parser.add_argument("--output-dir", type=Path, default=project/"outputs/weather_report")
    args = parser.parse_args()
    payload = prepare_weather_coverage(args.data_dir, args.output_dir)
    paths = render_weather_coverage(payload, args.output_dir)
    print(json.dumps(dict(paths=paths, source=payload["source"], audit=payload["panels"][0]["audit"])))


if __name__ == "__main__":
    main()
