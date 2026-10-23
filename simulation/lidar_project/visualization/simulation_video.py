"""Recorded 3D LiDAR movies, independent of numerical experiment execution.

The video clock uses actual simulation seconds. Scans and endpoint-map voxels
appear only at their saved control time; the terminal robot state has no scan.
Rendering downsamples displayed voxels deterministically, never metric counts.
"""
from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
CONTROLLED_CASES = ("dense_bright", "dense_dark", "sparse_bright", "sparse_dark")
PALETTE = ("#2875A2", "#8B68B0", "#38988D", "#AF673E")


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


@dataclass
class RecordedRun:
    case_id: str
    metadata: dict
    arrays: dict
    mapping: dict
    map_metadata: dict
    source_paths: tuple[Path, ...]
    diagnostics: dict | None = None
    recording_kind: str = "runs"

    @property
    def end_time(self):
        return float(self.arrays["states_t"][-1])

    def at(self, time_s):
        """Causal snapshot; never extrapolate motion, scans or measured cells."""
        d, m = self.arrays, self.mapping
        state = min(len(d["states_t"]) - 1,
                    int(np.searchsorted(d["states_t"], time_s + 1e-9, side="right") - 1))
        scan = min(len(d["controls_t"]) - 1,
                   int(np.searchsorted(d["controls_t"], time_s + 1e-9, side="right") - 1))
        if state < 0:
            raise ValueError("Video time cannot precede the recorded initial state")
        if scan < 0:
            xyz = np.empty((0, 3))
        else:
            offsets = d["scan_offsets"]
            xyz = d["scan_xyz_world"][offsets[scan]:offsets[scan + 1]]
        map_mask = (m["map_first_seen"] >= 0) & (m["map_first_seen"] <= scan)
        seen = ((m["ground_first_seen"] >= 0) & (m["ground_first_seen"] <= scan)
                | (m["obstacle_first_seen"] >= 0) & (m["obstacle_first_seen"] <= scan))
        target = np.empty((0, 3))
        if scan >= 0 and "scan_target_mask" in d:
            target = xyz[d["scan_target_mask"][offsets[scan]:offsets[scan + 1]]]
        diagnostic = ({name: value[scan] for name, value in self.diagnostics.items()}
                      if scan >= 0 and self.diagnostics else {})
        return dict(state=state, scan=scan, xyz=xyz, target_xyz=target, diagnostics=diagnostic, map_mask=map_mask,
                    observed_fraction=float(seen.mean()), observed_cells=int(seen.sum()),
                    map_voxels=int(map_mask.sum()), ended=bool(time_s >= self.end_time - 1e-9))


def load_recording(directory, case_id, seed=1, *, kind="runs"):
    if kind not in ("runs", "replays"):
        raise ValueError("Recording kind must be runs or replays")
    run = Path(directory) / kind / case_id / f"seed_{seed}"
    paths = tuple(run / name for name in ("trajectory.json", "trajectory.npz", "mapping.json", "mapping.npz"))
    metadata, map_metadata = _read(paths[0]), _read(paths[2])
    with np.load(paths[1], allow_pickle=False) as archive:
        arrays = {k: archive[k].copy() for k in archive.files}
    with np.load(paths[3], allow_pickle=False) as archive:
        mapping = {k: archive[k].copy() for k in archive.files}
    k = len(arrays["controls_t"])
    if (not metadata.get("scans_available") or len(arrays["states_t"]) != k + 1
            or len(arrays["scan_offsets"]) != k + 1
            or not np.allclose(arrays["controls_t"], arrays["states_t"][:-1])):
        raise ValueError(f"Video requires aligned raw 3D scan histories: {run}")
    if not np.array_equal(np.diff(arrays["scan_offsets"]), arrays["controls_n_returns"]):
        raise ValueError(f"Raw scan counts do not match recorded metrics: {run}")
    if np.any(mapping["map_first_seen"] < 0) or np.any(mapping["map_first_seen"] >= k):
        raise ValueError(f"Map contains an invalid scan index: {run}")
    target_path = run / "scan_targets.npz"
    if target_path.exists():
        with np.load(target_path, allow_pickle=False) as archive:
            mask = archive["target_mask"].copy()
            if not np.array_equal(archive["scan_offsets"], arrays["scan_offsets"]):
                raise ValueError(f"Target labels do not align with raw scans: {run}")
        arrays["scan_target_mask"] = mask
        paths += (target_path,)
    if "scan_target_mask" in arrays and arrays["scan_target_mask"].shape != (len(arrays["scan_xyz_world"]),):
        raise ValueError(f"Target labels do not align with raw return count: {run}")
    diagnostic_path = run / "diagnostics.csv"
    diagnostics = None
    if diagnostic_path.exists():
        with diagnostic_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        if len(rows) != k:
            raise ValueError(f"Diagnostics do not align with recorded scans: {run}")
        diagnostics = {key: np.array([float(row[key]) if row[key] else np.nan for row in rows])
                       for key in rows[0]}
        if not np.allclose(diagnostics["t"], arrays["controls_t"]):
            raise ValueError(f"Diagnostic times do not align with scans: {run}")
        paths += (diagnostic_path,)
    return RecordedRun(case_id, metadata, arrays, mapping, map_metadata, paths, diagnostics, kind)


def _same_scene(runs, *, controlled=False):
    """Reject uncontrolled geometry, pose or material changes in comparisons.

    The factorial comparison allows exactly two changes: azimuth sample count
    and the designated diffuse target's rho_d. All poses, other materials,
    geometry, mounting, receiver noise, weather and angular FOV stay fixed.
    """
    first = runs[0]
    def world_signature(run):
        world = copy.deepcopy(run.metadata["world_objects"])
        if controlled:
            targets = [obj for obj in world if obj.get("label") == "study_target"]
            if len(targets) != 1 or not isinstance(targets[0].get("material"), dict):
                raise ValueError("Controlled comparison requires one diffuse study_target")
            if targets[0]["material"].get("rho_s", 0) != 0:
                raise ValueError("Controlled study_target must have zero specular reflectance")
            targets[0]["material"].pop("rho_d")
        return json.dumps(world, sort_keys=True)
    world = world_signature(first)
    reference = first.arrays["reference_xy"]
    sensor = copy.deepcopy(first.metadata["sensor"])
    sensor.pop("measurements_per_cycle", None)
    for run in runs[1:]:
        if world_signature(run) != world:
            raise ValueError("Video comparison must use identical obstacle and target geometry/material")
        if not np.array_equal(run.arrays["reference_xy"], reference):
            raise ValueError("Video comparison must use an identical planning reference")
        if controlled:
            other = copy.deepcopy(run.metadata["sensor"])
            other.pop("measurements_per_cycle", None)
            if other != sensor or any(run.metadata["config"][key] != first.metadata["config"][key]
                                      for key in ("sensor_mount", "mount_height_m")):
                raise ValueError("Controlled comparison cannot change angles, FOV, noise, weather or other sensor parameters")
            for key in ("states_t", "states_xy", "states_theta", "controls_t"):
                if not np.array_equal(run.arrays[key], first.arrays[key]):
                    raise ValueError("Controlled replay must use exactly the same sensor poses and scan times")


def _faces(obj):
    """Display-only tessellation of stored geometry; no ray casting."""
    kind = obj["type"]
    if kind == "box":
        vertices = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                             [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], float) * np.asarray(obj["size"]) / 2
        angle = np.radians(obj.get("yaw_deg", 0))
        rotation = np.array([[np.cos(angle),-np.sin(angle),0], [np.sin(angle),np.cos(angle),0], [0,0,1]])
        vertices = vertices @ rotation.T + obj["center"]
        return [vertices[i] for i in ([0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7])]
    if kind == "cylinder":
        p = np.linspace(0, 2*np.pi, 17)
        ring = np.c_[obj["radius"]*np.cos(p),obj["radius"]*np.sin(p),np.zeros(len(p))] + obj["base"]
        top = ring + [0,0,obj["height"]]
        return [ring[:-1],top[:-1]] + [np.array([ring[i],ring[i+1],top[i+1],top[i]]) for i in range(16)]
    if kind == "sphere":
        t, p = np.linspace(0,np.pi,7), np.linspace(0,2*np.pi,13)
        v = np.array([[np.sin(a)*np.cos(b),np.sin(a)*np.sin(b),np.cos(a)] for a in t for b in p]) * obj["radius"] + obj["center"]
        return [v[[i*13+j,i*13+j+1,(i+1)*13+j+1,(i+1)*13+j]] for i in range(6) for j in range(12)]
    return []


def _rotation(roll, pitch, yaw):
    r,p,y = np.radians([roll,pitch,yaw])
    rx = np.array([[1,0,0],[0,np.cos(r),-np.sin(r)],[0,np.sin(r),np.cos(r)]])
    ry = np.array([[np.cos(p),0,np.sin(p)],[0,1,0],[-np.sin(p),0,np.cos(p)]])
    rz = np.array([[np.cos(y),-np.sin(y),0],[np.sin(y),np.cos(y),0],[0,0,1]])
    return rz @ ry @ rx


def _sensor_glyph(run, xy, theta):
    """Nominal angular FOV drawn to 2.3 m only; glyph is not measured range."""
    config, sensor = run.metadata["config"], run.metadata["sensor"]
    mount = config.get("sensor_mount", {})
    heading = np.degrees(theta) + mount.get("yaw_deg", 0)
    rotation = _rotation(mount.get("roll_deg", 0), mount.get("pitch_deg", 0), heading)
    offset = np.asarray(mount.get("offset_xy_m", [0,0]), float)
    robot_rotation = _rotation(0,0,np.degrees(theta))
    origin = np.r_[xy, config["mount_height_m"]] + np.r_[offset,0] @ robot_rotation.T
    az = np.linspace(sensor["hfov_start_deg"], sensor["hfov_end_deg"], 37)
    boundaries = []
    for elevation in (sensor["vfov_lower_deg"],sensor["vfov_upper_deg"]):
        a,e = np.radians(az), np.radians(elevation)
        local = np.c_[np.cos(e)*np.cos(a),np.cos(e)*np.sin(a),np.full(len(a),np.sin(e))]
        boundaries.append(origin + 2.3*(local @ rotation.T))
    segments = [boundaries[0],boundaries[1]]
    for i in (0,len(az)//4,len(az)//2,3*len(az)//4,len(az)-1):
        segments.extend([np.vstack([origin,boundaries[0][i]]),np.vstack([origin,boundaries[1][i]])])
    return segments, origin


def _source_manifest(runs, summary_path):
    files = [summary_path] + [path for run in runs for path in run.source_paths]
    result = []
    for path in files:
        try:
            label = path.resolve().relative_to(PROJECT).as_posix()
        except ValueError:
            label = path.name
        result.append({"path":label,"sha256":hashlib.sha256(path.read_bytes()).hexdigest()})
    return result


def _frame_axes(fig, run, index, count, color):
    from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

    if count == 4:
        left, width = .035 + (index % 2)*.49, .445
        top = .874 - (index // 2)*.41
        plot_bounds = [left-.005, top-.30, width+.005, .265]
        positions = dict(label=top, params=top-.024, status=top-.31,
                         instant=top-.335, totals=top-.365, study=top-.39)
    else:
        left, width = .035 + index*.93/count, .91/count
        plot_bounds = [left-.005,.285,width+.005,.525]
        positions = dict(label=.855,params=.823,status=.293,instant=.249,totals=.19,study=.126)
    ax = fig.add_axes(plot_bounds,projection="3d")
    ax.set_facecolor("#F7F9FC")
    ax.set(xlim=(-2,21),ylim=(-6,7),zlim=(-.2,3.2))
    ax.set_box_aspect((23,13,5.2),zoom=1.06)
    ax.view_init(elev=47,azim=-102)
    ax.set_xticks([0,5,10,15,20]); ax.set_yticks([-5,0,5]); ax.set_zticks([0,2])
    ax.set_xlabel("x (m)",labelpad=-1);ax.set_ylabel("y (m)",labelpad=-3);ax.set_zlabel("z (m)",labelpad=-3)
    ax.tick_params(labelsize=8,pad=-1)
    for axis in (ax.xaxis,ax.yaxis,ax.zaxis):
        axis.pane.fill=False
        axis._axinfo["grid"]["color"]=(.78,.82,.87,.55)
        axis._axinfo["grid"]["linewidth"]=.45
    for obj in run.metadata["world_objects"]:
        faces = _faces(obj)
        if faces:
            target = obj.get("label") == "study_target"
            rho = obj.get("material", {}).get("rho_d", .3) if isinstance(obj.get("material"), dict) else .3
            target_color = "#CDB175" if rho >= .1 else "#45414B"
            ax.add_collection3d(Poly3DCollection(faces,facecolor=target_color if target else "#AEB8C4",
                                               edgecolor="#887766" if target else "#8995A3",alpha=.12,
                                               linewidth=.5))
            if target:
                center = np.asarray(obj["center"])
                ax.text(center[0],center[1],center[2]+1.1,"Study target",fontsize=8,color="#AF4B4B")
    reference=run.arrays["reference_xy"]
    ax.plot(reference[:,0],reference[:,1],np.full(len(reference),.025),c="#9AA5B3",ls=(0,(5,3)),lw=1.1)
    goal=run.metadata["config"]["robot"]["goal"]
    ax.scatter([goal[0]],[goal[1]],[.08],marker="*",s=110,c="#294352",depthshade=False)
    ax.text(goal[0]+.1,goal[1]+.4,.08,"Goal",fontsize=9,color="#294352")
    cloud=ax.scatter([],[],[],s=.85,c=color,alpha=.48,depthshade=False,linewidths=0)
    scan=ax.scatter([],[],[],s=1.7,c="#E4892E",alpha=.88,depthshade=False,linewidths=0)
    target_scan=ax.scatter([],[],[],s=9,c="#D94765",alpha=.95,depthshade=False,linewidths=0)
    path,=ax.plot([],[],[],c=color,lw=2.1)
    body=ax.scatter([],[],[],s=80,c="#173C59",edgecolors="white",linewidths=.8,depthshade=False)
    heading,=ax.plot([],[],[],c="#173C59",lw=2.4)
    glyph=Line3DCollection([np.zeros((2,3))],colors="#319C73",linewidths=.6,alpha=.52)
    ax.add_collection3d(glyph)
    sensor_point=ax.scatter([],[],[],s=23,c="#319C73",edgecolors="white",linewidths=.5,depthshade=False)
    return dict(ax=ax,cloud=cloud,scan=scan,target_scan=target_scan,path=path,body=body,heading=heading,glyph=glyph,
                sensor=sensor_point,left=left,width=width,positions=positions)


def render_comparison(runs, labels, parameter_lines, output, *, title, subtitle, summary,
                      summary_path, mode, selection=None, fps=15, width=1600, height=900,
                      hold_seconds=3., ffmpeg=None, display_voxels=12000):
    """Encode a synchronized movie of stored snapshots with ffmpeg/H.264."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    controlled = mode in ("sensor", "factorial")
    _same_scene(runs, controlled=controlled)
    if controlled and any(run.recording_kind != "replays" or not run.diagnostics
                          or "scan_target_mask" not in run.arrays for run in runs):
        raise ValueError("Controlled videos require matched-pose replays, target labels and per-scan diagnostics")
    if width%2 or height%2 or width<800 or height<450 or fps<5 or hold_seconds<0:
        raise ValueError("Use even dimensions >=800x450, fps >=5 and nonnegative hold time")
    executable=ffmpeg or shutil.which("ffmpeg")
    if not executable:
        raise RuntimeError("ffmpeg is required (macOS: brew install ffmpeg)")
    output=Path(output).expanduser().resolve();output.parent.mkdir(parents=True,exist_ok=True)
    qa=PROJECT/"build"/"video_qa"/output.stem;qa.mkdir(parents=True,exist_ok=True)
    count=len(runs); end=max(run.end_time for run in runs)
    # The numerical simulation records at 5 Hz. Encode repeated frames at 15 fps;
    # no invented intermediate measurements or time-normalized routes are added.
    dt=round(min(float(np.min(np.diff(run.arrays["controls_t"]))) for run in runs),9)
    snapshot_rate=1/dt
    frame_count=int(np.ceil((end+hold_seconds)*snapshot_rate-1e-8))
    fig=plt.figure(figsize=(width/100,height/100),dpi=100,facecolor="#F7F9FC")
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"text.color":"#233547", "axes.labelcolor":"#576777"})
    fig.text(.04,.948,title,fontsize=25 if count < 4 else 23,fontweight="bold",color="#19394F")
    fig.text(.04,.911,subtitle,fontsize=11,color="#627485")
    clock=fig.text(.96,.906,"",ha="right",fontsize=16,fontweight="bold",color="#19394F")
    panels=[]
    # A common thinning fraction preserves between-case point-density ratios.
    # Per-panel equal point caps would visually hide the sampling intervention.
    maximum_voxels=max(len(run.mapping["map_xyz"]) for run in runs)
    display_fraction=min(1.,display_voxels/maximum_voxels) if maximum_voxels else 1.
    for i,(run,label,params) in enumerate(zip(runs,labels,parameter_lines)):
        panel=_frame_axes(fig,run,i,count,PALETTE[i]);left,w=panel["left"],panel["width"];pos=panel["positions"]
        fig.text(left,pos["label"],label,fontsize=17 if count<4 else 14,fontweight="bold",color=PALETTE[i])
        fig.text(left,pos["params"],params,fontsize=10.2 if count<4 else 9.5,color="#5D6C7D",linespacing=1.45)
        panel["status"]=fig.text(left,pos["status"],"",fontsize=11.7 if count<4 else 9.7,fontweight="bold")
        panel["instant"]=fig.text(left,pos["instant"],"",va="top",fontsize=11 if count<4 else 9.5,color="#40576A",linespacing=1.6 if count<4 else 1.15)
        panel["totals"]=fig.text(left,pos["totals"],"",va="top",fontsize=11 if count<4 else 9.5,color="#40576A",linespacing=1.65)
        a=next(c for c in summary["cases"] if c["id"]==run.case_id)["aggregate"]
        text=f"Closed-loop study: {a['goals']}/{a['runs']} goals  |  {a['collisions']} collisions"
        if count < 4:
            text+=f"\nMean mapped endpoint cells: {100*a['mean_observed_fraction']:.1f}%"
        fig.text(left,pos["study"],text,fontsize=10 if count<4 else 8.7,color="#718091",linespacing=1.7)
        m=run.mapping["map_xyz"]
        display_count=min(len(m),int(np.ceil(len(m)*display_fraction)))
        panel["display_index"]=np.linspace(0,len(m)-1,display_count,dtype=int)
        panels.append(panel)
    legend="COLORED  accumulated measured endpoints     ORANGE  latest scan     GREEN  fixed nominal FOV"
    if controlled:
        legend+="     PINK  latest detected study-target returns"
    fig.text(.04,.045 if count==4 else .065,legend,fontsize=9.2,color="#586E81")
    if mode=="environment":
        note="Weather changes optical attenuation only. No atmospheric backscatter / false weather returns. Same obstacles, target and sensor."
    else:
        note=("Same exact poses / angles / FOV / noise / clear air. Only azimuth samples and one target's diffuse reflectance change. "
              + ("Compare within rows or columns to isolate one factor." if mode=="factorial" else "Joint endpoints; use the four-case video for one-factor comparisons."))
    fig.text(.04,.025 if count==4 else .037,note,fontsize=8.7 if count==4 else 9.2,color="#586E81")
    fig.text(.96,.012,f"Recorded seed {runs[0].metadata['seed']}  |  exact simulator poses  |  display-only voxel thinning / fixed scene crop  |  native snapshots: {snapshot_rate:g} Hz",ha="right",fontsize=8,color="#8B98A7")
    command=[executable,"-hide_banner","-loglevel","error","-y","-f","rawvideo","-vcodec","rawvideo","-pix_fmt","rgb24",
             "-s",f"{width}x{height}","-framerate",f"{snapshot_rate:.8g}","-i","pipe:0","-an","-c:v","libx264",
             "-preset","fast","-crf","20","-pix_fmt","yuv420p","-r",str(fps),"-movflags","+faststart",str(output)]
    with (qa/"ffmpeg.log").open("wb") as log:
        process=subprocess.Popen(command,stdin=subprocess.PIPE,stderr=log)
        try:
            for frame in range(frame_count):
                time_s=min(frame/snapshot_rate,end)
                clock.set_text(f"t = {time_s:04.1f} s")
                for run,panel in zip(runs,panels):
                    s=run.at(time_s);d=run.arrays;k=s["state"];xy=d["states_xy"][k];theta=d["states_theta"][k]
                    index=panel["display_index"];index=index[s["map_mask"][index]]
                    points=run.mapping["map_xyz"][index]
                    panel["cloud"]._offsets3d=points.T if len(points) else ([],[],[])
                    p=s["xyz"]
                    crop=(p[:,0]>=-2)&(p[:,0]<=21)&(p[:,1]>=-6)&(p[:,1]<=7)&(p[:,2]>=-.2)&(p[:,2]<=3.2)
                    shown=p[crop];panel["scan"]._offsets3d=shown.T if len(shown) else ([],[],[])
                    target=s["target_xyz"];panel["target_scan"]._offsets3d=target.T if len(target) else ([],[],[])
                    path=d["states_xy"][:k+1];panel["path"].set_data_3d(path[:,0],path[:,1],np.full(len(path),.09))
                    panel["body"]._offsets3d=([xy[0]],[xy[1]],[.19])
                    arrow=np.array([np.r_[xy,.22],np.r_[xy+.85*np.array([np.cos(theta),np.sin(theta)]),.22]])
                    panel["heading"].set_data_3d(*arrow.T)
                    segments,origin=_sensor_glyph(run,xy,theta);panel["glyph"].set_segments(segments)
                    panel["sensor"]._offsets3d=([origin[0]],[origin[1]],[origin[2]])
                    status=run.metadata["status"]
                    if controlled:
                        text="MATCHED-POSE REPLAY COMPLETE" if s["ended"] else "MATCHED-POSE REPLAY  |  target detections highlighted"
                    elif s["ended"]:
                        text=f"{'GOAL REACHED' if status=='goal' else status.upper()} at {run.end_time:.1f} s  |  recording ended"
                    else:
                        text="NAVIGATING  |  measurements update every 0.2 s"
                    panel["status"].set_text(text)
                    panel["status"].set_color("#AE473E" if s["ended"] and status=="collision" else "#237B63" if s["ended"] else "#40576A")
                    scan_time=float(d["controls_t"][s["scan"]]) if s["scan"]>=0 else 0
                    if controlled:
                        q=s["diagnostics"];eligible=int(q.get("eligible_echoes",0));detected=int(q.get("detected_returns",0))
                        power=q.get("mean_eligible_power_w",np.nan);prob=q.get("expected_detection_probability",np.nan)
                        power_text=f"{power*1e9:.3g} nW" if np.isfinite(power) else "no eligible echo"
                        probability_text=f"{100*prob:.1f}%" if np.isfinite(prob) else "--"
                        visible=int(q.get("visible_target_echoes",0));geometric=int(q.get("target_opportunities",0))
                        panel["instant"].set_text(f"Target detected: {detected} / {eligible} eligible  |  LOS clear: {visible} / {geometric} target rays\n"
                                                  f"Power: {power_text}  |  mean P_D: {probability_text}  |  all returns: {len(p):,}")
                        map_text=f"Measured map: {s['map_voxels']:,} voxels  |  endpoint cells: {s['observed_fraction']*100:.1f}%"
                        panel["totals"].set_text(map_text)
                    else:
                        panel["instant"].set_text(f"{'Last scan' if s['ended'] else 'Latest scan'}: {len(p):,} points at {scan_time:.1f} s" + ("  (frozen)" if s["ended"] else ""))
                        panel["totals"].set_text(f"Accumulated map: {s['map_voxels']:,} measured voxels\nObserved endpoint cells: {s['observed_cells']:,} / {s['observed_fraction']*100:.1f}%")
                fig.canvas.draw()
                if frame in (0,int(2.6*snapshot_rate),int(end*snapshot_rate/2),int(round(end*snapshot_rate))):
                    fig.savefig(qa/f"frame_{frame:04d}.png",dpi=100)
                rgb=np.asarray(fig.canvas.buffer_rgba())[:,:,:3].copy()
                process.stdin.write(rgb.tobytes())
                if frame%max(1,frame_count//10)==0:
                    print(f"{output.name}: rendered {frame+1}/{frame_count} simulation snapshots",flush=True)
            process.stdin.close()
            if process.wait()!=0:
                raise RuntimeError((qa/"ffmpeg.log").read_text())
        except BaseException:
            process.kill();process.wait();raise
        finally:
            plt.close(fig)
    metadata={"format_version":2,"video":output.name,"mode":mode,"frame_size":[width,height],"encoded_fps":fps,
              "simulation_snapshot_hz":snapshot_rate,"rendered_snapshot_count":frame_count,
              "duration_s":frame_count/snapshot_rate,"common_time_axis_s":[0,end],"end_hold_s":hold_seconds,
              "chronology":"scan k at controls_t[k] = states_t[k]; terminal state has no scan; ended runs freeze without new measurements",
              "map_semantics":"first measured endpoint per 0.12 m voxel; observed XY cells are measured-endpoint cells, not free space or true map accuracy",
              "display":f"small endpoint markers; common retained map-voxel fraction {display_fraction:.6f}, at most {display_voxels} displayed voxels per panel, preserving case density ratios; latest scans are visually cropped to x[-2,21], y[-6,7], z[-0.2,3.2] m; all raw returns retained in metrics; static translucent meshes are truth context",
              "limitations":["exact simulator poses, static objects, ideal snapshot scans; no SLAM", "illustrative attenuation coefficients; no calibrated real-weather validation", "rain/fog attenuation only; no weather particles/backscatter"],
              "comparison_design":selection,"source_files":_source_manifest(runs,Path(summary_path)),"cases":[]}
    if controlled:
        metadata["limitations"] += ["Matched-pose replay isolates measurements; closed-loop outcomes are reported separately",
                                    "Bright/dark diffuse reflectances are illustrative virtual coefficients, not calibrated materials",
                                    "Sampling density is sensor configuration; sensor state is its changing position/orientation along the common route"]
    for run in runs:
        case=next(c for c in summary["cases"] if c["id"]==run.case_id)
        metadata["cases"].append(dict(id=run.case_id,seed=run.metadata["seed"],status=run.metadata["status"],
            end_time_s=run.end_time,last_scan_time_s=float(run.arrays["controls_t"][-1]),sensor=run.metadata["sensor"],
            sensor_mount=run.metadata["config"].get("sensor_mount"),mount_height_m=run.metadata["config"]["mount_height_m"],
            recording_kind=run.recording_kind,aggregate=case["aggregate"],seed_map=run.map_metadata,
            study_target=next((o for o in run.metadata["world_objects"] if o.get("label")=="study_target"),None)))
    generator={"environment":"environment","sensor":"sensor_state","factorial":"target_features"}[mode]
    metadata["reproduce"]=f"python tools/generate_{generator}_video.py"
    output.with_suffix(".json").write_text(json.dumps(metadata,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    print(f"Saved {output} ({output.stat().st_size/1e6:.2f} MB)",flush=True)
    return metadata


def _inputs(directory, requested, *, regenerate=False, seed=1, allow_generate=True,
            controlled=False):
    directory=Path(directory).expanduser().resolve()
    required=list(CONTROLLED_CASES) if controlled else list(requested)
    seeds=(1,2,3) if controlled else (seed,)
    kind="replays" if controlled else "runs"
    valid=(directory/"summary.json").exists()
    if valid:
        summary=_read(directory/"summary.json")
        valid=set(required).issubset({c["id"] for c in summary["cases"]})
        names=("trajectory.json","trajectory.npz","mapping.json","mapping.npz")
        if controlled:
            names+=("diagnostics.csv","scan_targets.npz")
        valid=valid and all((directory/kind/case/f"seed_{s}"/name).exists()
                           for case in required for s in seeds for name in names)
    if regenerate or not valid:
        if not allow_generate:
            raise FileNotFoundError("Saved video inputs are incomplete. Run the corresponding generator with --prepare-only first.")
        if controlled:
            from ..experiments.sampling_reflectance import run_study
            print(f"Generating controlled sampling / reflectance inputs in {directory}",flush=True)
            run_study(directory,seeds=seeds)
        else:
            from ..experiments.navigation_dependencies import run_study
            # Never overwrite frozen report data when raw archives are absent in a clone.
            isolated=PROJECT/"data"/"video_inputs"
            if not directory.is_relative_to(isolated):
                directory=isolated/"environments"
            print(f"Generating missing weather inputs in {directory}",flush=True)
            run_study(directory,seeds=seeds,case_ids=required)
        summary=_read(directory/"summary.json")
    return directory,summary


def video_main(mode, argv=None):
    controlled=mode in ("sensor","factorial")
    if mode not in ("environment","sensor","factorial"):
        raise ValueError(mode)
    parser=argparse.ArgumentParser(description="Render saved 3D simulation histories as a synchronized H.264 video")
    parser.add_argument("--data-dir",type=Path,default=PROJECT/"data"/("sampling_reflectance" if controlled else "navigation_dependencies"))
    parser.add_argument("--output",type=Path)
    parser.add_argument("--seed",type=int,default=1,choices=(1,2,3) if controlled else None)
    parser.add_argument("--fps",type=int,default=15,help="Encoded frame rate; numerical snapshots remain at their native 5 Hz")
    parser.add_argument("--width",type=int,default=1920 if mode=="factorial" else 1600)
    parser.add_argument("--height",type=int,default=1080 if mode=="factorial" else 900)
    parser.add_argument("--hold-seconds",type=float,default=3.)
    parser.add_argument("--ffmpeg",help="Optional ffmpeg executable path")
    parser.add_argument("--regenerate",action="store_true",help="Rerun the required numerical study before rendering")
    parser.add_argument("--use-saved",action="store_true",help="Read-only rendering: fail if raw inputs are absent; never run numerical simulations")
    parser.add_argument("--prepare-only",action="store_true",help="Prepare only this video's numerical study, without encoding a video")
    args=parser.parse_args(argv)
    if args.regenerate and args.use_saved:
        parser.error("--regenerate and --use-saved are mutually exclusive")
    requested=CONTROLLED_CASES if controlled else ("baseline","fog","rain")
    directory,summary=_inputs(args.data_dir,requested,regenerate=args.regenerate,seed=args.seed,
                              allow_generate=not args.use_saved,controlled=controlled)
    if args.prepare_only:
        print(f"Video inputs ready: {directory}",flush=True)
        return {"data_directory":str(directory)}
    if mode=="environment":
        ids=("baseline","fog","rain")
        runs=[load_recording(directory,c,args.seed) for c in ids]
        labels=["NORMAL","FOG","RAIN"]
        params=["Clear air  |  extinction 0.0002 /m","Extinction 0.6 /m","Rainfall 50 mm/h  |  extinction 0.1048 /m"]
        title="3D LiDAR simulation in three environments"
        subtitle="One fixed obstacle scene and one sensor configuration. Synchronized navigation, raw scans and accumulated measured maps."
        selection=None;filename="environment_conditions.mp4"
    else:
        ids=CONTROLLED_CASES if mode=="factorial" else ("dense_bright","sparse_dark")
        runs=[load_recording(directory,c,args.seed,kind="replays") for c in ids]
        names={"dense_bright":"DENSE SAMPLING / BRIGHT TARGET","dense_dark":"DENSE SAMPLING / DARK TARGET",
               "sparse_bright":"SPARSE SAMPLING / BRIGHT TARGET","sparse_dark":"SPARSE SAMPLING / DARK TARGET"}
        labels=[names[c] for c in ids]
        if mode=="sensor":
            labels=["BEST ENDPOINT  |  dense + bright","WORST ENDPOINT  |  sparse + dark"]
        params=[]
        for run in runs:
            target=next(o for o in run.metadata["world_objects"] if o.get("label")=="study_target")
            sensor=run.metadata["sensor"];m=sensor["measurements_per_cycle"];rho=target["material"]["rho_d"]
            params.append(f"{m} azimuths x {sensor['n_channels']} channels = {m*sensor['n_channels']:,} rays/scan  |  target diffuse reflectance {rho:g}")
        title=("Sampling density and target reflectance: four controlled cases" if mode=="factorial"
               else "Best vs worst: sampling density and target reflectance")
        subtitle=("Columns isolate target reflectance; rows isolate sampling density. Same recorded sensor position and orientation in every panel."
                  if mode=="factorial" else "Favourable and adverse configured endpoints. Mount angles, FOV, noise, weather, scene geometry and replayed poses are identical.")
        selection={"configured_endpoints":["dense_bright","sparse_dark"],"factorial_cases":list(CONTROLLED_CASES),
                   "varied_parameters":["azimuth samples per scan","study_target diffuse reflectance"],
                   "fixed_parameters":["mount height, pitch, roll and yaw","horizontal/vertical FOV","receiver noise","weather","scene geometry","all other object materials","exact replay poses and scan times"],
                   "scope":"configured favourable/adverse endpoints for measurement yield, not a globally optimal sensor or a navigation safety ranking",
                   "causal_limit":"joint endpoints change two factors; paired rows/columns in the four-case video isolate each factor",
                   "reflectance_limit":"illustrative virtual diffuse coefficients, not calibrated real material measurements"}
        filename="sampling_reflectance_factorial.mp4" if mode=="factorial" else "sensor_state_best_vs_worst.mp4"
    return render_comparison(runs,labels,params,args.output or PROJECT/"simulalted_environments"/filename,
        title=title,subtitle=subtitle,summary=summary,summary_path=directory/"summary.json",mode=mode,selection=selection,
        fps=args.fps,width=args.width,height=args.height,hold_seconds=args.hold_seconds,ffmpeg=args.ffmpeg)
