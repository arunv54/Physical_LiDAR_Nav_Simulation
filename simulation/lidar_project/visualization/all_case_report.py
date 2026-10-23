"""Weather-only report figures from an unchanged obstacle setup and saved runs.

Every plot reads existing JSON, CSV, or NPZ results. No sensing, trajectory
planning, map building, smoothing, or experiment rerun occurs here. The three
weather conditions are clear, rain 50 mm/h, and illustrative extinction 0.6/m;
all three seeds are retained, including their original chronology and terminal
states. This module intentionally does not compare different scene layouts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

PROJECT = Path(__file__).resolve().parents[2]
CASES = ("baseline", "rain", "fog")
LABELS = {"baseline": "Clear air", "rain": "Rain 50 mm/h", "fog": "Extinction 0.6 m⁻¹"}
COLORS = {"baseline": "#2579A8", "rain": "#3D9C94", "fog": "#C8753F"}
SEED_STYLES = {1: "-", 2: "--", 3: ":"}

def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _num(value, scale=1):
    return None if value is None else float(value) * scale




def _load_paths(directory, runs):
    paths = []
    for run in runs:
        source = directory / run["run_directory"] / "trajectory.npz"
        metadata_path = source.with_suffix(".json")
        with np.load(source, allow_pickle=False) as d:
            xy = d["states_xy"].copy()
            t = d["states_t"].copy()
            reference = d["reference_xy"].copy()
        if len(xy) != run["state_count"] or len(xy) != len(t):
            raise ValueError(f"State count mismatch: {source}")
        if not np.all(np.isfinite(xy)) or not np.all(np.diff(t) > 0):
            raise ValueError(f"Invalid recorded trajectory: {source}")
        if not np.isclose(t[-1], run["elapsed_s"], atol=1e-8):
            raise ValueError(f"Terminal time mismatch: {source}")
        metadata = _read(metadata_path)
        if metadata["status"] != run["status"] or metadata["seed"] != run["seed"]:
            raise ValueError(f"Run identity mismatch: {metadata_path}")
        paths.append({"case_id": run["case_id"], "layout_id": run.get("layout_id"),
                      "seed": run["seed"], "status": run["status"], "xy": xy, "t": t,
                      "reference": reference, "metadata": metadata,
                      "source": source, "sha256": hashlib.sha256(source.read_bytes()).hexdigest()})
    return paths


def _style():
    return {"font.family": "DejaVu Sans", "font.size": 7, "axes.labelsize": 6.5,
            "axes.titlesize": 7.5, "xtick.labelsize": 6, "ytick.labelsize": 6,
            "text.color": "#24333F", "axes.labelcolor": "#24333F", "axes.edgecolor": "#B8C2C9",
            "savefig.facecolor": "white", "figure.facecolor": "white", "pdf.fonttype": 42,
            "ps.fonttype": 42, "axes.unicode_minus": True}


def _save(fig, output, name):
    import matplotlib.pyplot as plt

    fig.savefig(output / f"{name}.png", dpi=260)
    fig.savefig(output / f"{name}.pdf")
    plt.close(fig)




def _object_faces(obj):
    """Create display polygons directly from stored geometry, without simulation."""
    kind = obj["type"]
    if kind == "box":
        c, half = np.asarray(obj["center"], float), np.asarray(obj["size"], float)/2
        corners = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1],
                            [-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]], float)*half
        angle = np.radians(obj.get("yaw_deg", 0))
        R = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0,0,1]])
        vertices = corners @ R.T + c
        return [vertices[index] for index in ([0,1,2,3],[4,5,6,7],[0,1,5,4],[1,2,6,5],[2,3,7,6],[3,0,4,7])]
    if kind == "cylinder":
        phi = np.linspace(0, 2*np.pi, 17)
        base = np.asarray(obj["base"], float)
        ring = base + np.c_[obj["radius"]*np.cos(phi), obj["radius"]*np.sin(phi), np.zeros(len(phi))]
        top = ring + [0, 0, obj["height"]]
        return [top[:-1], ring[:-1]] + [np.array([ring[i],ring[i+1],top[i+1],top[i]]) for i in range(len(phi)-1)]
    if kind == "sphere":
        theta = np.linspace(0, np.pi, 7)
        phi = np.linspace(0, 2*np.pi, 13)
        vertices = np.array([[np.sin(t)*np.cos(p),np.sin(t)*np.sin(p),np.cos(t)] for t in theta for p in phi])*obj["radius"]+obj["center"]
        return [vertices[[i*13+j,i*13+j+1,(i+1)*13+j+1,(i+1)*13+j]] for i in range(6) for j in range(12)]
    return []


def _draw_scene(ax, run, *, alpha=.11):
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    metadata = run["metadata"]
    withheld = {o["label"] for o in metadata["config"]["world"]["unexpected_objects"]}
    for obj in metadata["world_objects"]:
        faces = _object_faces(obj)
        if not faces:
            continue
        unknown = obj.get("label") in withheld
        collection = Poly3DCollection(faces, facecolor="#C8753F" if unknown else "#8E9CA7",
                                     edgecolor="#A16C4B" if unknown else "#7F8D97",
                                     alpha=alpha, linewidth=.25)
        ax.add_collection3d(collection)


def _axes_3d(ax, title):
    ax.set(xlim=(-2,21), ylim=(-6,7), zlim=(0,3), title=title)
    ax.set_box_aspect((23,13,3), zoom=1.1)
    ax.view_init(elev=37, azim=-112)
    ax.set_xticks([0,10,20]); ax.set_yticks([-5,0,5]); ax.set_zticks([0,2])
    ax.set_xlabel("x (m)", labelpad=-6); ax.set_ylabel("y (m)", labelpad=-8)
    ax.set_zlabel("z (m)", labelpad=-8)
    ax.tick_params(labelsize=5.5, pad=-2, length=1)
    ax.set_title(title, loc="left", pad=3, fontsize=7.2)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.fill = False
        axis._axinfo["grid"]["color"] = (.83,.86,.88,.65)
        axis._axinfo["grid"]["linewidth"] = .3




def _reference(ax, run):
    r = run["reference"]
    ax.plot(r[:,0],r[:,1],np.zeros(len(r)), color="#4B5560", ls=(0,(3,2)), lw=.55, alpha=.6)
    start, goal = run["metadata"]["config"]["robot"]["start"], run["metadata"]["config"]["robot"]["goal"]
    ax.plot(*start,0, marker="o", ms=2.6, color="#24333F")
    ax.plot(*goal,0, marker="*", ms=5, color="#24333F")




def _csv(path, rows):
    with path.open("w",encoding="utf-8",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _weather_metrics(summary):
    result=[]
    for case_id in CASES:
        case=next(c for c in summary["cases"] if c["id"]==case_id)
        a,r=case["aggregate"],case["replay"]
        result.append({"case_id":case_id,"label":LABELS[case_id],"runs":a["runs"],
                       "goals":a["goals"],"collisions":a["collisions"],"timeouts":a["timeouts"],
                       "mean_points_per_scan":a["mean_returns_per_control_step"],
                       "mean_observed_fraction":a["mean_observed_fraction"],
                       "target_range_rms_mm":_num(r["range_rmse_m"],1000),
                       "target_detection_probability":r["observed_detection_probability"],
                       "mean_eligible_power_w":r["mean_eligible_power_w"],
                       "mean_elapsed_s":a["mean_elapsed_s"],"mean_path_length_m":a["mean_path_length_m"],
                       "worst_surface_gap_m":a["worst_surface_gap_m"],
                       "mean_spatial_route_rmse_m":a["mean_spatial_route_rmse_m"],
                       "target_detected_returns":r["detected_returns"],"target_eligible_echoes":r["eligible_echoes"]})
    return result


def _weather_paths(directory, summary):
    selected=[r for r in summary["runs"] if r["case_id"] in CASES]
    paths=_load_paths(directory,selected)
    sources=[]
    reference_world=None
    reference_reference=None
    for path in paths:
        run_dir=path["source"].parent
        world=json.dumps(path["metadata"]["config"]["world"],sort_keys=True)
        if reference_world is None:
            reference_world=world
            reference_reference=path["reference"]
        if world!=reference_world:
            raise ValueError("Weather comparison changed obstacle geometry or material")
        if not np.array_equal(path["reference"],reference_reference):
            raise ValueError("Weather comparison changed the geometric reference route")
        with np.load(run_dir/"trajectory.npz",allow_pickle=False) as d:
            path["scan_t"]=d["controls_t"].copy()
            path["returns"]=d["controls_n_returns"].copy()
            path["gap"]=d["controls_swept_surface_gap"].copy()
            path["route_error"]=d["states_tracking_error"].copy()
        with np.load(run_dir/"mapping.npz",allow_pickle=False) as d:
            ground=d["ground_first_seen"]
            obstacle=d["obstacle_first_seen"]
            first=d["map_first_seen"]
            path["map_xyz"]=d["map_xyz"].copy() if path["seed"]==1 else None
            path["map_first_seen"]=first.copy() if path["seed"]==1 else None
        if len(path["scan_t"])!=len(path["t"])-1 or not np.allclose(path["scan_t"],path["t"][:-1],rtol=0,atol=1e-10):
            raise ValueError("Saved scan chronology differs from control snapshots")
        if len(first) and (first.min()<0 or first.max()>=len(path["scan_t"])):
            raise ValueError("Map contains a point observed after the final saved scan")
        counts=[]
        for step in range(len(path["scan_t"])):
            observed=((ground>=0)&(ground<=step))|((obstacle>=0)&(obstacle<=step))
            counts.append(int(observed.sum()))
        if np.any(np.diff(counts)<0):
            raise ValueError("Cumulative endpoint-cell count decreases")
        path["observed_cells"]=np.asarray(counts)
        path["roi_cells"]=int(ground.size)
        path["observed_percent"]=100*np.asarray(counts)/ground.size
        mapping=_read(run_dir/"mapping.json")
        if counts[-1]!=mapping["observed_cell_count"]:
            raise ValueError("Endpoint-cell chronology disagrees with saved final map summary")
        if len(first)!=mapping["map_point_count"]:
            raise ValueError("Saved map count disagrees with voxel representatives")
        path["map_summary"]=mapping
        sources += [run_dir/name for name in ("trajectory.npz","trajectory.json","mapping.npz","mapping.json")]
    identities={(p["case_id"],p["seed"]) for p in paths}
    if identities!={(c,s) for c in CASES for s in (1,2,3)}:
        raise ValueError("Expected every weather case and all three saved seeds")
    return paths,sources,hashlib.sha256(reference_world.encode()).hexdigest()


def _weather_replays(directory,case_ids=CASES):
    replays=[];sources=[]
    for case_id in case_ids:
        for seed in (1,2,3):
            source=directory/"replays"/case_id/f"seed_{seed}"/"traces.csv"
            with source.open(encoding="utf-8",newline="") as stream:
                rows=[{key:(float(value) if value!="" else None) for key,value in row.items()} for row in csv.DictReader(stream)]
            replays.append({"case_id":case_id,"seed":seed,"rows":rows})
            sources.append(source)
    for seed in (1,2,3):
        traces=[next(r["rows"] for r in replays if r["case_id"]==c and r["seed"]==seed) for c in case_ids]
        pose_fields=("t","robot_x_m","robot_y_m","robot_theta_rad")
        ref=np.array([[r[k] for k in pose_fields] for r in traces[0]])
        for trace in traces[1:]:
            if not np.array_equal(ref,np.array([[r[k] for k in pose_fields] for r in trace])):
                raise ValueError("Weather target measurements were not replayed at identical poses")
    return replays,sources


def _measurement_profiles(replays):
    """Pool at actual common control timestamps, never average per-pose RMS."""
    result=[]
    for case_id in CASES:
        rows=[row for replay in replays if replay["case_id"]==case_id for row in replay["rows"]]
        times=sorted({r["t"] for r in rows})
        sum_squared=0.;detections=0
        for time in times:
            current=[row for row in rows if row["t"]==time]
            count=sum(r["detected_returns"] for r in current)
            eligible=sum(r["eligible_echoes"] for r in current)
            power=sum(r["eligible_power_sum_w"] for r in current)
            sum_squared+=sum(r["range_error_sq_sum_m2"] for r in current)
            detections+=count
            result.append({"case_id":case_id,"time_s":time,"seed_snapshots":len(current),
                           "mean_target_returns":count/len(current),
                           "mean_eligible_power_w":power/eligible if eligible else None,
                           "pooled_cumulative_target_rms_mm":1000*np.sqrt(sum_squared/detections) if detections else None,
                           "pooled_cumulative_error_sq_m2":sum_squared,"pooled_cumulative_target_returns":detections,
                           "eligible_echoes_at_time":eligible})
    return result


def _weather_legend(fig, *, seeds=False, y=.99):
    from matplotlib.lines import Line2D

    handles=[Line2D([0],[0],color=COLORS[c],lw=1.5,label=LABELS[c]) for c in CASES]
    if seeds:
        handles += [Line2D([0],[0],color="#5C6770",lw=.8,ls=SEED_STYLES[s],label=f"Seed {s}") for s in (1,2,3)]
    fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.51,y),ncol=6 if seeds else 3,
               fontsize=6.5,frameon=False,handlelength=1.8,columnspacing=1.3)


def _tidy_axes(ax):
    ax.spines[["top","right"]].set_visible(False)
    ax.grid(axis="y",color="#DFE5E8",lw=.45)
    ax.set_axisbelow(True)
    ax.tick_params(length=2,pad=2)


def render_weather_measurements(profiles,output):
    import matplotlib.pyplot as plt

    with plt.rc_context(_style()):
        fig,axs=plt.subplots(1,3,figsize=(6.5,2.6))
        fig.subplots_adjust(left=.078,right=.985,bottom=.22,top=.79,wspace=.47)
        for case_id in CASES:
            rows=[r for r in profiles if r["case_id"]==case_id]
            t=np.array([r["time_s"] for r in rows])
            for ax,key,scale in zip(axs,("mean_target_returns","mean_eligible_power_w","pooled_cumulative_target_rms_mm"),(1,1e6,1)):
                y=np.array([np.nan if r[key] is None else r[key]*scale for r in rows])
                ax.plot(t,y,color=COLORS[case_id],lw=1.05)
        labels=[("a) Target returns at shared poses","Detected target returns / pose"),
                ("b) Received-power dependency","Eligible mean echo power (μW)"),
                ("c) Detected-return accuracy","Cumulative target RMS (mm)")]
        for ax,(title,label) in zip(axs,labels):
            _tidy_axes(ax);ax.set_title(title,loc="left",fontsize=7.1,pad=6)
            ax.set_xlabel("Saved replay time (s)");ax.set_ylabel(label)
            ax.set_xlim(0,18.2);ax.set_xticks([0,6,12,18])
        axs[1].set_yscale("log")
        axs[2].set_ylim(bottom=0)
        _weather_legend(fig,y=.99)
        fig.text(.045,.075,"All 3 seeds use identical saved poses across weather cases. Power is conditioned on an eligible geometric echo.",fontsize=6)
        fig.text(.045,.025,"RMS = √(Σ detected-return squared errors / Σ detected returns), pooled cumulatively; gaps indicate no eligible echo / no detections.",fontsize=6)
        _save(fig,output,"weather_measurements")


def render_weather_trajectories(paths,output):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    with plt.rc_context(_style()):
        fig=plt.figure(figsize=(6.5,3.0))
        ax=fig.add_axes([.035,.11,.62,.72],projection="3d")
        baseline=next(p for p in paths if p["case_id"]=="baseline" and p["seed"]==1)
        _axes_3d(ax,"")
        _draw_scene(ax,baseline,alpha=.14);_reference(ax,baseline)
        ax.view_init(elev=37,azim=-112)
        ax.set_box_aspect((23,13,3),zoom=1.15)
        zoom=fig.add_axes([.735,.245,.24,.52])
        # The zoom is exact world coordinates around the withheld target.
        for obj in baseline["metadata"]["world_objects"]:
            if obj["type"] in ("box","cylinder","sphere"):
                faces=_object_faces(obj)
                if obj["type"]=="box":
                    polygon=faces[0][:,:2]
                elif obj["type"]=="cylinder":
                    polygon=faces[0][:,:2]
                else:
                    phi=np.linspace(0,2*np.pi,40)
                    polygon=np.array(obj["center"][:2])+obj["radius"]*np.c_[np.cos(phi),np.sin(phi)]
                unknown=obj["label"].startswith("unexpected") or obj["label"]=="study_target"
                zoom.add_patch(Polygon(polygon,fc="#C8753F" if unknown else "#8E9CA7",ec="none",alpha=.18))
        for p in paths:
            xy=p["xy"];style=SEED_STYLES[p["seed"]];color=COLORS[p["case_id"]]
            ax.plot(xy[:,0],xy[:,1],np.zeros(len(xy)),color=color,lw=.85,ls=style,alpha=.9,zorder=9)
            zoom.plot(xy[:,0],xy[:,1],color=color,lw=.85,ls=style,alpha=.9)
        zoom.set(xlim=(8.5,14.5),ylim=(.3,4.2),xlabel="x (m)",ylabel="y (m)")
        zoom.set_aspect("equal",adjustable="box")
        zoom.set_xticks([9,11,13]);zoom.set_yticks([1,2,3,4]);zoom.grid(color="#E2E6E9",lw=.4)
        zoom.set_title("Exact top-down zoom",fontsize=7.1,pad=5)
        zoom.tick_params(labelsize=6,pad=1)
        _weather_legend(fig,seeds=True,y=1.0)
        fig.text(.045,.075,"All 9 saved paths in one unchanged obstacle setup. Ground-plane robot motion (z = 0); grey / orange: known / withheld objects.",fontsize=6)
        fig.text(.045,.025,"No path smoothing or spatial offsets. Clear and rain paths nearly coincide; seed line styles preserve all original trajectories.",fontsize=6)
        _save(fig,output,"weather_trajectories")


def render_weather_profiles(paths,output):
    import matplotlib.pyplot as plt

    with plt.rc_context(_style()):
        fig,axs=plt.subplots(2,2,figsize=(6.5,2.5),sharex=True)
        fig.subplots_adjust(left=.085,right=.985,bottom=.17,top=.79,hspace=.32,wspace=.29)
        for p in paths:
            color,style=COLORS[p["case_id"]],SEED_STYLES[p["seed"]]
            axs[0,0].plot(p["scan_t"],p["returns"],color=color,ls=style,lw=.72,alpha=.85)
            # The terminal state has no additional scan; hold the final recorded map.
            axs[0,1].step(np.r_[p["scan_t"],p["t"][-1]],np.r_[p["observed_percent"],p["observed_percent"][-1]],
                          where="post",color=color,ls=style,lw=.72,alpha=.85)
            axs[1,0].plot(p["scan_t"],p["gap"],color=color,ls=style,lw=.72,alpha=.85)
            axs[1,1].plot(p["t"],p["route_error"],color=color,ls=style,lw=.72,alpha=.85)
        axs[1,0].axhline(0,color="#9A5B3A",ls=(0,(3,2)),lw=.65)
        for ax,label in zip(axs.flat,("Points / scan","Endpoint cells (%)","Swept gap (m)","Route error (m)")):
            _tidy_axes(ax);ax.set_ylabel(label,fontsize=6.6)
            ax.set_xlim(0,18.2);ax.set_xticks([0,6,12,18]);ax.tick_params(labelsize=6)
        for ax in axs[1]:ax.set_xlabel("Saved simulation time (s)")
        axs[0,1].set_ylim(0,100)
        _weather_legend(fig,seeds=True,y=1.015)
        _save(fig,output,"weather_profiles")


def render_weather_maps(paths,output):
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize

    with plt.rc_context(_style()):
        fig=plt.figure(figsize=(6.5,2.2))
        normalization=Normalize(0,2.6)
        for index,case_id in enumerate(CASES):
            p=next(p for p in paths if p["case_id"]==case_id and p["seed"]==1)
            xyz=p["map_xyz"]
            xyz=xyz[xyz[:,2]>.12]
            ax=fig.add_axes([.005+index*.315,.08,.305,.77],projection="3d")
            ax.scatter(*xyz.T,c=xyz[:,2],cmap="viridis",norm=normalization,s=.42,
                       linewidths=0,alpha=.95,depthshade=False,rasterized=True)
            r=p["reference"];xy=p["xy"]
            ax.plot(r[:,0],r[:,1],np.zeros(len(r)),color="#7E8790",ls=(0,(3,2)),lw=.55)
            ax.plot(xy[:,0],xy[:,1],np.zeros(len(xy)),color=COLORS[case_id],lw=1.0)
            _axes_3d(ax,"")
            ax.set_box_aspect((23,13,3),zoom=1.0)
            ax.set_zlabel("");ax.set_zticklabels([]);ax.set_ylabel("")
            fig.text(.02+index*.315,.92,LABELS[case_id],fontsize=7.4,color=COLORS[case_id])
            fig.text(.02+index*.315,.85,f"{p['map_summary']['observed_cell_fraction']*100:.1f}% observed cells",fontsize=6.3)
        cax=fig.add_axes([.954,.26,.012,.48])
        bar=fig.colorbar(plt.cm.ScalarMappable(norm=normalization,cmap="viridis"),cax=cax,ticks=[0,1,2])
        bar.set_label("Measured height (m)",fontsize=6,labelpad=2);bar.ax.tick_params(labelsize=6,pad=1,length=2)
        fig.text(.015,.025,"Seed 1 measured voxel endpoints only; display hides z ≤ 0.12 m. Coverage uses all heights. Known exact pose; no SLAM.",fontsize=5.8)
        _save(fig,output,"weather_maps")


TARGET_CASES=("baseline","reflectance_dark","reflectance_bright","target_yaw_45","target_yaw_75",
              "target_short","target_tall","target_near","target_far")
TARGET_LABELS=("Baseline: ρ = 0.30","Reflectance 0.02","Reflectance 0.85","Target yaw 45°","Target yaw 75°",
               "Target height 0.3 m","Target height 2.0 m","Target x = 9 m","Target x = 13 m")


def _target_metrics(summary):
    result=[]
    for case_id,label in zip(TARGET_CASES,TARGET_LABELS):
        case=next(c for c in summary["cases"] if c["id"]==case_id)
        r=case["replay"]
        result.append({"case_id":case_id,"label":label,"replay_poses":r["pose_count"],
                       "target_opportunities_per_pose":r["target_opportunities"]/r["pose_count"],
                       "visible_target_echoes_per_pose":r["visible_target_echoes"]/r["pose_count"],
                       "eligible_echoes_per_pose":r["eligible_echoes"]/r["pose_count"],
                       "detected_target_returns_per_pose":r["detected_returns"]/r["pose_count"],
                       "eligible_mean_power_w":r["mean_eligible_power_w"],
                       "eligible_mean_range_m":r["mean_eligible_range_m"],
                       "eligible_mean_incidence_deg":r["mean_eligible_incidence_deg"],
                       "detected_target_range_rms_mm":_num(r["range_rmse_m"],1000),
                       "detected_target_returns":r["detected_returns"],
                       "target_detection_probability":r["observed_detection_probability"]})
    return result


def render_target_characteristics(rows,output):
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    with plt.rc_context(_style()):
        fig=plt.figure(figsize=(6.5,3.2))
        axes=fig.subplots(1,4,gridspec_kw={"width_ratios":[1.1,1.25,1.05,1.1]})
        fig.subplots_adjust(left=.213,right=.995,bottom=.34,top=.79,wspace=.40)
        y=np.arange(len(rows))
        for ax in axes:
            ax.set_ylim(len(rows)-.45,-.65)
            ax.set_yticks(y);ax.set_yticklabels([]);ax.tick_params(axis="y",length=0)
            ax.spines[["top","right","left"]].set_visible(False)
            ax.grid(axis="x",color="#E2E7EA",lw=.4);ax.set_axisbelow(True)
        powers=np.array([r["eligible_mean_power_w"] for r in rows])*1000
        axes[0].barh(y,powers,height=.53,color="#2579A8",alpha=.85)
        axes[0].set_yticklabels([r["label"] for r in rows],fontsize=6.4)
        axes[0].set_xlim(0,2.6);axes[0].set_xticks([0,1,2])
        axes[0].set_xlabel("Mean power (mW)")
        axes[0].set_title("a) Eligible echo power",loc="left",fontsize=6.8,pad=7)
        for yi,value in zip(y,powers):axes[0].text(value+.04,yi,f"{value:.3f}",va="center",fontsize=5.3)
        for key,color,height in (("target_opportunities_per_pose","#D4DDE3",.72),
                                 ("eligible_echoes_per_pose","#3D9C94",.46),
                                 ("detected_target_returns_per_pose","#C8753F",.19)):
            axes[1].barh(y,[r[key] for r in rows],height=height,color=color)
        axes[1].set_xlim(0,340);axes[1].set_xticks([0,150,300]);axes[1].set_xlabel("Echoes / replay pose")
        axes[1].set_title("b) Geometric → detected",loc="left",fontsize=6.8,pad=7)
        axes[2].plot([r["detected_target_range_rms_mm"] for r in rows],y,"o",ms=3.3,color="#2579A8")
        axes[2].axvline(10,color="#BAC5CD",lw=.65,ls=(0,(3,2)))
        axes[2].set_xlim(9,11);axes[2].set_xticks([9,10,11]);axes[2].set_xlabel("Target RMS (mm)")
        axes[2].set_title("c) Conditional accuracy",loc="left",fontsize=6.8,pad=7)
        for yi,row in zip(y,rows):
            axes[2].text(10.31,yi,f"{row['detected_target_range_rms_mm']:.2f}",va="center",fontsize=5.7)
        axes[3].set_xlim(0,1);axes[3].axis("off")
        axes[3].text(.15,-.85,"r̄ (m)",ha="center",va="center",fontsize=6.6)
        axes[3].text(.77,-.85,"θ̄ (deg)",ha="center",va="center",fontsize=6.6)
        axes[3].set_title("d) Mean state",loc="left",fontsize=6.8,pad=7)
        for yi,row in zip(y,rows):
            axes[3].text(.15,yi,f"{row['eligible_mean_range_m']:.2f}",ha="center",va="center",fontsize=6.5)
            axes[3].text(.77,yi,f"{row['eligible_mean_incidence_deg']:.1f}",ha="center",va="center",fontsize=6.5)
        handles=[Line2D([0],[0],color=c,lw=4,label=l)for c,l in (("#D4DDE3","Target opportunities"),("#3D9C94","Eligible echoes"),("#C8753F","Detected returns"))]
        fig.legend(handles=handles,loc="upper center",bbox_to_anchor=(.55,.98),frameon=False,ncol=3,fontsize=6.4,handlelength=1.3)
        fig.text(.025,.19,"Nine target conditions · 27 replays of the same three saved baseline trajectories (271 poses per condition).",fontsize=6.2)
        fig.text(.025,.135,"RMS is pooled over detected target returns; eligible power/range/incidence are conditioned on geometric echoes within range gates.",fontsize=5.9)
        fig.text(.025,.08,"Position labels denote target world x, not a fixed sensing distance. Whole-route means mix changing range and incidence.",fontsize=5.9)
        fig.text(.025,.025,"Baseline target: x = 11 m, height 1.2 m, yaw 0°, diffuse reflectance 0.30. Other robot, sensor, and weather settings are fixed.",fontsize=5.9)
        _save(fig,output,"target_characteristics")


def render_weather_report(dependency_dir,output_dir):
    """Render weather-only figures and return the provenance manifest.

    ``dependency_dir`` is the saved ``data/navigation_dependencies`` directory.
    The same 3 seeds are used for all weather conditions. Scene and replay pose
    invariance are verified before rendering any figure.
    """
    directory,output=Path(dependency_dir).resolve(),Path(output_dir).resolve()
    output.mkdir(parents=True,exist_ok=True)
    summary_path=directory/"summary.json"
    summary=_read(summary_path)
    metrics=_weather_metrics(summary)
    paths,path_sources,world_hash=_weather_paths(directory,summary)
    replays,replay_sources=_weather_replays(directory)
    profiles=_measurement_profiles(replays)
    target_metrics=_target_metrics(summary)
    _,target_sources=_weather_replays(directory,TARGET_CASES)
    for row in metrics:
        last=next(r for r in reversed(profiles) if r["case_id"]==row["case_id"])
        if not np.isclose(last["pooled_cumulative_target_rms_mm"],row["target_range_rms_mm"],rtol=1e-12):
            raise ValueError("Pooled cumulative RMS does not reproduce saved case summary")
    render_weather_measurements(profiles,output)
    render_weather_trajectories(paths,output)
    render_weather_profiles(paths,output)
    render_weather_maps(paths,output)
    render_target_characteristics(target_metrics,output)
    _csv(output/"weather_metrics.csv",metrics)
    _csv(output/"target_metrics.csv",target_metrics)
    _csv(output/"measurement_profiles.csv",profiles)
    navigation=[];trajectories=[];map_endpoints=[]
    for p in paths:
        for step,(time,xy,error) in enumerate(zip(p["t"],p["xy"],p["route_error"])):
            terminal=step==len(p["scan_t"])
            scan_step=min(step,len(p["scan_t"])-1)
            identity={"case_id":p["case_id"],"seed":p["seed"],"step":step,"time_s":float(time)}
            trajectories.append(dict(identity,x_m=float(xy[0]),y_m=float(xy[1]),z_m=0.,status=p["status"]))
            navigation.append(dict(identity,has_scan=not terminal,
                                   detected_points=None if terminal else int(p["returns"][step]),
                                   observed_endpoint_cells=int(p["observed_cells"][scan_step]),
                                   observed_endpoint_percent=float(p["observed_percent"][scan_step]),
                                   swept_surface_gap_m=None if terminal else float(p["gap"][step]),
                                   spatial_route_error_m=float(error)))
        if p["seed"]==1:
            for xyz,first_step in zip(p["map_xyz"],p["map_first_seen"]):
                map_endpoints.append({"case_id":p["case_id"],"seed":1,"first_scan_step":int(first_step),
                                      "x_m":float(xyz[0]),"y_m":float(xyz[1]),"z_m":float(xyz[2]),
                                      "displayed_elevated_only":bool(xyz[2]>.12)})
    _csv(output/"navigation_profiles.csv",navigation)
    _csv(output/"trajectories.csv",trajectories)
    _csv(output/"map_endpoints.csv",map_endpoints)
    source_files=list(dict.fromkeys([summary_path]+path_sources+replay_sources+target_sources))
    manifest={"schema_version":1,"scope":"Weather changes only; one fixed primary obstacle setup; existing saved data only.",
              "conditions":[{"id":c,"label":LABELS[c]}for c in CASES],"runs":len(paths),"seeds":[1,2,3],
              "metrics":metrics,"run_identities":[{"case_id":p["case_id"],"seed":p["seed"],"status":p["status"],
                                                     "states":len(p["xy"]),"scans":len(p["scan_t"]),"terminal_time_s":float(p["t"][-1])}for p in paths],
              "target_characteristics":{"case_ids":list(TARGET_CASES),"conditions":9,"replays":27,"poses_per_condition":271,
                                        "identical_baseline_poses_per_seed":True,"metrics":target_metrics,
                                        "scope":"Target measurement dependencies only; no navigation causal claim from these altered target geometries."},
              "checks":{"identical_world_geometry_and_material":True,"world_config_sha256":world_hash,
                        "identical_reference_route":True,"identical_replay_poses_per_seed":True,
                        "scan_times_equal_nonterminal_states":True,"terminal_state_has_no_extra_scan":True,
                        "map_first_seen_within_saved_scan_range":True,"observed_endpoint_cells_monotone":True,
                        "cumulative_rms_reproduces_pooled_summary":True},
              "display":{"trajectory_smoothing":False,"trajectory_subsampling":False,"trajectory_z_m":0,
                         "figure_sizes_inches":{"weather_measurements":[6.5,2.6],"weather_trajectories":[6.5,3.0],
                                                "weather_profiles":[6.5,2.5],"weather_maps":[6.5,2.2],"target_characteristics":[6.5,3.2]},
                         "map_display":"Seed 1; all saved voxel representatives with z > 0.12 m; no additional map endpoints.",
                         "map_coverage":"All saved endpoint cells, including low and elevated endpoints; this is not map accuracy.",
                         "uncertainty":"Three recorded seed paths drawn separately; no confidence interval inferred from three seeds.",
                         "measurement_profiles":"Target-return counts average available same-time seed snapshots; power pools eligible geometric echoes; cumulative RMS pools squared errors/counts across all detected target echoes.",
                         "weather_limits":"Illustrative atmospheric extinction, not calibrated weather; attenuation only, without fog backscatter."},
              "figures":[name+ext for name in ("target_characteristics","weather_measurements","weather_trajectories","weather_profiles","weather_maps")for ext in (".png",".pdf")],
              "sources":[{"path":str(p.relative_to(directory)),"bytes":p.stat().st_size,
                          "sha256":hashlib.sha256(p.read_bytes()).hexdigest()}for p in source_files]}
    (output/"manifest.json").write_text(json.dumps(manifest,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dependency-dir",type=Path,default=PROJECT/"data/navigation_dependencies")
    parser.add_argument("--output-dir",type=Path,default=PROJECT/"outputs/all_case_report")
    args=parser.parse_args()
    manifest=render_weather_report(args.dependency_dir,args.output_dir)
    print(json.dumps({"output_dir":str(args.output_dir),"runs":manifest["runs"],"figures":manifest["figures"]},indent=2))


if __name__=="__main__":
    main()


