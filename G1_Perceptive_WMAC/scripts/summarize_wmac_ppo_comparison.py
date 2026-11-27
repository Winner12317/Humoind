"""Aggregate paired physical trials and plot saved WMAC/PPO comparisons."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"axes.unicode_minus": False,
    "axes.spines.top": False, "axes.spines.right": False, "savefig.facecolor": "white"})


def wilson(successes, count):
    z = 1.959963984540054
    p = successes / count
    scale = 1 + z*z/count
    center = (p + z*z/(2*count)) / scale
    radius = z * math.sqrt(p*(1-p)/count + z*z/(4*count*count)) / scale
    return [center-radius, center+radius]


def aggregate(job, reports):
    walking = [x["results"]["walk"] for x in reports]
    recovery = [x["results"]["recovery"] for x in reports]
    n = sum(x["num_envs"] for x in recovery)
    successes = sum(x["recovery_success_count"] for x in recovery)
    times = [value for x in recovery for value in x["per_env"]["first_stable_recovery_seconds"] if value is not None]
    def weighted(data, key):
        return sum(x[key]*x["first_episode_control_steps"] for x in data) / sum(x["first_episode_control_steps"] for x in data)
    def mean_fraction(data, key):
        return sum(x[key]*x["num_envs"] for x in data) / n
    return {"label": job["label"], "algorithm": job["algorithm"], "iteration": job["iteration"],
        "seeds": [r["seed"] for r in reports], "trials_per_mode": n,
        "walk_xy_error": weighted(walking, "velocity_xy_error_mean"),
        "walk_yaw_error": weighted(walking, "yaw_velocity_error_mean"),
        "walk_survival": mean_fraction(walking,"first_episode_survival_fraction"),
        "walk_final_upright": mean_fraction(walking,"final_upright_fraction"),
        "recovery_success_count": successes, "recovery_success": successes/n,
        "recovery_success_wilson95": wilson(successes,n),
        "recovery_time_success_mean": float(np.mean(times)) if times else None,
        "recovery_time_success_median": float(np.median(times)) if times else None,
        "recovery_final_upright": mean_fraction(recovery,"final_upright_fraction"),
        "recovery_xy_error_all_phase": weighted(recovery,"velocity_xy_error_mean"),
        "recovery_yaw_error_all_phase": weighted(recovery,"yaw_velocity_error_mean"),
        "strict_then_walk": mean_fraction(recovery,"success_then_walk_fraction_all_starts"),
        "strict_walk_and_finish_upright": mean_fraction(recovery,"success_walk_and_finish_upright_fraction"),
        "task_reward_walk": weighted(walking,"task_reward_per_control_step"),
        "task_reward_recovery": weighted(recovery,"task_reward_per_control_step"),
        "initial_conditions_verified": all(r["initial_conditions_match_reference_within_1e_6"]
            for report in reports for r in report["results"].values())}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory",type=Path,required=True)
    parser.add_argument("--allow-incomplete",action="store_true")
    args=parser.parse_args()
    root=args.directory.resolve()
    manifest=json.loads((root/"manifest.json").read_text())
    seeds=manifest["protocol"]["seeds"]
    summaries=[]
    raw={}
    initial_reference={}
    initial_audit=[]
    for job in manifest["checkpoints"]:
        reports=[]
        for seed in seeds:
            path=root/"evaluations"/job["label"]/f"seed_{seed}"/"metrics.json"
            if path.exists():
                d=json.loads(path.read_text())
                if d["seed"] != seed or d["checkpoint_sha256"] != job["sha256"] or d["protocol"] != manifest["protocol"]:
                    raise SystemExit(f"Mismatched protocol/checkpoint identity: {path}")
                if set(d["results"])=={"walk","recovery"}:
                    for mode,result in d["results"].items():
                        if not result["initial_conditions_match_reference_within_1e_6"]:
                            raise SystemExit(f"Initial pairing failed: {path} {mode}")
                        if result["num_envs"]!=manifest["protocol"]["num_envs"] or result["steps_requested"]!=manifest["protocol"]["steps"]:
                            raise SystemExit(f"Mismatched evaluation size: {path} {mode}")
                        with np.load(path.parent/f"initial_{mode}.npz") as data:
                            arrays={name:data[name].copy() for name in data.files}
                        expected_shapes={name:(manifest["protocol"]["num_envs"],width)
                            for name,width in {"state":71,"context":35,"command":3,"actor_history":384}.items()}
                        if set(arrays)!=set(expected_shapes):raise SystemExit(f"Initial array fields differ: {path}")
                        key=(seed,mode)
                        if key not in initial_reference:initial_reference[key]=arrays
                        for name,values in arrays.items():
                            ref=initial_reference[key][name]
                            if values.shape!=expected_shapes[name] or values.dtype!=ref.dtype or not np.isfinite(values).all():
                                raise SystemExit(f"Initial array shape/dtype/finite check failed: {path} {name}")
                            difference=float(np.max(np.abs(values-ref)))
                            if difference>1e-6:raise SystemExit(f"Initial array values differ: {path} {name}: {difference}")
                            initial_audit.append({"policy":job["label"],"seed":seed,"mode":mode,"field":name,
                                "shape":list(values.shape),"dtype":str(values.dtype),"exactly_equal":bool(np.array_equal(values,ref)),
                                "max_absolute_difference":difference})
                    reports.append(d)
        if len(reports)!=len(seeds):
            if args.allow_incomplete:continue
            raise SystemExit(f"Incomplete results: {job['label']}")
        raw[job["label"]]=reports
        summaries.append(aggregate(job,reports))
    if not summaries:raise SystemExit("No complete policy evaluations yet")
    tilt_rows=[]
    for label,reports in raw.items():
        counts={name:[0,0] for name in ("up_z_below_0.5","up_z_0.5_to_0.8","up_z_at_least_0.8")}
        for report in reports:
            initial=root/"evaluations"/label/f"seed_{report['seed']}"/"initial_recovery.npz"
            with np.load(initial) as data:quat=data["state"][:,3:7]
            quat=quat/np.linalg.norm(quat,axis=-1,keepdims=True)
            up_z=1-2*(quat[:,1]**2+quat[:,2]**2)
            success=np.asarray(report["results"]["recovery"]["per_env"]["recovered"],dtype=bool)
            selections={"up_z_below_0.5":up_z<.5,"up_z_0.5_to_0.8":(up_z>=.5)&(up_z<.8),"up_z_at_least_0.8":up_z>=.8}
            for name,selection in selections.items():
                counts[name][0]+=int((success&selection).sum());counts[name][1]+=int(selection.sum())
        tilt_rows.append({"label":label,"groups":{name:{"success":s,"count":n} for name,(s,n) in counts.items()}})
    (root/"recovery_by_initial_tilt.json").write_text(json.dumps(tilt_rows,ensure_ascii=False,indent=2)+"\n")
    (root/"initial_state_pairing_audit.json").write_text(json.dumps({"tolerance":1e-6,"arrays":initial_audit,
        "maximum_absolute_difference":max(x["max_absolute_difference"] for x in initial_audit),
        "all_exactly_equal":all(x["exactly_equal"] for x in initial_audit)},ensure_ascii=False,indent=2)+"\n")
    by_label={x["label"]:x for x in summaries}
    pair_specs=[("same_iteration_16000","wmac_16000","ppo_16000"),
        ("same_iteration_20000","wmac_20000","ppo_20000"),
        ("same_iteration_26000","wmac_26000","ppo_26000"),
        ("same_real_samples_20000","wmac_20000","ppo_2100"),
        ("same_real_samples_26000","wmac_26000","ppo_2700"),
        ("same_training_time_20000","wmac_20000","ppo_6200"),
        ("same_training_time_26000","wmac_26000","ppo_8100"),
        ("historical_best_reference","wmac_26000","ppo_60000")]
    pairs=[]
    for kind,new,old in pair_specs:
        if new not in raw or old not in raw:continue
        records=[]
        for seed in seeds:
            a=next(r for r in raw[new] if r["seed"]==seed)
            b=next(r for r in raw[old] if r["seed"]==seed)
            av=np.asarray(a["results"]["recovery"]["per_env"]["recovered"],dtype=bool)
            bv=np.asarray(b["results"]["recovery"]["per_env"]["recovered"],dtype=bool)
            records.append({"seed":seed,"both_success":int((av&bv).sum()),
                "new_only_success":int((av&~bv).sum()),"old_only_success":int((~av&bv).sum()),
                "both_fail":int((~av&~bv).sum())})
        pairs.append({"comparison":kind,"new":new,"old":old,
            "recovery_success_difference_pp":100*(by_label[new]["recovery_success"]-by_label[old]["recovery_success"]),
            "walk_error_new_minus_old":by_label[new]["walk_xy_error"]-by_label[old]["walk_xy_error"],
            "paired_recovery_outcomes":records})
    payload={"protocol":manifest["protocol"],"policy_summaries":summaries,"comparisons":pairs,
        "complete":len(summaries)==len(manifest["checkpoints"]),
        "uncertainty_note":"Wilson intervals describe only approximate uncertainty from these evaluation trials, not variation across training seeds; related mocap frames are correlated."}
    (root/"summary.json").write_text(json.dumps(payload,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
    with (root/"summary.csv").open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=list(summaries[0]));writer.writeheader();writer.writerows(summaries)
    # The training selection curve is deliberately separate from unseen tests.
    periodic=json.loads((root/"wmac_periodic_evaluations.json").read_text())
    xs=[x["iteration"] for x in periodic]
    fig,axes=plt.subplots(2,2,figsize=(11.4,7.2))
    axes[0,0].plot(xs,[x["results"]["recovery"]["recovery_success_fraction"]*100 for x in periodic],"o-",label="Stood up at least once")
    axes[0,0].plot(xs,[x["results"]["recovery"]["final_upright_fraction"]*100 for x in periodic],"s--",label="Upright at the end")
    axes[0,0].set(ylabel="Recovery starts (%)",ylim=(-5,105),title="Recovery rates vary across checkpoints")
    axes[0,0].legend(fontsize=8)
    axes[0,1].plot(xs,[x["results"]["walk"]["velocity_xy_error_mean"] for x in periodic],"o-",label="Walking XY")
    axes[0,1].plot(xs,[x["results"]["recovery"]["velocity_xy_error_mean"] for x in periodic],"s--",label="Full recovery episode XY")
    axes[0,1].set(ylabel="Velocity error (m/s)",title="Velocity tracking in physical simulation")
    axes[0,1].legend(fontsize=8)
    with (root/"wmac_training_cutoff26000.csv").open() as f:train=list(csv.DictReader(f))
    sampled=train[::50]
    axes[1,0].plot([int(x["iteration"]) for x in sampled],[float(x["validation_model_error"]) for x in sampled],lw=.9,label="Held-out state error")
    axes[1,0].plot([int(x["iteration"]) for x in sampled],[float(x["validation_recovery_error"]) for x in sampled],lw=.7,alpha=.7,label="Recovery subset")
    axes[1,0].set(yscale="log",ylabel="Normalized prediction error",title="Model error and policy stability can diverge")
    axes[1,0].legend(fontsize=8)
    axes[1,1].plot(xs,[x["results"]["recovery"]["success_then_walk_fraction_all_starts"]*100 for x in periodic],"o-")
    axes[1,1].set(ylabel="Stood up and kept walking (%)",ylim=(-5,105),title="Periodic test: continued-walking tolerance 0.5")
    for ax in axes.flat:
        ax.set_xlabel("WMAC outer iteration");ax.grid(alpha=.18)
    fig.suptitle("WMAC through iteration 26,000: periodic 5 s evaluations, seed 2026, 32 environments",fontsize=14)
    fig.tight_layout(rect=(0,0,1,.94));fig.savefig(root/"wmac_convergence.png",dpi=170);plt.close(fig)

    available=[i for i in (16000,20000,26000) if f"wmac_{i}" in by_label and f"ppo_{i}" in by_label]
    if available:
        fig,axes=plt.subplots(1,3,figsize=(12,4.0))
        x=np.arange(len(available));width=.36
        for ax,metric,title,scale in [
            (axes[0],"recovery_success","Stood up within 10 s (%)",100),
            (axes[1],"walk_xy_error","Walking velocity error (m/s, lower is better)",1),
            (axes[2],"recovery_final_upright","Upright at end of recovery (%)",100)]:
            for off,algo,name,color in [(-width/2,"wmac","WMAC world model","#26798b"),(width/2,"ppo","Nov 4 PPO","#b98650")]:
                vals=[by_label[f"{algo}_{i}"][metric]*scale for i in available]
                bars=ax.bar(x+off,vals,width,label=name,color=color)
                for b,v in zip(bars,vals):ax.text(b.get_x()+width/2,v+(.018 if scale==1 else 1.2),f"{v:.3f}" if scale==1 else f"{v:.1f}",ha="center",fontsize=8)
            ax.set_title(title,fontsize=10);ax.set_xticks(x,[str(i) for i in available]);ax.set_xlabel("Outer iteration")
            if scale==100:ax.set_ylim(0,114)
            else:ax.set_ylim(0,max(by_label[f"{a}_{i}"][metric] for a in ("wmac","ppo") for i in available)*1.25)
            ax.grid(axis="y",alpha=.15);ax.set_axisbelow(True)
        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, fontsize=9, loc="upper center", bbox_to_anchor=(.5,.92), ncol=2, frameon=False)
        fig.suptitle("Physical evaluation at matched iterations: seeds 2030 and 2031, 256 starts per mode",fontsize=13)
        fig.tight_layout(rect=(0,0,1,.85));fig.savefig(root/"same_iteration_comparison.png",dpi=170);plt.close(fig)
    print(json.dumps({"completed_policies":len(summaries),"complete":payload["complete"],"output":str(root)},ensure_ascii=False))


if __name__=="__main__":main()
