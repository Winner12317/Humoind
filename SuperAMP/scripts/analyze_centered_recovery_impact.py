"""Analyze same-snapshot PPO/current WMAC physics with per-start uncertainty.

No simulator, checkpoint execution, optimizer, or model fitting is used here.
Inputs are four physical rollouts per scenario from compare_recovery_effort.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
TRIALS = ("a", "a_repeat", "b", "b_repeat")
RMS = {"qacc", "force", "velocity", "fd_acc", "physical_jerk", "torque_rate",
       "action_delta", "action_second_difference", "base_linear_acc_fd", "root_linear_acc",
       "root_rotational_qacc", "constraint_torque"}
UNITS = {"qacc": "rad/s²", "force": "Nm", "velocity": "rad/s", "fd_acc": "rad/s²",
         "physical_jerk": "rad/s³", "torque_rate": "Nm/s", "action_delta": "action units",
         "action_second_difference": "action units", "base_linear_acc_fd": "m/s²",
         "last_qacc_cost": "sum(rad/s²)²", "fd_acc_cost": "sum(rad/s²)²",
         "clip_fraction": "fraction", "task_reward": "reward/control step",
         "root_linear_acc": "m/s²", "root_rotational_qacc": "free-joint rotational qacc units",
         "constraint_torque": "Nm (all constraints)"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(4 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def clean(value):
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def mean_sq(value):
    value = np.asarray(value, dtype=np.float64)
    return float(np.square(value).mean()) if value.size else np.nan


def magnitude(value):
    value = np.asarray(value, dtype=np.float64).reshape(-1)
    finite = value[np.isfinite(value)]
    if not finite.size:
        return dict(samples=int(value.size), nonfinite=int(value.size), rms=None, p95=None, p99=None, max=None)
    return dict(samples=int(value.size), nonfinite=int(value.size - finite.size),
                rms=float(np.sqrt(np.square(finite).mean())),
                p95=float(np.quantile(np.abs(finite), .95)),
                p99=float(np.quantile(np.abs(finite), .99)), max=float(np.abs(finite).max()))


def per_start(arrays, mask, dt, decimation, *, valid_history=None):
    """Each scalar is one initial state's time/joint mean, never a frame sample."""
    n = mask.shape[1]
    history = mask if valid_history is None else valid_history
    physical_mask = np.repeat(mask, decimation, axis=0)
    physical_history = np.repeat(history, decimation, axis=0)
    adjacent_physical = physical_mask[1:] & physical_history[:-1]
    delta_mask = mask.copy()
    delta_mask[0] = False
    delta_mask[1:] &= history[:-1]
    second_mask = delta_mask.copy()
    second_mask[:2] = False
    second_mask[2:] &= history[:-2] & history[1:-1]
    qacc = arrays["qacc"].astype(np.float64)
    force = arrays["force"].astype(np.float64)
    fd_acc = (arrays["next_state"][..., 42:71].astype(np.float64)
              - arrays["state"][..., 42:71].astype(np.float64)) / dt
    action_delta = arrays["action_delta"].astype(np.float64)
    second = np.zeros_like(action_delta)
    second[1:] = action_delta[1:] - action_delta[:-1]
    base_acc = (arrays["next_state"][..., 7:10].astype(np.float64)
                - arrays["state"][..., 7:10].astype(np.float64)) / dt
    rows = []
    for env in range(n):
        cm, pm = mask[:, env], physical_mask[:, env]
        dm, sm, jm = delta_mask[:, env], second_mask[:, env], adjacent_physical[:, env]
        row = {
            "qacc": mean_sq(qacc[pm, env]),
            "force": mean_sq(force[pm, env]),
            "velocity": mean_sq(arrays["next_state"][cm, env, 42:71]),
            "fd_acc": mean_sq(fd_acc[cm, env]),
            "physical_jerk": mean_sq(np.diff(qacc[:, env], axis=0)[jm] / (dt / decimation)),
            "torque_rate": mean_sq(np.diff(force[:, env], axis=0)[jm] / (dt / decimation)),
            "action_delta": mean_sq(action_delta[dm, env]),
            "action_second_difference": mean_sq(second[sm, env]),
            "base_linear_acc_fd": mean_sq(base_acc[cm, env]),
            "last_qacc_cost": float(arrays["qacc_cost"][cm, env].mean()) if cm.any() else np.nan,
            "fd_acc_cost": float(np.square(fd_acc[cm, env]).sum(-1).mean()) if cm.any() else np.nan,
            "clip_fraction": float((np.abs(arrays["request"][pm, env]
                                        - arrays["clipped"][pm, env]) > 1e-5).mean()) if pm.any() else np.nan,
            "task_reward": float(arrays["reward"][cm, env].mean()) if cm.any() else np.nan,
        }
        if "physical_root_qacc" in arrays:
            row.update(root_linear_acc=mean_sq(arrays["physical_root_qacc"][pm, env, :3]),
                       root_rotational_qacc=mean_sq(arrays["physical_root_qacc"][pm, env, 3:]))
        if "constraint_torque" in arrays:
            row["constraint_torque"] = mean_sq(arrays["constraint_torque"][pm, env])
        rows.append(row)
    return {key: np.asarray([row[key] for row in rows]) for key in rows[0]}


def compare_scalars(first, second, *, bootstrap=10000, seed=20261117):
    result = {}
    for key in first:
        a, b = np.asarray(first[key]), np.asarray(second[key])
        valid = np.isfinite(a) & np.isfinite(b)
        a, b = a[valid], b[valid]
        if not len(a):
            result[key] = {"starts": 0}
            continue
        transform = np.sqrt if key in RMS else lambda x: x
        mean_a, mean_b = float(transform(a.mean())), float(transform(b.mean()))
        rng = np.random.default_rng(seed)
        indices = rng.integers(0, len(a), size=(bootstrap, len(a)))
        ba, bb = transform(a[indices].mean(1)), transform(b[indices].mean(1))
        ratios = np.full(bootstrap, np.nan)
        np.divide(bb, ba, out=ratios, where=ba != 0)
        ratios -= 1
        fully_defined = bool(np.isfinite(ratios).all())
        result[key] = {"unit": UNITS[key], "starts": len(a), "ppo": mean_a, "wmac": mean_b,
                       "relative_change": mean_b / mean_a - 1 if mean_a else None,
                       "mean_difference": mean_b - mean_a,
                       "relative_change_paired_bootstrap_95pct": (np.quantile(ratios, [.025, .975])
                                                                  if fully_defined else [np.nan, np.nan]),
                       "relative_bootstrap_defined_fraction": float(np.isfinite(ratios).mean()),
                       "difference_paired_bootstrap_95pct": np.quantile(bb - ba, [.025, .975]),
                       "starts_wmac_higher": int((b > a).sum()),
                       "starts_wmac_lower": int((b < a).sum()),
                       "per_start_ppo_raw": a, "per_start_wmac_raw": b,
                       "raw_statistic": "mean square before final root" if key in RMS else "mean"}
    return result


def pooled_description(data, mask, decimation):
    pm = np.repeat(mask, decimation, axis=0)
    result = {"physical_qacc": magnitude(np.concatenate([x["qacc"][pm] for x in data])),
            "physical_force": magnitude(np.concatenate([x["force"][pm] for x in data])),
            "startup_action_jump": magnitude(np.concatenate([x["action_delta"][0] for x in data])),
            "scope": "Pooled descriptive quantiles only; frames/repeats are not independent replicates"}
    if "physical_root_qacc" in data[0]:
        result["physical_root_linear_qacc"] = magnitude(np.concatenate([x["physical_root_qacc"][pm, :3] for x in data]))
    if "constraint_torque" in data[0]:
        result["constraint_torque"] = magnitude(np.concatenate([x["constraint_torque"][pm] for x in data]))
    return result


def joint_description(data, mask, decimation, names):
    pm = np.repeat(mask, decimation, axis=0)
    result = {}
    for j, name in enumerate(names):
        q = np.concatenate([x["qacc"][:, :, j][pm] for x in data])
        force = np.concatenate([x["force"][:, :, j][pm] for x in data])
        clip = np.concatenate([(np.abs(x["request"][:, :, j] - x["clipped"][:, :, j]) > 1e-5)[pm]
                               for x in data])
        result[name] = {"qacc": magnitude(q), "force": magnitude(force),
                        "custom_clip_fraction": float(clip.mean()) if clip.size else None}
    return result


def group_description(data, mask, decimation, names):
    pm = np.repeat(mask, decimation, axis=0)
    groups = {name: [j for j, joint in enumerate(names) if pattern in joint]
              for name, pattern in (("hips", "hip"), ("knees", "knee"), ("ankles", "ankle"),
                                    ("waist", "waist"), ("shoulders", "shoulder"),
                                    ("elbows", "elbow"), ("wrists", "wrist"))}
    result = {}
    for name, ids in groups.items():
        if not ids:
            continue
        result[name] = {field: magnitude(np.concatenate([x[field][pm][:, ids] for x in data]))
                       for field in ("qacc", "force")}
        result[name]["custom_clip_fraction"] = float(np.concatenate([
            (np.abs(x["request"][pm][:, ids] - x["clipped"][pm][:, ids]) > 1e-5) for x in data]).mean())
    return result


def reward_description(data, mask):
    if "reward_terms" not in data[0]:
        return None
    names = data[0]["reward_term_names"].tolist()
    # Equal initial-state weight, then repeated rollout weight; not equal frame
    # weight across different surviving episode lengths.
    result = {}
    for j, name in enumerate(names):
        per_start_values = [[x["reward_terms"][:, env, j][mask[:, env]].mean()
                             if mask[:, env].any() else np.nan for env in range(mask.shape[1])]
                            for x in data]
        result[name] = float(np.nanmean(per_start_values))
    return result


def analyze(directory, bootstrap):
    metrics = json.loads((directory / "metrics.json").read_text())
    data = {label: dict(np.load(directory / f"transitions_{label}.npz")) for label in TRIALS}
    protocol = metrics["protocol"]
    dt, decimation = protocol["control_dt"], protocol["physics_substeps"]
    steps = data["a"]["state"].shape[0]
    common = np.logical_and.reduce([x["valid"] & ~x["boundary"] for x in data.values()])
    time = np.arange(steps) * dt
    windows = {"all_0_to_5": time >= 0, "startup_0_to_0p1": time < .1,
               "early_0p1_to_2": (time >= .1) & (time < 2.), "late_2_to_5": time >= 2.}
    report = {"directory": str(directory), "protocol": protocol, "checkpoints": metrics["checkpoints"],
              "success": {label: {key: value for key, value in metrics["trials"][label].items()
                                   if key in ("recovery_success_count", "continued_walk_count",
                                              "first_episode_survival_count", "first_stable_recovery_seconds",
                                              "first_stable_recovery_seconds_mean")} for label in TRIALS},
              "source_sha256": {f"transitions_{label}.npz": sha256(directory / f"transitions_{label}.npz")
                                for label in TRIALS}, "windows": {}}
    raw = {}
    for label, window in windows.items():
        mask = common & window[:, None]
        per_trial = {trial: per_start(x, mask, dt, decimation, valid_history=common) for trial, x in data.items()}
        average = {policy: {key: (per_trial[policy][key] + per_trial[f"{policy}_repeat"][key]) / 2
                            for key in per_trial[policy]} for policy in ("a", "b")}
        raw[label] = average
        report["windows"][label] = {
            "common_control_steps_per_start": mask.sum(0),
            "paired_repeat_average": compare_scalars(average["a"], average["b"], bootstrap=bootstrap),
            "repeat_effects": {policy: compare_scalars(per_trial[policy], per_trial[f"{policy}_repeat"],
                                                        bootstrap=bootstrap) for policy in ("a", "b")},
            "descriptive": {policy: pooled_description([data[policy], data[f"{policy}_repeat"]], mask, decimation)
                            for policy in ("a", "b")},
            "per_joint": {policy: joint_description([data[policy], data[f"{policy}_repeat"]], mask, decimation,
                                                     metrics["joint_names"]) for policy in ("a", "b")},
            "joint_groups": {policy: group_description([data[policy], data[f"{policy}_repeat"]], mask, decimation,
                                                        metrics["joint_names"]) for policy in ("a", "b")},
            "weighted_physical_task_reward_terms_per_control_step": {
                policy: reward_description([data[policy], data[f"{policy}_repeat"]], mask) for policy in ("a", "b")},
        }
    # This tests substep-vs-control averaging using REAL states, without model error.
    for label, average in raw.items():
        report["windows"][label]["real_fd_cost_to_last_qacc_cost_ratio"] = {
            policy: float(average[policy]["fd_acc_cost"].mean() / average[policy]["last_qacc_cost"].mean())
            for policy in ("a", "b")}
    initial = dict(np.load(directory / "initial_state.npz"))
    initial_velocity_rms = np.sqrt(np.square(initial["state"][:, 42:71].astype(np.float64)).mean(-1))
    order = np.argsort(initial_velocity_rms)
    report["initial_velocity_robustness"] = {
        "initial_joint_velocity_rms_per_start_rad_s": initial_velocity_rms,
        "groups": {name: {
            "start_ids": ids,
            "initial_joint_velocity_rms_mean": float(initial_velocity_rms[ids].mean()),
            "early_qacc_rms_ppo": float(np.sqrt(raw["early_0p1_to_2"]["a"]["qacc"][ids].mean())),
            "early_qacc_rms_wmac": float(np.sqrt(raw["early_0p1_to_2"]["b"]["qacc"][ids].mean())),
            "early_action_second_difference_rms_ppo": float(np.sqrt(raw["early_0p1_to_2"]["a"]["action_second_difference"][ids].mean())),
            "early_action_second_difference_rms_wmac": float(np.sqrt(raw["early_0p1_to_2"]["b"]["action_second_difference"][ids].mean())),
        } for name, ids in (("lower_half_initial_velocity", order[:len(order)//2]),
                            ("upper_half_initial_velocity", order[len(order)//2:]))},
        "interpretation": "Descriptive post-hoc strata; not a randomized intervention or independent training replication",
    }
    return report, raw, data


def figures(reports, raw_data, prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    recovery = [key for key in reports if reports[key]["protocol"]["mode"] == "recovery"]
    if not recovery:
        return
    key = recovery[0]
    data = raw_data[key]
    dt, decimation = reports[key]["protocol"]["control_dt"], reports[key]["protocol"]["physics_substeps"]
    t = (np.arange(len(data["a"]["state"])) + 1) * dt
    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    specs = (("qacc", "Joint acceleration RMS (rad/s²)", True),
             ("force", "Actual actuator torque RMS (Nm)", True),
             ("action_delta", "Action change RMS (action units)", False),
             ("fd_acc", "Real 20 ms velocity-FD acceleration RMS (rad/s²)", False))
    for ax, (field, title, physics) in zip(axes.flat, specs):
        for policy, color, label in (("a", "#3a68a8", "PPO 60000"), ("b", "#b95036", "Current WMAC")):
            series = []
            for trial in (policy, f"{policy}_repeat"):
                x = data[trial]
                values = ((x["next_state"][..., 42:71] - x["state"][..., 42:71]) / dt
                          if field == "fd_acc" else x[field])
                if physics:
                    values = values.reshape(len(t), decimation, values.shape[1], values.shape[2])
                axes_to_reduce = (1, 3) if physics else (2,)
                energy = np.square(values.astype(np.float64)).mean(axes_to_reduce)
                energy[~(x["valid"] & ~x["boundary"])] = np.nan
                series.append(energy)
            line = np.sqrt(np.nanmean(np.stack(series), (0, 2)))
            ax.plot(t, line, color=color, label=label, lw=1.2)
        ax.axvline(.1, color="grey", ls=":", lw=.7)
        ax.axvline(2., color="grey", ls=":", lw=.7)
        ax.set(title=title, xlabel="Time (s)", xlim=(0, 5))
        ax.grid(alpha=.2)
    axes[0, 0].legend()
    fig.suptitle(f"Same physical starting snapshots; {key}; both policy repeats\n"
                 "5 ms physics displayed as 20 ms RMS; frozen trained policies under common evaluation conditions")
    fig.savefig(str(prefix) + "_time_profiles.png", dpi=180)
    fig.savefig(str(prefix) + "_time_profiles.svg")
    plt.close(fig)
    groups = reports[key]["windows"]["early_0p1_to_2"]["joint_groups"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, field, title in zip(axes, ("qacc", "force"), ("Joint acceleration RMS", "Actual torque RMS")):
        labels = list(groups["a"])
        positions = np.arange(len(labels))
        a = [groups["a"][name][field]["rms"] for name in labels]
        b = [groups["b"][name][field]["rms"] for name in labels]
        ax.bar(positions - .18, a, width=.36, color="#3a68a8", label="PPO 60000")
        ax.bar(positions + .18, b, width=.36, color="#b95036", label="Current WMAC")
        ax.set_xticks(positions, labels, rotation=35)
        ax.set(title=title, ylabel="rad/s²" if field == "qacc" else "Nm")
        ax.grid(axis="y", alpha=.2)
    axes[0].legend()
    fig.suptitle(f"Physical get-up effort by joint group, 0.1–2.0 s, {key}")
    fig.savefig(str(prefix) + "_joint_groups.png", dpi=180)
    fig.savefig(str(prefix) + "_joint_groups.svg")
    plt.close(fig)


def markdown(report, path):
    lines = ["# 当前居中 AMP WMAC 与旧 PPO：起身力度物理对照", "", "## 对照与统计口径", "",
             "固定完整物理初态和物理参数，A 为旧 PPO 60000，B 为当前 WMAC；每个策略回放两次。"
             "主表先对同一初态的两次测量平均，再对初态等权汇总。RMS 先平均平方量再开平方。"
             "bootstrap 仅在初态层配对抽样，不把帧或重复回放当独立样本；它不代表跨训练种子的可靠性。", "",
             "主掩码为四次回放共同首次 episode 有效且非边界的帧。固定 0.1–2 秒窗口与 2–5 秒窗口独立报告。"
             "控制周期 20 ms，物理子步 5 ms。物理 jerk 是 qacc 的 5 ms 数值差分，含接触和求解器高频成分，"
             "不能单凭它宣称硬件冲击或损伤。动作一阶/二阶差分是动作空间量，不是物理 jerk。", ""]
    selected = ("qacc", "force", "physical_jerk", "torque_rate", "action_delta",
                "action_second_difference", "last_qacc_cost", "fd_acc_cost", "clip_fraction",
                "root_linear_acc", "constraint_torque")
    lines += ["## 两批恢复初态合并：0.1–2.0 秒", "", "每批32个初态，合计64个初态；每个策略各有两次回放。",
              "", "| 指标 | PPO | WMAC | 相对变化 | 初态配对95%区间 |", "| --- | ---: | ---: | ---: | --- |"]
    for name in selected:
        row = report["pooled_recovery"]["early_0p1_to_2"].get(name, {})
        if not row.get("starts"):
            continue
        lo, hi = row["relative_change_paired_bootstrap_95pct"]
        change = f"{row['relative_change']:+.1%}" if row['relative_change'] is not None else "基线为零"
        lines.append(f"| {name} ({row['unit']}) | {row['ppo']:.5g} | {row['wmac']:.5g} | {change} | {lo:+.1%} 至 {hi:+.1%} |")
    lines += ["", "constraint_torque 为所有约束的广义关节力矩，包含接触与关节约束，不是接触力传感器读数。"
              "根平动加速度是物理子步 free-joint qacc 前三维的逐分量 RMS；根转动 qacc 为广义坐标量。", ""]
    for scenario, result in report["scenarios"].items():
        lines += [f"## {scenario}", "", f"源目录：`{result['directory']}`。", ""]
        for window in ("early_0p1_to_2", "late_2_to_5"):
            data = result["windows"][window]
            lines += [f"### {window}", "", "| 指标 | PPO | WMAC | 相对变化 | 初态配对95%区间 |", "| --- | ---: | ---: | ---: | --- |"]
            for name in selected:
                row = data["paired_repeat_average"][name]
                if not row.get("starts"):
                    continue
                lo, hi = row["relative_change_paired_bootstrap_95pct"]
                change = f"{row['relative_change']:+.1%}" if row['relative_change'] is not None else "基线为零"
                lines.append(f"| {name} ({row['unit']}) | {row['ppo']:.5g} | {row['wmac']:.5g} | "
                             f"{change} | {lo:+.1%} 至 {hi:+.1%} |")
            lines += ["", "真实 20 ms FD 成本 / 末物理子步瞬时 qacc 成本："
                      + "; ".join(f"{policy}={ratio:.4f}" for policy, ratio in
                                  data["real_fd_cost_to_last_qacc_cost_ratio"].items()) + "。", ""]
            terms = data["weighted_physical_task_reward_terms_per_control_step"]
            if terms["a"] is not None:
                lines += ["物理任务奖励分解（已乘各项权重和 dt，尚未乘 WMAC 外层 task_weight=0.75）：", "",
                          "| 项 | PPO | WMAC | B−A |", "| --- | ---: | ---: | ---: |"]
                for term in terms["a"]:
                    a, b = terms["a"][term], terms["b"][term]
                    lines.append(f"| {term} | {a:.6g} | {b:.6g} | {b-a:+.6g} |")
                lines += ["", "腰 pitch 的自定义电机裁剪率："
                          f"PPO={data['per_joint']['a']['waist_pitch_joint']['custom_clip_fraction']:.3%}，"
                          f"WMAC={data['per_joint']['b']['waist_pitch_joint']['custom_clip_fraction']:.3%}。", ""]
        lines += ["### 起身与续走", "", "| 回放 | 稳定起身 | 续走 | 存活至末尾 | 成功者平均确认时刻(s) |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for trial, values in result["success"].items():
            lines.append(f"| {trial} | {values['recovery_success_count']} | {values['continued_walk_count']} | "
                         f"{values['first_episode_survival_count']} | {values['first_stable_recovery_seconds_mean']} |")
        lines += [""]
    lines += ["## 结论边界", "", "这些对照能确认当前物理行为差异和奖励成本的采样口径差异，不能单独判定"
              "训练机制的唯一原因。真实状态的 FD 已会损失子步高频加速度；世界模型误差是额外问题。"
              "同初态 A/B 接触轨迹仍可能数值分岔，应同时查看 JSON 的 repeat_effects。", ""]
    path.write_text("\n".join(lines))


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, default=ROOT / "logs/impact_20261117")
    parser.add_argument("--output-prefix", type=Path, default=ROOT / "docs/validation/impact_20261117_physics")
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    scenarios, raw_stats, raw_data = {}, {}, {}
    for directory in sorted(args.root.iterdir()):
        if directory.is_dir() and (directory / "metrics.json").is_file():
            scenarios[directory.name], raw_stats[directory.name], raw_data[directory.name] = analyze(directory, args.bootstrap)
    if not scenarios:
        raise SystemExit("No completed physical paired datasets found")
    report = {"schema": "centered-amp-paired-impact-v1", "bootstrap_resamples": args.bootstrap,
              "physical_rollouts_generated_by_this_script": 0, "optimizer_updates": 0,
              "scope": "Fixed checkpoints, not matched training budget or causal intervention",
              "scenarios": scenarios, "pooled_recovery": {}}
    recovery = [name for name, item in scenarios.items() if item["protocol"]["mode"] == "recovery"]
    for window in ("early_0p1_to_2", "late_2_to_5", "all_0_to_5"):
        if recovery:
            pooled = {policy: {metric: np.concatenate([raw_stats[name][window][policy][metric] for name in recovery])
                               for metric in raw_stats[recovery[0]][window][policy]} for policy in ("a", "b")}
            report["pooled_recovery"][window] = compare_scalars(pooled["a"], pooled["b"], bootstrap=args.bootstrap)
    report["pooled_recovery_descriptive"] = {}
    for window, low, high in (("early_0p1_to_2", .1, 2.), ("late_2_to_5", 2., 5.), ("all_0_to_5", 0., 5.)):
        selected = {policy: {field: [] for field in ("qacc", "force")} for policy in ("a", "b")}
        for name in recovery:
            decimation = scenarios[name]["protocol"]["physics_substeps"]
            dt = scenarios[name]["protocol"]["control_dt"]
            trials = raw_data[name]
            common = np.logical_and.reduce([x["valid"] & ~x["boundary"] for x in trials.values()])
            t = np.arange(common.shape[0]) * dt
            pm = np.repeat(common & ((t >= low) & (t < high))[:, None], decimation, axis=0)
            for policy in ("a", "b"):
                for field in ("qacc", "force"):
                    selected[policy][field].extend([trials[trial][field][pm] for trial in (policy, policy + "_repeat")])
        if recovery:
            report["pooled_recovery_descriptive"][window] = {
                policy: {field: magnitude(np.concatenate(values)) for field, values in fields.items()}
                for policy, fields in selected.items()}
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    args.output_prefix.with_suffix(".json").write_text(json.dumps(clean(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    markdown(report, args.output_prefix.with_suffix(".md"))
    figures(scenarios, raw_data, args.output_prefix)
    print(json.dumps({"scenarios": list(scenarios), "json": str(args.output_prefix.with_suffix('.json')),
                      "markdown": str(args.output_prefix.with_suffix('.md'))}, ensure_ascii=False))


if __name__ == "__main__":
    main()
