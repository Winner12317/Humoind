"""CPU-only analysis of paired, matched-budget smooth-coefficient pilots.

No simulator, network construction, training runtime, fitting, or GPU operation.
Checkpoint reads use CPU mmap for provenance/configuration only. Adoption is a
human decision; this script reports measurements and never marks a pilot passed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from analyze_centered_recovery_impact import per_start, RMS, UNITS as BASE_UNITS


ROOT = Path(__file__).resolve().parents[1]
TRIALS = ("a", "a_repeat", "b", "b_repeat")
SCENARIOS = ("recovery_seed2026", "recovery_seed2027", "walk_seed2026")
COEFFICIENTS = ("smooth_action_second_coef", "smooth_joint_velocity_coef", "smooth_torque_coef", "smooth_power_coef")
BUDGET_KEYS = (
    "seed", "num_envs", "actor_batch_size", "actor_microbatch_size", "actor_updates",
    "wm_updates", "wm_batch_size", "wm_window", "imagination_horizon", "critic_epochs",
    "critic_minibatches", "discriminator_updates", "discriminator_batch_size", "utd",
    "collect_steps", "warmup_steps", "model_warmup_iterations", "critic_warmup_iterations",
    "precision", "compile", "optimizer_algorithm", "actor_lr", "world_model_lr", "critic_lr",
    "gamma", "gae_lambda", "model_hidden", "policy_hidden", "critic_hidden", "discriminator_hidden",
    "reference_alignment", "amp_reward_mapping", "amp_score_scale", "amp_score_ema_decay",
    "task_weight", "style_weight", "action_limit", "real_critic_coef", "mirror_walk_motions",
    "amp_feature_schema", "recovery_counter_semantics", "recovery_height_reward",
)
UNITS = {**BASE_UNITS, "root_upward_vz_mean": "m/s", "root_signed_vz_mean": "m/s",
         "root_lift_fd_mean": "m/s", "tilt_reduction_rate_mean": "rad/s",
         "root_angular_speed_mean": "rad/s", "xy_tracking_error_mean": "m/s",
         "yaw_tracking_error_mean": "rad/s", "root_height_mean": "m",
         "pose_tilt_mean": "rad", "confirmed_recovery_time": "s"}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
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
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def paired_stats(first, second, *, bootstrap, seed):
    """Input is one raw mean/square per start; repeats are already averaged."""
    result = {}
    for metric in first:
        a, b = (np.asarray(x[metric], dtype=np.float64) for x in (first, second))
        valid = np.isfinite(a) & np.isfinite(b)
        ids = np.flatnonzero(valid)
        a, b = a[valid], b[valid]
        if not len(a):
            result[metric] = {"starts": 0, "unit": UNITS[metric]}
            continue
        if metric in RMS and (np.any(a < 0) or np.any(b < 0)):
            raise ValueError("RMS input must be a nonnegative mean square")
        transform = np.sqrt if metric in RMS else lambda x: x
        ma, mb = float(transform(a.mean())), float(transform(b.mean()))
        rng = np.random.default_rng(seed)
        ix = rng.integers(len(a), size=(bootstrap, len(a)))
        ba, bb = transform(a[ix].mean(1)), transform(b[ix].mean(1))
        ratio = np.full(bootstrap, np.nan)
        np.divide(bb, ba, out=ratio, where=ba > 0)
        ratio -= 1
        result[metric] = {
            "unit": UNITS[metric], "starts": len(a), "usable_start_indices": ids,
            "a": ma, "b": mb, "B_minus_A": mb - ma,
            "relative_change": mb / ma - 1 if ma > 0 else None,
            "difference_paired_bootstrap_95pct": np.quantile(bb - ba, [.025, .975]),
            "relative_change_paired_bootstrap_95pct": np.quantile(ratio, [.025, .975]) if np.isfinite(ratio).all() else None,
            "starts_B_higher": int((b > a).sum()), "starts_B_lower": int((b < a).sum()),
            "per_start_a_raw": a, "per_start_b_raw": b,
            "raw_statistic": "mean square before final root" if metric in RMS else "mean",
        }
    return result


def mean_per_start(values, mask):
    values = np.asarray(values, dtype=np.float64)
    count = mask.sum(0)
    return np.sum(np.where(mask, values, 0), axis=0) / np.where(count, count, np.nan)


def up_z(state):
    quat = state[..., 3:7].astype(np.float64)
    quat /= np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-12)
    return (1 - 2 * (quat[..., 1] ** 2 + quat[..., 2] ** 2)).clip(-1, 1)


def physical_start_stats(row, mask, common, dt, decimation):
    values = per_start(row, mask, dt, decimation, valid_history=common)
    before, after = row["state"], row["next_state"]
    extra = {
        "root_upward_vz_mean": np.maximum(after[..., 9], 0),
        "root_signed_vz_mean": after[..., 9],
        "root_lift_fd_mean": (after[..., 2] - before[..., 2]) / dt,
        "tilt_reduction_rate_mean": (np.arccos(up_z(before)) - np.arccos(up_z(after))) / dt,
        "root_angular_speed_mean": np.linalg.norm(after[..., 10:13], axis=-1),
        "xy_tracking_error_mean": row["xy_error"], "yaw_tracking_error_mean": row["yaw_error"],
        "root_height_mean": after[..., 2], "pose_tilt_mean": np.arccos(up_z(after)),
    }
    values.update({key: mean_per_start(value, mask) for key, value in extra.items()})
    return values


def checkpoint_metadata(path):
    import torch  # CPU storage reads only; no CUDA availability probe.
    torch.set_num_threads(1)
    payload = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    if payload.get("format") != "amp-wmac-v1":
        raise ValueError("Both pilot policies must be WMAC checkpoints")
    cfg = payload["config"]
    if cfg.get("smooth_reward_mode") != "host_humanup_v1":
        raise ValueError("Expected the same active smooth formula in both arms")
    source = Path(cfg["resume"]).expanduser().resolve() if cfg.get("resume") else None
    if source is None or not source.is_file():
        raise ValueError("Pilot checkpoint must identify its accessible common resume source")
    source_payload = torch.load(source, map_location="cpu", weights_only=False, mmap=True)
    source_iteration = int(source_payload["iteration"])
    source_steps = int(source_payload["sim_steps"])
    result = {
        "path": str(Path(path).resolve()), "sha256": sha256(path),
        "implementation_version": payload["implementation_version"], "iteration": int(payload["iteration"]),
        "config": cfg, "smooth_reward_protocol": payload.get("smooth_reward_protocol"),
        "motor_metadata_sha256": payload.get("smooth_reward_model", {}).get("sha256"),
        "source_checkpoint": str(source), "source_sha256": sha256(source),
        "source_iteration": source_iteration, "additional_updates": int(payload["iteration"]) - source_iteration,
        "additional_physical_transitions": int(payload["sim_steps"]) - source_steps,
        "source_num_envs": source_payload["config"]["num_envs"],
        "source_actor_batch_size": source_payload["config"]["actor_batch_size"],
        "source_budget": {key: source_payload["config"].get(key) for key in BUDGET_KEYS},
        "smooth_reward_retune_history": payload.get("smooth_reward_retune_history", []),
        "training_cost": payload.get("training_cost"),
    }
    return result


def validate_pilot_budget(metadata, expected_updates):
    a, b = metadata["a"], metadata["b"]
    conflicts = {key: [a["config"].get(key), b["config"].get(key)] for key in BUDGET_KEYS
                 if a["config"].get(key) != b["config"].get(key)}
    if a["source_sha256"] != b["source_sha256"]:
        conflicts["source_sha256"] = [a["source_sha256"], b["source_sha256"]]
    if a["motor_metadata_sha256"] != b["motor_metadata_sha256"]:
        conflicts["motor_metadata_sha256"] = [a["motor_metadata_sha256"], b["motor_metadata_sha256"]]
    if a["additional_physical_transitions"] != b["additional_physical_transitions"]:
        conflicts["additional_physical_transitions"] = [a["additional_physical_transitions"], b["additional_physical_transitions"]]
    if any(x["additional_updates"] != expected_updates for x in metadata.values()):
        conflicts["additional_updates"] = [a["additional_updates"], b["additional_updates"]]
    if all(a["config"][key] == b["config"][key] for key in COEFFICIENTS):
        conflicts["coefficients"] = "Both arms use identical coefficients; this is not the requested coefficient pilot"
    if conflicts:
        raise ValueError(f"Pilot source/budget mismatch: {conflicts}")
    current, source = a["config"], a["source_budget"]
    wm_current = current["wm_updates"] * current["wm_batch_size"] * current["utd"] / current["num_envs"]
    wm_source = source["wm_updates"] * source["wm_batch_size"] * source["utd"] / source["num_envs"]
    return {"matched_source_and_budget": True, "compared_keys": list(BUDGET_KEYS),
            "expected_additional_updates": expected_updates,
            "additional_physical_transitions_each": a["additional_physical_transitions"],
            "budget_each": {key: a["config"].get(key) for key in BUDGET_KEYS},
            "weights": {p: {key: metadata[p]["config"][key] for key in COEFFICIENTS} for p in ("a", "b")},
            "nominal_wm_window_starts_per_real_transition": {"source": wm_source, "pilot": wm_current,
                "pilot_over_source": wm_current / wm_source,
                "formula": "wm_updates*wm_batch_size*utd/num_envs; mask validity and seven scored transitions are separate"}}


def analyze_scenario(directory, bootstrap, seed):
    metric_path = directory / "metrics.json"
    metrics = json.loads(metric_path.read_text())
    protocol = metrics["protocol"]
    if protocol["world_model_used_for_control"] or protocol["network_parameters_updated"]:
        raise ValueError("Physical paired recordings must use frozen policies")
    checks = metrics["snapshot_restore_checks"]
    if {x["trial"] for x in checks} != set(TRIALS) or any(
            x["initial_state_sha256"] != protocol["initial_state_sha256"] or
            x["physical_model_sha256"] != protocol["physical_model_sha256"] for x in checks):
        raise ValueError("Incomplete same-snapshot checks")
    data, sources = {}, {}
    for trial in TRIALS:
        path = directory / f"transitions_{trial}.npz"
        with np.load(path, allow_pickle=False) as archive:
            data[trial] = {key: archive[key].copy() for key in archive.files}
        sources[trial] = {"path": str(path), "sha256": sha256(path)}
    dt, decimation = protocol["control_dt"], protocol["physics_substeps"]
    common = np.logical_and.reduce([x["valid"] & ~x["boundary"] for x in data.values()])
    time = np.arange(len(common))[:, None] * dt
    time_windows = {"startup_0_to_0p1": time < .1, "early_0p1_to_2": (time >= .1) & (time < 2.),
                    "late_2_to_5": (time >= 2.) & (time < 5.), "all_0_to_5": time < 5.}
    shared_recovery = np.logical_and.reduce([x["recovery_phase"] for x in data.values()])
    shared_upright = np.logical_and.reduce([(x["state"][..., 2] > .65) & (up_z(x["state"]) > .8) for x in data.values()])
    phases = {"common_valid": np.ones_like(common), "shared_recovery": shared_recovery,
              "shared_upright": shared_upright}
    report = {"directory": str(directory), "protocol": protocol, "checkpoints": metrics["checkpoints"],
              "metrics_sha256": sha256(metric_path), "recordings": sources,
              "success_by_trial": {trial: {key: metrics["trials"][trial][key] for key in (
                  "recovery_success_count", "continued_walk_count", "first_episode_survival_count",
                  "first_stable_recovery_seconds", "first_stable_recovery_seconds_mean")} for trial in TRIALS},
              "windows": {}, "phase_exposure_seconds_per_start": {}}
    raw = {}
    for name, time_mask in time_windows.items():
        report["windows"][name], raw[name] = {}, {}
        for phase, pmask in phases.items():
            mask = common & time_mask & pmask
            trial_values = {trial: physical_start_stats(x, mask, common, dt, decimation) for trial, x in data.items()}
            average = {p: {key: (trial_values[p][key] + trial_values[p + "_repeat"][key]) * .5
                           for key in trial_values[p]} for p in ("a", "b")}
            raw[name][phase] = average
            report["windows"][name][phase] = {
                "common_control_steps_per_start": mask.sum(0), "common_control_steps": int(mask.sum()),
                "starts_with_observations": int(mask.any(0).sum()),
                "paired_repeat_average": paired_stats(average["a"], average["b"], bootstrap=bootstrap, seed=seed),
                "within_policy_repeat_differences": {
                    p: paired_stats(trial_values[p], trial_values[p + "_repeat"], bootstrap=bootstrap, seed=seed)
                    for p in ("a", "b")},
            }
        report["phase_exposure_seconds_per_start"][name] = {
            trial: {"own_recovery": ((x["valid"] & ~x["boundary"] & time_mask & x["recovery_phase"]).sum(0) * dt),
                    "own_valid": ((x["valid"] & ~x["boundary"] & time_mask).sum(0) * dt)} for trial, x in data.items()}
    times = {trial: np.asarray(report["success_by_trial"][trial]["first_stable_recovery_seconds"], dtype=float) for trial in TRIALS}
    average_times = {p: (times[p] + times[p + "_repeat"]) * .5 for p in ("a", "b")}
    report["confirmed_time_only_all_four_success"] = paired_stats(
        {"confirmed_recovery_time": average_times["a"]}, {"confirmed_recovery_time": average_times["b"]},
        bootstrap=bootstrap, seed=seed)
    return report, raw, average_times


def pool(reports, raw, times, bootstrap, seed):
    recovery = [name for name, row in reports.items() if row["protocol"]["mode"] == "recovery"]
    output = {"scenarios": recovery, "windows": {}}
    for window in raw[recovery[0]]:
        output["windows"][window] = {}
        for phase in raw[recovery[0]][window]:
            average = {p: {key: np.concatenate([raw[name][window][phase][p][key] for name in recovery])
                           for key in raw[recovery[0]][window][phase][p]} for p in ("a", "b")}
            output["windows"][window][phase] = {
                "common_control_steps": sum(reports[n]["windows"][window][phase]["common_control_steps"] for n in recovery),
                "starts_with_observations": sum(reports[n]["windows"][window][phase]["starts_with_observations"] for n in recovery),
                "paired_repeat_average": paired_stats(average["a"], average["b"], bootstrap=bootstrap, seed=seed)}
    output["confirmed_time_only_all_four_success"] = paired_stats(
        {"confirmed_recovery_time": np.concatenate([times[n]["a"] for n in recovery])},
        {"confirmed_recovery_time": np.concatenate([times[n]["b"] for n in recovery])}, bootstrap=bootstrap, seed=seed)
    return output


def write_markdown(report, path):
    labels = report["policy_labels"]
    def number(value):
        return "无数据" if value is None or not math.isfinite(value) else f"{value:.6g}"
    def table(lines, values, keys):
        lines.extend(["", f"| 指标 | A {labels['a']} | B {labels['b']} | B相对A | 初态配对95%区间 |",
                      "| --- | ---: | ---: | ---: | --- |"])
        for key in keys:
            row = values[key]
            if not row.get("starts"):
                lines.append(f"| {key} | 无共同初态 | 无共同初态 | — | — |")
                continue
            ci = row.get("relative_change_paired_bootstrap_95pct")
            change = row.get("relative_change")
            ci_text = f"{ci[0]:+.1%} 至 {ci[1]:+.1%}" if ci else "用JSON绝对差区间"
            change_text = f"{change:+.1%}" if change is not None else "见绝对差"
            lines.append(f"| {key} ({row['unit']}) | {number(row['a'])} | {number(row['b'])} | {change_text} | {ci_text} |")
    b = report["budget_validation"]
    matches_source_scale = (b["budget_each"]["num_envs"] == report["checkpoint_metadata"]["a"]["source_num_envs"]
                            and b["budget_each"]["actor_batch_size"] == report["checkpoint_metadata"]["a"]["source_actor_batch_size"])
    scale_note = ("本轮恢复原采样器/Actor总batch规模；仍是有限续训与单一训练源，不代表长期生产效果。"
                  if matches_source_scale else
                  "这是缩小采样器/Actor batch的短续训筛选，不是原4096环境/Actor batch4096的等价验证。")
    lines = ["# 平滑奖励系数：有限预算续训筛选结果", "",
             f"A={labels['a']}；B={labels['b']}。两者同源断点SHA：`{report['checkpoint_metadata']['a']['source_sha256']}`。",
             f"各增加{b['expected_additional_updates']}次更新、{b['additional_physical_transitions_each']}真实环境转移。",
             f"试验num_envs={b['budget_each']['num_envs']}、Actor batch={b['budget_each']['actor_batch_size']}、H={b['budget_each']['imagination_horizon']}；完整网络不变。",
             scale_note + "是否采用由实际结果和任务约束另行决定，脚本不自动判通过。", "",
             "两策略各重复两次。先在初态内平均，再平均重复，最后初态等权；RMS先汇总平方量再开根。bootstrap只重采样初态，不把重复/帧当独立样本。", "",
             "## 实际权重", "", "| 参数 | A | B |", "| --- | ---: | ---: |"]
    reference = report.get("source_reference")
    if reference and reference.get("matches_current_control"):
        lines[1:1] = ["", "**独立源参照警示：此A控制组相对原32000已经出现能力/力度漂移。B比A更好不充分代表相对原策略改善；采用前必须核对源策略。**",
                       f"独立对照：`{reference['path']}`（包含失败、续走与力度差异）。"]
    if report.get("adoption_decision") == "not_adopted_pending_full_budget_recheck":
        lines[1:1] = ["", "**本轮结论：不采用512环境筛选结果，等待4096环境/Actor batch4096的同源复查。**",
                       "候选虽优于本轮漂移控制组，仍存在起身后续走损失；不能冒充正式调优已成功。"]
    reuse = b["nominal_wm_window_starts_per_real_transition"]
    for key in COEFFICIENTS:
        lines.append(f"| {key} | {b['weights']['a'][key]:.8g} | {b['weights']['b'][key]:.8g} |")
    if not math.isclose(reuse["pilot_over_source"], 1.):
        lines.extend(["", f"名义WM窗口起点训练量/真实转移：源={reuse['source']:.6g}，本pilot={reuse['pilot']:.6g}，比值={reuse['pilot_over_source']:.6g}。",
                      "因此仅保留UTD数值并未保留原真实交互对应的模型训练量。该执行预算差异是明确混淆，不能仅凭本次结果认定它导致退化。"])
    lines.extend(["", "## 两恢复seed合并", ""])
    primary = ("qacc", "force", "velocity", "action_delta", "action_second_difference", "clip_fraction")
    for window in ("early_0p1_to_2", "late_2_to_5"):
        for phase in ("common_valid", "shared_recovery", "shared_upright"):
            row = report["pooled_recovery"]["windows"][window][phase]
            lines.extend(["", f"### {window} / {phase}", "",
                          f"共同帧数={row['common_control_steps']}，有观测初态数={row['starts_with_observations']}。"])
            keys = primary if phase == "common_valid" else ("root_upward_vz_mean", "root_lift_fd_mean", "tilt_reduction_rate_mean", "qacc", "force", "velocity")
            table(lines, row["paired_repeat_average"], keys)
    lines.extend(["", "### 稳定确认时间：仅双方两次都成功的初态", ""])
    table(lines, report["pooled_recovery"]["confirmed_time_only_all_four_success"], ("confirmed_recovery_time",))
    for name, scenario in report["scenarios"].items():
        lines.extend(["", f"## {name}：任务能力与暴露", "",
                      "| 回放 | 起身/直立确认 | 续走 | 首回合末尾存活 | 成功者确认均时s |",
                      "| --- | ---: | ---: | ---: | ---: |"])
        for trial, row in scenario["success_by_trial"].items():
            lines.append(f"| {labels[trial[0]]} {trial} | {row['recovery_success_count']} | {row['continued_walk_count']} | {row['first_episode_survival_count']} | {number(row['first_stable_recovery_seconds_mean'])} |")
        exposure = scenario["phase_exposure_seconds_per_start"]["early_0p1_to_2"]
        lines.append("\n各回放0.1–2秒自身恢复暴露时长均值：" + "；".join(
            f"{trial}={np.mean(row['own_recovery']):.4f}s" for trial, row in exposure.items()) + "。")
        if scenario["protocol"]["mode"] == "walk":
            for window in ("early_0p1_to_2", "late_2_to_5"):
                lines.extend(["", f"### 独立行走 {window}", ""])
                table(lines, scenario["windows"][window]["common_valid"]["paired_repeat_average"],
                      ("qacc", "force", "velocity", "xy_tracking_error_mean", "yaw_tracking_error_mean"))
    lines.extend(["", "## 限制", "",
                  "共同恢复是同一时刻四回放均标记恢复的交集，不是中间姿态匹配。更早退出恢复会改变阶段暴露；各回放自身恢复时长另列，不能仅凭全窗速度判断起身变慢。",
                  "恢复时间包含25步稳定确认；有失败时不能只看成功者均时。共同有效窗在终止后截断，不能替代成功/存活表。",
                  "qacc、实际力矩来自全部5ms物理子步；动作一/二阶差在20ms动作空间，不等于真实jerk。",
                  "物理task reward使用共同基线，不含各自训练的四成本或AMP，不用于宣称完整训练目标更优。",
                  "置信区间只覆盖本次初态差异；不同仿真回放仍可能接触分岔，应查看JSON的within_policy_repeat_differences。有限续训、单训练源与执行差异仍限制长期泛化结论。",
                  "所有原始权重、source、预算、记录哈希、逐初态量与回放暴露保存在同名JSON。没有自动采用或成功预判。"])
    path.write_text("\n".join(lines) + "\n")


def plot(report, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    measures = ("qacc", "force", "action_delta", "action_second_difference")
    row = report["pooled_recovery"]["windows"]["early_0p1_to_2"]["common_valid"]["paired_repeat_average"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.3), constrained_layout=True)
    for ax, metric in zip(axes, measures):
        ax.bar((0, 1), (row[metric]["a"], row[metric]["b"]), color=("#3a68a8", "#b95036"))
        ax.set_xticks((0, 1), (report["policy_labels"]["a"], report["policy_labels"]["b"]), rotation=15)
        ax.set_title(metric)
        ax.set_ylabel(UNITS[metric])
        ax.grid(axis="y", alpha=.2)
    fig.suptitle("Matched-budget pilot; paired physical recovery 0.1–2s; screening evidence")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--label-a", default="baseline500")
    parser.add_argument("--label-b", default="balanced_v2_500")
    parser.add_argument("--expected-updates", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261117)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--screening-decision", choices=("not_adopted_pending_full_budget_recheck",))
    args = parser.parse_args()
    if args.bootstrap < 1 or args.expected_updates < 1:
        parser.error("Positive bootstrap/expected-updates required")
    directories = {name: args.root / name for name in SCENARIOS}
    missing = [name for name, p in directories.items() if not (p / "metrics.json").is_file()]
    if missing:
        raise SystemExit(f"Required completed paired scenarios missing: {missing}")
    first = json.loads((directories[SCENARIOS[0]] / "metrics.json").read_text())
    metadata = {p: checkpoint_metadata(Path(first["checkpoints"][p]["path"])) for p in ("a", "b")}
    budget = validate_pilot_budget(metadata, args.expected_updates)
    reports, raw, times = {}, {}, {}
    for name, directory in directories.items():
        metrics = json.loads((directory / "metrics.json").read_text())
        for p in ("a", "b"):
            if metrics["checkpoints"][p]["sha256"] != metadata[p]["sha256"]:
                raise ValueError(f"Policy/checkpoint changes between scenarios: {name}/{p}")
        reports[name], raw[name], times[name] = analyze_scenario(directory, args.bootstrap, args.bootstrap_seed)
        print("Completed CPU physical analysis:", name, flush=True)
    report = {
        "schema": "matched-budget-smooth-weight-pilot-physics-v1", "policy_labels": {"a": args.label_a, "b": args.label_b},
        "adoption_decision": args.screening_decision,
        "decision_origin": "Explicit caller instruction" if args.screening_decision else None,
        "scope": "Screening evidence only; no automatic pass or coefficient adoption",
        "new_simulator_steps": 0, "network_or_optimizer_updates": 0, "gpu_operations": 0,
        "bootstrap": {"samples": args.bootstrap, "seed": args.bootstrap_seed,
                      "unit": "paired initial state, repeats averaged first", "training_seed_uncertainty_covered": False},
        "analysis_script_sha256": sha256(__file__), "shared_metric_helper_sha256": sha256(Path(__file__).with_name("analyze_centered_recovery_impact.py")),
        "checkpoint_metadata": metadata, "budget_validation": budget, "scenarios": reports,
        "pooled_recovery": pool(reports, raw, times, args.bootstrap, args.bootstrap_seed),
        "limitations": [("Sampler/Actor total batch match source; finite continuation and one training source."
                         if all(x["config"]["num_envs"] == x["source_num_envs"] and
                                x["config"]["actor_batch_size"] == x["source_actor_batch_size"]
                                for x in metadata.values()) else
                         "Sampler/Actor total batch differ from source; finite continuation and one training source."),
                        "Fixed physical evaluation command and finite initial-state suite, no long-term/general-terrain claim.",
                        "Common phase masks condition on outcomes and do not match intermediate poses.",
                        "Failure and survival counts must accompany conditional effort/time statistics."]}
    reference_path = ROOT / "docs/diagnostics/2026-11-17/pilot_control_drift.json"
    if reference_path.is_file():
        reference = json.loads(reference_path.read_text())
        report["source_reference"] = {
            "path": str(reference_path), "sha256": sha256(reference_path),
            "matches_current_control": reference["control"]["sha256"] == metadata["a"]["sha256"],
            "matches_common_source": reference["source"]["sha256"] == metadata["a"]["source_sha256"],
            "source_success_by_trial": reference["physical_analysis"]["success_by_trial"],
            "adoption_constraint": "Superiority to a drifted control alone cannot establish source-policy or production improvement.",
        }
    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    output = args.output_prefix.with_suffix(".json")
    output.write_text(json.dumps(clean(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    write_markdown(clean(report), args.output_prefix.with_suffix(".md"))
    if args.plot:
        plot(clean(report), args.output_prefix.with_suffix(".png"))
    print(output)


if __name__ == "__main__":
    main()
