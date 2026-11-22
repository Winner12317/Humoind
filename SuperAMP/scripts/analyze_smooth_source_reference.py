"""CPU physical source reference, intentionally not matched continuation budget.

A is the original source policy; B is a coefficient candidate continued from A.
This complements candidate/control pilots and never automatically adopts B.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from analyze_smooth_weight_pilots import (
    BUDGET_KEYS, COEFFICIENTS, SCENARIOS, analyze_scenario, checkpoint_metadata,
    clean, plot, pool, sha256,
)


def write_markdown(report, path):
    labels, candidate = report["policy_labels"], report["candidate_metadata"]
    lines = ["# 续训候选与原策略：独立源参照", "",
             f"A={labels['a']}；B={labels['b']}。A没有额外更新，B从同一A断点增加{candidate['additional_updates']}次更新、{candidate['additional_physical_transitions']}物理环境转移。",
             "这不是相同训练预算的因果对照，用于检查候选有没有明显损害原策略的起身、续走、行走与力度。候选胜过另一个漂移控制组不充分代表优于原策略；本脚本不自动采用权重。",
             f"来源SHA：`{candidate['source_sha256']}`；候选SHA：`{candidate['sha256']}`。", "",
             "## 配置与预算", "",
             f"候选num_envs={candidate['config']['num_envs']}、Actor总batch={candidate['config']['actor_batch_size']}、microbatch={candidate['config']['actor_microbatch_size']}，H={candidate['config']['imagination_horizon']}。",
             "所有共同物理场景中两策略各重复两次；先每初态内平均，再平均重复，最后初态等权。bootstrap只重采样初态，不把帧/重复当独立试验。", "",
             "| 参数 | 原策略A | 候选B |", "| --- | ---: | ---: |"]
    for key in COEFFICIENTS:
        lines.append(f"| {key} | {report['source_config'][key]:.8g} | {candidate['config'][key]:.8g} |")
    lines.extend(["", "## 逐场景任务能力", ""])
    for name, scenario in report["scenarios"].items():
        lines.extend([f"### {name}", "", "| 回放 | 起身/直立确认 | 续走 | 首回合末存活 | 成功者确认均时s |",
                      "| --- | ---: | ---: | ---: | ---: |"])
        for trial, value in scenario["success_by_trial"].items():
            mean = value["first_stable_recovery_seconds_mean"]
            text = f"{mean:.6g}" if mean is not None and math.isfinite(mean) else "无成功样本"
            lines.append(f"| {labels[trial[0]]} {trial} | {value['recovery_success_count']} | {value['continued_walk_count']} | {value['first_episode_survival_count']} | {text} |")
        lines.append("")
    def table(values, keys):
        lines.extend(["", f"| 指标 | A {labels['a']} | B {labels['b']} | B相对A | 初态配对95%区间 |",
                      "| --- | ---: | ---: | ---: | --- |"])
        for key in keys:
            row = values[key]
            if not row.get("starts"):
                lines.append(f"| {key} | 无共同样本 | 无共同样本 | — | — |")
                continue
            rel, interval = row.get("relative_change"), row.get("relative_change_paired_bootstrap_95pct")
            reltext = f"{rel:+.1%}" if rel is not None else "见JSON绝对差"
            citext = f"{interval[0]:+.1%} 至 {interval[1]:+.1%}" if interval else "见JSON绝对差"
            lines.append(f"| {key} ({row['unit']}) | {row['a']:.6g} | {row['b']:.6g} | {reltext} | {citext} |")
    lines.extend(["", "## 两批恢复初态合并", ""])
    for window in ("early_0p1_to_2", "late_2_to_5"):
        for phase in ("common_valid", "shared_recovery"):
            row = report["pooled_recovery"]["windows"][window][phase]
            lines.extend(["", f"### {window} / {phase}", "",
                          f"共同时间点{row['common_control_steps']}；有观测初态{row['starts_with_observations']}。"])
            keys = ("qacc", "force", "velocity", "action_delta", "action_second_difference", "clip_fraction") if phase == "common_valid" else (
                "root_upward_vz_mean", "root_lift_fd_mean", "tilt_reduction_rate_mean", "qacc", "force")
            table(row["paired_repeat_average"], keys)
    lines.extend(["", "### 双方两次都成功初态的确认时间", ""])
    table(report["pooled_recovery"]["confirmed_time_only_all_four_success"], ("confirmed_recovery_time",))
    walk = report["scenarios"]["walk_seed2026"]
    for window in ("early_0p1_to_2", "late_2_to_5"):
        lines.extend(["", f"## 独立行走 {window}", ""])
        table(walk["windows"][window]["common_valid"]["paired_repeat_average"],
              ("qacc", "force", "velocity", "xy_tracking_error_mean", "yaw_tracking_error_mean"))
    lines.extend(["", "## 限制", "",
                  "A与B更新数不同，因此不能将差异全部归因为权重；同预算baseline-versus-candidate试验另行分析。有限续训和一个训练源也不代表长期效果。",
                  "应优先保留起身、续走和存活能力。仅比较成功者确认时间或共同有效窗口会隐藏失败，必须同时读逐次任务计数。",
                  "共同恢复是四次回放同时间均处恢复的交集，不是姿态或动作相位匹配；暴露时长另存JSON。",
                  "真实qacc/实际力矩来自5ms子步，动作差来自20ms控制；原共同task不是各自完整训练目标。",
                  "CPU脚本只读已经完成的物理回放与checkpoint元数据，不执行GPU、仿真、网络或优化。当前无自动采用结论。"])
    path.write_text("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--label-a", default="source32000")
    parser.add_argument("--label-b", default="balanced32250")
    parser.add_argument("--expected-updates", type=int, default=250)
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=20261117)
    parser.add_argument("--plot", action="store_true")
    args = parser.parse_args()
    if args.bootstrap < 1 or args.expected_updates < 1:
        parser.error("Positive bootstrap/expected-updates required")
    directories = {name: args.root / name for name in SCENARIOS}
    missing = [name for name, directory in directories.items() if not (directory / "metrics.json").is_file()]
    if missing:
        raise SystemExit(f"Required completed source-reference scenarios missing: {missing}")
    first = json.loads((directories[SCENARIOS[0]] / "metrics.json").read_text())
    candidate = checkpoint_metadata(Path(first["checkpoints"]["b"]["path"]))
    if first["checkpoints"]["a"]["sha256"] != candidate["source_sha256"]:
        raise ValueError("A is not B's actual source checkpoint")
    if candidate["additional_updates"] != args.expected_updates:
        raise ValueError("Candidate continuation budget differs from requested reference")
    import torch
    payload = torch.load(candidate["source_checkpoint"], map_location="cpu", weights_only=False, mmap=True)
    source_config = payload["config"]
    reports, raw, times = {}, {}, {}
    for name, directory in directories.items():
        metrics = json.loads((directory / "metrics.json").read_text())
        if metrics["checkpoints"]["a"]["sha256"] != candidate["source_sha256"] or metrics["checkpoints"]["b"]["sha256"] != candidate["sha256"]:
            raise ValueError("Policies differ between source-reference scenarios")
        reports[name], raw[name], times[name] = analyze_scenario(directory, args.bootstrap, args.bootstrap_seed)
        print("Completed CPU source reference:", name, flush=True)
    report = {
        "schema": "smooth-weight-candidate-independent-source-reference-v1",
        "policy_labels": {"a": args.label_a, "b": args.label_b}, "adoption_decision": None,
        "purpose": "Source competence and effort reference; intentionally not matched update budget and not a coefficient-causal estimate",
        "source_additional_updates": 0, "candidate_additional_updates": candidate["additional_updates"],
        "new_simulator_steps": 0, "network_or_optimizer_updates": 0, "gpu_operations": 0,
        "candidate_metadata": candidate, "source_config": source_config,
        "configuration_differences": {key: {"source": source_config.get(key), "candidate": candidate["config"].get(key)}
                                      for key in (*BUDGET_KEYS, *COEFFICIENTS) if source_config.get(key) != candidate["config"].get(key)},
        "analysis_script_sha256": sha256(__file__), "scenarios": reports,
        "pooled_recovery": pool(reports, raw, times, args.bootstrap, args.bootstrap_seed),
        "bootstrap": {"samples": args.bootstrap, "seed": args.bootstrap_seed,
                      "unit": "paired initial state after averaging two repeats", "training_seed_uncertainty_covered": False},
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
