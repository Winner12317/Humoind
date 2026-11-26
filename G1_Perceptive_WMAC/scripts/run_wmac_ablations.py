"""Prepare reproducible ablations; execute serially only with --execute."""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict
from datetime import datetime, timezone
from fractions import Fraction
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.wmac.config import PRESETS, PROFILES, WmacConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    parser.add_argument("--presets", nargs="+", choices=PRESETS, default=list(PRESETS))
    parser.add_argument("--profile", choices=PROFILES, default="balanced")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--collect-steps", type=int)
    parser.add_argument("--utd", type=float)
    parser.add_argument("--warmup-steps", type=int)
    parser.add_argument("--model-warmup-iterations", type=int)
    parser.add_argument("--imagination-horizon", type=int)
    parser.add_argument("--wm-window", type=int)
    parser.add_argument("--baseline-steps-per-env", type=int, default=24)
    parser.add_argument("--strict-baseline-budget", action="store_true",
                        help="Require the WMAC interaction budget to divide into full PPO rollouts.")
    parser.add_argument("--log-root", type=Path, default=REPO / "logs" / "ablations")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--include-baseline", action="store_true")
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    cfg = WmacConfig.for_profile(args.profile)
    for name in ("iterations", "num_envs", "collect_steps", "utd", "warmup_steps",
                 "model_warmup_iterations", "imagination_horizon", "wm_window"):
        if getattr(args, name) is not None:
            setattr(cfg, name, getattr(args, name))
    try:
        cfg.validate()
    except ValueError as error:
        parser.error(str(error))
    if args.baseline_steps_per_env < 1:
        parser.error("--baseline-steps-per-env must be positive")
    if cfg.collect_steps:
        iteration_steps = cfg.iterations * cfg.collect_steps
    else:
        # Rational arithmetic keeps the planning budget independent of float
        # accumulation at rates such as 0.3 updates per vectorized control step.
        total = Fraction(cfg.iterations, 1) / Fraction(str(cfg.utd))
        iteration_steps = total.numerator // total.denominator
    vector_steps = cfg.warmup_steps + iteration_steps
    baseline_iters = (vector_steps + args.baseline_steps_per_env - 1) // args.baseline_steps_per_env
    baseline_vector_steps = baseline_iters * args.baseline_steps_per_env
    extra_vector_steps = baseline_vector_steps - vector_steps
    if args.include_baseline and args.strict_baseline_budget and extra_vector_steps:
        parser.error(
            f"WMAC budget is {vector_steps} vectorized steps, which is not divisible by "
            f"PPO rollout length {args.baseline_steps_per_env}. Adjust the total iteration "
            "budget while retaining WMAC's sampling cadence, or omit --strict-baseline-budget "
            f"to allow PPO {extra_vector_steps} extra vectorized steps (less than one rollout). "
            "The launcher preserves the WMAC UTD and PPO rollout length."
        )

    run_dir = args.log_root.expanduser().resolve() / f"{datetime.now():%Y-%m-%d_%H-%M-%S_%f}"
    run_dir.mkdir(parents=True)
    manifest_path = run_dir / "manifest.json"
    source = subprocess.run(
        ["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True, check=False,
    ).stdout.strip()
    transitions = cfg.num_envs * vector_steps
    baseline_transitions = cfg.num_envs * baseline_vector_steps
    extra_transitions = cfg.num_envs * extra_vector_steps
    jobs = []
    for seed in args.seeds:
        if args.include_baseline:
            jobs.append({
                "preset": "ppo_amp_baseline", "seed": seed,
                "physical_transitions": baseline_transitions,
                "target_physical_transitions": transitions,
                "extra_physical_transitions": extra_transitions,
                "resolved_config": {"task": cfg.task, "seed": seed, "num_envs": cfg.num_envs,
                    "max_iterations": baseline_iters, "num_steps_per_env": args.baseline_steps_per_env},
                "command": [
                    args.python, str(REPO / "scripts" / "train.py"), cfg.task,
                    "--env.scene.num-envs", str(cfg.num_envs), "--agent.seed", str(seed),
                    "--agent.num-steps-per-env", str(args.baseline_steps_per_env),
                    "--agent.max-iterations", str(baseline_iters), "--agent.logger", "tensorboard",
                    "--agent.run-name", f"ablation_ppo_seed{seed}_{run_dir.name}",
                ], "status": "planned",
            })
        for preset in args.presets:
            resolved = copy.deepcopy(cfg)
            resolved.seed, resolved.preset = seed, preset
            resolved.log_root, resolved.run_name = str(run_dir / "wmac"), "ablation"
            resolved.apply_preset()
            resolved.validate()
            jobs.append({
                "preset": preset, "seed": seed, "physical_transitions": transitions,
                "extra_physical_transitions": 0,
                "resolved_config": asdict(resolved),
                "command": [
                    args.python, str(REPO / "scripts" / "train_wmac.py"),
                    "--profile", cfg.profile, "--preset", preset, "--seed", str(seed),
                    "--iterations", str(cfg.iterations), "--num-envs", str(cfg.num_envs),
                    "--collect-steps", str(cfg.collect_steps), "--utd", str(cfg.utd),
                    "--warmup-steps", str(cfg.warmup_steps),
                    "--model-warmup-iterations", str(cfg.model_warmup_iterations),
                    "--imagination-horizon", str(cfg.imagination_horizon),
                    "--wm-window", str(cfg.wm_window),
                    "--log-root", str(run_dir / "wmac"), "--run-name", "ablation",
                ], "status": "planned",
            })
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(), "source_directory": str(REPO),
        "source_commit": source, "execute": args.execute,
        "profile": cfg.profile, "resolved_base_config": asdict(cfg),
        "vectorized_steps_per_run": vector_steps,
        "physical_transitions_per_run": transitions,
        "wmac_vectorized_steps_per_run": vector_steps,
        "wmac_physical_transitions_per_run": transitions,
        "ppo_vectorized_steps_per_run": baseline_vector_steps if args.include_baseline else None,
        "ppo_physical_transitions_per_run": baseline_transitions if args.include_baseline else None,
        "ppo_extra_vectorized_steps": extra_vector_steps if args.include_baseline else None,
        "ppo_extra_physical_transitions": extra_transitions if args.include_baseline else None,
        "strict_baseline_budget": args.strict_baseline_budget,
        "budget_note": (
            "WMAC includes initial warmup; collect_steps=0 uses floor(iterations/utd). "
            "The original budget fields describe the WMAC target. PPO retains its explicit "
            "rollout length and rounds its iterations upward to full rollouts; its extra "
            "physical interactions are reported separately and are less than one rollout. "
            "No WMAC collection or UTD parameter is changed to match PPO."
        ),
        "jobs": jobs,
    }

    def save_manifest() -> None:
        temporary = manifest_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2) + "\n")
        temporary.replace(manifest_path)

    save_manifest()
    print(f"Manifest: {manifest_path}")
    print(f"Runs: {len(jobs)}; WMAC physical simulator transitions per run: {transitions:,}")
    if args.include_baseline:
        print(f"PPO physical simulator transitions per run: {baseline_transitions:,}; "
              f"extra: {extra_transitions:,} ({extra_vector_steps} vectorized steps, "
              f"rollout length {args.baseline_steps_per_env}).")
    for job in jobs:
        print(shlex.join(job["command"]), flush=True)
    if not args.execute:
        print("Dry run complete. Add --execute to run these experiments serially.")
        return

    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("OMP_NUM_THREADS", "1")
    for index, job in enumerate(jobs):
        job["status"] = "running"
        job["started_at"] = datetime.now(timezone.utc).isoformat()
        job["console_log"] = str(run_dir / f"{index:03d}_{job['preset']}_seed{job['seed']}.log")
        save_manifest()
        print(f"[{index + 1}/{len(jobs)}] {job['preset']} seed={job['seed']}", flush=True)
        try:
            with open(job["console_log"], "w") as log:
                result = subprocess.run(job["command"], cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
            job["return_code"] = result.returncode
            job["status"] = "completed" if result.returncode == 0 else "failed"
        except BaseException:
            job["status"] = "interrupted"
            raise
        finally:
            job["ended_at"] = datetime.now(timezone.utc).isoformat()
            save_manifest()
        if result.returncode:
            raise SystemExit(f"Experiment failed; stopped sweep. See {job['console_log']}")


if __name__ == "__main__":
    main()
