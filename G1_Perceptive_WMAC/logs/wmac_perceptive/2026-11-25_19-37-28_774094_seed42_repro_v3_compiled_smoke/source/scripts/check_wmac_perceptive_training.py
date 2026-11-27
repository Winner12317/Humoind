"""Review retained policy updates and first-episode physical performance.

This creates a new review directory. It never resumes training or edits the
checkpoint, metrics CSV, or earlier evaluation records.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--baseline-checkpoint", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--seed", type=int, default=2032)
    parser.add_argument("--window", type=int, default=200)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if min(args.num_envs, args.steps, args.window) < 1 or args.seed < 0:
        parser.error("num-envs, steps and window must be positive; seed must be nonnegative")
    for name in ("checkpoint", "baseline_checkpoint"):
        path = getattr(args, name)
        if path is not None:
            path = path.expanduser().resolve()
            if not path.is_file(): parser.error(f"{name.replace('_', '-')} does not exist: {path}")
            setattr(args, name, path)
    return args


def _number(value, default=0.):
    try:
        result = float(value)
        return result if math.isfinite(result) else default
    except (TypeError, ValueError):
        return default


def training_summary(checkpoint, window=200):
    import torch
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if payload.get("format") != "amp-wmac-perceptive-v1":
        raise ValueError("Review requires a perceptive checkpoint")
    config, progress = payload.get("config", {}), payload.get("perception_training_progress", {})
    limit = int(payload["iteration"])
    recent, horizons, rows = deque(maxlen=window), Counter(), 0
    totals, previous = [0, 0], [None, None]
    csv_path = Path(checkpoint).parent/"metrics.csv"
    if csv_path.is_file():
        with csv_path.open(newline="") as stream:
            for row in csv.DictReader(stream):
                if _number(row.get("iteration"), float("inf")) > limit: continue
                active = _number(row.get("actor_active")) > 0
                rejected = _number(row.get("actor_update_rejected"))
                updates = config.get("actor_updates", 1)
                deltas = [round(updates*(1-rejected)) if active else 0,
                          round(updates*rejected) if active else 0]
                for index, key in enumerate(("actor_updates_accepted", "actor_updates_rejected")):
                    current = _number(row.get(key), None)
                    if current is not None and previous[index] is not None:
                        deltas[index] = max(0, int(current-previous[index]))
                    previous[index] = current
                    totals[index] += deltas[index]
                horizon = row.get("active_imagination_horizon", str(config.get("imagination_horizon", "unknown")))
                horizons[str(horizon)] += 1
                recent.append((deltas, horizon, row)); rows += 1
    last = recent[-1][2] if recent else {}
    origin = progress.get("_last_actor_update_iteration")
    if origin is None: origin = progress.get("_progress_start_iteration", limit)
    stall = int(_number(last.get("actor_update_stall_iterations"), max(0, limit-int(origin))))
    recent_accepted = sum(item[0][0] for item in recent)
    status = ("stalled" if stall >= config.get("actor_stall_warning_iterations", 200)
              else "updating_recently" if recent_accepted else "no_recent_accepted_updates" if rows else "insufficient_metrics")
    optimizer = payload.get("opt_actor", {}).get("state", {})
    steps = sorted({_number(state["step"]) for state in optimizer.values() if "step" in state})
    return {"checkpoint_iteration": limit, "perception_training_version": config.get("perception_training_version", 1),
            "status": status, "csv": str(csv_path), "csv_rows_to_checkpoint": rows,
            "protocol_updates_accepted": progress.get("actor_updates_accepted", previous[0]),
            "protocol_updates_rejected": progress.get("actor_updates_rejected", previous[1]),
            "csv_observed_updates_accepted": totals[0], "csv_observed_updates_rejected": totals[1],
            "recent_rows": len(recent), "recent_updates_accepted": recent_accepted,
            "recent_updates_rejected": sum(item[0][1] for item in recent),
            "actor_update_stall_iterations": stall, "horizon_distribution": dict(horizons),
            "recent_gate_reasons": dict(Counter(item[2].get("gate_reason", "unrecorded") for item in recent)),
            "actor_optimizer_parameter_states": len(optimizer), "actor_optimizer_steps": steps,
            "counter_note": "Protocol counters can span resumed CSV files; Adam steps can include pre-migration updates."}


def compare_evaluations(candidate, baseline):
    """Compare primary physical outcomes only for recorded identical starts."""
    if candidate.get("status") != "completed" or baseline.get("status") != "completed":
        return {"comparable": False, "reason": "evaluation_incomplete"}
    if candidate.get("protocol") != baseline.get("protocol"):
        return {"comparable": False, "reason": "evaluation_protocol_differs"}
    modes=tuple(candidate.get("results",{}))
    if set(modes)!=set(baseline.get("results",{})) or "walk" not in modes:
        return {"comparable":False,"reason":"motion_domains_differ"}
    for mode in modes:
        first = candidate.get("results", {}).get(mode, {}).get("initial_state_sha256")
        other = baseline.get("results", {}).get(mode, {}).get("initial_state_sha256")
        if first is None or first != other:
            return {"comparable": False, "reason": f"{mode}_initial_states_differ_or_missing"}
    measures = [("walk_survival", "walk", "first_episode_survival_fraction")]
    if "recovery" in modes:
        measures.extend((("recovery_success", "recovery", "recovery_success_fraction"),
                         ("recovery_then_walk", "recovery", "success_then_walk_fraction_all_starts")))
    deltas = {name: candidate["results"][mode][key]-baseline["results"][mode][key]
              for name, mode, key in measures}
    positive, negative = any(x > 0 for x in deltas.values()), any(x < 0 for x in deltas.values())
    verdict = ("mixed_outcomes" if positive and negative else "observed_improvement" if positive
               else "observed_regression" if negative else "no_observed_primary_improvement")
    return {"comparable": True, "verdict": verdict, "primary_fraction_deltas": deltas,
            "scope": "One fixed-seed evaluation; no statistical significance or convergence claim."}


def _evaluate(checkpoint, args, output):
    command = [sys.executable, str(ROOT/"scripts/evaluate_wmac_perceptive.py"),
               "--checkpoint", str(checkpoint), "--mode", "auto", "--device", args.device,
               "--num-envs", str(args.num_envs), "--steps", str(args.steps),
               "--seed", str(args.seed), "--output", str(output)]
    with output.with_suffix(".log").open("w") as log:
        subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads((output/"metrics.json").read_text())


def main(argv=None):
    args = parse_args(argv)
    output = (args.output.expanduser().resolve() if args.output else
              args.checkpoint.parent/"training_reviews"/f"review_{datetime.now():%Y-%m-%d_%H-%M-%S_%f}")
    output.mkdir(parents=True, exist_ok=False)
    review = {"format": "amp-wmac-perceptive-training-review-v1", "status": "running",
              "checkpoint": str(args.checkpoint), "convergence_established": False,
              "scope": "Retained updates and physical first episodes; short tests do not establish convergence."}
    code = 0
    try:
        review["training"] = training_summary(args.checkpoint, args.window)
        print(f"Updates: {review['training']['status']}; recent accepted={review['training']['recent_updates_accepted']}", flush=True)
        review["candidate_evaluation"] = _evaluate(args.checkpoint, args, output/"candidate")
        if args.baseline_checkpoint:
            review["baseline_checkpoint"] = str(args.baseline_checkpoint)
            review["baseline_evaluation"] = _evaluate(args.baseline_checkpoint, args, output/"baseline")
            review["comparison"] = compare_evaluations(review["candidate_evaluation"], review["baseline_evaluation"])
        else:
            review["comparison"] = {"comparable": False, "reason": "baseline_not_supplied"}
        review["status"] = "completed"
    except Exception as error:
        review.update(status="failed", error={"type": type(error).__name__, "message": str(error)})
        code = 1
    (output/"review.json").write_text(json.dumps(review, ensure_ascii=False, indent=2, allow_nan=False)+"\n")
    print(f"Review: {output/'review.json'}; convergence has not been established.")
    if review["status"] == "completed":
        for mode, result in review["candidate_evaluation"]["results"].items():
            print(f"{mode}: survival={result['first_episode_survival_fraction']:.3f}, "
                  f"recovery={result.get('recovery_success_fraction', 'n/a')}")
        print("Comparison:", json.dumps(review["comparison"], ensure_ascii=False))
    else: print("Review failed:", review["error"]["message"], file=sys.stderr)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
