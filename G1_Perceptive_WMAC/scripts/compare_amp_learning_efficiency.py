"""Compare checkpoint resources at equal measured physical performance.

Read-only: reads evaluate_wmac.py metrics JSON, never loads policies or starts
training/evaluation. Threshold crossings describe the supplied checkpoints;
they are not evidence of stable convergence between evaluations or seeds.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from src.training_cost import WALL_TIME_SCOPE


@dataclass(frozen=True)
class Condition:
    mode: str
    metric: str
    operator: str
    threshold: float


def parse_condition(value: str, operator: str) -> Condition:
    try:
        name, raw_threshold = value.rsplit("=", 1)
        mode, metric = name.split(".", 1)
        threshold = float(raw_threshold)
    except (ValueError, TypeError):
        raise ValueError(f"Invalid threshold {value!r}; expected MODE.METRIC=value") from None
    if not mode or not metric or not math.isfinite(threshold):
        raise ValueError(f"Invalid finite threshold {value!r}; expected MODE.METRIC=value")
    return Condition(mode, metric, operator, threshold)


def discover_metrics(paths):
    result = set()
    for value in paths:
        path = Path(value).expanduser().resolve()
        if path.is_dir():
            result.update(path.rglob("metrics.json"))
        elif path.is_file():
            result.add(path)
        else:
            raise ValueError(f"Metrics input does not exist: {path}")
    if not result:
        raise ValueError("No metrics.json files were found")
    return sorted(result)


def _finite_number(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _read_record(path, conditions):
    try:
        payload = json.loads(path.read_text())
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read evaluation {path}: {error}") from error
    if not isinstance(payload, dict) or not isinstance(payload.get("results"), dict) or not payload["results"]:
        raise ValueError(f"Missing physical evaluation results: {path}")
    if not isinstance(payload.get("protocol"), dict) or not payload["protocol"]:
        raise ValueError(f"Missing evaluation protocol: {path}")
    if not isinstance(payload.get("arguments"), dict):
        raise ValueError(f"Missing evaluation arguments: {path}")
    for mode, result in payload["results"].items():
        if not isinstance(result, dict) or not result.get("initial_state_sha256"):
            raise ValueError(f"Missing initial-state hash for {mode}: {path}")
        for field in ("num_envs", "steps_requested", "control_dt"):
            if not _finite_number(result.get(field)) or result[field] <= 0:
                raise ValueError(f"Missing/invalid physical protocol field {mode}.{field}: {path}")
    values, passed = {}, True
    for condition in conditions:
        result = payload["results"].get(condition.mode)
        if not isinstance(result, dict) or condition.metric not in result:
            raise ValueError(f"Missing metric {condition.mode}.{condition.metric}: {path}")
        value = result[condition.metric]
        if value is not None and not _finite_number(value):
            raise ValueError(f"Metric must be finite numeric or null: {condition.mode}.{condition.metric} in {path}")
        values[f"{condition.mode}.{condition.metric}"] = value
        passed &= value is not None and (
            value >= condition.threshold if condition.operator == ">=" else value <= condition.threshold)
    # Ignore only artifact locations, record-saving switches, and evaluator
    # compute device. Every other argument participates, including future
    # options that could change the physical evaluation.
    ignored = {"checkpoint", "output", "device", "save_transitions", "save_trace"}
    evaluation_args = {key: value for key, value in payload["arguments"].items() if key not in ignored}
    signature = {
        "protocol": payload["protocol"],
        "arguments": evaluation_args,
        "modes": {
            mode: {key: result[key] for key in (
                "initial_state_sha256", "num_envs", "steps_requested", "control_dt")}
            for mode, result in sorted(payload["results"].items())
        },
    }
    return {
        "metrics_path": str(path),
        "checkpoint": payload.get("checkpoint"),
        "training_cost": payload.get("training_cost"),
        "thresholds_passed": bool(passed),
        "metric_values": values,
        "signature": signature,
    }


def _cost_problems(record):
    cost = record["training_cost"]
    if not isinstance(cost, dict):
        return ["checkpoint has no training_cost metadata"]
    problems = []
    if cost.get("history_complete") is not True:
        problems.append("training cost history is incomplete")
    transitions = cost.get("physical_transitions")
    if not isinstance(transitions, int) or isinstance(transitions, bool) or transitions < 0:
        problems.append("physical transition count is unknown/invalid")
    wall = cost.get("wall_seconds")
    if not _finite_number(wall) or wall < 0:
        problems.append("cumulative training wall time is unknown/invalid")
    gpus = cost.get("gpu_count")
    if not isinstance(gpus, int) or isinstance(gpus, bool) or gpus < 0:
        problems.append("GPU count is unknown/invalid")
    if cost.get("wall_time_scope") != WALL_TIME_SCOPE:
        problems.append("wall time accounting scope does not match the shared runner scope")
    return problems


def _summarize_arm(records):
    issues = [{"metrics_path": row["metrics_path"], "problems": problems}
              for row in records if (problems := _cost_problems(row))]
    # Do not silently discard early checkpoints with unknown costs and claim
    # the remaining observed crossing was the first one in a complete history.
    if issues:
        return {
            "status": "cost_unavailable", "cost_history_trusted": False,
            "checkpoints_evaluated": len(records), "first_threshold_crossing": None,
            "thresholds_passed_anywhere": any(row["thresholds_passed"] for row in records),
            "issues": issues,
        }
    ordered = sorted(records, key=lambda row: (
        row["training_cost"]["physical_transitions"], row["training_cost"]["wall_seconds"]))
    previous_wall = -1.
    gpu_counts = set()
    parent_dirs = set()
    trajectory_ids = set()
    all_trajectory_ids_present = True
    for row in ordered:
        cost = row["training_cost"]
        if cost["wall_seconds"] < previous_wall:
            raise ValueError("Checkpoint costs are not monotonic; use one training trajectory per arm")
        previous_wall = cost["wall_seconds"]
        gpu_counts.add(cost["gpu_count"])
        trajectory_id = cost.get("trajectory_id")
        if isinstance(trajectory_id, str) and trajectory_id:
            trajectory_ids.add(trajectory_id)
        else:
            all_trajectory_ids_present = False
        if row["checkpoint"]:
            parent_dirs.add(str(Path(row["checkpoint"]).parent))
    if len(trajectory_ids) > 1:
        raise ValueError("Multiple training trajectory_id values in one arm; compare one trajectory at a time")
    if not all_trajectory_ids_present and len(parent_dirs) > 1:
        raise ValueError("Multiple training checkpoint directories in one arm; compare one trajectory at a time")
    if len(gpu_counts) != 1:
        raise ValueError("GPU count changes within a trajectory; cumulative GPU hours cannot be inferred")

    first = next((row for row in ordered if row["thresholds_passed"]), None)
    public = [{key: value for key, value in row.items() if key != "signature"} for row in ordered]
    crossing = None
    if first is not None:
        crossing = {key: value for key, value in first.items() if key != "signature"}
        cost = first["training_cost"]
        crossing["gpu_hours"] = cost["wall_seconds"] * cost["gpu_count"] / 3600.
    return {
        "status": "reached" if first is not None else "not_reached",
        "cost_history_trusted": True, "checkpoints_evaluated": len(records),
        "first_threshold_crossing": crossing,
        "observed_budget": ordered[-1]["training_cost"],
        "checkpoints": public,
    }


def compare_learning_efficiency(wm_paths, ppo_paths, *, minimum=(), maximum=()):
    conditions = [parse_condition(value, ">=") for value in minimum]
    conditions += [parse_condition(value, "<=") for value in maximum]
    if not conditions:
        raise ValueError("Specify at least one --minimum or --maximum physical performance threshold")
    records = {
        "wm": [_read_record(path, conditions) for path in discover_metrics(wm_paths)],
        "ppo": [_read_record(path, conditions) for path in discover_metrics(ppo_paths)],
    }
    reference = records["wm"][0]["signature"]
    for arm in records:
        for row in records[arm]:
            if row["signature"] != reference:
                raise ValueError(
                    f"Physical evaluation protocol/arguments/initial-state hashes differ: {row['metrics_path']}")
    arms = {name: _summarize_arm(rows) for name, rows in records.items()}
    trusted = all(arm["cost_history_trusted"] for arm in arms.values())
    reached = trusted and all(arm["status"] == "reached" for arm in arms.values())
    ratios = None
    different_gpus = None
    if trusted:
        different_gpus = arms["wm"]["observed_budget"]["gpu_count"] != arms["ppo"]["observed_budget"]["gpu_count"]
    if reached:
        a, b = (arms[name]["first_threshold_crossing"] for name in ("wm", "ppo"))
        ratios = {}
        for field in ("physical_transitions", "wall_seconds"):
            denominator = b["training_cost"][field]
            ratios[f"wm_over_ppo_{field}"] = a["training_cost"][field] / denominator if denominator else None
        ratios["wm_over_ppo_gpu_hours"] = a["gpu_hours"] / b["gpu_hours"] if b["gpu_hours"] else None
    return {
        "schema": "amp-learning-efficiency-v1",
        "conditions": [asdict(condition) for condition in conditions],
        "evaluation_protocol_verified": True,
        "training_cost_comparison_available": reached,
        "interpretation": (
            "First threshold crossing among supplied physical evaluations; not stable convergence. "
            "Use repeated checkpoints and independent seeds to assess persistence. "
            "Wall time excludes environment construction and external evaluation; GPU hours are wall time times GPU count. "
            "GPU model/utilization equivalence is not established by this metadata."
        ),
        "different_gpu_counts": different_gpus,
        "arms": arms,
        "ratios": ratios,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wm", nargs="+", type=Path, required=True, help="WM evaluation metrics JSON or directories")
    parser.add_argument("--ppo", nargs="+", type=Path, required=True, help="PPO evaluation metrics JSON or directories")
    parser.add_argument("--minimum", action="append", default=[], metavar="MODE.METRIC=VALUE")
    parser.add_argument("--maximum", action="append", default=[], metavar="MODE.METRIC=VALUE")
    parser.add_argument("--output", type=Path, help="Optional comparison JSON output; input files are never changed")
    args = parser.parse_args(argv)
    if not args.minimum and not args.maximum:
        parser.error("At least one --minimum or --maximum threshold is required")
    return parser, args


def main(argv=None):
    parser, args = parse_args(argv)
    try:
        report = compare_learning_efficiency(args.wm, args.ppo, minimum=args.minimum, maximum=args.maximum)
    except ValueError as error:
        parser.error(str(error))
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if args.output:
        # Exclusive creation also protects an input metrics.json if it was
        # accidentally supplied as the output destination.
        try:
            with args.output.expanduser().resolve().open("x") as handle:
                handle.write(encoded)
        except OSError as error:
            parser.error(f"Cannot create comparison output without overwriting an existing file: {error}")
    print(encoded, end="")


if __name__ == "__main__":
    main()
