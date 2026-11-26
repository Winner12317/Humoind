"""Run saved WMAC/PPO Actors under one physical protocol, without training."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def digest(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    output = args.manifest.resolve().parent
    protocol = manifest["protocol"]
    os.environ["MUJOCO_GL"] = "egl"
    os.environ.setdefault("OMP_NUM_THREADS", "1")

    import numpy as np
    import torch
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.utils.torch import configure_torch_backends
    from evaluate_wmac import evaluate_mode, load_actor

    configure_torch_backends()
    initial_references = {}
    status = {"completed": [], "started_at": time.time(), "finished": False}
    status_path = output / "progress.json"
    save_json(status_path, status)
    for job in manifest["checkpoints"]:
        checkpoint = Path(job["checkpoint"])
        before = digest(checkpoint)
        if before != job["sha256"]:
            raise RuntimeError(f"Checkpoint changed before evaluation: {checkpoint}")
        actor, kind = load_actor(checkpoint, protocol["device"])
        for seed in protocol["seeds"]:
            target = output / "evaluations" / job["label"] / f"seed_{seed}"
            target.mkdir(parents=True, exist_ok=True)
            report_path = target / "metrics.json"
            report = {"label": job["label"], "algorithm": job["algorithm"],
                "iteration": job["iteration"], "checkpoint": str(checkpoint),
                "checkpoint_sha256": before, "checkpoint_format": kind,
                "seed": seed, "protocol": protocol, "results": {}}
            settings = argparse.Namespace(**{key: value for key, value in protocol.items() if key != "seeds"})
            settings.seed = seed
            for mode in ("walk", "recovery"):
                started = time.perf_counter()
                print(f"START {job['label']} seed={seed} mode={mode}", flush=True)
                result = evaluate_mode(actor, settings, mode, target)
                key = (seed, mode)
                initial_path = target / f"initial_{mode}.npz"
                with np.load(initial_path) as data:
                    current = {name: data[name].copy() for name in data.files}
                if key not in initial_references:
                    initial_references[key] = current
                differences = {name: float(np.max(np.abs(value - initial_references[key][name])))
                    for name, value in current.items()}
                matched = all(value <= 1e-6 for value in differences.values())
                result["initial_conditions_match_reference_within_1e_6"] = matched
                result["initial_max_absolute_difference"] = differences
                report["results"][mode] = result
                save_json(report_path, report)
                if not matched:
                    raise RuntimeError(f"Initial states differ: {job['label']} seed={seed} {mode}: {differences}")
                compact = {"label": job["label"], "seed": seed, "mode": mode,
                    "seconds": time.perf_counter() - started,
                    "walk_survival": result["first_episode_survival_fraction"],
                    "xy_error": result["velocity_xy_error_mean"],
                    "recovery_success": result.get("recovery_success_fraction"),
                    "final_upright": result["final_upright_fraction"]}
                status["completed"].append(compact)
                save_json(status_path, status)
                print("DONE " + json.dumps(compact, ensure_ascii=False), flush=True)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        del actor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if digest(checkpoint) != before:
            raise RuntimeError(f"Checkpoint changed during evaluation: {checkpoint}")
    status.update(finished=True, finished_at=time.time())
    save_json(status_path, status)


if __name__ == "__main__":
    main()
