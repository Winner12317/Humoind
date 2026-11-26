"""Four-arm fixed-data short continuation of the two recovery-semantics fixes.

Networks and Adam start from one checkpoint in every arm. This is not an
on-policy training experiment: recorded data are immutable, no physical rollout
is collected for training, and old physical reward labels are used only for
reward-gap diagnostics. A source with a real-data Critic loss is refused.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime
import gc
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.diagnose_wmac import (
    MODULES, atomic_json, audit_evaluation_pair, build_replay, close_runner,
    diagnostic_config, module_fingerprint, phase_rng, phase_seed, read_dataset,
    resolved_config, sha256, snapshot_source, state_fingerprint,
)
from src.recovery_rewards import configure_recovery_height_reward

ARMS = {
    "legacy": ("soft_v1", "height_only_v1"),
    "counter_only": ("hard_forward_v2", "height_only_v1"),
    "height_only": ("soft_v1", "upright_v2"),
    "both": ("hard_forward_v2", "upright_v2"),
}
OPTIMIZERS = ("opt_actor", "opt_model", "opt_disc", "opt_critic")
COMMON_EVALUATION_HEIGHT_REWARD = "upright_v2"


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset", type=Path, nargs="+", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--updates", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--wm-batch-size", type=int, default=256)
    p.add_argument("--wm-updates", type=int, default=1)
    p.add_argument("--data-steps", type=int, default=249,
                   help="Use start-zero recorded prefixes of at most249 steps, strictly before the original recovery deadline")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--evaluation-seed", type=int, default=2030)
    p.add_argument("--evaluation-envs", type=int, default=32)
    p.add_argument("--evaluation-steps", type=int, default=500)
    p.add_argument("--no-final-evaluation", action="store_true")
    return p


def parse_args(argv=None):
    p = parser()
    args = p.parse_args(argv)
    for name in ("updates", "batch_size", "wm_batch_size", "wm_updates", "data_steps",
                 "evaluation_envs", "evaluation_steps"):
        if getattr(args, name) < 1:
            p.error(f"--{name.replace('_', '-')} must be positive")
    if args.data_steps > 249:
        p.error("--data-steps must be at most249 so recorded counters remain strictly before250")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.dataset = [path.expanduser().resolve() for path in args.dataset]
    args.output = args.output.expanduser().resolve()
    if args.output.exists():
        p.error("--output must be a new directory; existing evidence cannot be overwritten")
    return args


def validate_source(payload):
    source = resolved_config(payload)
    if payload.get("replay") is not None:
        raise ValueError("Source checkpoints with saved replay are not accepted; the supplied fixed dataset is the only replay")
    if source.real_critic_coef != 0.:
        raise ValueError("real_critic_coef must be0: recorded physical rewards retain their old labels after a height-objective change")
    if int(payload.get("iteration", 0)) < source.model_warmup_iterations:
        raise ValueError("The source has not completed model warmup")
    calibrated = int(payload.get("critic_calibration_updates", 0))
    if source.critic_bootstrap and calibrated < source.critic_warmup_iterations:
        raise ValueError("The source Critic calibration is incomplete; all arms must share an already calibrated Critic")
    missing = set(MODULES).union(OPTIMIZERS) - payload.keys()
    if missing:
        raise ValueError(f"Source is missing trained networks or Adam state: {sorted(missing)}")
    return source


def crop_start_zero_data(chunks, metadata, data_steps):
    """Retain genuine start-zero prefixes and refuse any recorded250 counter."""
    if not 1 <= data_steps <= 249:
        raise ValueError("Use1..249 recorded steps")
    cropped, records = [], []
    for chunk, description in zip(chunks, metadata):
        path = Path(description["path"])
        with np.load(path, allow_pickle=False) as archive:
            if "step_index" not in archive:
                raise ValueError(f"Recording {path} has no step_index proving a start-zero prefix")
            index = np.asarray(archive["step_index"])
        available = len(chunk["state"])
        if index.shape != (available,) or index.dtype.kind not in "iu" or not np.array_equal(index, np.arange(available)):
            raise ValueError("Every recording must start at control step0 and have consecutive step_index values")
        if available < data_steps:
            raise ValueError(f"Recording {path} has fewer than the requested {data_steps} steps")
        if not bool(chunk["first_episode_valid"][0].all()):
            raise ValueError("Every recording must begin in the first physical episode")
        first_counter = chunk["context"][0, :, 34]
        if not bool(torch.isfinite(first_counter).all()) or not bool((first_counter == 0).all()):
            raise ValueError("Every source recording must begin with recovery counter0")
        values = {key: value[:data_steps].clone() for key, value in chunk.items()}
        valid = values["first_episode_valid"]
        maxima = {}
        for key, maximum in (("context", torch.arange(data_steps)[:, None]),
                             ("next_context", torch.arange(1, data_steps+1)[:, None])):
            counter = values[key][..., 34]
            selected = counter[valid]
            if (not bool(torch.isfinite(selected).all()) or bool((selected < 0).any())
                    or bool((selected >= 250).any()) or bool((counter[valid] > maximum.expand_as(counter)[valid]).any())
                    or not bool((selected == selected.round()).all())):
                raise ValueError("Used physical recovery counters must be integer, nonnegative,<250 and consistent with a start-zero prefix")
            maxima[key] = float(selected.max()) if selected.numel() else None
        cropped.append(values)
        records.append({**description, "used_steps": data_steps,
                        "used_first_episode_transitions": int(valid.sum()),
                        "used_tensor_sha256": state_fingerprint(values),
                        "maximum_used_recovery_counter": maxima,
                        "recorded_reward_use": "Only validation_reward_gap diagnostics; never an Actor or real-data Critic target"})
    return cropped, records


def initial_fingerprints(runner):
    return {"modules": {name: module_fingerprint(getattr(runner, name)) for name in MODULES},
            "optimizers": {name: state_fingerprint(getattr(runner, name).state_dict()) for name in OPTIMIZERS},
            "amp_normalizer_settings": {key: getattr(runner.amp_norm, key)
                                         for key in ("eps", "variance_epsilon", "clip", "max_count")},
            "critic_calibration_updates": runner.critic_calibration_updates}


def actor_phase_updates(runner, gate):
    """Retain the source Actor update count and aggregate as the training loop does."""
    if not gate:
        return {}
    samples = [runner.update_actor_critic() for _ in range(runner.cfg.actor_updates)]
    return {key: float(np.mean([sample[key] for sample in samples])) for key in samples[0]}


def source_copy_snapshot(output):
    snapshot = snapshot_source(output)
    for relative in (Path("scripts/compare_wmac_recovery_fixes.py"), Path("src/recovery_rewards.py")):
        content = (REPO/relative).read_bytes()
        target = output/"source"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        snapshot["files_sha256"][str(relative)] = sha256(REPO/relative)
    return snapshot


def evaluate_common(checkpoint, output, args):
    """All arm and source policies face the same physical upright_v2 reward."""
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(REPO/"scripts/evaluate_wmac.py"),
               "--checkpoint", str(checkpoint), "--mode", "both",
               "--seed", str(args.evaluation_seed), "--num-envs", str(args.evaluation_envs),
               "--steps", str(args.evaluation_steps), "--device", args.device,
               "--recovery-height-reward", COMMON_EVALUATION_HEIGHT_REWARD,
               "--save-trace", "--output", str(output)]
    with output.with_suffix(".log").open("w") as log:
        subprocess.run(command, cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO),
            "MUJOCO_GL": "egl", "OMP_NUM_THREADS": "1"}, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads((output/"metrics.json").read_text())


def audit_common_protocol(reference, candidate):
    audit = audit_evaluation_pair(reference, candidate)
    extra = {key: [reference.get("protocol", {}).get(key), candidate.get("protocol", {}).get(key)]
             for key in ("recovery_height_reward", "amp_feature_schema")
             if reference.get("protocol", {}).get(key) != candidate.get("protocol", {}).get(key)}
    expected = all(result.get("protocol", {}).get("recovery_height_reward") == COMMON_EVALUATION_HEIGHT_REWARD
                   for result in (reference, candidate))
    if extra or not expected:
        audit["paired_comparison_allowed"] = audit["causal_comparison_eligible"] = False
        audit["status"] = "paired_comparison_refused"
        audit["reasons"].append("Common physical upright_v2 reward or AMP schema was not verified")
    audit["recovery_reward_and_schema_differences"] = extra
    audit["common_physical_height_reward_verified"] = expected
    return audit


def main(argv=None):
    args = parse_args(argv)
    if not args.checkpoint.is_file() or any(not path.is_file() for path in args.dataset):
        raise FileNotFoundError("All source checkpoint and dataset paths must be existing files")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    source = validate_source(payload)
    if args.data_steps < source.wm_window:
        raise ValueError("--data-steps must cover the saved world-model fitting window")
    input_hashes = {str(path): sha256(path) for path in (args.checkpoint, *args.dataset)}
    chunks, descriptions = read_dataset(args.dataset)
    chunks, descriptions = crop_start_zero_data(chunks, descriptions, args.data_steps)
    replay = build_replay(chunks)
    del chunks
    fixed_replay_hash = state_fingerprint(replay.data)
    args.precision = source.precision
    args.output.mkdir(parents=True, exist_ok=False)
    manifest = {"created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "scope": "Four-arm fixed-data short continuation; not on-policy resume, convergence evidence, or training-speed benchmark",
        "arguments": vars(args), "inputs_sha256": input_hashes, "source": source_copy_snapshot(args.output),
        "source_config": asdict(source), "source_iteration": payload["iteration"], "data": descriptions,
        "fixed_replay": {"sha256": fixed_replay_hash, "steps": replay.size, "num_envs": replay.num_envs,
            "train_transitions": replay.num_valid("train"), "validation_transitions": replay.num_valid("validation")},
        "arms": {key: {"recovery_counter_semantics": pair[0], "recovery_height_reward": pair[1]}
                 for key, pair in ARMS.items()},
        "physical_protocol": {"height_reward": COMMON_EVALUATION_HEIGHT_REWARD,
            "seed": args.evaluation_seed, "num_envs": args.evaluation_envs, "steps": args.evaluation_steps,
            "paired_arrays_tolerance": 1e-6, "physical_delay_rule": "original DelayedTerminationManager"},
        "limitations": ["Only source/data prefixes before step249 are used; source counters must start at0 and remain<250.",
            "Source real_critic_coef must be0 because saved physical rewards have stale semantics; validation_reward_gap is diagnostic and never an Actor gate criterion.",
            "No BC/PPO loss, new replay collection, or normalizer fitting from collected physics is added.",
            "World-model and Critic normalizers remain fixed; the AMP normalizer follows the same saved D-update rule in every arm.",
            "Every arm preserves source networks, Adam, learning-rate schedule and calibrated Critic before the two explicit semantics interventions.",
            "The shared diagnostic batch/model-update budget is reduced from production and listed in config differences.",
            "A short rollout/update check cannot establish persistent recovery success, causal sufficiency or training acceleration."],
        "status": "running", "results": {}}
    atomic_json(args.output/"manifest.json", manifest)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.torch import configure_torch_backends
    from src.wmac.env import CaptureEnv
    from src.wmac.runner import WmacRunner
    from src.wmac.state import StateAdapter

    configure_torch_backends(allow_tf32=source.precision == "tf32")
    baseline_fingerprints = None
    env = runner = None
    try:
        for label, (counter_semantics, height_semantics) in ARMS.items():
            arm = args.output/label
            arm.mkdir()
            cfg = diagnostic_config(source, replay, args, arm)
            cfg.recovery_counter_semantics, cfg.recovery_height_reward = counter_semantics, height_semantics
            cfg.allow_recovery_semantics_migration = True
            cfg.iterations = int(payload["iteration"]) + args.updates
            cfg.checkpoint_replay = False
            cfg.validate()
            env_cfg = load_env_cfg(source.task)
            configure_recovery_height_reward(env_cfg, height_semantics)
            env_cfg.scene.num_envs, env_cfg.seed = 2, args.seed
            env = CaptureEnv(cfg=env_cfg, device=args.device)
            env.adapter = StateAdapter(env, recovery_counter_semantics=counter_semantics)
            runner = WmacRunner(env, cfg, arm)
            runner.load(str(args.checkpoint), allow_recovery_semantics_migration=True)
            runner.replay = replay
            runner._train_env_mask = replay.train_env_mask.to(runner.device)
            runner._cost_history_complete = False
            if runner.device.type == "cpu":
                for name in OPTIMIZERS:
                    for group in getattr(runner, name).param_groups:
                        group["fused"] = False
            initial = initial_fingerprints(runner)
            if baseline_fingerprints is None:
                baseline_fingerprints = initial
            if initial != baseline_fingerprints:
                raise RuntimeError(f"Arm {label} did not start with identical source networks, Adam, normalization settings and Critic calibration")
            result = {"status": "updating", "initial_state": initial, "initial_state_matches_legacy": True,
                "actual_config": asdict(cfg), "config_differences": {key: {"source": asdict(source)[key], "diagnostic": value}
                    for key, value in asdict(cfg).items() if value != asdict(source)[key]},
                "metadata_environment_num_envs": 2, "new_physical_training_transitions": 0, "updates": []}
            manifest["results"][label] = result
            atomic_json(args.output/"manifest.json", manifest)
            started = time.perf_counter()
            for index in range(args.updates):
                runner._set_learning_rates()
                with phase_rng(args.seed, index, "model", args.device):
                    model = runner.fit_model()
                with phase_rng(args.seed, index, "discriminator", args.device):
                    discriminator = runner.update_discriminator()
                with phase_rng(args.seed, index, "validation", args.device):
                    validation = runner.validate_model()
                gate = runner._actor_gate(model)
                with phase_rng(args.seed, index, "actor", args.device):
                    update = actor_phase_updates(runner, gate)
                runner.iteration += 1
                row = {"diagnostic_update": index+1, "source_iteration_plus_updates": runner.iteration,
                    "actor_gate_passed": gate, "actor_update_attempted": gate,
                    "actor_updates_attempted": cfg.actor_updates if gate else 0,
                    "actor_lr": runner.opt_actor.param_groups[0]["lr"],
                    "phase_seeds": {name: phase_seed(args.seed, index, name)
                                    for name in ("model", "discriminator", "validation", "actor")},
                    **model, **discriminator, **validation, **update}
                row["actor_gradient_finite"] = math.isfinite(row["actor_grad_norm"]) if "actor_grad_norm" in row else None
                row["nonfinite_fields"] = [key for key, value in row.items() if isinstance(value, float) and not math.isfinite(value)]
                result["updates"].append(row)
                atomic_json(arm/"updates.json", result["updates"])
            result["fixed_data_update_wall_seconds"] = time.perf_counter()-started
            result["final_state"] = initial_fingerprints(runner)
            result["fixed_replay_data_unchanged"] = state_fingerprint(replay.data) == fixed_replay_hash
            checkpoint = arm/f"model_{runner.iteration}.pt"
            runner.save(checkpoint, export=False)
            result["checkpoint"] = str(checkpoint)
            result["checkpoint_sha256"] = sha256(checkpoint)
            result["inputs_unchanged_after_arm"] = all(sha256(Path(path)) == value for path, value in input_hashes.items())
            if not result["inputs_unchanged_after_arm"] or not result["fixed_replay_data_unchanged"]:
                raise RuntimeError("An input file or fixed replay tensor changed during an arm")
            result["status"] = "completed"
            atomic_json(args.output/"manifest.json", manifest)
            close_runner(runner)
            runner = None
            env.close()
            env = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[recovery-fixes] {label}: {args.updates} fixed-data update attempts completed", flush=True)

        values = manifest["results"]
        consistency = {}
        for name in ("world_model", "discriminator", "amp_norm"):
            hashes = {label: record["final_state"]["modules"][name] for label, record in values.items()}
            consistency[name] = {"all_equal": len(set(hashes.values())) == 1, "hashes": hashes}
        for name in ("opt_model", "opt_disc"):
            hashes = {label: record["final_state"]["optimizers"][name] for label, record in values.items()}
            consistency[name] = {"all_equal": len(set(hashes.values())) == 1, "hashes": hashes}
        manifest["independent_update_consistency"] = consistency
        if not args.no_final_evaluation:
            evaluations = args.output/"evaluations"
            reference = evaluate_common(args.checkpoint, evaluations/"source", args)
            manifest["physical_source_evaluation"] = reference
            atomic_json(args.output/"manifest.json", manifest)
            for label, record in values.items():
                evaluated = evaluate_common(Path(record["checkpoint"]), evaluations/label, args)
                record["physical_evaluation"] = evaluated
                record["pairing_with_source"] = audit_common_protocol(reference, evaluated)
                atomic_json(args.output/"manifest.json", manifest)
        manifest["inputs_unchanged_at_completion"] = all(sha256(Path(path)) == value for path, value in input_hashes.items())
        if not manifest["inputs_unchanged_at_completion"]:
            raise RuntimeError("An input checkpoint or dataset changed during evaluation")
        manifest["status"] = "completed"
        atomic_json(args.output/"manifest.json", manifest)
        atomic_json(args.output/"results.json", {"scope": manifest["scope"],
            "physical_protocol": manifest["physical_protocol"], "independent_update_consistency": consistency,
            "results": values, "source_evaluation": manifest.get("physical_source_evaluation"),
            "conclusion": "Short fixed-data intervention evidence only; do not infer persistent recovery or training-speed improvement.",
            "inputs_unchanged": manifest["inputs_unchanged_at_completion"]})
        print(f"Results: {args.output/'results.json'}")
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        atomic_json(args.output/"manifest.json", manifest)
        raise
    finally:
        if runner is not None:
            close_runner(runner)
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
