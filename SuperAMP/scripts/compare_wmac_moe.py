"""Compare MLP, dense MoE and sparse MoE on one immutable physical dataset.

All arms reset Actor Adam, including the MLP control. Other networks, Adam
states and normalization inherit one trained MLP checkpoint. This is a bounded
fixed-data intervention, not on-policy resume or a training-speed benchmark.
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

from scripts import compare_wmac_recovery_fixes as common
from scripts.diagnose_wmac import (
    MODULES, atomic_json, audit_evaluation_pair, build_replay, close_runner,
    diagnostic_config, module_fingerprint, phase_rng, phase_seed, read_dataset,
    sha256, snapshot_source, state_fingerprint,
)
from src.recovery_rewards import configure_recovery_height_reward
from src.wmac.actors import actor_architecture, create_actor

ARMS = {
    "mlp": {"actor_type": "mlp", "moe_top_k": 2, "moe_load_balance_coef": 0., "moe_router_z_coef": 0.},
    "moe_dense": {"actor_type": "moe", "moe_top_k": 4, "moe_load_balance_coef": 0., "moe_router_z_coef": 0.},
    "moe_top2": {"actor_type": "moe", "moe_top_k": 2, "moe_load_balance_coef": 0., "moe_router_z_coef": 0.},
    "moe_top2_regularized": {"actor_type": "moe", "moe_top_k": 2, "moe_load_balance_coef": .01, "moe_router_z_coef": .001},
}
AUXILIARIES = tuple(name for name in MODULES if name != "actor")
AUX_OPTIMIZERS = ("opt_model", "opt_disc", "opt_critic")


def parse_args(argv=None):
    parser = common.parser()
    parser.description = __doc__
    args = parser.parse_args(argv)
    for name in ("updates", "batch_size", "wm_batch_size", "wm_updates", "data_steps", "evaluation_envs", "evaluation_steps"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.data_steps > 249:
        parser.error("--data-steps must be at most 249; do not cross the original recovery guard")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.dataset = [path.expanduser().resolve() for path in args.dataset]
    args.output = args.output.expanduser().resolve()
    if args.output.exists():
        parser.error("--output must be a new directory")
    return args


def validate_source(payload):
    if "replay" in payload:
        raise ValueError("Source must not contain saved replay")
    source = common.validate_source(payload)
    if source.actor_type != "mlp":
        raise ValueError("This comparison requires one trained MLP source")
    return source


def arm_config(source, replay, args, directory, label, source_iteration):
    cfg = diagnostic_config(source, replay, args, directory)
    for key, value in ARMS[label].items():
        setattr(cfg, key, value)
    # This reproduces the historical four-arm intervention, not the new
    # Hiking Actor (which cannot preserve this MLP's function by cloning).
    cfg.moe_variant, cfg.moe_gate_hidden = "legacy", (256,)
    cfg.moe_num_experts, cfg.moe_init_noise_std = 4, 0.
    cfg.iterations = source_iteration + args.updates
    cfg.checkpoint_replay, cfg.allow_recovery_semantics_migration = False, False
    cfg.validate()
    return cfg


def converted_initial_payload(source_payload, runner, source_path, source_hash, label, initialization_seed):
    """Change only Actor representation/Adam and explicit experiment metadata."""
    if runner.opt_actor.state_dict()["state"]:
        raise ValueError("Actor Adam must be freshly constructed and empty in every arm")
    if runner.cfg.actor_type == "mlp":
        runner.actor.load_state_dict(source_payload["actor"])
    else:
        runner.actor.load_mlp_state_dict(source_payload["actor"], noise_std=0.)
    result = dict(source_payload)
    result.update(config=asdict(runner.cfg), actor_type=runner.cfg.actor_type,
        actor_architecture=actor_architecture(runner.cfg),
        implementation_version=5 if runner.cfg.actor_type == "moe" else 4,
        actor={key: value.detach().cpu().clone() for key, value in runner.actor.state_dict().items()},
        opt_actor=runner.opt_actor.state_dict(), best_eval_score=None,
        recovery_counter_semantics=runner.cfg.recovery_counter_semantics,
        recovery_height_reward=runner.cfg.recovery_height_reward)
    history = list(source_payload.get("actor_initialization_history", []))
    history.append({"kind": "fixed_data_actor_comparison", "arm": label,
        "source_checkpoint": str(source_path), "source_sha256": source_hash,
        "source_iteration": source_payload["iteration"],
        "source_architecture": actor_architecture(common.resolved_config(source_payload)),
        "target_architecture": actor_architecture(runner.cfg),
        "actor_adam_reset": True, "other_networks_and_adam_preserved": True,
        "expert_initialization": "exact source MLP clone; no expert perturbation" if runner.cfg.actor_type == "moe" else "source MLP weights",
        "initialization_seed": initialization_seed, "fixed_data_not_on_policy_resume": True})
    result["actor_initialization_history"] = history
    if isinstance(source_payload.get("training_cost"), dict):
        result["training_cost"] = {**source_payload["training_cost"], "history_complete": False}
    return result


def auxiliary_fingerprints(states):
    return {name: state_fingerprint(states[name]) for name in (*AUXILIARIES, *AUX_OPTIMIZERS)}


def initial_action_audit(cfg, state, observations, reference_actions):
    """Check the cloned initial function in CPU float32, independent of TF32."""
    with torch.random.fork_rng(devices=[]):
        actor = create_actor(cfg).cpu().eval()
        actor.load_state_dict(state)
    actor.precision = "float32"
    with torch.no_grad():
        actions = actor(observations.cpu().float())
        maximum = float((actions-reference_actions).abs().max())
        if not bool(torch.isfinite(actions).all()) or not math.isfinite(maximum) or maximum >= 1e-5:
            raise ValueError(f"Initial Actor function differs from the MLP source: maximum action gap {maximum}")
        arrays = {"actions": actions.numpy()}
        if cfg.actor_type == "moe":
            route = actor.route(observations.cpu().float())
            arrays.update({name: getattr(route, name).numpy() for name in ("indices", "weights", "probabilities")})
    return {"maximum_absolute_action_difference": maximum, "passed": True,
            "protocol": "same fixed held-out physical observations; CPU float32; training retains source precision",
            "actor_state_sha256": module_fingerprint(actor), "normalizer_sha256": module_fingerprint(actor.norm),
            "router_sha256": module_fingerprint(actor.router) if cfg.actor_type == "moe" else None,
            "experts_sha256": [module_fingerprint(expert) for expert in actor.experts] if cfg.actor_type == "moe" else None}, arrays


def evaluate_common(checkpoint, output, args, height_reward):
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, str(REPO/"scripts/evaluate_wmac.py"), "--checkpoint", str(checkpoint),
        "--mode", "both", "--seed", str(args.evaluation_seed), "--num-envs", str(args.evaluation_envs),
        "--steps", str(args.evaluation_steps), "--device", args.device,
        "--recovery-height-reward", height_reward, "--save-trace", "--output", str(output)]
    with output.with_suffix(".log").open("w") as log:
        subprocess.run(command, cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO),
            "MUJOCO_GL": "egl", "OMP_NUM_THREADS": "1"}, stdout=log, stderr=subprocess.STDOUT, check=True)
    return json.loads((output/"metrics.json").read_text())


def audit_common_protocol(reference, candidate, height_reward):
    audit = audit_evaluation_pair(reference, candidate)
    differences = {key: [reference.get("protocol", {}).get(key), candidate.get("protocol", {}).get(key)]
        for key in ("recovery_height_reward", "amp_feature_schema")
        if reference.get("protocol", {}).get(key) != candidate.get("protocol", {}).get(key)}
    verified = all(value.get("protocol", {}).get("recovery_height_reward") == height_reward
                   for value in (reference, candidate))
    if differences or not verified:
        audit["paired_comparison_allowed"] = audit["causal_comparison_eligible"] = False
        audit["status"] = "paired_comparison_refused"
        audit["reasons"].append("The common source height reward or AMP schema differs")
    audit.update(recovery_reward_and_schema_differences=differences, common_source_height_reward_verified=verified)
    return audit


def routing_summary(path):
    with np.load(path, allow_pickle=False) as archive:
        indices, valid = archive["indices"], archive["first_episode_valid"]
    if indices.ndim != 3 or valid.shape != indices.shape[:2] or valid.dtype.kind != "b":
        raise ValueError("Routing archive must contain [time,env,k] indices and boolean validity")
    with np.load(path, allow_pickle=False) as archive:
        protocol = json.loads(str(archive["routing_protocol_json"].item()))
    experts = int(protocol["num_experts"])
    if indices.dtype.kind not in "iu" or (indices < 0).any() or (indices >= experts).any():
        raise ValueError("Invalid expert indices in routing archive")
    chosen = indices[valid].reshape(-1)
    count = np.bincount(chosen, minlength=experts)
    pairs = valid[1:] & valid[:-1]
    membership_change = (np.sort(indices[1:], axis=-1) != np.sort(indices[:-1], axis=-1)).any(-1)
    top_change = indices[1:, :, 0] != indices[:-1, :, 0]
    return {"path": str(path), "sha256": sha256(path), "valid_actions": int(valid.sum()),
        "expert_selection_fraction": (count/max(len(chosen), 1)).tolist(),
        "consecutive_first_episode_pairs": int(pairs.sum()),
        "selected_set_switch_fraction": float(membership_change[pairs].mean()) if pairs.any() else None,
        "top1_switch_fraction": float(top_change[pairs].mean()) if pairs.any() else None,
        "limitation": "Routing churn is descriptive; hard route boundaries may cause action discontinuities."}


def main(argv=None):
    args = parse_args(argv)
    if not args.checkpoint.is_file() or any(not path.is_file() for path in args.dataset):
        raise FileNotFoundError("All source checkpoint and data paths must exist")
    payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    source = validate_source(payload)
    if args.data_steps < source.wm_window:
        raise ValueError("--data-steps must cover the source model window")
    args.precision = source.precision
    input_hashes = {str(path): sha256(path) for path in (args.checkpoint, *args.dataset)}
    chunks, descriptions = read_dataset(args.dataset)
    chunks, descriptions = common.crop_start_zero_data(chunks, descriptions, args.data_steps)
    replay = build_replay(chunks)
    del chunks
    fixed_hash = state_fingerprint(replay.data)
    with phase_rng(args.seed, 0, "probe_starts"):
        batch = replay.sample_starts(args.batch_size, source.recovery_fraction, fields=("actor_history",), split="validation")
    observations = batch["actor_history"].cpu().float()
    with torch.random.fork_rng(devices=[]):
        reference = create_actor(source).cpu().eval()
        reference.load_state_dict(payload["actor"])
    reference.precision = "float32"
    with torch.no_grad():
        reference_actions = reference(observations)
    args.output.mkdir(parents=True, exist_ok=False)
    np.savez_compressed(args.output/"initial_observations_and_source_actions.npz", observations=observations.numpy(),
        source_actions=reference_actions.numpy(), time_index=batch["time_index"].numpy(), env_index=batch["env_index"].numpy())
    del reference
    snapshot = snapshot_source(args.output)
    for relative in (Path("scripts/compare_wmac_moe.py"), Path("scripts/compare_wmac_recovery_fixes.py")):
        target = args.output/"source"/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((REPO/relative).read_bytes())
        snapshot["files_sha256"][str(relative)] = sha256(REPO/relative)
    manifest = {"status": "running", "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "scope": "Historical legacy-MoE four-arm comparison, not Hiking; all Actor Adam states reset; not on-policy resume or a speed benchmark",
        "arguments": vars(args), "source_snapshot": snapshot, "inputs_sha256": input_hashes,
        "source_config": asdict(source), "source_iteration": payload["iteration"], "data": descriptions,
        "fixed_replay": {"sha256": fixed_hash, "steps": replay.size, "num_envs": replay.num_envs,
            "train_transitions": replay.num_valid("train"), "validation_transitions": replay.num_valid("validation")},
        "arm_interventions": ARMS, "common_moe_num_experts": 4, "initial_actions": {},
        "initialization_seed": phase_seed(args.seed, 0, "probe_noise"),
        "physical_protocol": {"height_reward": source.recovery_height_reward, "seed": args.evaluation_seed,
            "num_envs": args.evaluation_envs, "steps": args.evaluation_steps},
        "limitations": ["One fixed off-policy dataset and short updates do not establish convergence, recovery improvements or a speedup.",
            "Actor Adam is reset in every arm; this is a deliberate fresh-optimizer comparison, not exact continuation.",
            "Source real_critic_coef is zero; stale physical reward labels enter reward-gap diagnostics only, never targets or Actor gates.",
            "WM/Critic normalizers stay loaded; AMP moments follow the same source D-update rule in each arm.",
            "MoE adds total parameters; dense selects four experts and sparse selects two. Capacity and dispatch differ.",
            "Initial function equivalence is audited in CPU float32; GPU roundoff can still differ with dispatch batch sizes.",
            "Hard routing boundaries can cause action jumps; matched initial arrays do not imply bitwise physical trajectories."],
        "results": {}}
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
    expected_aux = auxiliary_fingerprints(payload)
    moe_initial_hash, normalizer_hash = None, None
    env = runner = None
    try:
        for label in ARMS:
            arm = args.output/label
            arm.mkdir()
            cfg = arm_config(source, replay, args, arm, label, int(payload["iteration"]))
            env_cfg = load_env_cfg(source.task)
            configure_recovery_height_reward(env_cfg, source.recovery_height_reward)
            env_cfg.scene.num_envs, env_cfg.seed = 2, args.seed
            env = CaptureEnv(cfg=env_cfg, device=args.device)
            env.adapter = StateAdapter(env, recovery_counter_semantics=source.recovery_counter_semantics)
            with phase_rng(args.seed, 0, "probe_noise", args.device):
                runner = WmacRunner(env, cfg, arm)
            converted = converted_initial_payload(payload, runner, args.checkpoint, input_hashes[str(args.checkpoint)],
                label, manifest["initialization_seed"])
            initial_path = arm/"initial.pt"
            torch.save(converted, initial_path)
            runner.load(initial_path)
            loaded_states = {name: getattr(runner, name).state_dict() for name in (*AUXILIARIES, *AUX_OPTIMIZERS)}
            if auxiliary_fingerprints(loaded_states) != expected_aux:
                raise RuntimeError("An auxiliary network or its Adam state differs from the source")
            audit, arrays = initial_action_audit(cfg, converted["actor"], observations, reference_actions)
            np.savez_compressed(arm/"initial_actions.npz", **arrays)
            manifest["initial_actions"][label] = audit
            if normalizer_hash is None:
                normalizer_hash = audit["normalizer_sha256"]
            if audit["normalizer_sha256"] != normalizer_hash:
                raise RuntimeError("Actor normalization differs between arms")
            if cfg.actor_type == "moe":
                if moe_initial_hash is None:
                    moe_initial_hash = audit["actor_state_sha256"]
                if audit["actor_state_sha256"] != moe_initial_hash:
                    raise RuntimeError("MoE expert/router initial weights differ across arms")
            if runner.opt_actor.state_dict()["state"]:
                raise RuntimeError("Converted initial checkpoint restored nonempty Actor Adam")
            del converted, loaded_states, arrays
            runner.replay, runner._train_env_mask = replay, replay.train_env_mask.to(runner.device)
            runner._cost_history_complete = False
            if runner.device.type == "cpu":
                for name in common.OPTIMIZERS:
                    for group in getattr(runner, name).param_groups:
                        group["fused"] = False
            result = {"status": "updating", "initial_checkpoint": str(initial_path),
                "initial_checkpoint_sha256": sha256(initial_path), "actor_adam_reset": True,
                "source_auxiliaries_and_adam_preserved": True, "initial_action_audit": audit,
                "initial_state": common.initial_fingerprints(runner), "actual_config": asdict(cfg),
                "config_differences": {key: {"source": asdict(source)[key], "diagnostic": value}
                    for key, value in asdict(cfg).items() if value != asdict(source)[key]},
                "new_physical_training_transitions": 0, "metadata_environment_num_envs": 2, "updates": []}
            manifest["results"][label] = result
            atomic_json(args.output/"initial_action_audit.json", manifest["initial_actions"])
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
                    update = common.actor_phase_updates(runner, gate)
                runner.iteration += 1
                row = {"diagnostic_update": index+1, "source_iteration_plus_updates": runner.iteration,
                    "actor_gate_passed": gate, "actor_updates_attempted": cfg.actor_updates if gate else 0,
                    "actor_lr": float(runner.opt_actor.param_groups[0]["lr"]),
                    "phase_seeds": {name: phase_seed(args.seed, index, name)
                        for name in ("model", "discriminator", "validation", "actor")},
                    **model, **discriminator, **validation, **update}
                row["actor_gradient_finite"] = math.isfinite(row["actor_grad_norm"]) if "actor_grad_norm" in row else None
                row["nonfinite_fields"] = [key for key, value in row.items() if isinstance(value, float) and not math.isfinite(value)]
                result["updates"].append(row)
                atomic_json(arm/"updates.json", result["updates"])
            result["fixed_data_update_wall_seconds"] = time.perf_counter()-started
            result["final_state"] = common.initial_fingerprints(runner)
            result["fixed_replay_unchanged"] = state_fingerprint(replay.data) == fixed_hash
            path = arm/f"model_{runner.iteration}.pt"
            runner.save(path, export=False)
            result.update(checkpoint=str(path), checkpoint_sha256=sha256(path), status="completed")
            if not result["fixed_replay_unchanged"] or any(sha256(Path(path)) != value for path, value in input_hashes.items()):
                raise RuntimeError("An input checkpoint/data file or fixed replay changed")
            atomic_json(args.output/"manifest.json", manifest)
            close_runner(runner)
            runner = None
            env.close()
            env = None
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[moe-comparison] {label}: {args.updates} paired fixed-data update attempts", flush=True)
        consistency = {}
        for name in ("world_model", "discriminator", "amp_norm"):
            hashes = {label: value["final_state"]["modules"][name] for label, value in manifest["results"].items()}
            consistency[name] = {"all_equal": len(set(hashes.values())) == 1, "hashes": hashes}
        for name in ("opt_model", "opt_disc"):
            hashes = {label: value["final_state"]["optimizers"][name] for label, value in manifest["results"].items()}
            consistency[name] = {"all_equal": len(set(hashes.values())) == 1, "hashes": hashes}
        manifest["independent_update_consistency"] = consistency
        if not args.no_final_evaluation:
            root = args.output/"evaluations"
            reference = evaluate_common(args.checkpoint, root/"source", args, source.recovery_height_reward)
            manifest["physical_source_evaluation"] = reference
            for label, record in manifest["results"].items():
                evaluated = evaluate_common(Path(record["checkpoint"]), root/label, args, source.recovery_height_reward)
                record["physical_evaluation"] = evaluated
                record["pairing_with_source"] = audit_common_protocol(reference, evaluated, source.recovery_height_reward)
                record["routing"] = {mode: routing_summary(value["routing_recording"]["path"])
                    for mode, value in evaluated["results"].items() if "routing_recording" in value}
                atomic_json(args.output/"manifest.json", manifest)
        manifest["inputs_unchanged_at_completion"] = all(sha256(Path(path)) == value for path, value in input_hashes.items())
        if not manifest["inputs_unchanged_at_completion"]:
            raise RuntimeError("An input changed during evaluation")
        manifest["status"] = "completed"
        atomic_json(args.output/"manifest.json", manifest)
        atomic_json(args.output/"results.json", {"scope": manifest["scope"], "results": manifest["results"],
            "initial_actions": manifest["initial_actions"], "independent_update_consistency": consistency,
            "source_evaluation": manifest.get("physical_source_evaluation"),
            "conclusion": "Short fixed-data intervention only; do not infer improved persistent recovery, convergence or speed."})
        print(f"Results: {args.output/'results.json'}")
    except BaseException as error:
        manifest.update(status="failed", error={"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()})
        atomic_json(args.output/"manifest.json", manifest)
        raise
    finally:
        if runner is not None:
            close_runner(runner)
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
