"""Probe finite-scale WMAC action directions above measured physical rollout noise.

An optional closed-loop Actor prelude selects a later physical starting state.
Each component then uses one fixed physical nominal open-loop action sequence. The
same smooth task/style objective is evaluated through the world model and real
simulation; no terminal Critic value is included. These are finite-scale local
response measurements, not an exact physical Jacobian validation.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import traceback
from zoneinfo import ZoneInfo

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
COMPONENTS = ("task", "style", "mixed")


def noise_summary(samples):
    """Per-environment empirical noise from [repeat, environment] returns."""
    values = np.asarray(samples, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("Return samples must have shape [at least two repeats, environments]")
    finite = np.isfinite(values).all(axis=0)
    safe = np.where(finite[None], values, 0.)
    result = {"mean": safe.mean(0), "std": safe.std(0, ddof=1),
              "range": np.ptp(safe, axis=0), "finite": finite,
              "nonfinite_count": (~np.isfinite(values)).sum(axis=0)}
    for key in ("mean", "std", "range"):
        result[key] = np.where(finite, result[key], np.nan)
    return result


def finite_scale_response(baseline, plus, minus, model_projection, epsilon,
                          noise_multiplier=3., absolute_noise_floor=1e-6,
                          gradient_floor=1e-12):
    """Classify a response only when its signal exceeds conservative noise.

    Noise is measured in return-difference units. The reference amplitude is
    the maximum of each observed return range and twice each sample standard
    deviation. The threshold is max(absolute floor, multiplier * amplitude).
    This empirical threshold is deliberately not a statistical confidence bound.
    """
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon must be finite and positive")
    if not math.isfinite(noise_multiplier) or noise_multiplier < 1:
        raise ValueError("noise_multiplier must be finite and at least one")
    if not math.isfinite(absolute_noise_floor) or absolute_noise_floor <= 0:
        raise ValueError("absolute_noise_floor must be finite and positive")
    stats = [noise_summary(values) for values in (baseline, plus, minus)]
    count = len(stats[0]["mean"])
    projection = np.asarray(model_projection, dtype=np.float64)
    if any(len(s["mean"]) != count for s in stats) or projection.shape != (count,):
        raise ValueError("All response inputs must share the environment dimension")
    finite = np.isfinite(projection) & np.logical_and.reduce([s["finite"] for s in stats])
    delta = stats[1]["mean"] - stats[2]["mean"]
    amplitude = np.maximum.reduce([value for s in stats for value in (s["range"], 2*s["std"])])
    threshold = np.maximum(absolute_noise_floor, noise_multiplier*amplitude)
    reference = np.maximum(amplitude, absolute_noise_floor/noise_multiplier)
    resolved = finite & (np.abs(projection) > gradient_floor) & (np.abs(delta) > threshold)
    rows = []
    for index in range(count):
        reason = ("nonfinite measurement" if not finite[index] else
                  "model direction has zero or negligible gradient" if abs(projection[index]) <= gradient_floor else
                  "finite-scale response does not exceed measured noise" if not resolved[index] else
                  "finite-scale response exceeds measured noise")
        rows.append({"environment": index, "resolved": bool(resolved[index]), "reason": reason,
            "model_projection": float(projection[index]),
            "model_linearized_return_difference": float(2*epsilon*projection[index]),
            "physical_return_difference": float(delta[index]),
            "physical_central_difference": float(delta[index]/(2*epsilon)),
            "noise_reference_amplitude": float(amplitude[index]),
            "required_return_difference": float(threshold[index]),
            "signal_to_noise_ratio": float(abs(delta[index])/reference[index]),
            "baseline_return_mean": float(stats[0]["mean"][index]),
            "baseline_return_std": float(stats[0]["std"][index]),
            "baseline_return_range": float(stats[0]["range"][index]),
            "plus_return_mean": float(stats[1]["mean"][index]),
            "minus_return_mean": float(stats[2]["mean"][index]),
            "plus_return_std": float(stats[1]["std"][index]),
            "minus_return_std": float(stats[2]["std"][index]),
            "sign_agrees": bool(projection[index]*delta[index] > 0) if resolved[index] else None})
    return rows


def summarize_responses(rows):
    resolved = [row for row in rows if row["resolved"]]
    agrees = sum(row["sign_agrees"] is True for row in resolved)
    return {"comparisons": len(rows), "resolved": len(resolved),
            "unresolved": len(rows)-len(resolved), "sign_agrees": agrees,
            "sign_disagrees": len(resolved)-agrees,
            "sign_agreement_fraction_resolved": agrees/len(resolved) if resolved else None,
            "interpretation": "finite-scale local response only; unresolved responses are not counted as agreement"}


def clean(value):
    if isinstance(value, dict):
        return {str(key): clean(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(child) for child in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def atomic_json(path, value):
    temporary = Path(path).with_suffix(".json.tmp")
    temporary.write_text(json.dumps(clean(value), indent=2, allow_nan=False)+"\n")
    temporary.replace(path)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for part in iter(lambda: handle.read(4 << 20), b""):
            digest.update(part)
    return digest.hexdigest()


def probe(args, output, report):
    import torch
    import warp as wp
    from mjlab.tasks.registry import load_env_cfg
    from scripts.check_wmac_physics import PhysicsSnapshot, load_checkpoint
    from scripts.evaluate_wmac import keep_fallen_recovery_frames, seed_all
    from src.wmac.env import CaptureEnv
    from src.wmac.networks import frozen_parameters
    from src.wmac.runner import differentiable_amp_reward
    from src.wmac.state import StateAdapter, quat_to_matrix
    from src.recovery_rewards import configure_recovery_height_reward

    modules = load_checkpoint(args.checkpoint, args.device)
    cfg, actor, model, unused_critic, discriminator, normalizer = modules
    for module in (actor, model):
        module.precision = "float32"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    env_cfg = load_env_cfg(cfg.task)
    configure_recovery_height_reward(env_cfg, cfg.recovery_height_reward)
    env_cfg.scene.num_envs, env_cfg.seed = args.num_envs, args.seed
    env_cfg.curriculum = {}
    env_cfg.events.pop("push_robot", None)
    for group in env_cfg.observations.values():
        group.enable_corruption = False
    init = env_cfg.events["init_motion_loader"].params
    init["delay_reset_env_ratio"] = 0. if args.mode == "walk" else 1.
    if args.mode == "recovery":
        init["max_delay_steps"] = max(int(init.get("max_delay_steps", 250)),
                                      args.prelude_steps+args.horizon+1)
    command = env_cfg.commands["twist"]
    command.rel_standing_envs = command.rel_heading_envs = 0.
    command.heading_command = False
    command.ranges.heading = None
    command.resampling_time_range = ((args.prelude_steps+args.horizon+1)*
                                    env_cfg.decimation*env_cfg.sim.mujoco.timestep+1.,)*2
    command.ranges.lin_vel_x = (args.command_vx,)*2
    command.ranges.lin_vel_y = (args.command_vy,)*2
    command.ranges.ang_vel_z = (args.command_yaw,)*2
    seed_all(args.seed)
    env = CaptureEnv(cfg=env_cfg, device=args.device)
    try:
        env.adapter = adapter = StateAdapter(env, recovery_counter_semantics=cfg.recovery_counter_semantics)
        candidates = keep_fallen_recovery_frames(env) if args.mode == "recovery" else None
        seed_all(args.seed)
        env.reset(seed=args.seed)
        # These changes are instance-local and affect only this disposable probe.
        env.command_manager.compute = lambda *unused_args, **unused_kwargs: None
        original_terminations = env.termination_manager.compute
        def no_reset():
            tm = env.termination_manager
            tm._terminated_buf.zero_()
            tm._truncated_buf.zero_()
            return tm._terminated_buf | tm._truncated_buf
        def prelude_no_reset():
            # Keep the real delayed-termination counters during the closed-loop
            # prelude, while suppressing automatic episode resets.
            original_terminations()
            return no_reset()
        env.termination_manager.compute = prelude_no_reset if args.prelude_steps else no_reset
        original_events = env.event_manager.apply
        def fixed_events(*positional, **keywords):
            mode = keywords.get("mode", positional[0] if positional else None)
            return None if mode in ("interval", "step") else original_events(*positional, **keywords)
        env.event_manager.apply = fixed_events
        if args.disable_cuda_graphs:
            env.sim.use_cuda_graph = False
        report["prelude"] = {"requested_steps": args.prelude_steps, "completed_steps": 0,
            "policy": "checkpoint Actor with physical closed-loop observations",
            "delay_counters": "real termination-manager counters during prelude; frozen during direction rollouts"}
        with torch.no_grad():
            for _ in range(args.prelude_steps):
                env.step(actor(env.obs_buf["actor"]))
                report["prelude"]["completed_steps"] += 1
        env.termination_manager.compute = no_reset
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        wp.synchronize()
        snapshot = PhysicsSnapshot(env)
        up = quat_to_matrix(snapshot.state[:, 3:7])[:, 2, 2]
        phase = (snapshot.state[:, 2] < .55) | (up < .5)
        initial = {"state": snapshot.state.cpu().numpy(), "context": snapshot.context.cpu().numpy(),
                   "command": snapshot.command.cpu().numpy(),
                   "actor_history": snapshot.obs["actor"].cpu().numpy()}
        np.savez_compressed(output / "initial.npz", **initial)
        digest = hashlib.sha256()
        for key in sorted(initial):
            digest.update(key.encode())
            digest.update(np.ascontiguousarray(initial[key]).tobytes())
        report["initial_arrays_sha256"] = digest.hexdigest()
        report["initial_arrays_path"] = str(output / "initial.npz")
        report["environments"] = [{"environment": index, "start_height": float(snapshot.state[index, 2]),
            "start_up_z": float(up[index]), "phase": "recovery" if phase[index] else "upright",
            "requested_mode": args.mode, "prelude_steps": args.prelude_steps,
            "snapshot_delay_enabled": bool(snapshot.context[index, 33]),
            "snapshot_delay_counter": int(snapshot.context[index, 34]),
            "snapshot_episode_length_steps": int(snapshot.episode_length[index])}
            for index in range(env.num_envs)]
        report["protocol"].update({"task": cfg.task, "amp_feature_schema": cfg.amp_feature_schema,
            "recovery_counter_semantics": cfg.recovery_counter_semantics,
            "recovery_height_reward": cfg.recovery_height_reward,
            "control_dt": adapter.dt, "recovery_start_candidates": candidates,
            "cuda_graphs_used": env.sim.use_cuda_graph,
            "smooth_delay_max_steps": adapter.max_delay_steps,
            "task_weight": cfg.task_weight, "style_weight": cfg.style_weight,
            "snapshot_warp_arrays": len(snapshot.warp_arrays), "snapshot_torch_tensors": len(snapshot.tensors)})
        with torch.no_grad():
            snapshot.restore()
            actions = []
            for _ in range(args.horizon):
                action = actor(env.obs_buf["actor"])
                actions.append(action.detach().clone())
                env.step(action)
            nominal = torch.stack(actions)

        def objective(actions, physical):
            if physical:
                snapshot.restore()
            state, context = snapshot.state.clone(), snapshot.context.clone()
            previous, command = snapshot.previous_action.clone(), snapshot.command
            survival = state.new_ones(env.num_envs)
            task_sum, style_sum = torch.zeros_like(survival), torch.zeros_like(survival)
            trajectory = []
            for index, action in enumerate(actions):
                if physical:
                    env.step(action)
                    following = env.transition["next_state"]
                    contact = env.transition["contact_cost"]
                else:
                    following, contact = model.step_with_contact(state, action, context)
                next_context, terminated = adapter.advance_context(following, context)
                task = adapter.task_reward(state, following, action, previous, command,
                                           next_context, terminated, contact)
                style, _ = differentiable_amp_reward(discriminator, normalizer,
                    adapter.amp_features(state), adapter.amp_features(following), cfg.amp_reward_coef)
                weight = cfg.gamma**index*survival
                task_sum = task_sum+weight*task
                style_sum = style_sum+weight*style
                survival = survival*(1-terminated)
                state, context, previous = following, next_context, action
                trajectory.append(following.detach().clone())
            return {"task": task_sum, "style": style_sum,
                    "mixed": cfg.task_weight*task_sum+cfg.style_weight*style_sum}, torch.stack(trajectory)

        baseline, trajectories = {key: [] for key in COMPONENTS}, []
        for repeat in range(args.baseline_repeats):
            with torch.no_grad():
                values, state_trajectory = objective(nominal, physical=True)
            for key in COMPONENTS:
                baseline[key].append(values[key].cpu().numpy())
            trajectories.append(state_trajectory.cpu().numpy())
            print(f"[physical probe] baseline {repeat+1}/{args.baseline_repeats}", flush=True)
        baseline = {key: np.stack(values) for key, values in baseline.items()}
        trajectories = np.stack(trajectories)
        baseline_state_range = np.ptp(trajectories.astype(np.float64), axis=0)
        slices = {"root_position_m": slice(0, 3), "root_quaternion": slice(3, 7),
                  "root_linear_velocity_m_s": slice(7, 10), "root_angular_velocity_rad_s": slice(10, 13),
                  "joint_position_rad": slice(13, 42), "joint_velocity_rad_s": slice(42, 71)}
        report["baseline"] = {"repeats": args.baseline_repeats,
            "returns": {key: noise_summary(value) for key, value in baseline.items()},
            "trajectory_max_range_per_environment": baseline_state_range.max(axis=(0, 2)),
            "trajectory_max_range_by_state_group": {key: baseline_state_range[..., selection].max(axis=(0, 2))
                                                    for key, selection in slices.items()}}
        np.savez_compressed(output / "baseline.npz", nominal_actions=nominal.cpu().numpy(),
                            states=trajectories, **baseline)
        report["components"] = {}
        for key in COMPONENTS:
            candidate = nominal.detach().clone().requires_grad_(True)
            with frozen_parameters(model, discriminator, normalizer):
                prediction, _ = objective(candidate, physical=False)
                gradient = torch.autograd.grad(prediction[key].sum(), candidate)[0].detach()
            norms = gradient.square().sum((0, 2)).sqrt()
            finite = torch.isfinite(gradient).all(dim=0).all(dim=-1)
            safe = torch.where(finite[None, :, None], gradient, torch.zeros_like(gradient))
            direction = safe / safe.square().sum((0, 2), keepdim=True).sqrt().clamp_min(1e-12)
            projected = (safe*direction).sum((0, 2)).cpu().numpy()
            projected[~finite.cpu().numpy()] = np.nan
            record = {"model_gradient_norm": norms.cpu().numpy(), "model_projection": projected,
                      "direction": "per-environment unit L2 model action gradient over the fixed open-loop sequence",
                      "epsilons": {}}
            for epsilon in args.epsilons:
                measurements = {"plus": [], "minus": []}
                for repeat in range(args.perturbation_repeats):
                    for sign, multiplier in (("plus", 1.), ("minus", -1.)):
                        with torch.no_grad():
                            response, _ = objective(nominal+multiplier*epsilon*direction, physical=True)
                        measurements[sign].append(response[key].cpu().numpy())
                plus, minus = np.stack(measurements["plus"]), np.stack(measurements["minus"])
                rows = finite_scale_response(baseline[key], plus, minus, projected, epsilon,
                                            args.noise_multiplier, args.absolute_noise_floor)
                for row, initial_metadata in zip(rows, report["environments"]):
                    row.update(initial_metadata)
                record["epsilons"][str(epsilon)] = {"epsilon": epsilon, "plus_returns": plus,
                    "minus_returns": minus, "rows": rows, "summary": summarize_responses(rows)}
                report["components"][key] = record
                atomic_json(output / "report.json", report)
                print(f"[physical probe] {key} epsilon={epsilon}: {summarize_responses(rows)}", flush=True)
            cross_scale = []
            for index in range(env.num_envs):
                resolved = [entry["rows"][index] for entry in record["epsilons"].values()
                            if entry["rows"][index]["resolved"]]
                signs = [row["sign_agrees"] for row in resolved]
                cross_scale.append({"environment": index, "resolved_epsilon_count": len(signs),
                    "consistent_sign_across_resolved_epsilons": len(set(signs)) == 1 if len(signs) >= 2 else None,
                    "all_resolved_responses_agree_with_model": all(signs) if signs else None,
                    "interpretation": "unresolved" if not signs else "requires more resolved scales" if len(signs) < 2
                        else "scale-sensitive" if len(set(signs)) > 1 else "consistent finite-scale sign"})
            record["cross_scale"] = cross_scale
        snapshot.restore()
        report["status"] = "completed"
        return report
    finally:
        env.close()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=("walk", "recovery"), default="recovery")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=4)
    parser.add_argument("--prelude-steps", type=int, default=0,
                        help="Physical closed-loop Actor steps before taking the probe snapshot (0..128)")
    parser.add_argument("--baseline-repeats", type=int, default=5)
    parser.add_argument("--perturbation-repeats", type=int, default=2)
    parser.add_argument("--epsilons", nargs="+", type=float, default=[.01, .03, .1])
    parser.add_argument("--noise-multiplier", type=float, default=3.)
    parser.add_argument("--absolute-noise-floor", type=float, default=1e-6)
    parser.add_argument("--command-vx", type=float, default=.5)
    parser.add_argument("--command-vy", type=float, default=0.)
    parser.add_argument("--command-yaw", type=float, default=0.)
    parser.add_argument("--disable-cuda-graphs", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.num_envs <= 32 or not 1 <= args.horizon <= 32:
        parser.error("Use 1..32 environments and 1..32 control steps")
    if not 0 <= args.prelude_steps <= 128:
        parser.error("Use 0..128 prelude control steps")
    if args.baseline_repeats < 5 or args.perturbation_repeats < 2:
        parser.error("Use at least five baseline and two plus/minus repeats")
    if any(not math.isfinite(value) or value <= 0 for value in args.epsilons):
        parser.error("All epsilons must be finite and positive")
    if args.noise_multiplier < 1 or not math.isfinite(args.noise_multiplier):
        parser.error("noise multiplier must be finite and at least one")
    if args.absolute_noise_floor <= 0 or not math.isfinite(args.absolute_noise_floor):
        parser.error("absolute noise floor must be finite and positive")
    if not all(math.isfinite(getattr(args, key)) for key in ("command_vx", "command_vy", "command_yaw")):
        parser.error("Velocity commands must be finite")
    args.epsilons = sorted(set(args.epsilons))
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    source_paths = [Path(__file__), REPO / "scripts/check_wmac_physics.py", REPO / "scripts/evaluate_wmac.py",
                    REPO / "src/recovery_rewards.py", REPO / "src/tasks/amp_loco/mdp/rewards.py",
                    *sorted((REPO / "src/wmac").glob("*.py"))]
    report = {"status": "running", "created_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "arguments": vars(args), "checkpoint_sha256": file_hash(args.checkpoint),
        "source_sha256": {str(path.relative_to(REPO)): file_hash(path) for path in source_paths},
        "protocol": {"critic_included": False, "pushes": False, "observation_noise": False,
            "commands_frozen": True, "physical_resets_disabled": True, "interval_and_step_events_disabled": True,
            "prelude_steps": args.prelude_steps,
            "snapshot_timing": "after physical closed-loop Actor prelude; before nominal open-loop actions",
            "action_sequence": "physical nominal Actor sequence generated once, then fixed open-loop",
            "objectives": {"task": "discounted configured-survival-weighted task proxy, unweighted",
                "style": "discounted configured-survival-weighted AMP reward, unweighted",
                "mixed": "checkpoint task_weight * task + style_weight * style; no terminal Critic"},
            "noise_threshold": "max(absolute_floor, noise_multiplier * max(observed baseline/plus/minus ranges and twice sample standard deviations))",
            "maximum_environment_step_calls": args.prelude_steps+args.horizon*(1+args.baseline_repeats+
                len(COMPONENTS)*len(args.epsilons)*2*args.perturbation_repeats)},
        "limitations": ["Finite-scale response is not an exact Jacobian validation.",
            "The empirical noise threshold is not a statistical confidence interval.",
            "Identical restored state does not guarantee bitwise deterministic GPU contact solving.",
            "The same configured task/delay/survival proxy is evaluated in both branches; a surrogate backward rule is not an exact physical termination derivative.",
            "Resolved agreement at short horizons does not establish full recovery skill or long-horizon stability."]}
    atomic_json(output / "report.json", report)
    try:
        probe(args, output, report)
    except BaseException as error:
        report["status"] = "failed"
        report["error"] = {"type": type(error).__name__, "message": str(error), "traceback": traceback.format_exc()}
        raise
    finally:
        atomic_json(output / "report.json", report)
    print(f"Physical direction probe: {output / 'report.json'}")


if __name__ == "__main__":
    main()
