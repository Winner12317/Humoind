"""Compare two policies from the same complete physical starting snapshot.

The only policy inputs are physical observations; no world model controls the
simulator. Optional repeats of both policies measure rollout repeatability
separately from the A/B difference. Old PPO clipping must be supplied explicitly if its
original run overrode the default and the checkpoint lacks that metadata.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import types

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def content_hash(arrays):
    import numpy as np

    digest = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        value = np.ascontiguousarray(value)
        digest.update(name.encode())
        digest.update(str(value.dtype).encode())
        digest.update(str(value.shape).encode())
        digest.update(value.tobytes())
    return digest.hexdigest()


def clean(value):
    import numpy as np

    if isinstance(value, dict):
        return {str(key): clean(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(item) for item in value]
    if isinstance(value, np.ndarray):
        return clean(value.tolist())
    if isinstance(value, np.generic):
        return clean(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def magnitude_stats(values):
    import numpy as np

    values = np.abs(np.asarray(values, dtype=np.float64)).reshape(-1)
    finite = np.isfinite(values)
    usable = values[finite]
    result = {"samples": int(values.size), "nonfinite": int((~finite).sum())}
    if not usable.size:
        return {**result, "abs_mean": None, "rms": None,
                "p95": None, "p99": None, "max": None}
    return {**result, "abs_mean": float(usable.mean()),
            "rms": float(np.sqrt(np.mean(usable * usable))),
            "p95": float(np.quantile(usable, .95)),
            "p99": float(np.quantile(usable, .99)), "max": float(usable.max())}


def summarize(arrays, control_mask, decimation):
    import numpy as np

    substep_mask = np.repeat(control_mask, decimation, axis=0)
    difference_mask = control_mask.copy()
    # The startup jump from reset action zero is reported separately.
    difference_mask[0] = False
    requested = arrays["request"][substep_mask]
    clipped = arrays["clipped"][substep_mask]
    last_qacc = arrays["qacc"][decimation - 1::decimation][control_mask]
    return {
        "environment_control_steps": int(control_mask.sum()),
        "action": magnitude_stats(arrays["action"][control_mask]),
        "action_delta_excluding_first_step": magnitude_stats(arrays["action_delta"][difference_mask]),
        "actuator_force_Nm": magnitude_stats(arrays["force"][substep_mask]),
        "physical_substep_qacc_rad_s2": magnitude_stats(arrays["qacc"][substep_mask]),
        "last_substep_qacc_squared_sum_mean": (
            float(np.square(last_qacc.astype(np.float64)).sum(-1).mean())
            if last_qacc.size else None),
        "custom_clip_saturation_fraction": (
            float((np.abs(requested - clipped) > 1e-5).mean()) if requested.size else None),
        "joint_velocity_rad_s": magnitude_stats(arrays["next_state"][..., 42:71][control_mask]),
    }


def joint_summaries(arrays, control_mask, decimation, names):
    import numpy as np

    mask = np.repeat(control_mask, decimation, axis=0)
    result = {}
    for joint, name in enumerate(names):
        force = arrays["force"][:, :, joint][mask]
        request = arrays["request"][:, :, joint][mask]
        clipped = arrays["clipped"][:, :, joint][mask]
        result[name] = {
            "actuator_force_Nm": magnitude_stats(force),
            "custom_clip_saturation_fraction": (float((np.abs(request - clipped) > 1e-5).mean())
                                                  if request.size else None),
        }
    return result


def paired_cost_differences(a, b, mask):
    """Per-start paired scalar costs; no confidence/significance claim."""
    import numpy as np

    differences = []
    for environment in range(mask.shape[1]):
        chosen = mask[:, environment]
        if not chosen.any():
            differences.append(None)
            continue
        delta = b["qacc_cost"][chosen, environment] - a["qacc_cost"][chosen, environment]
        differences.append(float(np.mean(delta)))
    usable = [value for value in differences if value is not None and math.isfinite(value)]
    return {"qacc_cost_mean_B_minus_A_per_environment": differences,
            "mean": float(np.mean(usable)) if usable else None,
            "median": float(np.median(usable)) if usable else None,
            "environments_with_B_lower": sum(value < 0 for value in usable),
            "environments_with_B_higher": sum(value > 0 for value in usable),
            "environments_compared": len(usable)}


def physical_model_arrays(env):
    import numpy as np
    import torch
    from mjlab.sim.sim_data import TorchArray

    arrays = {}
    for name in ("geom_friction", "body_ipos", "body_mass", "body_inertia",
                 "dof_damping", "dof_armature", "actuator_gainprm",
                 "actuator_biasprm", "actuator_forcerange", "jnt_range"):
        value = getattr(env.sim.model, name, None)
        if isinstance(value, (torch.Tensor, TorchArray)):
            arrays[name] = value.detach().cpu().numpy().copy()
        elif isinstance(value, np.ndarray):
            arrays[name] = value.copy()
    if len(arrays) < 5:
        raise RuntimeError("Could not record enough randomized physical parameters")
    arrays["encoder_bias"] = env.scene["robot"].data.encoder_bias.detach().cpu().numpy().copy()
    return arrays


class EffortRecorder:
    """Instance-only hooks that return the original motor/physics results."""

    def __init__(self, env):
        import torch

        self.env, self.robot = env, env.scene["robot"]
        self.enabled, self.pending, self.rows = False, {}, []
        self.groups = []
        self._originals = []
        covered = []
        for key, actuator in enumerate(self.robot.actuators):
            if not hasattr(actuator, "_clip_effort"):
                raise ValueError("This probe requires the G1 Unitree custom actuators")
            ids, controls = actuator.target_ids.long(), actuator.ctrl_ids.long()
            covered.extend(ids.detach().cpu().tolist())
            self.groups.append((key, ids, controls))
            original = actuator._clip_effort
            self._originals.append((actuator, original))

            def clip(owner, effort, original=original, key=key):
                clipped = original(effort)
                if self.enabled:
                    same = owner._joint_vel * effort > 0
                    bound = torch.where(same, owner._effort_y1, owner._effort_y2)
                    bound = bound.clamp_max(float(owner.cfg.effort_limit))
                    bound = torch.where(owner._joint_vel.abs() < owner._velocity_x1,
                                        bound, owner._compute_effort_limit(bound))
                    self.pending[key] = tuple(value.detach().clone()
                                              for value in (effort, clipped, bound))
                return clipped

            actuator._clip_effort = types.MethodType(clip, actuator)
        if sorted(covered) != list(range(29)):
            raise ValueError("Actuator/joint mapping is not a complete 29-joint mapping")
        self._original_step = env.sim.step

        def step():
            self._original_step()
            if not self.enabled:
                return
            if len(self.pending) != len(self.groups):
                raise RuntimeError("Missing motor samples before a physical step")
            source_force = self.robot.data.actuator_force
            row = {name: torch.zeros(env.num_envs, 29, device=env.device)
                   for name in ("request", "clipped", "bound", "force")}
            row["qacc"] = self.robot.data.joint_acc.detach().clone()
            for key, ids, controls in self.groups:
                for name, value in zip(("request", "clipped", "bound"), self.pending[key]):
                    row[name][:, ids] = value
                row["force"][:, ids] = source_force[:, controls]
            self.rows.append(row)
            self.pending.clear()

        env.sim.step = step

    def begin(self):
        self.pending.clear()
        self.rows.clear()
        self.enabled = True

    def end(self):
        self.enabled = False
        self.pending.clear()

    def close(self):
        self.end()
        for actuator, original in self._originals:
            actuator._clip_effort = original
        self.env.sim.step = self._original_step


def capture_feet_contact_at_reward_clock(env):
    """Capture per-foot contacts before any reset can replace the sensor data."""
    sensor = env.scene['feet_ground_contact']
    compute = env.metrics_manager.compute

    def compute_and_capture(*args, **kwargs):
        result = compute(*args, **kwargs)
        if env._capture_enabled:
            env._physical_end['feet_contact'] = (sensor.data.found > 0).detach().clone()
        return result

    env.metrics_manager.compute = compute_and_capture


def rollout(env, actor, recorder, steps):
    import numpy as np
    import torch
    from src.wmac.state import heading_quat, quat_apply, quat_to_matrix

    adapter, n = env.adapter, env.num_envs
    active = torch.ones(n, dtype=torch.bool, device=env.device)
    stable = torch.zeros(n, dtype=torch.long, device=env.device)
    walking = torch.zeros_like(stable)
    recovered, continued = torch.zeros_like(active), torch.zeros_like(active)
    recovery_time = torch.full((n,), float("nan"), device=env.device)
    observations = env.obs_buf
    records = []
    recorder.begin()
    started = time.perf_counter()
    try:
        with torch.no_grad():
            for step in range(steps):
                state, context = adapter.state().clone(), adapter.context().clone()
                previous = env.action_manager.action.clone()
                history = observations["actor"].clone()
                raw = actor(history)
                limit = getattr(actor, "source_action_limit", None)
                action = raw.clamp(-limit, limit) if limit is not None else raw
                observations, *_ = env.step(action)
                transition = env.transition
                next_state = transition["next_state"]
                done = transition["terminated"] | transition["timeout"]
                up_z = quat_to_matrix(next_state[:, 3:7])[:, 2, 2]
                upright = (next_state[:, 2] > .65) & (up_z > .8)
                stable = torch.where(active & upright & ~done, stable + 1, torch.zeros_like(stable))
                new_success = active & ~recovered & (stable >= 25)
                recovery_time[new_success] = (step + 1) * env.step_dt
                recovered |= new_success
                _, rotations, velocities, angular = adapter.kinematics(next_state)
                torso = adapter.anchor_id
                command = transition["command"]
                target = quat_apply(heading_quat(rotations[:, torso]),
                                    torch.cat((command[:, :2], torch.zeros_like(command[:, :1])), -1))
                xy_error = (target[:, :2] - velocities[:, torso, :2]).norm(dim=-1)
                yaw_error = (command[:, 2] - angular[:, torso, 2]).abs()
                good_walk = (active & recovered & upright & ~done
                             & (command[:, :2].norm(dim=-1) > .1)
                             & (xy_error < .5) & (yaw_error < .5))
                walking = torch.where(good_walk, walking + 1, torch.zeros_like(walking))
                continued |= walking >= 50
                row = {name: transition[name].detach().clone() for name in (
                    "state", "next_state", "context", "next_context", "previous_action",
                    "command", "reward", "terminated", "timeout", "boundary", "contact_cost")}
                if "feet_contact" in transition:
                    row["feet_contact"] = transition["feet_contact"].detach().clone()
                row.update(raw_action=raw.detach().clone(), action=action.detach().clone(),
                           action_delta=(action - previous).detach().clone(), actor_history=history,
                           valid=active.clone(), upright=upright.clone(),
                           recovery_phase=((state[:, 2] < .55) | (context[:, -1] > 0)),
                           xy_error=xy_error, yaw_error=yaw_error)
                records.append(row)
                active &= ~done
    finally:
        recorder.end()
    torch.cuda.synchronize() if str(env.device).startswith("cuda") else None
    elapsed = time.perf_counter() - started
    if len(recorder.rows) != steps * env.cfg.decimation:
        raise RuntimeError("Physical substep count does not match requested control steps")
    arrays = {name: torch.stack([row[name] for row in records]).cpu().numpy() for name in records[0]}
    arrays.update({name: torch.stack([row[name] for row in recorder.rows]).cpu().numpy()
                   for name in recorder.rows[0]})
    arrays["qacc_cost"] = np.square(arrays["qacc"][env.cfg.decimation - 1::env.cfg.decimation]
                                   .astype(np.float64)).sum(-1)
    result = {"recovery_success_count": int(recovered.sum().item()),
              "continued_walk_count": int(continued.sum().item()),
              "first_episode_survival_count": int(active.sum().item()),
              "first_stable_recovery_seconds": recovery_time.cpu().tolist(),
              "first_stable_recovery_seconds_mean": (float(recovery_time[recovered].mean().item())
                                                      if recovered.any() else None),
              "rollout_wall_seconds_not_a_speed_benchmark": elapsed}
    return arrays, result


def compare(args, output):
    import numpy as np
    import torch
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg
    from scripts.check_wmac_physics import PhysicsSnapshot
    from scripts.evaluate_wmac import keep_fallen_recovery_frames, load_actor, seed_all
    from src.recovery_rewards import configure_recovery_height_reward
    from src.wmac.env import CaptureEnv
    from src.wmac.state import StateAdapter

    actors, provenance = {}, {}
    for label, path, override in (("a", args.checkpoint_a, args.action_limit_a),
                                   ("b", args.checkpoint_b, args.action_limit_b)):
        actor, kind = load_actor(path, args.device)
        actor.precision = "float32"
        if override is not None:
            actor.source_action_limit = override
        actors[label] = actor
        provenance[label] = {
            "path": str(path), "sha256": file_hash(path), "kind": kind,
            "policy_state_sha256": content_hash({
                name: tensor.detach().cpu().numpy()
                for name, tensor in actor.state_dict().items()}),
            "architecture": getattr(actor, "source_actor_architecture", None),
            "outer_action_limit": getattr(actor, "source_action_limit", None),
            "action_limit_explicit_override": override,
            "ppo_clip_metadata_caveat": ("Original run params were not checked; default None unless explicitly overridden"
                                          if kind == "ppo-amp-baseline" else None)}
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = load_env_cfg("Unitree-G1-AMP-Flat-Aligned")
    configure_recovery_height_reward(cfg, "upright_v2")
    cfg.scene.num_envs, cfg.seed, cfg.curriculum = args.num_envs, args.seed, {}
    cfg.events.pop("push_robot", None)
    for group in cfg.observations.values():
        group.enable_corruption = False
    cfg.events["init_motion_loader"].params.update(
        delay_reset_env_ratio=1. if args.mode == "recovery" else 0., max_delay_steps=250)
    command = cfg.commands["twist"]
    command.rel_standing_envs = command.rel_heading_envs = 0.
    command.heading_command = False
    command.ranges.heading = None
    command.resampling_time_range = (1e9, 1e9)
    command.ranges.lin_vel_x, command.ranges.lin_vel_y, command.ranges.ang_vel_z = (.5, .5), (0., 0.), (0., 0.)
    seed_all(args.seed)
    env = CaptureEnv(cfg=cfg, device=args.device)
    recorder = None
    try:
        env.adapter = StateAdapter(env)
        if args.record_foot_contacts:
            capture_feet_contact_at_reward_clock(env)
        candidates = keep_fallen_recovery_frames(env) if args.mode == "recovery" else None
        seed_all(args.seed)
        env.reset(seed=args.seed)
        initial = {name: value.detach().cpu().numpy().copy() for name, value in {
            "state": env.adapter.state(), "context": env.adapter.context(),
            "command": env.adapter.command(), "actor_history": env.obs_buf["actor"],
            "previous_action": env.action_manager.action}.items()}
        initial_hash = content_hash(initial)
        model_arrays = physical_model_arrays(env)
        model_hash = content_hash(model_arrays)
        np.savez_compressed(output / "initial_state.npz", **initial)
        np.savez_compressed(output / "physical_model.npz", **model_arrays)
        snapshot = PhysicsSnapshot(env)
        recorder = EffortRecorder(env)
        all_arrays, summaries = {}, {}
        sequence = [("a", "a"), ("a_repeat", "a"), ("b", "b"), ("b_repeat", "b")]
        if args.no_repeatability:
            sequence = [("a", "a"), ("b", "b")]
        protocol_checks = []
        for trial, policy in sequence:
            snapshot.restore()
            current = {name: value.detach().cpu().numpy().copy() for name, value in {
                "state": env.adapter.state(), "context": env.adapter.context(),
                "command": env.adapter.command(), "actor_history": env.obs_buf["actor"],
                "previous_action": env.action_manager.action}.items()}
            if content_hash(current) != initial_hash:
                raise RuntimeError(f"Initial observations/context differ for {trial}")
            if content_hash(physical_model_arrays(env)) != model_hash:
                raise RuntimeError(f"Physical parameters differ before {trial}")
            arrays, summary = rollout(env, actors[policy], recorder, args.steps)
            if content_hash(physical_model_arrays(env)) != model_hash:
                raise RuntimeError("A reset changed model parameters; snapshot protocol must be extended")
            np.savez_compressed(output / f"transitions_{trial}.npz", **arrays)
            all_arrays[trial], summaries[trial] = arrays, summary
            protocol_checks.append({"trial": trial, "initial_state_sha256": initial_hash,
                                    "physical_model_sha256": model_hash})
        a, b = all_arrays["a"], all_arrays["b"]
        common = a["valid"] & b["valid"]
        start_times = np.arange(args.steps) * env.step_dt
        windows = {"all": np.ones(args.steps, dtype=bool),
                   "startup_0_to_0p1_seconds": start_times < .1,
                   "recovery_0p1_to_2_seconds": (start_times >= .1) & (start_times < 2.),
                   "followthrough_2_to_5_seconds": (start_times >= 2.) & (start_times < 5.)}
        common_summaries = {}
        for label, arrays in all_arrays.items():
            summaries[label]["own_first_episode"] = summarize(arrays, arrays["valid"], cfg.decimation)
            summaries[label]["own_recovery_phase_secondary"] = summarize(
                arrays, arrays["valid"] & arrays["recovery_phase"], cfg.decimation)
            summaries[label]["startup_action_jump_from_reset"] = magnitude_stats(arrays["action_delta"][0])
        for name, window in windows.items():
            mask = common & window[:, None]
            common_summaries[name] = {"a": summarize(a, mask, cfg.decimation),
                                      "b": summarize(b, mask, cfg.decimation),
                                      "paired_qacc_cost": paired_cost_differences(a, b, mask),
                                      "per_joint_a": joint_summaries(a, mask, cfg.decimation,
                                                                     env.scene["robot"].joint_names),
                                      "per_joint_b": joint_summaries(b, mask, cfg.decimation,
                                                                     env.scene["robot"].joint_names)}
        repeats = {}
        for label in ("a", "b"):
            if f"{label}_repeat" not in all_arrays:
                continue
            first, repeat = all_arrays[label], all_arrays[f"{label}_repeat"]
            valid = first["valid"] & repeat["valid"]
            substep_valid = np.repeat(valid, cfg.decimation, axis=0)
            repeats[label] = {
                name: magnitude_stats((first[name].astype(np.float64)
                                      - repeat[name].astype(np.float64))[
                    substep_valid if name in ("force", "qacc") else valid])
                for name in ("action", "next_state", "force", "qacc")}
            repeats[label]["qacc_cost_by_fixed_window"] = {
                name: paired_cost_differences(first, repeat, valid & window[:, None])
                for name, window in windows.items()}
        identical_policy = (
            provenance["a"]["policy_state_sha256"] == provenance["b"]["policy_state_sha256"]
            and provenance["a"]["outer_action_limit"] == provenance["b"]["outer_action_limit"]
            and provenance["a"]["architecture"] == provenance["b"]["architecture"])
        report = {
            "schema": "amp-wmac-paired-effort-v1", "checkpoints": provenance,
            "protocol": {"mode": args.mode, "num_envs": args.num_envs, "steps": args.steps,
                         "seed": args.seed, "control_dt": env.step_dt, "physics_substeps": cfg.decimation,
                         "task": "Unitree-G1-AMP-Flat-Aligned", "recovery_delay_steps": 250,
                         "physical_reward": "upright_v2", "command": [.5, 0., 0.],
                         "push": False, "observation_noise": False, "exploration_noise": False,
                         "world_model_used_for_control": False, "network_parameters_updated": False,
                         "real_terminations_and_resets_retained": True, "recovery_candidates": candidates,
                         "initial_state_sha256": initial_hash, "physical_model_sha256": model_hash},
            "snapshot_restore_checks": protocol_checks, "trials": summaries,
            "common_first_episode_windows": common_summaries,
            "a_repeatability": repeats.get("a"), "b_repeatability": repeats.get("b"),
            "comparison_status": ("identical_policy_self_control_only" if identical_policy
                                  else "descriptive_paired_measurements_inspect_both_repeatability_records"),
            "joint_names": list(env.scene["robot"].joint_names),
            "limitations": [
                "Same-snapshot pairing does not guarantee bitwise deterministic GPU contacts; inspect both policy repeats.",
                "An A/B delta is not evidence of improvement by itself; both policy repeats measure contact/rollout variability.",
                "Identical policy states are a tool self-control, never a PPO comparison or optimization effect.",
                "Custom clipping is measured before motor friction; actual actuator_force is measured after physics.",
                "Own recovery-phase masks can differ in duration; primary effort comparison uses fixed windows and common-valid steps.",
                "Startup action jump is separate from continued action differences.",
                "A missing historical checkpoint or unknown historical CLI clipping cannot be replaced by another policy.",
                "These short trials do not establish training convergence or the causal reason for historical learning differences."]}
        (output / "metrics.json").write_text(json.dumps(clean(report), ensure_ascii=False, indent=2, allow_nan=False) + "\n")
        return report
    finally:
        if recorder is not None:
            recorder.close()
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=32)
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("walk", "recovery"), default="recovery")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--action-limit-a", type=float)
    parser.add_argument("--action-limit-b", type=float)
    parser.add_argument("--no-repeatability", action="store_true")
    parser.add_argument("--record-foot-contacts", action="store_true",
                        help="Record each foot's physical ground contact before resets")
    args = parser.parse_args()
    if args.num_envs < 1 or args.steps < 1:
        parser.error("--num-envs and --steps must be positive")
    for name in ("action_limit_a", "action_limit_b"):
        limit = getattr(args, name)
        if limit is not None and (not math.isfinite(limit) or limit <= 0):
            parser.error("Explicit action limits must be finite and positive")
    for name in ("checkpoint_a", "checkpoint_b"):
        value = getattr(args, name).expanduser().resolve()
        if not value.is_file():
            parser.error(f"Checkpoint not found: {value}")
        setattr(args, name, value)
    args.output = args.output.expanduser().resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    compare(args, args.output)
    print(f"Paired effort metrics: {args.output / 'metrics.json'}", flush=True)


if __name__ == "__main__":
    main()
