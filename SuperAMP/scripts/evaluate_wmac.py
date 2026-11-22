"""Evaluate WMAC and baseline AMP checkpoints in independent physical trials."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def evaluation_source_metadata(actor):
    """Describe training provenance without changing the physical protocol.

    Policies trained with either recovery proxy are evaluated by the original
    physical DelayedTerminationManager. Training semantics must not participate
    in the signature used to pair physical trials.
    """
    semantics = getattr(actor, "source_recovery_counter_semantics", "unknown")
    height_reward = getattr(actor, "source_recovery_height_reward", "unknown")
    architecture = getattr(actor, "source_actor_architecture", None)
    actor_type = getattr(actor, "source_actor_type", "unknown")
    smooth = getattr(actor, "source_smooth_reward_protocol", {"smooth_reward_mode": "none"})
    return {
        "checkpoint_actor_type": actor_type, "checkpoint_actor_architecture": architecture,
        "checkpoint_training_recovery_semantics": semantics,
        "checkpoint_training_recovery_height_reward": height_reward,
        "checkpoint_training_smooth_reward_protocol": smooth,
        "evaluation_task_reward_scope": "baseline_physical_task_without_training_smooth_terms",
        "source_provenance": {
            "checkpoint_actor_type": actor_type, "checkpoint_actor_architecture": architecture,
            "checkpoint_training_recovery_semantics": semantics,
            "checkpoint_training_recovery_height_reward": height_reward,
            "checkpoint_training_smooth_reward_protocol": smooth,
            "physical_delay_rule": "original_DelayedTerminationManager",
            "training_recovery_proxy_used_during_physical_evaluation": False,
            "training_smooth_terms_used_during_physical_evaluation": False,
        },
    }


class EvaluationRecorder:
    """Optional CPU snapshots of physical transitions and first-episode traces.

    Arrays retain the real pre-reset successor even on terminal steps. The
    ``first_episode_valid`` mask includes that terminal transition and excludes
    all transitions of later automatically reset episodes. These records are
    observations of a rollout, not complete simulator restart snapshots.
    """

    TRANSITION_FIELDS = (
        "state", "next_state", "action", "previous_action", "context",
        "next_context", "command", "reward", "terminated", "timeout",
        "boundary", "actor_history", "contact_cost",
    )
    TRACE_FIELDS = (
        "root_height", "upright_cos", "velocity_xy_error", "velocity_yaw_error",
        "action_rms", "recovered", "terminated", "timeout",
    )

    def __init__(self, num_envs, max_steps, *, save_transitions=False,
                 save_trace=False, protocol=None, source_provenance=None):
        if num_envs < 1 or max_steps < 1:
            raise ValueError("Recording dimensions must be positive")
        self.num_envs, self.max_steps = num_envs, max_steps
        self.save_transitions, self.save_trace = save_transitions, save_trace
        self.protocol = dict(protocol or {})
        self.source_provenance = dict(source_provenance or {})
        self.transitions = []
        self.traces = []
        self.valid = []

    @staticmethod
    def _snapshot(value):
        import numpy as np

        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise ValueError("Recording does not accept object arrays")
        return array.copy()

    def _env_snapshot(self, value, name):
        array = self._snapshot(value)
        if array.ndim < 1 or array.shape[0] != self.num_envs:
            raise ValueError(f"Recording field {name!r} must start with num_envs")
        return array

    def append(self, transition, first_episode_valid, trace=None):
        import numpy as np

        if len(self.valid) >= self.max_steps:
            raise ValueError("Recording exceeds the requested control-step bound")
        valid = self._env_snapshot(first_episode_valid, "first_episode_valid")
        if valid.shape != (self.num_envs,) or valid.dtype != np.bool_:
            raise ValueError("first_episode_valid must be a boolean environment mask")
        # Validate and copy a whole step before changing the recorder, so a
        # missing or malformed field cannot leave differently sized columns.
        transitions = ({key: self._env_snapshot(transition[key], key)
                        for key in self.TRANSITION_FIELDS}
                       if self.save_transitions else None)
        traces = ({key: self._env_snapshot(trace[key], key)
                   for key in self.TRACE_FIELDS} if self.save_trace else None)
        self.valid.append(valid)
        if transitions is not None:
            self.transitions.append(transitions)
        if traces is not None:
            self.traces.append(traces)

    @staticmethod
    def _content_hash(arrays):
        import numpy as np

        digest = hashlib.sha256()
        for name, value in sorted(arrays.items()):
            array = np.ascontiguousarray(value)
            digest.update(name.encode())
            digest.update(array.dtype.str.encode())
            digest.update(json.dumps(array.shape).encode())
            digest.update(array.tobytes())
        return digest.hexdigest()

    def finish(self, output, mode, control_dt):
        import numpy as np

        if not self.valid:
            raise ValueError("Cannot save an empty rollout recording")
        protocol = {
            **self.protocol, "schema": "wmac-physical-recording-v1", "mode": mode,
            "num_envs": self.num_envs, "steps_requested": self.max_steps,
            "steps_recorded": len(self.valid), "control_dt": control_dt,
            "layout": "time, environment, feature",
            "successor": "physical next state before automatic reset",
            "first_episode_valid": "includes terminal transition; excludes reset episodes",
            "restart_snapshot": False,
        }
        protocol_json = json.dumps(protocol, sort_keys=True, allow_nan=False)
        protocol_hash = hashlib.sha256(protocol_json.encode()).hexdigest()
        summary = {}
        for kind, records, enabled in (
            ("transitions", self.transitions, self.save_transitions),
            ("trace", self.traces, self.save_trace),
        ):
            if not enabled:
                continue
            arrays = {key: np.stack([row[key] for row in records]) for key in records[0]}
            arrays["first_episode_valid"] = np.stack(self.valid)
            arrays["recording_protocol_json"] = np.asarray(protocol_json)
            if self.source_provenance:
                # Provenance is deliberately separate from recording_protocol:
                # different training proxies can share one physical protocol.
                arrays["recording_source_provenance_json"] = np.asarray(json.dumps(
                    self.source_provenance, sort_keys=True, allow_nan=False))
            arrays["step_index"] = np.arange(len(self.valid), dtype=np.int64)
            arrays["control_time_seconds"] = (arrays["step_index"] + 1) * control_dt
            path = Path(output) / f"{kind}_{mode}.npz"
            temporary = path.with_suffix(path.suffix + ".tmp")
            try:
                with temporary.open("wb") as handle:
                    np.savez_compressed(handle, **arrays)
                os.replace(temporary, path)
            finally:
                temporary.unlink(missing_ok=True)
            summary[f"{kind}_recording"] = {
                "path": str(path), "schema": protocol["schema"],
                "protocol_sha256": protocol_hash,
                "content_sha256": self._content_hash(arrays),
                "steps_recorded": len(self.valid),
                "first_episode_transitions": int(arrays["first_episode_valid"].sum()),
                "shapes": {key: list(value.shape) for key, value in arrays.items()},
            }
            if self.source_provenance:
                summary[f"{kind}_recording"]["source_provenance"] = dict(self.source_provenance)
        return summary


class RoutingRecorder:
    """Optional MoE routes at the exact observation/action submission time."""

    def __init__(self, num_envs, max_steps, num_experts, top_k, architecture=None, num_actions=29):
        if min(num_envs, max_steps, num_experts, top_k) < 1 or top_k > num_experts:
            raise ValueError("Invalid routing recorder dimensions")
        self.num_envs, self.max_steps, self.num_experts, self.top_k = num_envs, max_steps, num_experts, top_k
        self.num_actions = num_actions
        self.architecture, self.rows = architecture, []

    def append(self, route, action, first_episode_valid):
        import numpy as np

        if len(self.rows) >= self.max_steps:
            raise ValueError("Routing recording exceeds requested control-step bound")
        values = {key: EvaluationRecorder._snapshot(getattr(route, key))
                  for key in ("indices", "weights", "probabilities")}
        values["action"] = EvaluationRecorder._snapshot(action)
        values["first_episode_valid"] = EvaluationRecorder._snapshot(first_episode_valid)
        shapes = {"indices": (self.num_envs, self.top_k), "weights": (self.num_envs, self.top_k),
                  "probabilities": (self.num_envs, self.num_experts), "action": (self.num_envs, self.num_actions),
                  "first_episode_valid": (self.num_envs,)}
        if any(values[key].shape != shape for key, shape in shapes.items()):
            raise ValueError("Routing fields have incompatible environment/expert/action dimensions")
        indices = values["indices"]
        if indices.dtype.kind not in "iu" or (indices < 0).any() or (indices >= self.num_experts).any():
            raise ValueError("Routing expert indices must be valid integers")
        if values["first_episode_valid"].dtype != np.bool_:
            raise ValueError("Routing first_episode_valid must be boolean")
        for key in ("weights", "probabilities", "action"):
            if values[key].dtype.kind != "f" or not np.isfinite(values[key]).all():
                raise ValueError("Routing actions, weights and probabilities must be finite floating-point values")
        for key in ("weights", "probabilities"):
            if (values[key] < 0).any() or not np.allclose(values[key].sum(-1), 1., rtol=1e-5, atol=1e-6):
                raise ValueError("Routing weights and probabilities must be normalized nonnegative rows")
        self.rows.append(values)

    def finish(self, output, mode, control_dt):
        import numpy as np

        if not self.rows:
            raise ValueError("Cannot save an empty routing recording")
        protocol = {"schema": "wmac-policy-routing-v1", "mode": mode,
            "num_envs": self.num_envs, "num_experts": self.num_experts, "top_k": self.top_k,
            "steps_requested": self.max_steps, "steps_recorded": len(self.rows),
            "layout": "time, environment, selected_expert_or_expert",
            "timing": "same raw Actor observation as submitted action; before env.step",
            "action": "Actor output submitted to env.step, before environment action processing",
            "first_episode_valid": "includes terminal action; excludes reset episodes",
            "actor_architecture": self.architecture,
            "limitation": "Hard top-k route changes can cause action discontinuities; this trace measures routes and does not prove smooth control."}
        arrays = {key: np.stack([row[key] for row in self.rows]) for key in self.rows[0]}
        arrays["step_index"] = np.arange(len(self.rows), dtype=np.int64)
        arrays["action_time_seconds"] = arrays["step_index"] * control_dt
        arrays["routing_protocol_json"] = np.asarray(json.dumps(protocol, sort_keys=True, allow_nan=False))
        path = Path(output)/f"routing_{mode}.npz"
        temporary = path.with_suffix(".npz.tmp")
        try:
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return {"routing_recording": {"path": str(path), "schema": protocol["schema"],
            "steps_recorded": len(self.rows), "first_episode_actions": int(arrays["first_episode_valid"].sum()),
            "content_sha256": EvaluationRecorder._content_hash(arrays),
            "limitation": protocol["limitation"]}}


def policy_action_with_optional_routing(actor, observation, first_episode_valid, recorder=None):
    """Trace only when requested; use the very same raw observation for both."""
    action = actor(observation)
    limit = getattr(actor, "source_action_limit", None)
    if limit is not None:
        action = action.clamp(-limit, limit)
    if recorder is not None:
        import torch
        with torch.no_grad():
            recorder.append(actor.route(observation), action, first_episode_valid)
    return action


def load_actor(checkpoint, device):
    """Reconstruct only the deployed policy; no discriminator is loaded."""
    import torch
    from src.wmac.checkpoints import checkpoint_height_reward_semantics, checkpoint_recovery_semantics, resolve_checkpoint_config
    from src.wmac.actors import actor_architecture, create_actor
    from src.wmac.networks import Actor
    from src.wmac.reference import is_official, smooth_reward_protocol
    from src.robot_profiles import PROFILES, profile_for_joint_count, profile_for_task

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    robot = None
    if payload.get("format") == "amp-wmac-v1":
        cfg = resolve_checkpoint_config(payload)
        robot = profile_for_task(cfg.task)
        actor = create_actor(cfg, obs_dim=4 * (9 + 3 * robot.num_joints), action_dim=robot.num_joints)
        actor.load_state_dict(payload["actor"])
        if is_official(cfg):
            actor.source_action_limit = cfg.action_limit
        actor.source_actor_architecture = actor_architecture(cfg)
        actor.source_actor_type = cfg.actor_type
        actor.source_recovery_counter_semantics = checkpoint_recovery_semantics(payload)
        actor.source_smooth_reward_protocol = smooth_reward_protocol(cfg)
        kind = "amp-wmac-v1"
    else:
        model = payload.get("model_state_dict", payload)
        weights = sorted(
            ((int(key.split(".")[1]), value) for key, value in model.items()
             if key.startswith("actor.") and key.endswith(".weight") and value.ndim == 2),
            key=lambda item: item[0],
        )
        # The flat actor sees four frames of 9 + 3n values and outputs n actions:
        # 384 and 29 for the G1, 312 and 23 for the PM01.
        actions = int(weights[-1][1].shape[0]) if weights else 0
        if not weights or actions not in {profile.num_joints for profile in PROFILES} \
                or weights[0][1].shape[1] != 4 * (9 + 3 * actions):
            raise ValueError("Expected a 384-observation, 29-action AMP ActorCritic checkpoint.")
        robot = profile_for_joint_count(actions)
        hidden = tuple(int(value.shape[0]) for _, value in weights[:-1])
        actor = Actor(obs_dim=4 * (9 + 3 * actions), action_dim=actions, hidden=hidden, action_limit=None)
        actor.load_baseline(payload)
        actor.source_actor_type = "mlp"
        actor.source_actor_architecture = actor_architecture({
            "actor_type": "mlp", "policy_hidden": hidden, "action_limit": None})
        actor.source_recovery_counter_semantics = "not_applicable"
        actor.source_smooth_reward_protocol = {"smooth_reward_mode": "none"}
        kind = "ppo-amp-baseline"
    actor.source_recovery_height_reward = checkpoint_height_reward_semantics(payload)
    saved_config = payload.get("config", {})
    schema = payload.get("amp_feature_schema", saved_config.get("amp_feature_schema", "legacy"))
    task = saved_config.get("task") or payload.get("amp_task_id") or (
        robot.flat_tasks[1] if schema == "grounded_v2" else robot.flat_tasks[0])
    if task not in tuple(name for profile in PROFILES for name in profile.flat_tasks):
        raise ValueError(f"Unsupported checkpoint evaluation task: {task}")
    actor.amp_task_id, actor.amp_feature_schema = task, schema
    # External rollout time is deliberately separate from the checkpoint's
    # accumulated training resources. Legacy checkpoints have unknown costs.
    actor.training_cost = payload.get("training_cost")
    return actor.to(device).eval(), kind


def seed_all(seed):
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def keep_fallen_recovery_frames(env):
    """Use existing recovery data, restricted to genuinely low or tilted poses."""
    from src.tasks.amp_loco.mdp.events import MotionResetManager
    from src.wmac.state import quat_to_matrix

    path = env.cfg.events["init_motion_loader"].params["motion_dir"]
    manager = MotionResetManager.get()
    if path not in manager.recovery_frames:
        raise ValueError("Recovery evaluation requires the Recovery NPZ motion group.")
    frames = manager.recovery_frames[path]
    up_z = quat_to_matrix(frames["root_quat"])[..., 2, 2]
    # Below the robot's recovery height (0.55 m for the G1) or tilted.
    recovery_height = getattr(getattr(env, "adapter", None), "recovery_height", None)
    if recovery_height is None:
        from src.robot_profiles import G1, profile_for_bodies
        scene = getattr(env, "scene", None)
        recovery_height = (profile_for_bodies(scene["robot"].body_names) if scene is not None else G1).recovery_height
    fallen = (frames["root_pos"][:, 2] < recovery_height) | (up_z < 0.5)
    count = int(fallen.sum().item())
    if not count:
        raise ValueError("Recovery group contains no low or tilted start frames.")
    manager.recovery_frames[path] = {key: value[fallen] for key, value in frames.items()}
    return count


def evaluate_mode(actor, args, mode, output):
    import numpy as np
    import torch
    from mjlab.tasks.registry import load_env_cfg
    from src.wmac.env import CaptureEnv
    from src.wmac.state import StateAdapter, heading_quat, quat_apply, quat_to_matrix
    from src.recovery_rewards import configure_recovery_height_reward

    task = getattr(actor, "amp_task_id", "Unitree-G1-AMP-Flat")
    cfg = load_env_cfg(task)
    # Evaluate every policy under the explicitly selected physical objective;
    # never silently switch it to match a checkpoint's training reward.
    height_reward = getattr(args, "recovery_height_reward", "height_only_v1")
    configure_recovery_height_reward(cfg, height_reward)
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.curriculum = {}
    if not args.push:
        cfg.events.pop("push_robot", None)
    for group in cfg.observations.values():
        group.enable_corruption = args.observation_noise
    init = cfg.events["init_motion_loader"].params
    init["delay_reset_env_ratio"] = 0.0 if mode == "walk" else 1.0
    if mode == "recovery":
        init["max_delay_steps"] = args.steps + 1
    command_cfg = cfg.commands["twist"]
    command_cfg.rel_standing_envs = 0.0
    command_cfg.rel_heading_envs = 0.0
    command_cfg.heading_command = False
    command_cfg.ranges.heading = None
    command_cfg.resampling_time_range = (args.steps * cfg.decimation * cfg.sim.mujoco.timestep + 1,) * 2
    command_cfg.ranges.lin_vel_x = (args.command_vx,) * 2
    command_cfg.ranges.lin_vel_y = (args.command_vy,) * 2
    command_cfg.ranges.ang_vel_z = (args.command_yaw,) * 2

    # Policy construction consumes random numbers. Reset before both simulator
    # creation (randomized physics) and reset (motion-frame sampling/history).
    seed_all(args.seed)
    env = CaptureEnv(cfg=cfg, device=args.device)
    try:
        env.adapter = adapter = StateAdapter(env)
        recovery_candidates = keep_fallen_recovery_frames(env) if mode == "recovery" else None
        seed_all(args.seed)
        observations, _ = env.reset(seed=args.seed)
        initial = {
            "state": adapter.state().detach().cpu().numpy(),
            "context": adapter.context().detach().cpu().numpy(),
            "command": adapter.command().detach().cpu().numpy(),
            "actor_history": observations["actor"].detach().cpu().numpy(),
        }
        digest = hashlib.sha256()
        for name in sorted(initial):
            digest.update(name.encode())
            digest.update(np.ascontiguousarray(initial[name]).tobytes())
        initial_hash = digest.hexdigest()
        npz_path = output / f"initial_{mode}.npz"
        np.savez_compressed(npz_path, **initial)

        n, dev, dt = env.num_envs, torch.device(args.device), env.step_dt
        active = torch.ones(n, dtype=torch.bool, device=dev)
        succeeded = torch.zeros(n, dtype=torch.bool, device=dev)
        kept_walking = torch.zeros(n, dtype=torch.bool, device=dev)
        ever_fell = torch.zeros(n, dtype=torch.bool, device=dev)
        previously_bad = torch.zeros(n, dtype=torch.bool, device=dev)
        stable_count = torch.zeros(n, dtype=torch.long, device=dev)
        walking_count = torch.zeros(n, dtype=torch.long, device=dev)
        stable_time = torch.full((n,), float("nan"), device=dev)
        fall_count = torch.zeros(n, dtype=torch.long, device=dev)
        observed_steps = torch.zeros(n, dtype=torch.long, device=dev)
        xy_error_sum = torch.zeros(n, device=dev)
        yaw_error_sum = torch.zeros(n, device=dev)
        reward_sum = torch.zeros(n, device=dev)
        terminated_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        timeout_ever = torch.zeros(n, dtype=torch.bool, device=dev)
        final_upright = torch.zeros(n, dtype=torch.bool, device=dev)
        steps_executed = 0
        record_transitions = getattr(args, "save_transitions", False)
        record_trace = getattr(args, "save_trace", False)
        routing_recorder = (RoutingRecorder(n, args.steps, actor.num_experts, actor.top_k,
                            getattr(actor, "source_actor_architecture", None), num_actions=adapter.num_joints)
                            if record_trace and callable(getattr(actor, "route", None)) else None)
        recorder = EvaluationRecorder(
            n, args.steps, save_transitions=record_transitions, save_trace=record_trace,
            source_provenance=evaluation_source_metadata(actor)["source_provenance"],
            protocol={
                "task": task, "checkpoint": str(args.checkpoint),
                "recovery_height_reward": height_reward,
                "seed": args.seed, "initial_state_sha256": initial_hash,
                "push": args.push, "observation_noise": args.observation_noise,
                "command": [args.command_vx, args.command_vy, args.command_yaw],
                "height_threshold": args.height_threshold, "upright_cos": args.upright_cos,
                "stable_steps": args.stable_steps, "walk_hold_steps": args.walk_hold_steps,
                "walk_velocity_tolerance": args.walk_velocity_tolerance,
                "walk_yaw_tolerance": args.walk_yaw_tolerance,
            },
        ) if record_transitions or record_trace else None
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        started_at = time.perf_counter()
        with torch.no_grad():
            for step in range(args.steps):
                action = policy_action_with_optional_routing(actor, observations["actor"], active, routing_recorder)
                observations, _, _, _, _ = env.step(action)
                transition = env.transition
                terminated, timeout = transition["terminated"], transition["timeout"]
                state = transition["next_state"]  # true physical successor, before reset
                _, rotations, velocities, angular_velocities = adapter.kinematics(state)
                torso = adapter.anchor_id
                desired_local = torch.cat((transition["command"][:, :2], torch.zeros(n, 1, device=dev)), -1)
                desired_world = quat_apply(heading_quat(rotations[:, torso]), desired_local)
                xy_error = (desired_world[:, :2] - velocities[:, torso, :2]).norm(dim=-1)
                yaw_error = (transition["command"][:, 2] - angular_velocities[:, torso, 2]).abs()
                up_z = quat_to_matrix(state[:, 3:7])[:, 2, 2]
                upright = (state[:, 2] > args.height_threshold) & (up_z > args.upright_cos)
                bad = (state[:, 2] < adapter.height_threshold) | (up_z < adapter.tilt_cos)
                done = terminated | timeout

                observed_steps += active.long()
                xy_error_sum += xy_error * active
                yaw_error_sum += yaw_error * active
                reward_sum += transition["reward"] * active
                fall_count += (bad & ~previously_bad & active).long()
                ever_fell |= bad & active
                previously_bad = torch.where(active, bad, previously_bad)
                final_upright = torch.where(active, upright, final_upright)
                stable_count = torch.where(active & upright & ~done, stable_count + 1, torch.zeros_like(stable_count))
                first_success = active & ~succeeded & (stable_count >= args.stable_steps)
                stable_time[first_success] = (step + 1) * dt
                succeeded |= first_success

                # Continue walking requires commanded translation, sustained
                # upright posture and independent velocity tracking errors.
                moving = transition["command"][:, :2].norm(dim=-1) > 0.1
                good_walk = (active & succeeded & upright & ~done & moving
                             & (xy_error < args.walk_velocity_tolerance)
                             & (yaw_error < args.walk_yaw_tolerance))
                walking_count = torch.where(good_walk, walking_count + 1, torch.zeros_like(walking_count))
                kept_walking |= walking_count >= args.walk_hold_steps
                terminated_ever |= terminated & active
                timeout_ever |= timeout & active
                if recorder is not None:
                    trace = {
                        "root_height": state[:, 2], "upright_cos": up_z,
                        "velocity_xy_error": xy_error, "velocity_yaw_error": yaw_error,
                        "action_rms": transition["action"].square().mean(-1).sqrt(),
                        "recovered": succeeded, "terminated": terminated, "timeout": timeout,
                    } if record_trace else None
                    recorder.append(transition, active, trace)
                active &= ~done
                steps_executed = step + 1
                if not active.any():
                    break
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        wall = time.perf_counter() - started_at
        total_steps = int(observed_steps.sum().item())
        succeeded_count = int(succeeded.sum().item())

        def per_env_float(values):
            return [float(value) if math.isfinite(float(value)) else None for value in values.cpu().tolist()]

        summary = {
            **evaluation_source_metadata(actor),
            "physical_recovery_height_reward": height_reward,
            "mode": mode, "num_envs": n, "steps_requested": args.steps,
            "steps_executed": steps_executed, "control_dt": dt,
            "first_episode_control_steps": total_steps,
            "wall_seconds_rollout": wall,
            "velocity_xy_error_mean": float(xy_error_sum.sum().item() / max(total_steps, 1)),
            "yaw_velocity_error_mean": float(yaw_error_sum.sum().item() / max(total_steps, 1)),
            "task_reward_per_control_step": float(reward_sum.sum().item() / max(total_steps, 1)),
            "baseline_task_reward_per_control_step": float(reward_sum.sum().item() / max(total_steps, 1)),
            "fall_episode_fraction": float(ever_fell.float().mean().item()),
            "fall_events": int(fall_count.sum().item()),
            "fall_events_per_1000_control_steps": 1000 * float(fall_count.sum().item()) / max(total_steps, 1),
            "terminated_first_episode_fraction": float(terminated_ever.float().mean().item()),
            "timeout_first_episode_fraction": float(timeout_ever.float().mean().item()),
            "first_episode_survival_fraction": float(active.float().mean().item()),
            "final_upright_fraction": float(final_upright.float().mean().item()),
            "initial_state_npz": str(npz_path), "initial_state_sha256": initial_hash,
            "recovery_start_candidates": recovery_candidates,
            "per_env": {
                "observed_control_steps": observed_steps.cpu().tolist(),
                "fall_events": fall_count.cpu().tolist(),
                "terminated": terminated_ever.cpu().tolist(),
                "timed_out": timeout_ever.cpu().tolist(),
            },
        }
        if mode == "recovery":
            summary.update({
                "recovery_success_fraction": float(succeeded.float().mean().item()),
                "recovery_success_count": succeeded_count,
                "first_stable_recovery_seconds_mean": float(stable_time[succeeded].mean().item()) if succeeded_count else None,
                "success_then_walk_fraction_all_starts": float(kept_walking.float().mean().item()),
                "success_then_walk_fraction_successes": float(kept_walking.sum().item() / succeeded_count) if succeeded_count else None,
                "success_walk_and_finish_upright_fraction": float((kept_walking & final_upright & active).float().mean().item()),
            })
            summary["per_env"].update({
                "recovered": succeeded.cpu().tolist(), "continued_walking": kept_walking.cpu().tolist(),
                "first_stable_recovery_seconds": per_env_float(stable_time),
            })
        if recorder is not None:
            summary.update(recorder.finish(output, mode, dt))
        if routing_recorder is not None:
            summary.update(routing_recorder.finish(output, mode, dt))
        return summary
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--mode", choices=("walk", "recovery", "both"), default="both")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--push", action="store_true")
    parser.add_argument("--observation-noise", action="store_true")
    parser.add_argument("--recovery-height-reward", choices=("height_only_v1", "upright_v2"),
                        default="height_only_v1",
                        help="Physical evaluation reward, independent of checkpoint training semantics")
    parser.add_argument("--save-transitions", action="store_true",
                        help="Save bounded pre-reset physical transitions and first-episode masks")
    parser.add_argument("--save-trace", action="store_true",
                        help="Save per-step physical recovery, tracking, and action diagnostics")
    parser.add_argument("--command-vx", type=float, default=0.5)
    parser.add_argument("--command-vy", type=float, default=0.0)
    parser.add_argument("--command-yaw", type=float, default=0.0)
    parser.add_argument("--height-threshold", type=float, default=None,
                        help="Root height of a stable recovery; default: the robot's upright "
                             "height (0.65 m for the G1, 0.67 m for the PM01)")
    parser.add_argument("--upright-cos", type=float, default=0.8)
    parser.add_argument("--stable-steps", type=int, default=25)
    parser.add_argument("--walk-hold-steps", type=int, default=50)
    parser.add_argument("--walk-velocity-tolerance", type=float, default=0.5)
    parser.add_argument("--walk-yaw-tolerance", type=float, default=0.5)
    parser.add_argument("--output", type=Path, help="Output directory for metrics.json and initial-state NPZs")
    args = parser.parse_args()
    for name in ("num_envs", "steps", "stable_steps", "walk_hold_steps"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not -1 <= args.upright_cos <= 1:
        parser.error("--upright-cos must lie in [-1, 1]")
    if ((args.height_threshold is not None and args.height_threshold <= 0)
            or args.walk_velocity_tolerance <= 0 or args.walk_yaw_tolerance <= 0):
        parser.error("Height and velocity tolerances must be positive")
    args.checkpoint = args.checkpoint.expanduser().resolve()
    if not args.checkpoint.is_file():
        parser.error(f"Checkpoint not found: {args.checkpoint}")
    os.environ["MUJOCO_GL"] = "egl"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if args.device.startswith("cuda") and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = args.device.partition(":")[2] or "0"
        args.device = "cuda:0"

    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.utils.torch import configure_torch_backends

    configure_torch_backends()
    actor, kind = load_actor(args.checkpoint, args.device)
    if args.height_threshold is None:
        from src.robot_profiles import profile_for_task
        args.height_threshold = profile_for_task(actor.amp_task_id).upright_height
    output = args.output.expanduser().resolve() if args.output else (
        REPO / "logs" / "wmac" / "evaluations" /
        f"{datetime.now(ZoneInfo('Asia/Shanghai')):%Y-%m-%d_%H-%M-%S_%f}_{args.checkpoint.stem}"
    )
    output.mkdir(parents=True, exist_ok=False)
    modes = ("walk", "recovery") if args.mode == "both" else (args.mode,)
    metrics = {
        **evaluation_source_metadata(actor),
        "checkpoint": str(args.checkpoint), "checkpoint_format": kind,
        "training_cost": actor.training_cost,
        "arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        "protocol": {
            "first_episode_only": True, "task": actor.amp_task_id,
            "recovery_height_reward": args.recovery_height_reward,
            "amp_feature_schema": actor.amp_feature_schema,
            "task_reward_scope": "baseline_physical_task_without_training_smooth_terms",
            "policy": "deterministic mean; normalization frozen", "discriminator_used": False,
            "stable_recovery": "height above threshold and pelvis up_z above upright_cos for stable_steps consecutive non-terminal steps",
            "continued_walk": "after recovery, upright and velocity/yaw errors below tolerance with nonzero translation command for walk_hold_steps",
            "initial_states": "recorded raw state/context/command/history; not a simulator snapshot or exact restart format",
        },
        "results": {},
    }
    for mode in modes:
        metrics["results"][mode] = evaluate_mode(actor, args, mode, output)
        (output / "metrics.json").write_text(json.dumps(metrics, indent=2, allow_nan=False) + "\n")
    for mode, result in metrics["results"].items():
        compact = {key: value for key, value in result.items() if key != "per_env"}
        print(f"[{mode}] {json.dumps(compact, sort_keys=True)}")
    print(f"Metrics: {output / 'metrics.json'}")


if __name__ == "__main__":
    main()
