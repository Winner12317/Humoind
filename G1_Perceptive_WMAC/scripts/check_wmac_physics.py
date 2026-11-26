"""Check WMAC's physics interfaces and, optionally, model gradient directions.

This is a short diagnostic, not a training or policy-quality benchmark. The
direction check differentiates an open-loop action sequence through the model
and compares it with central finite differences of the same smooth objective
in physical simulation. It checks all snapshot restores and repeated physical
rollouts before reporting a finite-difference comparison.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
import os
from pathlib import Path
import random
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import torch
import warp as wp

from mjlab.tasks.registry import load_env_cfg
import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from rsl_rl.modules.discriminator import Discriminator
from src.wmac.config import WmacConfig
from src.wmac.checkpoints import resolve_checkpoint_config
from src.wmac.actors import create_actor
from src.recovery_rewards import configure_recovery_height_reward
from src.wmac.env import CaptureEnv
from src.wmac.networks import Critic, RunningNorm, WorldModel, frozen_parameters
from src.wmac.runner import differentiable_amp_reward
from src.wmac.state import StateAdapter
from src.amp_features import amp_feature_dim


class PhysicsSnapshot:
    """In-place copies preserve CUDA graph addresses and solver warm starts.

    All arrays in MJWarp Data, not just qpos/qvel, are saved. The manager and
    scene tensors include action history, actuator state, observation history,
    command/event timers, reward sums and delayed-termination counters. The
    diagnostic disables resets, interval events and command updates while using
    this snapshot; mutable episode logging dictionaries are not physics state.
    """

    def __init__(self, env):
        self.env = env
        self.warp_arrays = []
        self.tensors = []
        self.scalars = []
        seen = set()

        def walk_warp(value):
            if id(value) in seen:
                return
            seen.add(id(value))
            if isinstance(value, wp.array):
                self.warp_arrays.append((value, wp.clone(value)))
            elif isinstance(value, dict):
                for child in value.values():
                    walk_warp(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    walk_warp(child)
            elif hasattr(value, "__dict__") and type(value).__module__.startswith("mujoco_warp"):
                for child in vars(value).values():
                    walk_warp(child)

        walk_warp(env.sim.data.struct)
        seen = {id(env), id(env.sim), id(env.adapter)}
        skip_keys = {"_env", "env", "cfg", "adapter", "sim"}

        def walk_torch(value):
            if id(value) in seen:
                return
            seen.add(id(value))
            if isinstance(value, torch.Tensor):
                self.tensors.append((value, value.detach().clone()))
            elif isinstance(value, dict):
                for child in value.values():
                    walk_torch(child)
            elif isinstance(value, (list, tuple)):
                for child in value:
                    walk_torch(child)
            elif (hasattr(value, "__dict__") and not callable(value)
                  and type(value).__module__.startswith(("mjlab.", "src."))
                  and "sim_data" not in type(value).__module__):
                for key, child in vars(value).items():
                    if key in skip_keys:
                        continue
                    if isinstance(child, (int, float, bool)):
                        self.scalars.append((value, key, child))
                    else:
                        walk_torch(child)

        for name in ("scene", "action_manager", "observation_manager", "command_manager",
                     "event_manager", "termination_manager", "reward_manager", "metrics_manager"):
            walk_torch(getattr(env, name))
        self.episode_length = env.episode_length_buf.clone()
        self.counters = (env.common_step_counter, env._sim_step_counter)
        self.obs = {k: v.clone() for k, v in env.obs_buf.items()}
        self.state = env.adapter.state().clone()
        self.context = env.adapter.context().clone()
        self.command = env.adapter.command().clone()
        self.previous_action = env.action_manager.action.clone()
        self.rng = (torch.get_rng_state(), torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    np.random.get_state(), random.getstate())
        wp.synchronize()

    @torch.no_grad()
    def restore(self):
        for target, saved in self.warp_arrays:
            wp.copy(target, saved)
        wp.synchronize()
        for target, saved in self.tensors:
            target.copy_(saved)
        for owner, name, saved in self.scalars:
            setattr(owner, name, saved)
        self.env.episode_length_buf.copy_(self.episode_length)
        self.env.common_step_counter, self.env._sim_step_counter = self.counters
        self.env.obs_buf = {k: v.clone() for k, v in self.obs.items()}
        self.env.observation_manager._obs_buffer = self.env.obs_buf
        self.env.transition = None
        self.env._physical_end = None
        torch.set_rng_state(self.rng[0])
        if self.rng[1] is not None:
            torch.cuda.set_rng_state_all(self.rng[1])
        np.random.set_state(self.rng[2])
        random.setstate(self.rng[3])
        for name, actual, expected in (
            ("raw state", self.env.adapter.state(), self.state),
            ("context", self.env.adapter.context(), self.context),
            ("command", self.env.adapter.command(), self.command),
            ("action history", self.env.action_manager.action, self.previous_action),
        ):
            if not torch.equal(actual, expected):
                raise RuntimeError(f"Snapshot did not exactly restore {name}")


@torch.no_grad()
def check_interfaces(env, steps, interval_control):
    adapter = env.adapter
    maxima = dict(position=0., quaternion=0., linear_velocity=0., angular_velocity=0., actor_frame=0., amp=0.)
    for _ in range(steps):
        state = adapter.state()
        p, q, v, w = adapter.kinematics(state)
        data = env.scene["robot"].data
        errors = {
            "position": (p-data.body_link_pos_w).abs().max(),
            "quaternion": torch.minimum((q-data.body_link_quat_w).abs().amax(-1),
                                         (q+data.body_link_quat_w).abs().amax(-1)).max(),
            "linear_velocity": (v-data.body_link_lin_vel_w).abs().max(),
            "angular_velocity": (w-data.body_link_ang_vel_w).abs().max(),
            "actor_frame": (adapter.actor_frame(state, env.action_manager.action,
                adapter.command(), adapter.context())-env.obs_buf["actor"][:, -96:]).abs().max(),
            "amp": (adapter.amp_features(state)-env.obs_buf["amp"]).abs().max(),
        }
        for key, error in errors.items():
            maxima[key] = max(maxima[key], float(error))
        env.step(torch.zeros(env.num_envs, 29, device=env.device))
    if max(maxima.values()) > 2e-4:
        raise AssertionError(f"Physics interface mismatch: {maxima}")
    # The interval timer is configured to fire every step. During the preceding
    # checks it was held fixed by the instance-local event wrapper.
    interval_control["enabled"] = True
    env.step(torch.zeros(env.num_envs, 29, device=env.device))
    transition = env.transition
    not_done = ~(transition["terminated"] | transition["timeout"])
    pushed = (adapter.state()-transition["next_state"]).abs().amax(-1) > 1e-6
    if not not_done.any() or not (transition["boundary"][not_done] & pushed[not_done]).all():
        raise AssertionError("A post-physics push was not recorded as a rollout boundary")
    push_count = int((not_done & pushed).sum())
    interval_control["enabled"] = False
    # Artificial time limits must preserve the physical final state before reset.
    env.episode_length_buf[:] = env.max_episode_length-1
    env.step(torch.zeros(env.num_envs, 29, device=env.device))
    transition = env.transition
    physical_not_reset = (transition["next_state"]-adapter.state()).abs().amax(-1) > 1e-5
    if not transition["timeout"].all() or not physical_not_reset.all() or not transition["boundary"].all():
        raise AssertionError("Timeout capture included a reset state")
    env.reset()
    state = adapter.state().detach().requires_grad_(True)
    # Re-enable gradients for the feature path even though this function otherwise
    # uses inference mode. This checks the rotations and FK at actual reset states.
    with torch.enable_grad():
        grad = torch.autograd.grad(adapter.amp_features(state).square().mean(), state)[0]
    if not torch.isfinite(grad).all():
        raise AssertionError("AMP feature derivative is not finite")
    return {"maximum_absolute_errors": maxima, "push_boundaries_verified": push_count,
            "pre_reset_timeout_capture_verified": True, "amp_input_gradient_finite": True}


def load_checkpoint(path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    if payload.get("format") != "amp-wmac-v1":
        raise ValueError("--checkpoint expects a WMAC amp-wmac-v1 checkpoint")
    cfg = resolve_checkpoint_config(payload)
    actor = create_actor(cfg).to(device)
    model = WorldModel(hidden=cfg.model_hidden).to(device)
    critic = Critic(136, hidden=cfg.critic_hidden).to(device)
    dim = amp_feature_dim(cfg.amp_feature_schema)
    discriminator = Discriminator(2*dim, cfg.amp_reward_coef, list(cfg.discriminator_hidden), str(device), cfg.task_weight).to(device)
    normalizer = RunningNorm(dim, eps=cfg.amp_normalizer_epsilon,
        max_count=cfg.amp_normalizer_until if cfg.amp_normalizer_until is not None else float('inf')).to(device)
    normalizer.variance_epsilon, normalizer.clip = cfg.amp_normalizer_variance_epsilon, cfg.amp_normalizer_clip
    for key, value in payload.get('amp_normalizer_settings', {}).items():
        if key in ('eps', 'variance_epsilon', 'clip', 'max_count'):
            setattr(normalizer, key, value)
    for name, module in (("actor", actor), ("world_model", model), ("critic", critic),
                         ("discriminator", discriminator), ("amp_norm", normalizer)):
        module.load_state_dict(payload[name])
        module.eval()
    return cfg, actor, model, critic, discriminator, normalizer


def check_directions(env, modules, horizon, directions, epsilon):
    cfg, actor, model, critic, discriminator, normalizer = modules
    adapter = env.adapter
    # Freeze exogenous goals and disable resets only inside this diagnostic.
    # Both physical and model objectives instead apply the identical smooth
    # survival/delay proxy externally. This allows safe comparison at falls.
    env.command_manager.compute = lambda *args, **kwargs: None
    def no_reset():
        tm = env.termination_manager
        tm._terminated_buf.zero_()
        tm._truncated_buf.zero_()
        return tm._terminated_buf | tm._truncated_buf
    env.termination_manager.compute = no_reset
    snapshot = PhysicsSnapshot(env)

    @torch.no_grad()
    def nominal_actions():
        snapshot.restore()
        result = []
        for _ in range(horizon):
            action = actor(env.obs_buf["actor"])
            result.append(action.clone())
            env.step(action)
        return torch.stack(result)

    def objective(actions, physical):
        if physical:
            snapshot.restore()
        state = snapshot.state.clone()
        context = snapshot.context.clone()
        previous = snapshot.previous_action.clone()
        command = snapshot.command
        survival = state.new_ones(len(state))
        value = torch.zeros_like(survival)
        for t, action in enumerate(actions):
            if physical:
                env.step(action)
                next_state = env.transition["next_state"]
                contact = env.transition["contact_cost"]
            else:
                next_state = model.step(state, action, context)
                contact = model.contact_cost(state, action, context)
            next_context, terminated = adapter.advance_context(next_state, context)
            task = adapter.task_reward(state, next_state, action, previous, command,
                                       next_context, terminated, contact)
            style, _ = differentiable_amp_reward(discriminator, normalizer,
                adapter.amp_features(state), adapter.amp_features(next_state), cfg.amp_reward_coef)
            value = value+cfg.gamma**t*survival*(cfg.task_weight*task+cfg.style_weight*style)
            survival = survival*(1-terminated)
            state, context, previous = next_state, next_context, action
        if cfg.critic_bootstrap:
            value = value+cfg.gamma**horizon*survival*critic(
                adapter.critic_features(state, previous, command, context))
        return value, state

    with torch.no_grad():
        actions = nominal_actions()
        first, final_first = objective(actions, physical=True)
        second, final_second = objective(actions, physical=True)
    repeat_state_error = float((final_first-final_second).abs().max())
    repeat_return_error = float((first-second).abs().max())
    if repeat_state_error > 1e-6 or repeat_return_error > 1e-6:
        return {"finite_difference_performed": False, "reason": "Physical snapshot repeatability check failed",
                "repeat_state_max_error": repeat_state_error, "repeat_return_max_error": repeat_return_error}
    actions = actions.detach().requires_grad_(True)
    with frozen_parameters(model, critic, discriminator, normalizer):
        imagined, _ = objective(actions, physical=False)
        gradient = torch.autograd.grad(imagined.sum(), actions)[0].detach()
    if not torch.isfinite(gradient).all():
        return {"finite_difference_performed": False, "reason": "Model action derivative is non-finite"}
    results = []
    for i in range(directions):
        direction = gradient.clone() if i == 0 else torch.randn_like(gradient)
        direction = direction/direction.square().sum((0, 2), keepdim=True).sqrt().clamp_min(1e-12)
        projected = (gradient*direction).sum((0, 2))
        with torch.no_grad():
            plus, _ = objective(actions.detach()+epsilon*direction, physical=True)
            minus, _ = objective(actions.detach()-epsilon*direction, physical=True)
        measured = (plus-minus)/(2*epsilon)
        for j in range(env.num_envs):
            stable = abs(float(measured[j])) > 1e-6 and abs(float(projected[j])) > 1e-6
            results.append({"direction": "model_gradient" if i == 0 else f"random_{i}", "environment": j,
                "start_height": float(snapshot.state[j, 2]),
                "phase": "recovery" if snapshot.state[j, 2] < .55 else "upright",
                "model_projection": float(projected[j]), "physical_central_difference": float(measured[j]),
                "sign_agrees": bool(projected[j]*measured[j] > 0) if stable else None})
    valid = [row["sign_agrees"] for row in results if row["sign_agrees"] is not None]
    snapshot.restore()
    return {"finite_difference_performed": True,
        "objective": "Full differentiable task+AMP reward, smooth delay/survival, optional fixed terminal critic; open-loop actions",
        "scope": "Local derivative diagnostic; it does not validate critic long-term accuracy or establish training speedup",
        "ablation_note": "The same full smooth objective is used in both branches; stop-gradient ablation switches are not applied",
        "checkpoint_critic_bootstrap": cfg.critic_bootstrap,
        "repeat_state_max_error": repeat_state_error, "repeat_return_max_error": repeat_return_error,
        "snapshot_warp_array_count": len(snapshot.warp_arrays), "snapshot_tensor_count": len(snapshot.tensors),
        "horizon": horizon, "epsilon": epsilon, "directions": directions,
        "sign_agreement_fraction": sum(valid)/len(valid) if valid else None, "comparisons": results}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--directions", type=int, default=4)
    parser.add_argument("--epsilon", type=float, default=.01)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not 1 <= args.num_envs <= 32 or not 1 <= args.steps <= 32:
        parser.error("Use 1..32 environments and interface steps for this short diagnostic")
    physical_steps = args.steps+2+(3+2*args.directions)*args.horizon if args.checkpoint else args.steps+2
    if args.horizon < 1 or args.directions < 1 or physical_steps > 128:
        parser.error("Diagnostic is limited to 128 environment.step calls; reduce horizon or directions")
    if not math.isfinite(args.epsilon) or args.epsilon <= 0:
        parser.error("epsilon must be finite and positive")
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    modules = load_checkpoint(args.checkpoint, args.device) if args.checkpoint else None
    source_cfg = modules[0] if modules else WmacConfig()
    env_cfg = load_env_cfg(source_cfg.task)
    configure_recovery_height_reward(env_cfg, source_cfg.recovery_height_reward)
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = args.seed
    env_cfg.observations["actor"].enable_corruption = False
    env_cfg.events["init_motion_loader"].params["delay_reset_env_ratio"] = 1.
    env_cfg.events["push_robot"].interval_range_s = (.001, .001)
    env_cfg.events["push_robot"].params["velocity_range"] = {"x": (.25, .25)}
    env = CaptureEnv(cfg=env_cfg, device=args.device)
    env.adapter = StateAdapter(env, recovery_counter_semantics=source_cfg.recovery_counter_semantics)
    interval_control = {"enabled": False}
    original_event_apply = env.event_manager.apply
    def controlled_events(*pos, **kwargs):
        mode = kwargs.get("mode", pos[0] if pos else None)
        if mode == "interval" and not interval_control["enabled"]:
            return None
        return original_event_apply(*pos, **kwargs)
    env.event_manager.apply = controlled_events
    report = {"checkpoint": str(args.checkpoint) if args.checkpoint else None,
              "device": args.device, "num_envs": args.num_envs, "maximum_physical_step_calls": physical_steps,
              "recovery_counter_semantics": source_cfg.recovery_counter_semantics,
              "recovery_height_reward": source_cfg.recovery_height_reward}
    try:
        env.reset()
        report["interfaces"] = check_interfaces(env, args.steps, interval_control)
        if modules:
            report["gradient_directions"] = check_directions(env, modules, args.horizon, args.directions, args.epsilon)
    finally:
        env.close()
    output = args.output or REPO/"logs"/"wmac_checks"/f"{datetime.now():%Y-%m-%d_%H-%M-%S_%f}.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2, allow_nan=False)+"\n"
    output.write_text(text)
    print(text)
    print(f"[WMAC physics check] report={output.resolve()}")


if __name__ == "__main__":
    main()
