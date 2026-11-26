"""Opt-in fixes for abrupt, off-reference get-ups under WMAC + AMP.

Every option here defaults to the historical behaviour, so existing profiles and
checkpoints train exactly as before. The options, by the problem they address:

Actor update dominated by the terminal value on recovery starts
    recovery_value_grad_scale     Scales d V(x_H) / d x_H for rollouts that start in
                                  recovery. The value itself (objective, critic
                                  targets) is unchanged.
    recovery_height_reward_ratio  Replaces the task's delayed-recovery weight of
                                  track_root_height (3.5), in the simulator reward and
                                  in imagination alike.
    recovery_tempo_mode           "reference_limits_v1": while a delayed-recovery
                                  environment is below the upright height, penalise
                                  base rise speed above a limit and each joint's speed
                                  above that joint's percentile in the Recovery
                                  reference clips. Same expression for the simulator
                                  reward and for imagined rewards.

Style gradient taken where the discriminator was never trained
    discriminator_imagined_fraction  Share of the discriminator's negatives drawn from
                                  the previous actor update's imagined transitions.
                                  They are a by-product of that update; no rollout is
                                  added and the batch size is unchanged.
    discriminator_penalty_points  "interpolated": the zero-centred gradient penalty is
                                  evaluated between expert and policy samples instead
                                  of on expert samples only (0-GP of Thanh-Tung et al.,
                                  ICLR 2019; InstantAMP uses the WGAN form).

World-model error in recovery
    recovery_refresh_seconds      A delayed-recovery environment that has been up for
                                  this long times out and restarts from a new Recovery
                                  frame. Without it such an environment walks on until
                                  the task's 20 s limit, and recovery makes up about
                                  1 % of the simulator data the world model learns from.
    wm_recovery_fraction          Recovery share of world-model training windows.
    recovery_model_error_gate     Leaves recovery starts out of the actor objective
                                  while the recovery model error is above this value.
    recovery_max_action_shift     Rejects an actor step that moves the actions on
                                  recovery starts by more than this (RMS).

Get-up route: the task reward pays for standing up as early as possible
    recovery_patience_seconds     For this long after a delayed-recovery environment
                                  goes down (its bad-pose counter), track_root_height
                                  pays a constant ``recovery_patience_reward`` instead
                                  of ratio * height reward, in the simulator and in
                                  imagination. Being down then costs little more than
                                  standing, so the task reward no longer pays for the
                                  fastest route up; after the window the usual height
                                  reward and the task's deadline apply. The reward must
                                  stay below what standing earns (the sum of the
                                  positive tracking weights), so lying down never pays.
    recovery_style_weight         Style weight for transitions of a delayed-recovery
                                  environment that is below the upright height or in a
                                  bad-pose countdown (the tempo cost's predicate). The
                                  part above ``style_weight`` is centred: it multiplies
                                  (style reward - its value at the robot's average
                                  score), so a better-than-average get-up step earns
                                  more and a worse one less, while merely staying down
                                  earns nothing extra. (The centred-tanh style reward
                                  is always positive; weighting all of it would pay the
                                  robot for being on the floor.) Requires
                                  amp_reward_mapping instantamp_centered_tanh_v1.
                                  Imagined objective and critic targets alike.

Reference data
    reference_motion_root         A motion directory (WalkandRun/, Recovery/) used for
                                  resets, tempo limits and AMP experts in place of the
                                  task's own, e.g. one written by
                                  scripts/clean_recovery_reference.py.

A checkpoint trained with any of them records ``recovery_style_protocol`` and
resumes only under the same settings. ``allow_recovery_style_migration`` forks an
existing checkpoint into a run with different settings.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .state import joint_velocities


RECOVERY_TEMPO_MODES = ("none", "reference_limits_v1")
RECOVERY_TEMPO_REWARD_TERM = "wmac_recovery_tempo_penalty"
RECOVERY_REFRESH_TERM = "wmac_recovery_refresh"
DISCRIMINATOR_PENALTY_POINTS = ("expert", "interpolated")
RECOVERY_STYLE_VERSION = 14
RECOVERY_PATIENCE_VERSION = 15  # first implementation with patience and recovery style weight
MIGRATION_FLAG = "allow_recovery_style_migration"
# Historical values. A configuration equal to these trains exactly as before.
DEFAULTS = {
    "recovery_value_grad_scale": 1.0,
    "recovery_height_reward_ratio": None,
    "recovery_tempo_mode": "none",
    "recovery_tempo_base_rise_limit": 0.9,
    "recovery_tempo_base_rise_coef": 10.0,
    "recovery_tempo_joint_speed_percentile": 99.0,
    "recovery_tempo_joint_speed_coef": 0.02,
    "discriminator_imagined_fraction": 0.0,
    "discriminator_penalty_points": "expert",
    "recovery_refresh_seconds": None,
    "wm_recovery_fraction": None,
    "recovery_model_error_gate": None,
    "recovery_max_action_shift": None,
    "recovery_patience_seconds": None,
    "recovery_patience_reward": 2.0,
    "recovery_style_weight": None,
    "reference_motion_root": None,
}
_TEMPO_KEYS = ("recovery_tempo_base_rise_limit", "recovery_tempo_base_rise_coef",
               "recovery_tempo_joint_speed_percentile", "recovery_tempo_joint_speed_coef")
_PATIENCE_KEYS = ("recovery_patience_seconds", "recovery_patience_reward")
# Settings whose change makes stored reward labels or the critic's bound stale.
_REWARD_KEYS = ("recovery_height_reward_ratio", "recovery_tempo_mode", *_TEMPO_KEYS, *_PATIENCE_KEYS,
                "recovery_style_weight")
# Keys a checkpoint older than RECOVERY_PATIENCE_VERSION cannot have used.
_VERSION_15_KEYS = (*_PATIENCE_KEYS, "recovery_style_weight")


def _get(cfg, key):
    return cfg.get(key, DEFAULTS[key]) if isinstance(cfg, Mapping) else getattr(cfg, key, DEFAULTS[key])


def _get_task(cfg):
    return cfg["task"] if isinstance(cfg, Mapping) else cfg.task


def _number(value, name, *, low=None, high=None, low_open=False, optional=False):
    if value is None and optional:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number" + (" or None" if optional else ""))
    if low is not None and (value <= low if low_open else value < low):
        raise ValueError(f"{name} must be {'>' if low_open else '>='} {low}")
    if high is not None and value > high:
        raise ValueError(f"{name} must be <= {high}")
    return float(value)


def recovery_style_protocol(cfg) -> dict:
    """Validated settings of this module; keys of an inactive tempo cost are left out.

    The reference motion directory is identified by content, not by path, so it is
    recorded separately (``motion_root_fingerprint``).
    """
    mode = _get(cfg, "recovery_tempo_mode")
    if mode not in RECOVERY_TEMPO_MODES:
        raise ValueError(f"recovery_tempo_mode must be one of {RECOVERY_TEMPO_MODES}")
    points = _get(cfg, "discriminator_penalty_points")
    if points not in DISCRIMINATOR_PENALTY_POINTS:
        raise ValueError(f"discriminator_penalty_points must be one of {DISCRIMINATOR_PENALTY_POINTS}")
    result = {
        "recovery_value_grad_scale": _number(_get(cfg, "recovery_value_grad_scale"),
                                             "recovery_value_grad_scale", low=0., high=1.),
        "recovery_height_reward_ratio": _number(_get(cfg, "recovery_height_reward_ratio"),
                                                "recovery_height_reward_ratio", low=0., optional=True),
        "recovery_tempo_mode": mode,
        "discriminator_imagined_fraction": _number(_get(cfg, "discriminator_imagined_fraction"),
                                                   "discriminator_imagined_fraction", low=0., high=1.),
        "discriminator_penalty_points": points,
        "recovery_refresh_seconds": _number(_get(cfg, "recovery_refresh_seconds"),
                                            "recovery_refresh_seconds", low=0., low_open=True, optional=True),
        "wm_recovery_fraction": _number(_get(cfg, "wm_recovery_fraction"), "wm_recovery_fraction",
                                        low=0., high=1., optional=True),
        "recovery_model_error_gate": _number(_get(cfg, "recovery_model_error_gate"),
                                             "recovery_model_error_gate", low=0., low_open=True, optional=True),
        "recovery_max_action_shift": _number(_get(cfg, "recovery_max_action_shift"),
                                             "recovery_max_action_shift", low=0., low_open=True, optional=True),
        "recovery_patience_seconds": _number(_get(cfg, "recovery_patience_seconds"),
                                             "recovery_patience_seconds", low=0., low_open=True, optional=True),
        "recovery_style_weight": _number(_get(cfg, "recovery_style_weight"), "recovery_style_weight",
                                         low=0., optional=True),
    }
    patience_reward = _number(_get(cfg, "recovery_patience_reward"), "recovery_patience_reward",
                              low=0., low_open=True)
    if result["recovery_patience_seconds"] is not None:
        result["recovery_patience_reward"] = patience_reward
    tempo = {
        "recovery_tempo_base_rise_limit": _number(_get(cfg, "recovery_tempo_base_rise_limit"),
                                                  "recovery_tempo_base_rise_limit", low=0.),
        "recovery_tempo_base_rise_coef": _number(_get(cfg, "recovery_tempo_base_rise_coef"),
                                                 "recovery_tempo_base_rise_coef", low=0.),
        "recovery_tempo_joint_speed_percentile": _number(
            _get(cfg, "recovery_tempo_joint_speed_percentile"),
            "recovery_tempo_joint_speed_percentile", low=0., low_open=True, high=100.),
        "recovery_tempo_joint_speed_coef": _number(_get(cfg, "recovery_tempo_joint_speed_coef"),
                                                   "recovery_tempo_joint_speed_coef", low=0.),
    }
    if mode != "none":
        result.update(tempo)
    return result


DEFAULT_PROTOCOL = recovery_style_protocol({})


def recovery_style_enabled(cfg) -> bool:
    """Whether a run uses any option of this module (the motion root is one of them)."""
    return recovery_style_protocol(cfg) != DEFAULT_PROTOCOL or _get(cfg, "reference_motion_root") is not None


def recovery_tempo_enabled(cfg) -> bool:
    return _get(cfg, "recovery_tempo_mode") != "none"


def recovery_patience_enabled(cfg) -> bool:
    return _get(cfg, "recovery_patience_seconds") is not None


def implementation_version(cfg) -> int | None:
    """Checkpoint implementation version these options need (None: none used)."""
    protocol = recovery_style_protocol(cfg)
    if any(protocol.get(key, DEFAULTS[key]) != DEFAULTS[key] for key in _VERSION_15_KEYS
           if key != "recovery_patience_reward"):
        return RECOVERY_PATIENCE_VERSION
    return RECOVERY_STYLE_VERSION if recovery_style_enabled(cfg) else None


def checkpoint_recovery_style_protocol(payload) -> dict:
    """Checkpoints written before these options existed used the historical objective."""
    saved = payload.get("config", {})
    if not isinstance(saved, Mapping):
        raise ValueError("Checkpoint config must be a mapping")
    result = recovery_style_protocol(saved)
    recorded = payload.get("recovery_style_protocol")
    if result != DEFAULT_PROTOCOL and (payload.get("implementation_version", 1) < RECOVERY_STYLE_VERSION
                                       or recorded is None):
        raise ValueError(f"Recovery-style options require implementation_version >= {RECOVERY_STYLE_VERSION} "
                         "and explicit checkpoint protocol metadata")
    if recorded is not None:
        # Records written before RECOVERY_PATIENCE_VERSION have no entry for options added
        # since; those options were off.
        recorded = {**{key: DEFAULT_PROTOCOL[key] for key in _VERSION_15_KEYS if key in DEFAULT_PROTOCOL},
                    **recorded}
    if recorded is not None and recorded != result:
        raise ValueError("Checkpoint recovery-style protocol conflicts with its saved config")
    if implementation_version(saved) == RECOVERY_PATIENCE_VERSION and \
            payload.get("implementation_version", 1) < RECOVERY_PATIENCE_VERSION:
        raise ValueError(f"Recovery patience or recovery style weight require implementation_version >= "
                         f"{RECOVERY_PATIENCE_VERSION}")
    return result


def recovery_style_changes(payload, cfg) -> dict:
    source, target = checkpoint_recovery_style_protocol(payload), recovery_style_protocol(cfg)
    return {key: {"source": source.get(key), "target": target.get(key)}
            for key in sorted(set(source) | set(target)) if source.get(key) != target.get(key)}


def validate_recovery_style_resume(payload, cfg) -> dict:
    """Return the authorised setting changes of a resume (empty for an ordinary one)."""
    changes = recovery_style_changes(payload, cfg)
    if changes and getattr(cfg, MIGRATION_FLAG, False) is not True:
        raise ValueError(f"Recovery-style settings differ from the checkpoint: {changes}. Ordinary resume keeps "
                         "the saved settings; pass --allow-recovery-style-migration to fork the checkpoint "
                         "into a separate experiment with the new ones.")
    return changes


def changes_reward(changes: Mapping) -> bool:
    return any(key in changes for key in _REWARD_KEYS)


def recovery_start_mask(state: torch.Tensor, context: torch.Tensor, recovery_height: float) -> torch.Tensor:
    """Starts that replay, statistics and evaluation count as recovery."""
    return (state[..., 2] < recovery_height) | (context[..., -1] > 0)


def recovery_style_mask(state: torch.Tensor, context: torch.Tensor, upright_height: float) -> torch.Tensor:
    """Transitions that ``recovery_style_weight`` and the tempo cost apply to (by their start)."""
    return (context[..., -2] > .5) & ((state[..., 2] < upright_height) | (context[..., -1] > 0))


def weighted_style(cfg, style: torch.Tensor, state: torch.Tensor, context: torch.Tensor,
                   upright_height: float) -> torch.Tensor:
    """Style part of the objective: ``style_weight * style``, plus on getting-up
    transitions ``(recovery_style_weight - style_weight) * (style - centre)``.

    ``centre`` is the centred-tanh reward at the robot's average score
    (amp_reward_coef / 2). Without the option this is exactly the historical
    ``style_weight * style``.
    """
    weight = cfg.style_weight
    recovery = getattr(cfg, "recovery_style_weight", None)
    if recovery is None:
        return weight * style
    mask = recovery_style_mask(state, context, upright_height)
    extra = (float(recovery) - float(weight)) * (style - .5 * float(cfg.amp_reward_coef))
    return weight * style + torch.where(mask, extra, torch.zeros_like(extra))


def scale_recovery_gradient(features: torch.Tensor, recovery: torch.Tensor, scale: float,
                            record: list | None = None) -> torch.Tensor:
    """Scale the gradient reaching ``features`` on recovery rows; values are unchanged.

    ``record`` receives, at backward time, the mean per-row gradient norm of the
    walking and of the recovery rows before scaling (NaN for an empty group).
    """
    if not features.requires_grad:
        return features
    features = features.view_as(features)  # a private node; the hook touches no other consumer
    factor = torch.where(recovery, scale, 1.).to(features.dtype).unsqueeze(-1)

    def hook(gradient):
        if record is not None:
            norms = gradient.detach().float().norm(dim=-1)
            count = recovery.sum()
            walking = (norms * ~recovery).sum() / (recovery.numel() - count).clamp_min(1)
            fallen = (norms * recovery).sum() / count.clamp_min(1)
            nan = norms.new_full((), float("nan"))
            record.append(torch.stack((torch.where(count < recovery.numel(), walking, nan),
                                       torch.where(count > 0, fallen, nan))))
        return gradient * factor if scale != 1. else None

    features.register_hook(hook)
    return features


class RecoveryTempoCost(nn.Module):
    """Cost rate for rising or moving joints faster than the reference get-up.

    Active on a transition whose start is a delayed-recovery environment that is
    below ``upright_height`` or inside a bad-pose countdown. Units: (m/s)^2 times the
    rise coefficient plus (rad/s)^2 times the joint coefficient; the caller applies
    the task's dt scale once, as for every other reward term.
    """

    def __init__(self, joint_speed_limit, *, base_rise_limit: float, base_rise_coef: float,
                 joint_speed_coef: float, upright_height: float, source: Mapping | None = None):
        super().__init__()
        limit = torch.as_tensor(joint_speed_limit, dtype=torch.float32).detach().clone()
        if limit.ndim != 1 or not bool(torch.isfinite(limit).all()) or not bool((limit > 0).all()):
            raise ValueError("Joint speed limits must be a finite positive vector")
        self.register_buffer("joint_speed_limit", limit)
        self.base_rise_limit, self.base_rise_coef = float(base_rise_limit), float(base_rise_coef)
        self.joint_speed_coef, self.upright_height = float(joint_speed_coef), float(upright_height)
        self.source = dict(source or {})

    @classmethod
    def from_reference(cls, recovery_dir, cfg, *, profile, joint_names):
        limits, source = reference_joint_speed_limits(
            recovery_dir, _get(cfg, "recovery_tempo_joint_speed_percentile"),
            profile=profile, joint_names=joint_names)
        return cls(limits, base_rise_limit=_get(cfg, "recovery_tempo_base_rise_limit"),
                   base_rise_coef=_get(cfg, "recovery_tempo_base_rise_coef"),
                   joint_speed_coef=_get(cfg, "recovery_tempo_joint_speed_coef"),
                   upright_height=profile.upright_height, source=source)

    def active(self, state, context):
        return recovery_style_mask(state, context, self.upright_height)

    def terms(self, state, next_state, context) -> dict[str, torch.Tensor]:
        active = self.active(state, context).to(next_state.dtype)
        rise = (next_state[..., 9] - self.base_rise_limit).clamp_min(0).square()
        joints = (joint_velocities(next_state).abs() - self.joint_speed_limit).clamp_min(0).square().sum(-1)
        return {"base_rise_excess": rise * active, "joint_speed_excess": joints * active}

    def forward(self, state, next_state, context) -> torch.Tensor:
        terms = self.terms(state, next_state, context)
        return (self.base_rise_coef * terms["base_rise_excess"]
                + self.joint_speed_coef * terms["joint_speed_excess"])

    def metadata(self) -> dict:
        result = {
            "formula": "recovery_tempo_reference_limits_v1",
            "definition": "active * (rise_coef * relu(next_base_vz - rise_limit)^2 "
                          "+ joint_coef * sum(relu(|next_joint_velocity| - joint_limit)^2))",
            "active": "delayed-recovery flag and (start height < upright_height or bad-pose counter > 0)",
            "base_rise_limit_m_s": self.base_rise_limit, "base_rise_coef": self.base_rise_coef,
            "joint_speed_coef": self.joint_speed_coef, "upright_height_m": self.upright_height,
            "joint_speed_limit_rad_s": [round(float(value), 6) for value in self.joint_speed_limit.cpu()],
            "source": self.source,
        }
        result["sha256"] = hashlib.sha256(
            json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return result


def reference_joint_speed_limits(recovery_dir, percentile: float, *, profile, joint_names):
    """Per-joint percentile of |joint velocity| over the Recovery reference clips.

    A left/right pair shares the larger of its two values: which side a reference
    performer happened to push off with is not a property of the robot.
    """
    names = list(joint_names)
    files = sorted(Path(recovery_dir).glob("*.npz"))
    if not files:
        raise ValueError(f"Recovery tempo limits need the Recovery reference clips; none in {recovery_dir}")
    speeds, hashes = [], {}
    for path in files:
        with np.load(path, allow_pickle=False) as motion:
            velocity = np.abs(np.asarray(motion["joint_vel"], dtype=np.float64))
        if velocity.ndim != 2 or velocity.shape[1] != len(names):
            raise ValueError(f"{path.name} has {velocity.shape[-1]} joints, the robot has {len(names)}")
        speeds.append(velocity)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    limit = np.percentile(np.concatenate(speeds), percentile, axis=0)
    partner = [names.index(profile.counterpart(name, names)) for name in names]
    limit = np.maximum(limit, limit[partner])
    source = {"clips_sha256": hashes, "percentile": float(percentile), "left_right": "max of the pair"}
    return torch.as_tensor(limit, dtype=torch.float32), source


def physical_recovery_tempo_cost(env) -> torch.Tensor:
    """Reward-manager term: the same cost on the simulator transition, before reset."""
    start = getattr(env, "_wmac_step_start", None)
    module = getattr(env, "_wmac_recovery_tempo", None)
    if start is None or module is None:
        raise RuntimeError("Recovery tempo cost requires CaptureEnv.step and an initialised WMAC runner")
    with torch.no_grad():
        return module(start["state"], env.adapter.state(), start["context"])


def delayed_recovery_refresh(env, seconds: float, recovery_height: float) -> torch.Tensor:
    """Time-out term: a delayed-recovery environment has been up for ``seconds``.

    "Up" is the complement of the recovery predicate used for replay and
    statistics: root at or above ``recovery_height`` and no bad-pose countdown.
    """
    manager = env.termination_manager
    delayed = getattr(manager, "_delay_env_mask", None)
    if delayed is None:
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    streak = getattr(env, "_wmac_up_steps", None)
    if streak is None or streak.shape != delayed.shape:
        streak = torch.zeros_like(delayed, dtype=torch.long)
    height = env.scene["robot"].data.root_link_pos_w[:, 2] - env.scene.env_origins[:, 2]
    up = (height >= recovery_height) & (manager._delay_counters == 0)
    # The first step of an episode starts a new count.
    streak = torch.where(up, torch.where(env.episode_length_buf <= 1, 0, streak) + 1, 0)
    env._wmac_up_steps = streak
    return delayed & (streak * env.step_dt >= seconds)


def configure_recovery_style_env(cfg, env_cfg) -> None:
    """Apply the reward and reset settings to a task configuration before the environment exists."""
    protocol = recovery_style_protocol(cfg)
    if protocol["recovery_refresh_seconds"] is not None:
        from mjlab.managers.termination_manager import TerminationTermCfg
        from src.robot_profiles import profile_for_task
        if RECOVERY_REFRESH_TERM in env_cfg.terminations:
            raise ValueError("Recovery refresh term was configured twice")
        env_cfg.terminations[RECOVERY_REFRESH_TERM] = TerminationTermCfg(
            func=delayed_recovery_refresh, time_out=True,
            params={"seconds": protocol["recovery_refresh_seconds"],
                    "recovery_height": profile_for_task(_get_task(cfg)).recovery_height})
    ratio = protocol["recovery_height_reward_ratio"]
    if ratio is not None:
        term = env_cfg.rewards["track_root_height"]
        if not term.params.get("mask_delay", False):
            raise ValueError("recovery_height_reward_ratio needs the delayed-recovery height reward")
        term.params = {**term.params, "delay_env_rew_ratio": ratio}
    seconds = protocol["recovery_patience_seconds"]
    if seconds is not None:
        term = env_cfg.rewards["track_root_height"]
        if not term.params.get("mask_delay", False):
            raise ValueError("recovery_patience_seconds needs the delayed-recovery height reward")
        loader = env_cfg.events["init_motion_loader"].params
        deadline = int(loader.get("max_delay_steps", 0))
        if deadline <= 0 or float(loader.get("delay_reset_env_ratio", 0.)) <= 0:
            raise ValueError("recovery_patience_seconds needs delayed-recovery environments")
        steps = patience_steps(seconds, env_cfg.decimation * env_cfg.sim.mujoco.timestep)
        if steps >= deadline:
            raise ValueError(f"recovery_patience_seconds ({steps} steps) must end before the recovery deadline "
                             f"({deadline} steps)")
        ceiling = standing_reward_ceiling(env_cfg.rewards)
        if not protocol["recovery_patience_reward"] < ceiling:
            raise ValueError(f"recovery_patience_reward must be below what standing earns ({ceiling:g}); "
                             "otherwise lying down pays as well as standing")
        term.params = {**term.params, "patience_reward": protocol["recovery_patience_reward"],
                       "patience_steps": steps}
    if protocol["recovery_tempo_mode"] != "none":
        from mjlab.managers.reward_manager import RewardTermCfg
        if not env_cfg.scale_rewards_by_dt:
            raise ValueError("Recovery tempo cost requires the existing dt-scaled task rewards")
        if RECOVERY_TEMPO_REWARD_TERM in env_cfg.rewards:
            raise ValueError("Recovery tempo reward term was configured twice")
        env_cfg.rewards[RECOVERY_TEMPO_REWARD_TERM] = RewardTermCfg(func=physical_recovery_tempo_cost, weight=-1.)


def patience_steps(seconds: float, step_dt: float) -> int:
    steps = round(seconds / step_dt)
    if steps < 1:
        raise ValueError("recovery_patience_seconds is shorter than one control step")
    return steps


def standing_reward_ceiling(rewards: Mapping) -> float:
    """Most an upright environment can earn per step: the positive (bounded) tracking terms."""
    return sum(float(term.weight) for name, term in rewards.items()
               if term is not None and float(term.weight) > 0 and name not in ("track_root_height", "is_terminated"))


def resolve_motion_root(root: str, repository: Path) -> Path:
    path = Path(root).expanduser()
    path = path if path.is_absolute() else Path(repository) / path
    path = path.resolve()
    for group in ("WalkandRun", "Recovery"):
        if not any((path / group).glob("*.npz")):
            raise ValueError(f"reference_motion_root needs {group}/*.npz under {path}")
    return path


def motion_root_fingerprint(root) -> dict:
    """Content of a reference motion directory, for checkpoint provenance."""
    root = Path(root)
    files = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
             for group in ("WalkandRun", "Recovery") for path in sorted((root / group).glob("*.npz"))}
    digest = hashlib.sha256(json.dumps(files, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return {"files": len(files), "content_sha256": digest}
