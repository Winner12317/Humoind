"""Pinned official details adapted to the existing flat AMP/BPTT method.

This protocol deliberately retains WMAC's local-residual dynamics, AMP reward,
recovery state machine and policy observations. It does not import tracking
goals, the latent actor, GenCoord integration or the perceptive project.
"""
from __future__ import annotations

from collections.abc import Mapping
import math


OFFICIAL_REPOSITORY = "https://github.com/HybridRobotics/supermimic"
OFFICIAL_COMMIT = "abc124c1bd5976341034668af3d12c087f2f1f55"
MIRROR_REFERENCE_COMMIT = "92ca7bb8a3d8ea959d934fc13b5aae7e30317757"
ALIGNMENTS = ("legacy_v1", "official_details_v1")
AMP_REWARD_MAPPINGS = ("clipped_lsgan_v1", "instantamp_centered_tanh_v1")
INSTANTAMP_REPOSITORY = "https://github.com/RoboticsResearchHubVN/InstantAMP"
INSTANTAMP_COMMIT = "9fc8031055a18d58e84d9de90639660e7aa6b491"
INSTANTAMP_SOURCE = INSTANTAMP_REPOSITORY + "/blob/" + INSTANTAMP_COMMIT + "/instantamp/amp.py#L106-L119"
_BASE_PROTOCOL_KEYS = ("reference_alignment", "optimizer_algorithm", "mirror_walk_motions")
SMOOTH_REWARD_MODES = ("none", "host_humanup_v1")
SMOOTH_REWARD_FORMULA = "host_humanup_rates_pd_request_v1"
HOST_COMMIT = "70bb580949a336a920833700e4b5dc3bf7fe87ce"
HUMANUP_COMMIT = "7516e0f27e6f4d1e7365cf64ea577a78247bd8cb"
_SMOOTH_COEFFICIENTS = {
    "smooth_action_second_coef": .01,
    "smooth_joint_velocity_coef": .00025,
    "smooth_torque_coef": .00025,
    "smooth_power_coef": .000025,
}
_SMOOTH_CONFIG_KEYS = ("smooth_reward_mode", *_SMOOTH_COEFFICIENTS)
IMAGINED_FOOT_SLIP_GEOMETRY = "geometry_detached_support_v1"
IMAGINED_FOOT_SLIP_INSTANTAMP = "geometry_instantamp_full_support_v1"
IMAGINED_FOOT_SLIP_GEOMETRIC_MODES = (IMAGINED_FOOT_SLIP_GEOMETRY, IMAGINED_FOOT_SLIP_INSTANTAMP)
_INSTANTAMP_FOOT_SLIP_PARAMETERS_M = (.001, .001)
_FOOT_SLIP_MODE_KEY = "imagined_recovery_foot_slip"
_FOOT_SLIP_GEOMETRY_KEYS = ("recovery_foot_slip_contact_height_m", "recovery_foot_slip_softness_m")
SMOOTH_REWARD_COEFFICIENT_KEYS = tuple(_SMOOTH_COEFFICIENTS)
# Resume-time execution choices may be shared by a bounded A/B pilot. All
# learning rules, data distributions, normalizers and model settings stay
# locked; new config fields are rejected by default until explicitly reviewed.
SMOOTH_RETUNE_EXECUTION_KEYS = frozenset((
    "resume", "run_name", "log_root", "iterations", "num_envs",
    "actor_batch_size", "actor_microbatch_size", "compile", "precision",
    "save_interval", "log_interval", "evaluation_interval",
    "device", "replay_device", "checkpoint_replay", "benchmark_sync",
))


def smooth_reward_protocol(cfg) -> dict:
    """Version reward formulas independently of the unchanged dynamics model.

    Torque uses the existing PD request before clipping to retain its Actor
    gradient. Power uses the analytic limited/frictional command. Neither is a
    prediction of physical 5 ms substep actuator_force. All four terms follow
    the existing task reward scale before the external task weight.
    """
    def get(key, default):
        return cfg.get(key, default) if isinstance(cfg, Mapping) else getattr(cfg, key, default)
    mode = get("smooth_reward_mode", "none")
    if mode not in SMOOTH_REWARD_MODES:
        raise ValueError(f"Unknown smooth_reward_mode: {mode!r}")
    coefficients = {}
    for key, default in _SMOOTH_COEFFICIENTS.items():
        value = get(key, default)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
            raise ValueError(f"{key} must be finite and nonnegative")
        coefficients[key] = float(value)
    result = {"smooth_reward_mode": mode}
    if mode != "none":
        result.update(coefficients)
        result.update(
            formula_version=SMOOTH_REWARD_FORMULA,
            cost_definitions={
                "action_second": "sum((action - 2*previous_action + previous_previous_action)^2)",
                "joint_velocity": "sum(next_joint_velocity^2)",
                "torque": "l2_norm(analytic_pd_requested_torque_before_limit)",
                "power": "l2_norm(analytic_actuator_command_torque * current_joint_velocity)",
            },
            scaling="negative_costs_inside_existing_task_scale_then_task_weight",
            torque_semantics="existing_pd_request_proxy_before_limit_and_command_power_after_torque_speed_clip_and_friction_not_physical_substep_force",
            references={"HoST": HOST_COMMIT, "HumanUP": HUMANUP_COMMIT},
            action_amplitude_curriculum=False,
            world_model_changed=False,
        )
    return result


def checkpoint_smooth_reward_protocol(payload) -> dict:
    """Old checkpoints stay on their historical reward; new ones fail closed."""
    saved = payload.get("config", {})
    if not isinstance(saved, Mapping):
        raise ValueError("Checkpoint config must be a mapping")
    result = smooth_reward_protocol(saved)
    version = payload.get("implementation_version", 1)
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("Checkpoint implementation_version must be a positive integer")
    if result["smooth_reward_mode"] != "none":
        if version < 12:
            raise ValueError("Smooth reward checkpoint requires implementation_version >= 12")
        missing = set(_SMOOTH_CONFIG_KEYS) - set(saved)
        if missing or "smooth_reward_protocol" not in payload:
            raise ValueError(f"Smooth reward checkpoint requires explicit metadata: {sorted(missing)}")
    if "smooth_reward_protocol" in payload and payload["smooth_reward_protocol"] != result:
        raise ValueError("Checkpoint smooth reward protocol conflicts with saved config or formulas")
    return result


def smooth_reward_resume_transition(payload, cfg) -> str:
    """Classify an authorized unchanged/enable/retune objective before mutation.

    Enabling historical rewards and retuning active coefficients are distinct:
    a retune must retain the saved motor metadata checked by the runner.
    """
    source, target = checkpoint_smooth_reward_protocol(payload), smooth_reward_protocol(cfg)
    if source == target:
        return "unchanged"
    allowed = (source["smooth_reward_mode"] == "none"
               and target["smooth_reward_mode"] == "host_humanup_v1"
               and getattr(cfg, "allow_smooth_reward_migration", False) is True)
    if allowed:
        return "enable"
    retune = (source["smooth_reward_mode"] == target["smooth_reward_mode"] == "host_humanup_v1"
              and getattr(cfg, "allow_smooth_reward_retune", False) is True)
    if retune:
        fixed_source = {key: value for key, value in source.items() if key not in _SMOOTH_COEFFICIENTS}
        fixed_target = {key: value for key, value in target.items() if key not in _SMOOTH_COEFFICIENTS}
        if fixed_source != fixed_target:
            raise ValueError("Smooth reward retune may change only the four cost coefficients")
        if checkpoint_amp_reward_protocol(payload) != amp_reward_protocol(cfg):
            raise ValueError("Smooth reward retune cannot also change the AMP reward protocol")
        # Local import avoids the module-level checkpoint/reference dependency.
        from .checkpoints import (checkpoint_recovery_semantics, checkpoint_height_reward_semantics,
                                  resolve_checkpoint_config)
        if (checkpoint_recovery_semantics(payload) != cfg.recovery_counter_semantics
                or checkpoint_height_reward_semantics(payload) != cfg.recovery_height_reward):
            raise ValueError("Smooth reward retune cannot also change recovery semantics")
        # Resolution reads only source metadata (checkpoint_protocol), never
        # resume preflight, so this local call does not recurse into this helper.
        from dataclasses import fields
        saved_cfg = resolve_checkpoint_config(payload)
        # An archived initialization checkpoint is provenance during resume;
        # parse_config intentionally does not warmstart the restored run again.
        saved_cfg.init_policy = None
        allowed_keys = (SMOOTH_RETUNE_EXECUTION_KEYS | set(SMOOTH_REWARD_COEFFICIENT_KEYS)
                        | {"allow_smooth_reward_retune"})
        forbidden_changes = {
            field.name:{"source":getattr(saved_cfg, field.name), "target":getattr(cfg, field.name)}
            for field in fields(saved_cfg)
            if field.name not in allowed_keys and getattr(saved_cfg, field.name) != getattr(cfg, field.name)}
        if forbidden_changes:
            raise ValueError("Smooth reward retune may change only the four cost coefficients and "
                             f"explicit execution fields; forbidden configuration changes: {forbidden_changes}")
        return "retune"
    raise ValueError("Smooth reward protocol mismatch; none to host_humanup_v1 requires "
                     "--allow-smooth-reward-migration, while active coefficient changes require "
                     "--allow-smooth-reward-retune for a separate reward experiment; "
                     "ordinary smooth resume must retain all coefficients.")


def validate_smooth_reward_resume(payload, cfg) -> bool:
    """Retain the historical True-only-for-none-to-active migration contract."""
    return smooth_reward_resume_transition(payload, cfg) == "enable"


def validate_smooth_reward_retune(payload, cfg) -> bool:
    """Return True only for a separately authorized active coefficient retune."""
    return smooth_reward_resume_transition(payload, cfg) == "retune"


def imagined_foot_slip_protocol(cfg) -> dict:
    def get(key, default=None):
        return cfg.get(key, default) if isinstance(cfg, Mapping) else getattr(cfg, key, default)

    mode = get(_FOOT_SLIP_MODE_KEY, "learned_v1")
    if mode not in ("learned_v1", *IMAGINED_FOOT_SLIP_GEOMETRIC_MODES):
        raise ValueError(f"Unknown imagined recovery foot slip: {mode!r}")
    result = {_FOOT_SLIP_MODE_KEY: mode}
    height, softness = (get(key) for key in _FOOT_SLIP_GEOMETRY_KEYS)
    if mode == "learned_v1":
        if height is not None or softness is not None:
            raise ValueError("Contact height and softness require geometric imagined foot slip")
        return result
    if (get("task") not in ("Unitree-G1-AMP-Flat", "Unitree-G1-AMP-Flat-Aligned")
            or get("recovery_counter_semantics") != "hard_forward_v2"):
        raise ValueError("Geometric imagined foot slip requires flat G1 with hard_forward_v2 recovery")
    if mode == IMAGINED_FOOT_SLIP_INSTANTAMP:
        parameters = (height, softness)
        if any(value is not None and value != pinned for value, pinned in
               zip(parameters, _INSTANTAMP_FOOT_SLIP_PARAMETERS_M)):
            raise ValueError("InstantAMP full-support foot slip pins contact height and softness at 0.001 m")
        height, softness = _INSTANTAMP_FOOT_SLIP_PARAMETERS_M
    if any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
           for value in (height, softness)) or softness <= 0:
        raise ValueError("Geometric foot slip requires explicit finite contact height and positive softness in metres")
    result.update(zip(_FOOT_SLIP_GEOMETRY_KEYS, (float(height), float(softness))))
    return result


def checkpoint_imagined_foot_slip_protocol(payload) -> dict:
    result = imagined_foot_slip_protocol(payload.get("config", {}))
    metadata = payload.get("imagined_foot_slip_protocol")
    if result[_FOOT_SLIP_MODE_KEY] in IMAGINED_FOOT_SLIP_GEOMETRIC_MODES and (
            payload.get("implementation_version", 1) < 13 or metadata is None):
        raise ValueError("Geometric foot slip requires version 13+ and explicit checkpoint protocol metadata")
    if "imagined_foot_slip_protocol" in payload and metadata != result:
        raise ValueError("Checkpoint imagined foot slip protocol conflicts with its saved config")
    return result


def validate_imagined_foot_slip_resume(payload, cfg) -> bool:
    source, target = checkpoint_imagined_foot_slip_protocol(payload), imagined_foot_slip_protocol(cfg)
    if source == target:
        return False
    if not (source[_FOOT_SLIP_MODE_KEY] == "learned_v1"
            and target[_FOOT_SLIP_MODE_KEY] in IMAGINED_FOOT_SLIP_GEOMETRIC_MODES
            and getattr(cfg, "allow_imagined_foot_slip_migration", False) is True):
        raise ValueError("Imagined foot slip protocol mismatch; use --allow-imagined-foot-slip-migration "
                         "for an explicit learned-to-geometry fork. Ordinary resume keeps the saved mode and geometry.")
    from dataclasses import fields
    from .checkpoints import resolve_checkpoint_config
    saved = resolve_checkpoint_config(payload)
    saved.init_policy = None
    allowed = SMOOTH_RETUNE_EXECUTION_KEYS | {
        _FOOT_SLIP_MODE_KEY, *_FOOT_SLIP_GEOMETRY_KEYS, "allow_imagined_foot_slip_migration"}
    changed = [field.name for field in fields(saved) if field.name not in allowed
               and getattr(saved, field.name) != getattr(cfg, field.name)]
    if changed:
        raise ValueError(f"Imagined foot slip migration must be a separate reward experiment; changed: {changed}")
    return True


def amp_reward_protocol(cfg) -> dict:
    """Return versioned reward semantics, independent of the world model.

    Scale 1 is in the retained LSGAN's +/-1 target units. InstantAMP's WGAN
    scale 5 is deliberately not reused. Legacy mode ignores these new knobs.
    """
    def get(key, default):
        return cfg.get(key, default) if isinstance(cfg, Mapping) else getattr(cfg, key, default)
    mode = get("amp_reward_mapping", "clipped_lsgan_v1")
    if mode not in AMP_REWARD_MAPPINGS:
        raise ValueError(f"Unknown amp_reward_mapping: {mode!r}")
    scale, decay = get("amp_score_scale", 1.0), get("amp_score_ema_decay", .99)
    if isinstance(scale, bool) or not isinstance(scale, (int, float)) or not math.isfinite(scale) or scale <= 0:
        raise ValueError("amp_score_scale must be finite and positive")
    if isinstance(decay, bool) or not isinstance(decay, (int, float)) or not math.isfinite(decay) or not 0 <= decay < 1:
        raise ValueError("amp_score_ema_decay must be finite and in [0, 1)")
    result = {"amp_reward_mapping": mode}
    if mode == "instantamp_centered_tanh_v1":
        result.update(amp_score_scale=float(scale), amp_score_ema_decay=float(decay))
    return result


def checkpoint_amp_reward_protocol(payload) -> dict:
    """Never reinterpret an unlabelled historical checkpoint as centered AMP."""
    saved = payload.get("config", {})
    if not isinstance(saved, Mapping):
        raise ValueError("Checkpoint config must be a mapping")
    result = amp_reward_protocol(saved)
    version = payload.get("implementation_version", 1)
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("Checkpoint implementation_version must be a positive integer")
    centered = result["amp_reward_mapping"] == "instantamp_centered_tanh_v1"
    if centered:
        if version < 11:
            raise ValueError("Centered AMP checkpoint requires implementation_version >= 11")
        missing = set(result) - set(saved)
        if missing or "amp_reward_protocol" not in payload:
            raise ValueError(f"Centered AMP checkpoint requires explicit reward protocol metadata: {sorted(missing)}")
    if "amp_reward_protocol" in payload and payload["amp_reward_protocol"] != result:
        raise ValueError("Checkpoint AMP reward protocol conflicts with its saved config")
    return result


def validate_training_protocol_resume(payload, cfg) -> bool:
    """Validate before mutation; return True for an explicit AMP migration.

    Smooth reward migration has its own independent authorization. The return
    value deliberately retains the historical AMP-only contract. Call
    validate_smooth_reward_resume separately when its migration flag is needed.
    """
    source, target = checkpoint_protocol(payload), config_protocol(cfg)
    if any(source[key] != target[key] for key in _BASE_PROTOCOL_KEYS):
        raise ValueError("Training protocol mismatch: official details, optimizer algorithm and mirrored data cannot be changed by ordinary resume")
    validate_smooth_reward_resume(payload, cfg)
    validate_imagined_foot_slip_resume(payload, cfg)
    saved_reward, requested_reward = checkpoint_amp_reward_protocol(payload), amp_reward_protocol(cfg)
    if saved_reward == requested_reward:
        return False
    allowed = (saved_reward["amp_reward_mapping"] == "clipped_lsgan_v1"
               and requested_reward["amp_reward_mapping"] == "instantamp_centered_tanh_v1"
               and getattr(cfg, "allow_amp_reward_migration", False) is True)
    if not allowed:
        raise ValueError("AMP reward protocol mismatch; only explicit clipped_lsgan_v1 to "
                         "instantamp_centered_tanh_v1 migration is supported. Pass "
                         "--allow-amp-reward-migration for a separate migration experiment; "
                         "ordinary centered resume must retain its scale and EMA decay.")
    return True


def is_official(cfg) -> bool:
    value = cfg.get("reference_alignment", "legacy_v1") if isinstance(cfg, Mapping) else getattr(cfg, "reference_alignment", "legacy_v1")
    if value not in ALIGNMENTS:
        raise ValueError("Unknown SuperMimic reference alignment")
    return value == "official_details_v1"


def checkpoint_protocol(payload) -> dict:
    saved = payload.get("config", {})
    if not isinstance(saved, Mapping):
        raise ValueError("Checkpoint config must be a mapping")
    defaults = {"reference_alignment": "legacy_v1", "optimizer_algorithm": "adam", "mirror_walk_motions": False}
    version = payload.get("implementation_version", 1)
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise ValueError("Checkpoint implementation_version must be a positive integer")
    if version >= 7:
        missing = set(defaults) - set(saved)
        if missing or "training_protocol" not in payload:
            raise ValueError(f"Version7 checkpoint requires explicit training protocol fields: {sorted(missing)}")
    result = {key: saved.get(key, default) for key, default in defaults.items()}
    if result["reference_alignment"] not in ALIGNMENTS or result["optimizer_algorithm"] not in ("adam", "muon_adam") or not isinstance(result["mirror_walk_motions"], bool):
        raise ValueError("Checkpoint training protocol is invalid")
    reward = checkpoint_amp_reward_protocol(payload)
    if reward["amp_reward_mapping"] == "instantamp_centered_tanh_v1":
        result.update(reward)
    smooth = checkpoint_smooth_reward_protocol(payload)
    if smooth["smooth_reward_mode"] != "none":
        result.update({key: smooth[key] for key in _SMOOTH_CONFIG_KEYS})
    foot_slip = checkpoint_imagined_foot_slip_protocol(payload)
    if foot_slip[_FOOT_SLIP_MODE_KEY] in IMAGINED_FOOT_SLIP_GEOMETRIC_MODES:
        result.update(foot_slip)
    if "training_protocol" in payload and payload["training_protocol"] != result:
        raise ValueError("Checkpoint training protocol conflicts with its config")
    if any(result[key] != default for key, default in defaults.items()) and "training_protocol" not in payload:
        raise ValueError("Aligned or mirrored checkpoints require explicit training protocol metadata")
    return result


def config_protocol(cfg) -> dict:
    result = {"reference_alignment": cfg.reference_alignment,
              "optimizer_algorithm": cfg.optimizer_algorithm,
              "mirror_walk_motions": cfg.mirror_walk_motions}
    reward = amp_reward_protocol(cfg)
    if reward["amp_reward_mapping"] == "instantamp_centered_tanh_v1":
        result.update(reward)
    smooth = smooth_reward_protocol(cfg)
    if smooth["smooth_reward_mode"] != "none":
        result.update({key: smooth[key] for key in _SMOOTH_CONFIG_KEYS})
    foot_slip = imagined_foot_slip_protocol(cfg)
    if foot_slip[_FOOT_SLIP_MODE_KEY] in IMAGINED_FOOT_SLIP_GEOMETRIC_MODES:
        result.update(foot_slip)
    return result


def amp_value_upper_bound(adapter, cfg) -> float:
    """Official critic cap, using the retained AMP task's reward upper bound.

    All positive flat-task terms below are bounded exponentials. Unbounded
    costs may have negative weights, but a positive cost weight has no finite
    bound and is rejected instead of silently applying a tracking-specific cap.
    """
    bounded = {"track_anchor_linear_velocity", "track_anchor_angular_velocity",
               "track_root_height", "body_ang_vel_xy_l2", "is_terminated"}
    total = 0.
    for name, term in adapter.reward_cfg.items():
        weight = float(term.weight)
        if weight <= 0:
            continue
        if name not in bounded:
            raise ValueError(f"Official critic bound needs a bound for positive reward term {name}")
        params = term.params
        if name == "track_root_height":
            amplitude = max(0., float(params.get("delay_env_rew_ratio", 1.))) if params.get("mask_delay", False) else 0.
        else:
            amplitude = max(1., float(params.get("delay_env_rew_ratio", 1.))) if params.get("mask_delay", False) else 1.
        total += weight * amplitude
    max_reward = cfg.task_weight * adapter.reward_scale * total + cfg.style_weight * cfg.amp_reward_coef
    bound = max_reward / (1. - cfg.gamma)
    if not math.isfinite(bound) or bound <= 0:
        raise ValueError("Official critic cap requires a finite positive AMP reward bound")
    return bound
