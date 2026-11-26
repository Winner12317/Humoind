"""Resolve saved WMAC behavior without inheriting newer experiment defaults."""

from __future__ import annotations

from dataclasses import fields
from collections.abc import Mapping

from .config import WmacConfig
from .actors import actor_architecture


RECOVERY_COUNTER_SEMANTICS = ("soft_v1", "hard_forward_v2")
RECOVERY_HEIGHT_REWARD_SEMANTICS = ("height_only_v1", "upright_v2")


def _saved_semantics(payload, key, allowed, legacy):
    if not isinstance(payload, Mapping):
        raise ValueError("Expected a checkpoint mapping")
    saved = payload.get("config", {})
    if not isinstance(saved, Mapping):
        raise ValueError("Checkpoint config must be a mapping")
    top_present, config_present = key in payload, key in saved
    top, configured = payload.get(key), saved.get(key)
    if top_present and config_present and top != configured:
        raise ValueError(f"Checkpoint {key} conflicts between top-level metadata and config")
    version = payload.get("implementation_version")
    if version is not None and (isinstance(version, bool) or not isinstance(version, int) or version < 1):
        raise ValueError("Checkpoint implementation_version must be a positive integer")
    if not top_present and not config_present:
        if version is not None and version >= 4:
            raise ValueError(f"Checkpoint implementation_version >= 4 is missing {key}")
        return legacy
    value = top if top_present else configured
    if value not in allowed:
        raise ValueError(f"Unknown checkpoint {key}: {value!r}; choose from {allowed}")
    return value


def checkpoint_recovery_semantics(payload):
    """Unlabelled historical checkpoints used the soft recovery counter."""
    return _saved_semantics(payload, "recovery_counter_semantics",
                            RECOVERY_COUNTER_SEMANTICS, "soft_v1")


def checkpoint_height_reward_semantics(payload):
    """Unlabelled historical checkpoints rewarded root height alone."""
    return _saved_semantics(payload, "recovery_height_reward",
                            RECOVERY_HEIGHT_REWARD_SEMANTICS, "height_only_v1")


def checkpoint_actor_architecture(payload) -> dict:
    """Old unlabelled policies are MLPs, independent of current defaults."""
    if not isinstance(payload, Mapping) or not isinstance(payload.get("config", {}), Mapping):
        raise ValueError("Checkpoint and saved config must be mappings")
    saved = payload.get("config", {})
    configured, top = saved.get("actor_type"), payload.get("actor_type")
    if "actor_type" in saved and "actor_type" in payload and configured != top:
        raise ValueError("Checkpoint actor_type conflicts between top-level metadata and config")
    explicit = "actor_type" in saved or "actor_type" in payload
    kind = configured if "actor_type" in saved else top if "actor_type" in payload else "mlp"
    version = payload.get("implementation_version")
    if version is not None and (isinstance(version, bool) or not isinstance(version, int) or version < 1):
        raise ValueError("Checkpoint implementation_version must be a positive integer")
    if version is not None and version >= 5 and (not explicit or "actor_architecture" not in payload):
        raise ValueError("Checkpoint implementation_version >= 5 requires explicit actor_type and actor_architecture metadata")
    defaults = WmacConfig.for_profile(saved.get("profile", "legacy"))
    values = {field.name: getattr(defaults, field.name) for field in fields(defaults)}
    if kind == "moe" and "moe_variant" not in saved:
        # Every MoE saved before impl6 was the old normalized/limited sparse
        # Actor. Never reinterpret it using the new Hiking defaults.
        values.update(moe_variant="legacy", moe_top_k=2, moe_gate_hidden=(256,),
                      moe_load_balance_coef=.01, moe_router_z_coef=.001)
    values.update(saved)
    values["actor_type"] = kind
    if version is not None and version >= 6 and kind == "moe" and "moe_variant" not in saved:
        raise ValueError("Checkpoint implementation_version >= 6 requires explicit moe_variant")
    architecture = actor_architecture(values)
    if "actor_architecture" in payload:
        recorded = payload["actor_architecture"]
        if not isinstance(recorded, Mapping) or set(recorded) != set(architecture):
            raise ValueError("Checkpoint actor_architecture metadata is incomplete or has unknown fields")
        normalized = dict(recorded)
        for key in ("policy_hidden", "moe_gate_hidden"):
            if key in normalized and isinstance(normalized[key], (tuple, list)):
                normalized[key] = list(normalized[key])
        if normalized != architecture:
            raise ValueError("Checkpoint actor_architecture conflicts with its saved config")
    state = payload.get("actor", {})
    if isinstance(state, Mapping):
        has_moe = any(str(key).startswith(("experts.", "router.")) for key in state)
        has_mlp = any(str(key).startswith("net.") for key in state)
        if (kind == "mlp" and has_moe) or (kind == "moe" and has_mlp):
            raise ValueError("Checkpoint Actor tensor layout conflicts with actor_type")
    return architecture


def resolve_checkpoint_config(payload) -> WmacConfig:
    """Rebuild saved profile, tuple fields and legacy Adam betas explicitly.

    Migration authorization belongs to the current invocation and is never
    inherited from an input checkpoint. Unlabelled old behavior is resolved
    before it can be replaced by the current dataclass/profile defaults.
    """
    if not isinstance(payload, Mapping) or payload.get("format") != "amp-wmac-v1":
        raise ValueError("Expected an amp-wmac-v1 checkpoint with its saved configuration")
    saved = payload.get("config")
    if not isinstance(saved, Mapping):
        raise ValueError("The WMAC checkpoint has no saved configuration")
    counter_mode = checkpoint_recovery_semantics(payload)
    height_mode = checkpoint_height_reward_semantics(payload)
    architecture = checkpoint_actor_architecture(payload)
    cfg = WmacConfig.for_profile(saved.get("profile", "legacy"))
    if architecture["actor_type"] == "moe" and "moe_variant" not in saved:
        cfg.moe_variant = "legacy"
        cfg.moe_top_k, cfg.moe_gate_hidden = 2, (256,)
        cfg.moe_load_balance_coef, cfg.moe_router_z_coef = .01, .001
    for field in fields(cfg):
        if field.name in saved and field.name != "allow_recovery_semantics_migration":
            value = saved[field.name]
            if isinstance(getattr(cfg, field.name), tuple) and isinstance(value, (tuple, list)):
                value = tuple(value)
            setattr(cfg, field.name, value)
    cfg.recovery_counter_semantics = counter_mode
    cfg.recovery_height_reward = height_mode
    cfg.actor_type = architecture["actor_type"]
    cfg.allow_recovery_semantics_migration = False
    if "actor_betas" not in saved:
        groups = payload.get("opt_actor", {}).get("param_groups", [])
        if groups and "betas" in groups[0]:
            cfg.actor_betas = tuple(groups[0]["betas"])
    return cfg
