"""Construct the saved policy architecture without changing legacy MLP keys."""

from __future__ import annotations

from collections.abc import Mapping
import math

from .networks import Actor


def _get(cfg, name, default):
    return cfg.get(name, default) if isinstance(cfg, Mapping) else getattr(cfg, name, default)


def _widths(value, name):
    if not isinstance(value, (tuple, list)) or any(
            isinstance(width, bool) or not isinstance(width, int) or width < 1 for width in value):
        raise ValueError(f"{name} must contain positive integer widths")
    return list(value)


def actor_architecture(cfg) -> dict:
    """Canonical JSON-compatible forward architecture; ignore unused MoE fields."""
    kind = _get(cfg, "actor_type", "mlp")
    if kind not in ("mlp", "moe"):
        raise ValueError(f"Unknown actor_type: {kind!r}; choose mlp or moe")
    limit = _get(cfg, "action_limit", 5.)
    if limit is not None and (isinstance(limit, bool) or not isinstance(limit, (int, float))
                              or not math.isfinite(limit) or limit <= 0):
        raise ValueError("Actor action_limit must be finite and positive or None")
    result = {"actor_type": kind,
              "policy_hidden": _widths(_get(cfg, "policy_hidden", (512, 256, 128)), "policy_hidden"),
              "action_limit": float(limit) if limit is not None else None}
    if kind == "mlp" and _get(cfg, "reference_alignment", "legacy_v1") == "official_details_v1":
        result.update(activation="gelu", layer_norm=True, action_limit=None)
    if kind == "moe":
        variant = _get(cfg, "moe_variant", "legacy")
        if variant not in ("hiking", "legacy"):
            raise ValueError("moe_variant must be hiking or legacy")
        if variant == "hiking":
            required = {"moe_num_experts": 4, "moe_top_k": 4,
                        "moe_expert_hidden": (256, 128, 64),
                        "moe_temperature": 1., "moe_load_balance_coef": 0.,
                        "moe_router_z_coef": 0., "moe_init_noise_std": 0.}
            for name, expected in required.items():
                value = _get(cfg, name, expected)
                if isinstance(expected, tuple) and isinstance(value, (list, tuple)):
                    value = tuple(value)
                if value != expected:
                    raise ValueError(f"Hiking alignment requires {name}={expected!r}")
            gate_hidden = _widths(_get(cfg, "moe_gate_hidden", (256,)), "moe_gate_hidden")
            if gate_hidden not in ([256], []):
                raise ValueError("Hiking gate hidden units must be [256] (paper default) or [] (historical checkpoint)")
            return dict(actor_type="moe", moe_variant="hiking", policy_hidden=[256, 128, 64],
                        action_limit=None, moe_num_experts=4, moe_top_k=4,
                        moe_gate_hidden=gate_hidden, moe_temperature=1., observation_normalization=False)
        experts, selected = _get(cfg, "moe_num_experts", 4), _get(cfg, "moe_top_k", 2)
        if (isinstance(experts, bool) or not isinstance(experts, int) or experts < 2
                or isinstance(selected, bool) or not isinstance(selected, int) or not 2 <= selected <= experts):
            raise ValueError("MoE requires integer num_experts >= top_k >= 2; top-1 routing is not supported")
        temperature = _get(cfg, "moe_temperature", 1.)
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("moe_temperature must be finite and positive")
        result.update(moe_num_experts=experts, moe_top_k=selected,
                      moe_gate_hidden=_widths(_get(cfg, "moe_gate_hidden", (256,)), "moe_gate_hidden"),
                      moe_temperature=float(temperature))
    return result


def create_actor(cfg, obs_dim=384, action_dim=29):
    """Return the original Actor for MLP; import sparse routing only on request."""
    architecture = actor_architecture(cfg)
    options = dict(obs_dim=obs_dim, action_dim=action_dim, hidden=tuple(architecture["policy_hidden"]),
                   action_limit=architecture["action_limit"],
                   output_init_scale=_get(cfg, "policy_output_init_scale", .01),
                   zero_output_bias=_get(cfg, "policy_output_zero_bias", True))
    if architecture["actor_type"] == "mlp":
        if "activation" in architecture:
            options.update(activation=architecture["activation"], layer_norm=architecture["layer_norm"])
        return Actor(**options)
    if architecture.get("moe_variant") == "hiking":
        from .hiking import HikingMoEActor
        return HikingMoEActor(obs_dim=obs_dim, action_dim=action_dim,
                              gate_hidden=tuple(architecture["moe_gate_hidden"]))
    from .moe import SparseMoEActor
    return SparseMoEActor(**options, num_experts=architecture["moe_num_experts"],
                          top_k=architecture["moe_top_k"],
                          gate_hidden=tuple(architecture["moe_gate_hidden"]),
                          temperature=architecture["moe_temperature"])
