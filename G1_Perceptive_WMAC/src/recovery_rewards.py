"""Shared physical/imagination recovery-height reward semantics.

The legacy version rewards height alone. ``upright_v2`` multiplies exactly the
same height term by positive root uprightness squared. This module contains no
environment state, cached kinematics, or discriminator-dependent calculation.
"""
from __future__ import annotations

import math

import torch


RECOVERY_HEIGHT_SEMANTICS = ("height_only_v1", "upright_v2")


def recovery_upright_gate(up_z: torch.Tensor) -> torch.Tensor:
    """Positive world-up alignment squared: upright one, horizontal/inverted zero."""
    return up_z.clamp(0., 1.).square()


def recovery_root_up_z(root_quaternion: torch.Tensor) -> torch.Tensor:
    """Negative projected-gravity z of a root wxyz quaternion.

    Normalization retains the same quaternion-input gradients as the WMAC state
    rotations. Inputs come from the root body's link quaternion in the physical
    MDP, not its torso quaternion or a cached gravity observation.
    """
    quaternion = root_quaternion / root_quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    return 1. - 2. * (quaternion[..., 1].square() + quaternion[..., 2].square())


def recovery_height_reward(height: torch.Tensor, desired_height: float | torch.Tensor,
                           std: float, *, up_z: torch.Tensor | None = None,
                           upright_power: float = 0.0) -> torch.Tensor:
    """Original Gaussian height term with an optional root-uprightness factor."""
    if not math.isfinite(upright_power) or upright_power < 0:
        raise ValueError("upright_power must be finite and nonnegative")
    reward = torch.exp(-(height - desired_height).square() / std**2)
    if upright_power == 0.:
        return reward
    if up_z is None:
        raise ValueError("An upright-gated height reward requires root up_z")
    gate = recovery_upright_gate(up_z) if upright_power == 2. else up_z.clamp(0., 1.).pow(upright_power)
    return reward * gate


def configure_recovery_height_reward(env_cfg, semantics: str):
    """Select explicit physical height semantics before environment construction.

    Only this reward term's parameter mapping changes. Copying the mapping
    avoids modifying another configuration that shares the original dictionary.
    """
    if semantics not in RECOVERY_HEIGHT_SEMANTICS:
        raise ValueError(f"Unknown recovery-height semantics {semantics!r}; expected {RECOVERY_HEIGHT_SEMANTICS}")
    rewards = getattr(env_cfg, "rewards", None)
    if rewards is None or "track_root_height" not in rewards:
        raise ValueError("Recovery-height semantics require a track_root_height reward term")
    term = rewards["track_root_height"]
    params = getattr(term, "params", None)
    if not isinstance(params, dict):
        raise ValueError("track_root_height must expose a reward-parameter dictionary")
    term.params = {**params, "upright_power": 0. if semantics == "height_only_v1" else 2.}
    return env_cfg
