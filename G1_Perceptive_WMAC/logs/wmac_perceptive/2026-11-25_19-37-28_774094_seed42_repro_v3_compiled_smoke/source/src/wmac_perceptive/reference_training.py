"""Official SuperMimic critic and return limits for the reference schedule.

The network and return equations follow HybridRobotics/supermimic commit
5efbf114e11f725c31d83e217be987c13c0e4ce2 (Apache-2.0). See
THIRD_PARTY_NOTICES.md. The AMP contribution to the reward ceiling is the
local task's adaptation; the upstream task uses its positive reward weights.
"""
from __future__ import annotations

from collections.abc import Sequence
import math

import torch
from torch import nn

from src.wmac.networks import RunningNorm, mlp, run_mlp


class ReferenceCritic(nn.Module):
    """Unnormalized ``Linear → LayerNorm → GELU`` scalar value network."""

    def __init__(self, in_dim: int, hidden: Sequence[int] = (256, 128)):
        super().__init__()
        self.norm = RunningNorm(in_dim)
        self.net = mlp(in_dim, hidden, 1, nn.GELU, layer_norm=True)
        self.precision = "float32"

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # ``norm`` is only a shape/checkpoint compatibility buffer. The
        # reference critic reads the original features without normalization.
        return run_mlp(self.net, features, self.precision).squeeze(-1)


def build_reference_critic(
    in_dim: int,
    hidden: Sequence[int] = (256, 128),
    device: str | torch.device = "cpu",
) -> ReferenceCritic:
    return ReferenceCritic(in_dim, hidden).to(device)


def reference_value_ceiling(adapter, cfg) -> float:
    """Analytic discounted upper bound, with task and AMP reward scaling.

    The task reward is scaled by ``adapter.reward_scale`` exactly once. The
    bounded AMP reward already includes its own coefficient and is not
    multiplied by the simulator time step.
    """
    gamma = float(cfg.gamma)
    if not math.isfinite(gamma) or not 0.0 <= gamma < 1.0:
        raise ValueError("Reference value ceiling requires gamma in [0, 1)")
    reward_scale = float(adapter.reward_scale)
    task_weight = float(cfg.task_weight)
    style_weight = float(cfg.style_weight)
    amp_coefficient = float(cfg.amp_reward_coef)
    scales = (reward_scale, task_weight, style_weight, amp_coefficient)
    if any(not math.isfinite(value) or value < 0.0 for value in scales):
        raise ValueError("Reference reward scales must be finite and nonnegative")
    weights = [float(term.weight) for term in adapter.reward_cfg.values()]
    if any(not math.isfinite(value) for value in weights):
        raise ValueError("Reference reward weights must be finite")
    positive_max = sum(weight for weight in weights if weight > 0.0)
    if getattr(cfg, "hiking_training_version", 1) >= 3:
        # SuperMimic sums the configured positive weights and scales by dt.
        # Here AMP is a separate, bounded reward already in per-step units.
        return (task_weight * reward_scale * positive_max + style_weight * amp_coefficient) / (1.0 - gamma)
    if getattr(adapter,"full_hiking",False):
        # feet_air_time is measured in seconds, not bounded by one. The
        # original episode ends at20s. This is the real-task value bound used
        # by SuperMimic, not a cap on the predicted airtime or source reward.
        # A learned predictor can leave its target domain; validation reports
        # its error. feet_close_xy is nonpositive despite +weight.
        positive_max=2.+2.+3.+.5*20.
    standing = adapter.reward_cfg.get("stand_still")
    if standing is not None:
        # Hiking subtracts an offset before multiplying by a negative weight:
        # the default pose contributes +1.2, despite weight=-0.3. Include it
        # so SuperMimic's return cap remains an actual upper bound.
        offset = float(standing.params.get("offset", 4. if getattr(adapter,"full_hiking",False) else 1.))
        if float(standing.weight) > 0. or not math.isfinite(offset):
            raise ValueError("Standing return bound requires a nonpositive weight and finite offset")
        positive_max += max(0., -float(standing.weight)*offset)
    task_max = reward_scale * positive_max
    return (task_weight * task_max + style_weight * amp_coefficient) / (1.0 - gamma)


def reference_bootstrap(
    last_value: torch.Tensor,
    survival: torch.Tensor,
    gamma: float,
    horizon: int,
    value_ceiling: float,
) -> torch.Tensor:
    """Reference bootstrap, including gradients through cumulative survival."""
    return gamma**horizon * last_value.clamp(max=value_ceiling) * survival


def reference_gae_returns(
    values: torch.Tensor,
    last_value: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    gamma: float,
    lam: float,
    value_ceiling: float,
) -> torch.Tensor:
    """Time-major reference GAE; cap final targets, not its input values."""
    returns = torch.empty_like(values)
    advantage = 0.0
    next_value = last_value
    for t in reversed(range(values.shape[0])):
        not_done = 1.0 - dones[t]
        delta = rewards[t] + not_done * gamma * next_value - values[t]
        advantage = delta + not_done * gamma * lam * advantage
        returns[t] = advantage + values[t]
        next_value = values[t]
    return returns.clamp(max=value_ceiling)
