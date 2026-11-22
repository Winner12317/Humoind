"""InstantAMP's robot-score centering, adapted to the existing LSGAN.

Reference: RoboticsResearchHubVN/InstantAMP, commit
9fc8031055a18d58e84d9de90639660e7aa6b491, instantamp/amp.py:106-119,
138-148 (MIT). Only the positive fixed-scale reward branch is adapted.
The discriminator loss, normalization and world model stay unchanged.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

import torch
from torch import nn


def clipped_lsgan_reward(score: torch.Tensor, coefficient: float) -> torch.Tensor:
    """The historical forward AND backward, retained for legacy checkpoints."""
    return coefficient * (1 - .25 * (score - 1).square()).clamp_min(0)


class CenteredAmpReward(nn.Module):
    """Bounded, monotone reward with a detached EMA of real policy scores.

    Scale 1 is the half-span of this repository's LSGAN targets (-1, +1),
    rather than InstantAMP's WGAN-specific scale 5. Expert EMA is diagnostic
    only: the reward is centered at the robot mean, not at the midpoint.
    No parameters, optimizer, discriminator forward, or custom backward.
    """
    def __init__(self, coefficient: float = .1, scale: float = 1., decay: float = .99):
        super().__init__()
        if not math.isfinite(coefficient) or coefficient < 0:
            raise ValueError("AMP coefficient must be finite and nonnegative")
        if not math.isfinite(scale) or scale <= 0:
            raise ValueError("AMP score scale must be finite and positive")
        if not math.isfinite(decay) or not 0 <= decay < 1:
            raise ValueError("AMP score EMA decay must be in [0, 1)")
        self.coefficient, self.scale, self.decay = coefficient, scale, decay
        self.register_buffer("robot_mean", torch.zeros(()))
        self.register_buffer("expert_mean", torch.zeros(()))
        self.register_buffer("initialized", torch.zeros((), dtype=torch.bool))
        self.register_buffer("updates", torch.zeros((), dtype=torch.int64))

    @torch.no_grad()
    def update(self, policy_score: torch.Tensor, expert_score: torch.Tensor) -> None:
        """Reuse scores from a successful existing D step; no host synchronization.

        As in the source, use pre-optimizer scores and initialize on the first
        usable batch. Unlike its zero-sentinel, an explicit flag also handles
        genuinely zero means. A nonfinite/empty batch leaves all state intact.
        """
        if policy_score.numel() == 0 or expert_score.numel() == 0:
            return
        robot = policy_score.detach().float().mean().to(self.robot_mean)
        expert = expert_score.detach().float().mean().to(self.expert_mean)
        live = torch.isfinite(robot) & torch.isfinite(expert)
        for target, current in ((self.robot_mean, robot), (self.expert_mean, expert)):
            candidate = torch.where(self.initialized,
                                    self.decay * target + (1 - self.decay) * current,
                                    current)
            target.copy_(torch.where(live, candidate, target))
        self.initialized.logical_or_(live)
        self.updates.add_(live.to(self.updates.dtype))

    def forward(self, score: torch.Tensor) -> torch.Tensor:
        # Statistics are constants for BPTT; never center on imagined scores.
        z = (score - self.robot_mean.detach()) / self.scale
        return self.coefficient * .5 * (1 + torch.tanh(z))

    def score_slope(self, score: torch.Tensor) -> torch.Tensor:
        """Actual tanh derivative, including numerical tail saturation."""
        t = torch.tanh((score - self.robot_mean.detach()) / self.scale)
        return self.coefficient / (2 * self.scale) * (1 - t.square())

    @staticmethod
    def validate_state(state) -> None:
        """Reject malformed checkpoint statistics before any runtime mutation."""
        expected = {"robot_mean", "expert_mean", "initialized", "updates"}
        if not isinstance(state, Mapping) or set(state) != expected:
            raise ValueError("Centered AMP checkpoint requires complete amp_reward_state")
        if any(not isinstance(v, torch.Tensor) or v.shape != torch.Size([]) for v in state.values()):
            raise ValueError("Centered AMP statistics must be scalar tensors")
        # Runtime EMA buffers are fp32. Reject wider types before loading so a
        # finite fp64 checkpoint value cannot overflow when copied into fp32.
        if any(state[k].dtype != torch.float32 or not bool(torch.isfinite(state[k]))
               for k in ("robot_mean", "expert_mean")):
            raise ValueError("Centered AMP score means must be finite fp32 scalars")
        if state["initialized"].dtype != torch.bool or state["updates"].dtype != torch.int64:
            raise ValueError("Centered AMP initialized/updates have invalid dtypes")
        initialized, updates = bool(state["initialized"]), int(state["updates"])
        if updates < 0 or initialized != (updates > 0):
            raise ValueError("Centered AMP initialized/updates disagree")
        if not initialized and any(float(state[k]) != 0 for k in ("robot_mean", "expert_mean")):
            raise ValueError("Uninitialized centered AMP means must be zero")
