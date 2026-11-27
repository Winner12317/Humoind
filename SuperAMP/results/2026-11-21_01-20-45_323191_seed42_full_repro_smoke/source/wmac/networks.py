"""Differentiable dynamics and independent actor/critic networks for WMAC.

The frame-local increment parameterization follows the SuperMimic reproduction
at https://github.com/zita-ch/supermimic_vibed (MIT, Copyright (c) 2026
ETH Zurich, Chong Zhang); see THIRD_PARTY_NOTICES.md for its license.

The optional GELU/LayerNorm blocks and default Linear output initialization
follow HybridRobotics/supermimic at abc124c1bd5976341034668af3d12c087f2f1f55.
They leave this framework's local-residual state and integration method intact.
"""

from __future__ import annotations

from contextlib import contextmanager
from collections.abc import Iterator, Sequence
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .state import (
    heading_quat,
    quat_apply,
    quat_from_rotvec,
    quat_inv,
    quat_mul,
    quat_normalize,
    quat_to_matrix,
)


def mlp(in_dim: int, hidden: Sequence[int], out_dim: int,
        activation: type[nn.Module] = nn.ELU, layer_norm: bool = False) -> nn.Sequential:
    layers: list[nn.Module] = []
    width = in_dim
    for next_width in hidden:
        layers.append(nn.Linear(width, next_width))
        if layer_norm:
            layers.append(nn.LayerNorm(next_width))
        layers.append(activation())
        width = next_width
    layers.append(nn.Linear(width, out_dim))
    return nn.Sequential(*layers)


def _activation_class(name: str) -> type[nn.Module]:
    """Resolve the explicit network dialect without changing legacy defaults."""
    activations = {"elu": nn.ELU, "silu": nn.SiLU, "gelu": nn.GELU}
    if name not in activations:
        raise ValueError(f"Unsupported network activation {name!r}; choose from {tuple(activations)}")
    return activations[name]


def run_mlp(module: nn.Module, values: torch.Tensor, precision: str = 'float32') -> torch.Tensor:
    """Only dense matmuls use bf16; dynamics, rotations and returns stay fp32."""
    if precision == 'bf16' and values.device.type == 'cuda':
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return module(values).float()
    return module(values)


class RunningNorm(nn.Module):
    """Explicitly updated moments; forward always retains input gradients.

    Forward never updates statistics, even in train mode. This keeps the model,
    policy and critic normalization fixed throughout each imagined trajectory.
    The additive epsilon also matches the baseline observation normalizer.
    """

    def __init__(self, dim: int, eps: float = 1e-2, max_count: float = 2e8):
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("var", torch.ones(dim))
        self.register_buffer("count", torch.zeros((), dtype=torch.float64))
        self.eps = eps
        self.max_count = max_count
        self.variance_epsilon = False
        self.clip = None

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        x = values.detach().reshape(-1, self.mean.numel()).float()
        if x.shape[0] == 0 or self.count >= self.max_count:
            return
        if not torch.isfinite(x).all():
            raise ValueError("RunningNorm cannot fit non-finite samples")
        n = x.shape[0]
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        total = self.count + n
        delta = batch_mean - self.mean
        fraction = (n / total).to(self.mean.dtype)
        merged_var = (
            self.var * self.count.to(self.var.dtype)
            + batch_var * n
            + delta.square() * self.count.to(self.var.dtype) * fraction
        ) / total.to(self.var.dtype)
        self.mean.add_(delta * fraction)
        self.var.copy_(merged_var.clamp_min(0))
        self.count.copy_(total)

    def std(self) -> torch.Tensor:
        if self.variance_epsilon:
            return (self.var.clamp_min(0) + self.eps).sqrt()
        return self.var.clamp_min(0).sqrt() + self.eps

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        result = (values - self.mean) / self.std()
        return result if self.clip is None else result.clamp(-self.clip, self.clip)


@contextmanager
def frozen_parameters(*modules: nn.Module) -> Iterator[None]:
    """Freeze parameter gradients without disabling gradients through inputs."""
    parameters: list[tuple[nn.Parameter, bool]] = []
    seen: set[int] = set()
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) not in seen:
                seen.add(id(parameter))
                parameters.append((parameter, parameter.requires_grad))
                parameter.requires_grad_(False)
    try:
        yield
    finally:
        for parameter, requires_grad in parameters:
            parameter.requires_grad_(requires_grad)


def _quat_to_rotvec(quat: torch.Tensor) -> torch.Tensor:
    q = quat_normalize(quat)
    q = torch.where(q[..., :1] < 0, -q, q)
    vector = q[..., 1:]
    norm_sq = vector.square().sum(-1, keepdim=True)
    norm = (norm_sq + 1e-12).sqrt()
    angle = 2 * torch.atan2(norm, q[..., :1])
    scale = torch.where(norm_sq < 1e-8, 2 + norm_sq / 3, angle / norm)
    return vector * scale


def invariant_features(state: torch.Tensor) -> torch.Tensor:
    """Height, heading-free orientation, body velocities and joint state (71D)."""
    q = quat_normalize(state[..., 3:7])
    tilt = quat_mul(quat_inv(heading_quat(q)), q)
    orientation = quat_to_matrix(tilt)[..., :, :2].flatten(-2)
    return torch.cat((
        state[..., 2:3], orientation,
        quat_apply(quat_inv(q), state[..., 7:10]),
        quat_apply(quat_inv(q), state[..., 10:13]),
        state[..., 13:42], state[..., 42:71],
    ), -1)


class WorldModel(nn.Module):
    """One control-period, local residual dynamics; no reset transitions.

    Raw state: position(3), quaternion wxyz(4), world velocities(3+3),
    joint positions(29), joint velocities(29). Output regression has 70
    dimensions because the orientation increment is a 3D rotation vector.
    """

    def __init__(self, context_dim: int = 35, hidden: Sequence[int] = (512, 512, 512),
                 activation: str = "silu", output_zero_init: bool = True):
        super().__init__()
        self.context_dim = context_dim
        self.precision = 'float32'
        self.in_dim = 71 + 29 + context_dim
        self.in_norm = RunningNorm(self.in_dim)
        self.out_norm = RunningNorm(70, eps=1e-4)
        self.net = mlp(self.in_dim, hidden, 70, _activation_class(activation), layer_norm=True)
        if output_zero_init:
            nn.init.zeros_(self.net[-1].weight)
            nn.init.zeros_(self.net[-1].bias)
        self.contact_net = mlp(self.in_dim, (256, 128), 2, nn.SiLU)
        nn.init.zeros_(self.contact_net[-1].weight)
        nn.init.constant_(self.contact_net[-1].bias, -4.0)
        self.register_buffer("error_floor", torch.cat((
            torch.full((3,), 3e-3), torch.full((3,), 5e-3),
            torch.full((3,), 2e-2), torch.full((3,), 5e-2),
            torch.full((29,), 3e-3), torch.full((29,), 5e-2),
        )))

    def inputs(self, state: torch.Tensor, action: torch.Tensor,
               context: torch.Tensor) -> torch.Tensor:
        return torch.cat((invariant_features(state), action, context), -1)

    @staticmethod
    def targets(state: torch.Tensor, next_state: torch.Tensor) -> torch.Tensor:
        """Frame-local difference; also used for physical state regression error."""
        q = quat_normalize(state[..., 3:7])
        inv_q = quat_inv(q)
        return torch.cat((
            quat_apply(quat_inv(heading_quat(q)), next_state[..., :3] - state[..., :3]),
            _quat_to_rotvec(quat_mul(inv_q, next_state[..., 3:7])),
            quat_apply(inv_q, next_state[..., 7:10] - state[..., 7:10]),
            quat_apply(inv_q, next_state[..., 10:13] - state[..., 10:13]),
            next_state[..., 13:42] - state[..., 13:42],
            next_state[..., 42:71] - state[..., 42:71],
        ), -1)

    @torch.no_grad()
    def update_normalizers(self, state: torch.Tensor, action: torch.Tensor,
                           context: torch.Tensor, next_state: torch.Tensor) -> None:
        self.in_norm.update(self.inputs(state, action, context))
        self.out_norm.update(self.targets(state, next_state))

    def step(self, state: torch.Tensor, action: torch.Tensor,
             context: torch.Tensor) -> torch.Tensor:
        output = run_mlp(self.net, self.in_norm(self.inputs(state, action, context)), self.precision)
        return self.integrate(state, output)

    def integrate(self, state: torch.Tensor, output: torch.Tensor) -> torch.Tensor:
        increment = output * self.out_norm.std() + self.out_norm.mean
        dp, drot, dv, dw, dq, dqd = increment.split((3, 3, 3, 3, 29, 29), -1)
        q = quat_normalize(state[..., 3:7])
        return torch.cat((
            state[..., :3] + quat_apply(heading_quat(q), dp),
            quat_normalize(quat_mul(q, quat_from_rotvec(drot))),
            state[..., 7:10] + quat_apply(q, dv),
            state[..., 10:13] + quat_apply(q, dw),
            state[..., 13:42] + dq,
            state[..., 42:71] + dqd,
        ), -1)

    def step_with_contact(self, state: torch.Tensor, action: torch.Tensor,
                          context: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.in_norm(self.inputs(state, action, context))
        output = run_mlp(self.net, features, self.precision)
        contact = F.softplus(run_mlp(self.contact_net, features, self.precision))
        return self.integrate(state, output), contact

    def forward(self, state: torch.Tensor, action: torch.Tensor,
                context: torch.Tensor) -> torch.Tensor:
        return self.step(state, action, context)

    def loss(self, predicted_state: torch.Tensor,
             target_state: torch.Tensor) -> torch.Tensor:
        scale = torch.maximum(self.out_norm.std(), self.error_floor)
        return (self.targets(target_state, predicted_state) / scale).square().mean(-1)

    def contact_cost(self, state: torch.Tensor, action: torch.Tensor,
                     context: torch.Tensor) -> torch.Tensor:
        """Predict non-negative raw foot-slip and self-collision costs."""
        features = self.in_norm(self.inputs(state, action, context))
        return F.softplus(run_mlp(self.contact_net, features, self.precision))


class Actor(nn.Module):
    """One deployed policy with the baseline's 384D history and 29D action."""

    def __init__(self, obs_dim: int = 384, action_dim: int = 29,
                 hidden: Sequence[int] = (512, 256, 128), action_limit: float | None = 5.0,
                 output_init_scale: float = 0.01, zero_output_bias: bool = True,
                 activation: str = "elu", layer_norm: bool = False):
        super().__init__()
        if action_limit is not None and action_limit <= 0:
            raise ValueError("action_limit must be positive or None")
        self.norm = RunningNorm(obs_dim)
        self.net = mlp(obs_dim, hidden, action_dim, _activation_class(activation), layer_norm=layer_norm)
        self.action_limit = action_limit
        self.precision = 'float32'
        with torch.no_grad():
            self.net[-1].weight.mul_(output_init_scale)
            if zero_output_bias:
                self.net[-1].bias.zero_()

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        raw_action = run_mlp(self.net, self.norm(observation), self.precision)
        if self.action_limit is None:
            return raw_action
        return self.action_limit * torch.tanh(raw_action / self.action_limit)

    @torch.no_grad()
    def load_baseline(self, checkpoint: dict[str, Any]) -> None:
        """Import baseline AMP ActorCritic actor weights and observation moments.

        Set action_limit=None when an exactly identical unbounded policy is
        required. Other modules and optimizers deliberately retain fresh state.
        """
        model = checkpoint.get("model_state_dict", checkpoint)
        actor_weights = {key[len("actor."):]: value for key, value in model.items()
                         if key.startswith("actor.")}
        if not actor_weights:
            raise ValueError("Checkpoint contains no baseline actor.* weights")
        self.net.load_state_dict(actor_weights)
        moments = checkpoint.get("obs_norm_state_dict")
        if moments is not None:
            self.norm.mean.copy_(moments["_mean"].reshape_as(self.norm.mean))
            self.norm.var.copy_(moments["_var"].reshape_as(self.norm.var))
            self.norm.count.copy_(moments["count"].reshape_as(self.norm.count))


class Critic(nn.Module):
    """Separate scalar value network; no shared parameters with the actor."""

    def __init__(self, in_dim: int, hidden: Sequence[int] = (256, 128),
                 activation: str = "elu", layer_norm: bool = False):
        super().__init__()
        self.norm = RunningNorm(in_dim)
        self.net = mlp(in_dim, hidden, 1, _activation_class(activation), layer_norm=layer_norm)
        self.precision = 'float32'

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return run_mlp(self.net, self.norm(features), self.precision).squeeze(-1)
