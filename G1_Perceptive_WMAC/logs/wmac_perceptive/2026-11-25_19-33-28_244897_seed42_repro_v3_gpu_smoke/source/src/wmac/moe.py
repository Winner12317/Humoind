"""Stateless sparse Top-k Actor with the existing AMP action/normalizer contract.

Hard Top-k selection has nondifferentiable routing boundaries. Gradients pass
through selected mixing weights and experts, not through a change of selected
expert IDs. Top-1 is intentionally unsupported: renormalizing its sole selected
weight to one would remove the task gradient to the router.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import NamedTuple

import torch
from torch import nn

from .networks import Actor, RunningNorm, mlp, run_mlp


class RoutingResult(NamedTuple):
    logits: torch.Tensor
    probabilities: torch.Tensor
    indices: torch.Tensor
    weights: torch.Tensor


class SparseMoEActor(nn.Module):
    """Shared normalized observations, independently routed full Actor experts.

    Dynamic sparse dispatch runs eagerly behind a local compiler-disable fence.
    The router and other surrounding operations remain compilable with graph
    breaks permitted; ``torch.compile(fullgraph=True)`` is unsupported. Dense
    export mode computes all experts then applies the same Top-k mask. It is a
    mathematically equivalent deployment fallback, not a sparse speed claim.
    """

    def __init__(self, obs_dim: int = 384, action_dim: int = 29,
                 hidden: Sequence[int] = (512, 256, 128), action_limit: float | None = 5.0,
                 output_init_scale: float = 0.01, zero_output_bias: bool = True,
                 num_experts: int = 4, top_k: int = 2,
                 gate_hidden: Sequence[int] = (256,), temperature: float = 1.0):
        super().__init__()
        if isinstance(num_experts, bool) or not isinstance(num_experts, int) or num_experts < 2:
            raise ValueError("num_experts must be an integer at least 2")
        if isinstance(top_k, bool) or not isinstance(top_k, int) or not 2 <= top_k <= num_experts:
            raise ValueError("top_k must satisfy 2 <= top_k <= num_experts; renormalized Top-1 has no task gradient to its router")
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature must be finite and positive")
        if action_limit is not None and (not math.isfinite(action_limit) or action_limit <= 0):
            raise ValueError("action_limit must be finite and positive or None")
        if obs_dim < 1 or action_dim < 1:
            raise ValueError("Observation and action widths must be positive")
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.hidden, self.gate_hidden = tuple(hidden), tuple(gate_hidden)
        self.num_experts, self.top_k = num_experts, top_k
        self.temperature, self.action_limit = float(temperature), action_limit
        self.precision = "float32"
        self.export_dense = False
        self.norm = RunningNorm(obs_dim)
        self.experts = nn.ModuleList([mlp(obs_dim, self.hidden, action_dim) for _ in range(num_experts)])
        self.router = mlp(obs_dim, self.gate_hidden, num_experts)
        with torch.no_grad():
            for expert in self.experts:
                expert[-1].weight.mul_(output_init_scale)
                if zero_output_bias:
                    expert[-1].bias.zero_()
            # Small random logits avoid a deterministic all-zero Top-k tie at
            # initialization without imposing a hand-written recovery router.
            self.router[-1].weight.mul_(.01)
            self.router[-1].bias.mul_(.01)

    def _normalized_route(self, flat: torch.Tensor) -> RoutingResult:
        logits = run_mlp(self.router, flat, self.precision)
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        probabilities = torch.softmax(logits / self.temperature, dim=-1)
        selected, indices = torch.topk(probabilities, self.top_k, dim=-1, sorted=True)
        weights = selected / selected.sum(dim=-1, keepdim=True)
        return RoutingResult(logits, probabilities, indices, weights)

    def _flatten(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim < 1 or observation.shape[-1] != self.obs_dim:
            raise ValueError(f"Expected observations with final width {self.obs_dim}")
        return self.norm(observation).reshape(-1, self.obs_dim)

    def route(self, observation: torch.Tensor) -> RoutingResult:
        """Return a fresh route graph; no saved graph, counters or statistics.

        Logits/probabilities have final width ``num_experts`` and indices/weights
        width ``top_k``. All leading observation dimensions are preserved.
        Callers can independently compute router losses or detached statistics.
        """
        values = self._normalized_route(self._flatten(observation))
        prefix = observation.shape[:-1]
        return RoutingResult(*(value.reshape(*prefix, width) for value, width in zip(
            values, (self.num_experts, self.num_experts, self.top_k, self.top_k))))

    @torch.compiler.disable
    def _raw_sparse(self, flat: torch.Tensor, route: RoutingResult) -> torch.Tensor:
        """Evaluate only the tokens actually assigned to each selected expert."""
        output = flat.new_zeros((flat.shape[0], self.action_dim))
        for expert_id, expert in enumerate(self.experts):
            tokens, slots = torch.where(route.indices == expert_id)
            if not tokens.numel():
                continue
            proposal = run_mlp(expert, flat.index_select(0, tokens), self.precision)
            contribution = proposal * route.weights[tokens, slots, None]
            output = output.index_add(0, tokens, contribution)
        return output

    def _raw_dense(self, flat: torch.Tensor, route: RoutingResult) -> torch.Tensor:
        proposals = torch.stack([run_mlp(expert, flat, self.precision) for expert in self.experts], dim=1)
        mask = torch.zeros_like(route.probabilities).scatter(1, route.indices, route.weights)
        return (proposals * mask[..., None]).sum(1)

    def _actions(self, raw: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        if self.action_limit is not None:
            raw = self.action_limit * torch.tanh(raw / self.action_limit)
        return raw.reshape(*observation.shape[:-1], self.action_dim)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        flat = self._flatten(observation)
        route = self._normalized_route(flat)
        raw = self._raw_dense(flat, route) if self.export_dense else self._raw_sparse(flat, route)
        return self._actions(raw, observation)

    def forward_dense(self, observation: torch.Tensor) -> torch.Tensor:
        """Equivalent masked Top-k export path, with all experts computed."""
        flat = self._flatten(observation)
        return self._actions(self._raw_dense(flat, self._normalized_route(flat)), observation)

    @torch.no_grad()
    def load_mlp_state_dict(self, state: Mapping[str, torch.Tensor], noise_std: float = 0.):
        """Clone an ordinary Actor's ``norm.*``/``net.*`` state into every expert.

        Router parameters are retained. Zero noise preserves the original raw
        proposal and configured action transform up to floating-point reduction
        error. Positive noise independently perturbs expert output weights only;
        callers must record that intervention. Normalizer attributes and action
        bounds are configuration, not tensors in this legacy state dictionary.
        Validation precedes mutation; this helper does not accept MoE state.
        """
        if not math.isfinite(noise_std) or noise_std < 0:
            raise ValueError("noise_std must be finite and nonnegative")
        expected = {**{f"norm.{key}": value for key, value in self.norm.state_dict().items()},
                    **{f"net.{key}": value for key, value in self.experts[0].state_dict().items()}}
        if set(state) != set(expected):
            raise ValueError("Expected an ordinary Actor MLP state with matching norm.* and net.* tensors")
        for key, value in expected.items():
            supplied = state[key]
            if not isinstance(supplied, torch.Tensor) or supplied.shape != value.shape:
                raise ValueError(f"MLP Actor tensor {key!r} has an incompatible shape")
            if not bool(torch.isfinite(supplied).all()):
                raise ValueError(f"MLP Actor tensor {key!r} is non-finite")
        self.norm.load_state_dict({key[5:]: value for key, value in state.items() if key.startswith("norm.")})
        expert_state = {key[4:]: value for key, value in state.items() if key.startswith("net.")}
        for expert in self.experts:
            expert.load_state_dict(expert_state)
            if noise_std:
                expert[-1].weight.add_(noise_std * torch.randn_like(expert[-1].weight))
        return self

    @torch.no_grad()
    def load_baseline(self, checkpoint, noise_std: float = 0.):
        """Reuse the existing PPO Actor importer, then clone its tensor state."""
        with torch.random.fork_rng(devices=[]):
            original = Actor(self.obs_dim, self.action_dim, self.hidden, self.action_limit)
            dtype = next(self.experts[0].parameters()).dtype
            if dtype != torch.float32:
                original.to(dtype=dtype)
        original.load_baseline(checkpoint)
        return self.load_mlp_state_dict(original.state_dict(), noise_std=noise_std)
