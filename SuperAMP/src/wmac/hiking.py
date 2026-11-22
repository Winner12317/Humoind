"""Dense Hiking Actor with the paper's [256] gate as the new-run default.

Reference: instinct_rl 574c8d50 / InstinctLab de6b92f0. The paper's gate
hidden width [256] differs from the executable G1 configuration's empty gate
hidden list. Both are represented explicitly for checkpoint compatibility;
new runs use the paper gate. WMAC supplies its own observation interface and
training objective.
"""
from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .moe import RoutingResult, SparseMoEActor
from .networks import RunningNorm, mlp, run_mlp


class HikingMoEActor(SparseMoEActor):
    """Four ELU experts, ELU gate MLP, full Softmax blend, linear action output.

    A dormant norm buffer preserves the runner/checkpoint interface, but is
    neither applied nor updated. All Linear layers retain PyTorch defaults.
    The dense forward has no Top-k, dispatch fence, or post-mixture transform.
    """

    def __init__(self, obs_dim: int = 384, action_dim: int = 29,
                 gate_hidden: Sequence[int] = (256,)):
        nn.Module.__init__(self)
        if obs_dim < 1 or action_dim < 1:
            raise ValueError("Observation and action widths must be positive")
        self.obs_dim, self.action_dim = obs_dim, action_dim
        self.hidden, self.gate_hidden = (256, 128, 64), tuple(gate_hidden)
        if self.gate_hidden not in ((256,), ()):
            raise ValueError("Hiking gate hidden units must be [256] or historical []")
        self.num_experts, self.top_k = 4, 4
        self.temperature, self.action_limit = 1.0, None
        self.precision, self.export_dense = "float32", False
        self.normalize_observations = False
        self.norm = RunningNorm(obs_dim)
        # Author construction order: gate first, then independent experts.
        self.router = mlp(obs_dim, self.gate_hidden, 4)
        self.experts = nn.ModuleList([mlp(obs_dim, self.hidden, action_dim) for _ in range(4)])

    def _flatten(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim < 1 or observation.shape[-1] != self.obs_dim:
            raise ValueError(f"Expected observations with final width {self.obs_dim}")
        return observation.reshape(-1, self.obs_dim)

    def _normalized_route(self, flat: torch.Tensor) -> RoutingResult:
        logits = run_mlp(self.router, flat, self.precision)
        if logits.dtype in (torch.float16, torch.bfloat16):
            logits = logits.float()
        probabilities = torch.softmax(logits, dim=-1)
        indices = torch.arange(4, device=flat.device).expand(flat.shape[0], 4)
        return RoutingResult(logits, probabilities, indices, probabilities)

    def _raw_dense(self, flat: torch.Tensor, route: RoutingResult) -> torch.Tensor:
        proposals = torch.stack([run_mlp(expert, flat, self.precision) for expert in self.experts], dim=1)
        return torch.einsum("be,beo->bo", route.probabilities, proposals)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        flat = self._flatten(observation)
        raw = self._raw_dense(flat, self._normalized_route(flat))
        return raw.reshape(*observation.shape[:-1], self.action_dim)

    def forward_dense(self, observation: torch.Tensor) -> torch.Tensor:
        return self.forward(observation)

    def load_mlp_state_dict(self, state, noise_std=0.):
        raise ValueError("Hiking uses different expert widths, no Actor normalization and no output tanh; "
                         "MLP-to-Hiking cloning cannot preserve the policy. Start a fresh Hiking run.")

    def load_baseline(self, checkpoint, noise_std=0.):
        raise ValueError("PPO/MLP-to-Hiking cloning is not supported; start a fresh Hiking run.")
