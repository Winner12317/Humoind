"""InstinctMJ visual-latent training adapted to local GenCoord dynamics.

Reference: project-instinct/InstinctMJ ``supermimic`` branch, commit
03563ee02665f327425106372264070315673748. The Actor's existing stack CNN
defines z; an online reconstruction loss trains that encoder. The Actor sees
z through a stop-gradient. World-model rollout predicts residual z without
rendering or decoding pixels. New depth-only inputs match the reference;
historical checkpoints retain explicit depth-valid channels. Local physical
adaptations are official GenCoord and the existing contact-cost proxies.
"""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F

from src.wmac.gencoord import gencoord_features
from src.wmac.networks import RunningNorm, mlp, run_mlp
from .config import TRANSFORMER_ACTOR_TYPE
from .hiking_context import normalize_hiking_sim_context
from .latent_perception import (
    processed_depth_stack, processed_stack_shape, unpack_latent_actor_input, unpack_processed_actor_input,
)
from .models import PerceptiveActor, TerrainPerceptiveWorldModel
from .perception import PerceptionSpec, unpack_actor_input


class LatentPerceptiveActor(PerceptiveActor):
    """Joint depth-stack CNN plus a policy with a shared latent entry point."""

    def __init__(self, spec: PerceptionSpec = PerceptionSpec(), latent_dim: int = 128,
                 action_dim: int = 29,gate_hidden: Sequence[int] = (256,), *,
                 actor_type: str = "moe"):
        if isinstance(latent_dim, bool) or not isinstance(latent_dim, int) or latent_dim != 128:
            raise ValueError("Instinct latent Actor preserves the 128-dimensional stack encoder")
        policy = None
        if actor_type == TRANSFORMER_ACTOR_TYPE:
            from .attention_policy import PerceptiveAttentionPolicy
            if spec.proprio_dim != PerceptiveAttentionPolicy.PROPRIO_DIM:
                raise ValueError("Transformer Actor requires eight 96-dimensional Hiking observations")
            policy = PerceptiveAttentionPolicy(action_dim=action_dim)
        elif actor_type != "moe":
            raise ValueError(f"Unsupported latent actor_type: {actor_type!r}")
        super().__init__(spec, action_dim,gate_hidden=gate_hidden,policy=policy)
        self.latent_dim = latent_dim
        self.latent_obs_dim = spec.proprio_dim + latent_dim
        shape = processed_stack_shape(spec)
        self.processed_obs_dim = spec.proprio_dim + shape[0] * shape[1] * shape[2]
        if len({self.latent_obs_dim, self.processed_obs_dim, self.obs_dim}) != 3:
            raise ValueError("Latent, processed and raw packed widths must differ for unambiguous dispatch")

    def encode_processed(self, stack: torch.Tensor) -> torch.Tensor:
        """Encode a preprocessed stack, preserving encoder/input gradients."""
        shape = processed_stack_shape(self.spec)
        if stack.ndim < 3 or tuple(stack.shape[-3:]) != shape:
            raise ValueError("Processed depth stack differs from the saved encoder architecture")
        prefix = stack.shape[:-3]
        values = stack.reshape(-1, *shape).to(self.vision.conv[0].weight.dtype)
        autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                    if self.precision == "bf16" and stack.device.type == "cuda" else nullcontext())
        with autocast:
            encoded = self.vision.projection(self.vision.conv(values).flatten(1))
        if encoded.dtype == torch.bfloat16:
            encoded = encoded.float()
        return encoded.reshape(*prefix, self.latent_dim)

    def encode_history(self, raw_ring: torch.Tensor) -> torch.Tensor:
        values = raw_ring.to(self.vision.conv[0].weight.dtype)
        return self.encode_processed(processed_depth_stack(values, self.spec))

    def features_latent(self, proprio_history: torch.Tensor, latent: torch.Tensor):
        if (proprio_history.ndim < 1 or proprio_history.shape[-1] != self.spec.proprio_dim
                or latent.ndim < 1 or latent.shape[-1] != self.latent_dim
                or latent.shape[:-1] != proprio_history.shape[:-1]):
            raise ValueError("Proprioception and stack latent dimensions differ")
        # This upstream gradient boundary applies to both real and imagined z.
        # Reconstruction calls encode_processed directly and can still train E.
        return torch.cat((proprio_history.to(latent.dtype), latent.detach()), -1)

    def features(self, packed: torch.Tensor):
        if packed.ndim >= 1 and packed.shape[-1] == self.latent_obs_dim:
            return self.features_latent(*unpack_latent_actor_input(packed, self.spec, self.latent_dim))
        if packed.ndim >= 1 and packed.shape[-1] == self.processed_obs_dim:
            proprio, stack = unpack_processed_actor_input(packed, self.spec)
            return self.features_latent(proprio, self.encode_processed(stack))
        proprio, raw_ring = unpack_actor_input(packed, self.spec)
        return self.features_latent(proprio, self.encode_history(raw_ring))

    def forward_latent(self, proprio_history: torch.Tensor, latent: torch.Tensor):
        return self.policy(self.features_latent(proprio_history, latent))

    def route_latent(self, proprio_history: torch.Tensor, latent: torch.Tensor):
        return self.policy.route(self.features_latent(proprio_history, latent))


class LatentPerceptiveWorldModel(TerrainPerceptiveWorldModel):
    """Single-head GenCoord, contact proxy and residual visual-latent model."""

    def __init__(self, terrain_oracle, spec: PerceptionSpec = PerceptionSpec(),
                 hidden: Sequence[int] = (512, 512, 512), context_dim: int = 35,
                 dt: float = .02, decimation: int = 4, latent_dim: int = 128,
                 contact_dim: int = 2, *, continuous_normalization: bool = False,
                 action_transform: tuple[torch.Tensor, torch.Tensor] | None = None):
        if isinstance(latent_dim, bool) or not isinstance(latent_dim, int) or latent_dim < 1:
            raise ValueError("latent_dim must be a positive integer")
        super().__init__(terrain_oracle, spec, hidden, context_dim,
                         dynamics_parameterization="supermimic_gencoord_v1", dt=dt, decimation=decimation)
        del self.depth_encoder
        del self.depth_net
        del self.contact_net
        self.latent_dim, self.context_dim = latent_dim, context_dim
        self.z_dim, self.num_contact = latent_dim, contact_dim
        self.in_dim = 68 + 29 + context_dim + self.terrain_patch_dim
        # SuperMimic keeps collecting WM moments throughout training (until=None).
        # The explicit flag preserves the original cap for historical protocols;
        # its value comes from the runner's saved training protocol, not tensor
        # state, so loading old moments into a new protocol cannot restore a cap.
        self.reference_inputs = action_transform is not None
        if self.reference_inputs:
            for name, value in zip(("action_scale", "action_offset"), action_transform, strict=True):
                value = torch.as_tensor(value, dtype=torch.float32)
                if value.ndim == 2:
                    value = value[0]
                self.register_buffer(name, value.expand(29).detach().clone())
        normalization_dim = 68 + 29 if self.reference_inputs else self.in_dim
        self.in_norm = RunningNorm(
            normalization_dim, max_count=float("inf") if continuous_normalization else 2e8)
        self.net = mlp(self.in_dim + latent_dim, hidden, 35 + contact_dim + latent_dim, nn.GELU, layer_norm=True)
        self.register_buffer("z_mean", torch.zeros(latent_dim))
        self.register_buffer("z_std", torch.ones(latent_dim))
        self.register_buffer("dz_std", torch.ones(latent_dim))
        with torch.no_grad():
            # Official GenCoord uses nn.Linear's default initialization. The
            # supplemental contact and residual latent rows initialize separately.
            start = 35 if self.reference_inputs else 0
            self.net[-1].weight[start:].zero_()
            self.net[-1].bias[start:].zero_()
            self.net[-1].bias[35:35+contact_dim].fill_(-4.)

    def _physical_inputs(self, state, action, context):
        prefix = state.shape[:-1]
        if state.ndim < 1 or state.shape[-1] != 71:
            raise ValueError("Physical state must have final width 71")
        for name, value, width in (("action", action, 29), ("context", context, self.context_dim)):
            if tuple(value.shape) != (*prefix, width):
                raise ValueError(f"{name} and physical state batch dimensions differ")
            if value.device != state.device:
                raise ValueError(f"{name} and physical state devices differ")
        if self.reference_inputs:
            action = action * self.action_scale + self.action_offset
            # SuperMimic maps DR coefficients using their configured ranges,
            # independently from its running state/action normalization.
            context = normalize_hiking_sim_context(context)
        else:
            ground = self.terrain_oracle.height_at(state[..., :2])
            state = torch.cat((state[..., :2], (state[..., 2] - ground).unsqueeze(-1), state[..., 3:]), -1)
        patch = self.terrain_oracle.local_patch(state)
        if tuple(patch.shape) != (*prefix, self.terrain_patch_dim):
            raise ValueError("Terrain oracle patch dimension differs from saved architecture")
        return torch.cat((gencoord_features(state), action, context, patch), -1)

    def inputs(self, state, action, context, depth_frame=None):
        if depth_frame is not None:
            raise ValueError("Latent world model has no image encoder; pass explicit z to inputs_latent")
        return self._physical_inputs(state, action, context)

    def inputs_latent(self, state, action, context, latent):
        """Normalized full network input; z uses separate upstream EMA stats."""
        if tuple(latent.shape) != (*state.shape[:-1], self.latent_dim):
            raise ValueError("Stack latent and physical state batch dimensions differ")
        if latent.device != state.device:
            raise ValueError("Stack latent and physical state devices differ")
        raw = self._physical_inputs(state, action, context)
        if self.reference_inputs:
            dim = self.in_norm.mean.numel()
            physical = torch.cat((self.in_norm(raw[..., :dim]), raw[..., dim:]), -1)
        else:
            physical = self.in_norm(raw)
        return torch.cat((physical, (latent.to(state.dtype) - self.z_mean) / self.z_std), -1)

    @torch.no_grad()
    def update_normalizers_latent(self, state, action, context, next_state, latent):
        del next_state
        if tuple(latent.shape) != (*state.shape[:-1], self.latent_dim):
            raise ValueError("Stack latent and physical state batch dimensions differ")
        self.update_normalizers(state, action, context, None)

    @torch.no_grad()
    def update_normalizers(self, state, action, context, next_state, depth_frame=None):
        if depth_frame is not None:
            raise ValueError("Use update_normalizers_latent with an explicitly encoded stack latent")
        del next_state
        if self.reference_inputs:
            self.in_norm.update(torch.cat((gencoord_features(state), action * self.action_scale + self.action_offset), -1))
        else:
            self.in_norm.update(self._physical_inputs(state, action, context))

    @torch.no_grad()
    def update_z_stats(self, z, dz, w, rate: float = .01):
        from .instinct_latent import update_latent_statistics
        update_latent_statistics(self, z, dz, w, rate)

    def step_with_latent(self, state, action, context, latent):
        prediction = run_mlp(self.net, self.inputs_latent(state, action, context, latent), self.precision)
        next_state = self.integrate(state, prediction[..., :35])
        contact = F.softplus(prediction[..., 35:35+self.num_contact])
        next_latent = latent + prediction[..., 35+self.num_contact:] * self.dz_std
        return next_state, contact, next_latent

    def step_with_perception(self, state, action, context, depth_frame):
        raise ValueError("Use step_with_latent; the world model has no image encoder or pixel decoder")

    def step(self, state, action, context, depth_frame=None):
        raise ValueError("Use step_with_latent with the current stack latent")

    def step_with_contact(self, state, action, context, depth_frame=None):
        raise ValueError("Use step_with_latent with the current stack latent")

    def contact_cost(self, state, action, context, depth_frame=None):
        raise ValueError("Use step_with_latent with the current stack latent")
