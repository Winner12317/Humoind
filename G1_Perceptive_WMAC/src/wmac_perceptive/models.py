"""Perceptive Actor and terrain/depth-conditioned learned dynamics.

The Actor and world model own separate image encoders. World-model pixel
targets therefore do not move whenever the Actor's encoder learns. Imagined
views are predictions at each step, not a frozen first image or true future
observations. Their errors must be checked before allowing Actor updates.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import nullcontext

import torch
from torch import nn
from torch.nn import functional as F

from src.wmac.hiking import HikingMoEActor
from src.wmac.networks import RunningNorm, WorldModel, invariant_features, mlp, run_mlp
from .perception import PerceptionSpec, select_depth_frames, unpack_actor_input
from .visual_metrics import preprocess_depth_channels


class DepthEncoder(nn.Module):
    """Hiking CNN with depth-only inputs or historical depth/valid pairs."""

    def __init__(self, channels: int, height: int, width: int, embedding: int,
                 spec: PerceptionSpec | None = None, *, include_valid: bool = True):
        super().__init__()
        if height < 2 or width < 2:
            raise ValueError("Depth images must support 2x2 pooling")
        self.channels, self.height, self.width = channels, height, width
        self.spec = spec
        self.include_valid = include_valid
        cropped_height = height if spec is None else height-spec.crop_top-spec.crop_bottom
        cropped_width = width if spec is None else width-spec.crop_left-spec.crop_right
        self.conv = nn.Sequential(nn.Conv2d(channels, 4, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2))
        self.projection = mlp(4 * (cropped_height // 2) * (cropped_width // 2), (256, 256), embedding, nn.ReLU)

    def preprocess(self, values):
        """Hiking crop/blur/clip, preserving the legacy depth/valid protocol."""
        if self.spec is None:
            return values
        result = preprocess_depth_channels(values, self.spec, channels=self.channels,
                                           include_valid=self.include_valid)
        return result if self.include_valid or self.spec.depth_protocol=="g1_rpl_v1" else result.clamp(0, 1)

    def forward(self, image):
        if tuple(image.shape[-3:]) != (self.channels, self.height, self.width):
            raise ValueError("Depth encoder image dimensions differ from its saved architecture")
        prefix = image.shape[:-3]
        # Replay images may be FP16 on CPU. Cast only selected samples, keeping
        # autograd through predicted future pixels and the encoder itself.
        values = image.reshape(-1, self.channels, self.height, self.width).to(self.conv[0].weight.dtype)
        values = self.projection(self.conv(self.preprocess(values)).flatten(1))
        return values.reshape(*prefix, values.shape[-1])


def _encode(encoder, image, precision):
    autocast = (torch.autocast("cuda", dtype=torch.bfloat16)
                if precision == "bf16" and image.device.type == "cuda" else nullcontext())
    with autocast:
        features = encoder(image)
    return features.float() if features.dtype == torch.bfloat16 else features


class PerceptiveActor(nn.Module):
    """Depth CNN plus the paper-gate Hiking MoE; no privileged terrain input."""

    def __init__(self, spec: PerceptionSpec = PerceptionSpec(), action_dim: int = 29,
                 gate_hidden: Sequence[int] = (256,), *, policy: nn.Module | None = None):
        super().__init__()
        self.spec, self.action_dim = spec, action_dim
        self.obs_dim = spec.packed_dim
        self.normalize_observations = False
        self.norm = RunningNorm(spec.proprio_dim)  # Dormant runner interface.
        self.vision = DepthEncoder(spec.visual_channels * spec.frames, spec.height, spec.width, 128, spec,
                                   include_valid=spec.include_valid)
        self.policy = (HikingMoEActor(spec.proprio_dim + 128, action_dim, gate_hidden=gate_hidden)
                       if policy is None else policy)
        self.precision = "float32"

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, value):
        self._precision = value
        if hasattr(self, "policy"):
            self.policy.precision = value

    @property
    def experts(self): return self.policy.experts

    @property
    def router(self): return self.policy.router

    @property
    def num_experts(self): return self.policy.num_experts

    @property
    def top_k(self): return self.policy.top_k

    @property
    def temperature(self): return self.policy.temperature

    @property
    def action_limit(self): return self.policy.action_limit

    def features(self, packed):
        proprio, ring = unpack_actor_input(packed, self.spec)
        frames = select_depth_frames(ring, self.spec)
        if not self.spec.include_valid:
            frames = frames[..., :1, :, :]
        image = frames.flatten(-4, -3)
        embedding = _encode(self.vision, image, self.precision)
        return torch.cat((proprio.to(embedding.dtype), embedding), dim=-1)

    def forward(self, packed):
        return self.policy(self.features(packed))

    def route(self, packed):
        return self.policy.route(self.features(packed))

    def forward_dense(self, packed):
        return self.forward(packed)

    @torch.no_grad()
    def copy_from_flat_hiking(self, state: Mapping[str, torch.Tensor]):
        """Keep an existing paper-gate Hiking policy's exact initial function.

        Only first-layer input widths change: copy proprioceptive columns and
        initialize the 128 new visual columns to zero. Independent vision
        weights are fresh. This is a new-run initialization, not a resume, and
        the visual path initially receives task gradient after its zero input
        columns start learning. An ordinary MLP or linear-gate source is refused.
        """
        expected = self.policy.state_dict()
        if set(state) != set(expected):
            raise ValueError("Perceptive warmstart requires the paper-gate Hiking Actor layout")
        first_layers = {"router.0.weight", *(f"experts.{i}.0.weight" for i in range(4))}
        prepared = {}
        for name, target in expected.items():
            value = state[name]
            if not isinstance(value, torch.Tensor) or not bool(torch.isfinite(value).all()):
                raise ValueError(f"Invalid warmstart tensor {name}")
            if name in first_layers:
                if value.shape != (target.shape[0], self.spec.proprio_dim):
                    raise ValueError(f"Warmstart first-layer width mismatch at {name}")
                copied = torch.zeros_like(target)
                copied[:, :self.spec.proprio_dim] = value.to(copied)
            elif name in {"norm.mean", "norm.var"}:
                if value.shape != (self.spec.proprio_dim,):
                    raise ValueError("Warmstart source observation width differs")
                copied = target.clone()  # Both source and target norm are unused.
            else:
                if value.shape != target.shape:
                    raise ValueError(f"Warmstart shape mismatch at {name}")
                copied = value.to(target).clone()
            prepared[name] = copied
        self.policy.load_state_dict(prepared)
        return self


class TerrainPerceptiveWorldModel(WorldModel):
    """Predict physical residuals, contact proxies, next depth and no-hit mask.

    Terrain oracle is privileged simulation geometry and belongs to the scene,
    not to the Actor or checkpoint parameters. Inputs query the same terrain as
    the physical simulator. Bilinear height-map derivatives are surrogates at
    mesh discontinuities, not a verified physical contact Jacobian.
    """

    def __init__(self, terrain_oracle, spec: PerceptionSpec = PerceptionSpec(),
                 hidden: Sequence[int] = (512, 512, 512), context_dim: int = 35,
                 dynamics_parameterization: str = "independent_residual_v1",
                 dt: float = .02, decimation: int = 4):
        super().__init__(context_dim=context_dim, hidden=hidden)
        if dynamics_parameterization not in ("independent_residual_v1","supermimic_gencoord_v1"):
            raise ValueError("Unknown physical dynamics parameterization")
        self.dynamics_parameterization=dynamics_parameterization
        self.dt,self.decimation=dt,decimation
        self.spec = spec
        object.__setattr__(self, "terrain_oracle", terrain_oracle)
        self.terrain_patch_dim = int(terrain_oracle.patch_dim)
        self.depth_encoder = DepthEncoder(2, spec.height, spec.width, 64, spec)
        self.physical_input_dim=68 if self.is_gencoord else 71
        self.prediction_dim=35 if self.is_gencoord else 70
        self.in_dim = self.physical_input_dim + 29 + context_dim + 64 + self.terrain_patch_dim
        self.in_norm = RunningNorm(self.in_dim)
        self.net = mlp(self.in_dim, hidden, self.prediction_dim, nn.GELU if self.is_gencoord else nn.SiLU, layer_norm=True)
        if self.is_gencoord:
            # Official GenCoord predicts physical velocity changes directly.
            # A learned output mean would change the meaning of zero delta.
            del self.out_norm
            del self.error_floor
        self.contact_net = mlp(self.in_dim, (256, 128), 2, nn.SiLU)
        self.depth_net = mlp(self.in_dim, (256, 256), 2 * spec.height * spec.width, nn.SiLU)
        with torch.no_grad():
            self.net[-1].weight.zero_(); self.net[-1].bias.zero_()
            self.contact_net[-1].weight.zero_(); self.contact_net[-1].bias.fill_(-4.)

    @property
    def is_gencoord(self):
        return self.dynamics_parameterization=="supermimic_gencoord_v1"

    def inputs(self, state, action, context, depth_frame=None):
        if depth_frame is None:
            raise ValueError("Perceptive dynamics requires the current depth/valid frame")
        if tuple(depth_frame.shape) != (*state.shape[:-1], *self.spec.frame_shape):
            raise ValueError("Depth frame and physical state batch dimensions differ")
        ground = self.terrain_oracle.height_at(state[..., :2])
        local_state = torch.cat((state[..., :2], (state[..., 2] - ground).unsqueeze(-1), state[..., 3:]), dim=-1)
        patch = self.terrain_oracle.local_patch(state)
        if patch.shape != (*state.shape[:-1], self.terrain_patch_dim):
            raise ValueError("Terrain oracle patch dimension differs from saved architecture")
        image = _encode(self.depth_encoder, depth_frame, self.precision)
        if self.is_gencoord:
            from src.wmac.gencoord import gencoord_features
            physical=gencoord_features(local_state)
        else:physical=invariant_features(local_state)
        return torch.cat((physical, action, context, image, patch), dim=-1)

    @torch.no_grad()
    def update_normalizers(self, state, action, context, next_state, depth_frame=None):
        self.in_norm.update(self.inputs(state, action, context, depth_frame))
        if not self.is_gencoord:self.out_norm.update(self.targets(state, next_state))

    def targets(self,state,next_state):
        if self.is_gencoord:
            from src.wmac.gencoord import velocity_increment_targets
            return velocity_increment_targets(state,next_state)
        return super().targets(state,next_state)

    def integrate(self,state,output):
        if self.is_gencoord:
            from src.wmac.gencoord import integrate_velocity_increment
            return integrate_velocity_increment(state,output,self.dt,self.decimation)
        return super().integrate(state,output)

    def loss(self,predicted_state,target_state):
        if self.is_gencoord:
            from src.wmac.gencoord import gencoord_loss
            return gencoord_loss(predicted_state,target_state)
        return super().loss(predicted_state,target_state)

    def step_with_perception(self, state, action, context, depth_frame):
        features = self.in_norm(self.inputs(state, action, context, depth_frame))
        increment = run_mlp(self.net, features, self.precision)
        contact = F.softplus(run_mlp(self.contact_net, features, self.precision))
        view = run_mlp(self.depth_net, features, self.precision).sigmoid()
        frame = view.reshape(*state.shape[:-1], *self.spec.frame_shape)
        return self.integrate(state, increment), contact, frame

    def step(self, state, action, context, depth_frame=None):
        if depth_frame is None:
            raise ValueError("Perceptive dynamics cannot step without depth conditioning")
        return self.step_with_perception(state, action, context, depth_frame)[0]

    def step_with_contact(self, state, action, context, depth_frame=None):
        if depth_frame is None:
            raise ValueError("Use step_with_perception with the current depth/valid frame")
        predicted, contact, _ = self.step_with_perception(state, action, context, depth_frame)
        return predicted, contact

    def contact_cost(self, state, action, context, depth_frame=None):
        if depth_frame is None:
            raise ValueError("Perceptive contact prediction requires depth conditioning")
        features = self.in_norm(self.inputs(state, action, context, depth_frame))
        return F.softplus(run_mlp(self.contact_net, features, self.precision))

    @staticmethod
    def pixel_loss_terms(predicted_frame, target_frame):
        if predicted_frame.shape != target_frame.shape or predicted_frame.shape[-3] != 2:
            raise ValueError("Pixel targets must match predicted (...,2,H,W) frames")
        target_frame = target_frame.to(predicted_frame)
        depth, valid = predicted_frame[..., 0, :, :], predicted_frame[..., 1, :, :]
        target_depth, target_valid = target_frame[..., 0, :, :], target_frame[..., 1, :, :].clamp(0, 1)
        # Target validity, not predicted validity, weights depth regression:
        # predicting all no-hit cannot hide errors at valid physical pixels.
        depth_loss = (F.smooth_l1_loss(depth, target_depth, reduction="none") * target_valid).sum((-2, -1))
        depth_loss = depth_loss / target_valid.sum((-2, -1)).clamp_min(1)
        valid_loss = F.binary_cross_entropy(valid.clamp(1e-6, 1-1e-6), target_valid, reduction="none").mean((-2, -1))
        return {"depth": depth_loss, "valid": valid_loss}

    @classmethod
    def pixel_loss(cls, predicted_frame, target_frame):
        terms = cls.pixel_loss_terms(predicted_frame, target_frame)
        return terms["depth"] + terms["valid"]
