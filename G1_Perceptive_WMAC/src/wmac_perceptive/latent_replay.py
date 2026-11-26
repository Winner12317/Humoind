"""Instinct-style image-stack replay for an online visual encoder.

Latents are recomputed with the current encoder, so replay never contains stale
latent coordinates. Cropped/blurred stacks replace the full 37-frame raw ring.
"""
from __future__ import annotations

from dataclasses import asdict

import torch

from src.wmac.replay import ReplayBuffer
from .replay import PerceptiveReplay
from .latent_perception import pack_processed_actor_input, processed_stack_shape
from .perception import PerceptionSpec


class LatentPerceptiveReplay(PerceptiveReplay):
    _DEPTH_FIELDS = ("depth_stack", "next_depth_stack")
    _REQUIRED = frozenset(("state", "context", "relative_height", "actor_history",
                          *_DEPTH_FIELDS))

    def __init__(self, capacity_steps, num_envs, device="cpu", validation_env_fraction=0.,
                 *, spec=PerceptionSpec()):
        super().__init__(capacity_steps, num_envs, device, validation_env_fraction, spec=spec)

    def _validate_perceptive_shapes(self, batch):
        missing = self._REQUIRED.difference(batch)
        if missing:
            raise ValueError(f"Latent replay is missing fields: {sorted(missing)}")
        expected = {"actor_history": (self.num_envs, self.spec.proprio_dim),
                    "relative_height": (self.num_envs,),
                    **{name: (self.num_envs, *processed_stack_shape(self.spec)) for name in self._DEPTH_FIELDS}}
        for name, shape in expected.items():
            value = batch[name]
            if (not isinstance(value, torch.Tensor) or tuple(value.shape) != shape
                    or not value.is_floating_point()):
                raise ValueError(f"Latent replay {name!r} must be a floating tensor of shape {shape}")

    def sample_starts(self, batch_size, recovery_fraction=0., fields=None, split="train"):
        selected = self._fields(fields)
        pack = "actor_history" in selected
        requested = tuple(dict.fromkeys((*selected, "depth_stack"))) if pack else selected
        result = ReplayBuffer.sample_starts(self, batch_size, recovery_fraction, requested, split)
        if pack:
            stack = result["depth_stack"].float()
            result["actor_history"] = pack_processed_actor_input(result["actor_history"].float(), stack, self.spec)
            if "depth_stack" not in selected:
                result.pop("depth_stack")
        from .hiking_amp import add_sample_history
        return add_sample_history(self, result)

    def state_dict(self):
        result = ReplayBuffer.state_dict(self)
        result.update(perceptive_format="instinct-depth-stack-replay-v1", perception_spec=asdict(self.spec))
        return result

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        if (state_dict.get("perceptive_format") != "instinct-depth-stack-replay-v1"
                or {**{"depth_protocol":"hiking_v1"}, **state_dict.get("perception_spec",{})} != asdict(self.spec)):
            raise ValueError("Latent replay depth-stack schema differs")
        if state_dict.get("size", 0):
            self._validate_perceptive_shapes({name: value[0] for name, value in state_dict["data"].items()})
        ReplayBuffer.load_state_dict(self, state_dict)
