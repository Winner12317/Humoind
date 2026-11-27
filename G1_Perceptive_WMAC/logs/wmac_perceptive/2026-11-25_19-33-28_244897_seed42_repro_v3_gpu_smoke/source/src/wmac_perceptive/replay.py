"""Terrain-relative replay with compact depth storage and exact proprioception."""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict

import torch

from src.wmac.replay import ReplayBuffer

from .perception import PerceptionSpec, pack_actor_input


class PerceptiveReplay(ReplayBuffer):
    """Keep fp32 proprioception separate from fp16 depth until Actor sampling.

    Model windows read physical transitions plus current/next depth frames;
    they never need the large ring. Actor starts fetch a ring using the exact
    same time/environment indices as the original proprioceptive history and
    pack it transiently in fp32. Split, boundary and invalid-data semantics are
    inherited unchanged from WMAC replay.
    """

    _DEPTH_FIELDS = ("depth_ring", "depth_frame", "next_depth_frame")
    _REQUIRED = frozenset(("state", "context", "relative_height", "actor_history", *_DEPTH_FIELDS))

    def __init__(self, capacity_steps, num_envs, device="cpu", validation_env_fraction=0.,
                 *, spec: PerceptionSpec = PerceptionSpec()):
        self.spec = spec
        super().__init__(capacity_steps, num_envs, device, validation_env_fraction)

    def _validate_perceptive_shapes(self, batch):
        missing = self._REQUIRED.difference(batch)
        if missing:
            raise ValueError(f"Perceptive replay is missing fields: {sorted(missing)}")
        expected = {"actor_history": (self.num_envs, self.spec.proprio_dim),
                    "relative_height": (self.num_envs,),
                    "depth_ring": (self.num_envs, *self.spec.ring_shape),
                    "depth_frame": (self.num_envs, *self.spec.frame_shape),
                    "next_depth_frame": (self.num_envs, *self.spec.frame_shape)}
        for name, shape in expected.items():
            value = batch[name]
            if not isinstance(value, torch.Tensor) or tuple(value.shape) != shape or not value.is_floating_point():
                raise ValueError(f"Perceptive replay {name!r} must be a floating tensor of shape {shape}")

    @torch.no_grad()
    def add(self, batch):
        self._validate_perceptive_shapes(batch)
        stored = dict(batch)
        # Do not quantize the 384-dimensional proprioceptive input along with
        # the much larger visual ring. Only depth is deliberately lossy fp16.
        stored["actor_history"] = batch["actor_history"].float()
        stored["relative_height"] = batch["relative_height"].float()
        for name in self._DEPTH_FIELDS:
            stored[name] = batch[name].to(dtype=torch.float16)
        from .hiking_amp import before_replay_add, after_replay_add
        before_replay_add(self)
        accepted = super().add(stored)
        after_replay_add(self, stored, accepted)
        return accepted

    @torch.no_grad()
    def mark_boundary(self):
        super().mark_boundary()
        if getattr(self, "_hiking_amp_history", False):
            self._amp_pending_reset.fill_(True)

    def _index_row(self, row, valid):
        if "relative_height" not in self.data:
            raise ValueError("Perceptive recovery indexing requires terrain-relative height")
        recovery = self.data["relative_height"][row] < .55
        if (self.spec.depth_protocol!="g1_rpl_v1" and "context" in self.data
                and self.data["context"].shape[-1] >= 35):
            recovery |= self.data["context"][row, :, 34] > 0
        self._valid[row].copy_(valid)
        self._recovery[row].copy_(recovery)
        category = 2*self._validation_env.long()+recovery.long()
        category = torch.where(valid, category, torch.full_like(category, 4))
        order = torch.argsort(category)
        counts = torch.bincount(category, minlength=5)[:4]
        self._env_pool[row].copy_(order)
        self._group_counts[row].copy_(counts)
        self._group_offsets[row].copy_(counts.cumsum(0)-counts)
        boundary = self._boundaries[row]
        boundary.zero_()
        for name in ("boundary", "terminated", "timeout"):
            if name in self.data:
                boundary |= self.data[name][row].reshape(self.num_envs, -1).bool().any(-1)

    def sample_starts(self, batch_size, recovery_fraction=.5, fields: Sequence[str] | None = None,
                      split="train"):
        selected = self._fields(fields)
        pack = "actor_history" in selected
        requested = tuple(dict.fromkeys((*selected, "depth_ring"))) if pack else selected
        result = super().sample_starts(batch_size, recovery_fraction, requested, split)
        if pack:
            result["actor_history"] = pack_actor_input(result["actor_history"].float(),
                                                       result.pop("depth_ring").float(), self.spec)
        from .hiking_amp import add_sample_history
        return add_sample_history(self, result)

    def state_dict(self):
        result = super().state_dict()
        result.update(perceptive_format="fp32-proprio-fp16-depth-v1", perception_spec=asdict(self.spec))
        return result

    @torch.no_grad()
    def load_state_dict(self, state_dict):
        if (state_dict.get("perceptive_format") != "fp32-proprio-fp16-depth-v1" or
                {**{"depth_protocol":"hiking_v1"}, **state_dict.get("perception_spec",{})} != asdict(self.spec)):
            raise ValueError("Replay checkpoint perceptive schema or sensor history differs")
        data = state_dict.get("data", {})
        if state_dict.get("size", 0):
            self._validate_perceptive_shapes({name: value[0] for name, value in data.items()})
            if data["actor_history"].dtype != torch.float32 or data["relative_height"].dtype != torch.float32:
                raise ValueError("Perceptive replay checkpoint must retain fp32 proprioception and height")
            if any(data[name].dtype != torch.float16 for name in self._DEPTH_FIELDS):
                raise ValueError("Perceptive replay checkpoint depth tensors must be fp16")
        super().load_state_dict(state_dict)
