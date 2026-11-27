"""Explicit, reset-safe depth history for real and imagined observations.

Raw packets contain image-plane distance divided by 2.5 m and a diagnostic
validity channel. Legacy packets are clipped to [0, 1], with no-hit depth 1.
Hiking depth-only packets retain no-hit depth 400000 in FP32 until crop/blur
and final clipping; their validity channel never enters the visual encoder.
Rings are chronological: index -1 is the current frame; the default delayed
policy reads -37..-2 at stride five, never the just-captured current image.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class PerceptionSpec:
    height: int = 36
    width: int = 64
    ring_length: int = 37
    frames: int = 8
    stride: int = 5
    delay: int = 1
    max_depth: float = 2.5
    proprio_dim: int = 384
    crop_top: int = 18
    crop_bottom: int = 0
    crop_left: int = 16
    crop_right: int = 16
    blur_kernel: int = 3
    blur_sigma: float = 1.0
    # Raw sensor packets retain validity for diagnostics. Only historical
    # policies feed that channel to their encoder and reconstruction target.
    include_valid: bool = True
    depth_protocol: str = "hiking_v1"

    def __post_init__(self):
        if self.depth_protocol not in ("hiking_v1", "g1_rpl_v1"):
            raise ValueError("Unsupported depth processing protocol")
        if not isinstance(self.include_valid, bool):
            raise ValueError("include_valid must explicitly select depth-only or depth/valid observations")
        for name in ("height", "width", "ring_length", "frames", "stride", "proprio_dim"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.delay, bool) or not isinstance(self.delay, int) or self.delay < 0:
            raise ValueError("delay must be a nonnegative integer")
        if self.ring_length < self.delay + self.stride * (self.frames - 1) + 1:
            raise ValueError("Depth ring is too short for the requested history/stride/delay")
        if not 0 < self.max_depth < float("inf"):
            raise ValueError("max_depth must be finite and positive")
        for name in ("crop_top", "crop_bottom", "crop_left", "crop_right"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.height-self.crop_top-self.crop_bottom < 2 or self.width-self.crop_left-self.crop_right < 2:
            raise ValueError("Cropped depth images must support 2x2 pooling; specify zero crop for tiny test images")
        if self.blur_kernel not in (1, 3) or not 0 < self.blur_sigma < float("inf"):
            raise ValueError("Supported blur kernels are 1/3 with finite positive sigma")

    @property
    def channels(self):
        return 2

    @property
    def visual_channels(self):
        """Channels per processed frame actually consumed by Actor/WM."""
        return 2 if self.include_valid else 1

    @property
    def frame_shape(self):
        return (2, self.height, self.width)

    @property
    def ring_shape(self):
        return (self.ring_length, *self.frame_shape)

    @property
    def packed_dim(self):
        return self.proprio_dim + self.ring_length * 2 * self.height * self.width

    @property
    def selected_indices(self):
        last = self.ring_length - 1 - self.delay
        return tuple(last - self.stride * index for index in reversed(range(self.frames)))


def _check_ring(ring: torch.Tensor, spec: PerceptionSpec):
    if ring.ndim < 4 or tuple(ring.shape[-4:]) != spec.ring_shape:
        raise ValueError(f"Expected depth ring (..., {spec.ring_shape}), got {tuple(ring.shape)}")


def select_depth_frames(ring: torch.Tensor, spec: PerceptionSpec = PerceptionSpec()):
    """Chronological strided/delayed frames, shape (..., frames, 2, H, W)."""
    _check_ring(ring, spec)
    indices = torch.tensor(spec.selected_indices, device=ring.device, dtype=torch.long)
    return ring.index_select(-4, indices)


def advance_depth_ring(ring: torch.Tensor, frame: torch.Tensor,
                       spec: PerceptionSpec = PerceptionSpec()):
    """Functional differentiable append, used for every imagined control step."""
    _check_ring(ring, spec)
    if tuple(frame.shape) != (*ring.shape[:-4], *spec.frame_shape):
        raise ValueError("New depth frame and ring batch dimensions differ")
    return torch.cat((ring[..., 1:, :, :, :], frame.unsqueeze(-4)), dim=-4)


def pack_actor_input(proprio_history: torch.Tensor, depth_ring: torch.Tensor,
                     spec: PerceptionSpec = PerceptionSpec()):
    """Pack only sampled replay rows; the replay should store both separately."""
    _check_ring(depth_ring, spec)
    if proprio_history.shape[-1] != spec.proprio_dim or proprio_history.shape[:-1] != depth_ring.shape[:-4]:
        raise ValueError("Proprioception and depth ring dimensions differ")
    return torch.cat((proprio_history, depth_ring.flatten(-4).to(proprio_history.dtype)), dim=-1)


def unpack_actor_input(packed: torch.Tensor, spec: PerceptionSpec = PerceptionSpec()):
    if packed.ndim < 1 or packed.shape[-1] != spec.packed_dim:
        raise ValueError(f"Perceptive policy expects packed width {spec.packed_dim}")
    proprio = packed[..., :spec.proprio_dim]
    ring = packed[..., spec.proprio_dim:].reshape(*packed.shape[:-1], *spec.ring_shape)
    return proprio, ring


class DepthHistory:
    """Real-sensor ring; resetting an environment fills its whole ring anew.

    ``append`` receives post-step/post-reset frames. Capture pre-reset next
    frames separately for model targets, before calling append with reset_mask.
    Neither reset nor append mutates a previously returned snapshot.
    """

    def __init__(self, spec: PerceptionSpec = PerceptionSpec(), num_envs: int = 1,
                 device="cpu", dtype=torch.float32):
        self.spec = spec
        if num_envs < 1:
            raise ValueError("num_envs must be positive")
        self.buffer = torch.zeros((num_envs, *spec.ring_shape), device=device, dtype=dtype)
        self.initialized = torch.zeros(num_envs, device=device, dtype=torch.bool)
        # Real collection can reuse these fixed indices. They are local
        # scratch tensors, outside module/checkpoint state and imagined graphs.
        self._sample_indices = self._successor_sample_indices = None
        self._fully_initialized = False

    @property
    def ring(self):
        return self.buffer

    def _indices(self, *, successor=False):
        if successor and self.spec.delay != 1:
            raise ValueError("Direct successor sampling requires the one-step Hiking delay")
        name = "_successor_sample_indices" if successor else "_sample_indices"
        indices = getattr(self, name)
        if indices is None or indices.device != self.buffer.device:
            indices = torch.tensor(tuple(index+int(successor) for index in self.spec.selected_indices),
                                   device=self.buffer.device, dtype=torch.long)
            setattr(self, name, indices)
        return indices

    def sampled_frames(self):
        """The real policy's selected frames, without rebuilding fixed indices."""
        _check_ring(self.buffer, self.spec)
        return self.buffer.index_select(-4, self._indices())

    def sampled_successor_frames(self, frame):
        """Same frames as selecting after a functional append, for delay=1.

        The delayed successor selects old indices1..L-1; its just-captured
        frame is not visible yet. Validate that frame's shape as the original
        append does, while avoiding a temporary copy of the complete ring.
        """
        _check_ring(self.buffer, self.spec)
        if tuple(frame.shape) != (*self.buffer.shape[:-4], *self.spec.frame_shape):
            raise ValueError("New depth frame and ring batch dimensions differ")
        indices = self._indices(successor=True)
        if frame.dtype != self.buffer.dtype or frame.device != self.buffer.device:
            # Preserve cat's dtype promotion/device checks for nonproduction
            # callers. The real processed collector uses matching FP32 tensors.
            return advance_depth_ring(self.buffer, frame, self.spec).index_select(-4, self._indices())
        return self.buffer.index_select(-4, indices)

    def _frame(self, frame):
        expected = (len(self.buffer), *self.spec.frame_shape)
        if tuple(frame.shape) != expected:
            raise ValueError(f"Expected frame shape {expected}")
        if not bool(torch.isfinite(frame).all()):
            raise ValueError("Depth history refuses non-finite observations")
        return frame.to(device=self.buffer.device, dtype=self.buffer.dtype)

    @torch.no_grad()
    def reset(self, frame: torch.Tensor, mask: torch.Tensor | None = None):
        frame = self._frame(frame)
        full_reset = mask is None
        if mask is None:
            mask = torch.ones(len(self.buffer), device=self.buffer.device, dtype=torch.bool)
        mask = mask.to(device=self.buffer.device, dtype=torch.bool)
        if tuple(mask.shape) != (len(self.buffer),):
            raise ValueError("Reset mask must have one entry per environment")
        expanded = frame[:, None].expand_as(self.buffer)
        self.buffer[mask] = expanded[mask]
        self.initialized[mask] = True
        # A partial reset conservatively keeps the existing flag. The next
        # append handles any uninitialized rows through its original mask.
        self._fully_initialized = self._fully_initialized or full_reset
        return self.buffer

    @torch.no_grad()
    def append(self, frame: torch.Tensor, reset_mask: torch.Tensor | None = None):
        frame = self._frame(frame)
        if reset_mask is None:
            reset_mask = torch.zeros_like(self.initialized)
        reset_mask = reset_mask.to(device=self.buffer.device, dtype=torch.bool)
        if tuple(reset_mask.shape) != (len(self.buffer),):
            raise ValueError("Reset mask must have one entry per environment")
        reset_mask = reset_mask | ~self.initialized
        updated = advance_depth_ring(self.buffer, frame, self.spec)
        repeated = frame[:, None].expand_as(updated)
        self.buffer = torch.where(reset_mask[:, None, None, None, None], repeated, updated)
        self.initialized.fill_(True)
        self._fully_initialized = True
        return self.buffer

    @torch.no_grad()
    def append_with_reset_ids(self, frame: torch.Tensor, reset_ids: torch.Tensor):
        """Append once and fill only reset rows in the newly allocated ring.

        ``reset_ids`` are the real collector's already computed termination
        indices. The old ring remains untouched, preserving snapshots and
        transition labels. Initial/partial histories use the ordinary path
        until all rows have been initialized.
        """
        frame = self._frame(frame)
        if (not isinstance(reset_ids, torch.Tensor) or reset_ids.ndim != 1
                or reset_ids.dtype not in (torch.int32, torch.int64)):
            raise ValueError("Depth reset indices must be a one-dimensional integer tensor")
        reset_ids = reset_ids.to(device=self.buffer.device, dtype=torch.long)
        if not self._fully_initialized:
            mask = torch.zeros_like(self.initialized)
            mask[reset_ids] = True
            return self.append(frame, reset_mask=mask)
        updated = advance_depth_ring(self.buffer, frame, self.spec)
        if reset_ids.numel():
            values = frame[reset_ids, None].expand(-1, self.spec.ring_length, -1, -1, -1)
            updated[reset_ids] = values
        self.buffer = updated
        self.initialized.fill_(True)
        return self.buffer

    def snapshot(self):
        return self.buffer.detach().clone()
