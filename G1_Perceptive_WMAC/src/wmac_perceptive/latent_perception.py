"""Processed depth stacks and compact Actor inputs for InstinctMJ latents.

One latent represents the whole delayed eight-frame stack. Real observations
retain the reset-safe raw DepthHistory; imagined trajectories advance z
directly and never construct a ring of independently encoded frame latents.
"""

from __future__ import annotations

import torch

from .perception import PerceptionSpec, select_depth_frames
from .visual_metrics import preprocess_depth_channels


def processed_stack_shape(spec: PerceptionSpec = PerceptionSpec()):
    return (spec.visual_channels * spec.frames, spec.height - spec.crop_top - spec.crop_bottom,
            spec.width - spec.crop_left - spec.crop_right)


def processed_depth_frame(raw_frame: torch.Tensor,
                          spec: PerceptionSpec = PerceptionSpec()) -> torch.Tensor:
    """Prepare one or more raw sensor packets; validity is never an input in depth-only mode.

    New Hiking packets contain image-plane depth divided by max_depth before
    clipping. Crop and Gaussian blur precede clipping, matching the reference
    observation modifier order. Legacy packets keep their original behavior.
    """
    if raw_frame.ndim < 3 or tuple(raw_frame.shape[-3:]) != spec.frame_shape:
        raise ValueError("Raw depth frame differs from the sensor protocol")
    values = raw_frame if spec.include_valid else raw_frame[..., :1, :, :]
    if values.device.type == "cpu" and values.dtype in (torch.float16, torch.bfloat16):
        values = values.float()
    values = preprocess_depth_channels(values, spec, channels=spec.visual_channels,
                                       include_valid=spec.include_valid)
    return values if spec.include_valid or spec.depth_protocol=="g1_rpl_v1" else values.clamp(0, 1)


def processed_depth_history(raw_history: torch.Tensor,
                            spec: PerceptionSpec = PerceptionSpec()) -> torch.Tensor:
    """Keep the full ring's time axis while preparing only encoder channels."""
    if raw_history.ndim < 4 or tuple(raw_history.shape[-4:]) != spec.ring_shape:
        raise ValueError("Raw depth history differs from the sensor protocol")
    return processed_depth_frame(raw_history, spec)


def processed_depth_stack(raw_ring: torch.Tensor,
                          spec: PerceptionSpec = PerceptionSpec()) -> torch.Tensor:
    """Select delayed frames, then apply the existing Actor crop and blur.

    The output is ``(..., frames*visual_channels, crop_height, crop_width)``.
    Historical checkpoints retain interleaved validity; new Hiking inputs are
    depth-only like the Instinct reference.
    """
    selected = select_depth_frames(raw_ring, spec)
    if not spec.include_valid:
        selected = selected[..., :1, :, :]
    values = selected.flatten(-4, -3)
    if values.device.type == "cpu" and values.dtype in (torch.float16, torch.bfloat16):
        values = values.float()
    values = preprocess_depth_channels(values, spec, channels=spec.frames*spec.visual_channels,
                                       include_valid=spec.include_valid)
    return values if spec.include_valid or spec.depth_protocol=="g1_rpl_v1" else values.clamp(0, 1)


def pack_processed_actor_input(proprio_history: torch.Tensor, stack: torch.Tensor,
                               spec: PerceptionSpec = PerceptionSpec()) -> torch.Tensor:
    """Transient packed replay input; the encoder consumes current weights."""
    if (proprio_history.ndim < 1 or proprio_history.shape[-1] != spec.proprio_dim
            or stack.ndim < 3 or tuple(stack.shape[-3:]) != processed_stack_shape(spec)
            or proprio_history.shape[:-1] != stack.shape[:-3]):
        raise ValueError("Proprioception and processed stack dimensions differ")
    return torch.cat((proprio_history, stack.flatten(-3).to(proprio_history.dtype)), -1)


def unpack_processed_actor_input(packed: torch.Tensor, spec: PerceptionSpec = PerceptionSpec()):
    shape = processed_stack_shape(spec)
    expected = spec.proprio_dim + shape[0] * shape[1] * shape[2]
    if packed.ndim < 1 or packed.shape[-1] != expected:
        raise ValueError(f"Processed policy expects packed width {expected}")
    return packed[..., :spec.proprio_dim], packed[..., spec.proprio_dim:].reshape(*packed.shape[:-1], *shape)


def pack_latent_actor_input(proprio_history: torch.Tensor, latent: torch.Tensor,
                            spec: PerceptionSpec = PerceptionSpec()) -> torch.Tensor:
    """Compact replay/imagination input ``[proprio_history, stack_latent]``."""
    if (proprio_history.ndim < 1 or proprio_history.shape[-1] != spec.proprio_dim
            or latent.ndim < 1 or latent.shape[-1] < 1
            or proprio_history.shape[:-1] != latent.shape[:-1]):
        raise ValueError("Proprioception and stack latent dimensions differ")
    return torch.cat((proprio_history, latent.to(proprio_history.dtype)), -1)


def unpack_latent_actor_input(packed: torch.Tensor, spec: PerceptionSpec = PerceptionSpec(),
                              latent_dim: int = 128):
    expected = spec.proprio_dim + latent_dim
    if packed.ndim < 1 or packed.shape[-1] != expected:
        raise ValueError(f"Latent policy expects packed width {expected}")
    return packed[..., :spec.proprio_dim], packed[..., spec.proprio_dim:]
