"""Fixed raw-history ONNX interface without tracing runtime shape dispatch.

Only the batch dimension is dynamic. Sensor history, crop, blur and the
policy head follow the saved architecture. The adapter owns a CPU evaluation
copy of the policy.
"""
from __future__ import annotations

import copy
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from src.wmac.hiking import HikingMoEActor
from .attention_policy import PerceptiveAttentionPolicy
from .models import PerceptiveActor


class RawPackedExportActor(nn.Module):
    """Export the raw ``[proprioception, depth/valid ring]`` entrypoint only.

    Pixel and latent Actors use the same real-image CNN path. Processed-stack
    and latent entrypoints intentionally have separate widths and are not part
    of this deployment interface. RPL packets are already centered upstream;
    their negative depths/resize overshoot remain intact here.
    """

    def __init__(self, actor: PerceptiveActor):
        super().__init__()
        if not isinstance(actor, PerceptiveActor) or not isinstance(
                actor.policy, (HikingMoEActor, PerceptiveAttentionPolicy)):
            raise TypeError("Raw ONNX adapter requires a perceptive Hiking MoE or attention Actor")
        policy = copy.deepcopy(actor).cpu().eval()
        policy.precision = "float32"
        spec = policy.spec
        self.input_width = spec.packed_dim
        self.proprio_width = spec.proprio_dim
        self.ring_width = spec.packed_dim - spec.proprio_dim
        self.ring_length, self.height, self.width = spec.ring_length, spec.height, spec.width
        self.frames, self.visual_channels = spec.frames, spec.visual_channels
        self.crop_height = spec.height - spec.crop_top - spec.crop_bottom
        self.crop_width = spec.width - spec.crop_left - spec.crop_right
        self.channels = spec.frames * spec.visual_channels
        self.include_valid = spec.include_valid
        self.apply_blur = spec.blur_kernel == 3
        self.clip_depth = not spec.include_valid and spec.depth_protocol != "g1_rpl_v1"
        self.conv, self.projection = policy.vision.conv, policy.vision.projection
        self.attention_policy = None
        if isinstance(policy.policy, PerceptiveAttentionPolicy):
            self.attention_policy = policy.policy
        else:
            self.router, self.experts = policy.policy.router, policy.policy.experts

        def indices(name, values):
            self.register_buffer(name, torch.tensor(tuple(values), dtype=torch.long))
        indices("time_indices", spec.selected_indices)
        indices("channel_indices", (0, 1) if spec.include_valid else (0,))
        indices("crop_rows", range(spec.crop_top, spec.height - spec.crop_bottom))
        indices("crop_columns", range(spec.crop_left, spec.width - spec.crop_right))
        indices("depth_indices", range(0, self.channels, 2) if spec.include_valid else range(self.channels))
        indices("valid_indices", range(1, self.channels, 2) if spec.include_valid else ())
        # ReflectionPad's legacy ONNX lowering builds reverse Slice nodes,
        # which trigger "steps!=1" constant-folding warnings. The exact pad
        # for radius one is two fixed gathers, including the corner pixels.
        indices("reflect_rows", (1, *range(self.crop_height), self.crop_height - 2))
        indices("reflect_columns", (1, *range(self.crop_width), self.crop_width - 2))
        dtype = self.conv[0].weight.dtype
        axis = torch.arange(-1, 2, dtype=dtype)
        gaussian = torch.exp(-axis.square() / (2 * spec.blur_sigma ** 2))
        gaussian = gaussian / gaussian.sum()
        kernel = (gaussian[:, None] * gaussian[None, :])[None, None]
        self.depth_channels = spec.frames
        self.register_buffer("blur_kernel", kernel.expand(self.depth_channels, 1, 3, 3).clone())
        self.eval()

    def forward(self, packed: torch.Tensor):
        # Fixed, contiguous raw-packet regions lower to unit-step Slice. They
        # need no large arange buffers or eager input Gather copies; unlike
        # ReflectionPad's reverse Slice, these support ordinary ONNX folding.
        proprio = packed.narrow(1, 0, self.proprio_width)
        ring = packed.narrow(1, self.proprio_width, self.ring_width).reshape(
            -1, self.ring_length, 2, self.height, self.width)
        selected = ring.index_select(1, self.time_indices).index_select(2, self.channel_indices)
        values = selected.reshape(-1, self.channels, self.height, self.width).to(self.conv[0].weight.dtype)
        values = values.index_select(-2, self.crop_rows).index_select(-1, self.crop_columns)
        if self.apply_blur:
            depth = values.index_select(1, self.depth_indices)
            reflected = depth.index_select(-2, self.reflect_rows).index_select(-1, self.reflect_columns)
            depth = F.conv2d(reflected, self.blur_kernel, groups=self.depth_channels)
            if self.include_valid:
                valid = values.index_select(1, self.valid_indices)
                values = torch.stack((depth, valid), dim=2).reshape(
                    -1, self.channels, self.crop_height, self.crop_width)
            else:
                values = depth
        if self.clip_depth:
            values = values.clamp(0, 1)
        embedding = self.projection(self.conv(values).flatten(1))
        features = torch.cat((proprio.to(embedding.dtype), embedding), dim=-1)
        if self.attention_policy is not None:
            return self.attention_policy(features)
        probabilities = torch.softmax(self.router(features), dim=-1)
        proposals = torch.stack([expert(features) for expert in self.experts], dim=1)
        return torch.einsum("be,beo->bo", probabilities, proposals)


def export_raw_packed_actor(actor: PerceptiveActor, path: str | Path, *,
                            input_name="proprio_and_depth_ring", output_name="actions",
                            opset_version=18) -> RawPackedExportActor:
    """Export a private CPU copy, leaving trained weights/modes/flags intact."""
    adapter = RawPackedExportActor(actor)
    # Keep the runner's existing raw FP32 input contract. The adapter casts
    # images/features to the saved network dtype internally, as the Actor does.
    sample = torch.zeros(1, adapter.input_width, dtype=torch.float32)
    torch.onnx.export(adapter, sample, str(path), input_names=[input_name], output_names=[output_name],
        dynamic_axes={input_name: {0: "batch"}, output_name: {0: "batch"}},
        opset_version=opset_version, dynamo=False)
    return adapter
