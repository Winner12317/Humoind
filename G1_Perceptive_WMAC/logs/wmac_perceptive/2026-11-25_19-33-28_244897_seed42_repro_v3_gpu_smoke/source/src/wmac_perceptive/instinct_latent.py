"""InstinctMJ SuperMimic's online visual-latent reconstruction and statistics.

Adapted from project-instinct/InstinctMJ, branch supermimic, commit
03563ee02665f327425106372264070315673748, networks.py and runner.py (MIT;
Copyright (c) 2026 ETH Zurich, Chong Zhang).
See THIRD_PARTY_NOTICES.md for the upstream licensing and attribution.

Production decoding preserves the upstream 18x32 architecture. Local images
also have validity channels, so the decoder's channel count is configurable.
The caller owns target stop-gradients and masked loss aggregation.
"""
from __future__ import annotations

import math
from numbers import Real

import torch
from torch import nn
from torch.nn import functional as F

INSTINCT_LATENT_REFERENCE_COMMIT = "03563ee02665f327425106372264070315673748"


class InstinctLatentDecoder(nn.Module):
    """Training-only decoder; imagined rollouts never need decoded pixels."""

    def __init__(self, latent_dim: int, channels: int, output_shape: tuple[int, int] = (18, 32),
                 *, bounded_output: bool = True):
        super().__init__()
        if any(isinstance(value, bool) or not isinstance(value, int) or value < 1
               for value in (latent_dim, channels)):
            raise ValueError("Decoder latent dimension and channels must be positive integers")
        if (not isinstance(output_shape, (tuple, list)) or len(output_shape) != 2
                or any(isinstance(value, bool) or not isinstance(value, int) or value < 1
                       for value in output_shape)):
            raise ValueError("Decoder output_shape must contain two positive integers")
        self.latent_dim, self.channels = latent_dim, channels
        self.output_shape = tuple(output_shape)
        self.bounded_output = bounded_output
        self.fc = nn.Sequential(nn.Linear(latent_dim, 32 * 5 * 8), nn.ELU())
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 3, 2, 1, output_padding=(0, 1)), nn.ELU(),
            nn.ConvTranspose2d(32, 16, 3, 2, 1, output_padding=(1, 1)), nn.ELU(),
            nn.Conv2d(16, channels, 3, 1, 1),
        )

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        if (not isinstance(latent, torch.Tensor) or latent.ndim < 1
                or latent.shape[-1] != self.latent_dim or not latent.is_floating_point()):
            raise ValueError("Decoder expects floating (...,latent_dim) inputs")
        prefix = latent.shape[:-1]
        # The caller controls precision/autocast. Upstream returns FP32 pixels.
        image = self.dec(self.fc(latent).reshape(-1, 32, 5, 8))
        image = (torch.sigmoid(image) if self.bounded_output else image).float()
        if self.output_shape != (18, 32):
            image = F.interpolate(image, size=self.output_shape, mode="bilinear", align_corners=False)
        return image.reshape(*prefix, self.channels, *self.output_shape)


@torch.no_grad()
def update_latent_statistics(module, latent: torch.Tensor, increment: torch.Tensor,
                             weight: torch.Tensor, rate: float = .01) -> None:
    """EMA mean/std and masked RMS increment, matching upstream's z buffers.

    The reference uses unbiased sample std. With exactly one sample its std
    is undefined; this adaptation uses correction=0 in that degenerate case.
    Empty batches leave the existing statistics unchanged.
    """
    if isinstance(rate, bool) or not isinstance(rate, Real) or not math.isfinite(rate) or not 0 <= rate <= 1:
        raise ValueError("Latent statistics rate must be finite and lie in [0,1]")
    buffers = (module.z_mean, module.z_std, module.dz_std)
    if (any(not isinstance(value, torch.Tensor) or value.ndim != 1 for value in buffers)
            or any(value.shape != buffers[0].shape for value in buffers[1:])):
        raise ValueError("Latent statistics require equal one-dimensional z_mean/z_std/dz_std buffers")
    dim = buffers[0].numel()
    if (latent.ndim < 1 or latent.shape[-1] != dim or increment.shape != latent.shape
            or weight.shape != latent.shape[:-1]):
        raise ValueError("Latent statistics inputs and validity weights must have matching shapes")
    if latent.numel() == 0:
        return
    z, dz, w = latent.reshape(-1, dim), increment.reshape(-1, dim), weight.reshape(-1, 1)
    module.z_mean.lerp_(z.mean(0), rate)
    module.z_std.lerp_(z.std(0, correction=1 if z.shape[0] > 1 else 0).clamp(min=1e-3), rate)
    module.dz_std.lerp_(((dz.square()*w).sum(0)/w.sum().clamp(min=1)).sqrt().clamp(min=1e-3), rate)


def normalized_latent_errors(predicted: torch.Tensor, target: torch.Tensor,
                             var_floor: float = 1e-6) -> torch.Tensor:
    """Per-step [B,K] MSE in units of one scalar target latent variance.

    Targets are not detached here: the WM learner must encode them under
    no_grad, matching InstinctMJ. Its training loss is (errors*mask).sum()/B.
    As with the statistics helper, a single sample uses population variance.
    """
    if (predicted.ndim != 3 or predicted.shape != target.shape
            or min(predicted.shape) < 1 or not predicted.is_floating_point()
            or not target.is_floating_point()):
        raise ValueError("Latent predictions and targets must have the same nonempty floating [B,K,D] shape")
    if (isinstance(var_floor, bool) or not isinstance(var_floor, Real)
            or not math.isfinite(var_floor) or var_floor <= 0):
        raise ValueError("Latent variance floor must be positive and finite")
    flat = target.flatten(0, 1)
    scale = flat.var(0, correction=1 if flat.shape[0] > 1 else 0).mean().clamp(min=var_floor)
    return (predicted-target).square().mean(-1)/scale
