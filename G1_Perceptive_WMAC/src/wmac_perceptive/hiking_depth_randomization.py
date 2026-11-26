"""RPL-inspired corruptions adapted to the unchanged Hiking camera pipeline.

Hiking packets contain optical-axis metres divided by 2.5 plus diagnostic
validity, at 36x64. This module changes only pixels in the Actor's 18x32 crop;
the existing crop -> Gaussian blur -> clip -> normalization still runs once
in the original preprocessing path. No resize, centered values, capture
decimation or camera-delay ring is introduced here.

The relative noise, dropout and terrain-biased side-mask ideas are adapted
from amazon-far/RPL 1ca0250ba3745381cfbfb2b72ebcf55e995076da and
amazon-far/holosoma 91647eed1ba82db3b242cd1057035ab76b63c5c0 (Apache-2.0).
Mask widths use fractions of Hiking's retained ROI, rather than copying the
external-camera native pixels. Synthetic dropout uses the policy's 2.5-m far
endpoint instead of RPL's negative near-fill; physical no-hits retain1e6m.
Synthetic mask values span Hiking's policy depth range.
These are explicit local adaptations, not an original Hiking noise preset.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

from .camera_calibration import load_g1_depth_calibration, raw_depth_to_metres


@dataclass(frozen=True)
class HikingDepthRandomizationCfg:
    height: int = 36
    width: int = 64
    crop_top: int = 18
    crop_bottom: int = 0
    crop_left: int = 16
    crop_right: int = 16
    max_depth: float = 2.5
    missing_depth_m: float = 1e6
    noise_enabled: bool = True
    std_multiplier: float = .1
    dropout_probability: float = .05
    dropout_depth_m: float = 2.5
    occlusion_enabled: bool = True
    occlusion_probability: float = 1.
    # Ratios are based on RPL's 240-pixel image, then applied to the retained
    # Hiking ROI so cropping cannot silently remove one side's augmentation.
    large_left_fraction: float = 90/240
    large_right_fraction: float = 50/240
    small_left_fraction: float = 60/240
    small_right_fraction: float = 20/240
    p_none: float = .1
    p_small: float = .45
    p_large: float = .45
    terrain_bias: float = .15

    def __post_init__(self):
        for name in ("height", "width"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 2:
                raise ValueError(f"{name} must be an integer at least two")
        for name in ("crop_top", "crop_bottom", "crop_left", "crop_right"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a nonnegative integer")
        if self.roi_height < 2 or self.roi_width < 2:
            raise ValueError("Hiking ROI must support the original blur and pooling")
        if not math.isfinite(self.max_depth) or self.max_depth <= 0:
            raise ValueError("max_depth must be finite and positive")
        if not math.isfinite(self.missing_depth_m) or self.missing_depth_m <= self.max_depth:
            raise ValueError("Hiking no-hit fill must remain beyond the policy depth range")
        if not math.isfinite(self.dropout_depth_m) or not 0 <= self.dropout_depth_m <= self.max_depth:
            raise ValueError("Synthetic dropout must stay in Hiking's policy depth range")
        if not math.isfinite(self.std_multiplier) or self.std_multiplier < 0:
            raise ValueError("std_multiplier must be finite and nonnegative")
        for name in ("noise_enabled", "occlusion_enabled"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"{name} must be a boolean")
        for name in ("dropout_probability", "occlusion_probability", "p_none", "p_small", "p_large", "terrain_bias",
                     "large_left_fraction", "large_right_fraction", "small_left_fraction", "small_right_fraction"):
            value = getattr(self, name)
            if not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"{name} must lie in [0,1]")
        if not math.isclose(self.p_none+self.p_small+self.p_large, 1., abs_tol=1e-7):
            raise ValueError("Occlusion mode probabilities must sum to one")
        if self.terrain_bias > min(self.p_small, self.p_large):
            raise ValueError("Terrain bias must preserve nonnegative probabilities")

    @property
    def roi_height(self):
        return self.height-self.crop_top-self.crop_bottom

    @property
    def roi_width(self):
        return self.width-self.crop_left-self.crop_right

    @property
    def roi_slices(self):
        return (slice(self.crop_top, self.height-self.crop_bottom),
                slice(self.crop_left, self.width-self.crop_right))

    @property
    def occlusion_widths(self):
        """(large_left, large_right, small_left, small_right) in ROI pixels."""
        return tuple(round(self.roi_width*value) for value in (
            self.large_left_fraction, self.large_right_fraction,
            self.small_left_fraction, self.small_right_fraction))


def _check_packet(packet, cfg):
    if (not isinstance(packet, torch.Tensor) or packet.ndim != 4
            or tuple(packet.shape[1:]) != (2, cfg.height, cfg.width)
            or not packet.is_floating_point()):
        raise ValueError("Expected floating Hiking packet [N,2,36,64]")
    # Both predicates are diagnostics over an already shape-checked packet.
    # Read them together once; preserve the original error priority.
    finite, bounded = torch.stack((torch.isfinite(packet).all(),
        ((packet[:, 1] >= 0) & (packet[:, 1] <= 1)).all())).cpu().tolist()
    if not finite:
        raise ValueError("Hiking depth randomization refuses non-finite packets")
    if not bounded:
        raise ValueError("Diagnostic validity must lie in [0,1]")


def _occlusion_strips(cfg, device):
    ll, lr, sl, sr = cfg.occlusion_widths
    cols = torch.arange(cfg.roi_width, device=device)
    return ((cols < ll) | (cols >= cfg.roi_width-lr),
            (cols < sl) | (cols >= cfg.roi_width-sr))


def augment_hiking_depth_packet(packet, terrain_is_stepping_stones=None,
                               cfg=HikingDepthRandomizationCfg(), *, enabled=True, generator=None):
    """Sample one ROI corruption, preserving raw scale and far/no-hit semantics.

    Noise and dropout apply only to physically valid ROI depths. Occlusion is
    independently allowed to replace a border regardless of its physical hit;
    its validity is marked false because it is synthetic sensor information.
    No crop, blur, clipping or delayed-history update happens in this function.
    """
    _check_packet(packet, cfg)
    if not isinstance(enabled, bool):
        raise ValueError("enabled must be a boolean")
    return _augment_checked_packet(packet, terrain_is_stepping_stones, cfg,
        enabled=enabled, generator=generator)


def _augment_checked_packet(packet, terrain_is_stepping_stones, cfg, *, enabled,
                            generator=None, strips=None):
    """Same draws and pixel formula after the caller has validated the packet."""
    if not enabled:
        return packet
    n = packet.shape[0]
    ys, xs = cfg.roi_slices
    result = packet.clone()
    depth = packet[:, 0, ys, xs]*cfg.max_depth
    valid = ((packet[:, 1, ys, xs] > .5) & (depth >= 0.) & (depth <= cfg.max_depth))
    if cfg.noise_enabled:
        # Do not perturb the 1e6-m missing sentinel or physically out-of-range
        # pixels; they must retain Hiking's blur-before-clipping behavior.
        noisy = torch.normal(depth, depth.abs()*cfg.std_multiplier, generator=generator)
        depth = torch.where(valid, noisy, depth)
        dropped = torch.bernoulli(torch.full_like(depth, cfg.dropout_probability), generator=generator).bool() & valid
        # Synthetic holes use the policy far endpoint, rather than a1e6-m
        # physical-miss sentinel: blur would otherwise turn a5% pixel loss
        # into saturated neighborhoods covering roughly37% of the image.
        depth = torch.where(dropped, torch.full_like(depth, cfg.dropout_depth_m), depth)
        result[:, 1, ys, xs] = torch.where(dropped, torch.zeros_like(depth), result[:, 1, ys, xs])
    if cfg.occlusion_enabled:
        if terrain_is_stepping_stones is None:
            raise ValueError("Terrain-biased occlusion needs a per-environment stepping-stones mask")
        stones = torch.as_tensor(terrain_is_stepping_stones, dtype=torch.bool, device=packet.device)
        if tuple(stones.shape) != (n,):
            raise ValueError("Expected one stepping-stones flag per environment")
        small_probability = cfg.p_small+torch.where(stones, cfg.terrain_bias, -cfg.terrain_bias)
        choice = torch.rand(n, device=packet.device, generator=generator)
        small = (choice >= cfg.p_none) & (choice < cfg.p_none+small_probability)
        large = choice >= cfg.p_none+small_probability
        apply = torch.rand(n, device=packet.device, generator=generator) < cfg.occlusion_probability
        large_strip, small_strip = (_occlusion_strips(cfg, packet.device)
                                    if strips is None else strips)
        mask = apply[:, None, None] & ((large[:, None, None] & large_strip[None, None])
                                       | (small[:, None, None] & small_strip[None, None]))
        synthetic = cfg.max_depth*torch.rand(n, cfg.roi_height, cfg.roi_width,
            device=packet.device, dtype=packet.dtype, generator=generator)
        depth = torch.where(mask, synthetic, depth)
        result[:, 1, ys, xs] = torch.where(mask, torch.zeros_like(depth), result[:, 1, ys, xs])
    result[:, 0, ys, xs] = depth/cfg.max_depth
    return result


class HikingDepthCapture:
    """One randomized raw packet per 50Hz control step, without extra delay.

    Repeated reads of a control step share the exact cached packet. Call
    ``reset(mask)`` after copying the terminal packet: the next read refreshes
    only those reset environments, using their new physical depth. The
    existing Hiking history owns its original one-control-step delay.
    """
    capture_interval = 1

    def __init__(self, num_envs, device="cpu", cfg=HikingDepthRandomizationCfg(), *,
                 enabled_randomization=True, generator=None):
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if not isinstance(enabled_randomization, bool):
            raise ValueError("enabled_randomization must be a boolean")
        self.num_envs, self.device, self.cfg = num_envs, torch.device(device), cfg
        self.enabled_randomization, self.generator = enabled_randomization, generator
        self._packet = torch.empty(num_envs, 2, cfg.height, cfg.width, device=self.device)
        self._pending = torch.ones(num_envs, dtype=torch.bool, device=self.device)
        self._has_pending = True
        self.last_control_step = None
        self.last_capture_step = None
        self.capture_count = 0
        self.captures_per_env = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        self._strips_cfg = cfg
        self._strips = _occlusion_strips(cfg, self.device)

    @property
    def packet(self):
        return self._packet

    def cached_packet(self, *, control_step):
        """Return an unchanged capture without constructing a new raw packet.

        A same-step reset invalidates this shortcut until the reset rows have
        been captured. Querying the cache neither samples randomness nor
        advances the capture counters.
        """
        if isinstance(control_step, bool) or not isinstance(control_step, int) or control_step < 0:
            raise ValueError("control_step must be a nonnegative integer")
        if self.last_control_step is not None and control_step < self.last_control_step:
            raise ValueError("Camera control steps cannot rewind")
        if self.last_control_step == control_step and not self._has_pending:
            return self._packet
        return None

    def reset(self, mask=None):
        if mask is None:
            self._pending.fill_(True)
            self._has_pending = True
            return
        value = torch.as_tensor(mask, device=self.device)
        if value.dtype == torch.bool:
            if tuple(value.shape) != (self.num_envs,):
                raise ValueError("Reset mask needs one boolean per environment")
            ids = value.nonzero(as_tuple=False).flatten()
        elif value.ndim == 1 and value.dtype in (torch.int32, torch.int64):
            ids = value.long()
            if ids.numel() and (bool((ids < 0).any()) or bool((ids >= self.num_envs).any())):
                raise ValueError("Reset environment ID is out of bounds")
        else:
            raise ValueError("Reset needs a boolean mask or integer environment IDs")
        if ids.numel():
            self._pending[ids] = True
            self._has_pending = True

    @torch.no_grad()
    def step(self, packet, *, control_step, terrain_is_stepping_stones=None):
        cached = self.cached_packet(control_step=control_step)
        if cached is not None:
            return cached
        same_step = self.last_control_step == control_step
        _check_packet(packet, self.cfg)
        if packet.shape[0] != self.num_envs or packet.device != self.device:
            raise ValueError("Physical packet batch/device differs from the capture cache")
        # Only a same-step reset needs advanced indexing. New physical frames
        # refresh every environment, so avoid gathering/scattering an entire
        # image batch through arange(N). Keep reset rows in the original
        # ascending order to preserve stochastic draw shapes and RNG state.
        ids = self._pending.nonzero(as_tuple=False).flatten() if same_step else None
        stones = None
        if terrain_is_stepping_stones is not None:
            stones = torch.as_tensor(terrain_is_stepping_stones, dtype=torch.bool, device=self.device)
            if tuple(stones.shape) != (self.num_envs,):
                raise ValueError("Expected one stepping-stones flag per environment")
            if ids is not None:
                stones = stones[ids]
        if self._strips_cfg != self.cfg:
            self._strips_cfg, self._strips = self.cfg, _occlusion_strips(self.cfg, self.device)
        # Advanced indexing previously made all sources contiguous, including
        # unusual externally supplied packet strides. Match that layout.
        source = packet[ids] if ids is not None else packet.contiguous()
        sampled = _augment_checked_packet(source, stones, self.cfg,
            enabled=self.enabled_randomization, generator=self.generator, strips=self._strips)
        if ids is None:
            self._packet.copy_(sampled)
            self._pending.fill_(False)
            self.captures_per_env.add_(1)
        else:
            self._packet[ids] = sampled
            self._pending[ids] = False
            self.captures_per_env[ids] += 1
        self._has_pending = False
        self.last_control_step = control_step
        self.last_capture_step = control_step
        self.capture_count += 1
        return self._packet


@torch.no_grad()
def preprocess_real_g1_hiking_z16(raw, calibration=None):
    """Real Z16 -> measured-scale depth -> Hiking 64x36/crop/blur/[0,1].

    ``nearest-exact`` selects source pixel centres consistently with resized K
    and keeps invalid/no-hit sentinels from being interpolated into nearby
    depths before Hiking's intentional Gaussian blur. No range pre-clipping,
    synthetic noise, centered normalization or RPL-size resize is applied.
    Real 30Hz input does not become 50Hz measurements: the deployment loop
    must hold the latest real frame between hardware arrivals.
    """
    from .perception import PerceptionSpec
    from .visual_metrics import preprocess_depth_channels
    calibration = calibration or load_g1_depth_calibration()
    if raw.ndim == 2:
        image = raw[None, None]
    elif raw.ndim == 3:
        image = raw[:, None]
    elif raw.ndim == 4 and raw.shape[1] == 1:
        image = raw
    else:
        raise ValueError("Real G1 depth must have shape [H,W], [N,H,W], or [N,1,H,W]")
    if tuple(image.shape[-2:]) != (calibration.height, calibration.width):
        raise ValueError("Raw image differs from the selected calibrated stream")
    depth, valid = raw_depth_to_metres(image, calibration)
    depth = torch.where(valid, depth, torch.full_like(depth, 1e6))
    if tuple(depth.shape[-2:]) != (36, 64):
        depth = F.interpolate(depth, size=(36, 64), mode="nearest-exact")
    spec = PerceptionSpec(include_valid=False)
    return preprocess_depth_channels(depth/2.5, spec, channels=1, include_valid=False).clamp(0, 1)
