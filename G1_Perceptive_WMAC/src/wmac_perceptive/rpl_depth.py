"""RPL/FALCON depth observation processing, independent of the learner.

Adapted from amazon-far/RPL 1ca0250ba3745381cfbfb2b72ebcf55e995076da,
``rpl_training/config_values/camera.py`` and ``terms/observation.py``, and
amazon-far/holosoma 91647eed1ba82db3b242cd1057035ab76b63c5c0,
``managers/camera/terms/warp_depth.py`` (Apache-2.0).

Input is finite optical-axis Z depth in metres, captured once per camera
update. Noise is drawn once here, before storing the observation/WM target;
these functions must never be called inside an Actor forward pass. Pose and
FOV sampling helpers are separate: upstream samples them once at sensor init,
not on every frame or automatically on episode reset.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch.nn import functional as F

RPL_REFERENCE_COMMIT = "1ca0250ba3745381cfbfb2b72ebcf55e995076da"
HOLOSOMA_REFERENCE_COMMIT = "91647eed1ba82db3b242cd1057035ab76b63c5c0"


@dataclass(frozen=True)
class RPLDepthConfig:
    """The enabled RPL depth-camera preset, without its external-camera rig."""

    native_width: int = 240
    native_height: int = 135
    output_width: int = 48
    output_height: int = 27
    near_clip: float = .1
    far_clip: float = 2.
    ray_far_plane: float = 2.5
    noise_enabled: bool = True
    std_multiplier: float = .1
    dropout_probability: float = .05
    dropout_value: float = -2.5
    occlusion_enabled: bool = True
    occlusion_probability: float = 1.
    occlusion_box: tuple[int, int, int, int] = (90, 50, 0, 0)
    occlusion_diff: int = 30
    terrain_bias: float = .15
    p_none: float = .1
    p_small: float = .45
    p_large: float = .45
    share_across_cameras: bool = True
    frame_rate: float = 10.
    delay: int = 2
    delay_buffer_length: int = 4

    def __post_init__(self):
        for name in ("native_width", "native_height", "output_width", "output_height", "delay_buffer_length"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if isinstance(self.delay, bool) or not isinstance(self.delay, int) or self.delay < 0:
            raise ValueError("delay must be a nonnegative integer")
        if self.delay >= self.delay_buffer_length:
            raise ValueError("delay_buffer_length must cover the selected delay")
        for name in ("near_clip", "far_clip", "ray_far_plane", "std_multiplier", "dropout_value", "frame_rate"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not 0 < self.near_clip < self.far_clip <= self.ray_far_plane or self.frame_rate <= 0:
            raise ValueError("Invalid RPL camera range/frame rate")
        if self.std_multiplier < 0:
            raise ValueError("std_multiplier must be nonnegative")
        for name in ("dropout_probability", "occlusion_probability", "p_none", "p_small", "p_large", "terrain_bias"):
            if not math.isfinite(getattr(self, name)) or not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be a probability")
        if not math.isclose(self.p_none + self.p_small + self.p_large, 1., abs_tol=1e-7):
            raise ValueError("Occlusion mode probabilities must sum to one")
        if self.terrain_bias > min(self.p_small, self.p_large):
            raise ValueError("terrain_bias must preserve nonnegative mode probabilities")
        if len(self.occlusion_box) != 4 or any(isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in self.occlusion_box):
            raise ValueError("occlusion_box must contain four nonnegative integer widths")
        if isinstance(self.occlusion_diff, bool) or not isinstance(self.occlusion_diff, int) or self.occlusion_diff < 0:
            raise ValueError("occlusion_diff must be a nonnegative integer")


def _check_depth(depth):
    if depth.ndim != 4 or not depth.is_floating_point():
        raise ValueError("RPL depth must be floating-point [N, cameras, H, W] in metres")


def rpl_depth_noise(depth_m, cfg=RPLDepthConfig(), *, generator=None):
    """Gaussian std=.1*abs(depth), followed by independent 5% near dropout."""
    _check_depth(depth_m)
    if not cfg.noise_enabled:
        return depth_m
    noisy = torch.normal(mean=depth_m, std=cfg.std_multiplier * depth_m.abs(), generator=generator)
    dropped = torch.bernoulli(torch.full_like(depth_m, cfg.dropout_probability), generator=generator).bool()
    return torch.where(dropped, torch.full_like(noisy, cfg.dropout_value), noisy)


def rpl_depth_occlusion(depth_m, terrain_is_stepping_stones, cfg=RPLDepthConfig(), *, generator=None):
    """RPL terrain-biased random-depth border strips, drawn per capture.

    RPL's index0 is stepping stones and indices1..3 are slopes/stairs. Passing
    the semantic boolean rather than those numeric indices lets another
    terrain generator preserve this rule without silently changing its map.
    """
    _check_depth(depth_m)
    if not cfg.occlusion_enabled:
        return depth_m
    n, cameras, height, width = depth_m.shape
    if terrain_is_stepping_stones is None:
        raise ValueError("RPL border occlusion requires a per-environment stepping-stones mask")
    stones = torch.as_tensor(terrain_is_stepping_stones, device=depth_m.device, dtype=torch.bool)
    if tuple(stones.shape) != (n,):
        raise ValueError("Expected one stepping-stones flag per environment")
    # Upstream shifts probability mass toward small strips on stones and
    # toward large strips on the other source terrain families.
    p_small = torch.full((n,), cfg.p_small, device=depth_m.device)
    p_small = p_small + torch.where(stones, cfg.terrain_bias, -cfg.terrain_bias)
    choices = torch.rand(n, device=depth_m.device, generator=generator)
    mode_small = (choices >= cfg.p_none) & (choices < cfg.p_none + p_small)
    mode_large = choices >= cfg.p_none + p_small
    apply = torch.rand(n, device=depth_m.device, generator=generator) < cfg.occlusion_probability
    x1, x2, y1, y2 = (
        max(0, min(limit, round(value * scale)))
        for value, scale, limit in zip(
            cfg.occlusion_box,
            (width/cfg.native_width, width/cfg.native_width, height/cfg.native_height, height/cfg.native_height),
            (width, width, height, height),
        )
    )
    diff = round(cfg.occlusion_diff * width/cfg.native_width)
    rows = torch.arange(height, device=depth_m.device)[:, None]
    cols = torch.arange(width, device=depth_m.device)[None, :]
    large = (rows < y1) | (rows >= height-y2) | (cols < x1) | (cols >= width-x2)
    small = (cols < max(0, x1-diff)) | (cols >= width-max(0, x2-diff))
    selected = apply[:, None, None] & (
        (mode_large[:, None, None] & large) | (mode_small[:, None, None] & small)
    )
    image_cameras = 1 if cfg.share_across_cameras else cameras
    random_depth = cfg.near_clip + (cfg.far_clip-cfg.near_clip) * torch.rand(
        n, image_cameras, height, width, device=depth_m.device, generator=generator
    )
    return torch.where(selected[:, None], random_depth, depth_m)


def resize_rpl_depth(depth_m, cfg=RPLDepthConfig()):
    """Training ordering: clip metric depth, bicubic resize, no second clip."""
    _check_depth(depth_m)
    return F.interpolate(depth_m.clamp(cfg.near_clip, cfg.far_clip),
        size=(cfg.output_height, cfg.output_width), mode="bicubic", align_corners=False, antialias=True)


def normalize_rpl_depth(processed_m, cfg=RPLDepthConfig()):
    """Zero-centred RPL observations; bicubic overshoot is intentionally kept."""
    return (processed_m-cfg.near_clip)/(cfg.far_clip-cfg.near_clip)-.5


def preprocess_rpl_depth(depth_m, cfg=RPLDepthConfig(), *, training=True,
                         terrain_is_stepping_stones=None, generator=None, normalize=True):
    """One capture's shared corruption/processing; temporal delay is separate.

    ``normalize=False`` returns the processed metric image for RPLDepthDelay.
    Deterministic evaluation disables corruption but keeps the training
    clip-before-resize contract. The upstream standalone sim2sim server uses
    a different resize-before-clip order and is not copied into this path.
    """
    value = depth_m
    if training:
        value = rpl_depth_noise(value, cfg, generator=generator)
        value = rpl_depth_occlusion(value, terrain_is_stepping_stones, cfg, generator=generator)
    processed = resize_rpl_depth(value, cfg)
    return normalize_rpl_depth(processed, cfg) if normalize else processed


class RPLDepthDelay:
    """Original four-capture ring, read at age2, with reset fill at depth1.05m."""

    def __init__(self, num_envs, cfg=RPLDepthConfig(), *, cameras=1, device="cpu", generator=None):
        if num_envs < 1 or cameras < 1:
            raise ValueError("num_envs/cameras must be positive")
        self.cfg = cfg
        self.buffer = torch.full((num_envs, cfg.delay_buffer_length, cameras, cfg.output_height,
                                  cfg.output_width), .5*(cfg.near_clip+cfg.far_clip), device=device)
        self.delay_idx = torch.randint(cfg.delay, cfg.delay+1, (num_envs,), device=device, generator=generator)
        self.env_arange = torch.arange(num_envs, device=device)

    def reset(self, env_ids, *, generator=None):
        ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.buffer.device)
        self.buffer[ids] = .5*(self.cfg.near_clip+self.cfg.far_clip)
        self.delay_idx[ids] = torch.randint(self.cfg.delay, self.cfg.delay+1, (len(ids),),
                                         device=self.buffer.device, generator=generator)

    def push(self, processed_m, *, reset_ids=None, generator=None):
        if tuple(processed_m.shape) != tuple(self.buffer[:, 0].shape):
            raise ValueError("Processed depth shape differs from the RPL delay buffer")
        if reset_ids is not None:
            self.reset(reset_ids, generator=generator)
        self.buffer[:, 1:] = self.buffer[:, :-1].clone()
        self.buffer[:, 0] = processed_m
        return self.output_depth

    @property
    def output_depth(self):
        return self.buffer[self.env_arange, self.delay_idx]

    @property
    def normalized(self):
        return normalize_rpl_depth(self.output_depth, self.cfg)


def sample_rpl_mount_offsets(num_envs, *, device="cpu", generator=None):
    """Init-only offsets: xyz in metres, roll/pitch/yaw in degrees.

    Upstream adds these Euler offsets to the nominal Euler angles before
    conversion to a quaternion. It does not compose a random quaternion onto
    the nominal quaternion; callers adapting the G1 mount should do likewise.
    """
    translation = .05*torch.rand(num_envs, 3, device=device, generator=generator)-.025
    limits = torch.tensor([2.5, 3., 2.5], device=device)
    rotation_deg = 2*limits*torch.rand(num_envs, 3, device=device, generator=generator)-limits
    return translation, rotation_deg


def sample_rpl_horizontal_fov(nominal_degrees, num_envs, *, device="cpu", generator=None):
    """Init-only ±5 degree variation, centred on the measured G1 HFOV."""
    if not 5 < nominal_degrees < 175:
        raise ValueError("Nominal FOV must keep its ±5-degree range within (0, 180)")
    return (nominal_degrees-5.) + 10.*torch.rand(num_envs, device=device, generator=generator)
