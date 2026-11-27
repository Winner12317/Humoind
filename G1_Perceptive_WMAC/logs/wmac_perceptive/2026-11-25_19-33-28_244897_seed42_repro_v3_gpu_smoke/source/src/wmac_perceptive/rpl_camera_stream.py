"""Capture-once RPL camera timing and the measured G1 real-depth adapter.

The RPL preset captures at 10 Hz and reads a four-capture ring at age two.
Its synthetic 200 ms delay is retained here; the G1 report's 30 Hz stream
period does not establish a measured exposure-to-policy delay. The caller
software-subsamples that stream, rather than interpreting two captures as
two 50 Hz policy steps. The learner's eight-frame latent history is separate.

RPL postprocessing is adapted under Apache-2.0 from the pinned versions in
``rpl_depth``. Resampling the measured G1 native image before that pipeline
is a local sensor adaptation, not an additional RPL randomization rule.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

from .camera_calibration import load_g1_depth_calibration, raw_depth_to_metres
from .rpl_depth import RPLDepthConfig, RPLDepthDelay, normalize_rpl_depth, preprocess_rpl_depth


class RPLCameraStream:
    """Cache one randomized packet per capture and hold it between captures.

    ``step`` returns the same owned [N,2,H,W] cache on repeated reads. Channel
    zero is the delayed, normalized depth; channel one is diagnostic raw-hit
    validity (nearest downsampling, before synthetic noise/occlusion). It is
    not a policy input. Copy a terminal packet before calling ``reset``:

    ``terminal = stream.step(...).clone(); stream.reset(done_mask)``.

    Reset clears only the selected environments to RPL's neutral 1.05 m
    delay fill. It does not rewind the capture clock, resample the camera
    pose/FOV, or recapture a post-reset image in the same control step.
    """

    def __init__(self, num_envs, device="cpu", cfg=RPLDepthConfig(), *,
                 capture_period=.1, control_dt=.02, enabled_randomization=True,
                 generator=None):
        if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if (not math.isfinite(capture_period) or capture_period <= 0
                or not math.isfinite(control_dt) or control_dt <= 0):
            raise ValueError("Camera and control periods must be finite and positive")
        interval = capture_period/control_dt
        if not math.isclose(interval, round(interval), rel_tol=0., abs_tol=1e-8) or round(interval) < 1:
            raise ValueError("Capture period must be an integer number of control steps")
        if not isinstance(enabled_randomization, bool):
            raise ValueError("enabled_randomization must be a boolean")
        self.num_envs, self.device, self.cfg = num_envs, torch.device(device), cfg
        self.capture_period, self.control_dt = float(capture_period), float(control_dt)
        self.capture_interval = int(round(interval))
        self.enabled_randomization, self.generator = enabled_randomization, generator
        self.delay = RPLDepthDelay(num_envs, cfg, cameras=1, device=self.device, generator=generator)
        self.device = self.delay.buffer.device
        self._valid_buffer = torch.zeros_like(self.delay.buffer, dtype=torch.bool)
        self._packet = torch.zeros((num_envs, 2, cfg.output_height, cfg.output_width), device=self.device)
        self.last_control_step = None
        self.last_capture_step = None
        self.capture_count = 0

    def _reset_ids(self, mask):
        if mask is None:
            return torch.arange(self.num_envs, device=self.device)
        value = torch.as_tensor(mask, device=self.device)
        if value.dtype == torch.bool:
            if tuple(value.shape) != (self.num_envs,):
                raise ValueError("Reset mask must contain one boolean per environment")
            return value.nonzero(as_tuple=False).flatten()
        if value.ndim != 1 or value.dtype not in (torch.int32, torch.int64):
            raise ValueError("Reset must use a boolean mask or a vector of integer environment IDs")
        if value.numel() and (bool((value < 0).any()) or bool((value >= self.num_envs).any())):
            raise ValueError("Reset environment ID is out of bounds")
        return value.long()

    @torch.no_grad()
    def reset(self, mask=None):
        ids = self._reset_ids(mask)
        if not ids.numel():
            return self._packet
        self.delay.reset(ids, generator=self.generator)
        self._valid_buffer[ids] = False
        # The source metric fill normalizes to zero. Keep an exact neutral
        # packet while the delay buffer itself retains its metric source fill.
        self._packet[ids] = 0.
        return self._packet

    @torch.no_grad()
    def step(self, depth_m, *, control_step, terrain_is_stepping_stones,
             reset_mask=None):
        if isinstance(control_step, bool) or not isinstance(control_step, int) or control_step < 0:
            raise ValueError("control_step must be a nonnegative integer")
        if self.last_control_step is not None and control_step < self.last_control_step:
            raise ValueError("Camera stream control steps cannot rewind")
        if reset_mask is not None:
            self.reset(reset_mask)
        if self.last_control_step == control_step:
            return self._packet
        if self.last_capture_step is not None and control_step-self.last_capture_step < self.capture_interval:
            self.last_control_step = control_step
            return self._packet
        expected = (self.num_envs, 1, self.cfg.native_height, self.cfg.native_width)
        if not isinstance(depth_m, torch.Tensor) or tuple(depth_m.shape) != expected or not depth_m.is_floating_point():
            raise ValueError(f"Expected floating metric camera depth with shape {expected}")
        if depth_m.device != self.device:
            raise ValueError("Camera depth and stream device differ")
        value = depth_m.float()
        valid = torch.isfinite(value) & (value > 0.) & (value <= self.cfg.far_clip)
        # Simulated ray misses and raw RealSense invalids are represented by
        # far range before entering the source noise/occlusion pipeline.
        value = torch.where(torch.isfinite(value) & (value > 0.),
                            value.clamp(max=self.cfg.ray_far_plane),
                            torch.full_like(value, self.cfg.ray_far_plane))
        processed = preprocess_rpl_depth(value, self.cfg,
            training=self.enabled_randomization,
            terrain_is_stepping_stones=terrain_is_stepping_stones,
            generator=self.generator, normalize=False)
        delayed = self.delay.push(processed)
        self._valid_buffer[:, 1:] = self._valid_buffer[:, :-1].clone()
        self._valid_buffer[:, 0] = F.interpolate(valid.float(),
            size=(self.cfg.output_height, self.cfg.output_width), mode="nearest").bool()
        self._packet[:, :1] = normalize_rpl_depth(delayed, self.cfg)
        self._packet[:, 1:] = self._valid_buffer[self.delay.env_arange, self.delay.delay_idx]
        self.last_control_step = control_step
        self.last_capture_step = control_step
        self.capture_count += 1
        return self._packet

    @property
    def packet(self):
        return self._packet


@torch.no_grad()
def preprocess_real_g1_z16(raw, calibration=None, *, cfg=RPLDepthConfig()):
    """Deterministic G1 deployment pixels, without synthetic training noise.

    Accept [H,W], [N,H,W], or [N,1,H,W] original, unaligned Z16 depth.
    Convert with the measured scale; fill invalids and cap beyond the 2.5 m
    simulated ray range; bicubic/antialias resize 848x480 to 240x135 using
    align_corners=False; then perform RPL's clip -> 48x27 bicubic resize ->
    normalization. A second native-size resize is avoided for native input.
    No crop, RGB alignment, camera extrinsics, or stochastic corruption is
    introduced. Capture cadence and the synthetic delay are stream concerns,
    not inferred from the real camera timestamps by this function.
    """
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
        raise ValueError("Raw G1 dimensions differ from the selected calibrated stream")
    depth, valid = raw_depth_to_metres(image, calibration)
    depth = torch.where(valid, depth.clamp(max=cfg.ray_far_plane),
                        torch.full_like(depth, cfg.ray_far_plane))
    if tuple(depth.shape[-2:]) != (cfg.native_height, cfg.native_width):
        depth = F.interpolate(depth, size=(cfg.native_height, cfg.native_width),
                              mode="bicubic", align_corners=False, antialias=True)
    return preprocess_rpl_depth(depth, cfg, training=False)
