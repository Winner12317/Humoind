"""Measured G1 depth intrinsics and RealSense optical-Z conventions.

RealSense deprojection uses integer pixel-centre coordinates, optical +X right,
+Y down and +Z forward. The ray-caster uses +X forward, +Y left and +Z up.
Only the captured depth profile is bundled; camera-internal RGB extrinsics do
not supply a camera-to-robot mounting transform.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from pathlib import Path

import torch


G1_DEPTH_CALIBRATION_PATH = Path(__file__).resolve().parents[1] / "assets/cameras/g1_depth_calibration.json"


@dataclass(frozen=True)
class DepthCameraCalibration:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    fps: float
    depth_scale_m_per_raw_unit: float
    distortion_model: str = "Brown Conrady"
    distortion_coefficients: tuple[float, ...] = (0., 0., 0., 0., 0.)
    source_report_sha256: str = ""

    def __post_init__(self):
        for name in ("width", "height"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("fx", "fy", "fps", "depth_scale_m_per_raw_unit"):
            value = getattr(self, name)
            if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("cx", "cy"):
            if not math.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if self.distortion_model not in ("Brown Conrady", "Inverse Brown Conrady", "None"):
            raise ValueError("Unsupported depth distortion model")
        if len(self.distortion_coefficients) != 5 or any(
            not math.isfinite(v) or v != 0. for v in self.distortion_coefficients
        ):
            raise ValueError("Nonzero distortion requires explicit rectification before pinhole ray generation")

    def resized(self, width: int, height: int) -> "DepthCameraCalibration":
        """Intrinsics for align_corners=False, pixel-centred image resizing.

        Destination pixel u samples source (u+.5)/sx-.5. Scaling cx without
        this half-pixel correction moves the optical axis after resizing.
        """
        if any(isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in (width, height)):
            raise ValueError("Resized width and height must be positive integers")
        sx, sy = width / self.width, height / self.height
        return replace(self, width=width, height=height, fx=self.fx*sx, fy=self.fy*sy,
                       cx=(self.cx+.5)*sx-.5, cy=(self.cy+.5)*sy-.5)

    def cropped(self, left: int, top: int, width: int, height: int) -> "DepthCameraCalibration":
        """Intrinsics after an integer crop, before any subsequent resizing."""
        if any(isinstance(v, bool) or not isinstance(v, int) for v in (left, top, width, height)):
            raise ValueError("Crop coordinates and size must be integers")
        if left < 0 or top < 0 or width < 1 or height < 1 or left+width > self.width or top+height > self.height:
            raise ValueError("Crop must lie inside the calibrated image")
        return replace(self, width=width, height=height, cx=self.cx-left, cy=self.cy-top)

    @property
    def fov_degrees(self) -> tuple[float, float]:
        """Boundary-to-boundary FOV, retaining the measured asymmetric centre."""
        return tuple(math.degrees(math.atan((c+.5)/f)+math.atan((n-c-.5)/f))
                     for c, f, n in ((self.cx, self.fx, self.width), (self.cy, self.fy, self.height)))


def load_g1_depth_calibration(path: str | Path | None = None) -> DepthCameraCalibration:
    """Load the sanitized active Depth/Z16 profile shipped with the code."""
    profile = json.loads(Path(path or G1_DEPTH_CALIBRATION_PATH).read_text())
    if profile.get("schema_version") != 1:
        raise ValueError("Unsupported depth calibration schema")
    if profile.get("stream") != "Depth" or profile.get("format") != "Z16" or profile.get("aligned_to_color") is not False:
        raise ValueError("Calibration must describe an unaligned Depth/Z16 stream")
    if profile.get("depth_convention") != "optical_axis_z_metres" or profile.get("pixel_coordinate_convention") != "integer_coordinates_are_pixel_centres":
        raise ValueError("Unsupported depth or pixel coordinate convention")
    fields = ("width", "height", "fx", "fy", "cx", "cy", "fps", "depth_scale_m_per_raw_unit",
              "distortion_model", "source_report_sha256")
    return DepthCameraCalibration(**{name: profile[name] for name in fields},
                                  distortion_coefficients=tuple(profile["distortion_coefficients"]))


def calibrated_optical_rays(calibration: DepthCameraCalibration | None = None, *,
                            width: int | None = None, height: int | None = None,
                            device="cpu", dtype=torch.float32):
    """Return unit rays [H*W,3] and optical-Z/range factors in torso-camera axes.

    Resized output pixels sample the same calibrated pinhole, with their own
    scaled focal lengths and asymmetric principal point. These directions
    still need the existing camera-to-torso rotation and position.
    """
    calibration = calibration or load_g1_depth_calibration()
    if (width is None) != (height is None):
        raise ValueError("Provide both output width and height")
    if width is not None:
        calibration = calibration.resized(width, height)
    if dtype not in (torch.float32, torch.float64):
        raise ValueError("Pinhole rays require float32 or float64")
    u, v = torch.meshgrid(torch.arange(calibration.width, device=device, dtype=dtype),
                         torch.arange(calibration.height, device=device, dtype=dtype), indexing="xy")
    x = (u-calibration.cx)/calibration.fx
    y = (v-calibration.cy)/calibration.fy
    raw = torch.stack((torch.ones_like(x), -x, -y), -1).reshape(-1, 3)
    cosines = raw.norm(dim=-1).reciprocal()
    return raw*cosines[:, None], cosines


def raw_depth_to_metres(raw: torch.Tensor, calibration: DepthCameraCalibration | None = None, *,
                         max_depth: float | None = None):
    """Convert raw RealSense Z16 units to optical-axis metres and a valid mask.

    Invalid, nonfinite and out-of-Z16-range values are returned as zero. The
    optional max_depth is a downstream application range, not a camera range
    inferred from the six-second capture. No RGB alignment is applied.
    """
    calibration = calibration or load_g1_depth_calibration()
    if raw.dtype == torch.bool or raw.is_complex():
        raise ValueError("Raw depth must contain numeric Z16 units")
    if max_depth is not None and (not math.isfinite(max_depth) or max_depth <= 0):
        raise ValueError("max_depth must be finite and positive")
    values = raw.to(torch.float32)
    metres = values*calibration.depth_scale_m_per_raw_unit
    valid = torch.isfinite(values) & (values > 0) & (values <= 65535)
    if max_depth is not None:
        valid = valid & (metres <= max_depth)
    return torch.where(valid, metres, torch.zeros_like(metres)), valid
