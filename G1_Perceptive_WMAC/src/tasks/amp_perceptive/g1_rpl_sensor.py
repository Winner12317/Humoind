"""Measured G1 body camera with RPL init-only placement and FOV variation.

Only camera geometry is randomized here. RPL depth corruption, clipping,
resizing and delay live in the observation pipeline. The G1 torso mount is
retained; RPL's added front/back camera rig is not transplanted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch
import warp as wp

from mjlab.sensor import PinholeCameraPatternCfg, RayCastSensorCfg
from mjlab.sensor.raycast_sensor import RayCastSensor
from mjlab.utils.lab_api.math import (
    euler_xyz_from_quat, matrix_from_quat, quat_apply, quat_from_euler_xyz,
)
from src.wmac_perceptive.camera_calibration import (
    calibrated_optical_rays, load_g1_depth_calibration,
)
from src.wmac_perceptive.rpl_depth import sample_rpl_horizontal_fov, sample_rpl_mount_offsets
from .sensors import HikingDepthPatternCfg


@dataclass
class G1RPLDepthPatternCfg(PinholeCameraPatternCfg):
    width: int = 240
    height: int = 135
    # Metadata only; projection below uses the measured non-centred K.
    fovy: float = 58.11279497342981
    position: tuple[float, float, float] = HikingDepthPatternCfg.position
    rotation: tuple[float, float, float, float] = HikingDepthPatternCfg.rotation
    sensor_protocol: str = "g1_rpl_depth_v1"

    def optical_rays(self, device="cpu"):
        return calibrated_optical_rays(width=self.width, height=self.height, device=device)

    def generate_rays(self, mj_model, device):
        del mj_model
        directions, _ = self.optical_rays(device)
        quaternion = torch.tensor(self.rotation, device=device)
        quaternion = quaternion/quaternion.norm()
        directions = quat_apply(quaternion.expand(len(directions), 4), directions)
        offsets = torch.tensor(self.position, device=device).expand(len(directions), 3).clone()
        return offsets, directions


@dataclass(frozen=True)
class G1CameraGeometry:
    positions: torch.Tensor
    quaternions: torch.Tensor
    horizontal_fov_degrees: torch.Tensor
    optical_cosines: torch.Tensor
    local_directions: torch.Tensor
    translation_offsets: torch.Tensor
    rotation_offsets_degrees: torch.Tensor


def sample_g1_camera_geometry(pattern: G1RPLDepthPatternCfg, num_envs: int, *,
                              randomize_camera=True, device="cpu", generator=None):
    """Construct persistent per-env rays; do not call this on episode reset.

    RPL adds roll/pitch/yaw offsets to nominal Euler angles before quaternion
    conversion. Both focal lengths vary together, preserving measured pixel
    aspect and the asymmetric principal point. The sampled HFOV describes
    the two image boundaries, rather than assuming a centred principal point.
    """
    if isinstance(num_envs, bool) or not isinstance(num_envs, int) or num_envs < 1:
        raise ValueError("num_envs must be a positive integer")
    if not isinstance(randomize_camera, bool):
        raise ValueError("randomize_camera must be a boolean")
    calibration = load_g1_depth_calibration().resized(pattern.width, pattern.height)
    nominal_fov = calibration.fov_degrees[0]
    nominal_position = torch.tensor(pattern.position, device=device, dtype=torch.float32)
    nominal_quaternion = torch.tensor(pattern.rotation, device=device, dtype=torch.float32)
    if tuple(nominal_position.shape) != (3,) or not bool(torch.isfinite(nominal_position).all()):
        raise ValueError("Camera position must contain three finite values")
    if tuple(nominal_quaternion.shape) != (4,) or not bool(torch.isfinite(nominal_quaternion).all()) or not bool(nominal_quaternion.norm() > 0):
        raise ValueError("Camera quaternion must contain four finite values and have nonzero norm")
    nominal_quaternion = nominal_quaternion/nominal_quaternion.norm()
    if randomize_camera:
        translation, rotation = sample_rpl_mount_offsets(num_envs, device=device, generator=generator)
        fov = sample_rpl_horizontal_fov(nominal_fov, num_envs, device=device, generator=generator)
        nominal_euler = torch.stack(euler_xyz_from_quat(nominal_quaternion[None]), -1)
        euler = nominal_euler + torch.deg2rad(rotation)
        quaternions = quat_from_euler_xyz(euler[:, 0], euler[:, 1], euler[:, 2])
        # For a=cx+.5, b=W-a, F=atan(a/fx)+atan(b/fx), the positive
        # root below exactly preserves boundary HFOV with a non-centred K.
        angle = torch.deg2rad(fov)
        a, b = calibration.cx+.5, calibration.width-calibration.cx-.5
        numerator = calibration.width*angle.cos()
        fx = (numerator + (numerator.square()+4*a*b*angle.sin().square()).sqrt())/(2*angle.sin())
        fy = fx*(calibration.fy/calibration.fx)
    else:
        translation = torch.zeros(num_envs, 3, device=device)
        rotation = torch.zeros_like(translation)
        fov = torch.full((num_envs,), nominal_fov, device=device)
        quaternions = nominal_quaternion.expand(num_envs, 4).clone()
        fx = torch.full((num_envs,), calibration.fx, device=device)
        fy = torch.full((num_envs,), calibration.fy, device=device)
    positions = nominal_position + translation
    u, v = torch.meshgrid(torch.arange(pattern.width, device=device, dtype=torch.float32),
                         torch.arange(pattern.height, device=device, dtype=torch.float32), indexing="xy")
    x = (u.flatten()-calibration.cx)[None]/fx[:, None]
    y = (v.flatten()-calibration.cy)[None]/fy[:, None]
    cosines = (1+x.square()+y.square()).rsqrt()
    optical = torch.stack((torch.ones_like(x), -x, -y), -1)
    optical.mul_(cosines[..., None])
    directions = torch.bmm(optical, matrix_from_quat(quaternions).transpose(1, 2))
    return G1CameraGeometry(positions.contiguous(), quaternions.contiguous(), fov.contiguous(),
                            cosines.contiguous(), directions.contiguous(), translation, rotation)


class G1RPLRayCastSensor(RayCastSensor):
    """MuJoCo sensor with per-env persistent G1 rays and RPL geometric DR."""

    def __init__(self, cfg):
        super().__init__(cfg)
        self._camera_geometry = None

    def initialize(self, mj_model, model, data, device):
        super().initialize(mj_model, model, data, device)
        self._initialize_camera_geometry(data.nworld, device)

    def _initialize_camera_geometry(self, num_envs, device, *, generator=None):
        if self._camera_geometry is not None:
            if len(self.camera_positions) != num_envs or self.camera_positions.device != torch.device(device):
                raise ValueError("An initialized camera cannot change environment count or device")
            return
        if not isinstance(self.cfg.pattern, G1RPLDepthPatternCfg):
            raise ValueError("G1RPLRayCastSensor requires a calibrated G1 ray pattern")
        geometry = sample_g1_camera_geometry(self.cfg.pattern, num_envs,
            randomize_camera=self.cfg.randomize_camera, device=device, generator=generator)
        self._camera_geometry = geometry
        self.camera_positions = geometry.positions
        self.camera_quaternions = geometry.quaternions
        self.horizontal_fov_degrees = geometry.horizontal_fov_degrees
        self.optical_cosines = geometry.optical_cosines
        self._camera_local_directions = geometry.local_directions
        # Origins are shared within each camera. This expanded view retains
        # [env,ray,3] shape without allocating repeated translation storage.
        self._camera_local_offsets = geometry.positions[:, None].expand_as(geometry.local_directions)

    def prepare_rays(self):
        if self._camera_geometry is None:
            raise RuntimeError("Initialize calibrated camera geometry before preparing rays")
        assert self._data is not None and self._model is not None
        if self._frame_type == "body":
            frame_pos = self._data.xpos[:, self._frame_body_id]
            frame_mat = self._data.xmat[:, self._frame_body_id].view(-1, 3, 3)
        elif self._frame_type == "site":
            frame_pos = self._data.site_xpos[:, self._frame_site_id]
            frame_mat = self._data.site_xmat[:, self._frame_site_id].view(-1, 3, 3)
        else:
            frame_pos = self._data.geom_xpos[:, self._frame_geom_id]
            frame_mat = self._data.geom_xmat[:, self._frame_geom_id].view(-1, 3, 3)
        rot_mat = self._compute_alignment_rotation(frame_mat)
        assert self._ray_pnt is not None and self._ray_vec is not None
        # Write directly into persistent Warp buffers; avoid duplicating two
        # native-resolution world-ray arrays merely for pre-graph staging.
        pnt = wp.to_torch(self._ray_pnt).view(len(frame_pos), self._num_rays, 3)
        vec = wp.to_torch(self._ray_vec).view(len(frame_pos), self._num_rays, 3)
        world_origins = frame_pos + torch.einsum("bij,bj->bi", rot_mat, self.camera_positions)
        pnt.copy_(world_origins[:, None])
        torch.bmm(self._camera_local_directions, rot_mat.transpose(1, 2), out=vec)
        self._cached_world_origins = pnt
        self._cached_world_rays = vec
        self._cached_frame_pos = frame_pos
        self._cached_frame_mat = frame_mat


@dataclass
class G1RPLRayCastSensorCfg(RayCastSensorCfg):
    pattern: G1RPLDepthPatternCfg = field(default_factory=G1RPLDepthPatternCfg)
    max_distance: float = 2.5
    exclude_parent_body: bool = False
    include_geom_groups: tuple[int, ...] | None = (0, 2)
    randomize_camera: bool = True

    def build(self):
        return G1RPLRayCastSensor(self)
