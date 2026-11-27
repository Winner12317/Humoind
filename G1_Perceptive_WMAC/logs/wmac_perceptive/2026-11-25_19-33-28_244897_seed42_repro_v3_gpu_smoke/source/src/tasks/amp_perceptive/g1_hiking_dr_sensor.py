"""Measured G1 intrinsics and geometric DR on the Hiking camera baseline.

The nominal body frame, mounting transform, 64x36 ray grid, near-hit rejection
and far range are Hiking's. Only bounded calibration/mount uncertainty is
added around that nominal camera. The initial default error magnitudes are
borrowed from RPL, not values enabled by the original Hiking G1 task; they
are explicit config parameters rather than a replacement external-camera rig.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import torch
from mjlab.sensor import RayCastSensorCfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat,matrix_from_quat,quat_from_euler_xyz

from src.wmac_perceptive.camera_calibration import load_g1_depth_calibration
from .g1_rpl_sensor import G1CameraGeometry,G1RPLDepthPatternCfg,G1RPLRayCastSensor
from .sensors import HikingDepthPatternCfg,HikingRayCastSensor,HIKING_CAMERA_MAX_DISTANCE


@dataclass
class G1HikingDRDepthPatternCfg(G1RPLDepthPatternCfg):
    width: int = 64
    height: int = 36
    position: tuple[float,float,float] = HikingDepthPatternCfg.position
    rotation: tuple[float,float,float,float] = HikingDepthPatternCfg.rotation
    sensor_protocol: str = "g1_hiking_dr_v1"


def _half_range(values,name):
    if (not isinstance(values,(tuple,list)) or len(values)!=3
            or any(isinstance(v,bool) or not isinstance(v,(int,float))
                   or not math.isfinite(v) or v<0 for v in values)):
        raise ValueError(f"{name} must contain three finite nonnegative half ranges")
    return values


def sample_hiking_camera_geometry(pattern,num_envs,*,randomize_camera=True,
        translation_half_range_m=(.025,.025,.025),rotation_half_range_deg=(2.5,3.,2.5),
        fov_half_range_degrees=5.,device="cpu",generator=None):
    """Sample fixed installation errors in the Hiking mounting/body frame.

    Translation is added to Hiking's torso-local mounting position; Euler
    offsets are added to its nominal orientation. Preserve the measured K's
    non-centred principal point and fx/fy ratio while varying boundary HFOV.
    Call once at initialization, never once per frame or automatically reset.
    """
    if isinstance(num_envs,bool) or not isinstance(num_envs,int) or num_envs<1:
        raise ValueError("num_envs must be a positive integer")
    if not isinstance(randomize_camera,bool):
        raise ValueError("randomize_camera must be boolean")
    _half_range(translation_half_range_m,"translation_half_range_m")
    _half_range(rotation_half_range_deg,"rotation_half_range_deg")
    c=load_g1_depth_calibration().resized(pattern.width,pattern.height)
    hfov=c.fov_degrees[0]
    if (isinstance(fov_half_range_degrees,bool) or not isinstance(fov_half_range_degrees,(int,float))
            or not math.isfinite(fov_half_range_degrees) or fov_half_range_degrees<0
            or hfov-fov_half_range_degrees<=0 or hfov+fov_half_range_degrees>=180):
        raise ValueError("FOV range must remain finite and inside (0,180) degrees")
    position=torch.tensor(pattern.position,device=device,dtype=torch.float32)
    quaternion=torch.tensor(pattern.rotation,device=device,dtype=torch.float32)
    if (position.shape!=(3,) or quaternion.shape!=(4,)
            or not bool(torch.isfinite(position).all()) or not bool(torch.isfinite(quaternion).all())
            or not bool(quaternion.norm()>0)):
        raise ValueError("Nominal Hiking camera transform must be finite with a nonzero quaternion")
    quaternion=quaternion/quaternion.norm()
    if randomize_camera:
        translation=(2*torch.rand(num_envs,3,device=device,generator=generator)-1)*position.new_tensor(translation_half_range_m)
        rotation=(2*torch.rand(num_envs,3,device=device,generator=generator)-1)*position.new_tensor(rotation_half_range_deg)
        fov=hfov+(2*torch.rand(num_envs,device=device,generator=generator)-1)*fov_half_range_degrees
        nominal_euler=torch.stack(euler_xyz_from_quat(quaternion[None]),-1)
        angles=nominal_euler+torch.deg2rad(rotation)
        quaternions=quat_from_euler_xyz(angles[:,0],angles[:,1],angles[:,2])
        angle=torch.deg2rad(fov)
        left,right=c.cx+.5,pattern.width-c.cx-.5
        numerator=pattern.width*angle.cos()
        fx=(numerator+(numerator.square()+4*left*right*angle.sin().square()).sqrt())/(2*angle.sin())
        fy=fx*(c.fy/c.fx)
    else:
        translation=torch.zeros(num_envs,3,device=device)
        rotation=torch.zeros_like(translation)
        fov=torch.full((num_envs,),hfov,device=device)
        quaternions=quaternion.expand(num_envs,4).clone()
        fx=torch.full((num_envs,),c.fx,device=device)
        fy=torch.full((num_envs,),c.fy,device=device)
    positions=position+translation
    u,v=torch.meshgrid(torch.arange(pattern.width,device=device,dtype=torch.float32),
                       torch.arange(pattern.height,device=device,dtype=torch.float32),indexing="xy")
    x=(u.flatten()-c.cx)[None]/fx[:,None]
    y=(v.flatten()-c.cy)[None]/fy[:,None]
    cosine=(1+x.square()+y.square()).rsqrt()
    optical=torch.stack((torch.ones_like(x),-x,-y),-1)*cosine[...,None]
    directions=torch.bmm(optical,matrix_from_quat(quaternions).transpose(1,2))
    return G1CameraGeometry(positions.contiguous(),quaternions.contiguous(),fov.contiguous(),
        cosine.contiguous(),directions.contiguous(),translation,rotation)


class G1HikingDRRayCastSensor(G1RPLRayCastSensor,HikingRayCastSensor):
    """Persistent per-env measured rays plus Hiking per-geometry near filtering."""

    def _initialize_camera_geometry(self,num_envs,device,*,generator=None):
        if self._camera_geometry is not None:
            if len(self.camera_positions)!=num_envs or self.camera_positions.device!=torch.device(device):
                raise ValueError("An initialized camera cannot change environment count or device")
            return
        if not isinstance(self.cfg.pattern,G1HikingDRDepthPatternCfg):
            raise ValueError("Hiking DR camera requires the 64x36 measured G1 pattern")
        geometry=sample_hiking_camera_geometry(self.cfg.pattern,num_envs,
            randomize_camera=self.cfg.randomize_camera,
            translation_half_range_m=self.cfg.translation_half_range_m,
            rotation_half_range_deg=self.cfg.rotation_half_range_deg,
            fov_half_range_degrees=self.cfg.fov_half_range_degrees,
            device=device,generator=generator)
        self._camera_geometry=geometry
        self.camera_positions=geometry.positions
        self.camera_quaternions=geometry.quaternions
        self.horizontal_fov_degrees=geometry.horizontal_fov_degrees
        self.optical_cosines=geometry.optical_cosines
        self._camera_local_directions=geometry.local_directions
        self._camera_local_offsets=geometry.positions[:,None].expand_as(geometry.local_directions)


@dataclass
class G1HikingDRRayCastSensorCfg(RayCastSensorCfg):
    pattern: G1HikingDRDepthPatternCfg = field(default_factory=G1HikingDRDepthPatternCfg)
    max_distance: float = 2*HIKING_CAMERA_MAX_DISTANCE
    exclude_parent_body: bool = False
    include_geom_groups: tuple[int,...] | None = (0,2)
    randomize_camera: bool = True
    # RPL-derived initial uncertainty ranges, adapted around the Hiking mount;
    # these are neither calibrated mounting errors nor Hiking source defaults.
    translation_half_range_m: tuple[float,float,float] = (.025,.025,.025)
    rotation_half_range_deg: tuple[float,float,float] = (2.5,3.,2.5)
    fov_half_range_degrees: float = 5.

    def build(self):
        return G1HikingDRRayCastSensor(self)
