"""Versioned Hiking camera geometry and depth measurement semantics.

The new camera follows InstinctLab's Hiking task at de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057.
Raw packets retain an auxiliary validity plane for diagnostics. Only depth is
encoded by the new policy; crop, blur and final 2.5 m clipping happen later.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cached_property
import math

import torch
import warp as wp
from mjlab.sensor import PinholeCameraPatternCfg, RayCastSensorCfg
from mjlab.sensor.raycast_sensor import RayCastSensor
from mjlab.utils.lab_api.math import quat_apply
from mujoco_warp._src.ray import _ray_geom_mesh_bvh
from mujoco_warp._src.types import vec6

from src.wmac_perceptive.camera_calibration import calibrated_optical_rays, load_g1_depth_calibration
from src.wmac_perceptive.config import (
    G1_HIKING_CALIBRATED_SENSOR_PROTOCOL, G1_HIKING_DR_SENSOR_PROTOCOL, G1_RPL_SENSOR_PROTOCOL,
    HIKING_FIXED_CAMERA_PROTOCOLS, HIKING_SENSOR_PROTOCOL, LEGACY_SENSOR_PROTOCOL,
)

HIKING_REFERENCE_COMMIT = "de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057"
HIKING_CAMERA_MAX_DISTANCE = 1e6
HIKING_CAMERA_MIN_DISTANCE = .1


def camera_protocol_metadata(protocol):
    """Numerical sensor settings, separate from task geometry and weights."""
    if protocol==G1_HIKING_DR_SENSOR_PROTOCOL:
        from src.wmac_perceptive.hiking_depth_randomization import HikingDepthRandomizationCfg
        from .g1_hiking_dr_sensor import G1HikingDRRayCastSensorCfg
        value=camera_protocol_metadata(HIKING_SENSOR_PROTOCOL)
        calibration=load_g1_depth_calibration()
        geometry=G1HikingDRRayCastSensorCfg
        value.update(protocol=protocol,calibration=asdict(calibration),
            ray_intrinsics=asdict(calibration.resized(64,36)),
            image_shape=[36,64],processed_image_shape=[18,32],capture_hz=50.,
            hardware_stream_hz=calibration.fps,delay_policy_steps=1,
            mounting_calibrated=False,normalization="clip(depth,0,2.5)/2.5",
            preprocess_order=["crop_ROI", "adapted_pixel_randomization", "gaussian_blur", "clip_0_2.5", "divide_2.5"],
            pixel_randomization=asdict(HikingDepthRandomizationCfg()),
            camera_randomization={"translation_half_range_m":list(geometry.translation_half_range_m),
                "rotation_half_range_deg":list(geometry.rotation_half_range_deg),
                "fov_half_range_degrees":geometry.fov_half_range_degrees,
                "centre":"Hiking_torso_local_mount_and_measured_G1_K",
                "sampling":"once_per_environment_at_sensor_initialization",
                "range_source":"RPL_starting_ranges_not_Hiking_defaults_or_measured_mount_uncertainty"},
            pixel_coordinates="RealSense_integer_pixel_centres_with_half_pixel_resize",
            stochastic_scope="one_draw_per_physical_frame_shared_by_Actor_and_WM",
            adaptation="Hiking_image_and_timing_baseline; only_RPL_randomization_ideas_adapted")
        return value
    if protocol==G1_RPL_SENSOR_PROTOCOL:
        from src.wmac_perceptive.rpl_depth import RPLDepthConfig,RPL_REFERENCE_COMMIT,HOLOSOMA_REFERENCE_COMMIT
        c=load_g1_depth_calibration()
        return dict(protocol=protocol,calibration=asdict(c),ray_intrinsics=asdict(c.resized(240,135)),
            frame="torso_link",position=list(HikingDepthPatternCfg.position),
            rotation_wxyz=list(HikingDepthPatternCfg.rotation),mounting_calibrated=False,
            projection="optical_axis_Z",hardware_stream_hz=c.fps,capture_hz=10.,
            delay_capture_ticks=2,synthetic_delay_seconds=.2,measured_end_to_end_latency=None,
            pixel_coordinates="RealSense_integer_pixel_centres_with_half_pixel_resize",
            image_shape=[135,240],processed_image_shape=[27,48],normalization="(depth-0.1)/1.9-0.5",
            preprocess_order=["relative_gaussian_noise","dropout","terrain_biased_border_occlusion",
                "clip_0.1_2","bicubic_antialias_resize_27x48","two_capture_tick_delay","centered_normalization"],
            resize_overshoot="preserved",randomization=asdict(RPLDepthConfig()),
            installation_randomization="sample_once_per_environment_at_initialization",
            rpl_commit=RPL_REFERENCE_COMMIT,holosoma_commit=HOLOSOMA_REFERENCE_COMMIT,
            adaptation="single_G1_body_camera; no external front/back rig; G1 30Hz software sampled at10Hz")
    if protocol != LEGACY_SENSOR_PROTOCOL and protocol not in HIKING_FIXED_CAMERA_PROTOCOLS:
        raise ValueError(f"Unsupported Hiking sensor protocol: {protocol}")
    official = protocol in HIKING_FIXED_CAMERA_PROTOCOLS
    pattern = HikingDepthPatternCfg(sensor_protocol=protocol)
    metadata = dict(protocol=protocol, reference_commit=HIKING_REFERENCE_COMMIT if official else None,
                image_shape=[pattern.height,pattern.width], fov_degrees=[pattern.fovx,pattern.fovy],
                frame="torso_link", position=list(pattern.position), rotation_wxyz=list(pattern.rotation),
                coordinate_convention="world_camera_x_forward_y_left_z_up",
                projection="distance_to_image_plane", include_geom_groups=[0,2],
                update_period_s=.02, raw_depth_scale_m=2.5,
                pixel_sampling="pixel_centers" if official else "fov_endpoints",
                alignment="full_body_pose", min_ray_distance=.1 if official else 0.,
                max_ray_distance=2*HIKING_CAMERA_MAX_DISTANCE if official else 4.,
                missing_depth_m=HIKING_CAMERA_MAX_DISTANCE if official else 2.5,
                minimum_distance_rule="per_geometry_hit_filter" if official else "image_plane_filter",
                preprocess_order=["crop", "gaussian_blur", "clip_0_2.5", "divide_2.5"] if official
                    else ["clip_and_fill", "divide_2.5", "crop", "gaussian_blur"],
                exclude_parent_body=not official,
                backend="MuJoCo Warp BVH over local terrain and robot visual geometry")
    if protocol == G1_HIKING_CALIBRATED_SENSOR_PROTOCOL:
        calibration = load_g1_depth_calibration()
        intrinsics = calibration.resized(pattern.width, pattern.height)
        metadata.update(
            calibration=asdict(calibration), ray_intrinsics=asdict(intrinsics),
            fov_degrees=list(intrinsics.fov_degrees),
            pixel_sampling="RealSense_integer_pixel_centres_with_half_pixel_resize",
            capture_hz=50., hardware_stream_hz=calibration.fps, delay_policy_steps=1,
            mounting_calibrated=False,
        )
    return metadata


@dataclass
class HikingDepthPatternCfg(PinholeCameraPatternCfg):
    width: int = 64
    height: int = 36
    fovy: float = 58.29
    fovx: float = 89.51
    position: tuple[float,float,float] = (.0487988662332928,.01,.4378029937970051)
    rotation: tuple[float,float,float,float] = (.9135367613482678,.004363309284746571,.4067366430758002,0.)
    sensor_protocol: str = LEGACY_SENSOR_PROTOCOL

    @cached_property
    def calibrated_intrinsics(self):
        return load_g1_depth_calibration().resized(self.width, self.height)

    def optical_rays(self,device="cpu"):
        if self.sensor_protocol == G1_HIKING_CALIBRATED_SENSOR_PROTOCOL:
            return calibrated_optical_rays(self.calibrated_intrinsics, device=device)
        if self.sensor_protocol == HIKING_SENSOR_PROTOCOL:
            # K^-1 [u+.5,v+.5,1], principal point (W/2,H/2).
            u=(torch.arange(self.width,device=device)+.5)*(2/self.width)-1
            v=(torch.arange(self.height,device=device)+.5)*(2/self.height)-1
        elif self.sensor_protocol == LEGACY_SENSOR_PROTOCOL:
            u=torch.linspace(-1,1,self.width,device=device)
            v=torch.linspace(-1,1,self.height,device=device)
        else:
            raise ValueError(f"Unsupported Hiking sensor protocol: {self.sensor_protocol}")
        x,y=torch.meshgrid(u,v,indexing="xy")
        # Hiking's world camera convention: +X forward, +Y left, +Z up.
        raw=torch.stack((torch.ones_like(x),-x*math.tan(math.radians(self.fovx)/2),
                         -y*math.tan(math.radians(self.fovy)/2)),-1).reshape(-1,3)
        return raw/raw.norm(dim=-1,keepdim=True),raw.norm(dim=-1).reciprocal()

    def generate_rays(self,mj_model,device):
        del mj_model
        directions,_=self.optical_rays(device)
        quaternion=torch.tensor(self.rotation,device=device)
        quaternion=quaternion/quaternion.norm()
        directions=quat_apply(quaternion.expand(len(directions),4),directions)
        offsets=torch.tensor(self.position,device=device).expand(len(directions),3).clone()
        return offsets,directions


def depth_packet_from_ranges(ranges,pattern: HikingDepthPatternCfg,maximum_depth=2.5,minimum_depth=.1,*,cosines=None):
    """Range→camera image-plane depth, [N,2,H,W] (scaled depth,valid).

    Legacy packets fill invalid pixels with depth1. Hiking packets preserve
    the camera's larger measurement range until the later blur/normalization.
    The auxiliary validity plane is independent of either image's fill value.
    """
    if cosines is None:
        _,cosines=pattern.optical_rays(ranges.device)
    if pattern.sensor_protocol in HIKING_FIXED_CAMERA_PROTOCOLS:
        # Hiking's camera clips at its 1e6 m sensor range, not the downstream
        # normalization range. Far/missing values must survive the Gaussian
        # filter; clipping first would change pixels near depth discontinuities.
        measured = ranges.float()*cosines.float()
        hit = torch.isfinite(ranges)&(ranges>HIKING_CAMERA_MIN_DISTANCE)
        measured = torch.where(hit, measured.clamp(max=HIKING_CAMERA_MAX_DISTANCE),
                               torch.full_like(measured,HIKING_CAMERA_MAX_DISTANCE))
        valid = hit&(measured<=maximum_depth)
        return torch.stack((measured/maximum_depth,valid.to(measured.dtype)),1).reshape(
            -1,2,pattern.height,pattern.width)
    depth=ranges*cosines.to(ranges)
    valid=torch.isfinite(depth)&(ranges>=0)&(depth>=minimum_depth)&(depth<=maximum_depth)
    depth=torch.where(valid,depth.clamp(minimum_depth,maximum_depth)/maximum_depth,torch.ones_like(depth))
    return torch.stack((depth,valid.to(depth.dtype)),1).reshape(-1,2,pattern.height,pattern.width)


@wp.kernel(enable_backward=False)
def _replace_near_hits(
    ngeom: int,
    body_weldid: wp.array[int], geom_type: wp.array[int], geom_bodyid: wp.array[int],
    geom_dataid: wp.array2d[int], geom_matid: wp.array2d[int], geom_group: wp.array[int],
    geom_size: wp.array2d[wp.vec3], geom_rgba: wp.array2d[wp.vec4], mat_rgba: wp.array2d[wp.vec4],
    geom_xpos: wp.array2d[wp.vec3], geom_xmat: wp.array2d[wp.mat33],
    origins: wp.array2d[wp.vec3], directions: wp.array2d[wp.vec3], groups: vec6,
    scene_bvh: wp.uint64, roots: wp.array[int], geom_ids: wp.array[int],
    mesh_bvh: wp.array[wp.uint64], hfield_bvh: wp.array[wp.uint64],
    near: float, far: float,
    distances: wp.array2d[float], ids: wp.array2d[int], normals: wp.array2d[wp.vec3],
):
    """Discard each geometry's nearest hit inside the Hiking near range.

    A global nearest-hit filter would hide farther geometry behind a near
    robot link. A shifted ray could instead hit the same link's back surface.
    Querying each candidate at the original origin applies the reference near
    rejection to local geometry. The Isaac target meshes and local MuJoCo
    geometry need not have identical aggregation. Ordinary rays keep the fast
    backend result.
    """
    world, ray = wp.tid()
    old = distances[world,ray]
    if old >= 0.0 and old <= near:
        closest = far
        selected = int(-1)
        normal = wp.vec3()
        query = wp.bvh_query_ray(scene_bvh,origins[world,ray],directions[world,ray],roots[world])
        bound = int(0)
        while wp.bvh_query_next(query,bound,closest):
            geom = geom_ids[bound-world*ngeom]
            distance, candidate_normal = _ray_geom_mesh_bvh(
                body_weldid,geom_type,geom_bodyid,geom_dataid,geom_matid,geom_group,
                geom_size,geom_rgba,mat_rgba,geom_xpos,geom_xmat,world,
                origins[world,ray],directions[world,ray],groups,True,-1,geom,
                mesh_bvh,hfield_bvh,closest)
            if distance > near and distance < closest:
                closest=distance
                selected=geom
                normal=candidate_normal
        distances[world,ray]=closest if selected>=0 else -1.0
        ids[world,ray]=selected
        normals[world,ray]=normal


class HikingRayCastSensor(RayCastSensor):
    """Standard MuJoCo sensor plus Hiking's per-mesh minimum ray distance."""

    @cached_property
    def _depth_cosines(self) -> torch.Tensor:
        return self.cfg.pattern.optical_rays(self.data.distances.device)[1]

    @cached_property
    def depth_packet(self) -> torch.Tensor:
        """Read-only depth and validity for the most recent sensed result."""
        return depth_packet_from_ranges(self.data.distances, self.cfg.pattern, cosines=self._depth_cosines)

    def postprocess_rays(self) -> None:
        super().postprocess_rays()
        # Called outside the captured raycast on every sense, including both
        # terminal and post-reset measurements at the same control step.
        self.__dict__.pop("depth_packet", None)

    def raycast_kernel(self, rc):
        super().raycast_kernel(rc)
        model,data=self._model.struct,self._data.struct
        wp.launch(_replace_near_hits,dim=(data.nworld,self._num_rays),inputs=[
            rc.bvh_ngeom,model.body_weldid,model.geom_type,model.geom_bodyid,
            model.geom_dataid,model.geom_matid,model.geom_group,model.geom_size,
            model.geom_rgba,model.mat_rgba,data.geom_xpos,data.geom_xmat,
            self._ray_pnt,self._ray_vec,self._geomgroup,rc.bvh_id,rc.group_root,
            rc.enabled_geom_ids,rc.mesh_bvh_id,rc.hfield_bvh_id,
            HIKING_CAMERA_MIN_DISTANCE,self.cfg.max_distance,
            self._ray_dist,self._ray_geomid,self._ray_normal],device=self._wp_device)


@dataclass
class HikingRayCastSensorCfg(RayCastSensorCfg):
    def build(self):
        return HikingRayCastSensor(self)


def _hiking_stepping_stones_mask(terrain, reference):
    """Reuse static family constants while reading current curriculum types."""
    cfg = terrain.cfg.terrain_generator
    keys = tuple(cfg.sub_terrains)
    proportions = tuple(cfg.sub_terrains[name].proportion for name in keys)
    signature = (id(cfg), keys, proportions, cfg.num_cols, reference.device, reference.dtype)
    cached = getattr(terrain, "_hiking_depth_family_constants", None)
    if cached is None or cached[0] != signature:
        values = reference.new_tensor(proportions)
        thresholds = values.cumsum(0) / values.sum()
        flags = torch.tensor(["stepping" in name for name in keys], device=reference.device)
        cached = (signature, thresholds, flags)
        terrain._hiking_depth_family_constants = cached
    _, thresholds, flags = cached
    fraction = (terrain.terrain_types.to(reference.dtype) + .5) / cfg.num_cols
    family = torch.searchsorted(thresholds, fraction).clamp_max(len(keys) - 1)
    return flags[family]


def read_depth_frame(env):
    """Read-only latest physical packet, shared by consumers of the same sense."""
    sensor=env.scene["hiking_depth"]
    if getattr(sensor.cfg.pattern,"sensor_protocol",None)==G1_HIKING_DR_SENSOR_PROTOCOL:
        from src.wmac_perceptive.hiking_depth_randomization import HikingDepthCapture
        if not hasattr(env,"_hiking_depth_capture"):
            env._hiking_depth_capture=HikingDepthCapture(env.num_envs,env.device,
                enabled_randomization=getattr(env.cfg,"_hiking_depth_randomization",True))
        capture=env._hiking_depth_capture
        control_step=int(env.common_step_counter)
        if getattr(env.cfg,"_hiking_training_version",1)>=2:
            cached=capture.cached_packet(control_step=control_step)
            if cached is not None:
                return cached
        ranges=sensor.data.distances
        measured=ranges.float()*sensor.optical_cosines
        hit=torch.isfinite(ranges)&(ranges>HIKING_CAMERA_MIN_DISTANCE)
        measured=torch.where(hit,measured.clamp(max=HIKING_CAMERA_MAX_DISTANCE),
            torch.full_like(measured,HIKING_CAMERA_MAX_DISTANCE))
        valid=hit&(measured<=2.5)
        packet=torch.stack((measured/2.5,valid.to(measured.dtype)),1).reshape(env.num_envs,2,36,64)
        stepping=_hiking_stepping_stones_mask(env.scene.terrain,packet)
        return capture.step(packet,control_step=control_step,
            terrain_is_stepping_stones=stepping)
    if getattr(sensor.cfg.pattern,"sensor_protocol",None)==G1_RPL_SENSOR_PROTOCOL:
        from src.wmac_perceptive.rpl_camera_stream import RPLCameraStream
        if not hasattr(env,"_rpl_camera_stream"):
            env._rpl_camera_stream=RPLCameraStream(env.num_envs,env.device,
                enabled_randomization=getattr(env.cfg,"_rpl_depth_randomization",True))
        stream=env._rpl_camera_stream
        step=int(env.common_step_counter)
        if (stream.last_control_step==step or
                (stream.last_capture_step is not None and step-stream.last_capture_step<stream.capture_interval)):
            return stream.step(None,control_step=step,terrain_is_stepping_stones=None)
        ranges=sensor.data.distances
        depth=torch.where(torch.isfinite(ranges)&(ranges>=0),ranges*sensor.optical_cosines,
                          torch.full_like(ranges,2.5)).reshape(env.num_envs,1,135,240)
        terrain=env.scene.terrain
        keys=list(terrain.cfg.terrain_generator.sub_terrains)
        proportions=torch.tensor([terrain.cfg.terrain_generator.sub_terrains[k].proportion for k in keys],
                                  device=env.device)
        thresholds=proportions.cumsum(0)/proportions.sum()
        fraction=(terrain.terrain_types.to(depth.dtype)+.5)/terrain.cfg.terrain_generator.num_cols
        family=torch.searchsorted(thresholds,fraction).clamp_max(len(keys)-1)
        stepping=torch.tensor(["stepping" in k for k in keys],device=env.device)[family]
        return stream.step(depth,control_step=step,
            terrain_is_stepping_stones=stepping)
    if isinstance(sensor, HikingRayCastSensor):
        return sensor.depth_packet
    return depth_packet_from_ranges(sensor.data.distances,sensor.cfg.pattern)


def depth_observation(env):
    return read_depth_frame(env).flatten(1)
