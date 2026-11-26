"""Hiking camera geometry, measurement ordering and optional real Warp rays."""
from __future__ import annotations

import ast
import hashlib
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.tasks.amp_perceptive.sensors import (
    G1_HIKING_CALIBRATED_SENSOR_PROTOCOL,
    HIKING_SENSOR_PROTOCOL, LEGACY_SENSOR_PROTOCOL, HIKING_CAMERA_MAX_DISTANCE,
    HikingDepthPatternCfg, HikingRayCastSensorCfg, camera_protocol_metadata,
    depth_packet_from_ranges, read_depth_frame,
)
from src.tasks.amp_perceptive.env_cfg import g1_amp_perceptive_hiking_env_cfg
from src.wmac_perceptive.perception import PerceptionSpec
from src.wmac_perceptive.latent_perception import processed_depth_frame


class HikingCameraTest(unittest.TestCase):
    def test_fixed_camera_reuses_packet_until_next_sense(self):
        from mjlab.sensor import ObjRef
        from mjlab.sensor.raycast_sensor import RayCastSensor

        for protocol in (HIKING_SENSOR_PROTOCOL, G1_HIKING_CALIBRATED_SENSOR_PROTOCOL):
            with self.subTest(protocol=protocol):
                pattern = HikingDepthPatternCfg(width=3, height=2, sensor_protocol=protocol)
                cfg = HikingRayCastSensorCfg(name="hiking_depth", frame=ObjRef(type="body", name="camera"),
                                            pattern=pattern)
                sensor = cfg.build()
                distances = torch.tensor([[.1, .2, 4., -1., 1., 3.]])
                env = SimpleNamespace(scene={"hiking_depth": sensor}, common_step_counter=7)
                before = depth_packet_from_ranges(distances, pattern)
                with patch.object(sensor, "_compute_data", return_value=SimpleNamespace(distances=distances)), \
                     patch.object(RayCastSensor, "postprocess_rays"):
                    terminal = read_depth_frame(env)
                    self.assertIs(terminal, read_depth_frame(env))
                    # Warp overwrites the same range buffer during the second
                    # sense in a step. The old packet must remain available.
                    distances.copy_(torch.tensor([[.5, .6, 1., .8, 2., -1.]]))
                    sensor.postprocess_rays()
                    reset = read_depth_frame(env)
                    self.assertIs(reset, read_depth_frame(env))
                self.assertIsNot(terminal, reset)
                torch.testing.assert_close(terminal, before, rtol=0, atol=0)
                torch.testing.assert_close(reset, depth_packet_from_ranges(distances, pattern), rtol=0, atol=0)
                self.assertFalse(torch.equal(reset, terminal))

    def test_measured_intrinsics_preserve_hiking_raycast_and_depth_processing(self):
        from src.wmac_perceptive.camera_calibration import load_g1_depth_calibration

        protocol = G1_HIKING_CALIBRATED_SENSOR_PROTOCOL
        sensor = g1_amp_perceptive_hiking_env_cfg(sensor_protocol=protocol).scene.sensors[-1]
        nominal = g1_amp_perceptive_hiking_env_cfg(sensor_protocol=HIKING_SENSOR_PROTOCOL).scene.sensors[-1]
        self.assertIsInstance(sensor, HikingRayCastSensorCfg)
        self.assertEqual((sensor.include_geom_groups, sensor.max_distance, sensor.exclude_parent_body),
                         (nominal.include_geom_groups, nominal.max_distance, nominal.exclude_parent_body))
        self.assertEqual((sensor.pattern.position, sensor.pattern.rotation),
                         (nominal.pattern.position, nominal.pattern.rotation))
        source = load_g1_depth_calibration()
        u, v = np.meshgrid(np.arange(64), np.arange(36))
        source_u = (u + .5) * source.width / 64 - .5
        source_v = (v + .5) * source.height / 36 - .5
        expected = np.stack((np.ones_like(u), -(source_u-source.cx)/source.fx,
                             -(source_v-source.cy)/source.fy), axis=-1).reshape(-1, 3)
        expected /= np.linalg.norm(expected, axis=-1, keepdims=True)
        rays, cosines = sensor.pattern.optical_rays()
        np.testing.assert_allclose(rays.numpy(), expected, atol=1e-7, rtol=2e-7)
        np.testing.assert_allclose(cosines.numpy(), expected[:, 0], atol=1e-7, rtol=2e-7)
        ranges = torch.full((1, 36 * 64), 1.)
        ranges[0, :3] = torch.tensor([.1, -1., 4.])
        packet = depth_packet_from_ranges(ranges, sensor.pattern)
        self.assertEqual(packet[0, 0, 0, :2].tolist(), [HIKING_CAMERA_MAX_DISTANCE/2.5] * 2)
        self.assertGreater(float(packet[0, 0, 0, 2]), 1.)
        metadata = camera_protocol_metadata(protocol)
        baseline = camera_protocol_metadata(HIKING_SENSOR_PROTOCOL)
        for key in ("preprocess_order", "update_period_s", "projection", "minimum_distance_rule"):
            self.assertEqual(metadata[key], baseline[key])
        self.assertEqual(processed_depth_frame(packet, PerceptionSpec(include_valid=False)).shape, (1, 1, 18, 32))

    def test_pixel_centers_match_independent_intrinsic_matrix_projection(self):
        pattern=HikingDepthPatternCfg(sensor_protocol=HIKING_SENSOR_PROTOCOL)
        width,height=pattern.width,pattern.height
        fx=width/(2*math.tan(math.radians(pattern.fovx)/2))
        fy=height/(2*math.tan(math.radians(pattern.fovy)/2))
        intrinsic=np.array([[fx,0,width/2],[0,fy,height/2],[0,0,1.]])
        pixels=np.array([[x+.5,y+.5,1.] for y in range(height) for x in range(width)])
        optical=pixels@np.linalg.inv(intrinsic).T
        robot=optical[:,[2,0,1]]*np.array([1.,-1.,-1.])
        expected=robot/np.linalg.norm(robot,axis=1,keepdims=True)
        actual,cosines=pattern.optical_rays()
        np.testing.assert_allclose(actual.numpy(),expected,atol=1e-7,rtol=2e-7)
        np.testing.assert_allclose(cosines.numpy(),expected[:,0],atol=1e-7,rtol=2e-7)
        legacy=HikingDepthPatternCfg().optical_rays()[0]
        self.assertGreater(float((actual-legacy).abs().max()),.005)

    def test_pinned_isaaclab_pinhole_function_matches_local_rays(self):
        # Hiking README pins IsaacLab f73c331738. This optional check executes
        # only that independently downloaded pure torch projection function.
        path=Path(os.environ.get("HIKING_ISAACLAB_CAMERA_REFERENCE",
                                 "/tmp/hiking-isaaclab-camera-reference"))/"patterns.py"
        if not path.is_file():
            self.skipTest("Optional pinned IsaacLab source is not present")
        source=path.read_bytes()
        self.assertEqual(hashlib.sha256(source).hexdigest(),
                         "bf91f91b92d56b0beab13f6192fff051e8674fa83e1e516c377c886f27a9a26d")
        node=next(x for x in ast.parse(source).body
                  if isinstance(x,ast.FunctionDef) and x.name=="pinhole_camera_pattern")
        for argument in node.args.args:argument.annotation=None
        node.returns=None
        namespace={"torch":torch}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),
                     str(path),"exec"),namespace)
        pattern=HikingDepthPatternCfg(sensor_protocol=HIKING_SENSOR_PROTOCOL)
        k=torch.eye(3)[None]
        k[:,0,0]=pattern.width/(2*math.tan(math.radians(pattern.fovx)/2))
        k[:,1,1]=pattern.height/(2*math.tan(math.radians(pattern.fovy)/2))
        k[:,0,2]=pattern.width/2;k[:,1,2]=pattern.height/2
        starts,expected=namespace["pinhole_camera_pattern"](pattern,k,"cpu")
        torch.testing.assert_close(pattern.optical_rays()[0],expected[0],atol=1e-7,rtol=2e-7)
        self.assertEqual(float(starts.abs().sum()),0.)

    def test_range_filter_uses_ray_distance_and_keeps_far_depth_before_blur(self):
        pattern=HikingDepthPatternCfg(width=2,height=2,sensor_protocol=HIKING_SENSOR_PROTOCOL)
        ranges=torch.tensor([[.11,4.,-1.,.1]])
        packet=depth_packet_from_ranges(ranges,pattern)
        cosines=pattern.optical_rays()[1]
        self.assertLess(float(.11*cosines[0]),.1)  # Valid ray, not image-plane threshold.
        self.assertEqual(float(packet[0,1,0,0]),1.)
        self.assertGreater(float(packet[0,0,0,1]),1.)
        self.assertEqual(float(packet[0,0,1,0]),HIKING_CAMERA_MAX_DISTANCE/2.5)
        self.assertEqual(float(packet[0,0,1,1]),HIKING_CAMERA_MAX_DISTANCE/2.5)
        self.assertEqual(packet.dtype,torch.float32)
        self.assertTrue(bool(torch.isfinite(packet).all()))

    def test_hiking_blurs_before_final_range_clipping(self):
        raw=torch.zeros(1,2,36,64)
        raw[:,0]=.2;raw[:,1]=1.
        raw[:,0,25,30]=HIKING_CAMERA_MAX_DISTANCE/2.5
        # Independently use the pinned upstream torchvision GaussianBlur
        # convention (a normalized Gaussian, reflection padding).
        from torchvision.transforms import GaussianBlur
        reference=GaussianBlur(3,1)(raw[:,:1,18:,16:-16]*2.5).clamp(0,2.5)/2.5
        actual=processed_depth_frame(raw,PerceptionSpec(include_valid=False))
        torch.testing.assert_close(actual,reference,atol=2e-7,rtol=2e-7)
        premature=GaussianBlur(3,1)(raw[:,:1,18:,16:-16].clamp(0,1))
        self.assertGreater(float((actual-premature).abs().max()),.5)
        self.assertTrue(bool(torch.isfinite(actual).all()))
        self.assertLessEqual(float(actual.max()),1.)

    def test_new_sensor_includes_torso_visuals_and_legacy_keeps_old_protocol(self):
        current=g1_amp_perceptive_hiking_env_cfg(sensor_protocol=HIKING_SENSOR_PROTOCOL).scene.sensors[-1]
        legacy=g1_amp_perceptive_hiking_env_cfg().scene.sensors[-1]
        self.assertIsInstance(current,HikingRayCastSensorCfg)
        self.assertFalse(current.exclude_parent_body)
        self.assertEqual(current.max_distance,2e6)
        self.assertEqual(current.ray_alignment,"base")
        self.assertEqual(current.include_geom_groups,(0,2))
        self.assertTrue(legacy.exclude_parent_body)
        self.assertEqual(legacy.max_distance,4.)
        self.assertEqual(legacy.pattern.sensor_protocol,LEGACY_SENSOR_PROTOCOL)
        meta=camera_protocol_metadata(HIKING_SENSOR_PROTOCOL)
        self.assertEqual(meta["position"],list(current.pattern.position))
        self.assertEqual(meta["rotation_wxyz"],list(current.pattern.rotation))
        with self.assertRaises(ValueError):
            g1_amp_perceptive_hiking_env_cfg(sensor_protocol="unknown")

    @unittest.skipUnless(os.environ.get("WMAC_RUN_CUDA_TESTS")=="1","Opt-in real CUDA raycast")
    def test_cuda_min_range_ignores_near_geometry_and_sees_background(self):
        self._check_min_range_scene("cuda:0",(1.,))

    def test_cpu_min_range_does_not_mix_world_specific_background_geometry(self):
        # Overlapping scene coordinates intentionally make a missing world
        # root / bounds offset observable: the farther world must stay farther.
        self._check_min_range_scene("cpu",(1.,2.))

    def _check_min_range_scene(self,device,wall_centers):
        import mujoco
        import mujoco_warp as mjw
        import warp as wp
        from mjlab.sensor import ObjRef
        xml='''<mujoco><worldbody>
          <body name="camera"><geom type="box" pos=".06 0 0" size=".01 .1 .1" group="2"/></body>
          <geom name="background" type="box" pos="1 0 0" size=".1 1 1" group="0"/>
        </worldbody></mujoco>'''
        host=mujoco.MjModel.from_xml_string(xml)
        state=mujoco.MjData(host);mujoco.mj_forward(host,state)
        worlds=len(wall_centers)
        with wp.ScopedDevice(device):
            model=mjw.put_model(host)
            data=mjw.put_data(host,state,nworld=worlds,nconmax=1,njmax=1)
            poses=data.geom_xpos.numpy()
            poses[:,host.geom("background").id,0]=wall_centers
            data.geom_xpos.assign(poses)
            context=mjw.create_render_context(host,nworld=worlds,enabled_geom_groups=[0,2])
            mjw.refit_bvh(model,data,context)
            cfg=HikingRayCastSensorCfg(name="test",frame=ObjRef(type="body",name="camera"),
                pattern=HikingDepthPatternCfg(width=3,height=1,sensor_protocol=HIKING_SENSOR_PROTOCOL),
                max_distance=2e6,exclude_parent_body=False,include_geom_groups=(0,2))
            sensor=cfg.build()
            sensor.initialize(host,SimpleNamespace(struct=model),SimpleNamespace(struct=data,nworld=worlds),device)
            origins=np.tile(np.array([[[0,0,0],[0,.5,0],[0,2,0]]],dtype=np.float32),(worlds,1,1))
            directions=np.tile(np.array([[[1,0,0]]*3],dtype=np.float32),(worlds,1,1))
            sensor._ray_pnt.assign(origins)
            sensor._ray_vec.assign(directions)
            sensor.raycast_kernel(context)
            expected=[[center-.1,center-.1,-1] for center in wall_centers]
            np.testing.assert_allclose(sensor._ray_dist.numpy(),expected,atol=1e-6)
            # The ray did not move beyond the close geometry's entry surface.
            np.testing.assert_array_equal(sensor._ray_pnt.numpy(),origins)


if __name__=="__main__":unittest.main()
