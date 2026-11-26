"""Measured G1 projection and init-only RPL camera geometry randomization."""
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import warp as wp

from mjlab.sensor import ObjRef
from src.tasks.amp_perceptive.g1_rpl_sensor import (
    G1RPLDepthPatternCfg, G1RPLRayCastSensor, G1RPLRayCastSensorCfg,
    sample_g1_camera_geometry,
)
from src.tasks.amp_perceptive.sensors import HikingDepthPatternCfg
from src.wmac_perceptive.camera_calibration import calibrated_optical_rays, load_g1_depth_calibration


def _rotation_matrix(quaternion):
    q = np.asarray(quaternion)
    return Rotation.from_quat(q[..., [1, 2, 3, 0]]).as_matrix()


class G1RPLSensorTest(unittest.TestCase):
    def test_nominal_pattern_uses_measured_depth_K_and_existing_G1_mount(self):
        pattern = G1RPLDepthPatternCfg()
        self.assertEqual((pattern.width, pattern.height), (240, 135))
        self.assertEqual(pattern.position, HikingDepthPatternCfg.position)
        self.assertEqual(pattern.rotation, HikingDepthPatternCfg.rotation)
        rays, cosine = pattern.optical_rays()
        expected, expected_cosine = calibrated_optical_rays(width=240, height=135)
        torch.testing.assert_close(rays, expected)
        torch.testing.assert_close(cosine, expected_cosine)
        offsets, directions = pattern.generate_rays(None, "cpu")
        expected_mount = expected.numpy() @ _rotation_matrix(pattern.rotation).T
        np.testing.assert_allclose(directions.numpy(), expected_mount, atol=2e-7)
        np.testing.assert_allclose(offsets.numpy(), np.tile(pattern.position, (240*135, 1)), atol=1e-8)

    def test_disabled_randomization_preserves_nominal_rays_for_every_env(self):
        pattern = G1RPLDepthPatternCfg(width=16, height=9)
        actual = sample_g1_camera_geometry(pattern, 5, randomize_camera=False)
        offsets, directions = pattern.generate_rays(None, "cpu")
        torch.testing.assert_close(actual.local_directions, directions[None].expand(5, -1, -1), atol=2e-7, rtol=2e-7)
        torch.testing.assert_close(actual.positions, offsets[0:1].expand(5, -1))
        torch.testing.assert_close(actual.translation_offsets, torch.zeros(5, 3))
        torch.testing.assert_close(actual.rotation_offsets_degrees, torch.zeros(5, 3))
        torch.testing.assert_close(actual.optical_cosines, pattern.optical_rays()[1][None].expand(5, -1))

    def test_pose_randomization_uses_source_ranges_and_additive_Euler_angles(self):
        pattern = G1RPLDepthPatternCfg(width=8, height=6)
        actual = sample_g1_camera_geometry(pattern, 128, generator=torch.Generator().manual_seed(123))
        nominal_euler = Rotation.from_quat(np.asarray(pattern.rotation)[[1, 2, 3, 0]]).as_euler("xyz")
        expected_angles = nominal_euler[None] + np.deg2rad(actual.rotation_offsets_degrees.numpy())
        expected_rotation = Rotation.from_euler("xyz", expected_angles).as_matrix()
        np.testing.assert_allclose(_rotation_matrix(actual.quaternions.numpy()), expected_rotation, atol=2e-7)
        np.testing.assert_allclose(actual.positions.numpy(), np.asarray(pattern.position)[None]+actual.translation_offsets.numpy(), atol=2e-8)
        self.assertLessEqual(float(actual.translation_offsets.abs().max()), .025)
        limits = torch.tensor([2.5, 3., 2.5])
        self.assertTrue(bool((actual.rotation_offsets_degrees.abs() <= limits).all()))
        self.assertTrue(bool((actual.rotation_offsets_degrees.max(0).values > limits*.9).all()))
        self.assertTrue(bool((actual.rotation_offsets_degrees.min(0).values < -limits*.9).all()))

    def test_random_fov_keeps_measured_principal_point_and_pixel_aspect(self):
        pattern = G1RPLDepthPatternCfg(width=16, height=9)
        actual = sample_g1_camera_geometry(pattern, 64, generator=torch.Generator().manual_seed(7))
        calibration = load_g1_depth_calibration().resized(pattern.width, pattern.height)
        # Independently undo the mount with scipy, recover the focal lengths
        # from pixel0, then verify both image boundary angles and principal K.
        camera = np.einsum("bnj,bjk->bnk", actual.local_directions.numpy(), _rotation_matrix(actual.quaternions.numpy()))
        normalized_x = -camera[:, :, 1]/camera[:, :, 0]
        normalized_y = -camera[:, :, 2]/camera[:, :, 0]
        fx = -calibration.cx/normalized_x[:, 0]
        fy = -calibration.cy/normalized_y[:, 0]
        cx = 0-normalized_x[:, 0]*fx
        cy = 0-normalized_y[:, 0]*fy
        np.testing.assert_allclose(cx, calibration.cx, atol=1e-8)
        np.testing.assert_allclose(cy, calibration.cy, atol=1e-8)
        np.testing.assert_allclose(fx/fy, calibration.fx/calibration.fy, atol=4e-7)
        boundary_fov = np.rad2deg(np.arctan((cx+.5)/fx)+np.arctan((pattern.width-cx-.5)/fx))
        np.testing.assert_allclose(boundary_fov, actual.horizontal_fov_degrees.numpy(), atol=1.5e-5)
        nominal = calibration.fov_degrees[0]
        self.assertTrue(bool((actual.horizontal_fov_degrees >= nominal-5).all()))
        self.assertTrue(bool((actual.horizontal_fov_degrees <= nominal+5).all()))
        np.testing.assert_allclose(actual.optical_cosines.numpy(), camera[:, :, 0], atol=2e-7)

    def test_seeded_geometry_is_reproducible(self):
        pattern = G1RPLDepthPatternCfg(width=8, height=6)
        a = sample_g1_camera_geometry(pattern, 7, generator=torch.Generator().manual_seed(50))
        b = sample_g1_camera_geometry(pattern, 7, generator=torch.Generator().manual_seed(50))
        for field in ("positions", "quaternions", "local_directions", "optical_cosines", "horizontal_fov_degrees"):
            torch.testing.assert_close(getattr(a, field), getattr(b, field), rtol=0, atol=0)

    @staticmethod
    def make_sensor(randomize=True, num_envs=3):
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="torso"><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
        data = SimpleNamespace(nworld=num_envs, xpos=torch.zeros(num_envs, model.nbody, 3),
                               xmat=torch.eye(3).expand(num_envs, model.nbody, 3, 3).clone())
        cfg = G1RPLRayCastSensorCfg(name="depth", frame=ObjRef(type="body", name="torso"),
            pattern=G1RPLDepthPatternCfg(width=8, height=6), randomize_camera=randomize)
        sensor = cfg.build()
        sensor.initialize(model, SimpleNamespace(), data, "cpu")
        return sensor, model, data

    def test_sensor_init_geometry_is_persistent_through_reset_update_and_prepare(self):
        sensor, _, _ = self.make_sensor()
        self.assertIsInstance(sensor, G1RPLRayCastSensor)
        before = sensor._camera_local_directions.clone()
        pointer = sensor._camera_local_directions.data_ptr()
        for ids in (None, torch.tensor([0, 2]), slice(None)):
            sensor.reset(ids)
            sensor.update(.02)
            sensor.prepare_rays()
            sensor._initialize_camera_geometry(3, "cpu", generator=torch.Generator().manual_seed(9))
            self.assertEqual(sensor._camera_local_directions.data_ptr(), pointer)
            torch.testing.assert_close(sensor._camera_local_directions, before, rtol=0, atol=0)
        self.assertEqual(sensor._camera_local_offsets.stride()[1], 0)

    def test_prepare_rays_uses_each_env_pose_and_fixed_Warp_buffers(self):
        sensor, model, data = self.make_sensor()
        torso = model.body("torso").id
        positions = torch.tensor([[1., 2., 3.], [-4., 2., 0.], [0., 0., 9.]])
        rotations = Rotation.from_euler("xyz", [[.1, .2, .3], [-.5, .1, .2], [0., 0., np.pi/2]]).as_matrix()
        data.xpos[:, torso] = positions
        data.xmat[:, torso] = torch.tensor(rotations, dtype=torch.float32)
        pnt_pointer = wp.to_torch(sensor._ray_pnt).data_ptr()
        vec_pointer = wp.to_torch(sensor._ray_vec).data_ptr()
        sensor.prepare_rays()
        expected_origins = positions.numpy()+np.einsum("bij,bj->bi", rotations, sensor.camera_positions.numpy())
        expected_rays = np.einsum("bij,bnj->bni", rotations, sensor._camera_local_directions.numpy())
        np.testing.assert_allclose(sensor._cached_world_origins.numpy(), np.broadcast_to(expected_origins[:, None], (3, 48, 3)), atol=8e-7)
        np.testing.assert_allclose(sensor._cached_world_rays.numpy(), expected_rays, atol=2e-7)
        self.assertEqual(sensor._cached_world_origins.data_ptr(), pnt_pointer)
        self.assertEqual(sensor._cached_world_rays.data_ptr(), vec_pointer)
        sensor.prepare_rays()
        self.assertEqual(sensor._cached_world_origins.data_ptr(), pnt_pointer)
        self.assertEqual(sensor._cached_world_rays.data_ptr(), vec_pointer)

    def test_ray_cast_defaults_keep_G1_mount_and_RPL_far_range_without_near_filter(self):
        sensor, _, _ = self.make_sensor(randomize=False)
        self.assertEqual(sensor.cfg.max_distance, 2.5)
        self.assertFalse(sensor.cfg.exclude_parent_body)
        self.assertEqual(sensor.cfg.include_geom_groups, (0, 2))
        # Only the standard BVH caster is inherited; Hiking's near-hit
        # discard/recast kernel must not run for RPL's clip-after-render path.
        from mjlab.sensor.raycast_sensor import RayCastSensor
        self.assertIs(type(sensor).raycast_kernel, RayCastSensor.raycast_kernel)


if __name__ == "__main__":
    unittest.main()
