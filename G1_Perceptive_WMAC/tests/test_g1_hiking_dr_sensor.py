"""G1 geometric randomization retains Hiking image geometry and near filtering."""
from types import SimpleNamespace
import unittest

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch
import warp as wp

from mjlab.sensor import ObjRef
from src.tasks.amp_perceptive.g1_hiking_dr_sensor import (
    G1HikingDRDepthPatternCfg,G1HikingDRRayCastSensor,G1HikingDRRayCastSensorCfg,
    sample_hiking_camera_geometry,
)
from src.tasks.amp_perceptive.sensors import (
    HikingDepthPatternCfg,HikingRayCastSensor,HIKING_CAMERA_MAX_DISTANCE,
)
from src.wmac_perceptive.camera_calibration import calibrated_optical_rays,load_g1_depth_calibration


def matrix(q):
    return Rotation.from_quat(np.asarray(q)[...,[1,2,3,0]]).as_matrix()


class G1HikingDRSensorTest(unittest.TestCase):
    def test_baseline_keeps_Hiking_size_mount_and_measured_K(self):
        p=G1HikingDRDepthPatternCfg()
        self.assertEqual((p.width,p.height,p.sensor_protocol),(64,36,'g1_hiking_dr_v1'))
        self.assertEqual(p.position,HikingDepthPatternCfg.position)
        self.assertEqual(p.rotation,HikingDepthPatternCfg.rotation)
        ray,cosine=p.optical_rays()
        expected,c=calibrated_optical_rays(width=64,height=36)
        torch.testing.assert_close(ray,expected,rtol=0,atol=0)
        torch.testing.assert_close(cosine,c,rtol=0,atol=0)
        positions,directions=p.generate_rays(None,'cpu')
        np.testing.assert_allclose(directions.numpy(),expected.numpy()@matrix(p.rotation).T,atol=2e-7)
        np.testing.assert_allclose(positions.numpy(),np.broadcast_to(p.position,(2304,3)),atol=1e-8)

    def test_disabled_randomization_preserves_measured_nominal_projection(self):
        p=G1HikingDRDepthPatternCfg()
        geometry=sample_hiking_camera_geometry(p,3,randomize_camera=False)
        offsets,rays=p.generate_rays(None,'cpu')
        torch.testing.assert_close(geometry.local_directions,rays[None].expand(3,-1,-1),atol=2e-7,rtol=2e-7)
        torch.testing.assert_close(geometry.positions,offsets[:1].expand(3,-1))
        torch.testing.assert_close(geometry.translation_offsets,torch.zeros(3,3))
        torch.testing.assert_close(geometry.rotation_offsets_degrees,torch.zeros(3,3))

    def test_configurable_errors_are_relative_to_Hiking_mount(self):
        p=G1HikingDRDepthPatternCfg(width=8,height=6)
        xyz=(.001,.002,.003);rpy=(.2,.4,.6)
        g=sample_hiking_camera_geometry(p,256,translation_half_range_m=xyz,
            rotation_half_range_deg=rpy,fov_half_range_degrees=.5,
            generator=torch.Generator().manual_seed(31))
        torch.testing.assert_close(g.positions,torch.tensor(p.position)[None]+g.translation_offsets)
        self.assertTrue(bool((g.translation_offsets.abs()<=torch.tensor(xyz)).all()))
        self.assertTrue(bool((g.rotation_offsets_degrees.abs()<=torch.tensor(rpy)).all()))
        self.assertTrue(bool((g.translation_offsets.abs().max(0).values>torch.tensor(xyz)*.95).all()))
        nominal=Rotation.from_quat(np.asarray(p.rotation)[[1,2,3,0]]).as_euler('xyz')
        expected=Rotation.from_euler('xyz',nominal[None]+np.deg2rad(g.rotation_offsets_degrees.numpy())).as_matrix()
        np.testing.assert_allclose(matrix(g.quaternions.numpy()),expected,atol=2e-7)

    def test_FOV_errors_preserve_noncentred_K_and_pixel_aspect(self):
        p=G1HikingDRDepthPatternCfg(width=16,height=9)
        g=sample_hiking_camera_geometry(p,64,fov_half_range_degrees=1.5,
            generator=torch.Generator().manual_seed(43))
        c=load_g1_depth_calibration().resized(p.width,p.height)
        optical=np.einsum('bnj,bjk->bnk',g.local_directions.numpy(),matrix(g.quaternions.numpy()))
        x=-optical[:,:,1]/optical[:,:,0];y=-optical[:,:,2]/optical[:,:,0]
        # Recover fx/fy independently from endpoint displacement; do not
        # assume the optical centre sits at half the image width/height.
        fx=(p.width-1)/(x[:,p.width-1]-x[:,0])
        fy=(p.height-1)/(y[:,-p.width]-y[:,0])
        np.testing.assert_allclose(-x[:,0]*fx,c.cx,atol=3e-6)
        np.testing.assert_allclose(-y[:,0]*fy,c.cy,atol=2e-6)
        np.testing.assert_allclose(fx/fy,c.fx/c.fy,atol=6e-7)
        boundary=np.rad2deg(np.arctan((c.cx+.5)/fx)+np.arctan((p.width-c.cx-.5)/fx))
        np.testing.assert_allclose(boundary,g.horizontal_fov_degrees.numpy(),atol=1.4e-5)
        self.assertTrue(bool((g.horizontal_fov_degrees>=c.fov_degrees[0]-1.5).all()))
        self.assertTrue(bool((g.horizontal_fov_degrees<=c.fov_degrees[0]+1.5).all()))
        np.testing.assert_allclose(g.optical_cosines.numpy(),optical[:,:,0],atol=2e-7)

    @staticmethod
    def make_sensor():
        model=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body name="torso"><geom type="sphere" size=".1"/></body></worldbody></mujoco>')
        data=SimpleNamespace(nworld=3,xpos=torch.zeros(3,model.nbody,3),
            xmat=torch.eye(3).expand(3,model.nbody,3,3).clone())
        cfg=G1HikingDRRayCastSensorCfg(name='depth',frame=ObjRef(type='body',name='torso'),
            translation_half_range_m=(.001,.002,.003),rotation_half_range_deg=(.2,.4,.6),
            fov_half_range_degrees=.5)
        sensor=cfg.build();sensor.initialize(model,SimpleNamespace(),data,'cpu')
        return sensor,model,data

    def test_initialization_is_fixed_through_reset_and_updates(self):
        sensor,_,_=self.make_sensor()
        self.assertIsInstance(sensor,G1HikingDRRayCastSensor)
        before=sensor._camera_local_directions.clone();pointer=before.data_ptr()
        actual_pointer=sensor._camera_local_directions.data_ptr()
        for ids in [None,torch.tensor([0,2]),slice(None)]:
            sensor.reset(ids);sensor.update(.02);sensor.prepare_rays()
            sensor._initialize_camera_geometry(3,'cpu',generator=torch.Generator().manual_seed(928))
            self.assertEqual(sensor._camera_local_directions.data_ptr(),actual_pointer)
            torch.testing.assert_close(sensor._camera_local_directions,before,rtol=0,atol=0)
        self.assertNotEqual(pointer,actual_pointer)
        self.assertEqual(sensor._camera_local_offsets.stride()[1],0)
        self.assertTrue(bool((sensor._camera_geometry.translation_offsets.abs()<=torch.tensor([.001,.002,.003])).all()))

    def test_world_transform_writes_persistent_native_Warp_arrays(self):
        sensor,model,data=self.make_sensor();body=model.body('torso').id
        data.xpos[:,body]=torch.tensor([[1.,2.,3.],[-2.,1.,0.],[0.,3.,4.]])
        rotations=Rotation.from_euler('xyz',[[.1,.2,.3],[.2,-.3,.4],[-.1,-.2,-.3]]).as_matrix()
        data.xmat[:,body]=torch.tensor(rotations,dtype=torch.float32)
        pp=wp.to_torch(sensor._ray_pnt).data_ptr();vp=wp.to_torch(sensor._ray_vec).data_ptr()
        sensor.prepare_rays()
        expected_pos=data.xpos[:,body].numpy()+np.einsum('bij,bj->bi',rotations,sensor.camera_positions.numpy())
        expected_ray=np.einsum('bij,bnj->bni',rotations,sensor._camera_local_directions.numpy())
        np.testing.assert_allclose(sensor._cached_world_origins.numpy(),np.broadcast_to(expected_pos[:,None],(3,2304,3)),atol=8e-7)
        np.testing.assert_allclose(sensor._cached_world_rays.numpy(),expected_ray,atol=2e-7)
        self.assertEqual(sensor._cached_world_origins.data_ptr(),pp)
        self.assertEqual(sensor._cached_world_rays.data_ptr(),vp)

    def test_sensor_retains_Hiking_near_filter_far_range_and_robot_geometry(self):
        sensor,_,_=self.make_sensor()
        self.assertEqual(sensor.cfg.max_distance,2*HIKING_CAMERA_MAX_DISTANCE)
        self.assertFalse(sensor.cfg.exclude_parent_body)
        self.assertEqual(sensor.cfg.include_geom_groups,(0,2))
        self.assertIs(type(sensor).raycast_kernel,HikingRayCastSensor.raycast_kernel)

    def test_invalid_uncertainty_configuration_is_rejected(self):
        p=G1HikingDRDepthPatternCfg(width=8,height=6)
        for kwargs in [{'translation_half_range_m':(-.01,0,0)},
                       {'rotation_half_range_deg':(1,2)},
                       {'fov_half_range_degrees':float('nan')},
                       {'fov_half_range_degrees':180.}]:
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):
                sample_hiking_camera_geometry(p,2,**kwargs)


if __name__=='__main__':unittest.main()
