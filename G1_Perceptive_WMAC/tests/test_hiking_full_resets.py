"""Changing the free-root convention must preserve Hiking's torso reset."""
from __future__ import annotations

import unittest

import mujoco
import numpy as np
import torch

from src.wmac.state import quat_apply,quat_inv
from src.tasks.amp_perceptive.hiking_resets import torso_state_to_pelvis,sample_hiking_reset_state
from test_wmac_state_fast import make_adapter


class HikingFullResetTest(unittest.TestCase):
    def test_inverse_fk_matches_torso_pose_and_com_twist_in_mujoco(self):
        adapter,model=make_adapter();count=5
        torch.manual_seed(318)
        pose=torch.randn(count,7,dtype=torch.float64)
        pose[:,3:]/=pose[:,3:].norm(dim=-1,keepdim=True)
        velocity=torch.randn(count,6,dtype=torch.float64)*.2
        joints=torch.randn(count,29,dtype=torch.float64)*.12+adapter.default_joint_pos
        speed=torch.randn(count,29,dtype=torch.float64)*.15
        com=adapter.default_torso_com.expand(count,-1)
        state=torso_state_to_pelvis(adapter,pose,velocity,joints,speed,com)
        position,quaternion,linear,angular=adapter.kinematics(state)
        torso=adapter.anchor_id
        actual_com_velocity=linear[:,torso]+torch.cross(angular[:,torso],quat_apply(quaternion[:,torso],com),dim=-1)
        torch.testing.assert_close(position[:,torso],pose[:,:3],atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(quaternion[:,torso],pose[:,3:],atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(actual_com_velocity,velocity[:,:3],atol=1e-12,rtol=1e-12)
        torch.testing.assert_close(angular[:,torso],velocity[:,3:],atol=1e-12,rtol=1e-12)
        for row in range(count):
            data=mujoco.MjData(model)
            data.qpos[:7]=state[row,:7].numpy();data.qpos[7:]=joints[row].numpy()
            data.qvel[:3]=state[row,7:10].numpy()
            data.qvel[3:6]=quat_apply(quat_inv(state[row,3:7]),state[row,10:13]).numpy()
            data.qvel[6:]=speed[row].numpy()
            mujoco.mj_forward(model,data)
            actual=np.zeros(6)
            mujoco.mj_objectVelocity(model,data,mujoco.mjtObj.mjOBJ_BODY,adapter.torso_gid,actual,0)
            # The shared fixture builds production float32 FK constants,
            # then promotes them to float64; MuJoCo keeps double constants.
            np.testing.assert_allclose(data.xpos[adapter.torso_gid],pose[row,:3].numpy(),rtol=2e-8,atol=2e-8)
            np.testing.assert_allclose(actual[:3],velocity[row,3:].numpy(),rtol=2e-8,atol=2e-8)
            np.testing.assert_allclose(actual[3:],velocity[row,:3].numpy(),rtol=2e-8,atol=2e-8)

    def test_training_ranges_are_in_torso_frame_after_random_waist(self):
        adapter,_=make_adapter();count=200
        origin=torch.rand(count,3,dtype=torch.float64)*4.
        default=adapter.default_joint_pos.expand(count,-1)
        dq=torch.zeros_like(default)
        limits=adapter.joint_limits.expand(count,-1,-1)
        com=adapter.default_torso_com.expand(count,-1)
        state=sample_hiking_reset_state(adapter,origin,default,dq,limits,com)
        p,q,v,w=adapter.kinematics(state);torso=adapter.anchor_id
        relative=p[:,torso]-origin
        self.assertTrue(bool((relative[:,:2].abs()<=.1+1e-12).all()))
        torch.testing.assert_close(relative[:,2],torch.full((count,),.9,dtype=state.dtype),atol=1e-12,rtol=1e-12)
        gravity=quat_apply(quat_inv(q[:,torso]),state.new_tensor((0.,0.,-1.)))
        torch.testing.assert_close(gravity,state.new_tensor((0.,0.,-1.)).expand(count,-1),atol=1e-12,rtol=1e-12)
        yaw=2.*torch.atan2(q[:,torso,3],q[:,torso,0])
        self.assertTrue(bool((yaw.abs()<=.1+1e-12).all()))
        com_velocity=v[:,torso]+torch.cross(w[:,torso],quat_apply(q[:,torso],com),dim=-1)
        self.assertTrue(bool((torch.cat((com_velocity,w[:,torso]),-1).abs()<=.2+1e-12).all()))
        self.assertGreater(float(state[:,3:7][:,1:3].abs().max()),.01)
        self.assertTrue(bool((state[:,13:42]>=limits[...,0]).all()))
        self.assertTrue(bool((state[:,13:42]<=limits[...,1]).all()))
        torch.testing.assert_close(state[:,42:71],torch.zeros_like(dq),rtol=0,atol=0)

    def test_controlled_standing_has_zero_com_velocity_and_source_height(self):
        adapter,_=make_adapter();count=2
        default=adapter.default_joint_pos.expand(count,-1)
        state=sample_hiking_reset_state(adapter,torch.zeros(count,3,dtype=torch.float64),
            default,torch.zeros_like(default),adapter.joint_limits.expand(count,-1,-1),
            adapter.default_torso_com.expand(count,-1),joint_position_range=(0.,0.),
            pose_range={},velocity_range={})
        p,q,v,w=adapter.kinematics(state)
        torch.testing.assert_close(p[:,adapter.anchor_id],state.new_tensor((0.,0.,.9)).expand(count,-1))
        torch.testing.assert_close(q[:,adapter.anchor_id],state.new_tensor((1.,0.,0.,0.)).expand(count,-1))
        torch.testing.assert_close(v,torch.zeros_like(v),rtol=0,atol=0)
        torch.testing.assert_close(w,torch.zeros_like(w),rtol=0,atol=0)


if __name__=="__main__":unittest.main()
