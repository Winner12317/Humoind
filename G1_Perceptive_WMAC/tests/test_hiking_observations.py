from types import SimpleNamespace
import unittest

import torch

from src.wmac.state import quat_from_rotvec
from src.wmac_perceptive.hiking_observations import (
    configure_hiking_observations,hiking_actor_frame,hiking_angular_velocity,
    hiking_projected_gravity,hiking_proprio_frame,hiking_proprio_noise_amplitudes,
    pack_hiking_proprio_history,unpack_hiking_proprio_history,
)


class HikingObservationsTest(unittest.TestCase):
    def test_real_and_imagined_use_torso_not_pelvis_and_match_scales(self):
        q=quat_from_rotvec(torch.tensor([[.2,-.3,.4]]))
        w=torch.tensor([[1.,2.,3.]])
        state=torch.zeros(1,71);state[:,3]=1.;state[:,42:]=2.
        adapter=SimpleNamespace(anchor_id=1,default_joint_pos=torch.zeros(29),default_joint_vel=torch.zeros(29),
            kinematics=lambda state:(None,torch.stack((state[:,3:7],q),1),None,
                                     torch.stack((torch.zeros_like(w),w),1)))
        data=SimpleNamespace(body_link_quat_w=torch.stack((state[:,3:7],q),1),
                             body_link_ang_vel_w=torch.stack((torch.zeros_like(w),w),1))
        env=SimpleNamespace(scene={"robot":SimpleNamespace(data=data)})
        cfg=SimpleNamespace(name="robot",body_ids=[1])
        frame=hiking_actor_frame(adapter,state,torch.ones(1,29),torch.tensor([[.5,0.,.2]]))
        self.assertEqual(frame.shape,(1,96))
        torch.testing.assert_close(frame[:,:3],hiking_angular_velocity(env,cfg)*.25)
        torch.testing.assert_close(frame[:,3:6],hiking_projected_gravity(env,cfg))
        torch.testing.assert_close(frame[:,38:67],torch.full((1,29),.1))
        self.assertGreater(float(frame[:,:3].abs().sum()),0.)

    def test_noise_is_added_before_scale_with_zero_command_and_action_noise(self):
        n=5000;q=torch.zeros(n,4);q[:,0]=1.
        args=(q,torch.zeros(n,3),torch.zeros(n,29),torch.zeros(n,29),
              torch.zeros(n,29),torch.zeros(n,3),torch.zeros(29),torch.zeros(29))
        torch.manual_seed(49)
        clean=hiking_proprio_frame(*args)
        noisy=hiking_proprio_frame(*args,noise=True)
        amplitudes=hiking_proprio_noise_amplitudes()
        self.assertTrue(bool(((noisy-clean).abs()<=amplitudes+1e-7).all()))
        self.assertTrue(bool(((noisy-clean).abs().amax(0)[amplitudes>0]>.97*amplitudes[amplitudes>0]).all()))
        torch.testing.assert_close(noisy[:,6:9],clean[:,6:9],rtol=0,atol=0)
        torch.testing.assert_close(noisy[:,67:],clean[:,67:],rtol=0,atol=0)

    def test_frame_preserves_gradients_and_compile(self):
        rot=torch.tensor([[.2,.1,.05]],requires_grad=True)
        q=quat_from_rotvec(rot)
        joints=torch.ones(1,29,requires_grad=True)
        args=(q,torch.tensor([[.1,.2,.3]]),joints,joints,torch.zeros(1,29),
              torch.tensor([[.5,0.,.1]]),torch.zeros(29),torch.zeros(29))
        expected=hiking_proprio_frame(*args)
        compiled=torch.compile(hiking_proprio_frame,backend="eager",fullgraph=True)(*args)
        torch.testing.assert_close(compiled,expected)
        compiled.square().sum().backward()
        self.assertTrue(bool(torch.isfinite(rot.grad).all()))
        self.assertTrue(bool((joints.grad>0).all()))

    def test_term_major_source_wrapper_layout_roundtrips_and_preserves_gradients(self):
        history=torch.arange(2*8*96,dtype=torch.float32).reshape(2,8,96).requires_grad_()
        packed=pack_hiking_proprio_history(history)
        expected=torch.cat([history[...,start:end].flatten(1) for start,end in
                            ((0,3),(3,6),(6,9),(9,38),(38,67),(67,96))],-1)
        torch.testing.assert_close(packed,expected,rtol=0,atol=0)
        torch.testing.assert_close(unpack_hiking_proprio_history(packed),history,rtol=0,atol=0)
        unpack_hiking_proprio_history(packed).sum().backward()
        torch.testing.assert_close(history.grad,torch.ones_like(history),rtol=0,atol=0)
        compiled=torch.compile(pack_hiking_proprio_history,backend="eager",fullgraph=True)
        torch.testing.assert_close(compiled(history),packed)

    def test_configuration_has_eight_term_major_frames_and_source_noise(self):
        from src.tasks.amp_loco.config.g1.env_cfgs import g1_amp_flat_aligned_env_cfg
        cfg=configure_hiking_observations(g1_amp_flat_aligned_env_cfg())
        actor=cfg.observations["actor"]
        self.assertEqual(actor.history_length,8)
        self.assertEqual(actor.history_ordering,"term")
        self.assertEqual(tuple(actor.terms),('base_ang_vel','projected_gravity','command','joint_pos','joint_vel','actions'))
        self.assertEqual(actor.terms["base_ang_vel"].scale,.25)
        self.assertEqual(actor.terms["joint_vel"].scale,.05)
        self.assertEqual(actor.terms["joint_vel"].noise.n_min,-.5)


if __name__=="__main__":unittest.main()
