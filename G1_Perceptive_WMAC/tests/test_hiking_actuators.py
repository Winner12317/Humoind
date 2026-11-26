from types import SimpleNamespace
import unittest

import mujoco
import torch

from mjlab.actuator.actuator import ActuatorCmd
from src.assets.robots.unitree_g1.g1_constants import G1_XML
from src.wmac_perceptive.hiking_actuators import (
    HikingDelayedPdActuator,HikingDelayedPdActuatorCfg,ResetFixedDelay,
    configure_hiking_actuators,get_hiking_joint_parameters,
    read_hiking_actuator_parameters,read_hiking_delay_per_joint,
)


class HikingActuatorTest(unittest.TestCase):
    def test_source_parameters_all_29_joints_and_action_scales(self):
        model=mujoco.MjModel.from_xml_path(str(G1_XML))
        names=[model.joint(i).name for i in range(1,model.njnt)]
        p=get_hiking_joint_parameters(names)
        self.assertEqual(len(names),29)
        self.assertEqual(torch.bincount(p["delay_group"]).tolist(),[8,4,2,1,14])
        i=names.index("left_wrist_pitch_joint")
        self.assertEqual(float(p["effort_limit"][i]),5.)
        self.assertAlmostEqual(float(p["armature"][i]),.00425,places=8)
        i=names.index("left_knee_joint")
        self.assertEqual(float(p["effort_limit"][i]),139.)
        self.assertEqual(float(p["velocity_limit"][i]),20.)
        self.assertAlmostEqual(float(p["kp"][i]),.025101925*(20*3.1415926535)**2,places=4)
        from src.tasks.amp_loco.config.g1.env_cfgs import g1_amp_flat_aligned_env_cfg
        cfg=g1_amp_flat_aligned_env_cfg()
        previous=cfg.scene.entities["robot"].articulation
        configure_hiking_actuators(cfg)
        self.assertIsNot(previous,cfg.scene.entities["robot"].articulation)
        self.assertEqual(len(cfg.scene.entities["robot"].articulation.actuators),5)
        self.assertAlmostEqual(cfg.actions["joint_pos"].scale['.*_wrist_pitch_joint'],
                               .25*5/(.00425*(20*3.1415926535)**2),places=10)

    def test_reset_fixed_lags_and_first_sample_backfill(self):
        delay=ResetFixedDelay(3,1,"cpu");delay.lags.copy_(torch.tensor([0,1,2]))
        expected=([10,10,10],[20,10,10],[30,20,10],[40,30,20])
        for index,answer in enumerate(expected):
            output=delay.append_and_get(torch.full((3,1),float(10*(index+1))))
            torch.testing.assert_close(output[:,0],torch.tensor(answer,dtype=torch.float32))
        torch.testing.assert_close(delay.lags,torch.tensor([0,1,2]))
        delay.reset(torch.tensor([1]))
        output=delay.append_and_get(torch.full((3,1),50.))
        torch.testing.assert_close(output[:,0],torch.tensor([50.,50.,30.]))

    def test_lag_sampling_is_uniform_and_constant_between_resets(self):
        torch.manual_seed(37)
        delay=ResetFixedDelay(9000,2,"cpu");delay.reset()
        counts=torch.bincount(delay.lags).float()/9000
        self.assertTrue(bool(((counts>.31)&(counts<.36)).all()))
        lags=delay.lags.clone()
        for _ in range(4):delay.append_and_get(torch.ones(9000,2))
        torch.testing.assert_close(delay.lags,lags)

    def test_pd_clips_effort_without_motor_speed_curve_and_runtime_mapping(self):
        actuator=object.__new__(HikingDelayedPdActuator)
        actuator._target_ids=torch.tensor([1,0])
        actuator.delay=ResetFixedDelay(1,6,"cpu")
        actuator.delay.lags.zero_()
        actuator.stiffness=torch.tensor([[10.,20.]])
        actuator.damping=torch.tensor([[1.,2.]])
        actuator.force_limit=torch.tensor([[5.,8.]])
        actuator.velocity_limit=torch.tensor([[37.,22.]])
        cmd=ActuatorCmd(torch.ones(1,2),torch.zeros(1,2),torch.zeros(1,2),
                       torch.zeros(1,2),torch.tensor([[100.,-100.]]))
        torch.testing.assert_close(actuator.compute(cmd),torch.tensor([[-5.,8.]]))
        robot=SimpleNamespace(data=SimpleNamespace(joint_pos=torch.zeros(1,2)),actuators=[actuator])
        p=read_hiking_actuator_parameters(robot)
        torch.testing.assert_close(p["kp"],torch.tensor([[20.,10.]]))
        actuator.delay.lags.fill_(2)
        torch.testing.assert_close(read_hiking_delay_per_joint(robot),torch.tensor([[2.,2.]]))

    def test_cfg_edits_mjcf_to_motor_with_source_armature(self):
        from src.assets.robots.unitree_g1.g1_constants import get_spec
        from mjlab.entity import Entity
        from src.tasks.amp_loco.config.g1.env_cfgs import g1_amp_flat_aligned_env_cfg
        cfg=configure_hiking_actuators(g1_amp_flat_aligned_env_cfg())
        entity=Entity(cfg.scene.entities["robot"])
        self.assertEqual(len(entity.actuators),5)
        model=entity.spec.compile()
        self.assertEqual(model.nu,29)
        names=[model.joint(i).name for i in range(1,model.njnt)]
        p=get_hiking_joint_parameters(names,dtype=torch.float64)
        torch.testing.assert_close(torch.from_numpy(model.dof_armature[6:]),p["armature"],rtol=0,atol=1e-12)


if __name__=="__main__":unittest.main()
