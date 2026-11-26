"""Hiking posture reward parity across physical and imagined trajectories."""
from __future__ import annotations

import ast
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from mjlab.managers.reward_manager import RewardManager, RewardTermCfg
from src.tasks.amp_perceptive.env_cfg import g1_amp_perceptive_hiking_env_cfg
from src.tasks.amp_perceptive.mdp import hiking_stand_still, hiking_stand_still_term
from src.wmac_perceptive.state import PerceptiveStateAdapter


class HikingStandingRewardTest(unittest.TestCase):
    def test_command_gates_are_strict_and_use_xy_norm_and_absolute_yaw(self):
        command=torch.tensor([[0.,0.,0.],[.149,0.,0.],[.15,0.,0.],
            [0.,-.15,0.],[.11,.11,0.],[0.,0.,.149],[0.,0.,-.15],
            [.5,0.,0.],[0.,0.,1.]],dtype=torch.float64)
        joint=torch.zeros(len(command),29,dtype=torch.float64)
        raw=hiking_stand_still_term(joint,torch.zeros(29,dtype=joint.dtype),command)
        torch.testing.assert_close(raw,raw.new_tensor([-4.,-4.,0.,0.,0.,-4.,0.,0.,0.]),rtol=0,atol=0)

    def test_reward_manager_and_imagined_adapter_match_and_apply_dt_once(self):
        config=g1_amp_perceptive_hiking_env_cfg(motion_data_protocol="hiking_walk_v1",
            locomotion_objective="hiking_standing_v1")
        term=config.rewards["stand_still"]
        n=4
        state=torch.zeros(n,71);state[:,2]=.8;state[:,3]=1.
        default=torch.linspace(-.3,.3,29)
        successor=state.clone();successor[:,13:42]=default
        successor[-1,13:42]+=.2
        successor.requires_grad_(True)
        command=torch.tensor([[0.,0.,0.],[.5,0.,0.],[0.,0.,1.],[.1,0.,.1]])
        # Large encoder bias must have no effect on a physical posture reward.
        robot_data=SimpleNamespace(joint_pos=successor[:,13:42],
            default_joint_pos=default.expand(n,-1),
            joint_pos_biased=successor[:,13:42]+10.,encoder_bias=torch.full((n,29),10.))
        env=SimpleNamespace(num_envs=n,device="cpu",scene={"robot":SimpleNamespace(data=robot_data)},
            command_manager=SimpleNamespace(get_command=lambda name:command))
        manager=RewardManager({"stand_still":term},env,scale_by_dt=True)
        physical=manager.compute(.02).clone()

        # Exercise the actual adapter aggregation (including its base method).
        # The unrelated mandatory tracking terms are zero-weighted here.
        adapter=PerceptiveStateAdapter.__new__(PerceptiveStateAdapter)
        adapter.anchor_id=0;adapter.dt=.02;adapter.reward_scale=.02
        adapter.default_joint_pos=default
        adapter.joint_limits=torch.tensor([[-100.,100.]]).expand(29,-1)
        adapter.terrain=SimpleNamespace(height_at=lambda points:torch.zeros_like(points[...,0]))
        adapter.reward_cfg={name:RewardTermCfg(func=lambda env:None,weight=0.,params={"std":1.})
            for name in ("track_anchor_linear_velocity","track_anchor_angular_velocity","body_ang_vel_xy_l2")}
        adapter.reward_cfg["stand_still"]=term
        fk=(successor[:,:3,None].transpose(1,2),successor[:,None,3:7],
            successor[:,None,7:10],successor[:,None,10:13])
        imagined=adapter.task_reward(state,successor,torch.zeros(n,29),torch.zeros(n,29),
            command,torch.zeros(n,35),torch.zeros(n),torch.zeros(n,2),kinematics=fk)
        torch.testing.assert_close(imagined,physical,rtol=1e-6,atol=1e-8)
        torch.testing.assert_close(imagined,torch.tensor([1.2*.02,0.,0.,-.3*(29*.2-4.)*.02]),
            rtol=1e-6,atol=1e-8)
        gradient=torch.autograd.grad(imagined.sum(),successor)[0]
        torch.testing.assert_close(gradient[-1,13:42],torch.full((29,),-.3*.02))
        torch.testing.assert_close(gradient[:-1],torch.zeros_like(gradient[:-1]),rtol=0,atol=0)
        # An ordinary legacy adapter does not acquire this new objective.
        adapter.reward_cfg.pop("stand_still")
        legacy=adapter.task_reward(state,successor,torch.zeros(n,29),torch.zeros(n,29),
            command,torch.zeros(n,35),torch.zeros(n),torch.zeros(n,2),kinematics=fk)
        torch.testing.assert_close(legacy,torch.zeros(n),rtol=0,atol=0)

    def test_batched_imagination_preserves_l1_joint_gradient(self):
        joint=torch.full((3,4,29),.2,dtype=torch.float64,requires_grad=True)
        with torch.no_grad():joint[0,0,0]=-.2
        command=torch.zeros(3,4,3,dtype=torch.float64)
        command[1,:,2]=.5
        reward=-.3*.02*hiking_stand_still_term(joint,torch.zeros(29,dtype=joint.dtype),command)
        gradient=torch.autograd.grad(reward.sum(),joint)[0]
        expected=torch.full_like(joint,-.006);expected[0,0,0]=.006;expected[1]=0.
        torch.testing.assert_close(gradient,expected,rtol=0,atol=1e-14)

    def test_compiled_pure_term_has_same_values_and_gradients(self):
        fn=torch.compile(hiking_stand_still_term,backend="eager",fullgraph=True)
        joint=torch.linspace(-.3,.3,58).reshape(2,29).requires_grad_(True)
        command=torch.tensor([[0.,0.,0.],[0.,0.,-.7]])
        reference=hiking_stand_still_term(joint,torch.zeros(29),command)
        actual=fn(joint,torch.zeros(29),command)
        torch.testing.assert_close(actual,reference,rtol=0,atol=0)
        expected_gradient=torch.autograd.grad(reference.sum(),joint,retain_graph=True)[0]
        actual_gradient=torch.autograd.grad(actual.sum(),joint)[0]
        torch.testing.assert_close(actual_gradient,expected_gradient,rtol=0,atol=0)

    def test_configuration_enables_only_explicit_new_hiking_objective(self):
        legacy=g1_amp_perceptive_hiking_env_cfg(motion_data_protocol="hiking_walk_v1")
        updated=g1_amp_perceptive_hiking_env_cfg(motion_data_protocol="hiking_walk_v1",
            locomotion_objective="hiking_standing_v1")
        self.assertNotIn("stand_still",legacy.rewards)
        self.assertEqual(legacy.commands["twist"].rel_standing_envs,0.)
        self.assertEqual(set(updated.rewards)-set(legacy.rewards),{"stand_still"})
        self.assertEqual(updated.rewards["stand_still"].weight,-.3)
        self.assertEqual(updated.rewards["stand_still"].params,
            {"command_name":"twist","threshold":.15,"offset":4.})
        self.assertEqual(updated.commands["twist"].rel_standing_envs,.05)
        self.assertEqual(updated.commands["twist"].resampling_time_range,(8.,12.))
        self.assertEqual(updated.commands["twist"].in_place_turn_fraction,.2)
        for name in legacy.rewards:
            self.assertEqual(updated.rewards[name].weight,legacy.rewards[name].weight)
            self.assertIs(updated.rewards[name].func,legacy.rewards[name].func)
        for name in ("reset_base","reset_robot_joints"):
            self.assertEqual(updated.events[name].params,legacy.events[name].params)
        with self.assertRaisesRegex(ValueError,"locomotion objective"):
            g1_amp_perceptive_hiking_env_cfg(locomotion_objective="unknown")
        with self.assertRaisesRegex(ValueError,"requires forward-walk/turn"):
            g1_amp_perceptive_hiking_env_cfg(locomotion_objective="hiking_standing_v1")
        with self.assertRaises(ValueError):
            g1_amp_perceptive_hiking_env_cfg(motion_scope="joint_recovery_v1",
                locomotion_objective="hiking_standing_v1")

    def test_matches_executed_pinned_hiking_function(self):
        root=Path(os.environ.get("HIKING_REFERENCE_CHECKOUT","/tmp/hiking_locomotion_source"))
        path=root/"source/instinctlab/instinctlab/tasks/parkour/mdp/rewards.py"
        if not path.is_file():self.skipTest("Optional pinned Hiking source checkout is not present")
        function=next(node for node in ast.parse(path.read_text()).body
            if isinstance(node,ast.FunctionDef) and node.name=="stand_still")
        # Run the exact upstream body without importing IsaacLab in this env.
        function.returns=None
        for argument in function.args.args:argument.annotation=None
        module=ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[]))
        scope={"torch":torch,"SceneEntityCfg":lambda name:SimpleNamespace(name=name)}
        exec(compile(module,str(path),"exec"),scope)
        torch.manual_seed(391)
        joint=torch.randn(200,29,dtype=torch.float64)
        default=torch.randn(200,29,dtype=torch.float64)
        commands=torch.randn(200,3,dtype=torch.float64)*.2
        env=SimpleNamespace(scene={"robot":SimpleNamespace(data=SimpleNamespace(
            joint_pos=joint,default_joint_pos=default))},
            command_manager=SimpleNamespace(get_command=lambda name:commands))
        expected=scope["stand_still"](env,"twist",offset=4.)
        actual=hiking_stand_still(env,offset=4.)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)


if __name__=="__main__":unittest.main()
