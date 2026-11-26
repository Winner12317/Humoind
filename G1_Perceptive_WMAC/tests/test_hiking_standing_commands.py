"""Standing must remain a zero command in collection and imagined motion."""
from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from src.tasks.amp_perceptive.commands import (
    FlatPatchVelocityCommand, FlatPatchVelocityCommandCfg,
    command_from_state, navigation_terrain_curriculum,
)
from src.tasks.amp_perceptive.terrain import TerrainHeightField
from src.wmac.env import CaptureEnv
from src.wmac_perceptive.env import PerceptiveCaptureEnv


class Scene(dict):
    pass


def command_fixture(count=4, standing=.05, turns=.2, duration=(8., 12.)):
    cfg=FlatPatchVelocityCommandCfg(
        entity_name="robot", resampling_time_range=duration,
        rel_standing_envs=standing, in_place_turn_fraction=turns,
        ranges=FlatPatchVelocityCommandCfg.Ranges(
            lin_vel_x=(.2,.7), lin_vel_y=(0.,0.), ang_vel_z=(-1.,1.)))
    data=SimpleNamespace(
        root_link_pos_w=torch.tensor([[0.,0.,.79]]).repeat(count,1),
        root_link_quat_w=torch.tensor([[1.,0.,0.,0.]]).repeat(count,1),
        root_link_lin_vel_b=torch.zeros(count,3), root_link_ang_vel_b=torch.zeros(count,3))
    scene=Scene(robot=SimpleNamespace(data=data))
    scene.env_origins=torch.zeros(count,3)
    scene.terrain=SimpleNamespace(
        flat_patches={"target":torch.tensor([[[[2.,0.,0.],[-2.,0.,0.],[0.,2.,0.],[0.,-2.,0.]]]])},
        terrain_levels=torch.zeros(count,dtype=torch.long),
        terrain_types=torch.zeros(count,dtype=torch.long))
    env=SimpleNamespace(num_envs=count,device=torch.device("cpu"),scene=scene,step_dt=.02)
    return FlatPatchVelocityCommand(cfg,env),env,data


class HikingStandingCommandTest(unittest.TestCase):
    def test_standing_stays_zero_after_predicted_drift_but_zero_speed_turn_is_preserved(self):
        state=torch.zeros(4,71,dtype=torch.float64,requires_grad=True)
        with torch.no_grad():
            state[:,3]=1.
            state[0,:2]=torch.tensor([.3,-.2])
        target=torch.tensor([[2.,1.,0.]]*4,dtype=torch.float64)
        speed=torch.tensor([0.,0.,.5,.5],dtype=torch.float64)
        standing=torch.tensor([True,False,True,False])
        expected=command_from_state(state,target,speed,standing_mask=standing)
        compiled=torch.compile(command_from_state,backend="eager",fullgraph=True)
        actual=compiled(state,target,speed,standing_mask=standing)
        torch.testing.assert_close(actual,expected,rtol=0,atol=0)
        self.assertTrue(bool((actual[standing]==0.).all()))
        self.assertEqual(float(actual[1,0]),0.)
        self.assertGreater(float(actual[1,2]),0.)
        self.assertGreater(float(actual[3,0]),0.)
        actual.sum().backward()
        self.assertTrue(bool(torch.isfinite(state.grad).all()))
        self.assertTrue(bool((state.grad[standing]==0.).all()))

    def test_sampling_separates_five_percent_standing_from_remaining_turns(self):
        torch.manual_seed(2032)
        command,env,_=command_fixture(count=10000)
        command._resample_command(torch.arange(env.num_envs))
        standing=command.is_standing_env
        turning=command.is_in_place_turn
        self.assertTrue(.04<float(standing.float().mean())<.06)
        self.assertTrue(.18<float(turning[~standing].float().mean())<.22)
        self.assertFalse(bool((standing&turning).any()))
        self.assertTrue(bool((command.command[standing]==0.).all()))
        self.assertFalse(bool(command.goal_is_traversal[standing].any()))
        self.assertTrue(bool((command.desired_speed[standing]==0.).all()))
        self.assertTrue(bool((command.command[turning,2].abs()>0.).all()))

    def test_reset_immediately_publishes_sampled_command_without_changing_physics(self):
        torch.manual_seed(2033)
        command,env,data=command_fixture(count=128)
        initial_pos=data.root_link_pos_w.clone()
        initial_quat=data.root_link_quat_w.clone()
        command.vel_command_b.fill_(.123)
        command.reset(torch.arange(env.num_envs))
        expected=command_from_state(command._root_pose(),command.target_pos_w,command.desired_speed,
            standing_mask=command.is_standing_env)
        torch.testing.assert_close(command.command,expected,rtol=0,atol=0)
        self.assertTrue(bool(command.is_standing_env.any()))
        self.assertTrue(bool((command.command_counter==1).all()))
        self.assertTrue(bool(((command.time_left>=8.)&(command.time_left<=12.)).all()))
        torch.testing.assert_close(data.root_link_pos_w,initial_pos,rtol=0,atol=0)
        torch.testing.assert_close(data.root_link_quat_w,initial_quat,rtol=0,atol=0)
        self.assertTrue(bool((data.root_link_lin_vel_b==0.).all()))

    def test_standing_ignores_goal_completion_until_full_resampling_interval(self):
        command,env,_=command_fixture(standing=1.,duration=(8.,8.))
        ids=torch.arange(env.num_envs)
        command.reset(ids)
        command.target_pos_w.copy_(command._root_pose()[:,:3])
        for _ in range(399):
            command.compute(.02)
        self.assertTrue(bool((command.command==0.).all()))
        self.assertTrue(bool((command.command_counter==1).all()))
        self.assertTrue(bool((command.goals_reached==0.).all()))
        self.assertTrue(bool((command.turn_goals_reached==0.).all()))
        command.compute(.04)
        self.assertTrue(bool((command.command_counter==2).all()))
        self.assertTrue(bool((command.time_left==8.).all()))

    def test_goal_finished_into_standing_gets_full_duration_and_new_command_same_step(self):
        command,env,_=command_fixture(standing=1.,duration=(8.,8.))
        command.is_standing_env.zero_()
        command.is_in_place_turn.zero_()
        command.goal_is_traversal.fill_(True)
        command.target_pos_w.copy_(command._root_pose()[:,:3])
        command.desired_speed.fill_(.5)
        command.time_left.fill_(.04)
        command.vel_command_b.fill_(.5)
        command._update_command()
        self.assertTrue(bool(command.is_standing_env.all()))
        self.assertTrue(bool((command.command==0.).all()))
        self.assertTrue(bool((command.time_left==8.).all()))
        self.assertTrue(bool((command.command_counter==1).all()))
        self.assertTrue(bool((command.goals_reached==1.).all()))
        command._update_command()
        self.assertTrue(bool((command.goals_reached==1.).all()))

    def test_no_standing_legacy_keeps_reset_and_finished_target_timing(self):
        command,env,_=command_fixture(standing=0.,turns=0.)
        command.vel_command_b.fill_(.123)
        command.reset(torch.arange(env.num_envs))
        self.assertTrue(bool((command.command==.123).all()))
        self.assertFalse(bool(command.is_standing_env.any()))
        command.target_pos_w.copy_(command._root_pose()[:,:3])
        command.time_left.fill_(.04)
        before_counter=command.command_counter.clone()
        command._update_command()
        self.assertTrue(bool((command.command==0.).all()))
        self.assertTrue(bool((command.time_left==.04).all()))
        torch.testing.assert_close(command.command_counter,before_counter)

    def test_standing_never_counts_as_terrain_traversal(self):
        command,env,_=command_fixture(standing=1.)
        command.reset(torch.arange(env.num_envs))
        env.episode_length_buf=torch.full((env.num_envs,),500)
        env.reset_terminated=torch.zeros(env.num_envs,dtype=torch.bool)
        env.command_manager=SimpleNamespace(get_term=lambda name:command)
        env._perceptive_terrain_oracle=TerrainHeightField(
            torch.zeros(1,5,5),torch.zeros(1,2),torch.full((1,2),4.))
        def update(ids,up,down):
            env.scene.terrain.last_update=(up.clone(),down.clone())
        env.scene.terrain.update_env_origins=update
        navigation_terrain_curriculum(env,torch.arange(env.num_envs))
        up,down=env.scene.terrain.last_update
        self.assertFalse(bool(up.any()))
        self.assertFalse(bool(down.any()))

    def test_capture_stores_pre_step_mode_and_flags_mode_only_switch_as_boundary(self):
        command=SimpleNamespace(target_pos_w=torch.ones(2,3),desired_speed=torch.zeros(2),
                                is_standing_env=torch.tensor([True,False]))
        env=object.__new__(PerceptiveCaptureEnv)
        env.obs_buf={"actor":torch.zeros(2,384)}
        env.command_manager=SimpleNamespace(get_term=lambda name:command)
        env.adapter=SimpleNamespace(relative_height=lambda state:state[:,2])
        def physical_step(self,action):
            self.transition={"state":torch.zeros(2,71),"boundary":torch.tensor([False,True])}
            command.is_standing_env[0]=False
            return "physical_result"
        with patch.object(CaptureEnv,"step",physical_step):
            self.assertEqual(env.step(torch.zeros(2,29)),"physical_result")
        self.assertEqual(env.transition["command_standing"].tolist(),[True,False])
        self.assertEqual(env.transition["command_boundary"].tolist(),[True,False])
        self.assertEqual(env.transition["boundary"].tolist(),[True,True])


if __name__=="__main__":
    unittest.main()
