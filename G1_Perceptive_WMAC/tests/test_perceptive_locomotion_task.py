"""Walking-only task scope, explicit turns and immediate fall termination."""
from __future__ import annotations

import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from mjlab.managers.termination_manager import TerminationManager
from src.tasks.amp_perceptive.commands import (
    FlatPatchVelocityCommand, command_from_state, navigation_terrain_curriculum,
)
from src.tasks.amp_perceptive.env_cfg import g1_amp_perceptive_hiking_env_cfg
from src.tasks.amp_perceptive.mdp import init_perceptive_motion_loader
from src.tasks.amp_perceptive.terrain import (
    HIKING_PROPORTIONS, WALKING_TERRAIN_PROPORTIONS, TerrainHeightField,
)


class Scene(dict):
    pass


def command_fixture(count=16,turn_fraction=1.):
    cfg=g1_amp_perceptive_hiking_env_cfg().commands["twist"]
    cfg.in_place_turn_fraction=turn_fraction
    data=SimpleNamespace(root_link_pos_w=torch.tensor([[0.,0.,.79]]).repeat(count,1),
                         root_link_quat_w=torch.tensor([[1.,0.,0.,0.]]).repeat(count,1))
    scene=Scene(robot=SimpleNamespace(data=data))
    scene.terrain=SimpleNamespace(
        flat_patches={"target":torch.tensor([[[[1.,0.,0.],[-1.,0.,0.],[0.,1.,0.],[0.,-1.,0.]]]])},
        terrain_levels=torch.zeros(count,dtype=torch.long),terrain_types=torch.zeros(count,dtype=torch.long))
    env=SimpleNamespace(num_envs=count,device=torch.device("cpu"),scene=scene,step_dt=.02)
    return FlatPatchVelocityCommand(cfg,env),env,data


class WalkingOnlyTaskTest(unittest.TestCase):
    def test_default_scope_excludes_recovery_rewards_loader_and_flat_family(self):
        cfg=g1_amp_perceptive_hiking_env_cfg()
        loader=cfg.events["init_motion_loader"].params
        self.assertEqual(loader["motion_scope"],"forward_walk_turn_v1")
        self.assertIsNone(loader["recovery_dir"])
        self.assertEqual(loader["delay_reset_env_ratio"],0.)
        self.assertEqual(loader["max_delay_steps"],0)
        self.assertNotIn("track_root_height",cfg.rewards)
        self.assertNotIn("mean_delay_steps",cfg.metrics)
        for reward in cfg.rewards.values():
            self.assertFalse(reward.params.get("mask_delay",False))
            self.assertEqual(reward.params.get("delay_env_rew_ratio",1.),1.)
        command=cfg.commands["twist"]
        self.assertEqual(command.ranges.lin_vel_x,(.2,.7))
        self.assertEqual(command.ranges.lin_vel_y,(0.,0.))
        self.assertEqual(command.ranges.ang_vel_z,(-1.,1.))
        self.assertEqual(command.in_place_turn_fraction,.2)
        terrains=cfg.scene.terrain.terrain_generator.sub_terrains
        self.assertEqual(set(terrains),set(WALKING_TERRAIN_PROPORTIONS))
        self.assertAlmostEqual(sum(t.proportion for t in terrains.values()),1.)
        self.assertNotIn("randomize_terrain",g1_amp_perceptive_hiking_env_cfg(play=True).events)

    def test_historical_scope_requires_explicit_selection_and_keeps_protocol(self):
        cfg=g1_amp_perceptive_hiking_env_cfg(motion_scope="joint_recovery_v1")
        params=cfg.events["init_motion_loader"].params
        self.assertEqual(params["motion_scope"],"joint_recovery_v1")
        self.assertEqual(params["delay_reset_env_ratio"],.4)
        self.assertEqual(params["max_delay_steps"],250)
        self.assertTrue(params["recovery_dir"].endswith("Recovery"))
        self.assertIn("track_root_height",cfg.rewards)
        self.assertEqual(cfg.commands["twist"].ranges.lin_vel_x,(.45,1.))
        self.assertEqual(cfg.commands["twist"].in_place_turn_fraction,0.)
        self.assertAlmostEqual(cfg.scene.terrain.terrain_generator.sub_terrains["recovery_flat"].proportion,.25)
        with self.assertRaises(ValueError):g1_amp_perceptive_hiking_env_cfg(motion_scope="unknown")

    def test_loader_never_installs_a_delay_manager_for_walking(self):
        original=object()
        env=SimpleNamespace(num_envs=4,device="cpu",termination_manager=original)
        with patch("src.tasks.amp_loco.mdp.events.MotionResetManager") as manager:
            init_perceptive_motion_loader(env,None,"filtered/WalkandRun")
            manager.return_value.init.assert_called_once_with(env,"filtered/WalkandRun",None)
        self.assertIs(env.termination_manager,original)
        for kwargs in ({"recovery_dir":"Recovery"},{"delay_reset_env_ratio":.4},{"max_delay_steps":250}):
            with self.subTest(kwargs=kwargs),self.assertRaisesRegex(ValueError,"Walking-only"):
                init_perceptive_motion_loader(env,None,"filtered/WalkandRun",**kwargs)

    def test_height_or_tilt_falls_terminate_on_first_control_step(self):
        cfg=g1_amp_perceptive_hiking_env_cfg()
        qpos=torch.tensor([[0.,0.,.2,1.,0.,0.,0.],[2.,0.,2.79,0.,1.,0.,0.]])
        asset=SimpleNamespace(data=SimpleNamespace(data=SimpleNamespace(qpos=qpos)),
                              indexing=SimpleNamespace(free_joint_q_adr=torch.arange(7)))
        oracle=TerrainHeightField(torch.stack((torch.zeros(5,5),torch.full((5,5),2.))),
                                 torch.tensor([[0.,0.],[2.,0.]]),torch.ones(2,2))
        env=SimpleNamespace(num_envs=2,device="cpu",scene={"robot":asset},_perceptive_terrain_oracle=oracle)
        manager=TerminationManager({name:cfg.terminations[name] for name in ("bad_base_height","bad_orientation")},env)
        self.assertEqual(manager.compute().tolist(),[True,True])
        self.assertFalse(hasattr(manager,"_delay_counters"))

    def test_commands_never_request_lateral_backward_or_fast_motion(self):
        state=torch.zeros(4,71,dtype=torch.float64,requires_grad=True)
        with torch.no_grad():state[:,3]=1.
        target=torch.tensor([[1.,1.,0.],[-1.,0.,0.],[1.,-1.,0.],[0.,0.,0.]],dtype=torch.float64)
        command=command_from_state(state,target,torch.tensor([.7,.7,0.,.7],dtype=torch.float64))
        self.assertTrue(bool((command[:,0]>=0).all()))
        self.assertTrue(bool((command[:,0]<=.7).all()))
        self.assertTrue(bool((command[:,1]==0).all()))
        self.assertEqual(command[2,0],0.)
        self.assertLess(command[2,2],0.)
        command.sum().backward()
        self.assertTrue(bool(torch.isfinite(state.grad).all()))

    def test_explicit_in_place_turns_cover_both_signs_and_are_not_traversals(self):
        torch.manual_seed(41)
        command,env,_=command_fixture(count=64)
        command._resample_command(torch.arange(env.num_envs))
        command._update_command()
        self.assertTrue(bool(command.is_in_place_turn.all()))
        self.assertTrue(bool((command.desired_speed==0).all()))
        self.assertFalse(bool(command.goal_is_traversal.any()))
        self.assertTrue(bool((command.command[:,:2]==0).all()))
        self.assertTrue(bool((command.command[:,2]>0).any()))
        self.assertTrue(bool((command.command[:,2]<0).any()))
        self.assertTrue(bool((command.command[:,2].abs()<=1.).all()))

    def test_completed_turns_resample_without_counting_a_navigation_goal(self):
        command,env,data=command_fixture(count=2)
        command._resample_command(torch.arange(2))
        target=command.target_pos_w-data.root_link_pos_w
        yaw=torch.atan2(target[:,1],target[:,0])
        data.root_link_quat_w[:,0]=torch.cos(yaw/2)
        data.root_link_quat_w[:,3]=torch.sin(yaw/2)
        command.goals_reached.fill_(3.)
        command._update_command()
        self.assertEqual(command.turn_goals_reached.tolist(),[1.,1.])
        self.assertEqual(command.goals_reached.tolist(),[3.,3.])
        self.assertTrue(bool((command.desired_speed==0).all()))

    def test_in_place_turns_cannot_advance_terrain_curriculum(self):
        command,env,_=command_fixture(count=2)
        env.scene.env_origins=torch.zeros(2,3)
        env.scene.terrain.terrain_levels=torch.ones(2,dtype=torch.long)
        def update(ids,up,down):env.scene.terrain.last_update=(up.clone(),down.clone())
        env.scene.terrain.update_env_origins=update
        env.command_manager=SimpleNamespace(get_term=lambda name:command)
        env.episode_length_buf=torch.full((2,),500)
        env.reset_terminated=torch.zeros(2,dtype=torch.bool)
        env._perceptive_terrain_oracle=TerrainHeightField(torch.zeros(1,5,5),torch.zeros(1,2),torch.full((1,2),4.))
        command.desired_speed.zero_()
        command.turn_goals_reached.fill_(20.)
        navigation_terrain_curriculum(env,torch.arange(2))
        up,down=env.scene.terrain.last_update
        self.assertFalse(bool(up.any()))
        self.assertFalse(bool(down.any()))


if __name__=="__main__":unittest.main()
