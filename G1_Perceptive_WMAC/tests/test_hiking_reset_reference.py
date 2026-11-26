"""Hiking standing resets are independent of the full AMP reference motion."""
from __future__ import annotations

import ast
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from mjlab.envs import mdp as envs_mdp
from src.tasks.amp_perceptive.env_cfg import g1_amp_perceptive_hiking_env_cfg
from src.tasks.amp_perceptive.mdp import hiking_reset_protocol_metadata


class Scene(dict):
    pass


class ResetAsset:
    is_fixed_base=False

    def __init__(self,count=5):
        states=torch.zeros(count,13)
        states[:,2]=.78  # Local pelvis default, not the official torso's .9 m.
        states[:,3]=math.cos(.3/2)
        states[:,6]=math.sin(.3/2)
        self.data=SimpleNamespace(
            default_root_state=states,
            default_joint_pos=torch.linspace(-.2,.2,29).repeat(count,1),
            default_joint_vel=torch.zeros(count,29),
            soft_joint_pos_limits=torch.tensor([[-.22,.22]]).repeat(count,29,1),
        )
        self.writes={}

    def write_root_link_pose_to_sim(self,value,env_ids):
        self.writes["root_pose"]=(value.clone(),env_ids.clone())

    def write_root_link_velocity_to_sim(self,value,env_ids):
        self.writes["root_velocity"]=(value.clone(),env_ids.clone())

    def write_joint_state_to_sim(self,position,velocity,env_ids,joint_ids):
        self.writes["joint_position"]=(position.clone(),env_ids.clone())
        self.writes["joint_velocity"]=(velocity.clone(),env_ids.clone())


class HikingResetReferenceTest(unittest.TestCase):
    def test_new_protocol_has_no_motion_loader_and_uses_standard_reset_functions(self):
        cfg=g1_amp_perceptive_hiking_env_cfg(motion_data_protocol="hiking_walk_v1")
        self.assertNotIn("init_motion_loader",cfg.events)
        self.assertNotIn("reset_from_motion",cfg.events)
        self.assertIs(cfg.events["reset_base"].func,envs_mdp.reset_root_state_uniform)
        self.assertIs(cfg.events["reset_robot_joints"].func,envs_mdp.reset_joints_by_offset)
        for name in ("reset_base","reset_robot_joints"):
            self.assertEqual(cfg.events[name].mode,"reset")
        self.assertNotIn("track_root_height",cfg.rewards)
        self.assertEqual(cfg.commands["twist"].ranges.lin_vel_y,(0.,0.))
        self.assertNotIn("randomize_terrain",g1_amp_perceptive_hiking_env_cfg(
            play=True,motion_data_protocol="hiking_walk_v1").events)

    def test_reset_parameters_match_pinned_hiking_source(self):
        root=Path(os.environ.get("HIKING_REFERENCE_CHECKOUT","/tmp/hiking_locomotion_source"))
        relative="source/instinctlab/instinctlab/tasks/parkour/config/parkour_env_cfg.py"
        path=root/relative
        if not path.is_file():
            self.skipTest("Optional pinned Hiking source checkout is not present")
        tree=ast.parse(path.read_text())
        cfg=g1_amp_perceptive_hiking_env_cfg(motion_data_protocol="hiking_walk_v1")
        for name in ("reset_base","reset_robot_joints"):
            assignments=[node for node in ast.walk(tree) if isinstance(node,ast.Assign)
                         and any(isinstance(target,ast.Name) and target.id==name
                                 for target in node.targets)]
            self.assertEqual(len(assignments),1)
            call=assignments[0].value
            expected=ast.literal_eval(next(k.value for k in call.keywords if k.arg=="params"))
            self.assertEqual(cfg.events[name].params,expected)

    def test_subset_reset_matches_independent_numeric_oracle_on_terrain_origin(self):
        asset=ResetAsset()
        scene=Scene(robot=asset)
        scene.env_origins=torch.tensor([[0.,0.,0.],[10.,20.,2.],[20.,40.,4.],
                                        [30.,60.,6.],[40.,80.,8.]])
        env=SimpleNamespace(scene=scene,num_envs=5,device=torch.device("cpu"))
        ids=torch.tensor([1,3])
        cfg=g1_amp_perceptive_hiking_env_cfg(motion_data_protocol="hiking_walk_v1")

        torch.manual_seed(713)
        raw_pose=torch.rand(2,6)
        pose=(raw_pose*2-1)*torch.tensor([.1,.1,0.,0.,0.,.1])
        expected_position=scene.env_origins[ids]+asset.data.default_root_state[ids,:3]+pose[:,:3]
        yaw=.3+pose[:,5]
        expected_quaternion=torch.stack((torch.cos(yaw/2),torch.zeros(2),
                                         torch.zeros(2),torch.sin(yaw/2)),-1)
        expected_velocity=torch.rand(2,6)*.4-.2
        expected_joints=(asset.data.default_joint_pos[ids]+torch.rand(2,29)*.3-.15).clamp(-.22,.22)

        torch.manual_seed(713)
        for name in ("reset_base","reset_robot_joints"):
            term=cfg.events[name]
            term.func(env,ids,**term.params)
        torch.testing.assert_close(asset.writes["root_pose"][0],
                                   torch.cat((expected_position,expected_quaternion),-1))
        torch.testing.assert_close(asset.writes["root_velocity"][0],expected_velocity)
        torch.testing.assert_close(asset.writes["joint_position"][0],expected_joints)
        self.assertEqual(float(asset.writes["joint_velocity"][0].abs().max()),0.)
        for value,selected in asset.writes.values():
            self.assertEqual(value.shape[0],2)
            self.assertEqual(selected.tolist(),[1,3])
        # Standing root height depends only on the model and sampled surface,
        # never an AMP frame's altitude on a staircase/platform.
        torch.testing.assert_close(asset.writes["root_pose"][0][:,2],torch.tensor([2.78,6.78]))

    def test_legacy_motion_resets_and_scope_validation_remain_explicit(self):
        cfg=g1_amp_perceptive_hiking_env_cfg()
        self.assertIn("init_motion_loader",cfg.events)
        self.assertIn("reset_from_motion",cfg.events)
        self.assertNotIn("reset_base",cfg.events)
        self.assertNotIn("reset_robot_joints",cfg.events)
        with self.assertRaisesRegex(ValueError,"forward-walk/turn"):
            g1_amp_perceptive_hiking_env_cfg(motion_scope="joint_recovery_v1",
                                           motion_data_protocol="hiking_walk_v1")
        with self.assertRaisesRegex(ValueError,"motion-data protocol"):
            g1_amp_perceptive_hiking_env_cfg(motion_data_protocol="unknown")
        metadata=hiking_reset_protocol_metadata()
        self.assertEqual(metadata["local_root"],"pelvis")
        self.assertEqual(metadata["reference_root"],"torso_link")
        self.assertFalse(metadata["reference_motion_used_for_reset"])


if __name__=="__main__":
    unittest.main()
