"""Actor-only schema checks and first-episode physical scoring semantics."""
from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import torch
from torch import nn

from scripts.evaluate_wmac_perceptive import (
    FirstEpisodeMetrics, PolicyHistory, _assert_no_noise_or_push, load_actor,
    evaluate_mode, parser, restrict_recovery_starts,
)
from src.wmac_perceptive.config import FORMAT, PerceptiveConfig
from src.wmac_perceptive.checkpoints import architecture_metadata
from src.wmac_perceptive.models import PerceptiveActor
from src.wmac_perceptive.perception import PerceptionSpec, unpack_actor_input


class FakeActor(nn.Module):
    spec = PerceptionSpec(height=2, width=3, ring_length=4, frames=2,
                          stride=2, delay=1, crop_top=0, crop_left=0, crop_right=0)

    def forward(self, packed):
        proprio, depth = unpack_actor_input(packed, self.spec)
        return (proprio[:, :1]+depth[:, -1, 0].mean((-2,-1))[:, None]).expand(-1,29)


class PerceptiveEvaluationTest(unittest.TestCase):
    def test_actor_only_load_preserves_network_and_refuses_flat_or_visual_protocol_changes(self):
        torch.manual_seed(82)
        actor = PerceptiveActor()
        cfg = PerceptiveConfig.defaults()
        payload = {"format":FORMAT, "config":asdict(cfg), "visual_spec":asdict(actor.spec),
                   "actor":actor.state_dict(), "world_model":{"not_constructed_or_loaded":True},
                   "architecture":architecture_metadata(cfg,actor.spec),"amp_grounding":cfg.ground_height_semantics}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"perceptive.pt"
            torch.save(payload,path)
            restored, restored_cfg, _ = load_actor(path)
            self.assertEqual(restored_cfg.task,cfg.task)
            self.assertEqual(asdict(restored.spec),asdict(actor.spec))
            for name,value in actor.state_dict().items():
                torch.testing.assert_close(restored.state_dict()[name],value,rtol=0,atol=0)
            bad = copy.deepcopy(payload); bad["format"]="amp-wmac-v1"
            torch.save(bad,path)
            with self.assertRaisesRegex(ValueError,"flat WMAC/PPO"):
                load_actor(path)
            bad = copy.deepcopy(payload); bad["visual_spec"]["delay"]=0
            torch.save(bad,path)
            with self.assertRaisesRegex(ValueError,"visual_spec differs"):
                load_actor(path)
            bad = copy.deepcopy(payload); bad["visual_spec"].pop("crop_top")
            torch.save(bad,path)
            with self.assertRaisesRegex(ValueError,"complete visual_spec"):
                load_actor(path)

    def test_physical_history_uses_pre_reset_frame_then_refills_reset_environment(self):
        actor = FakeActor()
        controller = PolicyHistory(actor,2,action_limit=.5)
        frame = torch.ones(2,*actor.spec.frame_shape)
        controller.reset(frame)
        transition={"next_depth_frame":torch.full_like(frame,.2),
                    "terminated":torch.tensor([False,True]),"timeout":torch.tensor([False,False])}
        post = torch.full_like(frame,.8)
        controller.after_step(transition,post)
        history=controller.history.buffer
        torch.testing.assert_close(history[0,:-1],frame[0].expand(3,*actor.spec.frame_shape))
        torch.testing.assert_close(history[0,-1],transition["next_depth_frame"][0])
        torch.testing.assert_close(history[1],post[1].expand(*actor.spec.ring_shape))
        action=controller.action(torch.ones(2,384))
        torch.testing.assert_close(action,torch.full((2,29),.5))

    def test_recovery_requires_half_second_then_independent_full_second_of_tracking(self):
        score=FirstEpisodeMetrics(1,.1)
        def update(yaw=.1,relative=.8,done=False):
            values={"relative_height":torch.tensor([relative]),"up_z":torch.tensor([1.]),
                    "xy_error":torch.tensor([.1]),"yaw_error":torch.tensor([yaw]),
                    "moving":torch.tensor([True]),"terminated":torch.tensor([done]),
                    "timeout":torch.tensor([False]),"sensor_valid":torch.tensor([.7]),
                    "target_progress":torch.tensor([.01]),"target_distance":torch.tensor([1.])}
            return score.update(**values)
        for _ in range(4): update()
        self.assertFalse(bool(score.recovered[0]))
        update()
        self.assertTrue(bool(score.recovered[0]))
        self.assertAlmostEqual(float(score.recovery_time[0]),.5)
        for _ in range(9): update()
        self.assertFalse(bool(score.continued[0]))
        update()
        self.assertTrue(bool(score.continued[0]))
        self.assertEqual(score.summary("recovery")["success_then_walk_count"],1)

    def test_terminal_successor_is_included_but_auto_reset_episodes_cannot_score(self):
        score=FirstEpisodeMetrics(1,.1)
        values={"relative_height":torch.tensor([.4]),"up_z":torch.tensor([.2]),
                "xy_error":torch.tensor([2.]),"yaw_error":torch.tensor([1.]),
                "moving":torch.tensor([True]),"terminated":torch.tensor([True]),
                "timeout":torch.tensor([False]),"sensor_valid":torch.tensor([.25]),
                "target_progress":torch.tensor([-.5]),"target_distance":torch.tensor([3.])}
        row=score.update(**values)
        self.assertTrue(bool(row["first_episode_valid"][0]))
        self.assertFalse(bool(score.active[0]))
        values.update(relative_height=torch.tensor([1.2]),up_z=torch.tensor([1.]),
                      xy_error=torch.tensor([0.]),yaw_error=torch.tensor([0.]),terminated=torch.tensor([False]))
        for _ in range(20):
            row=score.update(**values)
            self.assertFalse(bool(row["first_episode_valid"][0]))
        result=score.summary("recovery")
        self.assertEqual(result["first_episode_control_steps"],1)
        self.assertEqual(result["recovery_success_count"],0)
        self.assertEqual(result["velocity_xy_error_mean"],2.)
        self.assertEqual(result["target_progress_meters_sum"],-.5)

    def test_yaw_violation_resets_continue_clock_without_erasing_recovery(self):
        score=FirstEpisodeMetrics(1,.1)
        values={"relative_height":torch.tensor([.8]),"up_z":torch.tensor([1.]),
                "xy_error":torch.tensor([0.]),"yaw_error":torch.tensor([0.]),
                "moving":torch.tensor([True]),"terminated":torch.tensor([False]),
                "timeout":torch.tensor([False]),"sensor_valid":torch.tensor([1.]),
                "target_progress":torch.tensor([.01]),"target_distance":torch.tensor([1.])}
        for _ in range(13): score.update(**values)
        values["yaw_error"]=torch.tensor([.8]);score.update(**values)
        self.assertTrue(bool(score.recovered[0]))
        self.assertEqual(int(score.walk_clock[0]),0)
        values["yaw_error"]=torch.tensor([0.])
        for _ in range(9): score.update(**values)
        self.assertFalse(bool(score.continued[0]))
        score.update(**values)
        self.assertTrue(bool(score.continued[0]))

    def test_fallen_pool_filter_is_local_and_noise_push_contract_is_checked(self):
        frames={"root_pos":torch.tensor([[0.,0.,.3],[0.,0.,.8]]),
                "root_quat":torch.tensor([[1.,0.,0.,0.],[1.,0.,0.,0.]]),
                "other":torch.tensor([[1.],[2.]])}
        other_manager=SimpleNamespace(recovery_frames={"walk":copy.deepcopy(frames)})
        env=SimpleNamespace(_perceptive_motion_reset_manager=SimpleNamespace(recovery_frames={"walk":copy.deepcopy(frames)}),
            cfg=SimpleNamespace(events={"init_motion_loader":SimpleNamespace(params={"motion_dir":"walk"})},
                                observations={"actor":SimpleNamespace(enable_corruption=False)}))
        self.assertEqual(restrict_recovery_starts(env),1)
        self.assertEqual(len(env._perceptive_motion_reset_manager.recovery_frames["walk"]["root_pos"]),1)
        self.assertEqual(len(other_manager.recovery_frames["walk"]["root_pos"]),2)
        _assert_no_noise_or_push(env)
        env.cfg.events["push_robot"]=object()
        with self.assertRaisesRegex(ValueError,"pushes disabled"):_assert_no_noise_or_push(env)
        env.cfg.events.pop("push_robot")
        env.cfg.observations["actor"].enable_corruption=True
        with self.assertRaisesRegex(ValueError,"corruption disabled"):_assert_no_noise_or_push(env)
        self.assertEqual(parser().parse_args(["--checkpoint","anything.pt"]).device,"cpu")

    def test_evaluation_loop_scores_pre_reset_states_and_records_first_episode_masks(self):
        class Scene(dict):
            pass
        class FakeEnv:
            num_envs=2;device="cpu";step_dt=.02
            def __init__(self):
                self.cfg=SimpleNamespace(events={},observations={"actor":SimpleNamespace(enable_corruption=False)})
                self.closed=False
                self.target=torch.tensor([[1.,0.,2.],[1.,0.,2.]])
                self.command_manager=SimpleNamespace(get_term=lambda unused:SimpleNamespace(target_pos_w=self.target,desired_speed=torch.full((2,),.5)))
                self.scene=Scene(hiking_depth=SimpleNamespace(cfg=SimpleNamespace(pattern=SimpleNamespace(
                    width=3,height=2,fovx=89.51,fovy=58.29,position=(0.,0.,0.),rotation=(1.,0.,0.,0.)))))
                self.scene.terrain=SimpleNamespace(terrain_types=torch.tensor([0,1]),terrain_levels=torch.tensor([0,0]))
                self.adapter=SimpleNamespace(state=lambda:self.state,context=lambda:torch.zeros(2,35),
                    command=lambda:torch.tensor([[.5,0.,0.],[.5,0.,0.]]),
                    relative_height=lambda state:state[:,2]-2.,
                    anchor_id=0,height_threshold=.5,tilt_cos=.5,max_delay_steps=250)
                def kinematics(state):
                    return state[:,:3,None].transpose(1,2),state[:,None,3:7],torch.tensor([[[.5,0.,0.]],[[.5,0.,0.]]]),torch.zeros(2,1,3)
                self.adapter.kinematics=kinematics
                self.reset()
            def reset(self,**kwargs):
                self.t=0;self.state=torch.zeros(2,71);self.state[:,2]=2.8;self.state[:,3]=1.
                self.obs_buf={"actor":torch.zeros(2,384)}
                return self.obs_buf,{}
            def step(self,action):
                previous=self.state.clone();self.state[:,0]+=.01;self.t+=1
                successor=self.state.clone();done=torch.tensor([False,self.t==2])
                if self.t==2:
                    successor[1,2]=2.2
                self.transition={"state":previous,"next_state":successor,"command":self.adapter.command(),
                    "navigation_target":self.target.clone(),"next_depth_frame":torch.ones(2,2,2,3),
                    "terminated":done,"timeout":torch.zeros(2,dtype=torch.bool)}
                return self.obs_buf,None,done,torch.zeros_like(done),{}
            def close(self):self.closed=True
        env=FakeEnv();cfg=PerceptiveConfig.defaults()
        args=SimpleNamespace(num_envs=2,seed=19,device="cpu",steps=4,save_trace=True)
        with tempfile.TemporaryDirectory() as directory, \
             patch("scripts.train_wmac_perceptive.make_environment",return_value=env), \
             patch("src.tasks.amp_perceptive.sensors.read_depth_frame",return_value=torch.ones(2,2,2,3)):
            result=evaluate_mode(FakeActor(),cfg,args,"walk",Path(directory))
            trace=np.load(result["trace"])
            np.testing.assert_array_equal(trace["first_episode_valid"],[[True,True],[True,True],[True,False],[True,False]])
            self.assertAlmostEqual(float(trace["relative_height"][1,1]),.2,places=5)
        self.assertTrue(env.closed)
        self.assertEqual(result["first_episode_control_steps"],6)
        self.assertEqual(result["per_terrain_type"]["1"]["first_episode_control_steps"],2)
        self.assertAlmostEqual(result["target_progress_meters_sum"],.06,places=5)
        self.assertEqual(result["physical_recovery_window_steps"],250)


if __name__=="__main__":
    unittest.main()
