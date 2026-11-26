from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from src.wmac.env import CaptureEnv
from src.wmac_perceptive.env import PerceptiveCaptureEnv


def make_env(full,terminated=False,timeout=False):
    def base_init(self):
        self.cfg=SimpleNamespace(_hiking_full_protocol=full)
        self._capture_enabled=False;self._physical_end=None
        self.pose=torch.tensor([2.,3.]);self.packet=torch.tensor([1.,1.])
        self.calls=[]
        self.obs_buf={"actor":torch.zeros(2,768)}
        self.command=SimpleNamespace(target_pos_w=torch.zeros(2,3),desired_speed=torch.ones(2),
                                     is_standing_env=torch.zeros(2,dtype=torch.bool))
        self.command_manager=SimpleNamespace(get_term=lambda name:self.command)
        self.adapter=SimpleNamespace(relative_height=lambda state:state[:,2])
        self.done=torch.tensor([terminated or timeout,False])
        def metric():
            self._physical_end={"state":torch.zeros(2,71),"terminated":torch.tensor([terminated,False]),
                                "timeout":torch.tensor([timeout,False]),"boundary":self.done.clone()}
        self.metrics_manager=SimpleNamespace(compute=metric)
        def reset(ids):self.calls.append("reset");self.pose[ids]=9.
        self._reset_idx=reset
        def forward():self.calls.append("forward")
        def sense():self.calls.append("sense");self.packet=self.pose.clone()
        self.sim=SimpleNamespace(forward=forward,sense=sense)
    with patch.object(CaptureEnv,"__init__",base_init):return PerceptiveCaptureEnv()


def base_step(self,action):
    self._capture_enabled=True
    self.metrics_manager.compute()
    ids=self.done.nonzero().squeeze(-1)
    if len(ids):self._reset_idx(ids)
    self.sim.forward();self.sim.sense()
    self.policy_depth=self.packet.clone()
    self.history_updates=1
    self._capture_enabled=False
    self.transition=self._physical_end.copy()
    return self.policy_depth


class HikingDepthTimingTest(unittest.TestCase):
    def test_full_capture_reuses_actor_observation_and_legacy_preserves_compute(self):
        env=make_env(True)
        calls=[]
        env.observation_manager=SimpleNamespace(compute=lambda:(calls.append(True) or {"actor":torch.ones(2,768)}))
        self.assertIs(env._actor_history_for_capture(),env.obs_buf["actor"])
        self.assertEqual(calls,[])
        env._fresh_depth_capture=False
        self.assertTrue(bool((env._actor_history_for_capture()==1).all()))
        self.assertEqual(calls,[True])

    def step(self,env):
        with patch.object(CaptureEnv,"step",base_step),patch(
            "src.tasks.amp_perceptive.sensors.read_depth_frame",lambda env:env.packet[:,None,None,None]),patch(
            "src.tasks.amp_perceptive.hiking_rewards.hiking_auxiliary_targets",lambda env:torch.zeros(2,13)):
            return env.step(torch.zeros(2,29))

    def test_normal_step_measures_current_successor_with_one_sense(self):
        env=make_env(True);self.step(env)
        torch.testing.assert_close(env.transition["next_depth_frame"].flatten(),torch.tensor([2.,3.]))
        self.assertEqual(env.calls,["forward","sense"])
        self.assertEqual(env.history_updates,1)

    def test_terminal_depth_is_pre_reset_but_policy_depth_is_reset_fresh(self):
        for timeout in (False,True):
            with self.subTest(timeout=timeout):
                env=make_env(True,terminated=not timeout,timeout=timeout);self.step(env)
                torch.testing.assert_close(env.transition["next_depth_frame"].flatten(),torch.tensor([2.,3.]))
                torch.testing.assert_close(env.policy_depth,torch.tensor([9.,3.]))
                self.assertEqual(env.calls,["forward","sense","reset","forward","sense"])
                self.assertEqual(env.history_updates,1)
                self.assertNotIn("terminal_depth_frame",env.transition)

    def test_legacy_checkpoint_retains_previous_packet_without_extra_sense(self):
        env=make_env(False,terminated=True);self.step(env)
        torch.testing.assert_close(env.transition["next_depth_frame"].flatten(),torch.ones(2))
        self.assertEqual(env.calls,["reset","forward","sense"])


if __name__=="__main__":unittest.main()
