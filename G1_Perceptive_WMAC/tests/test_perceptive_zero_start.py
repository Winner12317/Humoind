"""Zero-state, physical command timing and first-episode scoring contracts."""
from __future__ import annotations

import copy
from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np
import torch

from scripts.evaluate_wmac_zero_start import (
    ScheduledCommand, SuccessThresholds, configure_zero_start,
    initial_state_audit, parser, score_trace, steps_for,
)


class ZeroStartEvaluationTest(unittest.TestCase):
    def test_command_change_uses_native_history_and_never_writes_physical_state(self):
        n = 2
        term = SimpleNamespace(vel_command_b=torch.zeros(n, 3), target_pos_w=torch.zeros(n, 3),
            desired_speed=torch.zeros(n), goal_is_traversal=torch.ones(n, dtype=torch.bool),
            is_in_place_turn=torch.ones(n, dtype=torch.bool), is_heading_env=torch.ones(n, dtype=torch.bool),
            is_standing_env=torch.ones(n, dtype=torch.bool))
        env = SimpleNamespace(step_dt=.1, scene=SimpleNamespace(env_origins=torch.zeros(n, 3)),
            command_manager=SimpleNamespace(get_term=lambda name:term),
            physical_state=torch.tensor([[1., 2., 3.], [4., 5., 6.]]))
        before = env.physical_state.clone()
        clock = ScheduledCommand(env, "start", .2, .5)
        actor = torch.zeros(n, 4, 96)
        clock.assert_observation(actor.flatten(1))
        # A reset cannot sample random commands or provide initial velocity.
        term._resample_command(torch.arange(n))
        torch.testing.assert_close(term.vel_command_b, torch.zeros(n, 3))
        for _ in range(2):
            term.compute(.1)
            actor = torch.cat((actor[:, 1:], torch.zeros(n, 1, 96)), 1)
            actor[:, -1, 6:9] = term.vel_command_b
            clock.assert_observation(actor.flatten(1))
        torch.testing.assert_close(actor[0, :, 6], torch.tensor([0., 0., 0., .5]))
        torch.testing.assert_close(env.physical_state, before, rtol=0, atol=0)
        # A later auto-reset first-episode-inactive env has a refilled history.
        actor[1, :, 6] = .5
        clock.assert_observation(actor.flatten(1), torch.tensor([True, False]))
        with self.assertRaisesRegex(RuntimeError, "command history"):
            clock.assert_observation(actor.flatten(1))

    def test_immediate_start_initial_history_contains_the_real_initial_command(self):
        term = SimpleNamespace(vel_command_b=torch.zeros(1, 3), target_pos_w=torch.zeros(1, 3),
            desired_speed=torch.zeros(1), **{name:torch.zeros(1, dtype=torch.bool) for name in
                ("goal_is_traversal", "is_in_place_turn", "is_heading_env", "is_standing_env")})
        env = SimpleNamespace(step_dt=.02, scene=SimpleNamespace(env_origins=torch.zeros(1, 3)),
            command_manager=SimpleNamespace(get_term=lambda name:term))
        clock = ScheduledCommand(env, "immediate_start", 2., .5)
        actor = torch.zeros(1, 4, 96); actor[:, :, 6] = .5
        clock.assert_observation(actor.flatten(1))
        torch.testing.assert_close(term.vel_command_b, torch.tensor([[.5, 0., 0.]]))
        with self.assertRaisesRegex(ValueError, "once per physical"):
            term.compute(.01)

    def test_only_evaluation_config_changes_and_flat_boundary_matches_actual_geometry(self):
        from src.tasks.amp_perceptive.env_cfg import g1_amp_perceptive_hiking_env_cfg
        from src.wmac_perceptive.config import HIKING_MOTION_PROTOCOL, HIKING_SENSOR_PROTOCOL
        original = g1_amp_perceptive_hiking_env_cfg(play=True,
            motion_scope="forward_walk_turn_v1", sensor_protocol=HIKING_SENSOR_PROTOCOL,
            motion_data_protocol=HIKING_MOTION_PROTOCOL)
        evaluation = configure_zero_start(copy.deepcopy(original), terrain="flat", domain_randomization="retained")
        self.assertEqual(evaluation.events["reset_base"].params, {"pose_range":{}, "velocity_range":{}})
        self.assertEqual(evaluation.events["reset_robot_joints"].params["position_range"], (0., 0.))
        self.assertEqual(evaluation.events["reset_robot_joints"].params["velocity_range"], (0., 0.))
        self.assertTrue(original.events["reset_base"].params["velocity_range"])
        self.assertEqual(original.events["reset_robot_joints"].params["position_range"], (-.15, .15))
        self.assertEqual(evaluation.scene.terrain.terrain_generator.size, (12., 12.))
        self.assertEqual(original.scene.terrain.terrain_generator.size, (8., 8.))
        family = next(iter(evaluation.scene.terrain.terrain_generator.sub_terrains.values()))
        self.assertEqual(family.family, "recovery_flat")
        self.assertEqual(family.wall_probability, 0.)
        self.assertEqual(evaluation.terminations["terrain_boundary"].params["half_extent"], 6.)
        for name in ("foot_friction", "encoder_bias", "base_com"):
            self.assertIn(name, evaluation.events)
        nominal = configure_zero_start(copy.deepcopy(original), terrain="training_spawn", domain_randomization="nominal")
        self.assertEqual(nominal.scene.terrain.terrain_generator.size, (8., 8.))
        for name in ("foot_friction", "encoder_bias", "base_com"):
            self.assertNotIn(name, nominal.events)

    def test_zero_initial_state_is_exact_and_repeated_real_history_is_required(self):
        class Scene(dict):
            pass
        n = 2
        root_default = torch.zeros(n, 13); root_default[:, 2] = .8; root_default[:, 3] = 1.
        joint_default = torch.linspace(-.4, .6, 29).expand(n, -1).clone()
        origins = torch.tensor([[0., 0., 0.], [8., 2., 0.]])
        qpos = torch.cat((root_default[:, :7].clone(), joint_default), -1)
        qpos[:, :3] += origins
        qvel = torch.zeros(n, 35)
        robot = SimpleNamespace(indexing=SimpleNamespace(free_joint_q_adr=torch.arange(7),
            free_joint_v_adr=torch.arange(6), joint_q_adr=torch.arange(7, 36), joint_v_adr=torch.arange(6, 35)),
            data=SimpleNamespace(default_root_state=root_default, default_joint_pos=joint_default,
                data=SimpleNamespace(qpos=qpos, qvel=qvel)))
        history = torch.zeros(n, 4, 96)
        # Physical DR encoder biases may appear in observations, but must be
        # repeated from the true first frame, never substituted by zeros.
        history[:, :, 9:38] = .01
        scene = Scene(robot=robot); scene.env_origins = origins
        expected_frame=history[:,0].clone()
        adapter=SimpleNamespace(state=lambda:torch.zeros(n,71),context=lambda:torch.zeros(n,35),
            command=lambda:torch.zeros(n,3),actor_frame=lambda *args,**kwargs:expected_frame)
        env = SimpleNamespace(num_envs=n, scene=scene, adapter=adapter,
            action_manager=SimpleNamespace(action=torch.zeros(n, 29)), obs_buf={"actor":history.flatten(1)})
        frame = torch.rand(n, 2, 2, 3)
        controller = SimpleNamespace(history=SimpleNamespace(buffer=frame[:, None].expand(n, 4, 2, 2, 3).clone()))
        audit, raw = initial_state_audit(env, controller, frame)
        self.assertTrue(all(value == 0. for value in audit.values()))
        qvel[0, 0] = 1e-9
        with self.assertRaisesRegex(RuntimeError, "raw_qvel_max_abs"):
            initial_state_audit(env, controller, frame)
        qvel.zero_(); controller.history.buffer[0, 0, 0, 0, 0] += .1
        with self.assertRaisesRegex(RuntimeError, "depth_history_repeat"):
            initial_state_audit(env, controller, frame)

    def test_full_zero_start_removes_source_height_reset_and_nominal_material_randomizer(self):
        from src.tasks.amp_perceptive.env_cfg import g1_amp_perceptive_hiking_env_cfg
        cfg=g1_amp_perceptive_hiking_env_cfg(play=True,sensor_protocol="hiking_depth_only_v1",
            motion_data_protocol="hiking_walk_v1",locomotion_objective="hiking_full_v1")
        self.assertIn("reset_hiking_state",cfg.events)
        self.assertIn("physics_material",cfg.events)
        configure_zero_start(cfg,terrain="flat",domain_randomization="nominal")
        self.assertNotIn("reset_hiking_state",cfg.events)
        self.assertNotIn("physics_material",cfg.events)
        self.assertEqual(cfg.events["reset_base"].params,{"pose_range":{},"velocity_range":{}})
        self.assertEqual(cfg.events["reset_robot_joints"].params["position_range"],(0.,0.))
        terrain=next(iter(cfg.scene.terrain.terrain_generator.sub_terrains.values()))
        self.assertEqual(terrain.task_protocol,"hiking_full_v1")
        self.assertEqual(terrain.family,"recovery_flat")

    def test_full_eight_frame_schedule_keeps_standing_mode_and_ignores_random_navigation(self):
        from src.wmac_perceptive.hiking_observations import pack_hiking_proprio_history
        term=SimpleNamespace(vel_command_b=torch.zeros(1,3),target_pos_w=torch.zeros(1,3),
            desired_speed=torch.zeros(1),random_yaw_rate=torch.ones(1),
            random_velocity_mask=torch.ones(1,dtype=torch.bool),
            **{name:torch.zeros(1,dtype=torch.bool) for name in
                ("goal_is_traversal","is_in_place_turn","is_heading_env","is_standing_env")})
        env=SimpleNamespace(step_dt=.1,scene=SimpleNamespace(env_origins=torch.zeros(1,3)),
                            command_manager=SimpleNamespace(get_term=lambda name:term))
        schedule=ScheduledCommand(env,"start",.2,.5)
        history=torch.zeros(1,8,96)
        schedule.assert_observation(pack_hiking_proprio_history(history))
        self.assertTrue(bool(term.is_standing_env.all()))
        self.assertFalse(bool(term.random_velocity_mask.any()))
        for _ in range(2):
            schedule.compute(.1)
            history=torch.cat((history[:,1:],torch.zeros(1,1,96)),1)
            history[:,-1,6:9]=term.vel_command_b
        schedule.assert_observation(pack_hiking_proprio_history(history))
        self.assertFalse(bool(term.is_standing_env.any()))
        torch.testing.assert_close(history[0,:,6],torch.tensor([0.,0.,0.,0.,0.,0.,0.,.5]))

    @staticmethod
    def trace(steps, n):
        initial = np.zeros((n, 71), np.float32)
        initial[:, 2] = .8; initial[:, 3] = 1.
        return initial, {"next_state":np.repeat(initial[None], steps, 0),
            "first_episode_valid":np.ones((steps, n), bool),
            "terminated":np.zeros((steps, n), bool), "timeout":np.zeros((steps, n), bool),
            "up_z":np.ones((steps, n), np.float32), "relative_height":np.full((steps, n), .8, np.float32)}

    def test_standing_separately_rejects_drift_speed_and_tilt_even_without_fall(self):
        initial, trace = self.trace(10, 4)
        trace["next_state"][:, 1, 0] = .25
        trace["next_state"][:, 2, 7] = .2
        trace["up_z"][:, 3] = math_cos_degrees(25.)
        result = score_trace(trace, initial, case="stand", dt=.1, wait_seconds=2.,
            thresholds=SuccessThresholds(), bad_height=.5, bad_up_z=.5)
        self.assertEqual(result["first_episode_survival_fraction"], 1.)
        self.assertEqual(result["fall_episode_fraction"], 0.)
        self.assertEqual([row["success"] for row in result["per_env"]], [True, False, False, False])

    def test_terminal_successor_is_counted_and_auto_reset_cannot_rescue_start(self):
        initial, trace = self.trace(10, 2)
        trace["next_state"][2:, :, 7] = .5
        trace["next_state"][:, :, 0] = np.cumsum(trace["next_state"][:, :, 7], axis=0)*.1
        trace["terminated"][3, 1] = True
        trace["relative_height"][3, 1] = .2
        trace["first_episode_valid"][4:, 1] = False
        trace["next_state"][4:, 1, 0] = 100.
        thresholds = replace(SuccessThresholds(), start_min_forward_displacement_m=.1, final_speed_window_seconds=.2)
        result = score_trace(trace, initial, case="start", dt=.1, wait_seconds=.2,
            thresholds=thresholds, bad_height=.5, bad_up_z=.5)
        good, bad = result["per_env"]
        self.assertTrue(good["success"])
        self.assertAlmostEqual(good["start_forward_displacement_m"], .4, places=6)
        self.assertAlmostEqual(good["reach_speed_seconds"], .3)
        self.assertEqual(bad["observed_steps"], 4)
        self.assertTrue(bad["fallen"])
        self.assertFalse(bad["success"])
        self.assertLess(bad["start_forward_displacement_m"], 1.)
        self.assertIsNone(bad["final_forward_speed_m_s"])

    def test_yaw_is_reported_as_diagnostic_without_changing_fixed_v1_criteria(self):
        initial, trace = self.trace(10, 1)
        trace["next_state"][:, 0, 12] = .3
        result = score_trace(trace, initial, case="stand", dt=.1, wait_seconds=2.,
            thresholds=SuccessThresholds(), bad_height=.5, bad_up_z=.5)
        row = result["per_env"][0]
        self.assertAlmostEqual(row["stand_mean_abs_yaw_rate_rad_s"], .3, places=6)
        self.assertTrue(row["standing_window_complete"])
        self.assertTrue(row["success"])

    def test_clock_rejects_fractional_control_steps_and_cli_defaults_are_explicit(self):
        self.assertEqual(steps_for(2., .02), 100)
        with self.assertRaisesRegex(ValueError, "integer multiples"):
            steps_for(.025, .02)
        args = parser().parse_args(["--checkpoint", "actor.pt"])
        self.assertEqual(args.cases, ["stand", "start", "immediate_start"])
        self.assertEqual(args.terrain, "flat")
        self.assertEqual(args.domain_randomization, "retained")
        self.assertEqual(args.num_envs, 32)


def math_cos_degrees(degrees):
    return float(np.cos(np.radians(degrees)))


if __name__ == "__main__":
    unittest.main()
