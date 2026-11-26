"""CPU-only checks for calibration counter semantics and comparison reporting."""
import argparse
import unittest

import numpy as np

from scripts.measure_perceptive_sim_capacity import (
    Candidate, apply_grid_assignment, compare_traces, generated_action, grid_assignment,
    initial_capacity_requirement, select_snapshots, sorted_contact_rows, state71_to_qpos_qvel,
    summarize_counters, warning_counts,
)


class SimCapacityToolsTest(unittest.TestCase):
    def test_four_column_source_candidate(self):
        self.assertEqual(Candidate.parse("128:700:128:128"), Candidate(128, 700, 128, 128))
        for bad in ("128:700", "0:700:100", "128:700:100:0", "a:b:c"):
            with self.assertRaises(argparse.ArgumentTypeError): Candidate.parse(bad)

    def test_host_initial_contact_requirement_is_recorded_without_bypass(self):
        self.assertEqual(initial_capacity_requirement("nconmax overflow (nconmax must be >= 172)"), 172)
        self.assertIsNone(initial_capacity_requirement("naconmax overflow (naconmax must be >= 12345)"))

    def test_shared_contact_pool_is_not_a_single_world_cap(self):
        result = summarize_counters(80, [10, 20], 90, [70, 10], 128, 100)
        self.assertFalse(result["contact_pool_overflow"])
        self.assertEqual(result["contacts_per_world_max"], 70)
        self.assertFalse(result["constraint_at_or_over_capacity"])
        result = summarize_counters(129, [100, 101], 130, [70, 58], 128, 100)
        self.assertTrue(result["contact_pool_overflow"])
        self.assertTrue(result["broadphase_pool_overflow"])
        self.assertTrue(result["constraint_overflow"])
        self.assertTrue(summarize_counters(1, [100], 1, [1], 128, 100)["constraint_at_or_over_capacity"])

    def test_heightfield_and_workspace_warnings_stay_distinct(self):
        result = warning_counts("height field collision overflow >= 50\nCCD overflow\ncontact match overflow\n")
        self.assertEqual(result["heightfield_pair_overflow"], 1)
        self.assertEqual(result["ccd_workspace_overflow"], 1)
        self.assertEqual(result["contact_sensor_match_overflow"], 1)

    def test_scripted_actions_are_fixed_and_bounded(self):
        left = generated_action(41, 8, "scripted")
        np.testing.assert_array_equal(left, generated_action(41, 8, "scripted"))
        self.assertEqual(left.shape, (8, 29))
        self.assertLessEqual(float(np.abs(left).max()), .300001)
        self.assertFalse(np.array_equal(left, generated_action(42, 8, "scripted")))
        self.assertFalse(generated_action(41, 8, "zero").any())

    def test_trace_comparison_exposes_first_divergence_and_termination(self):
        reference = {key: np.zeros((3, 2, 4)) for key in ("initial_qpos", "initial_qvel", "qpos", "qvel", "reward")}
        reference.update(terminated=np.zeros((3, 2), dtype=bool), timeout=np.zeros((3, 2), dtype=bool))
        other = {key: value.copy() for key, value in reference.items()}
        other["qpos"][1, 0, 0] = .01
        other["terminated"][2, 1] = True
        result = compare_traces(reference, other, atol=1e-5, rtol=1e-5)
        self.assertFalse(result["qpos"]["allclose"])
        self.assertEqual(result["qpos"]["first_differing_step"], 1)
        self.assertEqual(result["terminated"]["first_differing_step"], 2)
        self.assertTrue(result["qvel"]["allclose"])

    def test_snapshot_selection_is_uniform_and_preserves_parallel_worlds(self):
        trace = {"qpos": np.arange(10*3*7).reshape(10, 3, 7).astype(float),
            "qvel": np.arange(10*3*6).reshape(10, 3, 6).astype(float),
            "initial_qpos": np.zeros((3, 7)), "initial_qvel": np.zeros((3, 6))}
        indices, position, velocity = select_snapshots(trace, 3, 3)
        np.testing.assert_array_equal(indices, [-1, 4, 9])
        self.assertEqual(position.shape, (3, 3, 7))
        np.testing.assert_array_equal(velocity[-1], trace["qvel"][-1])
        with self.assertRaises(ValueError): select_snapshots(trace, 3, 2)
        trace["qpos"][-1, 0, 0] = float("nan")
        with self.assertRaises(ValueError): select_snapshots(trace, 3, 3)

    def test_contact_rows_sort_by_world_geom_and_position(self):
        result = sorted_contact_rows([1, 0, 0], [[4, 5], [2, 3], [2, 3]],
            [[0, 0, 0], [1, 0, 0], [0, 0, 0]], [-.1, -.2, -.3])
        np.testing.assert_array_equal(result[:, :3], [[0, 2, 3], [0, 2, 3], [1, 4, 5]])
        np.testing.assert_array_equal(result[:2, 3], [0, 1])
        self.assertEqual(sorted_contact_rows([], np.empty((0, 2)), np.empty((0, 3)), []).shape, (0, 7))

    def test_frozen_comparison_handles_different_contact_counts(self):
        reference = {"qacc": np.zeros((3, 2, 6)), "nefc": np.zeros((3, 2)),
            "contacts_case__0000": np.zeros((5, 7))}
        other = {key: value.copy() for key, value in reference.items()}
        other["contacts_case__0000"] = np.zeros((4, 7))
        result = compare_traces(reference, other, atol=1e-5, rtol=1e-5)
        self.assertTrue(result["qacc"]["allclose"])
        self.assertFalse(result["contacts_case__0000"]["same_shape"])

    def test_grid_assigns_every_existing_cell_and_keeps_shared_origins(self):
        import torch
        from types import SimpleNamespace
        levels, kinds = grid_assignment(256, 10, 20)
        self.assertEqual(len(set(zip(levels, kinds))), 200)
        old_types = torch.div(torch.arange(256), 256/20, rounding_mode="floor").long()
        terrain = SimpleNamespace(terrain_origins=torch.arange(10*20*3).reshape(10, 20, 3).float(),
            terrain_levels=torch.zeros(256, dtype=torch.long), terrain_types=old_types.clone(),
            env_origins=torch.zeros(256, 3))
        command = SimpleNamespace(_hiking_speed_low=old_types.float(),
            _hiking_speed_high=old_types.float()+1, random_velocity_mask=old_types==1)
        env = SimpleNamespace(num_envs=256, device="cpu",
            scene=SimpleNamespace(terrain=terrain, env_origins=terrain.env_origins),
            command_manager=SimpleNamespace(get_term=lambda _: command))
        pointer = terrain.env_origins.data_ptr()
        apply_grid_assignment(env)
        self.assertEqual(pointer, terrain.env_origins.data_ptr())
        np.testing.assert_array_equal(terrain.terrain_levels.numpy(), levels)
        np.testing.assert_array_equal(terrain.terrain_types.numpy(), kinds)
        np.testing.assert_array_equal(command._hiking_speed_low.numpy(), kinds)
        np.testing.assert_array_equal(command.random_velocity_mask.numpy(), kinds==1)
        torch.testing.assert_close(terrain.env_origins, terrain.terrain_origins[levels, kinds])

    def test_pre_reset_state_converts_world_angular_velocity_to_local(self):
        state = np.zeros((2, 3, 71), dtype=float)
        state[..., 3] = np.sqrt(.5); state[..., 6] = np.sqrt(.5)
        state[..., 7:10] = [2, 3, 4]
        state[..., 10:13] = [1, 0, 0]
        state[..., 13:42] = np.arange(29)
        state[..., 42:71] = np.arange(29)+100
        qpos, qvel = state71_to_qpos_qvel(state)
        self.assertEqual(qpos.shape, (2, 3, 36))
        self.assertEqual(qvel.shape, (2, 3, 35))
        np.testing.assert_allclose(qvel[..., 3:6], np.broadcast_to([0, -1, 0], (2, 3, 3)), atol=1e-12)
        np.testing.assert_array_equal(qvel[..., :3], state[..., 7:10])
        np.testing.assert_array_equal(qpos[..., 7:], state[..., 13:42])
        indices, actual_qpos, actual_qvel = select_snapshots({"next_state": state}, 2, 3, "pre-reset")
        np.testing.assert_array_equal(indices, [0, 1])
        np.testing.assert_array_equal(actual_qpos, qpos)
        np.testing.assert_array_equal(actual_qvel, qvel)


if __name__ == "__main__": unittest.main()
