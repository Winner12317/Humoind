"""Replay sampling distributions, held-out isolation and non-finite quarantine."""
from __future__ import annotations

import unittest

import torch

from src.wmac.replay import ReplayBuffer


def batch(time: int, envs: int = 10) -> dict[str, torch.Tensor]:
    state = torch.zeros(envs, 71)
    state[:, 0] = torch.arange(envs)
    state[:, 1] = time
    state[:, 2] = torch.where((torch.arange(envs) + time) % 3 == 0, .3, .8)
    state[:, 3] = 1
    return {
        "state": state,
        "next_state": state.clone() + .001,
        "context": torch.zeros(envs, 35),
        "action": torch.full((envs, 29), float(time)),
        "reward": torch.ones(envs),
        "terminated": torch.zeros(envs, dtype=torch.bool),
        "timeout": torch.zeros(envs, dtype=torch.bool),
        "boundary": torch.zeros(envs, dtype=torch.bool),
    }


class FastReplayTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261106)

    def test_environment_split_stays_disjoint_after_ring_wrap(self):
        replay = ReplayBuffer(4, 10, validation_env_fraction=.2)
        for time in range(7):
            replay.add(batch(time))
        self.assertEqual(replay.num_valid("train"), 32)
        self.assertEqual(replay.num_valid("validation"), 8)
        self.assertEqual(replay.num_valid("all"), 40)
        torch.testing.assert_close(replay.train_env_mask, torch.arange(10) < 8)
        torch.testing.assert_close(replay.validation_env_mask, torch.arange(10) >= 8)
        for split, low, high in (("train", 0, 8), ("validation", 8, 10)):
            sample = replay.sample_starts(512, fields=("state",), split=split)
            self.assertTrue(((sample["env_index"] >= low) & (sample["env_index"] < high)).all())
            torch.testing.assert_close(sample["state"][:, 0], sample["env_index"].float())
            window = replay.sample_windows(128, 3, fields=("state",), split=split)
            self.assertTrue(((window["env_index"] >= low) & (window["env_index"] < high)).all())
            torch.testing.assert_close(window["env_index"], window["env_index"][:, :1].expand(-1, 3))
            torch.testing.assert_close(window["time_index"][:, 1:] - window["time_index"][:, :-1], torch.ones(128, 2, dtype=torch.long))
            torch.testing.assert_close(window["state"][:, :, 0], window["env_index"].float())

    def test_selective_fields_keep_hidden_boundary_mask_and_terminal_labels(self):
        replay = ReplayBuffer(4, 2)
        for time in range(4):
            item = batch(time, 2)
            if time == 1:
                item["terminated"][0] = True
                item["next_state"][0, 2] = .123  # Actual pre-reset terminal height.
            replay.add(item)
        sample = replay.sample_windows(128, 4, fields=("next_state",), recovery_fraction=0.)
        self.assertEqual(set(sample), {"next_state", "mask", "time_index", "env_index"})
        selected = sample["env_index"][:, 0] == 0
        self.assertTrue(selected.any())
        torch.testing.assert_close(sample["next_state"][selected, 1, 2], torch.full((int(selected.sum()),), .123))
        torch.testing.assert_close(sample["mask"][selected], torch.tensor([1., 1., 0., 0.]).expand(int(selected.sum()), -1))
        self.assertEqual(set(replay.sample_starts(4, fields=())), {"time_index", "env_index"})
        with self.assertRaises(KeyError):
            replay.sample_starts(4, fields=("missing",))

    def test_nonfinite_transition_is_quarantined_without_rewriting_storage(self):
        replay = ReplayBuffer(4, 2)
        for time in range(4):
            item = batch(time, 2)
            if time == 1:
                item["next_state"][0, 2] = .123
            if time == 2:
                item["next_state"][0, 0] = float("nan")
            if time == 3:
                item["reward"][1] = float("inf")
            valid = replay.add(item)
            if time == 2:
                torch.testing.assert_close(valid, torch.tensor([False, True]))
        self.assertEqual(replay.invalid_transitions, 2)
        self.assertEqual(replay.num_valid(), 6)
        self.assertTrue(torch.isnan(replay.data["next_state"][2, 0, 0]))
        self.assertTrue(torch.isinf(replay.data["reward"][3, 1]))
        starts = replay.sample_starts(1000, fields=("next_state", "reward"), recovery_fraction=0.)
        invalid_start = ((starts["time_index"] == 2) & (starts["env_index"] == 0)) | ((starts["time_index"] == 3) & (starts["env_index"] == 1))
        self.assertFalse(invalid_start.any())
        for key in ("next_state", "reward"):
            self.assertTrue(torch.isfinite(starts[key]).all())
        windows = replay.sample_windows(128, 4, fields=("next_state", "reward"), recovery_fraction=0.)
        zero = windows["env_index"][:, 0] == 0
        one = ~zero
        torch.testing.assert_close(windows["mask"][zero], torch.tensor([1., 1., 0., 0.]).expand(int(zero.sum()), -1))
        torch.testing.assert_close(windows["mask"][one], torch.tensor([1., 1., 1., 0.]).expand(int(one.sum()), -1))
        torch.testing.assert_close(windows["next_state"][zero, 1, 2], torch.full((int(zero.sum()),), .123))
        for key in ("next_state", "reward"):
            self.assertTrue(torch.isfinite(windows[key]).all())

    def test_empty_train_or_validation_pool_never_leaks_across_the_split(self):
        replay = ReplayBuffer(2, 4, validation_env_fraction=.5)
        item = batch(0, 4)
        item["action"][:2] = float("inf")
        replay.add(item)
        self.assertEqual(replay.num_valid("train"), 0)
        self.assertEqual(replay.num_valid("validation"), 2)
        with self.assertRaisesRegex(ValueError, "No finite replay transitions"):
            replay.sample_starts(8, split="train")
        validation = replay.sample_starts(32, split="validation")
        self.assertTrue((validation["env_index"] >= 2).all())
        no_validation = ReplayBuffer(2, 2)
        no_validation.add(batch(0, 2))
        self.assertEqual(no_validation.num_valid("validation"), 0)
        with self.assertRaisesRegex(ValueError, "No finite replay transitions"):
            no_validation.sample_starts(8, split="validation")

    def test_class_sampling_is_transition_uniform_even_when_row_counts_differ(self):
        replay = ReplayBuffer(2, 10)
        first, second = batch(0), batch(1)
        first["state"][:, 2] = .8
        first["state"][0, 2] = .3
        second["state"][:, 2] = .3
        replay.add(first); replay.add(second)
        sample = replay.sample_starts(11000, recovery_fraction=1., fields=("state",))
        self.assertTrue((sample["state"][:, 2] < .55).all())
        # First row has 1 recovery transition, second row 10. Uniform by row
        # would incorrectly allocate half of the batch to the first row.
        fraction_first = float((sample["time_index"] == 0).float().mean())
        self.assertLess(abs(fraction_first - 1/11), .02)

    def test_prefixes_are_reused_between_many_samples_and_incrementally_invalidated(self):
        replay = ReplayBuffer(8, 10, validation_env_fraction=.2)
        for time in range(8):
            replay.add(batch(time))
        replay.sample_starts(16)
        replay.sample_windows(16, 3)
        initial_builds = replay.index_cache_builds
        for _ in range(20):
            replay.sample_starts(16)
            replay.sample_windows(16, 3)
        self.assertEqual(replay.index_cache_builds, initial_builds)
        replay.add(batch(8))
        replay.sample_starts(16)
        self.assertEqual(replay.index_cache_builds, initial_builds + 1)
        replay.sample_starts(16)
        self.assertEqual(replay.index_cache_builds, initial_builds + 1)

    def test_restore_rebuilds_finite_split_indices_and_automatically_cuts_the_tail(self):
        replay = ReplayBuffer(5, 10, validation_env_fraction=.2)
        for time in range(8):
            item = batch(time)
            if time == 7:
                item["next_state"][9, 0] = float("nan")
            replay.add(item)
        restored = ReplayBuffer(5, 10, validation_env_fraction=.2)
        restored.load_state_dict(replay.state_dict())
        self.assertEqual(restored.num_valid("train"), 40)
        self.assertEqual(restored.num_valid("validation"), 9)
        restored.add(batch(8)); restored.add(batch(9))
        sample = restored.sample_windows(128, 4, split="validation", fields=("next_state",))
        self.assertTrue((sample["env_index"] >= 8).all())
        self.assertTrue(torch.isfinite(sample["next_state"]).all())
        for times, mask in zip(sample["time_index"], sample["mask"]):
            i = int((times == 7).nonzero()[0])
            self.assertFalse(mask[i + 1:].any())

    def test_legacy_checkpoint_load_rebuilds_the_new_split(self):
        replay = ReplayBuffer(4, 4)
        for time in range(4):
            replay.add(batch(time, 4))
        legacy = {k: v for k, v in replay.state_dict().items()
                  if k not in ("version", "valid", "validation_env_fraction", "invalid_transitions")}
        restored = ReplayBuffer(4, 4, validation_env_fraction=.25)
        restored.load_state_dict(legacy)
        self.assertEqual(restored.num_valid("train"), 12)
        self.assertEqual(restored.num_valid("validation"), 4)
        validation = restored.sample_starts(128, split="validation")
        self.assertTrue((validation["env_index"] == 3).all())

    def test_invalid_batch_shape_does_not_advance_or_partially_overwrite_the_ring(self):
        replay = ReplayBuffer(2, 2)
        replay.add(batch(0, 2))
        before = replay.data["state"][0].clone()
        item = batch(1, 2)
        item["reward"] = torch.ones(2, 1)
        with self.assertRaisesRegex(ValueError, "changed shape"):
            replay.add(item)
        self.assertEqual((replay.ptr, replay.size, replay.total_steps), (1, 1, 1))
        torch.testing.assert_close(replay.data["state"][0], before)


if __name__ == "__main__":
    unittest.main()
