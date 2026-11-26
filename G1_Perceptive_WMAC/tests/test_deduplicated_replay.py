"""Lossless storage against the existing full-history replay implementation."""
from __future__ import annotations

import copy
from dataclasses import replace
import unittest

import torch

from src.wmac_perceptive.compact_gpu_replay import CompressedLatentReplay
from src.wmac_perceptive.deduplicated_replay import DeduplicatedHikingReplay
from src.wmac_perceptive.hiking_amp import configure_replay
from src.wmac_perceptive.hiking_context import advance_hiking_context
from src.wmac_perceptive.hiking_observations import pack_hiking_proprio_history
from src.wmac_perceptive.perception import PerceptionSpec


SPEC = PerceptionSpec(height=2, width=3, ring_length=4, frames=2, stride=2, delay=1,
    proprio_dim=768, crop_top=0, crop_bottom=0, crop_left=0, crop_right=0,
    blur_kernel=1, include_valid=False)


class Stream:
    def __init__(self, n=4, seed=701, unusual=True):
        self.n, self.unusual = n, unusual
        self.rng = torch.Generator().manual_seed(seed)
        self.frames = self.rand(n, 8, 96)  # Deliberately non-repeated initial history.
        self.images = self.rand(n, 4, 1, 2, 3)
        self.context = torch.zeros(n, 100)
        self.context[:, 0] = self.rand(n)
        self.context[:, 35:38] = self.rand(n, 3)
        self.context[:, 67:96] = self.rand(n, 29)
        self.time = 0

    def rand(self, *shape):
        return torch.rand(*shape, generator=self.rng)

    def reset_all(self):
        self.frames = self.rand(self.n, 1, 96).expand(-1, 8, -1).clone()
        self.images = self.rand(self.n, 1, 1, 2, 3).expand(-1, 4, -1, -1, -1).clone()
        self.context[:, 35:38] = self.rand(self.n, 3)
        self.context[:, 67:96] = self.rand(self.n, 29)

    def batch(self):
        n, step = self.n, self.time
        state = self.rand(n, 71)
        state[:, 2] = torch.tensor([.8, .4, .8, .4])[:n]
        state[:, 0] += step
        action, auxiliary = self.rand(n, 29), self.rand(n, 13)
        previous = self.context[:, 38:67].clone()
        next_context = advance_hiking_context(self.context, action, auxiliary)
        terminal = torch.zeros(n, dtype=torch.bool)
        timeout = terminal.clone(); boundary = terminal.clone()
        if self.unusual:
            if step in (1, 11, 21): terminal[0] = True
            if step in (4, 14): timeout[1] = True
            if step in (2, 12): boundary[2] = True
            if step == 3: state[1, 0] = float('nan')
            if step == 5: next_context[0, 12] += .025
            if step == 6: previous[2, 4] += .03
            if step == 7: self.frames[1, 2, 17] += .1
            if step == 8: self.frames[0, 3, 5] = float('nan')
            if step == 9: self.context[3, 0] += .3
            # Nonzero clock tests exact legacy sampling classification.
            self.context[:, 34] = float(step % 3)
            next_context[:, 34] = self.context[:, 34]
        next_images = torch.cat((self.images[:, 1:], self.rand(n, 1, 1, 2, 3)), 1)
        batch = dict(state=state, next_state=self.rand(n, 71), context=self.context.clone(),
            next_context=next_context.clone(), command=self.rand(n, 3), previous_action=previous,
            actor_history=pack_hiking_proprio_history(self.frames), reward=self.rand(n),
            contact_cost=auxiliary, action=action, navigation_target=self.rand(n, 3),
            desired_speed=self.rand(n), relative_height=self.rand(n), random_yaw_rate=self.rand(n),
            terminated=terminal, timeout=timeout, boundary=boundary,
            command_standing=torch.zeros(n, dtype=torch.bool), command_boundary=boundary.clone(),
            random_velocity_mask=torch.zeros(n, dtype=torch.bool),
            depth_stack=self.images[:, SPEC.selected_indices].flatten(1, 2),
            next_depth_stack=next_images[:, SPEC.selected_indices].flatten(1, 2))
        self.frames = torch.cat((self.frames[:, 1:], self.rand(n, 1, 96)), 1)
        self.images = next_images
        self.context = next_context.clone()
        done = terminal | timeout
        if done.any():
            self.frames[done] = self.rand(int(done.sum()), 1, 96).expand(-1, 8, -1)
            self.images[done] = self.rand(int(done.sum()), 1, 1, 2, 3).expand(-1, 4, -1, -1, -1)
            self.context[done, 35:38] = self.rand(int(done.sum()), 3)
            self.context[done, 67:96] = self.rand(int(done.sum()), 29)
        self.time += 1
        return batch


def make_pair(capacity=9, unusual=True, amp=True):
    stream = Stream(unusual=unusual)
    compact = DeduplicatedHikingReplay(capacity, stream.n, 'cpu', .25, spec=SPEC)
    reference = CompressedLatentReplay(capacity, stream.n, 'cpu', .25, spec=SPEC)
    for replay in (compact, reference):
        replay.bootstrap_processed_history(stream.images)
        if amp: configure_replay(replay)
    return compact, reference, stream


class DeduplicatedReplayTests(unittest.TestCase):
    def assert_oracle(self, compact, reference):
        for split in ('train', 'validation', 'all'):
            for recovery in (0., .5):
                for windows in (False, True):
                    for seed in (71, 32):
                        torch.manual_seed(seed)
                        actual = (compact.sample_windows(37, min(3, compact.size), recovery, split=split)
                                  if windows else compact.sample_starts(37, recovery, split=split))
                        torch.manual_seed(seed)
                        expected = (reference.sample_windows(37, min(3, reference.size), recovery, split=split)
                                    if windows else reference.sample_starts(37, recovery, split=split))
                        self.assertEqual(actual.keys(), expected.keys())
                        for name in actual:
                            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0,
                                                       equal_nan=True, msg=lambda m: name+': '+m)

    def test_exact_sample_oracle_with_noise_resets_boundaries_and_wrap(self):
        compact, reference, stream = make_pair()
        for step in range(29):
            if step == 17:
                compact.mark_boundary(); reference.mark_boundary()
                stream.reset_all()
            batch = stream.batch()
            torch.testing.assert_close(compact.add(batch), reference.add(batch))
            if step in (0, 5, 8, 9, 18, 28): self.assert_oracle(compact, reference)
        report = compact.storage_report()
        self.assertGreater(report['sparse_exceptions']['history']['total_entries'], 0)
        self.assertGreater(report['sparse_exceptions']['next_context']['total_entries'], 0)
        self.assertGreater(report['sparse_exceptions']['previous_action']['total_entries'], 0)

    def test_empty_and_populated_restore_then_new_simulator_data(self):
        compact, reference, stream = make_pair()
        empty = DeduplicatedHikingReplay(9, 4, 'cpu', .25, spec=SPEC)
        empty.load_state_dict(copy.deepcopy(compact.state_dict()))
        for _ in range(19):
            batch = stream.batch(); compact.add(batch); reference.add(batch)
        saved = copy.deepcopy(compact.state_dict())
        restored = DeduplicatedHikingReplay(9, 4, 'cpu', .25, spec=SPEC)
        restored.load_state_dict(saved)
        # Model restore starts a new simulator: preserve old replay, cut tail,
        # then start new camera/AMP histories, exactly as existing replay does.
        reference.mark_boundary()
        reference._pending_reset.fill_(True)
        self.assert_oracle(restored, reference)
        stream.reset_all()
        for _ in range(13):
            batch = stream.batch()
            torch.testing.assert_close(restored.add(batch), reference.add(batch))
        self.assert_oracle(restored, reference)

    def test_unexpected_history_after_mark_boundary_is_preserved(self):
        compact, reference, stream = make_pair(unusual=False)
        for _ in range(6):
            batch = stream.batch(); compact.add(batch); reference.add(batch)
        compact.mark_boundary(); reference.mark_boundary()
        # No reset in this fixture: clamping is not allowed to change values.
        batch = stream.batch(); compact.add(batch); reference.add(batch)
        self.assert_oracle(compact, reference)
        self.assertEqual(compact.storage_report()['sparse_exceptions']['history']['current_entries'], 4)

    def test_context_pool_reuses_live_slots_without_unbounded_growth(self):
        compact, reference, stream = make_pair(capacity=3, unusual=False)
        for step in range(120):
            stream.context[:, :33] = stream.rand(4, 33)
            stream.context[:, 35:38] = stream.rand(4, 3)
            batch = stream.batch(); compact.add(batch); reference.add(batch)
        self.assert_oracle(compact, reference)
        report = compact.storage_report()
        self.assertLessEqual(report['context_live_events'], 4*4)
        self.assertLessEqual(report['context_event_capacity'], 32)

    def test_ordinary_history_stays_sparse_free_and_uses_less_memory(self):
        compact, reference, stream = make_pair(capacity=128, unusual=False)
        for _ in range(160):
            batch = stream.batch(); compact.add(batch); reference.add(batch)
        self.assert_oracle(compact, reference)
        report = compact.storage_report()
        for value in report['sparse_exceptions'].values():
            self.assertEqual(value['current_entries'], 0)
            self.assertEqual(value['total_entries'], 0)
        self.assertEqual(report['context_total_events'], 4)
        self.assertLess(report['unique_tensor_storage_bytes'], report['original_layout_estimate_bytes']*.5)

    def test_old_layout_and_corrupt_reference_checkpoint_are_rejected(self):
        compact, reference, stream = make_pair()
        with self.assertRaisesRegex(ValueError, 'explicit conversion'):
            compact.load_state_dict(reference.state_dict())
        batch = stream.batch(); compact.add(batch)
        saved = copy.deepcopy(compact.state_dict())
        saved['context_events']['references'][0] += 1
        with self.assertRaisesRegex(ValueError, 'reference counts'):
            compact.load_state_dict(saved)

    def test_field_subsets_and_invalid_schema_fail_before_row_changes(self):
        compact, reference, stream = make_pair()
        batch = stream.batch(); compact.add(batch); reference.add(batch)
        fields = ('previous_action', 'next_context', 'context', 'actor_history')
        torch.manual_seed(5); actual = compact.sample_starts(11, fields=fields)
        torch.manual_seed(5); expected = reference.sample_starts(11, fields=fields)
        for name in actual: torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        count = compact.total_steps
        invalid = stream.batch(); invalid['context'] = invalid['context'][:, :99]
        with self.assertRaisesRegex(ValueError, 'context'):
            compact.add(invalid)
        self.assertEqual(count, compact.total_steps)

    def test_recovery_uses_terrain_relative_height_not_world_z(self):
        compact, reference, stream = make_pair(unusual=False)
        batch = stream.batch()
        batch['state'][:, 2] = torch.tensor([4., .2, 3., .1])
        batch['relative_height'] = torch.tensor([.2, .8, .3, .9])
        compact.add(batch); reference.add(batch)
        torch.testing.assert_close(compact._recovery[0], torch.tensor([True, False, True, False]))
        self.assert_oracle(compact, reference)

    def test_signed_zero_and_input_float_conversion_are_exact(self):
        compact, reference, stream = make_pair(unusual=False)
        for step in range(11):
            batch = stream.batch()
            if step % 2:
                batch['actor_history'] = batch['actor_history'].half()
                batch['relative_height'] = batch['relative_height'].half()
            batch['actor_history'][0, step % 24] = -0.
            compact.add(batch); reference.add(batch)
        self.assert_oracle(compact, reference)
        rows, envs = torch.tensor([compact.ptr-1]), torch.tensor([0])
        a = compact._gather_fields(rows, envs, ('actor_history',))['actor_history']
        b = reference._gather_fields(rows, envs, ('actor_history',))['actor_history']
        torch.testing.assert_close(a.view(torch.int32), b.view(torch.int32))

    def test_malformed_internal_layout_is_not_silently_reinterpreted(self):
        compact, _, stream = make_pair()
        compact.add(stream.batch())
        for kind in ('width', 'missing', 'event', 'amp'):
            saved = copy.deepcopy(compact.state_dict())
            if kind == 'width': saved['data']['context'] = torch.zeros(9, 4, 100)
            elif kind == 'missing': saved['data'].pop('contact_cost')
            elif kind == 'event': saved['context_events']['ids'][0, 1] = -1
            else: saved['hiking_amp_state'].pop('_amp_episode_starts')
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                compact.load_state_dict(saved)

    def test_observation_finiteness_is_checked_after_original_fp32_cast(self):
        compact, reference, stream = make_pair(unusual=False)
        batch = stream.batch()
        batch['actor_history'] = batch['actor_history'].double()
        # Only an older frame overflows: the compressed newest frame is finite.
        batch['actor_history'][0, 0] = 1e100
        batch['relative_height'] = batch['relative_height'].double()
        batch['relative_height'][1] = 1e100
        accepted = compact.add(batch); expected = reference.add(batch)
        torch.testing.assert_close(accepted, expected)
        torch.testing.assert_close(accepted, torch.tensor([False, False, True, True]))
        for name in ('actor_history', 'relative_height'):
            rows, envs = torch.tensor([0, 0]), torch.tensor([0, 1])
            actual = compact._gather_fields(rows, envs, (name,))[name]
            original = reference._gather_fields(rows, envs, (name,))[name]
            torch.testing.assert_close(actual, original, rtol=0, atol=0, equal_nan=True)
        self.assert_oracle(compact, reference)
        restored = DeduplicatedHikingReplay(9, 4, 'cpu', .25, spec=SPEC)
        restored.load_state_dict(copy.deepcopy(compact.state_dict()))
        torch.testing.assert_close(restored._valid, compact._valid)

    def test_nonproduction_physical_dtypes_are_rejected_before_storage(self):
        for field in ('state', 'next_state'):
            compact, _, stream = make_pair(unusual=False)
            batch = stream.batch(); batch[field] = batch[field].double()
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'FP32 '+field):
                compact.add(batch)
            self.assertEqual(compact.total_steps, 0)
            self.assertEqual(compact.data, {})


if __name__ == '__main__':
    unittest.main()
