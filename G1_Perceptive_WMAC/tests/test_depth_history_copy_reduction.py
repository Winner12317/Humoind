"""Real depth history sampling/appending matches the original cat/where path."""
from __future__ import annotations

from dataclasses import replace
import unittest

import torch

from src.wmac_perceptive.perception import (
    DepthHistory, PerceptionSpec, advance_depth_ring, select_depth_frames,
)


SPEC = PerceptionSpec(height=18, width=32, crop_top=0, crop_left=0, crop_right=0,
                      blur_kernel=1, include_valid=False, proprio_dim=768)


def assert_bits(test, actual, expected):
    test.assertEqual(actual.dtype, expected.dtype)
    integer = torch.int32 if actual.dtype == torch.float32 else torch.int64
    test.assertTrue(torch.equal(actual.contiguous().view(integer), expected.contiguous().view(integer)))


class DepthHistoryCopyReductionTest(unittest.TestCase):
    def history(self, n=7):
        torch.manual_seed(1924)
        history = DepthHistory(SPEC, n)
        history.reset(torch.rand(n, *SPEC.frame_shape))
        history.buffer.copy_(torch.rand_like(history.buffer))
        history.buffer.flatten()[::17] = -0.
        return history

    def test_selected_and_teacher_successor_frames_match_functional_append_bitwise(self):
        history = self.history()
        old = history.ring
        frozen = old.clone()
        frame = torch.rand(len(old), *SPEC.frame_shape)
        frame.flatten()[::13] = -0.
        current = select_depth_frames(old, SPEC)
        expected = select_depth_frames(advance_depth_ring(old, frame, SPEC), SPEC)
        assert_bits(self, history.sampled_frames(), current)
        assert_bits(self, history.sampled_successor_frames(frame), expected)
        current_pointer, next_pointer = history._sample_indices.data_ptr(), history._successor_sample_indices.data_ptr()
        # Changing a just-captured frame cannot change a one-step delayed view.
        assert_bits(self, history.sampled_successor_frames(frame+3.), expected)
        history.sampled_frames()
        self.assertEqual(history._sample_indices.data_ptr(), current_pointer)
        self.assertEqual(history._successor_sample_indices.data_ptr(), next_pointer)
        assert_bits(self, old, frozen)

    def test_append_with_no_partial_and_all_resets_matches_original_and_keeps_old_ring(self):
        history = self.history()
        expected = history.ring.clone()
        n = len(expected)
        for index, ids in enumerate(([], [1, 5], list(range(n)), [], [0, 6], [])):
            frame = torch.rand(n, *SPEC.frame_shape)
            frame.flatten()[::11] = -0.
            mask = torch.zeros(n, dtype=torch.bool)
            mask[ids] = True
            previous = history.ring
            snapshot = history.snapshot()
            expected = torch.where(mask[:, None, None, None, None], frame[:, None].expand_as(expected),
                                   torch.cat((expected[:, 1:], frame[:, None]), 1))
            history.append_with_reset_ids(frame, torch.tensor(ids, dtype=torch.int32 if index % 2 else torch.long))
            assert_bits(self, history.ring, expected)
            assert_bits(self, previous, snapshot)
            self.assertNotEqual(previous.data_ptr(), history.ring.data_ptr())
            self.assertTrue(bool(history.initialized.all()))
            self.assertTrue(history._fully_initialized)

    def test_initial_and_partial_reset_rows_use_original_initialization_rule(self):
        for initialized_ids in ([], [1, 4]):
            with self.subTest(initialized_ids=initialized_ids):
                history = DepthHistory(SPEC, 6)
                mask = torch.zeros(6, dtype=torch.bool)
                mask[initialized_ids] = True
                history.reset(torch.full((6, *SPEC.frame_shape), .2), mask)
                self.assertFalse(history._fully_initialized)
                previous, snapshot = history.ring, history.snapshot()
                frame = torch.rand(6, *SPEC.frame_shape)
                reset_ids = torch.tensor([3])
                reset = torch.zeros(6, dtype=torch.bool)
                reset[reset_ids] = True
                effective = reset | ~history.initialized
                expected = torch.where(effective[:, None, None, None, None], frame[:, None].expand_as(snapshot),
                    torch.cat((snapshot[:, 1:], frame[:, None]), 1))
                history.append_with_reset_ids(frame, reset_ids)
                assert_bits(self, history.ring, expected)
                assert_bits(self, previous, snapshot)
                self.assertTrue(bool(history.initialized.all()))
                self.assertTrue(history._fully_initialized)

    def test_wrong_shapes_nonfinite_frames_and_reset_index_schema_leave_history_unchanged(self):
        history = self.history()
        before, initialized = history.snapshot(), history.initialized.clone()
        wrong = torch.zeros(7, 1, 18, 32)
        with self.assertRaisesRegex(ValueError, "dimensions differ"):
            history.sampled_successor_frames(wrong)
        for invalid_ids in (torch.zeros(7, dtype=torch.bool), torch.zeros(1, 1, dtype=torch.long)):
            with self.subTest(ids=invalid_ids), self.assertRaisesRegex(ValueError, "integer tensor"):
                history.append_with_reset_ids(torch.zeros(7, *SPEC.frame_shape), invalid_ids)
        for value in (float('nan'), float('inf')):
            frame = torch.zeros(7, *SPEC.frame_shape)
            frame[2, 0, 0, 0] = value
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "non-finite"):
                history.append_with_reset_ids(frame, torch.tensor([2]))
        assert_bits(self, history.ring, before)
        self.assertTrue(torch.equal(history.initialized, initialized))

    def test_dtype_promotion_matches_teacher_and_delay_zero_keeps_the_old_api(self):
        history = self.history()
        frame = torch.zeros(len(history.ring), *SPEC.frame_shape, dtype=torch.float64)
        expected = select_depth_frames(advance_depth_ring(history.ring, frame, SPEC), SPEC)
        assert_bits(self, history.sampled_successor_frames(frame), expected)
        immediate = DepthHistory(replace(SPEC, delay=0), 2)
        with self.assertRaisesRegex(ValueError, "one-step Hiking delay"):
            immediate.sampled_successor_frames(torch.zeros(2, *SPEC.frame_shape))


if __name__ == "__main__":
    unittest.main()
