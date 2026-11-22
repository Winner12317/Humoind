import unittest

import torch

from src.wmac.runner import _collect_tick_statistics, _masked_stat_mean


class FlatStatisticsTests(unittest.TestCase):
    def test_collect_fields_keep_tick_weights_and_exclude_rejected_nan(self):
        batch = {
            'reward': torch.tensor([2., 6., float('nan'), float('inf')]),
            'state': torch.tensor([[0., 0., .4], [0., 0., .7], [0., 0., .9], [0., 0., .6]]),
            'context': torch.tensor([[0.], [0.], [1.], [0.]]),
            'terminated': torch.tensor([True, False, False, False]),
            'timeout': torch.tensor([False, False, False, False]),
            'boundary': torch.tensor([True, True, True, False]),
        }
        first = _collect_tick_statistics(batch, torch.tensor([True, True, False, False]))
        empty = _collect_tick_statistics(batch, torch.zeros(4, dtype=torch.bool))
        torch.testing.assert_close(first, torch.tensor([4., .25, 0., .5, .75, .5]), rtol=0, atol=0)
        torch.testing.assert_close(empty, torch.tensor([0., .25, 0., .5, .75, 1.]), rtol=0, atol=0)
        self.assertEqual(torch.stack((first, empty)).sum(0, dtype=torch.float64)[0].item() / 2, 2.)

    def test_model_phase_reductions_preserve_empty_and_selected_nonfinite(self):
        errors = torch.tensor([1., float('nan'), 5., float('inf')])
        selected = torch.tensor([[True, False, True, False], [False, False, False, False],
                                 [True, True, False, False], [False, False, True, True]])
        phases = _masked_stat_mean(errors, selected, empty=float('nan'))
        torch.testing.assert_close(phases, torch.tensor([3., float('nan'), float('nan'), float('inf')]),
                                   rtol=0, atol=0, equal_nan=True)


if __name__ == '__main__':
    unittest.main()
