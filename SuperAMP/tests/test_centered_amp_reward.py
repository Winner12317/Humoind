"""Numerical and state-lifecycle checks for source-backed centered AMP reward."""
import copy
import unittest

import torch

from src.wmac.amp_rewards import CenteredAmpReward, clipped_lsgan_reward


class CenteredAmpRewardTest(unittest.TestCase):
    def test_autograd_matches_finite_differences_across_old_dead_boundary(self):
        reward = CenteredAmpReward().double()
        reward.update(torch.tensor([-.8, -.2]), torch.tensor([.7, .9]))
        score = torch.tensor([-3., -1.01, -1., -.99, .0, 1., 3.], dtype=torch.float64,
                             requires_grad=True)
        self.assertTrue(torch.autograd.gradcheck(reward, (score,)))
        slope, = torch.autograd.grad(reward(score).sum(), score)
        self.assertTrue(bool((slope > 0).all()))
        torch.testing.assert_close(slope, reward.score_slope(score))

    def test_bounds_center_and_tail_saturation_are_honest(self):
        reward = CenteredAmpReward(coefficient=.1, scale=1.)
        reward.update(torch.tensor([-.4]), torch.tensor([.6]))
        scores = torch.tensor([-1e6, -1., -.4, 1., 1e6], requires_grad=True)
        values = reward(scores)
        self.assertTrue(bool(((values >= 0) & (values <= .1)).all()))
        self.assertAlmostEqual(float(values[2]), .05, places=7)
        slope, = torch.autograd.grad(values.sum(), scores)
        self.assertAlmostEqual(float(slope[2]), .05, places=7)
        self.assertEqual(float(slope[0]), 0.)
        self.assertEqual(float(slope[-1]), 0.)
        torch.testing.assert_close(slope, reward.score_slope(scores))

    def test_center_translation_and_expert_score_do_not_change_reward(self):
        first, shifted = CenteredAmpReward(), CenteredAmpReward()
        first.update(torch.tensor([-.5]), torch.tensor([.9]))
        shifted.update(torch.tensor([-.5 + 8]), torch.tensor([-300.]))
        score = torch.linspace(-3, 2, 50)
        torch.testing.assert_close(first(score), shifted(score + 8), atol=4e-8, rtol=1e-6)

    def test_zero_first_mean_is_initialized_once_and_not_a_sentinel(self):
        reward = CenteredAmpReward(decay=.99)
        reward.update(torch.tensor([0.]), torch.tensor([0.]))
        self.assertTrue(bool(reward.initialized))
        reward.update(torch.tensor([-1.]), torch.tensor([1.]))
        self.assertAlmostEqual(float(reward.robot_mean), -.01, places=7)
        self.assertAlmostEqual(float(reward.expert_mean), .01, places=7)
        self.assertEqual(int(reward.updates), 2)
        self.assertEqual(list(reward.parameters()), [])
        self.assertFalse(reward.robot_mean.requires_grad)

    def test_no_center_update_from_bad_or_empty_batches(self):
        reward = CenteredAmpReward()
        reward.update(torch.tensor([-.5]), torch.tensor([.5]))
        saved = copy.deepcopy(reward.state_dict())
        for policy, expert in (([float('nan')], [.5]), ([-.2], [float('inf')]), ([], [.2])):
            reward.update(torch.tensor(policy), torch.tensor(expert))
        for key, value in saved.items():
            torch.testing.assert_close(reward.state_dict()[key], value, rtol=0, atol=0)

    def test_legacy_mode_preserves_quadratic_values_and_derivatives(self):
        score = torch.tensor([-3., -1.5, -.5, 1., 2., 4.], requires_grad=True)
        value = clipped_lsgan_reward(score, .1)
        torch.testing.assert_close(value, torch.tensor([0., 0., .04375, .1, .075, 0.]))
        gradient, = torch.autograd.grad(value.sum(), score)
        torch.testing.assert_close(gradient, torch.tensor([0., 0., .075, 0., -.05, 0.]))

    def test_stat_checkpoint_validation_rejects_wrong_counter_and_partial_state(self):
        reward = CenteredAmpReward()
        reward.validate_state(reward.state_dict())
        for replacement in ({'updates': torch.tensor(1)}, {'initialized': torch.tensor(True)},
                            {'updates': torch.tensor(-1)}, {'robot_mean': torch.tensor([0.])},
                            {'robot_mean': torch.tensor(1e300, dtype=torch.float64)}):
            state = dict(reward.state_dict(), **replacement)
            with self.assertRaises(ValueError):
                reward.validate_state(state)
        with self.assertRaises(ValueError):
            reward.validate_state({'robot_mean': torch.tensor(0.)})


if __name__ == '__main__':
    unittest.main()
