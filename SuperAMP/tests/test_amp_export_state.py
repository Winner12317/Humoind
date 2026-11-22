"""The PPO comparison arm must keep training after an ONNX save."""
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.modules import ActorCritic
from src.tasks.amp_loco.rl.runner import AMPOnPolicyRunner


class Policy(nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = nn.Sequential(nn.Linear(4, 2))

    def act_inference(self, observation):
        return self.actor(observation)


class ExportStateTest(unittest.TestCase):
    def test_save_after_update_ignores_cached_nonleaf_action_distribution(self):
        policy = ActorCritic(4, 4, 2, actor_hidden_dims=[8], critic_hidden_dims=[8]).train()
        observations = torch.randn(3, 4)
        policy.act(observations)
        policy.action_mean.square().mean().backward()
        distribution = policy.distribution
        self.assertIsNotNone(distribution.mean.grad_fn)
        owner = SimpleNamespace(alg=SimpleNamespace(policy=policy),
                                empirical_normalization=False, device="cpu")

        def export(wrapper, *args, **kwargs):
            torch.testing.assert_close(wrapper(observations), policy.act_inference(observations))
            self.assertIsNone(wrapper.actor_critic.distribution)

        with tempfile.TemporaryDirectory() as directory, patch("torch.onnx.export", side_effect=export):
            AMPOnPolicyRunner._export_policy_to_onnx(owner, directory)
        self.assertIs(policy.distribution, distribution)
        self.assertTrue(policy.training)

    def test_save_uses_copies_and_live_normalizer_keeps_learning(self):
        policy = Policy().train()
        norm = EmpiricalNormalization([4]).train()
        norm(torch.randn(8, 4))
        old_count = norm.count.clone()
        weights = {key: value.clone() for key, value in policy.state_dict().items()}
        owner = SimpleNamespace(alg=SimpleNamespace(policy=policy), obs_normalizer=norm,
                                empirical_normalization=True, device='cpu')
        wrappers = []

        def export(wrapper, *args, **kwargs):
            wrappers.append(wrapper)
            self.assertIsNot(wrapper.actor_critic, policy)
            self.assertIsNot(wrapper.obs_normalizer, norm)
            self.assertFalse(wrapper.training)
            self.assertFalse(wrapper.obs_normalizer.training)

        with tempfile.TemporaryDirectory() as directory, patch('torch.onnx.export', side_effect=export):
            AMPOnPolicyRunner._export_policy_to_onnx(owner, directory)
        self.assertEqual(len(wrappers), 1)
        self.assertTrue(policy.training)
        self.assertTrue(norm.training)
        for key, value in policy.state_dict().items():
            torch.testing.assert_close(value, weights[key])
        norm(torch.randn(8, 4))
        self.assertEqual(int(norm.count), int(old_count) + 8)


if __name__ == '__main__':
    unittest.main()
