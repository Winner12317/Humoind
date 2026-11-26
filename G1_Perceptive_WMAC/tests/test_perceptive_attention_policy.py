"""Observation layout, state-conditioned readout, and one-block parity."""

from __future__ import annotations

import unittest

import torch
from torch.nn import functional as F

from src.wmac_perceptive.attention_policy import PerceptiveAttentionPolicy
from src.wmac_perceptive.hiking_observations import pack_hiking_proprio_history


class PerceptiveAttentionPolicyTest(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(312)
        self.policy = PerceptiveAttentionPolicy().double()
        self.history = torch.randn(2, 8, 96, dtype=torch.float64, generator=generator)
        self.latent = torch.randn(2, 128, dtype=torch.float64, generator=generator)

    def pack(self, history):
        return torch.cat((pack_hiking_proprio_history(history), self.latent), dim=-1)

    def test_readout_matches_full_self_attention_and_gradients(self):
        policy = self.policy
        history = self.history.requires_grad_()
        latent = self.latent.requires_grad_()
        proprio = policy.proprio_projection(history)
        visual = policy.visual_projection(latent)[:, None]
        command = policy.command_projection(history[:, -1, 6:9])[:, None]
        tokens = torch.cat((proprio, visual, command, proprio[:, -1:] + command), dim=1) + policy.position
        normalized = policy.attention_norm(tokens)
        query = policy.query(normalized).reshape(2, 11, 4, 32).transpose(1, 2)
        key, value = policy.key_value(normalized).chunk(2, dim=-1)
        key = key.reshape(2, 11, 4, 32).transpose(1, 2)
        value = value.reshape(2, 11, 4, 32).transpose(1, 2)
        attention = F.scaled_dot_product_attention(query, key, value).transpose(1, 2).reshape(2, 11, 128)
        hidden = tokens + policy.attention_output(attention)
        hidden = hidden + policy.feedforward(policy.feedforward_norm(hidden))
        expected = policy.output(hidden[:, -1])

        actual = policy(self.pack(history))
        torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)
        inputs = (history, latent, *policy.parameters())
        expected_grads = torch.autograd.grad(expected.square().sum(), inputs)
        actual_grads = torch.autograd.grad(actual.square().sum(), inputs)
        for actual_grad, expected_grad in zip(actual_grads, expected_grads, strict=True):
            torch.testing.assert_close(actual_grad, expected_grad, atol=1e-12, rtol=1e-10)
        self.assertTrue(all(bool(gradient.norm() > 0) for gradient in actual_grads))
        torch.testing.assert_close(policy(self.pack(history)[0]), actual[0])

    def test_latest_state_and_command_change_readout_query(self):
        queries = []
        handle = self.policy.query.register_forward_hook(lambda _module, _inputs, result: queries.append(result))
        try:
            earlier = self.history.clone()
            earlier[:, 0, 0] += 2.0
            latest = self.history.clone()
            latest[:, -1, 0] += 2.0
            command = self.history.clone()
            command[:, -1, 6] += 2.0
            with torch.no_grad():
                baseline_actions = self.policy(self.pack(self.history))
                earlier_actions = self.policy(self.pack(earlier))
                self.policy(self.pack(latest))
                self.policy(self.pack(command))
        finally:
            handle.remove()

        baseline, earlier_query, latest_query, command_query = queries
        torch.testing.assert_close(earlier_query, baseline, rtol=0, atol=0)
        self.assertFalse(torch.allclose(earlier_actions, baseline_actions))
        self.assertFalse(torch.allclose(latest_query, baseline))
        self.assertFalse(torch.allclose(command_query, baseline))


if __name__ == "__main__":
    unittest.main()
