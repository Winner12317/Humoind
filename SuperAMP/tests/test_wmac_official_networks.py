"""Bounded official block alignment without changing the AMP state interface."""
from __future__ import annotations

import unittest

import torch
from torch import nn

from src.wmac.networks import Actor, Critic, WorldModel


def official_blocks(input_dim, hidden, output_dim):
    # Pinned upstream models.py:31-43, with dimensions supplied by this task.
    layers = []
    for width in hidden:
        layers.extend((nn.Linear(input_dim, width), nn.LayerNorm(width), nn.GELU()))
        input_dim = width
    layers.append(nn.Linear(input_dim, output_dim))
    return nn.Sequential(*layers)


class OfficialNetworkDetailsTest(unittest.TestCase):
    def assert_same_state(self, first, second):
        self.assertEqual(tuple(first.state_dict()), tuple(second.state_dict()))
        for name, value in first.state_dict().items():
            torch.testing.assert_close(value, second.state_dict()[name], rtol=0, atol=0)

    def test_existing_constructor_defaults_still_select_the_legacy_dialect(self):
        pairs = (
            (lambda: WorldModel(hidden=(13, 9)),
             lambda: WorldModel(hidden=(13, 9), activation="silu", output_zero_init=True)),
            (lambda: Actor(obs_dim=7, action_dim=5, hidden=(13, 9)),
             lambda: Actor(obs_dim=7, action_dim=5, hidden=(13, 9), activation="elu", layer_norm=False)),
            (lambda: Critic(7, hidden=(13, 9)),
             lambda: Critic(7, hidden=(13, 9), activation="elu", layer_norm=False)),
        )
        for default, explicit in pairs:
            torch.manual_seed(176)
            actual = default()
            torch.manual_seed(176)
            expected = explicit()
            self.assert_same_state(actual, expected)
            actual.load_state_dict(expected.state_dict(), strict=True)
            expected_activations = nn.SiLU if isinstance(actual, WorldModel) else nn.ELU
            self.assertEqual(sum(isinstance(layer, expected_activations) for layer in actual.net), 2)
            self.assertEqual(sum(isinstance(layer, nn.LayerNorm) for layer in actual.net),
                             2 if isinstance(actual, WorldModel) else 0)

    def test_actor_official_blocks_and_linear_head_match_pinned_construction(self):
        torch.manual_seed(142)
        actor = Actor(obs_dim=7, action_dim=5, hidden=(13, 9), action_limit=None,
                      output_init_scale=1., zero_output_bias=False,
                      activation="gelu", layer_norm=True)
        torch.manual_seed(142)
        expected = official_blocks(7, (13, 9), 5)
        self.assert_same_state(actor.net, expected)
        observation = torch.linspace(-4., 4., 21).reshape(3, 7)
        torch.testing.assert_close(actor(observation), expected(actor.norm(observation)), rtol=0, atol=0)
        self.assertIsNone(actor.action_limit)
        self.assertEqual(actor.norm.mean.shape, (7,))

    def test_critic_official_blocks_match_pinned_construction(self):
        torch.manual_seed(308)
        critic = Critic(7, hidden=(13, 9), activation="gelu", layer_norm=True)
        torch.manual_seed(308)
        expected = official_blocks(7, (13, 9), 1)
        self.assert_same_state(critic.net, expected)
        features = torch.linspace(-2., 2., 28).reshape(4, 7)
        torch.testing.assert_close(critic(features), expected(critic.norm(features)).squeeze(-1), rtol=0, atol=0)

    def test_world_model_changes_only_block_details_and_preserves_contact_extension(self):
        torch.manual_seed(413)
        model = WorldModel(hidden=(13, 9), activation="gelu", output_zero_init=False)
        torch.manual_seed(413)
        expected = official_blocks(135, (13, 9), 70)
        self.assert_same_state(model.net, expected)
        self.assertGreater(int(torch.count_nonzero(model.net[-1].weight)), 0)
        self.assertGreater(int(torch.count_nonzero(model.net[-1].bias)), 0)
        torch.testing.assert_close(model.contact_net[-1].weight,
                                   torch.zeros_like(model.contact_net[-1].weight), rtol=0, atol=0)
        torch.testing.assert_close(model.contact_net[-1].bias,
                                   torch.full_like(model.contact_net[-1].bias, -4.), rtol=0, atol=0)
        self.assertEqual(sum(isinstance(layer, nn.SiLU) for layer in model.contact_net), 2)
        legacy = WorldModel(hidden=(13, 9))
        self.assertEqual(tuple(model.state_dict()), tuple(legacy.state_dict()))
        state = torch.zeros(3, 71)
        state[:, 2] = .8
        state[:, 3] = 1.
        predicted, contact = model.step_with_contact(state, torch.zeros(3, 29), torch.zeros(3, 35))
        self.assertEqual(predicted.shape, (3, 71))
        self.assertEqual(contact.shape, (3, 2))
        self.assertTrue(bool(torch.isfinite(predicted).all()))

    def test_layernorm_actor_and_critic_are_structurally_distinct_from_legacy(self):
        legacy_actor = Actor(obs_dim=7, action_dim=5, hidden=(13, 9))
        official_actor = Actor(obs_dim=7, action_dim=5, hidden=(13, 9), activation="gelu", layer_norm=True)
        with self.assertRaises(RuntimeError):
            official_actor.load_state_dict(legacy_actor.state_dict(), strict=True)
        legacy_critic = Critic(7, hidden=(13, 9))
        official_critic = Critic(7, hidden=(13, 9), activation="gelu", layer_norm=True)
        with self.assertRaises(RuntimeError):
            official_critic.load_state_dict(legacy_critic.state_dict(), strict=True)

    def test_unknown_activation_is_rejected_by_each_network(self):
        for construct in (lambda: WorldModel(hidden=(8,), activation="unknown"),
                          lambda: Actor(obs_dim=7, hidden=(8,), activation="unknown"),
                          lambda: Critic(7, hidden=(8,), activation="unknown")):
            with self.assertRaisesRegex(ValueError, "Unsupported network activation"):
                construct()


if __name__ == "__main__":
    unittest.main()
