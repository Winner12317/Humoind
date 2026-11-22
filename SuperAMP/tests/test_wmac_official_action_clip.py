"""Preserve physical action clipping around the official raw-output Actor."""
from __future__ import annotations

import copy
from dataclasses import asdict
from pathlib import Path
import tempfile
import unittest

import torch

from scripts.evaluate_wmac import load_actor, policy_action_with_optional_routing
from src.wmac.actors import actor_architecture, create_actor
from src.wmac.config import WmacConfig
from src.wmac.networks import Actor
from src.wmac.reference import config_protocol


def payload_for(profile, *, limit=20.):
    cfg = WmacConfig.for_profile(profile)
    cfg.policy_hidden = (8, 6)
    cfg.action_limit = limit
    actor = create_actor(cfg)
    with torch.no_grad():
        actor.net[-1].weight.zero_()
        actor.net[-1].bias.copy_(torch.linspace(-50., 50., 29))
    payload = dict(format="amp-wmac-v1", implementation_version=7,
                   config=asdict(cfg), actor=actor.state_dict(), actor_type=cfg.actor_type,
                   actor_architecture=actor_architecture(cfg), training_protocol=config_protocol(cfg))
    return cfg, actor, payload


class ActionClipTest(unittest.TestCase):
    def setUp(self):
        self.observation = torch.zeros(2, 384)
        self.active = torch.tensor([True, False])

    def test_clipping_is_external_and_preserves_the_raw_tensor(self):
        raw = torch.tensor([[-25., 1., 30.], [12., -20., 20.]])
        class Policy:
            source_action_limit = 20.
            def __call__(self, observation):
                return raw
        result = policy_action_with_optional_routing(Policy(), self.observation, self.active)
        torch.testing.assert_close(result, torch.tensor([[-20., 1., 20.], [12., -20., 20.]]), atol=0, rtol=0)
        torch.testing.assert_close(raw, torch.tensor([[-25., 1., 30.], [12., -20., 20.]]), atol=0, rtol=0)

    def test_legacy_or_explicit_none_returns_the_original_output_object(self):
        raw = torch.tensor([[-50., 50.]])
        class Policy:
            def __call__(self, observation):
                return raw
        for explicit_none in (False, True):
            actor = Policy()
            if explicit_none:
                actor.source_action_limit = None
            with self.subTest(explicit_none=explicit_none):
                self.assertIs(policy_action_with_optional_routing(actor, self.observation, self.active), raw)

    def test_optional_route_records_the_submitted_clipped_action(self):
        raw = torch.tensor([[-30., 30.], [2., 3.]])
        route = object()
        class Policy:
            source_action_limit = 20.
            def __call__(self, observation):
                self.observation = observation
                return raw
            def route(self, observation):
                self.route_observation = observation
                return route
        class Recorder:
            def append(self, actual_route, action, active):
                self.values = actual_route, action, active
        actor, recorder = Policy(), Recorder()
        action = policy_action_with_optional_routing(actor, self.observation, self.active, recorder)
        self.assertIs(actor.observation, actor.route_observation)
        self.assertIs(recorder.values[0], route)
        self.assertIs(recorder.values[1], action)
        self.assertIs(recorder.values[2], self.active)
        torch.testing.assert_close(action, raw.clamp(-20, 20), atol=0, rtol=0)

    def test_actual_official_checkpoint_loads_saved_limit_and_keeps_forward_raw(self):
        cfg, source, payload = payload_for("supermimic_official", limit=7.5)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"official.pt"
            torch.save(payload, path)
            actor, kind = load_actor(path, "cpu")
        self.assertEqual(kind, "amp-wmac-v1")
        self.assertEqual(actor.source_action_limit, cfg.action_limit)
        self.assertIsNone(actor.action_limit)
        before = copy.deepcopy(actor.state_dict())
        with torch.no_grad():
            raw = actor(self.observation)
            actual = policy_action_with_optional_routing(actor, self.observation, self.active)
            torch.testing.assert_close(raw, source(self.observation), atol=0, rtol=0)
        self.assertGreater(float(raw.abs().max()), 20.)
        torch.testing.assert_close(actual, raw.clamp(-7.5, 7.5), atol=0, rtol=0)
        for name, value in before.items():
            torch.testing.assert_close(actor.state_dict()[name], value, atol=0, rtol=0)

    def test_actual_legacy_wmac_checkpoint_keeps_its_existing_output(self):
        _, source, payload = payload_for("ppo_aligned")
        payload["implementation_version"] = 6
        payload.pop("training_protocol")
        for field in ("reference_alignment", "optimizer_algorithm", "mirror_walk_motions"):
            payload["config"].pop(field)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"legacy.pt"
            torch.save(payload, path)
            actor, kind = load_actor(path, "cpu")
        self.assertEqual(kind, "amp-wmac-v1")
        self.assertFalse(hasattr(actor, "source_action_limit"))
        with torch.no_grad():
            expected = source(self.observation)
            actual = policy_action_with_optional_routing(actor, self.observation, self.active)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_actual_ppo_checkpoint_remains_unclipped(self):
        source = Actor(hidden=(8, 6), action_limit=None, output_init_scale=1., zero_output_bias=False)
        with torch.no_grad():
            source.net[-1].weight.zero_()
            source.net[-1].bias.copy_(torch.linspace(-50., 50., 29))
        payload = {"model_state_dict": {"actor."+name: tensor for name, tensor in source.net.state_dict().items()}}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"ppo.pt"
            torch.save(payload, path)
            actor, kind = load_actor(path, "cpu")
        self.assertEqual(kind, "ppo-amp-baseline")
        self.assertFalse(hasattr(actor, "source_action_limit"))
        with torch.no_grad():
            expected = source(self.observation)
            actual = policy_action_with_optional_routing(actor, self.observation, self.active)
        self.assertGreater(float(actual.abs().max()), 20.)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
