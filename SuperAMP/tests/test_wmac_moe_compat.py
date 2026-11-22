"""CPU contracts for Actor factories, historical checkpoints and script consumers."""
from __future__ import annotations

from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
import numpy as np

from src.wmac.actors import actor_architecture, create_actor
from src.wmac.checkpoints import checkpoint_actor_architecture, resolve_checkpoint_config
from src.wmac.config import WmacConfig
from src.wmac.networks import Actor


REPO = Path(__file__).resolve().parents[1]


def script(name):
    specification = importlib.util.spec_from_file_location("moe_compat_"+name, REPO/"scripts"/(name+".py"))
    result = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(result)
    return result


EVALUATE = script("evaluate_wmac")
TRAIN = script("train_wmac")
DIAGNOSE = script("diagnose_wmac")
RECORDED = script("probe_wmac_recorded_phases")
BENCHMARK = script("benchmark_wmac")


def config(kind="mlp", **overrides):
    result = WmacConfig(device="cpu", replay_device="cpu", compile=False, policy_hidden=(8, 4),
                        actor_type=kind, moe_variant="legacy", action_limit=5., moe_num_experts=3, moe_top_k=2,
                        moe_gate_hidden=(5,), moe_temperature=.8)
    for name, value in overrides.items():
        setattr(result, name, value)
    return result


def checkpoint(cfg, actor=None):
    return {"format": "amp-wmac-v1", "implementation_version": 5 if cfg.actor_type == "moe" else 4,
            "config": asdict(cfg), "actor_type": cfg.actor_type,
            "actor_architecture": actor_architecture(cfg),
            "actor": (actor or create_actor(cfg)).state_dict()}


class ActorFactoryCompatibilityTest(unittest.TestCase):
    def test_default_factory_retains_original_mlp_keys_initialization_and_outputs(self):
        cfg = config(policy_output_init_scale=.2, policy_output_zero_bias=False)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(71)
            old = Actor(hidden=cfg.policy_hidden, action_limit=cfg.action_limit,
                        output_init_scale=.2, zero_output_bias=False)
            torch.manual_seed(71)
            new = create_actor(cfg)
        self.assertIs(type(new), Actor)
        self.assertEqual(list(old.state_dict()), list(new.state_dict()))
        for name, value in old.state_dict().items():
            torch.testing.assert_close(new.state_dict()[name], value, rtol=0, atol=0)
        observations = torch.randn(7, 384)
        torch.testing.assert_close(new(observations), old(observations), rtol=0, atol=0)

    def test_old_impl4_without_actor_metadata_remains_mlp_and_keeps_saved_moments(self):
        cfg = config()
        actor = create_actor(cfg)
        actor.norm.mean.fill_(.4)
        actor.norm.var.fill_(1.7)
        actor.norm.count.fill_(123.)
        payload = checkpoint(cfg, actor)
        payload.pop("actor_type")
        payload.pop("actor_architecture")
        for name in list(payload["config"]):
            if name == "actor_type" or name.startswith("moe_"):
                payload["config"].pop(name)
        self.assertEqual(checkpoint_actor_architecture(payload)["actor_type"], "mlp")
        restored_cfg = resolve_checkpoint_config(payload)
        self.assertEqual(restored_cfg.actor_type, "mlp")
        with patch("torch.load", return_value=payload):
            restored, _ = EVALUATE.load_actor("historical.pt", "cpu")
        self.assertIs(type(restored), Actor)
        for name, value in actor.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], value, rtol=0, atol=0)

    def test_moe_roundtrip_uses_factory_and_records_architecture_outside_physical_protocol(self):
        cfg = config("moe")
        original = create_actor(cfg)
        payload = checkpoint(cfg, original)
        with patch("torch.load", return_value=payload), patch("src.wmac.actors.create_actor", wraps=create_actor) as factory:
            restored, kind = EVALUATE.load_actor("moe.pt", "cpu")
        factory.assert_called_once()
        self.assertEqual(kind, "amp-wmac-v1")
        observations = torch.randn(9, 384)
        torch.testing.assert_close(original(observations), restored(observations), rtol=0, atol=0)
        metadata = EVALUATE.evaluation_source_metadata(restored)
        self.assertEqual(metadata["checkpoint_actor_architecture"], actor_architecture(cfg))
        self.assertEqual(metadata["source_provenance"]["checkpoint_actor_type"], "moe")
        recorder = EVALUATE.EvaluationRecorder(2, 1, protocol={"physical_delay_rule": "unchanged"},
                                             source_provenance=metadata["source_provenance"])
        self.assertNotIn("checkpoint_actor_type", recorder.protocol)

    def test_baseline_ppo_still_uses_original_actor_and_unbounded_output(self):
        original = Actor(hidden=(8, 4), action_limit=None)
        payload = {"model_state_dict": {"actor."+name: value for name, value in original.net.state_dict().items()}}
        with patch("torch.load", return_value=payload), patch("src.wmac.actors.create_actor") as factory:
            restored, kind = EVALUATE.load_actor("ppo.pt", "cpu")
        factory.assert_not_called()
        self.assertIs(type(restored), Actor)
        self.assertEqual(kind, "ppo-amp-baseline")
        self.assertIsNone(restored.action_limit)
        observations = torch.randn(3, 384)
        torch.testing.assert_close(restored(observations), original(observations), rtol=0, atol=0)

    def test_moe_invalid_top1_or_architecture_metadata_is_rejected(self):
        for changes in ({"moe_num_experts": 1}, {"moe_top_k": 1}, {"moe_temperature": 0.}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                create_actor(config("moe", **changes))
        payload = checkpoint(config("moe"))
        payload["actor_architecture"]["moe_temperature"] = .3
        with self.assertRaisesRegex(ValueError, "conflicts"):
            checkpoint_actor_architecture(payload)
        payload = checkpoint(config("moe"))
        payload.pop("actor_architecture")
        with self.assertRaisesRegex(ValueError, "requires explicit"):
            checkpoint_actor_architecture(payload)
        old_without_label = {"format": "amp-wmac-v1", "config": {}, "actor": {"experts.0.0.weight": torch.zeros(2, 2)}}
        with self.assertRaisesRegex(ValueError, "tensor layout"):
            checkpoint_actor_architecture(old_without_label)

    def test_mlp_ignores_inactive_moe_fields_but_moe_temperature_changes_architecture(self):
        first = config()
        self.assertEqual(actor_architecture(first), actor_architecture(replace(first, moe_num_experts=8)))
        moe = config("moe")
        self.assertNotEqual(actor_architecture(moe), actor_architecture(replace(moe, moe_temperature=1.)))


class ActorConsumersCompatibilityTest(unittest.TestCase):
    def test_resume_cli_keeps_historical_mlp_and_cannot_migrate_architecture_with_recovery_switch(self):
        payload = checkpoint(config())
        payload.pop("actor_type")
        payload.pop("actor_architecture")
        payload["config"].pop("actor_type")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"legacy.pt"
            torch.save(payload, path)
            restored = TRAIN.parse_config(["--resume", str(path)])
            self.assertEqual(restored.actor_type, "mlp")
            with self.assertRaisesRegex(ValueError, "--init-policy"):
                TRAIN.parse_config(["--resume", str(path), "--actor-type", "moe",
                                    "--allow-recovery-semantics-migration"])

    def test_component_swaps_and_recorded_probes_reject_moe_routing_changes(self):
        cfg = config("moe")
        def diagnostic_payload(value):
            return {"format": "amp-wmac-v1", "config": asdict(value),
                    **{name: {"weight": torch.ones(1)} for name in DIAGNOSE.MODULES}}
        source = diagnostic_payload(cfg)
        DIAGNOSE.validate_compatible(source, diagnostic_payload(cfg))
        for changed in (replace(cfg, actor_type="mlp"), replace(cfg, moe_top_k=3),
                        replace(cfg, moe_gate_hidden=(7,)), replace(cfg, moe_temperature=1.)):
            with self.subTest(changed=actor_architecture(changed)):
                with self.assertRaisesRegex(ValueError, "matching architecture"):
                    DIAGNOSE.validate_compatible(source, diagnostic_payload(changed))
                with self.assertRaisesRegex(ValueError, "incompatible semantics"):
                    RECORDED.validate_bundles({22000: (cfg,), 24000: (changed,)})

    def test_archived_benchmark_refuses_moe_weights_before_loading_any_module(self):
        runner = SimpleNamespace(cfg=config(), **{name: SimpleNamespace(load_state_dict=Mock()) for name in BENCHMARK.MODULE_NAMES})
        with self.assertRaisesRegex(ValueError, "Actor architecture"):
            BENCHMARK.load_shared_networks(torch, runner, {"config": asdict(config("moe"))})
        for name in BENCHMARK.MODULE_NAMES:
            getattr(runner, name).load_state_dict.assert_not_called()

    def test_parameter_audit_reports_moe_total_and_active_capacity_separately(self):
        comparison = script("compare_amp_parameters")
        cfg = WmacConfig.for_profile("ppo_aligned")
        cfg.actor_type, cfg.policy_hidden = "moe", (8, 4)
        cfg.moe_variant = "legacy"
        cfg.moe_num_experts, cfg.moe_top_k, cfg.moe_gate_hidden = 3, 2, (5,)
        report = comparison.build_comparison(4, wmac_config=cfg)
        budget = report["actor_architecture_and_parameters"]
        self.assertEqual(budget["architecture"]["actor_type"], "moe")
        self.assertEqual(budget["total_parameters"], budget["router_parameters"]+sum(budget["expert_parameters"]))
        self.assertEqual(budget["active_parameters_per_token"], budget["router_parameters"]+sum(budget["expert_parameters"][:2]))
        self.assertLess(budget["active_parameters_per_token"], budget["total_parameters"])
        self.assertFalse(report["all_common_parameters_match"])
        self.assertIn("actor_architecture_and_capacity", [item["parameter"] for item in report["remaining_differences"]])

    def test_physical_diagnostic_checkpoint_loader_uses_same_moe_factory(self):
        physics = script("check_wmac_physics")
        from src.wmac.networks import Critic, RunningNorm, WorldModel
        from rsl_rl.modules.discriminator import Discriminator
        cfg = config("moe", model_hidden=(8,), critic_hidden=(8,), discriminator_hidden=(8,))
        original = create_actor(cfg)
        payload = checkpoint(cfg, original)
        payload.update(world_model=WorldModel(hidden=cfg.model_hidden).state_dict(),
                       critic=Critic(136, cfg.critic_hidden).state_dict(),
                       discriminator=Discriminator(390, .1, [8], "cpu", .75).state_dict(),
                       amp_norm=RunningNorm(195).state_dict())
        with patch("torch.load", return_value=payload), patch.object(physics, "create_actor", wraps=create_actor) as factory:
            restored = physics.load_checkpoint("moe.pt", "cpu")
        factory.assert_called_once()
        self.assertEqual(restored[0].actor_type, "moe")
        observations = torch.randn(4, 384)
        torch.testing.assert_close(restored[1](observations), original(observations), rtol=0, atol=0)


class RoutingRecordingTest(unittest.TestCase):
    def test_optional_routing_uses_action_observation_and_first_episode_mask_without_extra_unrequested_gate(self):
        route = SimpleNamespace(indices=torch.tensor([[2, 1], [2, 1]]),
            weights=torch.tensor([[.625, .375], [.625, .375]]),
            probabilities=torch.tensor([[.2, .3, .5], [.2, .3, .5]]))
        observations = torch.arange(2*384, dtype=torch.float32).reshape(2, 384)
        active = torch.ones(2, dtype=torch.bool)
        class Policy:
            def __call__(self, values):
                self.last_action_observation = values
                return values[:, :29]*.01
        actor = Policy()
        def trace_route(values):
            self.assertIs(values, actor.last_action_observation)
            self.assertFalse(torch.is_grad_enabled())
            return route
        actor.route = Mock(side_effect=trace_route)
        EVALUATE.policy_action_with_optional_routing(actor, observations, active)
        actor.route.assert_not_called()
        recorder = EVALUATE.RoutingRecorder(2, 2, 3, 2, actor_architecture(config("moe")))
        first_action = EVALUATE.policy_action_with_optional_routing(actor, observations, active, recorder)
        expected_first = first_action.clone()
        active[0] = False
        EVALUATE.policy_action_with_optional_routing(actor, observations+1., active, recorder)
        observations.fill_(999.)
        active.zero_()
        with tempfile.TemporaryDirectory() as directory:
            result = recorder.finish(directory, "recovery", .02)
            with np.load(Path(directory)/"routing_recovery.npz", allow_pickle=False) as saved:
                np.testing.assert_array_equal(saved["first_episode_valid"], [[True, True], [False, True]])
                np.testing.assert_array_equal(saved["step_index"], [0, 1])
                np.testing.assert_allclose(saved["action_time_seconds"], [0., .02])
                np.testing.assert_array_equal(saved["action"][0], expected_first.numpy())
                self.assertEqual(saved["probabilities"].shape, (2, 2, 3))
                protocol = json.loads(str(saved["routing_protocol_json"].item()))
                self.assertIn("action discontinuities", protocol["limitation"])
            self.assertEqual(result["routing_recording"]["first_episode_actions"], 3)
            self.assertFalse((Path(directory)/"routing_recovery.npz.tmp").exists())


if __name__ == "__main__":
    unittest.main()
