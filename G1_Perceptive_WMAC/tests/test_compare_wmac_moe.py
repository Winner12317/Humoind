"""CPU checks for paired Actor conversion without inheriting Actor Adam."""
from __future__ import annotations

import contextlib
from dataclasses import asdict
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.wmac.actors import actor_architecture, create_actor
from src.wmac.config import WmacConfig

SCRIPT = Path(__file__).resolve().parents[1]/"scripts/compare_wmac_moe.py"
SPEC = importlib.util.spec_from_file_location("compare_wmac_moe_test", SCRIPT)
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


def source_payload(kind="mlp"):
    cfg = WmacConfig(device="cpu", replay_device="cpu", compile=False, precision="float32",
        actor_type=kind, moe_variant="legacy", moe_top_k=2,
        policy_hidden=(8, 4), moe_gate_hidden=(5,), real_critic_coef=0.,
        model_warmup_iterations=5, critic_warmup_iterations=3)
    actor = create_actor(cfg)
    actor.norm.mean.fill_(.4)
    actor.norm.var.fill_(1.2)
    actor.norm.count.fill_(127.)
    optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr, betas=cfg.actor_betas)
    for parameter in actor.parameters():
        parameter.grad = torch.full_like(parameter, .01)
    optimizer.step()
    result = {"format": "amp-wmac-v1", "implementation_version": 5 if kind == "moe" else 4,
        "config": asdict(cfg), "iteration": 1000, "critic_calibration_updates": 3,
        "actor": actor.state_dict(), "opt_actor": optimizer.state_dict(),
        "training_cost": {"history_complete": True, "physical_transitions": 1234}}
    if kind == "moe":
        result.update(actor_type="moe", actor_architecture=actor_architecture(cfg))
    for index, name in enumerate((*CLI.AUXILIARIES, *CLI.AUX_OPTIMIZERS)):
        result[name] = {"source_value": torch.tensor([float(index)])}
    return cfg, result


def arguments():
    return SimpleNamespace(batch_size=4, wm_batch_size=8, wm_updates=1, device="cpu",
                           precision="float32", updates=8)


class MoeComparisonTest(unittest.TestCase):
    def test_budget_guard_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"new"
            args = ["--checkpoint", "source.pt", "--dataset", "walk.npz", "recovery.npz", "--output", str(output)]
            parsed = CLI.parse_args(args)
            self.assertEqual((parsed.updates, parsed.batch_size, parsed.wm_batch_size, parsed.data_steps), (8, 128, 256, 249))
            self.assertEqual(parsed.evaluation_seed, 2030)
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    CLI.parse_args(args+["--data-steps", "250"])
                output.mkdir()
                with self.assertRaises(SystemExit):
                    CLI.parse_args(args)

    def test_source_requires_trained_mlp_fresh_replay_and_no_stale_real_critic_targets(self):
        _, payload = source_payload()
        payload["config"].pop("actor_type")  # The running impl4 trainer predates this field.
        self.assertEqual(CLI.validate_source(payload).actor_type, "mlp")
        payload["replay"] = None
        with self.assertRaisesRegex(ValueError, "replay"):
            CLI.validate_source(payload)
        payload.pop("replay")
        payload["config"]["real_critic_coef"] = .5
        with self.assertRaisesRegex(ValueError, "real_critic_coef"):
            CLI.validate_source(payload)
        payload["config"]["real_critic_coef"] = 0.
        payload["critic_calibration_updates"] = 0
        with self.assertRaisesRegex(ValueError, "calibration"):
            CLI.validate_source(payload)
        _, moe = source_payload("moe")
        with self.assertRaisesRegex(ValueError, "MLP source"):
            CLI.validate_source(moe)

    def test_all_actor_adams_are_fresh_other_states_preserved_and_moe_initial_functions_match(self):
        source, payload = source_payload()
        before = CLI.state_fingerprint(payload)
        replay = SimpleNamespace(num_envs=4, capacity_steps=249, validation_env_fraction=.1)
        observations = torch.randn(11, 384)
        original = create_actor(source)
        original.load_state_dict(payload["actor"])
        with torch.no_grad():
            reference_actions = original(observations)
        moe_states, normalizers = [], []
        for label in CLI.ARMS:
            with self.subTest(label=label):
                cfg = CLI.arm_config(source, replay, arguments(), Path("unused")/label, label, payload["iteration"])
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(123)
                    actor = create_actor(cfg)
                runner = SimpleNamespace(cfg=cfg, actor=actor,
                    opt_actor=torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr, betas=cfg.actor_betas))
                converted = CLI.converted_initial_payload(payload, runner, Path("source.pt"), "source-hash", label, 123)
                self.assertTrue(payload["opt_actor"]["state"])
                self.assertEqual(converted["opt_actor"]["state"], {})
                self.assertFalse(converted["training_cost"]["history_complete"])
                self.assertEqual(CLI.auxiliary_fingerprints(converted), CLI.auxiliary_fingerprints(payload))
                history = converted["actor_initialization_history"][-1]
                self.assertTrue(history["actor_adam_reset"])
                self.assertTrue(history["fixed_data_not_on_policy_resume"])
                audit, arrays = CLI.initial_action_audit(cfg, converted["actor"], observations, reference_actions)
                self.assertLess(audit["maximum_absolute_action_difference"], 1e-5)
                normalizers.append(audit["normalizer_sha256"])
                if cfg.actor_type == "moe":
                    moe_states.append(audit["actor_state_sha256"])
                    self.assertEqual(arrays["indices"].shape, (11, cfg.moe_top_k))
                    self.assertEqual(cfg.moe_num_experts, 4)
                self.assertEqual(cfg.imagination_horizon, source.imagination_horizon)
                self.assertEqual(cfg.critic_epochs, source.critic_epochs)
        self.assertEqual(len(set(moe_states)), 1)
        self.assertEqual(len(set(normalizers)), 1)
        self.assertEqual(CLI.state_fingerprint(payload), before)

    def test_conversion_refuses_a_nonfresh_target_actor_optimizer(self):
        source, payload = source_payload()
        actor = create_actor(source)
        optimizer = torch.optim.Adam(actor.parameters())
        for parameter in actor.parameters():
            parameter.grad = torch.ones_like(parameter)
        optimizer.step()
        with self.assertRaisesRegex(ValueError, "freshly constructed"):
            CLI.converted_initial_payload(payload, SimpleNamespace(cfg=source, actor=actor, opt_actor=optimizer),
                                          Path("source.pt"), "hash", "mlp", 123)

    def test_action_audit_rejects_a_changed_initial_policy(self):
        cfg, payload = source_payload()
        observations = torch.zeros(3, 384)
        with self.assertRaisesRegex(ValueError, "differs"):
            CLI.initial_action_audit(cfg, payload["actor"], observations, torch.full((3, 29), 999.))

    def test_common_evaluation_uses_source_reward_and_explicit_route_trace(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)/"evaluations"/"source"
            args = SimpleNamespace(evaluation_seed=2030, evaluation_envs=32, evaluation_steps=500, device="cpu")
            def fake_run(command, **unused):
                self.assertEqual(command[command.index("--recovery-height-reward")+1], "height_only_v1")
                self.assertIn("--save-trace", command)
                output.mkdir()
                (output/"metrics.json").write_text('{"results": {}}')
            with patch.object(CLI.subprocess, "run", side_effect=fake_run) as run:
                self.assertEqual(CLI.evaluate_common(Path("source.pt"), output, args, "height_only_v1"), {"results": {}})
            run.assert_called_once()
            reference = {"protocol": {"recovery_height_reward": "height_only_v1", "amp_feature_schema": "grounded_v2"}}
            changed = {"protocol": {"recovery_height_reward": "upright_v2", "amp_feature_schema": "grounded_v2"}}
            with patch.object(CLI, "audit_evaluation_pair", return_value={"paired_comparison_allowed": True, "reasons": []}):
                self.assertFalse(CLI.audit_common_protocol(reference, changed, "height_only_v1")["paired_comparison_allowed"])

    def test_routing_churn_excludes_reset_episodes_and_distinguishes_rank_from_membership(self):
        indices = np.asarray([[[0, 1], [2, 3]], [[1, 0], [1, 2]], [[2, 3], [2, 1]]])
        valid = np.asarray([[True, True], [True, True], [False, True]])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"routing_recovery.npz"
            np.savez_compressed(path, indices=indices, first_episode_valid=valid,
                routing_protocol_json=np.asarray(json.dumps({"num_experts": 4})))
            summary = CLI.routing_summary(path)
        self.assertEqual(summary["valid_actions"], 5)
        self.assertEqual(summary["consecutive_first_episode_pairs"], 3)
        self.assertAlmostEqual(summary["selected_set_switch_fraction"], 1/3)
        self.assertEqual(summary["top1_switch_fraction"], 1.)
        np.testing.assert_allclose(summary["expert_selection_fraction"], [.2, .4, .3, .1])


if __name__ == "__main__":
    unittest.main()
