"""Training-proxy provenance must not alter paired physical evaluation rules."""

from __future__ import annotations

import contextlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.training_cost import WALL_TIME_SCOPE
from src.wmac.diagnostics import objective_gradient_probe, open_loop_probe
from src.wmac.networks import Actor
from test_wmac_diagnostics import toy_physical_windows
from test_wmac_evaluation_trace import physical_trace, physical_transition
from test_wmac_optimization import make_runner

ROOT = Path(__file__).resolve().parents[1]


def load_script(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


EVALUATE = load_script("recovery_provenance_evaluation", "evaluate_wmac.py")
COMPARE = load_script("recovery_provenance_comparison", "compare_amp_learning_efficiency.py")


class CheckpointRecoveryProvenanceTest(unittest.TestCase):
    def test_legacy_and_explicit_wmac_semantics_are_labelled_without_changing_policy(self):
        original = Actor(hidden=(8, 4), action_limit=5.)
        observation = torch.randn(2, 384)
        for configured, height_mode in ((None, None), ("soft_v1", "height_only_v1"),
                                       ("hard_forward_v2", "height_only_v1"),
                                       ("soft_v1", "upright_v2"), ("hard_forward_v2", "upright_v2")):
            expected = configured or "soft_v1"
            expected_height = height_mode or "height_only_v1"
            with self.subTest(configured=configured, height_mode=height_mode):
                config = {"policy_hidden": (8, 4), "action_limit": 5.,
                          "amp_feature_schema": "grounded_v2", "task": "Unitree-G1-AMP-Flat-Aligned"}
                if configured is not None:
                    config["recovery_counter_semantics"] = configured
                if height_mode is not None:
                    config["recovery_height_reward"] = height_mode
                payload = {"format": "amp-wmac-v1", "config": config, "actor": original.state_dict()}
                with patch("torch.load", return_value=payload):
                    actor, kind = EVALUATE.load_actor("unused.pt", "cpu")
                self.assertEqual(kind, "amp-wmac-v1")
                self.assertEqual(actor.source_recovery_counter_semantics, expected)
                self.assertEqual(actor.source_recovery_height_reward, expected_height)
                torch.testing.assert_close(actor(observation), original(observation))
                self.assertEqual(actor.amp_task_id, config["task"])
                metadata = EVALUATE.evaluation_source_metadata(actor)
                self.assertEqual(metadata["checkpoint_training_recovery_semantics"], expected)
                self.assertEqual(metadata["checkpoint_training_recovery_height_reward"], expected_height)
                self.assertEqual(metadata["source_provenance"]["physical_delay_rule"],
                                 "original_DelayedTerminationManager")
                self.assertFalse(metadata["source_provenance"]["training_recovery_proxy_used_during_physical_evaluation"])

    def test_ppo_has_no_wmac_training_counter_proxy(self):
        original = Actor(hidden=(8, 4), action_limit=None)
        payload = {"model_state_dict": {f"actor.{key}": value for key, value in original.net.state_dict().items()}}
        for height_mode in (None, "upright_v2"):
            if height_mode is not None:
                payload["recovery_height_reward"] = height_mode
            with patch("torch.load", return_value=payload):
                actor, kind = EVALUATE.load_actor("ppo.pt", "cpu")
            self.assertEqual(kind, "ppo-amp-baseline")
            self.assertEqual(actor.source_recovery_counter_semantics, "not_applicable")
            self.assertEqual(actor.source_recovery_height_reward, height_mode or "height_only_v1")

    def test_new_checkpoint_cannot_silently_lose_training_provenance(self):
        actor = Actor(hidden=(8, 4))
        payload = {"format": "amp-wmac-v1", "implementation_version": 4,
                   "config": {"policy_hidden": (8, 4)}, "actor": actor.state_dict()}
        with patch("torch.load", return_value=payload), self.assertRaisesRegex(ValueError, "missing"):
            EVALUATE.load_actor("unlabelled-new.pt", "cpu")


class PhysicalProtocolProvenanceTest(unittest.TestCase):
    def test_explicit_evaluation_height_reward_is_applied_before_environment_construction(self):
        from src.recovery_rewards import configure_recovery_height_reward
        from src.tasks.amp_loco.config.g1.env_cfgs import g1_amp_flat_aligned_env_cfg

        for training_height, evaluation_height in (("upright_v2", None), ("height_only_v1", "upright_v2")):
            cfg = g1_amp_flat_aligned_env_cfg()
            actor = SimpleNamespace(amp_task_id="Unitree-G1-AMP-Flat-Aligned",
                                    source_recovery_height_reward=training_height)
            args = SimpleNamespace(num_envs=2, seed=2030, push=False, observation_noise=False,
                                   steps=1, command_vx=.5, command_vy=0., command_yaw=0., device="cpu")
            if evaluation_height is not None:
                args.recovery_height_reward = evaluation_height
            with patch("mjlab.tasks.registry.load_env_cfg", return_value=cfg), \
                 patch("src.recovery_rewards.configure_recovery_height_reward", wraps=configure_recovery_height_reward) as configure, \
                 patch("src.wmac.env.CaptureEnv", side_effect=RuntimeError("stop before simulator construction")) as environment, \
                 patch.object(EVALUATE, "seed_all"):
                with self.assertRaisesRegex(RuntimeError, "before simulator"):
                    EVALUATE.evaluate_mode(actor, args, "walk", Path("unused"))
                configure.assert_called_once_with(cfg, evaluation_height or "height_only_v1")
                self.assertIs(environment.call_args.kwargs["cfg"], cfg)

    def test_recording_keeps_training_metadata_separate_from_physical_protocol_hash(self):
        protocols, hashes, sources = [], [], []
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for semantics in ("soft_v1", "hard_forward_v2"):
                path = root / semantics
                path.mkdir()
                provenance = EVALUATE.evaluation_source_metadata(SimpleNamespace(
                    source_recovery_counter_semantics=semantics,
                    source_recovery_height_reward="upright_v2" if semantics == "hard_forward_v2" else "height_only_v1"))["source_provenance"]
                recorder = EVALUATE.EvaluationRecorder(2, 1, save_transitions=True, save_trace=True,
                    protocol={"seed": 2030, "initial_state_sha256": "same-physical-start"},
                    source_provenance=provenance)
                recorder.append(physical_transition(), torch.ones(2, dtype=torch.bool), physical_trace())
                result = recorder.finish(path, "recovery", .02)
                hashes.append(result["transitions_recording"]["protocol_sha256"])
                self.assertEqual(result["transitions_recording"]["source_provenance"], provenance)
                with np.load(path / "transitions_recovery.npz", allow_pickle=False) as saved:
                    protocols.append(saved["recording_protocol_json"].item())
                    sources.append(json.loads(saved["recording_source_provenance_json"].item()))
                    self.assertFalse(saved["recording_source_provenance_json"].dtype.hasobject)
            self.assertEqual(protocols[0], protocols[1])
            self.assertEqual(hashes[0], hashes[1])
            self.assertEqual(sources[0]["checkpoint_training_recovery_semantics"], "soft_v1")
            self.assertEqual(sources[1]["checkpoint_training_recovery_semantics"], "hard_forward_v2")

    def test_metrics_preserve_training_metadata_without_rejecting_same_physical_trials(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = root / "model.pt"
            checkpoint.touch()
            paths = []
            for semantics in ("soft_v1", "hard_forward_v2"):
                output = root / semantics
                actor = SimpleNamespace(amp_task_id="Unitree-G1-AMP-Flat-Aligned", amp_feature_schema="grounded_v2",
                    source_recovery_counter_semantics=semantics,
                    source_recovery_height_reward="upright_v2" if semantics == "hard_forward_v2" else "height_only_v1",
                    training_cost={"physical_transitions": 1000, "wall_seconds": 10., "gpu_count": 1,
                                   "wall_time_scope": WALL_TIME_SCOPE, "history_complete": True,
                                   "trajectory_id": semantics})
                result = {"num_envs": 2, "steps_requested": 1, "control_dt": .02,
                          "initial_state_sha256": "same-physical-start", "recovery_success_fraction": 1.}
                argv = [str(ROOT / "scripts/evaluate_wmac.py"), "--checkpoint", str(checkpoint), "--mode", "recovery",
                        "--device", "cpu", "--num-envs", "2", "--steps", "1", "--output", str(output)]
                with patch.object(sys, "argv", argv), patch.object(EVALUATE, "load_actor", return_value=(actor, "amp-wmac-v1")), \
                     patch.object(EVALUATE, "evaluate_mode", return_value=result), \
                     patch("mjlab.utils.torch.configure_torch_backends"), contextlib.redirect_stdout(io.StringIO()):
                    EVALUATE.main()
                paths.append(output / "metrics.json")
                saved = json.loads(paths[-1].read_text())
                self.assertEqual(saved["checkpoint_training_recovery_semantics"], semantics)
                self.assertEqual(saved["checkpoint_training_recovery_height_reward"], actor.source_recovery_height_reward)
                self.assertEqual(saved["protocol"]["recovery_height_reward"], "height_only_v1")
                self.assertNotIn("checkpoint_training_recovery_semantics", saved["protocol"])
                self.assertNotIn("source_provenance", saved["protocol"])
            comparison = COMPARE.compare_learning_efficiency([paths[0]], [paths[1]],
                minimum=["recovery.recovery_success_fraction=.95"])
            self.assertTrue(comparison["evaluation_protocol_verified"])
            self.assertTrue(comparison["training_cost_comparison_available"])
            # Different physical reward objectives must still fail protocol matching.
            changed = json.loads(paths[1].read_text())
            changed["protocol"]["recovery_height_reward"] = "upright_v2"
            changed["arguments"]["recovery_height_reward"] = "upright_v2"
            paths[1].write_text(json.dumps(changed))
            with self.assertRaisesRegex(ValueError, "Physical evaluation protocol"):
                COMPARE.compare_learning_efficiency([paths[0]], [paths[1]],
                    minimum=["recovery.recovery_success_fraction=.95"])


class DiagnosticRecoveryProvenanceTest(unittest.TestCase):
    def test_objective_probe_reports_configured_semantics_and_surrogate_limitation(self):
        for semantics in ("soft_v1", "hard_forward_v2"):
            runner, batch, noise = make_runner(batch_size=2, horizon=2)
            runner.cfg.recovery_counter_semantics = semantics
            runner.cfg.recovery_height_reward = "upright_v2"
            if hasattr(runner.adapter, "recovery_counter_semantics"):
                runner.adapter.recovery_counter_semantics = semantics
            result = objective_gradient_probe(runner, batch, noise)
            self.assertEqual(result["protocol"]["recovery_counter_semantics"], semantics)
            self.assertEqual(result["protocol"]["recovery_height_reward"], "upright_v2")
            self.assertIn("not verified physical", result["protocol"]["surrogate_gradient_caveat"])
            if semantics == "hard_forward_v2":
                self.assertIn("biased straight-through", result["protocol"]["training_survival_gradient"])
            json.dumps(result, allow_nan=False)

    def test_open_loop_probe_supports_legacy_custom_configs_and_explicit_semantics(self):
        runner, windows = toy_physical_windows()
        legacy = open_loop_probe(runner, windows, horizons=(1, 3))
        self.assertEqual(legacy["protocol"]["recovery_counter_semantics"], "soft_v1")
        self.assertEqual(legacy["protocol"]["recovery_height_reward"], "height_only_v1")
        runner.cfg.recovery_counter_semantics = "hard_forward_v2"
        runner.cfg.recovery_height_reward = "upright_v2"
        new = open_loop_probe(runner, windows, horizons=(1, 3))
        self.assertEqual(new["protocol"]["recovery_counter_semantics"], "hard_forward_v2")
        self.assertEqual(new["protocol"]["recovery_height_reward"], "upright_v2")
        self.assertEqual(legacy["horizons"], new["horizons"])
        self.assertIn("biased straight-through", new["protocol"]["training_survival_gradient"])


if __name__ == "__main__":
    unittest.main()
