"""Compiled-step G1 imagination parity, independent validation and resume.

CPU tests use torch.compile's eager backend: Dynamo capture is exercised
without invoking CPU Inductor. No GPU environment or long training is started.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from src.wmac.networks import frozen_parameters
from src.wmac_perceptive.compiled_imagination import CriticLoss, LatentImaginedStep, LatentTrajectoryTerms
from src.wmac_perceptive.latent_perception import unpack_processed_actor_input
from src.wmac_perceptive.latent_replay import LatentPerceptiveReplay
from src.wmac_perceptive.latent_training import rollout_latent_model_metrics
from src.wmac_perceptive.perception import PerceptionSpec
from src.wmac_perceptive.optimizers import build_reference_optimizer
from src.wmac_perceptive.runner import MODULES, OPTIMIZERS
from test_perceptive_latent_runner import latent_runner_fixture, latent_transition
import test_perceptive_runner_restore as restore_checks


def fast_runner_fixture(*, batch_size=3, horizon=3, wm_window=None, spec=None, adam_steps=0,
                        compiled=True):
    runner, batch, noise = latent_runner_fixture(batch_size, horizon, spec, adam_steps)
    window = runner.cfg.wm_window if wm_window is None else wm_window
    runner.cfg = replace(runner.cfg, execution_profile="supermimic", replay_storage="legacy",
                         validation_horizon=max(8, horizon), wm_window=window,
                         compile=False, cuda_graphs=False, validation_batch_size=batch_size)
    runner._setup_fast_execution()
    if compiled:
        runner._latent_step_fn = torch.compile(runner._latent_step_module, backend="eager", fullgraph=True, dynamic=False)
        runner._latent_trajectory_fn = torch.compile(runner._latent_trajectory_module, backend="eager", fullgraph=True, dynamic=False)
        runner._latent_critic_loss_fn = torch.compile(runner._latent_critic_loss_module, backend="eager", fullgraph=True, dynamic=False)
    return runner, batch, noise


def copy_networks(source, destination):
    for name in (*MODULES, "latent_decoder"):
        getattr(destination, name).load_state_dict(getattr(source, name).state_dict())


class FastImaginationParityTest(unittest.TestCase):
    def test_compiled_step_and_fk_terms_match_legacy_imagination_and_all_actor_gradients(self):
        legacy, batch, noise = latent_runner_fixture()
        fast, fast_batch, fast_noise = fast_runner_fixture()
        copy_networks(legacy, fast)
        outputs, gradients = [], []
        for runner, starts, noises in ((legacy, batch, noise), (fast, fast_batch, fast_noise)):
            with patch.object(runner.latent_decoder, "forward", side_effect=AssertionError("decoder entered imagination")), \
                    patch.object(runner.actor, "encode_processed", wraps=runner.actor.encode_processed) as encode, \
                    frozen_parameters(runner.world_model, runner.discriminator, runner.critic):
                values = runner.imagine(starts, noises=noises)
                grads = torch.autograd.grad(values["objective"].sum(), tuple(runner.actor.parameters()), allow_unused=True)
                self.assertEqual(encode.call_count, 1)
            outputs.append(values)
            gradients.append(grads)
        for name in outputs[0]:
            torch.testing.assert_close(outputs[1][name], outputs[0][name], rtol=2e-10, atol=2e-11)
        self.assertEqual(len(gradients[0]), len(gradients[1]))
        for expected, actual in zip(gradients[0], gradients[1]):
            if expected is None:
                self.assertIsNone(actual)
            else:
                self.assertTrue(bool(torch.isfinite(actual).all()))
                torch.testing.assert_close(actual, expected, rtol=3e-9, atol=3e-11)
        for runner in (legacy, fast):
            self.assertTrue(all(parameter.grad is None for module in
                (runner.world_model, runner.discriminator, runner.critic, runner.latent_decoder)
                for parameter in module.parameters()))

    def test_compiled_actor_step_detaches_visual_input_but_preserves_world_model_latent_derivative(self):
        runner, batch, noise = fast_runner_fixture()
        proprio, stack = unpack_processed_actor_input(batch["actor_history"], runner.spec)
        latent = runner.actor.encode_processed(stack).detach().requires_grad_(True)
        result = runner._latent_step_fn(batch["state"], batch["context"], proprio.reshape(-1, 4, 96), latent,
            batch["navigation_target"], batch["desired_speed"], noise[0])
        action_grad, = torch.autograd.grad(result[4].sum(), latent, allow_unused=True, retain_graph=True)
        latent_grad, = torch.autograd.grad(result[3].sum(), latent)
        self.assertTrue(action_grad is None or bool((action_grad == 0).all()))
        self.assertGreater(float(latent_grad.abs().sum()), 0.)

    def test_fast_critic_has_same_rng_minibatches_weights_and_adam_moments(self):
        legacy, _, _ = latent_runner_fixture()
        fast, _, _ = fast_runner_fixture()
        copy_networks(legacy, fast)
        for runner in (legacy, fast):
            runner.cfg.critic_epochs, runner.cfg.critic_minibatches = 2, 4
        torch.manual_seed(601)
        features = torch.randn(3, 3, 217, dtype=torch.float64)
        targets = torch.randn(3, 3, dtype=torch.float64)
        torch.manual_seed(602)
        expected = legacy._fit_critic(features, targets)
        torch.manual_seed(602)
        actual = fast._fit_critic(features, targets)
        for name in expected:
            self.assertAlmostEqual(actual[name], expected[name], places=11)
        for expected_parameter, actual_parameter in zip(legacy.critic.parameters(), fast.critic.parameters()):
            torch.testing.assert_close(actual_parameter, expected_parameter, rtol=2e-12, atol=2e-13)
            left, right = legacy.opt_critic.state[expected_parameter], fast.opt_critic.state[actual_parameter]
            for name in left:
                torch.testing.assert_close(right[name], left[name], rtol=2e-12, atol=2e-13)

    def test_super_critic_shuffles_once_and_reuses_four_minibatches_for_five_epochs(self):
        fast, _, _ = fast_runner_fixture(compiled=False)
        oracle, _, _ = latent_runner_fixture()
        copy_networks(fast, oracle)
        fast.cfg.execution_optimizer = "muon_adam"
        for runner in (fast, oracle):
            runner.cfg.critic_epochs, runner.cfg.critic_minibatches = 5, 4
            runner.cfg.max_grad_norm = 1.
            runner.cfg.critic_lr = .0002
            runner.opt_critic = build_reference_optimizer([runner.critic], .0002, device="cpu")
        torch.manual_seed(603)
        features = torch.randn(4, 3, 217, dtype=torch.float64)
        features[..., 0] = torch.arange(12).reshape(4, 3)
        targets = torch.randn(4, 3, dtype=torch.float64)
        flat, target = features.flatten(0, 1), targets.flatten()
        torch.manual_seed(604)
        permutation = torch.randperm(len(flat))
        expected_batches = [indices.tolist() for _ in range(5) for indices in permutation.view(4, -1)]
        for indices in [indices for _ in range(5) for indices in permutation.view(4, -1)]:
            oracle.opt_critic.zero_grad(set_to_none=True)
            loss = (oracle.critic(flat[indices])-target[indices]).square().mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(oracle.critic.parameters(), 1., foreach=True)
            oracle.opt_critic.step()
        calls = []
        original_loss = fast._latent_critic_loss_fn
        def capture_loss(values, target):
            calls.append(values[:, 0].long().tolist())
            return original_loss(values, target)
        torch.manual_seed(604)
        with patch("torch.randperm", wraps=torch.randperm) as shuffle, \
                patch.object(fast, "_latent_critic_loss_fn", side_effect=capture_loss):
            fast._fit_critic(features, targets)
        self.assertEqual(shuffle.call_count, 1)
        self.assertEqual(calls, expected_batches)
        for actual, expected in zip(fast.critic.parameters(), oracle.critic.parameters()):
            torch.testing.assert_close(actual, expected, rtol=2e-12, atol=2e-13)
            for name in oracle.opt_critic.state[expected]:
                torch.testing.assert_close(fast.opt_critic.state[actual][name], oracle.opt_critic.state[expected][name], rtol=2e-12, atol=2e-13)
        torch.manual_seed(604)
        with patch("torch.randperm", wraps=torch.randperm) as legacy_shuffle:
            oracle._fit_critic(features, targets)
        self.assertEqual(legacy_shuffle.call_count, 5)


class FastValidationHorizonTest(unittest.TestCase):
    def test_model_training_window_two_and_free_running_validation_four_are_independent(self):
        runner, batch, _ = fast_runner_fixture(horizon=4, wm_window=2, compiled=False)
        runner.cfg.validation_horizon = 4
        runner.replay = LatentPerceptiveReplay(8, 3, validation_env_fraction=1/3, spec=runner.spec)
        for index in range(6):
            runner.replay.add(latent_transition(runner, batch, index))
        windows = runner.replay.sample_windows(3, 2, 0., fields=runner.WM_FIELDS)
        with torch.no_grad():
            prepared = runner._latent_wm_execution.kernel.prepare(windows, update_statistics=False)
        loss, metrics = runner._latent_wm_execution.kernel(prepared)
        self.assertTrue(bool(torch.isfinite(loss)))
        names = runner._latent_wm_execution.metric_names
        self.assertIn("wm_h2_model_error", names)
        self.assertNotIn("wm_h4_model_error", names)
        self.assertEqual(metrics.numel(), len(names))
        with patch.object(runner.replay, "sample_windows", wraps=runner.replay.sample_windows) as sample:
            validation = runner.validate_model()
        self.assertEqual(sample.call_args.args[1], 4)
        self.assertEqual(sample.call_args.kwargs["split"], "validation")
        self.assertIn("validation_walk_h4_model_error", validation)
        self.assertIn("validation_walk_h4_latent_error", validation)
        self.assertEqual(runner.cfg.wm_window, 2)
        self.assertEqual(runner.validation_steps, 4)
        self.assertGreater(validation["validation_walk_h4_samples"], 0)

    def test_actor_gate_reads_training_min_h_k_but_demands_validation_at_actual_h(self):
        runner, _, _ = fast_runner_fixture(horizon=4, wm_window=2, compiled=False)
        runner.cfg.imagination_horizon_start = 1
        runner.cfg.validation_horizon = 4
        runner.actor_updates_accepted = 2 * runner.cfg.horizon_growth_updates
        runner.validation = {}
        for horizon in (1, 2, 4):
            prefix = f"validation_walk_h{horizon}_"
            runner.validation.update({prefix+"samples": 3., prefix+"finite_fraction": 1.,
                prefix+"model_error": .01, prefix+"contact_error": .01,
                prefix+"latent_error": .01})
        row = dict(wm_updates_applied_fraction=1., wm_h1_model_error=.01, wm_h2_model_error=.01)
        self.assertTrue(runner._actor_gate(dict(row)))
        self.assertEqual(runner._active_horizon, 4)
        self.assertNotIn("wm_h4_model_error", row)
        runner.validation["validation_walk_h4_finite_fraction"] = 0.
        self.assertTrue(runner._actor_gate(dict(row)))
        self.assertEqual(runner._active_horizon, 2)
        runner.validation["validation_walk_h4_finite_fraction"] = 1.
        self.assertTrue(runner._actor_gate({**row, "wm_h2_model_error": float("inf")}))
        self.assertEqual(runner._active_horizon, 1)
        self.assertFalse(runner._actor_gate({**row, "wm_h1_model_error": float("inf")}))


class FastRunnerCheckpointTest(unittest.TestCase):
    assert_tree_equal = restore_checks.PerceptiveRunnerRestoreTest.assert_tree_equal

    def test_fast_resume_rebuilds_execution_and_invalidates_captured_graph_state(self):
        source, _, _ = fast_runner_fixture(spec=PerceptionSpec(), adam_steps=1, compiled=False)
        restored, _, _ = fast_runner_fixture(spec=PerceptionSpec(), adam_steps=2, compiled=False)
        old_execution = restored._latent_wm_execution
        old_execution._graph = object()  # A stale captured graph must never survive model restore.
        old_execution._calls = 99
        source.iteration = 17
        with tempfile.TemporaryDirectory() as directory, patch("torch.cuda.is_available", return_value=False):
            path = Path(directory) / "fast.pt"
            source.save(path, export=False)
            payload = torch.load(path, map_location="cpu", weights_only=False)
            with patch("src.wmac_perceptive.runner.read_depth_frame", return_value=restored.current_depth):
                restored.load(path)
        self.assertIsNot(restored._latent_wm_execution, old_execution)
        self.assertIsNone(restored._latent_wm_execution._graph)
        self.assertEqual(restored._latent_wm_execution._calls, 0)
        self.assertIs(restored._latent_wm_execution.kernel.world_model, restored.world_model)
        self.assertIs(restored._latent_wm_execution.kernel.vision, restored.actor.vision)
        self.assertIs(restored._latent_wm_execution.kernel.decoder, restored.latent_decoder)
        for name in (*MODULES, "latent_decoder", *OPTIMIZERS):
            self.assert_tree_equal(getattr(restored, name).state_dict(), payload[name])


if __name__ == "__main__":
    unittest.main()
