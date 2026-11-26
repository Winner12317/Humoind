"""Tensor-only WM loss and guarded execution against the original eager fit."""
from __future__ import annotations

import copy
import os
from types import SimpleNamespace
import unittest

import torch

from src.wmac_perceptive.latent_execution import LatentWorldModelKernel, WMExecution, safe_optimizer_step
from src.wmac_perceptive.latent_training import rollout_latent_model_metrics
from test_perceptive_latent_runner import latent_runner_fixture, populated_latent_replay


def fixture(recon_images=5):
    runner, starts, _ = latent_runner_fixture(batch_size=3, horizon=3)
    runner.cfg.training_schedule = "validated_v2"
    runner.cfg.latent_recon_images = recon_images
    replay = populated_latent_replay(runner, starts, steps=5)
    batch = replay.sample_windows(3, 3, 0., fields=runner.WM_FIELDS)
    batch["mask"] = batch["mask"].new_tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0]])
    return runner, batch


def reference_loss(runner, batch):
    values = rollout_latent_model_metrics(runner, batch, update_statistics=True)
    mask = batch["mask"]
    dyn = runner._masked_mean(values["model_error"], mask)
    contact = runner._masked_mean(values["contact_error"], mask)
    latent = (values["latent_error"]*mask).sum()/len(mask)
    stack = batch["next_depth_stack"][mask.bool()].float()[:runner.cfg.latent_recon_images]
    recon = (runner.latent_decoder(runner.actor.encode_processed(stack))-stack).abs().mean()
    return dyn+contact+runner.cfg.latent_loss_coef*latent+runner.cfg.latent_recon_coef*recon


class LatentExecutionTest(unittest.TestCase):
    def test_original_wm_warmup_counts_ten_minibatches_per_macro(self):
        runner, _ = fixture()
        runner.cfg.training_schedule, runner.cfg.wm_updates = "supermimic_v1", 10
        self.assertEqual(WMExecution(runner, mode="eager", warmup=3).warmup, 30)
        runner.cfg.training_schedule = "validated_v2"
        self.assertEqual(WMExecution(runner, mode="eager", warmup=3).warmup, 3)

    def test_official_schedule_applies_nonfinite_update_instead_of_silently_skipping(self):
        runner, batch = fixture()
        runner.cfg.training_schedule = "supermimic_v1"
        execution = WMExecution(runner, mode="eager")
        self.assertFalse(execution.guard_nonfinite)
        batch["next_state"][0, 0, 0] = float("nan")
        _, norm, applied = execution.step(batch)
        self.assertTrue(bool(applied))
        self.assertFalse(bool(torch.isfinite(norm)))
        self.assertTrue(any(not bool(torch.isfinite(parameter).all()) for parameter in execution._parameters))
        self.assertTrue(all(float(state["step"]) == 1 for state in runner.opt_model.state.values()))

    def test_unguarded_optimizer_matches_unconditional_adam_with_nan_loss_and_finite_gradient(self):
        parameter = torch.nn.Parameter(torch.tensor([1.]))
        reference = torch.nn.Parameter(parameter.detach().clone())
        optimizer, official = (torch.optim.Adam([value], lr=.01) for value in (parameter, reference))
        parameter.grad = torch.tensor([.5])
        reference.grad = parameter.grad.clone()
        applied = safe_optimizer_step(torch.tensor(float("nan")), torch.tensor(.5), optimizer, guard=False)
        official.step()
        self.assertTrue(bool(applied))
        torch.testing.assert_close(parameter, reference, rtol=0, atol=0)
        for name, value in optimizer.state[parameter].items():
            torch.testing.assert_close(value, official.state[reference][name], rtol=0, atol=0)

    def test_fixed_shape_selection_loss_and_all_model_gradients_match_original_eager(self):
        for limit in (1, 5, 100):
            with self.subTest(recon_images=limit):
                source, batch = fixture(limit)
                target, _ = fixture(limit)
                target._model_training_modules.load_state_dict(source._model_training_modules.state_dict())
                kernel = LatentWorldModelKernel(target)
                prepared = kernel.prepare(batch)
                actual, metrics = kernel(prepared)
                expected = reference_loss(source, batch)
                torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
                self.assertEqual(metrics.shape, (len(kernel.metric_names),))
                self.assertEqual(float(metrics[-1]), 1.)
                actual.backward(); expected.backward()
                for (name, first), (_, second) in zip(kernel.named_parameters(), source._model_training_modules.named_parameters()):
                    with self.subTest(parameter=name):
                        self.assertIsNotNone(first.grad)
                        self.assertIsNotNone(second.grad)
                        torch.testing.assert_close(first.grad, second.grad, rtol=2e-5, atol=2e-7)
                for name in ("z_mean", "z_std", "dz_std"):
                    torch.testing.assert_close(getattr(target.world_model, name), getattr(source.world_model, name), rtol=0, atol=0)
                self.assertFalse(prepared["target_latent"].requires_grad)
                self.assertFalse(prepared["initial_latent"].requires_grad)

    def test_fullgraph_aot_compiled_forward_backward_matches_eager(self):
        source, batch = fixture()
        target, _ = fixture()
        target._model_training_modules.load_state_dict(source._model_training_modules.state_dict())
        first, second = LatentWorldModelKernel(source), LatentWorldModelKernel(target)
        eager_batch, compiled_batch = first.prepare(batch), second.prepare(batch)
        expected, expected_metrics = first(eager_batch)
        compiled = torch.compile(second, backend="aot_eager", fullgraph=True, dynamic=False)
        actual, metrics = compiled(compiled_batch)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-7)
        torch.testing.assert_close(metrics, expected_metrics, rtol=1e-6, atol=1e-7)
        actual.backward(); expected.backward()
        for left, right in zip(second.parameters(), first.parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=2e-5, atol=2e-7)

    def test_guarded_step_changes_joint_modules_but_rejects_nan_without_touching_adam(self):
        runner, batch = fixture()
        execution = WMExecution(runner, mode="eager")
        before = {name: value.clone() for name, value in execution.kernel.state_dict().items()}
        metrics, norm, ok = execution(batch)
        self.assertTrue(ok)
        self.assertTrue(bool(torch.isfinite(norm)))
        self.assertEqual(metrics.shape, (len(execution.metric_names),))
        self.assertTrue(any(not torch.equal(value, before[name])
                            for name, value in execution.kernel.state_dict().items() if not name.endswith(("z_mean", "z_std", "dz_std"))))
        parameters = [parameter.detach().clone() for parameter in execution.kernel.parameters()]
        optimizer = copy.deepcopy(runner.opt_model.state_dict())
        broken = {name: value.clone() for name, value in batch.items()}
        broken["next_state"][0, 0, 0] = float("nan")
        _, _, ok = execution.step(broken)
        self.assertFalse(ok)
        for saved, current in zip(parameters, execution.kernel.parameters()):
            torch.testing.assert_close(current, saved, rtol=0, atol=0)
        restored = runner.opt_model.state_dict()
        for identifier, state in optimizer["state"].items():
            for name, value in state.items():
                torch.testing.assert_close(restored["state"][identifier][name], value, rtol=0, atol=0)

    def test_graph_mode_is_explicitly_unavailable_on_cpu(self):
        runner, _ = fixture()
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            WMExecution(runner, mode="graph")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA fused-Adam protocol regression")
    def test_cuda_fused_adam_rejects_nan_inf_loss_gradients_and_recovers_without_moment_changes(self):
        parameter = torch.nn.Parameter(torch.tensor([.3, -.2, .8], device="cuda"))
        optimizer = torch.optim.Adam([parameter], lr=.01, fused=True)
        parameter.grad = torch.ones_like(parameter)
        self.assertTrue(bool(safe_optimizer_step(parameter.new_tensor(1.), parameter.new_tensor(1.), optimizer)))
        for problem in ("nan_loss", "inf_loss", "nan_gradient", "inf_gradient"):
            with self.subTest(problem=problem):
                before = parameter.detach().clone()
                state = {key: value.clone() for key, value in optimizer.state[parameter].items()}
                parameter.grad = torch.ones_like(parameter)
                loss = parameter.new_tensor(1.)
                if problem.endswith("loss"):
                    loss.fill_(float("nan") if problem.startswith("nan") else float("inf"))
                else:
                    parameter.grad[0] = float("nan") if problem.startswith("nan") else float("inf")
                norm = torch.nn.utils.clip_grad_norm_([parameter], 1.)
                self.assertFalse(bool(safe_optimizer_step(loss, norm, optimizer)))
                torch.testing.assert_close(parameter, before, rtol=0, atol=0)
                for key, value in state.items():
                    torch.testing.assert_close(optimizer.state[parameter][key], value, rtol=0, atol=0)
                self.assertFalse(hasattr(optimizer, "found_inf"))
                self.assertFalse(hasattr(optimizer, "grad_scale"))
                parameter.grad = torch.ones_like(parameter)
                norm = torch.nn.utils.clip_grad_norm_([parameter], 1.)
                self.assertTrue(bool(safe_optimizer_step(parameter.new_tensor(1.), norm, optimizer)))
                self.assertGreater(float((parameter-before).abs().sum()), 0.)
                torch.testing.assert_close(optimizer.state[parameter]["step"], state["step"]+1, rtol=0, atol=0)

    @unittest.skipUnless(os.environ.get("PERCEPTIVE_TEST_CUDA_GRAPH") == "1" and torch.cuda.is_available(),
                         "optional CUDA forward/backward graph check")
    def test_cuda_forward_backward_graph_replays_and_skips_a_bad_update_then_recovers(self):
        runner, batch = fixture()
        runner._model_training_modules.cuda().float()
        runner.opt_model = torch.optim.Adam(runner._model_training_modules.parameters(), lr=.001, fused=True)
        batch = {key: value.cuda().float() if value.is_floating_point() else value.cuda()
                 for key, value in batch.items()}
        execution = WMExecution(runner, mode="graph", warmup=3)
        retained = []
        for index in range(5):
            if index == 4:
                runner.opt_model.zero_grad(set_to_none=True)
            metrics, norm, ok = execution(batch)
            self.assertTrue(bool(ok))
            self.assertTrue(norm.is_cuda)
            self.assertTrue(bool(torch.isfinite(norm)))
            retained.append(metrics)
        self.assertIsNotNone(execution._graph)
        self.assertNotEqual(retained[-1].data_ptr(), retained[-2].data_ptr())
        parameters = [parameter.detach().clone() for parameter in execution.kernel.parameters()]
        state = copy.deepcopy(runner.opt_model.state_dict())
        broken = {key: value.clone() for key, value in batch.items()}
        broken["next_state"][0, 0, 0] = float("nan")
        _, _, ok = execution(broken)
        self.assertFalse(bool(ok))
        for before, after in zip(parameters, execution.kernel.parameters()):
            torch.testing.assert_close(after, before, rtol=0, atol=0)
        for identifier, moments in state["state"].items():
            for name, value in moments.items():
                torch.testing.assert_close(runner.opt_model.state_dict()["state"][identifier][name], value, rtol=0, atol=0)
        _, norm, ok = execution(batch)
        self.assertTrue(bool(ok))
        self.assertTrue(bool(torch.isfinite(norm)))

    def _production_graph_case(self, mode="graph", optimizer_factory=None):
        from src.wmac_perceptive.instinct_latent import InstinctLatentDecoder
        from src.wmac_perceptive.latent_models import LatentPerceptiveActor, LatentPerceptiveWorldModel
        from src.wmac_perceptive.perception import PerceptionSpec
        from test_perceptive_runner import PlaneOracle

        torch.manual_seed(782)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        spec, count, steps = PerceptionSpec(), 64, 8
        actor = LatentPerceptiveActor(spec)
        actor.vision.cuda()
        world = LatentPerceptiveWorldModel(PlaneOracle(), spec, hidden=(2048, 2048, 2048)).cuda()
        world.precision = actor.precision = "tf32"
        decoder = InstinctLatentDecoder(128, 16).cuda()
        modules = torch.nn.ModuleList((world, actor.vision, decoder))
        optimizer = (torch.optim.Adam(modules.parameters(), lr=torch.tensor(2e-4, device="cuda"),
                                      fused=True, capturable=True) if optimizer_factory is None
                     else optimizer_factory(modules))
        cfg = SimpleNamespace(wm_window=steps, latent_dim=128, latent_recon_images=512,
            latent_loss_coef=1., latent_recon_coef=1., latent_stat_rate=.01,
            precision="tf32", max_grad_norm=1., compile_mode="default")
        runner = SimpleNamespace(world_model=world, actor=actor, latent_decoder=decoder,
                                 opt_model=optimizer, cfg=cfg)
        state = torch.zeros(count, steps, 71, device="cuda")
        state[..., 2], state[..., 3], state[..., 7], state[..., 42:] = .8, 1., .2, .01
        successor = state.clone(); successor[..., 0] += .004; successor[..., 13:42] += .0002
        image = torch.rand(count, steps, 16, 18, 32, device="cuda"); image[..., 1::2, :, :] = 1.
        target = torch.rand_like(image); target[..., 1::2, :, :] = 1.
        mask = torch.ones(count, steps, device="cuda")
        mask[::7, -2:] = 0.; mask[::11, 3:] = 0.
        batch = dict(state=state, next_state=successor, action=torch.randn(count, steps, 29, device="cuda")*.1,
            context=torch.zeros(count, steps, 35, device="cuda"), contact_cost=torch.zeros(count, steps, 2, device="cuda"),
            depth_stack=image, next_depth_stack=target, mask=mask)
        execution = WMExecution(runner, mode=mode, warmup=3)
        representatives = (world.net[-1].weight, actor.vision.conv[0].weight, decoder.dec[-1].weight)
        for index in range(12):
            if index in (5, 10):
                optimizer.zero_grad(set_to_none=True)
                self.assertTrue(all(parameter.grad is None for parameter in execution._parameters))
            before = [parameter.detach().clone() for parameter in representatives]
            metrics, norm, ok = execution.step(batch)
            self.assertTrue(metrics.is_cuda and norm.is_cuda and ok.is_cuda)
            self.assertTrue(bool(ok))
            self.assertGreater(float(norm), 0.)
            for previous, parameter in zip(before, representatives):
                self.assertGreater(float((parameter-previous).abs().sum()), 0.)
            if execution._graph is not None:
                for parameter, gradient in zip(execution._parameters, execution._captured_gradients):
                    self.assertIs(parameter.grad, gradient)
        for parameter in execution._parameters:
            if "step" in optimizer.state[parameter]:
                self.assertEqual(int(optimizer.state[parameter]["step"]), 12)
        snapshots = [parameter.detach().clone() for parameter in execution._parameters]
        moments = copy.deepcopy(optimizer.state_dict())
        broken = {key: value.clone() for key, value in batch.items()}
        broken["next_state"][0, 0, 0] = float("nan")
        optimizer.zero_grad(set_to_none=True)
        _, norm, ok = execution.step(broken)
        self.assertTrue(norm.is_cuda)
        self.assertFalse(bool(ok))
        for parameter, saved in zip(execution._parameters, snapshots):
            torch.testing.assert_close(parameter, saved, rtol=0, atol=0)
        for identifier, values in moments["state"].items():
            for name, saved in values.items():
                torch.testing.assert_close(optimizer.state_dict()["state"][identifier][name], saved, rtol=0, atol=0)
        optimizer.zero_grad(set_to_none=True)
        _, norm, ok = execution.step(batch)
        self.assertTrue(norm.is_cuda)
        self.assertTrue(bool(ok))
        self.assertGreater(float(norm), 0.)
        for parameter in execution._parameters:
            if "step" in optimizer.state[parameter]:
                self.assertEqual(int(optimizer.state[parameter]["step"]), 13)
        if mode == "graph_full":
            # Learning-rate values may change in place without recapturing;
            # replacing LR storage must be rejected before the next replay.
            old_lr = optimizer.param_groups[0]["lr"]
            old_lr.fill_(1e-4)
            _, _, ok = execution.step(batch)
            self.assertTrue(bool(ok))
            optimizer.param_groups[0]["lr"] = old_lr.clone()
            with self.assertRaisesRegex(RuntimeError, "moved or was rebound"):
                execution.step(batch)
            optimizer.param_groups[0]["lr"] = old_lr
            old_buffer = world.z_mean
            world.z_mean = old_buffer.clone()
            with self.assertRaisesRegex(RuntimeError, "moved or was rebound"):
                execution.step(batch)
            world.z_mean = old_buffer
            parameter = execution._parameters[0]
            state_key = next(key for key, value in optimizer.state[parameter].items()
                             if isinstance(value, torch.Tensor) and value.ndim > 0)
            old_moment = optimizer.state[parameter][state_key]
            optimizer.state[parameter][state_key] = old_moment.clone()
            with self.assertRaisesRegex(RuntimeError, "moved or was rebound"):
                execution.step(batch)
            optimizer.state[parameter][state_key] = old_moment
            old_parameter = world.net[0].weight
            world.net[0].weight = torch.nn.Parameter(old_parameter.detach().clone())
            with self.assertRaisesRegex(RuntimeError, "moved or was rebound"):
                execution.step(batch)
            world.net[0].weight = old_parameter

    @unittest.skipUnless(os.environ.get("PERCEPTIVE_TEST_PRODUCTION_GRAPH") == "1" and torch.cuda.is_available(),
                         "optional production B64/K8/2048 WM graph regression")
    def test_production_cuda_graph_keeps_grad_bindings_across_outer_phase_clears(self):
        self._production_graph_case()

    @unittest.skipUnless(os.environ.get("PERCEPTIVE_TEST_PRODUCTION_GRAPH_FULL") == "1" and torch.cuda.is_available(),
                         "optional production full-step CUDA graph regression")
    def test_production_cuda_full_graph_native_adam_skip_recovery_and_pointer_guards(self):
        self._production_graph_case(mode="graph_full")

    @unittest.skipUnless(os.environ.get("PERCEPTIVE_TEST_PRODUCTION_GRAPH_FULL") == "1" and torch.cuda.is_available(),
                         "optional production Muon/Adam full-step CUDA graph regression")
    def test_production_cuda_full_graph_reference_muon_adam_skip_recovery_and_pointer_guards(self):
        from src.wmac_perceptive.optimizers import build_reference_optimizer
        self._production_graph_case(mode="graph_full", optimizer_factory=lambda modules:
            build_reference_optimizer(modules, lr=2e-4, device="cuda"))

    @unittest.skipUnless(os.environ.get("PERCEPTIVE_TEST_INDUCTOR") == "1" and torch.cuda.is_available(),
                         "optional CUDA Inductor code-generation check")
    def test_inductor_fullgraph_forward_backward(self):
        runner, batch = fixture()
        runner._model_training_modules.cuda().float()
        runner.opt_model = torch.optim.Adam(runner._model_training_modules.parameters(), lr=.001, fused=True)
        batch = {key: value.cuda().float() if value.is_floating_point() else value.cuda()
                 for key, value in batch.items()}
        execution = WMExecution(runner, mode="compile")
        metrics, norm, ok = execution(batch)
        self.assertTrue(ok)
        self.assertTrue(bool(torch.isfinite(metrics).all()))
        self.assertTrue(bool(torch.isfinite(norm)))


if __name__ == "__main__":
    unittest.main()
