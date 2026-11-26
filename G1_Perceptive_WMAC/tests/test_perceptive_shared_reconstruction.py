"""Independent loss/gradient oracles for sharing one WM reconstruction encoding.

The actual v2 G1 physical model uses a small depth-only sensor/network here.
The original eager reconstruction path remains an independent reference. No
codes may survive an optimizer update; all WM targets stay stop-gradient.
"""
from __future__ import annotations

from dataclasses import replace
import os
import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.wmac.networks import frozen_parameters
from src.wmac_perceptive.instinct_latent import InstinctLatentDecoder
from src.wmac_perceptive.latent_execution import LatentWorldModelKernel
from src.wmac_perceptive.models import DepthEncoder
from src.wmac_perceptive.optimizers import build_reference_optimizer
from src.wmac_perceptive.wm_macro_execution import WMMacroExecution
from test_hiking_training_v2 import v2_reference_fixture
from test_perceptive_latent_execution import reference_loss


CUDA = os.environ.get("PERCEPTIVE_TEST_SHARED_RECONSTRUCTION_CUDA") == "1" and torch.cuda.is_available()


def fixture(limit=4, *, codes=False, few_valid=False, device="cpu"):
    runner, starts, _ = v2_reference_fixture()
    runner.spec = replace(runner.spec, include_valid=False)
    runner.actor.spec = runner.spec
    channels = runner.spec.frames
    runner.actor.vision = DepthEncoder(channels, runner.spec.height, runner.spec.width,
        128, runner.spec, include_valid=False).double()
    runner.latent_decoder = InstinctLatentDecoder(128, channels, (4, 4)).double()
    runner.cfg.latent_recon_images = limit
    runner.cfg.wm_updates = 10
    runner.cfg.max_grad_norm = 1.
    runner.cfg.compile_fullgraph = True
    runner._model_training_modules = nn.ModuleList((runner.world_model, runner.actor.vision,
                                                   runner.latent_decoder))
    count, steps = 3, runner.cfg.wm_window
    expand = lambda tensor: tensor[:, None].expand(count, steps, *tensor.shape[1:]).clone()
    generator = torch.Generator().manual_seed(112218)
    image_shape = (channels, 4, 4)
    batch = dict(state=expand(starts["state"]), next_state=expand(starts["state"]),
        action=expand(starts["previous_action"]), context=expand(starts["context"]),
        contact_cost=torch.rand(count, steps, 13, generator=generator, dtype=torch.float64) * .2,
        depth_stack=torch.rand(count, 1, *image_shape, generator=generator),
        next_depth_stack=torch.rand(count, steps, *image_shape, generator=generator),
        mask=torch.tensor([[1, 0, 1], [1, 1, 0], [0, 1, 1]], dtype=torch.bool))
    if few_valid:
        batch["mask"] = torch.tensor([[1, 0, 0], [0, 0, 0], [0, 0, 1]], dtype=torch.bool)
    batch["next_state"][..., 13:42] += .001
    if codes:
        for name in ("depth_stack", "next_depth_stack"):
            batch[name] = (batch[name] * 255).round().to(torch.uint8)
    if device != "cpu":
        runner._model_training_modules.to(device).float()
        runner.device = torch.device(device)
        runner.cfg.precision = "float32"
        runner.world_model.precision = "float32"
        batch = {name: value.to(device=device, dtype=torch.float32 if value.is_floating_point() else value.dtype)
                 for name, value in batch.items()}
    runner.opt_model = build_reference_optimizer((runner._model_training_modules,), .001, device=device)
    return runner, batch


def decode(batch):
    return {name: (value.float() / 255. if value.dtype == torch.uint8 else value)
            for name, value in batch.items()}


def paired(limit=4, **kwargs):
    first, batch = fixture(limit, **kwargs)
    second, _ = fixture(limit, **kwargs)
    second._model_training_modules.load_state_dict(first._model_training_modules.state_dict())
    return first, second, batch


def assert_gradients(test, first, second, *, rtol=3e-5, atol=3e-7):
    for (name, left), (other, right) in zip(first.named_parameters(), second.named_parameters()):
        test.assertEqual(name, other)
        with test.subTest(parameter=name):
            if left.grad is None:
                test.assertIsNone(right.grad)
            else:
                test.assertIsNotNone(right.grad)
                torch.testing.assert_close(left.grad, right.grad, rtol=rtol, atol=atol)


class SharedReconstructionTest(unittest.TestCase):
    def test_loss_visual_decoder_physical_gradients_and_stats_match_independent_original(self):
        for limit, codes, sparse in ((4, False, False), (5, True, True), (100, False, False)):
            with self.subTest(limit=limit, uint8=codes, padded_selection=sparse):
                source, target, batch = paired(limit, codes=codes, few_valid=sparse)
                kernel = LatentWorldModelKernel(target)
                prepared = kernel.prepare_shared(batch)
                actual, _ = kernel(prepared)
                # This oracle encodes all targets under no_grad, then encodes
                # only actual valid reconstruction rows again with gradients.
                expected = reference_loss(source, decode(batch))
                torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-8)
                actual.backward(); expected.backward()
                assert_gradients(self, source._model_training_modules, target._model_training_modules)
                for name in ("z_mean", "z_std", "dz_std"):
                    torch.testing.assert_close(getattr(source.world_model, name),
                                               getattr(target.world_model, name), rtol=1e-10, atol=1e-12)
                self.assertFalse(prepared["initial_latent"].requires_grad)
                self.assertFalse(prepared["target_latent"].requires_grad)
                self.assertTrue(prepared["reconstruction_latent"].requires_grad)

    def test_stable_selection_inverse_scatter_and_exactly_one_visual_backward_encoding(self):
        runner, batch = fixture(4, codes=True)
        kernel = LatentWorldModelKernel(runner)
        calls = []
        handle = kernel.vision.projection.register_forward_hook(
            lambda module, inputs, output: calls.append((len(output), torch.is_grad_enabled())))
        try:
            with patch.object(runner.world_model, "update_z_stats", wraps=runner.world_model.update_z_stats) as stats:
                prepared = kernel.prepare_shared(batch)
                kernel(prepared)
                self.assertEqual(stats.call_count, 1)
        finally:
            handle.remove()
        self.assertEqual(calls, [(3, False), (4, True), (5, False)])
        mask = batch["mask"].flatten()
        original_order = list(range(mask.numel()))
        order = sorted(original_order, key=lambda index: not bool(mask[index]))
        selected = torch.tensor(order[:4])
        flat = batch["next_depth_stack"].flatten(0, 1).float() / 255.
        torch.testing.assert_close(prepared["reconstruction_stacks"], flat[selected], rtol=0., atol=0.)
        torch.testing.assert_close(prepared["reconstruction_mask"], mask[selected], rtol=0., atol=0.)
        with torch.no_grad():
            full = runner.actor.encode_processed(decode(batch)["next_depth_stack"])
        torch.testing.assert_close(prepared["target_latent"], full, rtol=1e-10, atol=1e-12)

    def test_encoder_gradient_comes_only_from_reconstruction_and_stats_can_be_frozen(self):
        runner, batch = fixture()
        kernel = LatentWorldModelKernel(runner)
        before = {name: getattr(runner.world_model, name).clone() for name in ("z_mean", "z_std", "dz_std")}
        prepared = kernel.prepare_shared(batch, update_statistics=False)
        loss, _ = kernel(prepared)
        parameters = tuple(kernel.vision.parameters())
        full_gradient = torch.autograd.grad(loss, parameters, retain_graph=True)
        error = (kernel.decoder(prepared["reconstruction_latent"]) - prepared["reconstruction_stacks"]).abs().flatten(1).mean(-1)
        valid = prepared["reconstruction_mask"]
        independent_reconstruction = error[valid].mean() * kernel.recon_weight
        recon_gradient = torch.autograd.grad(independent_reconstruction, parameters)
        self.assertGreater(sum(float(value.abs().sum()) for value in full_gradient), 0.)
        for left, right in zip(full_gradient, recon_gradient):
            torch.testing.assert_close(left, right, rtol=2e-8, atol=2e-10)
        for name, value in before.items():
            torch.testing.assert_close(getattr(runner.world_model, name), value, rtol=0., atol=0.)

    def test_no_grad_fallback_and_temporary_encoder_freeze_restore_correct_gradient_boundary(self):
        runner, batch = fixture()
        kernel = LatentWorldModelKernel(runner)
        with torch.no_grad():
            prepared = kernel.prepare_shared(batch, update_statistics=False)
        self.assertNotIn("reconstruction_latent", prepared)
        with frozen_parameters(kernel.vision):
            frozen = kernel.prepare_shared(batch, update_statistics=False)
            self.assertFalse(frozen["reconstruction_latent"].requires_grad)
        live = kernel.prepare_shared(batch, update_statistics=False)
        self.assertTrue(live["reconstruction_latent"].requires_grad)
        self.assertFalse(live["initial_latent"].requires_grad)
        self.assertFalse(live["target_latent"].requires_grad)

    def test_three_optimizer_updates_reencode_online_instead_of_reusing_stale_latents(self):
        source, target, batch = paired()
        old, shared = LatentWorldModelKernel(source), LatentWorldModelKernel(target)
        latents = []
        for _ in range(3):
            source.opt_model.zero_grad(set_to_none=True)
            target.opt_model.zero_grad(set_to_none=True)
            expected, _ = old(old.prepare(batch))
            prepared = shared.prepare_shared(batch)
            latents.append(prepared["reconstruction_latent"].detach().clone())
            actual, _ = shared(prepared)
            expected.backward(); actual.backward()
            assert_gradients(self, old, shared)
            for runner in (source, target):
                torch.nn.utils.clip_grad_norm_(runner._model_training_modules.parameters(), 1., foreach=True)
                runner.opt_model.step()
            for name, value in source._model_training_modules.state_dict().items():
                torch.testing.assert_close(target._model_training_modules.state_dict()[name], value,
                    rtol=3e-5, atol=3e-7)
        self.assertTrue(all(not torch.equal(left, right) for left, right in zip(latents, latents[1:])))

    def test_separately_compiled_prepare_and_loss_backward_match_eager_without_stat_version_errors(self):
        source, target, batch = paired(codes=True)
        eager, shared = LatentWorldModelKernel(source), LatentWorldModelKernel(target)
        prepare = torch.compile(shared.prepare_shared, backend="aot_eager", fullgraph=True, dynamic=False)
        loss_function = torch.compile(shared, backend="aot_eager", fullgraph=True, dynamic=False)
        for _ in range(2):
            source.opt_model.zero_grad(set_to_none=True)
            target.opt_model.zero_grad(set_to_none=True)
            expected, metrics = eager(eager.prepare(batch))
            prepared = prepare(batch)
            actual, actual_metrics = loss_function(prepared)
            torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-8)
            torch.testing.assert_close(actual_metrics, metrics, rtol=2e-6, atol=2e-8)
            actual.backward(); expected.backward()
            assert_gradients(self, eager, shared)
            source.opt_model.step(); target.opt_model.step()


@unittest.skipUnless(CUDA, "set PERCEPTIVE_TEST_SHARED_RECONSTRUCTION_CUDA=1 for small CUDA checks")
class SharedReconstructionCudaTest(unittest.TestCase):
    def test_gpu_single_step_legacy_loss_gradients_and_statistics_match_shared(self):
        source, target, batch = paired(codes=True, device="cuda")
        old, shared = LatentWorldModelKernel(source), LatentWorldModelKernel(target)
        expected, metrics = old(old.prepare(batch))
        actual, actual_metrics = shared(shared.prepare_shared(batch))
        torch.testing.assert_close(actual, expected, rtol=2e-4, atol=1e-6)
        torch.testing.assert_close(actual_metrics, metrics, rtol=2e-4, atol=1e-6)
        expected.backward(); actual.backward()
        assert_gradients(self, old, shared, rtol=3e-4, atol=1e-6)
        for name in ("z_mean", "z_std", "dz_std"):
            torch.testing.assert_close(getattr(source.world_model, name),
                                       getattr(target.world_model, name), rtol=2e-5, atol=1e-7)

    def test_actual_small_wm_macro_cross_prepare_backward_captures_and_replays(self):
        source, target, batch = paired(codes=True, device="cuda")
        source.cfg.compile = target.cfg.compile = True
        # Capture equivalence uses identical compiled arithmetic. Comparing
        # sixty low-precision Muon steps across different eager/Inductor and
        # encoder batch partitions is not a bitwise trajectory contract; the
        # independent legacy loss/gradient oracle is checked above and on CPU.
        reference = WMMacroExecution(source, backend="inductor", cuda_graphs=False)
        execution = WMMacroExecution(target, backend="inductor", cuda_graphs=True)
        captures = []
        for macro in range(1, 7):
            expected = reference.step_macro([batch] * 10)
            actual = execution.step_macro([batch] * 10)
            torch.cuda.synchronize()
            torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-7)
            for name, value in source._model_training_modules.state_dict().items():
                torch.testing.assert_close(target._model_training_modules.state_dict()[name], value,
                    rtol=2e-5, atol=2e-7)
            self.assertTrue(all(float(value["step"]) == macro * 10 for value in target.opt_model.state.values()
                                if "step" in value))
            captures.append(execution._graph is not None)
            target.opt_model.zero_grad(set_to_none=True)
        self.assertEqual(captures, [False, False, False, True, True, True])


if __name__ == "__main__":
    unittest.main()
