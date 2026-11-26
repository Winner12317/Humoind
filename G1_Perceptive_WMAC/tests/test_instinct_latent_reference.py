"""Latent decoder/statistics/loss parity against pinned MIT InstinctMJ files.

The optional source-parity checks use an existing hash-checked checkout and
never download code or start a simulator. Set INSTINCTMJ_REFERENCE_CHECKOUT to
another checkout of commit 03563ee02665f327425106372264070315673748.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from src.wmac_perceptive.instinct_latent import (
    InstinctLatentDecoder, normalized_latent_errors, update_latent_statistics,
)


REFERENCE = Path(os.environ.get("INSTINCTMJ_REFERENCE_CHECKOUT", "/tmp/instinctmj-supermimic-20261111"))
SOURCE_HASHES = {
    "networks.py": "b6356b59cef81a01b1c86ced8247f83a7060fab9407a7ad943f8425e092e38f5",
    "runner.py": "507f1c726f700597b6db89a20b1e70fd0d1ec6eddb7f3fa00a149896dc557e85",
}


def stats_holder(dim=7, dtype=torch.float64):
    return SimpleNamespace(z_dim=dim, z_mean=torch.linspace(-.3, .2, dim, dtype=dtype),
                           z_std=torch.linspace(.2, .8, dim, dtype=dtype),
                           dz_std=torch.linspace(.1, .9, dim, dtype=dtype))


def reference_source(name):
    path = REFERENCE/"src/instinct_mj/rl/supermimic"/name
    if not path.is_file():
        raise unittest.SkipTest("Optional pinned InstinctMJ reference checkout is absent")
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != SOURCE_HASHES[name]:
        raise AssertionError(f"Pinned InstinctMJ source hash differs: {name}")
    return ast.parse(source, filename=str(path))


def execute_ast(nodes):
    namespace = {"torch": torch, "nn": nn, "AMP_DTYPE": None}
    tree = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(tree, "<pinned InstinctMJ latent reference>", "exec"), namespace)
    return namespace


def reference_decoder_and_stats():
    tree = reference_source("networks.py")
    decoder = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "LatentDecoder")
    world_model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "WorldModel")
    stats = copy.deepcopy(next(node for node in world_model.body
                               if isinstance(node, ast.FunctionDef) and node.name == "update_z_stats"))
    stats.name = "reference_update_z_stats"
    namespace = execute_ast([decoder, stats])
    return namespace["LatentDecoder"], namespace["reference_update_z_stats"]


def reference_errors():
    tree = reference_source("runner.py")
    runner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "SuperMimicRunner")
    update = next(node for node in runner.body if isinstance(node, ast.FunctionDef) and node.name == "wm_update")
    assignments = {}
    for node in ast.walk(update):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if node.targets[0].id in {"z_scale", "lz", "latent_loss"}:
                assignments[node.targets[0].id] = copy.deepcopy(node)
    if set(assignments) != {"z_scale", "lz", "latent_loss"}:
        raise AssertionError("Pinned upstream latent-loss statements have changed")
    definition = ast.parse("def reference_errors(predicted, target, mask):\n"
                           "    Z_hat = predicted\n    z_tgt = target\n    Bz = target.shape[0]\n").body[0]
    definition.body.extend(assignments[name] for name in ("z_scale", "lz", "latent_loss"))
    definition.body.append(ast.Return(value=ast.Tuple(elts=[ast.Name(id="lz", ctx=ast.Load()),
        ast.Name(id="latent_loss", ctx=ast.Load())], ctx=ast.Load())))
    return execute_ast([definition])["reference_errors"]


class InstinctLatentInvariantTest(unittest.TestCase):
    def test_decoder_preserves_prefix_and_supports_local_depth_valid_channels_and_tiny_crop(self):
        torch.manual_seed(443)
        decoder = InstinctLatentDecoder(12, 16, output_shape=(4, 6)).double()
        latent = torch.randn(2, 3, 12, dtype=torch.float64, requires_grad=True)
        decoded = decoder(latent)
        self.assertEqual(decoded.shape, (2, 3, 16, 4, 6))
        self.assertEqual(decoded.dtype, torch.float32)
        self.assertTrue(bool(((decoded >= 0) & (decoded <= 1)).all()))
        decoded.square().mean().backward()
        self.assertTrue(bool(torch.isfinite(latent.grad).all()))
        self.assertGreater(float(latent.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all() for p in decoder.parameters()))

    def test_single_sample_and_empty_statistics_remain_finite(self):
        holder = stats_holder(3)
        initial = copy.deepcopy(holder)
        z = torch.tensor([[[.1, .2, .3]]], dtype=torch.float64, requires_grad=True)
        update_latent_statistics(holder, z, torch.zeros_like(z), torch.zeros(1, 1), rate=1.)
        torch.testing.assert_close(holder.z_mean, z.detach().reshape(3), rtol=0, atol=0)
        torch.testing.assert_close(holder.z_std, z.new_full((3,), .001), rtol=0, atol=0)
        torch.testing.assert_close(holder.dz_std, z.new_full((3,), .001), rtol=0, atol=0)
        self.assertTrue(all(not getattr(holder, name).requires_grad for name in ("z_mean", "z_std", "dz_std")))
        update_latent_statistics(initial, z.new_empty((0, 2, 3)), z.new_empty((0, 2, 3)), z.new_empty((0, 2)))
        for name in ("z_mean", "z_std", "dz_std"):
            torch.testing.assert_close(getattr(initial, name), getattr(stats_holder(3), name), rtol=0, atol=0)

    def test_scalar_target_variance_and_masked_loss_have_known_values(self):
        target = torch.tensor([[[0., 2.], [2., 4.]]], dtype=torch.float64)
        predicted = target+torch.tensor([[[1., 1.], [2., 2.]]], dtype=torch.float64)
        # Each dimension has unbiased sample variance 2; one scalar scale is 2.
        errors = normalized_latent_errors(predicted, target)
        torch.testing.assert_close(errors, torch.tensor([[.5, 2.]], dtype=torch.float64), rtol=0, atol=0)
        mask = torch.tensor([[1., 0.]], dtype=torch.float64)
        self.assertEqual(float((errors*mask).sum()/target.shape[0]), .5)
        singleton = torch.zeros(1, 1, 2, dtype=torch.float64)
        torch.testing.assert_close(normalized_latent_errors(singleton+1, singleton), singleton.new_full((1, 1), 1e6))

    def test_target_gradients_are_callers_responsibility_and_real_nan_is_not_hidden(self):
        target = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
        predicted = torch.randn_like(target, requires_grad=True)
        normalized_latent_errors(predicted, target).sum().backward()
        self.assertIsNotNone(target.grad)
        self.assertTrue(bool(torch.isfinite(target.grad).all()))
        broken = target.detach().clone(); broken[0, 0, 0] = float("nan")
        self.assertFalse(bool(torch.isfinite(normalized_latent_errors(predicted.detach(), broken)).all()))


class InstinctLatentPinnedReferenceTest(unittest.TestCase):
    def test_decoder_production_forward_and_input_parameter_gradients_match_upstream(self):
        torch.manual_seed(490)
        reference_class, _ = reference_decoder_and_stats()
        source = reference_class(13, channels=16).double()
        local = InstinctLatentDecoder(13, 16).double()
        local.load_state_dict(source.state_dict(), strict=True)
        first = torch.randn(2, 13, dtype=torch.float64, requires_grad=True)
        second = first.detach().clone().requires_grad_()
        expected, actual = source(first), local(second)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        multiplier = torch.randn_like(expected)
        (expected*multiplier).sum().backward(); (actual*multiplier).sum().backward()
        torch.testing.assert_close(second.grad, first.grad, rtol=0, atol=0)
        for left, right in zip(local.parameters(), source.parameters()):
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)

    def test_repeated_ema_statistics_and_weighted_increment_rms_match_upstream(self):
        torch.manual_seed(491)
        _, reference_update = reference_decoder_and_stats()
        actual, expected = stats_holder(), stats_holder()
        for rate in (.01, .2, 1.):
            z, dz = torch.randn(4, 3, 7, dtype=torch.float64), torch.randn(4, 3, 7, dtype=torch.float64)
            mask = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.float64)
            update_latent_statistics(actual, z, dz, mask, rate)
            reference_update(expected, z, dz, mask, rate)
            for name in ("z_mean", "z_std", "dz_std"):
                torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)

    def test_variance_normalized_errors_masked_total_and_gradients_match_upstream(self):
        torch.manual_seed(492)
        reference = reference_errors()
        target = torch.randn(3, 4, 7, dtype=torch.float64)
        predicted = torch.randn_like(target, requires_grad=True)
        source_prediction = predicted.detach().clone().requires_grad_()
        mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.float64)
        expected, expected_total = reference(source_prediction, target, mask)
        actual = normalized_latent_errors(predicted, target)
        actual_total = (actual*mask).sum()/predicted.shape[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_total, expected_total, rtol=0, atol=0)
        actual_total.backward(); expected_total.backward()
        torch.testing.assert_close(predicted.grad, source_prediction.grad, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
