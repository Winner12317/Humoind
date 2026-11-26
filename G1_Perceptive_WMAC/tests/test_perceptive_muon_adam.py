"""Pinned PyTorch Muon math, official partition and capturable finite guards."""

from __future__ import annotations

import ast
import copy
import hashlib
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from src.wmac_perceptive.optimizers import (
    NS_COEFFICIENTS, PYTORCH_MUON_REFERENCE_SHA256, REFERENCE_OPTIMIZER_ALGORITHM,
    ReferenceMuonAdam, _zeropower_via_newtonschulz, build_reference_optimizer,
    optimizer_algorithm_name, validate_reference_optimizer_state,
)


def network(device="cpu", dtype=torch.float32):
    return nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 6), nn.Tanh(), nn.Linear(6, 2)).to(device=device, dtype=dtype)


def assert_tree_equal(test, actual, expected, *, atol=0., rtol=0.):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    elif isinstance(expected, dict):
        test.assertEqual(actual.keys(), expected.keys())
        for key in expected:
            assert_tree_equal(test, actual[key], expected[key], atol=atol, rtol=rtol)
    elif isinstance(expected, (tuple, list)):
        test.assertEqual(len(actual), len(expected))
        for left, right in zip(actual, expected):
            assert_tree_equal(test, left, right, atol=atol, rtol=rtol)
    else:
        test.assertEqual(actual, expected)


def official_muon_functions():
    path = Path(os.environ.get("PYTORCH_MUON_REFERENCE", "/tmp/pytorch_v2_10_muon.py"))
    if not path.exists():
        raise unittest.SkipTest("Pinned PyTorch v2.10 Muon source unavailable")
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != PYTORCH_MUON_REFERENCE_SHA256:
        raise AssertionError("PyTorch v2.10 Muon source hash differs")
    tree = ast.parse(source, filename=str(path))
    names = {"_zeropower_via_newtonschulz", "_adjust_lr", "_single_tensor_muon"}
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    namespace = dict(torch=torch, Tensor=torch.Tensor, math=math, _to_scalar=lambda value: value)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


class ReferencePartitionAndStateTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(801)

    def test_middle_linear_weights_only_and_shared_parameters_are_deduplicated(self):
        model = network()
        other = nn.Sequential(nn.Linear(2, 3), nn.LayerNorm(3), nn.Linear(3, 1))
        conv = nn.Conv2d(2, 4, 3)
        optimizer = build_reference_optimizer([model, other, conv, model], .0002)
        groups = {group["kind"]: group for group in optimizer.param_groups}
        self.assertEqual([id(p) for p in groups["muon"]["params"]], [id(model[2].weight)])
        expected = {id(p) for module in (model, other, conv) for p in module.parameters()}
        self.assertEqual({id(p) for group in optimizer.param_groups for p in group["params"]}, expected)
        self.assertEqual(sum(len(group["params"]) for group in optimizer.param_groups), len(expected))
        self.assertEqual(groups["muon"]["weight_decay"], .1)
        self.assertEqual(groups["adam"]["weight_decay"], 0.)
        self.assertEqual(groups["muon"]["adjust_lr_fn"], "match_rms_adamw")
        self.assertEqual(optimizer_algorithm_name(optimizer), REFERENCE_OPTIMIZER_ALGORITHM)
        self.assertTrue(all(torch.is_tensor(group["lr"]) and group["lr"].ndim == 0 for group in optimizer.param_groups))

    def test_adam_only_path_matches_native_adam_parameters_and_moments(self):
        model = nn.Linear(3, 2).double()
        reference = copy.deepcopy(model)
        optimizer = build_reference_optimizer([model], .0002, betas=(.9, .95))
        native = torch.optim.Adam(reference.parameters(), lr=torch.tensor(.0002), betas=(.9, .95),
                                  foreach=False, fused=False, capturable=False)
        self.assertEqual(len(optimizer.param_groups), 1)
        for _ in range(4):
            for parameter, expected in zip(model.parameters(), reference.parameters()):
                gradient = torch.randn_like(parameter)
                parameter.grad, expected.grad = gradient.clone(), gradient.clone()
            optimizer.step()
            native.step()
        for parameter, expected in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(parameter, expected, atol=0, rtol=0)
            assert_tree_equal(self, optimizer.state[parameter], native.state[expected])

    def test_cpu_finite_guard_preserves_every_state_and_parameter_then_recovers(self):
        model = network()
        optimizer = build_reference_optimizer([model], .0002)
        model(torch.randn(3, 4)).square().mean().backward()
        optimizer.step()
        weights, state = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
        optimizer.found_inf, optimizer.grad_scale = torch.tensor(1.), None
        for parameter in model.parameters():
            parameter.grad.fill_(float("nan"))
        optimizer.step()
        assert_tree_equal(self, model.state_dict(), weights)
        assert_tree_equal(self, optimizer.state_dict(), state)
        optimizer.found_inf.zero_()
        optimizer.zero_grad(set_to_none=True)
        model(torch.randn(3, 4)).square().mean().backward()
        optimizer.step()
        self.assertTrue(any(not torch.equal(value, weights[name]) for name, value in model.state_dict().items()))

    def test_strict_roundtrip_and_reject_malformed_algorithm_momentum_and_adam_states_atomically(self):
        model = network()
        optimizer = build_reference_optimizer([model], .0002)
        model(torch.randn(3, 4)).square().mean().backward()
        optimizer.step()
        saved = copy.deepcopy(optimizer.state_dict())
        receiver = build_reference_optimizer([copy.deepcopy(model)], .001)
        receiver.load_state_dict(saved)
        assert_tree_equal(self, receiver.state_dict(), saved)
        adam_id = saved["param_groups"][0]["params"][0]
        muon_id = saved["param_groups"][1]["params"][0]
        for problem in ("marker", "kind", "names", "ids", "negative_id", "muon_shape", "muon_nan", "adam_negative", "step", "sparse", "lr_sparse"):
            bad = copy.deepcopy(saved)
            if problem == "marker":
                bad.pop("algorithm_name")
            elif problem == "kind":
                bad["param_groups"][1]["kind"] = "adam"
            elif problem == "names":
                bad["param_groups"][0]["param_names"].reverse()
            elif problem == "ids":
                bad["param_groups"][0]["params"].reverse()
            elif problem == "negative_id":
                bad["param_groups"][0]["params"][0] = -1
            elif problem == "muon_shape":
                bad["state"][muon_id]["momentum_buffer"] = torch.zeros(1)
            elif problem == "muon_nan":
                bad["state"][muon_id]["momentum_buffer"].flatten()[0] = float("nan")
            elif problem == "adam_negative":
                bad["state"][adam_id]["exp_avg_sq"].flatten()[0] = -.1
            elif problem == "step":
                bad["state"][adam_id]["step"].fill_(.5)
            elif problem == "sparse":
                bad["state"][muon_id]["momentum_buffer"] = bad["state"][muon_id]["momentum_buffer"].to_sparse()
            else:
                bad["param_groups"][0]["lr"] = bad["param_groups"][0]["lr"].to_sparse()
            before = copy.deepcopy(receiver.state_dict())
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                receiver.load_state_dict(bad)
            assert_tree_equal(self, receiver.state_dict(), before)


class PinnedMuonSourceParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.reference = official_muon_functions()

    def test_newton_schulz_matches_source_for_tall_wide_square_and_zero_matrices(self):
        torch.manual_seed(802)
        for shape in ((8, 3), (3, 8), (5, 5)):
            for dtype in (torch.float32, torch.float64):
                for zero in (False, True):
                    gradient = torch.zeros(shape, dtype=dtype) if zero else torch.randn(shape, dtype=dtype)
                    expected = self.reference._zeropower_via_newtonschulz(gradient.clone(), NS_COEFFICIENTS, 5, 1e-7)
                    actual = _zeropower_via_newtonschulz(gradient.clone())
                    torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_muon_parameter_and_momentum_updates_match_pinned_source_across_steps(self):
        torch.manual_seed(803)
        for shape in ((8, 3), (3, 8), (5, 5)):
            parameter = nn.Parameter(torch.randn(shape, dtype=torch.float64))
            expected = parameter.detach().clone()
            buffer = torch.zeros_like(expected)
            optimizer = ReferenceMuonAdam([], [parameter], lr=.0002)
            for _ in range(4):
                gradient = torch.randn_like(parameter)
                parameter.grad = gradient.clone()
                optimizer.step()
                self.reference._single_tensor_muon([expected], [gradient.clone()], [buffer], lr=.0002,
                    weight_decay=.1, momentum=.95, nesterov=True, ns_coefficients=NS_COEFFICIENTS,
                    ns_steps=5, eps=1e-7, adjust_lr_fn="match_rms_adamw", has_complex=False)
                torch.testing.assert_close(optimizer.state[parameter]["momentum_buffer"], buffer, atol=0, rtol=0)
                torch.testing.assert_close(parameter, expected, atol=3e-11, rtol=3e-11)
                torch.testing.assert_close(parameter.grad, gradient, atol=0, rtol=0)

    def test_partition_matches_official_ast_with_deduplication(self):
        checkout = Path(os.environ.get("SUPERMIMIC_REFERENCE_CHECKOUT", "/tmp/supermimic-official-audit-20261111"))
        path = checkout / "src/supermimic/learners/learner_utils.py"
        if not path.exists():
            self.skipTest("Official learner utility checkout unavailable")
        tree = ast.parse(path.read_bytes())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "build_optimizers")
        def capture(parameters, **kwargs):
            return SimpleNamespace(parameters=list(parameters), kwargs=kwargs)
        fake_torch = SimpleNamespace(tensor=torch.tensor, optim=SimpleNamespace(Adam=capture, Muon=capture, Optimizer=torch.optim.Optimizer))
        namespace = dict(torch=fake_torch, nn=nn, Sequence=list)
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(path), "exec"), namespace)
        model = network()
        official = namespace["build_optimizers"]([model], lr=.0002)
        local = build_reference_optimizer([model], .0002)
        self.assertEqual([[id(p) for p in group["params"]] for group in local.param_groups],
                         [[id(p) for p in optimizer.parameters] for optimizer in official])


@unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
class ReferenceOptimizerCudaTest(unittest.TestCase):
    def test_fused_adam_muon_device_guard_skips_nan_inf_and_preserves_all_state(self):
        torch.manual_seed(804)
        model = network("cuda:0")
        optimizer = build_reference_optimizer([model], .0002, device="cuda:0")
        model(torch.randn(3, 4, device="cuda:0")).square().mean().backward()
        optimizer.step()
        for bad_value in (float("nan"), float("inf")):
            weights, state = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
            optimizer.found_inf, optimizer.grad_scale = torch.ones((), device="cuda:0"), None
            for parameter in model.parameters():
                parameter.grad.fill_(bad_value)
            optimizer.step()
            assert_tree_equal(self, model.state_dict(), weights)
            assert_tree_equal(self, optimizer.state_dict(), state)
        optimizer.found_inf.zero_()
        optimizer.zero_grad(set_to_none=True)
        model(torch.randn(3, 4, device="cuda:0")).square().mean().backward()
        optimizer.step()
        self.assertTrue(all(bool(torch.isfinite(value).all()) for state in optimizer.state.values() for value in state.values()))

    def test_full_cuda_graph_loss_backward_clip_optimizer_lr_and_skip_recovery(self):
        torch.manual_seed(805)
        model = network("cuda:0")
        optimizer = build_reference_optimizer([model], .0002, device="cuda:0")
        image = torch.randn(8, 4, device="cuda:0")
        optimizer.found_inf, optimizer.grad_scale = torch.zeros((), device="cuda:0"), None
        def update():
            optimizer.zero_grad(set_to_none=False)
            loss = model(image).square().mean()
            loss.backward()
            norm = nn.utils.clip_grad_norm_(model.parameters(), 1., foreach=True)
            optimizer.found_inf.copy_((~(torch.isfinite(loss) & torch.isfinite(norm))).float())
            optimizer.step()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                update()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            update()
        graph.replay()
        torch.cuda.synchronize()
        state_addresses = {id(value): value.data_ptr() for state in optimizer.state.values() for value in state.values()}
        for bad_value in (float("nan"), float("inf")):
            weights, state = copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict())
            image.fill_(bad_value)
            graph.replay()
            torch.cuda.synchronize()
            self.assertEqual(float(optimizer.found_inf), 1.)
            assert_tree_equal(self, model.state_dict(), weights)
            assert_tree_equal(self, optimizer.state_dict(), state)
        image.normal_()
        graph.replay()
        torch.cuda.synchronize()
        self.assertEqual(float(optimizer.found_inf), 0.)
        self.assertTrue(any(not torch.equal(value, weights[name]) for name, value in model.state_dict().items()))
        before = copy.deepcopy(model.state_dict())
        lr_addresses = [group["lr"].data_ptr() for group in optimizer.param_groups]
        for group in optimizer.param_groups:
            group["lr"].zero_()
        graph.replay()
        torch.cuda.synchronize()
        assert_tree_equal(self, model.state_dict(), before)
        self.assertEqual(lr_addresses, [group["lr"].data_ptr() for group in optimizer.param_groups])
        self.assertEqual(state_addresses, {id(value): value.data_ptr() for state in optimizer.state.values() for value in state.values()})


if __name__ == "__main__":
    unittest.main()
