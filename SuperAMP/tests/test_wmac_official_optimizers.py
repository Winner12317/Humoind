"""Pinned PyTorch Muon math, official partition and capturable finite guards."""

from __future__ import annotations

import ast
import copy
import hashlib
import math
from types import SimpleNamespace
import unittest

import torch
from torch import nn

from src.wmac.optimizers import (
    NS_COEFFICIENTS, PYTORCH_MUON_REFERENCE_SHA256, REFERENCE_OPTIMIZER_ALGORITHM,
    ReferenceMuonAdam, _zeropower_via_newtonschulz, build_reference_optimizer,
    optimizer_algorithm_name, validate_reference_optimizer_state, set_learning_rate,
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


# Unmodified functions extracted from the pinned PyTorch v2.10 source above.
# This self-contained source fixture keeps parity checks independent of /tmp,
# the network, and availability of torch.optim.Muon in the training environment.
_PYTORCH_FUNCTIONS = r'''def _zeropower_via_newtonschulz(
    grad: Tensor, ns_coefficients: tuple[float, float, float], ns_steps: int, eps: float
) -> Tensor:
    """
    Newton-Schulz iteration to compute the zeroth power / orthogonalization of G. We opt to use a
    quintic iteration whose coefficients are selected to maximize the slope at zero. For the purpose
    of minimizing steps, it turns out to be empirically effective to keep increasing the slope at
    zero even beyond the point where the iteration no longer converges all the way to one everywhere
    on the interval. This iteration therefore does not produce UV^T but rather something like US'V^T
    where S' is diagonal with S_{ii}' ~ Uniform(0.5, 1.5), which turns out not to hurt model
    performance at all relative to UV^T, where USV^T = G is the SVD.

    Implementation reference: https://github.com/KellerJordan/Muon/blob/master/muon.py
    with suggestions by @jxbz, @leloykun, and @YouJiacheng.
    """
    if ns_steps >= 100:
        raise ValueError(
            "Number of steps must be less than 100 for computational efficiency"
        )
    if len(grad.shape) != 2:
        raise ValueError("Input tensor gradient must be a 2D matrix")
    if len(ns_coefficients) != 3:
        raise ValueError("Coefficients must be a tuple of exactly 3 values")
    a, b, c = ns_coefficients
    ortho_grad = grad.bfloat16()
    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    # Ensure spectral norm is at most 1
    ortho_grad.div_(ortho_grad.norm().clamp(min=eps))
    # Perform the NS iterations
    for _ in range(ns_steps):
        gram_matrix = ortho_grad @ ortho_grad.T
        gram_update = torch.addmm(
            gram_matrix, gram_matrix, gram_matrix, beta=b, alpha=c
        )
        ortho_grad = torch.addmm(ortho_grad, gram_update, ortho_grad, beta=a)

    if grad.size(0) > grad.size(1):
        ortho_grad = ortho_grad.T
    return ortho_grad

def _adjust_lr(lr: float, adjust_lr_fn: str | None, param_shape: torch.Size) -> float:
    """Default learning rate adjustment used by Muon."""
    A, B = param_shape[:2]

    if adjust_lr_fn is None or adjust_lr_fn == "original":
        # pyrefly: ignore [no-matching-overload]
        adjusted_ratio = math.sqrt(max(1, A / B))
    elif adjust_lr_fn == "match_rms_adamw":
        adjusted_ratio = 0.2 * math.sqrt(max(A, B))
    else:
        adjusted_ratio = 1.0
    return lr * adjusted_ratio

def _single_tensor_muon(
    params: list[Tensor],
    grads: list[Tensor],
    muon_momentum_bufs: list[Tensor],
    *,
    lr: float,
    weight_decay: float,
    momentum: float,
    nesterov: bool,
    ns_coefficients: tuple[float, float, float],
    ns_steps: int,
    eps: float,
    adjust_lr_fn: str | None,
    has_complex: bool,
) -> None:
    lr = _to_scalar(lr)
    if has_complex:
        raise ValueError("Complex parameters are not supported")

    for i, param in enumerate(params):
        grad = grads[i]
        if grad.ndim != 2:
            raise ValueError("Param gradient must be a 2D matrix")

        buf = muon_momentum_bufs[i]
        buf.lerp_(grad, 1 - momentum)
        update = grad.lerp(buf, momentum) if nesterov else buf

        update = _zeropower_via_newtonschulz(update, ns_coefficients, ns_steps, eps)

        adjusted_lr = _adjust_lr(lr, adjust_lr_fn, param.shape)

        param.mul_(1 - lr * weight_decay)
        param.add_(update, alpha=-adjusted_lr)'''
_PYTORCH_FUNCTIONS_SHA256 = 'eb09295cb60883224b51d452dc5c495c629f835c50cbeddb86b7c426a1e2b3de'
_OFFICIAL_PARTITION = r'''def build_optimizers(
  modules: Sequence[nn.Module],
  lr: float,
  betas: tuple[float, float] = (0.9, 0.999),
  device: str = "cpu",
) -> list[torch.optim.Optimizer]:
  """Muon on the Linear weights between the first and the last Linear of each
  ``nn.Sequential`` in ``modules``, fused capturable Adam on the rest, both at
  ``lr``. Muon's ``lr`` is a float, fixed once an update is captured."""
  hidden = [
    linear.weight
    for module in modules
    for seq in module.modules()
    if isinstance(seq, nn.Sequential)
    for linear in [m for m in seq if isinstance(m, nn.Linear)][1:-1]
  ]
  hidden_ids = {id(p) for p in hidden}
  rest = [p for m in modules for p in m.parameters() if id(p) not in hidden_ids]
  adam = torch.optim.Adam(
    rest, lr=torch.tensor(lr, device=device), betas=betas, fused=True, capturable=True
  )
  if not hidden:
    return [adam]
  return [adam, torch.optim.Muon(hidden, lr=lr, adjust_lr_fn="match_rms_adamw")]'''
_OFFICIAL_PARTITION_SHA256 = '85a46b6b6cc230248995cd1b86864a1141d62d62c97073c6e67bdd203690f7b4'


def official_muon_functions():
    assert hashlib.sha256(_PYTORCH_FUNCTIONS.encode()).hexdigest() == _PYTORCH_FUNCTIONS_SHA256
    namespace = dict(torch=torch, Tensor=torch.Tensor, math=math, _to_scalar=lambda value: value)
    exec(compile(_PYTORCH_FUNCTIONS, "pytorch-v2.10.0/_muon.py:source-fixture", "exec"), namespace)
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
        self.assertEqual(hashlib.sha256(_OFFICIAL_PARTITION.encode()).hexdigest(), _OFFICIAL_PARTITION_SHA256)
        tree = ast.parse(_OFFICIAL_PARTITION)
        function = tree.body[0]
        def capture(parameters, **kwargs):
            return SimpleNamespace(parameters=list(parameters), kwargs=kwargs)
        fake_torch = SimpleNamespace(tensor=torch.tensor, optim=SimpleNamespace(Adam=capture, Muon=capture, Optimizer=torch.optim.Optimizer))
        namespace = dict(torch=fake_torch, nn=nn, Sequence=list)
        exec(compile(ast.Module(body=[function], type_ignores=[]), "supermimic-abc124c1/learner_utils.py:source-fixture", "exec"), namespace)
        model = network()
        official = namespace["build_optimizers"]([model], lr=.0002)
        local = build_reference_optimizer([model], .0002)
        self.assertEqual([[id(p) for p in group["params"]] for group in local.param_groups],
                         [[id(p) for p in optimizer.parameters] for optimizer in official])




class LearningRateCompatibilityTest(unittest.TestCase):
    def test_schedule_preserves_tensor_addresses_and_supports_scalar_override(self):
        model = network()
        optimizer = build_reference_optimizer([model], .0002)
        addresses = [group["lr"].data_ptr() for group in optimizer.param_groups]
        set_learning_rate(optimizer, .0001)
        self.assertEqual(addresses, [group["lr"].data_ptr() for group in optimizer.param_groups])
        for group in optimizer.param_groups:
            torch.testing.assert_close(group["lr"], torch.tensor(.0001), atol=0, rtol=0)
        for group in optimizer.param_groups:
            group["lr"] = 0.0
        before = copy.deepcopy(model.state_dict())
        model(torch.randn(3, 4)).square().mean().backward()
        optimizer.step()
        assert_tree_equal(self, model.state_dict(), before)
        saved = copy.deepcopy(optimizer.state_dict())
        self.assertTrue(all(torch.is_tensor(group["lr"]) for group in saved["param_groups"]))
        receiver = build_reference_optimizer([copy.deepcopy(model)], .0002)
        receiver.load_state_dict(saved)
        assert_tree_equal(self, receiver.state_dict(), saved)

    def test_invalid_lr_fails_before_any_group_changes_and_adam_keeps_scalar_lr(self):
        model = network()
        optimizer = build_reference_optimizer([model], .0002)
        before = copy.deepcopy(optimizer.state_dict())
        for bad in (-.1, float("nan"), float("inf"), True):
            with self.subTest(lr=bad), self.assertRaises(ValueError):
                set_learning_rate(optimizer, bad)
            assert_tree_equal(self, optimizer.state_dict(), before)
        native = torch.optim.Adam(model.parameters(), lr=.001)
        set_learning_rate(native, .0003)
        self.assertEqual(native.param_groups[0]["lr"], .0003)

    def test_old_pure_adam_checkpoint_is_rejected_without_mutation(self):
        model = network()
        receiver = build_reference_optimizer([model], .0002)
        before = copy.deepcopy(receiver.state_dict())
        old = torch.optim.Adam(model.parameters(), lr=.0002).state_dict()
        with self.assertRaisesRegex(ValueError, "algorithm marker"):
            receiver.load_state_dict(old)
        assert_tree_equal(self, receiver.state_dict(), before)


if __name__ == "__main__":
    unittest.main()
