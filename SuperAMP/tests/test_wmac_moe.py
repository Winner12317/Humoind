"""CPU sparse-dispatch, branch-gradient and deployment-equivalence checks."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from src.wmac.moe import SparseMoEActor
from src.wmac.networks import Actor


def small_actor(experts=4, top_k=2):
    torch.manual_seed(537)
    actor = SparseMoEActor(obs_dim=4, action_dim=2, hidden=(7, 5),
        num_experts=experts, top_k=top_k, gate_hidden=(),
        output_init_scale=1., zero_output_bias=False).double()
    with torch.no_grad():
        actor.router[-1].weight.zero_()
        actor.router[-1].bias.copy_(torch.linspace(3., -3., experts, dtype=torch.float64))
    return actor


def dense_oracle(actor, observation):
    """All experts on all tokens, independently scatter the selected mixture."""
    normalized = actor.norm(observation)
    logits = actor.router(normalized)/actor.temperature
    probabilities = logits.softmax(-1)
    selected, indices = probabilities.topk(actor.top_k, dim=-1)
    weights = selected/selected.sum(-1, keepdim=True)
    dense_weights = torch.zeros_like(probabilities).scatter(-1, indices, weights)
    proposal = sum(expert(normalized)*dense_weights[..., expert_id, None]
                   for expert_id, expert in enumerate(actor.experts))
    return proposal if actor.action_limit is None else actor.action_limit*torch.tanh(proposal/actor.action_limit)


class SparseMoEActorTest(unittest.TestCase):
    def test_top1_and_invalid_hyperparameters_are_refused(self):
        for kwargs in ({"top_k": 1}, {"top_k": 0}, {"num_experts": 2, "top_k": 3},
                       {"num_experts": 1}, {"num_experts": True}, {"top_k": True},
                       {"temperature": 0.}, {"temperature": float("nan")},
                       {"action_limit": 0.}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                SparseMoEActor(obs_dim=4, action_dim=2, hidden=(3,), gate_hidden=(), **kwargs)

    def test_actual_expert_token_work_is_sparse_and_unused_experts_have_no_gradient(self):
        actor = small_actor()
        inputs = torch.randn(2, 3, 4, dtype=torch.float64, requires_grad=True)
        seen = [0]*actor.num_experts
        calls = [0]*actor.num_experts
        handles = []
        for expert_id, expert in enumerate(actor.experts):
            def record(module, args, output, index=expert_id):
                calls[index] += 1
                seen[index] += len(args[0])
            handles.append(expert.register_forward_hook(record))
        try:
            actor(inputs).square().sum().backward()
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(seen, [6, 6, 0, 0])
        self.assertEqual(calls, [1, 1, 0, 0])
        self.assertEqual(sum(seen), inputs.numel()//4*actor.top_k)
        for expert in actor.experts[2:]:
            self.assertTrue(all(parameter.grad is None for parameter in expert.parameters()))
        self.assertGreater(sum(float(parameter.grad.abs().sum()) for parameter in actor.router.parameters()), 0.)
        self.assertTrue(torch.isfinite(inputs.grad).all())

    def test_sparse_values_and_selected_branch_gradients_match_dense_oracle(self):
        sparse, dense = small_actor(), small_actor()
        original = torch.randn(5, 4, dtype=torch.float64)
        first, second = original.clone().requires_grad_(True), original.clone().requires_grad_(True)
        actual, expected = sparse(first), dense_oracle(dense, second)
        torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
        sparse_grad = torch.autograd.grad(actual.square().sum(), (first, *sparse.parameters()), allow_unused=True)
        dense_grad = torch.autograd.grad(expected.square().sum(), (second, *dense.parameters()), allow_unused=True)
        for actual_grad, expected_grad in zip(sparse_grad, dense_grad):
            if actual_grad is None:
                self.assertIsNotNone(expected_grad)
                torch.testing.assert_close(expected_grad, torch.zeros_like(expected_grad), rtol=0, atol=0)
            else:
                torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(sparse.forward_dense(first.detach()), actual.detach(), rtol=1e-12, atol=1e-12)

    def test_local_input_gradient_matches_finite_difference_inside_one_selected_subset(self):
        actor = small_actor(experts=3)
        with torch.no_grad():
            actor.router[-1].weight.normal_(0., .03)
        observation = torch.tensor([.15, -.2, .3, .1], dtype=torch.float64, requires_grad=True)
        direction = torch.tensor([.2, -.1, .5, .3], dtype=torch.float64)
        direction /= direction.norm()
        weights = observation.new_tensor([.3, -.2])
        objective = (actor(observation)*weights).sum()
        derivative = torch.autograd.grad(objective, observation)[0].dot(direction)
        epsilon = 1e-6
        plus, minus = observation.detach()+epsilon*direction, observation.detach()-epsilon*direction
        self.assertTrue(torch.equal(actor.route(plus).indices, actor.route(minus).indices))
        finite_difference = ((actor(plus)*weights).sum()-(actor(minus)*weights).sum())/(2*epsilon)
        torch.testing.assert_close(derivative, finite_difference, rtol=2e-6, atol=1e-9)

    def test_shapes_dense_k_action_transform_and_forward_normalizers_are_stateless(self):
        for leading in ((), (5,), (2, 3), (0,)):
            for top_k in (2, 4):
                with self.subTest(leading=leading, top_k=top_k):
                    actor = small_actor(top_k=top_k)
                    inputs = torch.randn(*leading, 4, dtype=torch.float64)
                    before = {key: value.clone() for key, value in actor.norm.state_dict().items()}
                    output = actor(inputs)
                    route = actor.route(inputs)
                    self.assertEqual(output.shape, (*leading, 2))
                    self.assertEqual(route.logits.shape, (*leading, 4))
                    self.assertEqual(route.indices.shape, (*leading, top_k))
                    torch.testing.assert_close(route.weights.sum(-1), torch.ones(leading, dtype=torch.float64))
                    torch.testing.assert_close(route.probabilities.sum(-1), torch.ones(leading, dtype=torch.float64))
                    for key, value in actor.norm.state_dict().items():
                        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
                    actor.export_dense = True
                    torch.testing.assert_close(actor(inputs), output, rtol=1e-12, atol=1e-12)
                    for module in actor.modules():
                        self.assertFalse(any(isinstance(value, torch.Tensor) and value.grad_fn is not None
                                             for value in vars(module).values()))
        actor = small_actor()
        actor.action_limit = None
        observation = torch.randn(4, 4, dtype=torch.float64)*100
        unbounded = actor(observation)
        actor.action_limit = 5.
        torch.testing.assert_close(actor(observation), 5*torch.tanh(unbounded/5))

    def test_mlp_clone_preserves_actions_and_noise_explicitly_breaks_expert_symmetry(self):
        original = Actor(4, 2, (7, 5), 5., output_init_scale=1., zero_output_bias=False).double()
        with torch.no_grad():
            original.norm.mean.copy_(torch.tensor([.1, -.1, .2, -.2], dtype=torch.float64))
            original.norm.var.fill_(2.)
            original.norm.count.fill_(12.)
        actor = small_actor()
        inputs = torch.randn(6, 4, dtype=torch.float64)
        state = copy.deepcopy(original.state_dict())
        rng = torch.get_rng_state().clone()
        actor.load_mlp_state_dict(state)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)
        torch.testing.assert_close(actor(inputs), original(inputs), rtol=1e-12, atol=1e-12)
        for expert in actor.experts:
            for key, value in expert.state_dict().items():
                torch.testing.assert_close(value, original.net.state_dict()[key], rtol=0, atol=0)
        actor(inputs).square().sum().backward()
        self.assertLess(sum(float(parameter.grad.abs().sum()) for parameter in actor.router.parameters()), 1e-10)
        actor.zero_grad(set_to_none=True)
        actor.load_mlp_state_dict(state, noise_std=.01)
        self.assertFalse(torch.equal(actor.experts[0][-1].weight, actor.experts[1][-1].weight))
        actor(inputs).square().sum().backward()
        self.assertGreater(sum(float(parameter.grad.abs().sum()) for parameter in actor.router.parameters()), 1e-9)
        for key, value in original.state_dict().items():
            torch.testing.assert_close(value, state[key], rtol=0, atol=0)
        before = copy.deepcopy(actor.state_dict())
        with self.assertRaisesRegex(ValueError, "MLP"):
            actor.load_mlp_state_dict({**state, "unexpected": torch.zeros(1)})
        for key, value in actor.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_ppo_import_reuses_original_actor_mapping(self):
        original = Actor(4, 2, (7, 5), 5.)
        payload = {"model_state_dict": {f"actor.{key}": value for key, value in original.net.state_dict().items()},
                   "obs_norm_state_dict": {"_mean": original.norm.mean[None],
                       "_var": original.norm.var[None], "count": original.norm.count}}
        actor = SparseMoEActor(4, 2, (7, 5), 5., num_experts=3, gate_hidden=())
        actor.load_baseline(payload)
        inputs = torch.randn(4, 4)
        torch.testing.assert_close(actor(inputs), original(inputs), rtol=1e-6, atol=1e-7)
        actor.load_baseline(payload, noise_std=.01)
        self.assertFalse(torch.equal(actor.experts[0][-1].weight, actor.experts[1][-1].weight))
        for expert in actor.experts:
            # Explicit symmetry-breaking noise changes only the final weights.
            for key, value in expert.state_dict().items():
                if key != f"{len(expert)-1}.weight":
                    torch.testing.assert_close(value, original.net.state_dict()[key], rtol=0, atol=0)

    def test_sparse_dispatch_can_graph_break_without_losing_input_or_router_gradients(self):
        actor = small_actor()
        compiled = torch.compile(actor, backend="eager", fullgraph=False)
        inputs = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        expected = actor(inputs).detach()
        output = compiled(inputs)
        torch.testing.assert_close(output, expected, rtol=1e-12, atol=1e-12)
        output.square().sum().backward()
        self.assertTrue(torch.isfinite(inputs.grad).all())
        self.assertGreater(sum(float(parameter.grad.abs().sum()) for parameter in actor.router.parameters()), 0.)

    def test_dense_export_matches_sparse_in_onnxruntime_with_dynamic_batch(self):
        if importlib.util.find_spec("onnxruntime") is None:
            self.skipTest("onnxruntime is not installed")
        import onnxruntime as ort
        actor = SparseMoEActor(4, 2, (7, 5), 5., num_experts=3, gate_hidden=()).eval()
        export = copy.deepcopy(actor)
        export.export_dense = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"moe.onnx"
            torch.onnx.export(export, torch.zeros(2, 4), str(path), input_names=["obs"], output_names=["actions"],
                dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}}, opset_version=18, dynamo=False)
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            inputs = torch.randn(5, 4)
            actual = session.run(None, {"obs": inputs.numpy()})[0]
            np.testing.assert_allclose(actual, actor(inputs).detach().numpy(), rtol=2e-5, atol=2e-7)


if __name__ == "__main__":
    unittest.main()
