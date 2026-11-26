"""Paper-gate Hiking MoE parity, dense derivatives and historical isolation.

The reference is deliberately built from plain PyTorch layers: four ELU
experts, a 256-wide ELU gate, and full Softmax output fusion. It does not import the
project's MLP helper or any upstream training/environment dependency.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch
from torch import nn

from src.wmac.actors import actor_architecture, create_actor
from src.wmac.checkpoints import checkpoint_actor_architecture, resolve_checkpoint_config
from src.wmac.config import WmacConfig
from src.wmac.hiking import HikingMoEActor
from src.wmac.moe import SparseMoEActor
from src.wmac.networks import Actor


class HikingReference(nn.Module):
    """Independent dense reference for the paper gate and historical code gate."""

    def __init__(self, observation_width=4, action_width=2, gate_hidden=(256,)):
        super().__init__()
        gate_layers = []
        width = observation_width
        for next_width in gate_hidden:
            gate_layers.extend((nn.Linear(width, next_width), nn.ELU()))
            width = next_width
        gate_layers.append(nn.Linear(width, 4))
        self.router = nn.Sequential(*gate_layers)
        experts = []
        for _ in range(4):
            layers = []
            width = observation_width
            for next_width in (256, 128, 64):
                layers.extend((nn.Linear(width, next_width), nn.ELU()))
                width = next_width
            layers.append(nn.Linear(width, action_width))
            experts.append(nn.Sequential(*layers))
        self.experts = nn.ModuleList(experts)

    def forward(self, observations):
        weights = self.router(observations).softmax(dim=-1)
        proposals = torch.stack([expert(observations) for expert in self.experts], dim=-2)
        return (weights.unsqueeze(-1) * proposals).sum(dim=-2)


def hiking_config(**changes):
    return replace(WmacConfig(device="cpu", replay_device="cpu", compile=False,
                              precision="float32"), actor_type="moe", **changes)


def legacy_config():
    return hiking_config(moe_variant="legacy", policy_hidden=(8, 4),
                         moe_num_experts=3, moe_top_k=2, moe_gate_hidden=(5,),
                         moe_temperature=.8, moe_load_balance_coef=.01,
                         moe_router_z_coef=.001)


class HikingMoEParityTest(unittest.TestCase):
    def test_new_default_moe_matches_hiking_paper_gate_and_mlp_stays_original(self):
        cfg = WmacConfig()
        self.assertEqual(cfg.actor_type, "mlp")
        self.assertEqual(cfg.moe_variant, "hiking")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(813)
            old = Actor(hidden=cfg.policy_hidden, action_limit=cfg.action_limit,
                        output_init_scale=cfg.policy_output_init_scale,
                        zero_output_bias=cfg.policy_output_zero_bias)
            torch.manual_seed(813)
            current = create_actor(cfg)
        self.assertIs(type(current), Actor)
        self.assertEqual(set(old.state_dict()), set(current.state_dict()))
        for name, value in old.state_dict().items():
            torch.testing.assert_close(current.state_dict()[name], value, rtol=0, atol=0)
        cfg = replace(cfg, actor_type="moe")
        cfg.validate()
        actor = create_actor(cfg)
        self.assertIs(type(actor), HikingMoEActor)
        self.assertEqual((actor.obs_dim, actor.action_dim), (384, 29))
        self.assertEqual((actor.num_experts, actor.top_k), (4, 4))
        self.assertEqual(actor.hidden, (256, 128, 64))
        self.assertEqual(actor.gate_hidden, (256,))
        self.assertEqual(actor.temperature, 1.)
        self.assertFalse(actor.normalize_observations)
        self.assertIsNone(actor.action_limit)
        self.assertEqual(cfg.moe_load_balance_coef, 0.)
        self.assertEqual(cfg.moe_router_z_coef, 0.)
        self.assertEqual(sum(isinstance(layer, nn.Linear) for layer in actor.router), 2)
        self.assertEqual(sum(isinstance(layer, nn.ELU) for layer in actor.router), 1)
        self.assertEqual(actor.router[0].in_features, 384)
        self.assertEqual(actor.router[0].out_features, 256)
        self.assertEqual(actor.router[-1].out_features, 4)
        self.assertEqual(sum(isinstance(layer, nn.ELU) for layer in actor.experts[0]), 3)
        self.assertEqual(actor_architecture(cfg)["moe_variant"], "hiking")

    def test_both_gate_initializations_forward_and_all_derivatives_match_independent_reference(self):
        for gate_hidden in ((256,), ()):
            with self.subTest(gate_hidden=gate_hidden):
                with torch.random.fork_rng(devices=[]):
                    torch.manual_seed(2718)
                    actual = create_actor(hiking_config(moe_gate_hidden=gate_hidden),
                                          obs_dim=4, action_dim=2).double()
                    torch.manual_seed(2718)
                    reference = HikingReference(gate_hidden=gate_hidden).double()
                actual_parameters = dict(actual.named_parameters())
                expected_parameters = dict(reference.named_parameters())
                self.assertEqual(set(actual_parameters), set(expected_parameters))
                for name, expected in expected_parameters.items():
                    # This detects any extra head/router scaling or zero bias.
                    torch.testing.assert_close(actual_parameters[name], expected, rtol=0, atol=0)
                data = torch.tensor([[.2, -.4, .1, 1.], [-.7, .5, .4, -.2], [.3, .8, -.6, .2]],
                                    dtype=torch.float64)
                left, right = data.clone().requires_grad_(True), data.clone().requires_grad_(True)
                output, expected = actual(left), reference(right)
                torch.testing.assert_close(output, expected, rtol=1e-12, atol=1e-12)
                objective_weights = data.new_tensor([[.3, -.5], [-.7, .2], [.4, .6]])
                (output * objective_weights).sum().backward()
                (expected * objective_weights).sum().backward()
                torch.testing.assert_close(left.grad, right.grad, rtol=1e-11, atol=1e-12)
                for name, expected_parameter in expected_parameters.items():
                    gradient = actual_parameters[name].grad
                    self.assertIsNotNone(gradient)
                    torch.testing.assert_close(gradient, expected_parameter.grad, rtol=1e-10, atol=1e-12)
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in actual.router.parameters()), 0.)
                for expert in actual.experts:
                    self.assertGreater(sum(float(p.grad.abs().sum()) for p in expert.parameters()), 0.)

    def test_raw_observations_and_unbounded_mean_have_no_hidden_transform(self):
        actor = create_actor(hiking_config(), obs_dim=4, action_dim=2).double()
        observations = torch.randn(3, 4, dtype=torch.float64)
        before = actor(observations).detach()
        with torch.no_grad():
            actor.norm.mean.fill_(120.)
            actor.norm.var.fill_(1e5)
            actor.norm.count.fill_(1234.)
        torch.testing.assert_close(actor(observations), before, rtol=0, atol=0)
        with torch.no_grad():
            actor.router[-1].weight.zero_()
            actor.router[-1].bias.zero_()
            for expert in actor.experts:
                expert[-1].weight.zero_()
                expert[-1].bias.copy_(torch.tensor([12., -9.], dtype=torch.float64))
        torch.testing.assert_close(actor(observations), observations.new_tensor([[12., -9.]]).expand(3, -1),
                                   rtol=0, atol=0)

    def test_all_experts_process_every_sample_and_routing_preserves_full_softmax(self):
        actor = create_actor(hiking_config(), obs_dim=4, action_dim=2).double()
        observation = torch.randn(2, 3, 4, dtype=torch.float64)
        counts = [0, 0, 0, 0]
        handles = []
        for index, expert in enumerate(actor.experts):
            def record(module, inputs, output, expert_index=index):
                counts[expert_index] += inputs[0].numel() // 4
            handles.append(expert.register_forward_hook(record))
        try:
            actor(observation)
        finally:
            for handle in handles:
                handle.remove()
        self.assertEqual(counts, [6, 6, 6, 6])
        for prefix in ((), (7,), (2, 3), (0,)):
            with self.subTest(prefix=prefix):
                data = torch.randn(*prefix, 4, dtype=torch.float64)
                route = actor.route(data)
                self.assertEqual(actor(data).shape, (*prefix, 2))
                expected = actor.router(data).softmax(-1)
                torch.testing.assert_close(route.probabilities, expected, rtol=1e-12, atol=1e-12)
                torch.testing.assert_close(route.weights, expected, rtol=0, atol=0)
                indices = torch.arange(4).expand(*prefix, 4)
                torch.testing.assert_close(route.indices.cpu(), indices, rtol=0, atol=0)
                torch.testing.assert_close(actor.forward_dense(data), actor(data), rtol=0, atol=0)

    def test_hiking_dense_actor_compiles_as_one_graph_with_input_and_gate_gradients(self):
        actor = create_actor(hiking_config(), obs_dim=4, action_dim=2).double()
        compiled = torch.compile(actor, backend="eager", fullgraph=True)
        inputs = torch.randn(3, 4, dtype=torch.float64, requires_grad=True)
        expected = actor(inputs).detach()
        actual = compiled(inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().sum().backward()
        self.assertTrue(bool(torch.isfinite(inputs.grad).all()))
        self.assertGreater(sum(float(p.grad.abs().sum()) for p in actor.router.parameters()), 0.)
        hiking_config(compile=True, compile_fullgraph=True).validate()

    def test_non_hiking_settings_cannot_be_silently_accepted_as_hiking(self):
        for changes in ({"moe_top_k": 2}, {"moe_num_experts": 5}, {"moe_gate_hidden": (128,)},
                        {"moe_gate_hidden": (256, 256)},
                        {"moe_temperature": .8}, {"moe_load_balance_coef": .01},
                        {"moe_router_z_coef": .001}, {"moe_expert_hidden": (256, 128)},
                        {"moe_init_noise_std": .01}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                hiking_config(**changes).validate()

    def test_historical_unlabelled_impl5_remains_sparse_with_identical_state_and_actions(self):
        cfg = legacy_config()
        cfg.validate()
        original = create_actor(cfg)
        self.assertIs(type(original), SparseMoEActor)
        with torch.no_grad():
            original.norm.mean.fill_(.3)
            original.norm.var.fill_(1.8)
            original.norm.count.fill_(140.)
        saved_config = asdict(cfg)
        for name in ("moe_variant", "moe_expert_hidden"):
            saved_config.pop(name)
        old_architecture = actor_architecture(cfg)
        # The historical architecture schema had exactly these fields.
        self.assertEqual(set(old_architecture), {"actor_type", "policy_hidden", "action_limit",
            "moe_num_experts", "moe_top_k", "moe_gate_hidden", "moe_temperature"})
        payload = {"format": "amp-wmac-v1", "implementation_version": 5,
                   "config": saved_config, "actor_type": "moe", "actor_architecture": old_architecture,
                   "actor": original.state_dict()}
        restored_config = resolve_checkpoint_config(payload)
        self.assertEqual(restored_config.moe_variant, "legacy")
        self.assertEqual(checkpoint_actor_architecture(payload), old_architecture)
        restored = create_actor(restored_config)
        self.assertIs(type(restored), SparseMoEActor)
        restored.load_state_dict(payload["actor"])
        data = torch.randn(9, 384)
        torch.testing.assert_close(restored(data), original(data), rtol=0, atol=0)

    def test_hiking_impl6_requires_explicit_variant_and_reconstructs_dense_policy(self):
        cfg = hiking_config()
        actor = create_actor(cfg)
        payload = {"format": "amp-wmac-v1", "implementation_version": 6,
                   "config": asdict(cfg), "actor_type": "moe", "actor_architecture": actor_architecture(cfg),
                   "actor": actor.state_dict()}
        restored_config = resolve_checkpoint_config(payload)
        self.assertEqual(restored_config.moe_variant, "hiking")
        restored = create_actor(restored_config)
        restored.load_state_dict(payload["actor"])
        data = torch.randn(3, 384)
        torch.testing.assert_close(restored(data), actor(data), rtol=0, atol=0)
        payload["config"].pop("moe_variant")
        with self.assertRaises(ValueError):
            resolve_checkpoint_config(payload)

    def test_historical_linear_hiking_impl6_restores_empty_gate_instead_of_new_default(self):
        cfg = hiking_config(moe_gate_hidden=())
        cfg.validate()
        actor = create_actor(cfg)
        saved_config = asdict(cfg)
        saved_config["moe_gate_hidden"] = []  # Historical JSON representation.
        payload = {"format": "amp-wmac-v1", "implementation_version": 6,
                   "config": saved_config, "actor_type": "moe", "actor_architecture": actor_architecture(cfg),
                   "actor": actor.state_dict()}
        self.assertEqual(payload["actor_architecture"]["moe_gate_hidden"], [])
        restored_config = resolve_checkpoint_config(payload)
        self.assertEqual(restored_config.moe_gate_hidden, ())
        restored_config.validate()
        restored = create_actor(restored_config)
        self.assertEqual(restored.gate_hidden, ())
        self.assertEqual(sum(isinstance(layer, nn.Linear) for layer in restored.router), 1)
        restored.load_state_dict(payload["actor"])
        for name, parameter in actor.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[name], parameter, rtol=0, atol=0)
        data = torch.randn(7, 384)
        torch.testing.assert_close(restored(data), actor(data), rtol=0, atol=0)

    def test_onnx_dense_hiking_matches_batches_without_a_topk_operator(self):
        if any(importlib.util.find_spec(name) is None for name in ("onnx", "onnxruntime")):
            self.skipTest("ONNX and ONNX Runtime are needed for deployment parity")
        import onnx
        import onnxruntime as ort
        actor = create_actor(hiking_config(), obs_dim=4, action_dim=2).eval()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "hiking.onnx"
            torch.onnx.export(actor, torch.zeros(2, 4), str(path), input_names=["obs"],
                              output_names=["actions"], dynamic_axes={"obs": {0: "batch"}, "actions": {0: "batch"}},
                              opset_version=18, dynamo=False)
            graph = onnx.load(str(path)).graph
            self.assertNotIn("TopK", [node.op_type for node in graph.node])
            session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
            for batch in (1, 7):
                with self.subTest(batch=batch):
                    inputs = torch.randn(batch, 4)
                    actual = session.run(None, {"obs": inputs.numpy()})[0]
                    np.testing.assert_allclose(actual, actor(inputs).detach().numpy(), rtol=2e-5, atol=3e-7)


if __name__ == "__main__":
    unittest.main()
