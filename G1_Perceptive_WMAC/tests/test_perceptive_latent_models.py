"""InstinctMJ stack latents, encoder training and visual gradient boundaries."""

from __future__ import annotations

import ast
import copy
import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch
from torch.nn import functional as F

from src.wmac.networks import frozen_parameters
from src.wmac_perceptive.instinct_latent import InstinctLatentDecoder
from src.wmac_perceptive.latent_models import LatentPerceptiveActor, LatentPerceptiveWorldModel
from src.wmac_perceptive.latent_perception import (
    pack_latent_actor_input, pack_processed_actor_input, processed_depth_stack, processed_stack_shape,
    unpack_latent_actor_input, unpack_processed_actor_input,
)
from src.wmac_perceptive.models import PerceptiveActor
from src.wmac_perceptive.perception import PerceptionSpec, pack_actor_input, select_depth_frames
from test_perceptive_models import PlaneOracle, SMALL, state_batch


class StackLatentActorTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(310)

    def test_processed_stack_is_existing_actor_view_with_chronological_frames(self):
        actor = LatentPerceptiveActor().double()
        spec = actor.spec
        ring = torch.rand(2, *spec.ring_shape, dtype=torch.float64)
        expected = actor.vision.preprocess(select_depth_frames(ring, spec).flatten(-4, -3))
        actual = processed_depth_stack(ring, spec)
        self.assertEqual(actual.shape, (2, 16, 18, 32))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(actor.encode_history(ring), actor.vision(select_depth_frames(ring, spec).flatten(-4, -3)), rtol=0, atol=0)

    def test_processed_and_latent_packing_preserve_arbitrary_batch_prefixes(self):
        proprio = torch.randn(2, 3, 384, dtype=torch.float64)
        stack = torch.rand(2, 3, *processed_stack_shape(SMALL), dtype=torch.float16)
        latent = torch.randn(2, 3, 128, dtype=torch.float16)
        for packed, unpack, expected in (
            (pack_processed_actor_input(proprio, stack, SMALL), unpack_processed_actor_input, stack.double()),
            (pack_latent_actor_input(proprio, latent, SMALL), unpack_latent_actor_input, latent.double()),
        ):
            actual_proprio, actual_visual = unpack(packed, SMALL)
            torch.testing.assert_close(actual_proprio, proprio, rtol=0, atol=0)
            torch.testing.assert_close(actual_visual, expected, rtol=0, atol=0)
        self.assertEqual(pack_latent_actor_input(proprio, latent, SMALL).shape[-1], 512)

    def test_actor_preserves_existing_vision_policy_state_and_initial_function(self):
        original = PerceptiveActor(SMALL).double()
        actor = LatentPerceptiveActor(SMALL).double()
        self.assertEqual(set(original.state_dict()), set(actor.state_dict()))
        actor.load_state_dict(original.state_dict(), strict=True)
        proprio = torch.randn(2, 384, dtype=torch.float64)
        ring = torch.rand(2, *SMALL.ring_shape, dtype=torch.float64)
        packed = pack_actor_input(proprio, ring, SMALL)
        torch.testing.assert_close(actor(packed), original(packed), rtol=0, atol=0)

    def test_raw_processed_and_compact_actor_inputs_have_equal_actions_and_routes(self):
        actor = LatentPerceptiveActor(SMALL).double()
        proprio = torch.randn(2, 3, 384, dtype=torch.float64)
        ring = torch.rand(2, 3, *SMALL.ring_shape, dtype=torch.float64)
        stack = processed_depth_stack(ring, SMALL)
        latent = actor.encode_processed(stack)
        inputs = (pack_actor_input(proprio, ring, SMALL), pack_processed_actor_input(proprio, stack, SMALL),
                  pack_latent_actor_input(proprio, latent, SMALL))
        expected = actor.forward_latent(proprio, latent)
        for packed in inputs:
            torch.testing.assert_close(actor(packed), expected, rtol=0, atol=0)
            torch.testing.assert_close(actor.route(packed).weights, actor.route_latent(proprio, latent).weights, rtol=0, atol=0)
        self.assertEqual(expected.shape, (2, 3, 29))
        self.assertEqual(actor.num_experts, 4)
        self.assertEqual(actor.top_k, 4)
        actor.precision = "bf16"
        torch.testing.assert_close(actor(inputs[1]), expected, rtol=0, atol=0)

    def test_policy_gradient_isolated_from_visual_encoder_and_latent_inputs(self):
        actor = LatentPerceptiveActor(SMALL).double()
        proprio = torch.randn(2, 384, dtype=torch.float64, requires_grad=True)
        ring = torch.rand(2, *SMALL.ring_shape, dtype=torch.float64, requires_grad=True)
        actor(pack_actor_input(proprio, ring, SMALL)).square().sum().backward()
        # Packing shares one concatenation with proprioception, so its backward
        # can materialize explicit zeros for the detached image slice.
        self.assertTrue(ring.grad is None or bool((ring.grad == 0).all()))
        self.assertGreater(float(proprio.grad.abs().sum()), 0.)
        self.assertTrue(all(parameter.grad is None for parameter in actor.vision.parameters()))
        self.assertTrue(all(parameter.grad is not None for parameter in actor.policy.parameters()))
        latent = torch.randn(2, 128, dtype=torch.float64, requires_grad=True)
        actor.forward_latent(proprio.detach(), latent).sum().backward()
        self.assertIsNone(latent.grad)

    def test_encoder_is_online_trainable_through_reconstruction_only(self):
        actor = LatentPerceptiveActor(SMALL).double()
        shape = processed_stack_shape(SMALL)
        decoder = InstinctLatentDecoder(128, shape[0], shape[1:]).double()
        stack = torch.rand(2, *shape, dtype=torch.float64)
        encoder_before = copy.deepcopy(actor.vision.state_dict())
        policy_before = copy.deepcopy(actor.policy.state_dict())
        optimizer = torch.optim.Adam([*actor.vision.parameters(), *decoder.parameters()], lr=.001)
        optimizer.zero_grad()
        reconstruction = decoder(actor.encode_processed(stack))
        (reconstruction - stack).abs().mean().backward()
        optimizer.step()
        self.assertTrue(any(not torch.equal(value, encoder_before[name]) for name, value in actor.vision.state_dict().items()))
        for name, value in actor.policy.state_dict().items():
            torch.testing.assert_close(value, policy_before[name], rtol=0, atol=0)
        self.assertTrue(all(parameter.requires_grad for parameter in actor.vision.parameters()))

    def test_explicit_encoder_retains_selected_image_input_gradients(self):
        actor = LatentPerceptiveActor(SMALL).double()
        ring = torch.rand(2, *SMALL.ring_shape, dtype=torch.float64, requires_grad=True)
        actor.encode_history(ring).square().sum().backward()
        self.assertGreater(float(ring.grad[:, (0, 2)].abs().sum()), 0.)
        self.assertEqual(float(ring.grad[:, (1, 3)].abs().sum()), 0.)

    def test_invalid_shapes_and_nonstandard_encoder_width_are_rejected(self):
        actor = LatentPerceptiveActor(SMALL)
        with self.assertRaises(ValueError):
            LatentPerceptiveActor(SMALL, latent_dim=64)
        with self.assertRaises(ValueError):
            actor.encode_processed(torch.zeros(2, 1, 4, 6))
        with self.assertRaises(ValueError):
            actor.forward_latent(torch.zeros(2, 384), torch.zeros(2, 4, 128))
        with self.assertRaises(ValueError):
            pack_processed_actor_input(torch.zeros(1, 384), torch.zeros(2, *processed_stack_shape(SMALL)), SMALL)
        with self.assertRaises(ValueError):
            unpack_latent_actor_input(torch.zeros(2, 8), SMALL)


class StackLatentWorldModelTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(311)
        self.model = LatentPerceptiveWorldModel(PlaneOracle(), SMALL, hidden=(32,)).double()
        self.state = state_batch()
        self.action, self.context = self.state.new_zeros(2, 29), self.state.new_zeros(2, 35)
        self.latent = torch.randn(2, 128, dtype=self.state.dtype)

    def test_reference_inputs_use_joint_targets_world_height_and_material_ranges(self):
        from src.wmac.gencoord import gencoord_features
        scale = torch.linspace(.1, .4, 29)
        offset = torch.linspace(-.3, .3, 29)
        model = LatentPerceptiveWorldModel(PlaneOracle(offset=.6), SMALL, hidden=(32,), context_dim=129,
                                          action_transform=(scale, offset), continuous_normalization=True).double()
        action = self.action + torch.arange(2, dtype=self.state.dtype)[:, None]
        context = self.state.new_zeros(2, 129)
        context[:, 0] = .95
        context[:, 100:] = self.state.new_tensor([.3, 1.6])[:, None]
        context[:, 38:67] = action
        context[:, 67:96] = self.state.new_tensor([0., 2.])[:, None]
        expected = torch.cat((gencoord_features(self.state), action * scale + offset), -1)
        model.update_normalizers_latent(self.state, action, context, self.state, self.latent)
        self.assertEqual(model.in_norm.mean.numel(), 97)
        torch.testing.assert_close(model.in_norm.mean, expected.mean(0))
        observed = model.inputs_latent(self.state, action, context, self.latent)
        torch.testing.assert_close(observed[:, :97], model.in_norm(expected))
        torch.testing.assert_close(observed[:, 97], torch.zeros_like(action[:, 0]))
        torch.testing.assert_close(observed[:, 197:226], self.state.new_tensor([-1., 1.])[:, None].expand(-1, 29))
        torch.testing.assert_close(observed[:, 97+38:97+67], action)
        torch.testing.assert_close(observed[:, 97+67:97+96], context[:, 67:96] - 1.)
        self.assertGreater(model.net[-1].weight[:35].abs().sum().item(), 0.)

    def test_single_head_excludes_encoder_decoder_and_separates_z_statistics(self):
        model = self.model
        self.assertEqual(model.in_dim, 68 + 29 + 35 + 3)
        self.assertEqual(model.in_norm.mean.numel(), model.in_dim)
        self.assertEqual(model.net[0].in_features, model.in_dim + 128)
        self.assertEqual(model.net[-1].out_features, 35 + 2 + 128)
        for name in ("depth_encoder", "depth_net", "contact_net", "latent_net", "codec", "out_norm"):
            self.assertFalse(hasattr(model, name))
        model.update_normalizers_latent(self.state, self.action, self.context, self.state, self.latent)
        self.assertEqual(float(model.in_norm.count), 2.)
        model.z_mean.fill_(.3)
        model.z_std.fill_(2.)
        values = model.inputs_latent(self.state, self.action, self.context, self.latent)
        torch.testing.assert_close(values[..., :model.in_dim], model.in_norm(model.inputs(self.state, self.action, self.context)))
        torch.testing.assert_close(values[..., model.in_dim:], (self.latent - .3) / 2)

    def test_zero_initialized_head_preserves_z_and_integrates_gen_coord_velocities(self):
        state = self.state.clone()
        state[:, 7] = 1
        state[:, 42] = .5
        predicted, contact, future = self.model.step_with_latent(state, self.action, self.context, self.latent)
        torch.testing.assert_close(future, self.latent, rtol=0, atol=0)
        torch.testing.assert_close(predicted[:, 0], state[:, 0] + .02)
        torch.testing.assert_close(predicted[:, 13], state[:, 13] + .01)
        torch.testing.assert_close(contact, self.state.new_full((2, 2), float(F.softplus(torch.tensor(-4.)))))
        with self.assertRaisesRegex(ValueError, "no image encoder"):
            self.model.step_with_perception(state, self.action, self.context, torch.empty(0))

    def test_latent_head_predicts_scaled_residual_and_contact_softplus_proxy(self):
        bias = self.model.net[-1].bias
        with torch.no_grad():
            bias[35:37] = self.state.new_tensor([-.5, .7])
            bias[37:] = torch.linspace(-1, 1, 128, dtype=bias.dtype)
            self.model.dz_std.copy_(torch.linspace(.1, 2, 128, dtype=bias.dtype))
        _, contact, future = self.model.step_with_latent(self.state, self.action, self.context, self.latent)
        torch.testing.assert_close(contact, F.softplus(bias[35:37]).expand(2, -1))
        torch.testing.assert_close(future, self.latent + bias[37:] * self.model.dz_std)

    def test_ema_statistics_use_real_latents_and_masked_increment_rms(self):
        z = torch.randn(2, 3, 128, dtype=self.state.dtype)
        dz = torch.randn_like(z)
        mask = self.state.new_tensor([[1, 1, 0], [1, 0, 0]])
        self.model.update_z_stats(z, dz, mask, rate=1.)
        torch.testing.assert_close(self.model.z_mean, z.reshape(-1, 128).mean(0))
        torch.testing.assert_close(self.model.z_std, z.reshape(-1, 128).std(0).clamp_min(.001))
        expected_rms = ((dz.square() * mask[..., None]).sum((0, 1)) / mask.sum()).sqrt().clamp_min(.001)
        torch.testing.assert_close(self.model.dz_std, expected_rms)
        self.assertFalse(self.model.z_mean.requires_grad)

    def test_world_model_latent_recurrence_keeps_action_gradients_while_actor_detaches_them(self):
        with torch.no_grad():
            self.model.net[-1].weight.normal_(0, .01)
        action = torch.randn_like(self.action, requires_grad=True)
        latent = self.latent.clone().requires_grad_(True)
        actor = LatentPerceptiveActor(SMALL).double()
        proprio = torch.randn(2, 384, dtype=self.state.dtype)
        with frozen_parameters(self.model):
            state, _, first_z = self.model.step_with_latent(self.state, action, self.context, latent)
            _, _, second_z = self.model.step_with_latent(state, self.action, self.context, first_z)
            action_grad, latent_grad = torch.autograd.grad(second_z.sum(), (action, latent), retain_graph=True)
            policy_action_grad, = torch.autograd.grad(actor.forward_latent(proprio, second_z).sum(), action, allow_unused=True)
        self.assertGreater(float(action_grad.abs().sum()), 0.)
        self.assertGreater(float(latent_grad.abs().sum()), 0.)
        self.assertIsNone(policy_action_grad)
        self.assertTrue(all(parameter.grad is None for parameter in self.model.parameters()))

    def test_world_model_loss_keeps_official_integrated_pose_supervision(self):
        target = self.state.clone()
        target[:, 0] += .01
        torch.testing.assert_close(self.model.loss(self.state, target), self.state.new_full((2,), 1 / 70))


def _official_latent_methods():
    path = Path(os.environ.get("INSTINCTMJ_REFERENCE_CHECKOUT", "/tmp/instinctmj-supermimic-20261111")) / "src/instinct_mj/rl/supermimic/networks.py"
    if not path.exists():
        raise unittest.SkipTest("pinned InstinctMJ checkout unavailable; set INSTINCTMJ_REFERENCE_CHECKOUT")
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != "b6356b59cef81a01b1c86ced8247f83a7060fab9407a7ad943f8425e092e38f5":
        raise AssertionError("Pinned InstinctMJ networks.py source hash mismatch")
    tree = ast.parse(source, filename=str(path))
    world_model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "WorldModel")
    stats = copy.deepcopy(next(node for node in world_model.body if isinstance(node, ast.FunctionDef) and node.name == "update_z_stats"))
    stats.decorator_list = []
    step = next(node for node in world_model.body if isinstance(node, ast.FunctionDef) and node.name == "step")
    z_assignment = copy.deepcopy(next(node for node in step.body if isinstance(node, ast.Assign)
                                    and isinstance(node.targets[0], ast.Name) and node.targets[0].id == "z1"))
    namespace = {"torch": torch}
    exec(compile(ast.Module(body=[stats], type_ignores=[]), str(path), "exec"), namespace)
    return namespace["update_z_stats"], compile(ast.Module(body=[z_assignment], type_ignores=[]), str(path), "exec")


class InstinctSourceVisualParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.methods = _official_latent_methods()

    def test_stats_match_hash_verified_upstream_function(self):
        torch.manual_seed(312)
        model = LatentPerceptiveWorldModel(PlaneOracle(), SMALL, hidden=(8,)).double()
        reference = SimpleNamespace(z_dim=128, z_mean=model.z_mean.clone(), z_std=model.z_std.clone(), dz_std=model.dz_std.clone())
        z, dz, mask = torch.randn(2, 3, 128, dtype=torch.float64), torch.randn(2, 3, 128, dtype=torch.float64), torch.tensor([[1., 1., 0.], [1., 0., 0.]], dtype=torch.float64)
        model.update_z_stats(z, dz, mask)
        self.methods[0](reference, z, dz, mask)
        for name in ("z_mean", "z_std", "dz_std"):
            torch.testing.assert_close(getattr(model, name), getattr(reference, name), rtol=0, atol=0)

    def test_scaled_residual_and_its_gradients_match_upstream_z_assignment(self):
        torch.manual_seed(313)
        model = LatentPerceptiveWorldModel(PlaneOracle(), SMALL, hidden=(8,)).double()
        model.dz_std.copy_(torch.rand_like(model.dz_std) + .01)
        latent = torch.randn(2, 128, dtype=torch.float64, requires_grad=True)
        y = torch.randn(2, 35 + 2 + 128, dtype=torch.float64, requires_grad=True)
        # Only the physical output offset changes from upstream's 70 to 35;
        # execute the source z1 assignment itself with that local offset.
        namespace = {"self": model, "z": latent, "y": y, "WM_OUT": 35}
        exec(self.methods[1], namespace)
        expected = namespace["z1"]
        actual = latent + y[..., 37:] * model.dz_std
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual_grads = torch.autograd.grad(actual.square().sum(), (latent, y), retain_graph=True)
        expected_grads = torch.autograd.grad(expected.square().sum(), (latent, y))
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
