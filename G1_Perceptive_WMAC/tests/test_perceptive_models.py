"""Depth chronology, non-blind learned dynamics and BPTT boundary checks."""
from __future__ import annotations

import copy
import unittest

import torch
from torch import nn

from src.wmac.hiking import HikingMoEActor
from src.wmac.networks import frozen_parameters
from src.wmac_perceptive.models import DepthEncoder, PerceptiveActor, TerrainPerceptiveWorldModel
from src.wmac_perceptive.perception import (
    DepthHistory, PerceptionSpec, advance_depth_ring, pack_actor_input,
    select_depth_frames, unpack_actor_input,
)


SMALL = PerceptionSpec(height=4, width=6, ring_length=4, frames=2, stride=2, delay=1,
                       crop_top=0, crop_left=0, crop_right=0)


class PlaneOracle(nn.Module):
    patch_dim = 3

    def __init__(self, offset=0., slope=.15):
        super().__init__()
        self.offset, self.slope = offset, slope
        self.register_buffer("map_asset", torch.ones(4))

    def height_at(self, xy):
        return self.offset + self.slope * xy[..., 0]

    def local_patch(self, state):
        # Relative samples from the same fixed plane, not policy information.
        return torch.stack([state[..., 0] * 0 + self.slope * x for x in (-.2, 0., .2)], -1)


def state_batch(count=2):
    state = torch.zeros(count, 71, dtype=torch.float64)
    state[:, 0] = torch.linspace(.1, .3, count, dtype=torch.float64)
    state[:, 2], state[:, 3] = .8, 1.
    return state


class DepthHistoryTest(unittest.TestCase):
    def test_default_chronology_delay_and_arbitrary_batch_pack(self):
        spec = PerceptionSpec()
        self.assertEqual(spec.selected_indices, (0, 5, 10, 15, 20, 25, 30, 35))
        self.assertEqual(spec.packed_dim, 170880)
        ring = torch.arange(37).reshape(1, 37, 1, 1, 1).expand(2, *spec.ring_shape).float()
        chosen = select_depth_frames(ring, spec)
        torch.testing.assert_close(chosen[:, :, 0, 0, 0], torch.tensor(spec.selected_indices).expand(2, 8).float())
        proprio = torch.randn(2, 3, 384, dtype=torch.float64)
        visual = torch.randn(2, 3, *SMALL.ring_shape, dtype=torch.float16)
        packed = pack_actor_input(proprio, visual, SMALL)
        restored, frames = unpack_actor_input(packed, SMALL)
        torch.testing.assert_close(restored, proprio, rtol=0, atol=0)
        torch.testing.assert_close(frames, visual.double(), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            PerceptionSpec(ring_length=36)

    def test_reset_clears_every_old_frame_only_for_reset_environments(self):
        history = DepthHistory(SMALL, 2, dtype=torch.float64)
        original = torch.ones(2, *SMALL.frame_shape, dtype=torch.float64)
        history.reset(original)
        snapshot = history.snapshot()
        next_frame = original.clone()
        next_frame[0].fill_(.2); next_frame[1].fill_(.8)
        history.append(next_frame, reset_mask=torch.tensor([True, False]))
        torch.testing.assert_close(history.buffer[0], next_frame[0].expand(*SMALL.ring_shape), rtol=0, atol=0)
        torch.testing.assert_close(history.buffer[1, :-1], original[1].expand(3, *SMALL.frame_shape), rtol=0, atol=0)
        torch.testing.assert_close(history.buffer[1, -1], next_frame[1], rtol=0, atol=0)
        torch.testing.assert_close(snapshot, original[:, None].expand_as(snapshot), rtol=0, atol=0)

    def test_prediction_advances_the_delayed_policy_view_with_gradients(self):
        ring = torch.zeros(1, *SMALL.ring_shape, dtype=torch.float64)
        future = torch.full((1, *SMALL.frame_shape), .4, dtype=torch.float64, requires_grad=True)
        ring = advance_depth_ring(ring, future, SMALL)
        self.assertEqual(float(select_depth_frames(ring, SMALL).sum()), 0.)  # One-frame sensor delay.
        ring = advance_depth_ring(ring, torch.full_like(future, .7), SMALL)
        chosen = select_depth_frames(ring, SMALL)
        torch.testing.assert_close(chosen[:, -1], future)
        chosen.sum().backward()
        torch.testing.assert_close(future.grad, torch.ones_like(future))


class PerceptiveNetworksTest(unittest.TestCase):
    def test_author_crop_and_blur_keep_hit_masks_and_exclude_outside_field_of_view(self):
        spec = PerceptionSpec()
        encoder = DepthEncoder(16, 36, 64, 128, spec)
        image = torch.zeros(1, 16, 36, 64)
        image[:, 1::2] = torch.randint(0, 2, image[:, 1::2].shape).float()
        image[:, 0::2, 24, 30] = 1.
        processed = encoder.preprocess(image)
        self.assertEqual(processed.shape, (1, 16, 18, 32))
        torch.testing.assert_close(processed[:, 1::2], image[:, 1::2, 18:, 16:-16], rtol=0, atol=0)
        self.assertGreater(float(processed[:, 0, 6, 14]), 0.)
        self.assertLess(float(processed[:, 0, 6, 14]), 1.)
        changed = image.clone(); changed[:, 0::2, :18] = 8.
        changed[:, 0::2, :, :16] = 8.; changed[:, 0::2, :, -16:] = 8.
        torch.testing.assert_close(encoder.preprocess(changed), processed, rtol=0, atol=0)

    def test_actor_uses_depth_and_learns_its_own_encoder_and_all_experts(self):
        torch.manual_seed(34)
        actor = PerceptiveActor(SMALL).double()
        proprio = torch.randn(2, 384, dtype=torch.float64)
        ring = torch.rand(2, *SMALL.ring_shape, dtype=torch.float64)
        packed = pack_actor_input(proprio, ring, SMALL).requires_grad_(True)
        actual = actor(packed)
        changed = actor(pack_actor_input(proprio, torch.zeros_like(ring), SMALL))
        self.assertEqual(actual.shape, (2, 29))
        self.assertGreater(float((actual-changed).abs().max()), 1e-8)
        actual.square().sum().backward()
        for module in (actor.vision, actor.router, *actor.experts):
            gradients = [p.grad for p in module.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0.)
        _, visual_grad = unpack_actor_input(packed.grad, SMALL)
        self.assertGreater(float(visual_grad[:, (0, 2)].abs().sum()), 0.)
        self.assertEqual(float(visual_grad[:, (1, 3)].abs().sum()), 0.)
        route = actor.route(packed.detach())
        torch.testing.assert_close(route.weights.sum(-1), torch.ones(2, dtype=torch.float64))
        self.assertEqual(route.weights.shape, (2, 4))

    def test_model_requires_depth_and_conditions_on_depth_and_terrain(self):
        torch.manual_seed(52)
        model = TerrainPerceptiveWorldModel(PlaneOracle(), SMALL, hidden=(32,)).double()
        state = state_batch(); action = torch.zeros(2, 29, dtype=torch.float64)
        context = torch.zeros(2, 35, dtype=torch.float64)
        frame = torch.rand(2, *SMALL.frame_shape, dtype=torch.float64)
        inputs = model.inputs(state, action, context, frame)
        self.assertEqual(inputs.shape[-1], 135+64+3)
        self.assertGreater(float((inputs-model.inputs(state, action, context, torch.zeros_like(frame))).abs().sum()), 0.)
        shifted_model = TerrainPerceptiveWorldModel(PlaneOracle(offset=10.), SMALL, hidden=(32,)).double()
        shifted_model.load_state_dict(model.state_dict())
        shifted_state = state.clone(); shifted_state[:, 2] += 10.
        torch.testing.assert_close(shifted_model.inputs(shifted_state, action, context, frame), inputs, rtol=1e-13, atol=1e-13)
        model.terrain_oracle.slope = .4
        self.assertGreater(float((inputs-model.inputs(state, action, context, frame)).abs().sum()), .01)
        with self.assertRaisesRegex(ValueError, "without depth"):
            model.step(state, action, context)
        self.assertFalse(any("map_asset" in name or "terrain_oracle" in name for name in model.state_dict()))
        predicted, contact, next_frame = model.step_with_perception(state, action, context, frame)
        self.assertEqual(predicted.shape, state.shape)
        self.assertEqual(contact.shape, (2, 2))
        self.assertEqual(next_frame.shape, frame.shape)
        self.assertTrue(bool(torch.isfinite(next_frame).all()))
        self.assertTrue(bool(((next_frame >= 0) & (next_frame <= 1)).all()))

    def test_pixel_loss_cannot_hide_visible_errors_by_predicting_no_hits(self):
        target = torch.ones(2, 2, 4, 6, dtype=torch.float64)
        target[:, 0].fill_(.2)
        bad_mask = target.clone(); bad_mask[:, 0].fill_(.8); bad_mask[:, 1].fill_(.01)
        good_mask = bad_mask.clone(); good_mask[:, 1].fill_(.99)
        first = TerrainPerceptiveWorldModel.pixel_loss_terms(bad_mask, target)
        second = TerrainPerceptiveWorldModel.pixel_loss_terms(good_mask, target)
        torch.testing.assert_close(first["depth"], second["depth"], rtol=0, atol=0)
        self.assertTrue(bool((first["valid"] > second["valid"]).all()))
        target[:, 1].zero_()
        self.assertEqual(float(TerrainPerceptiveWorldModel.pixel_loss_terms(good_mask, target)["depth"].sum()), 0.)
        self.assertTrue(bool(torch.isfinite(TerrainPerceptiveWorldModel.pixel_loss(good_mask, target)).all()))

    def test_frozen_model_keeps_future_image_input_gradient_to_actor_without_parameter_leaks(self):
        torch.manual_seed(63)
        actor = PerceptiveActor(SMALL).double()
        model = TerrainPerceptiveWorldModel(PlaneOracle(), SMALL, hidden=(32,)).double()
        state = state_batch(); context = torch.zeros(2, 35, dtype=torch.float64)
        proprio = torch.randn(2, 384, dtype=torch.float64)
        ring = torch.rand(2, *SMALL.ring_shape, dtype=torch.float64)
        before = {name: value.detach().clone() for name, value in model.state_dict().items()}
        with frozen_parameters(model):
            action = actor(pack_actor_input(proprio, ring, SMALL))
            action.retain_grad()
            state, _, future = model.step_with_perception(state, action, context, ring[:, -1])
            future.retain_grad()
            next_ring = advance_depth_ring(ring, future, SMALL)
            next_ring = advance_depth_ring(next_ring, torch.zeros_like(future), SMALL)
            later = actor(pack_actor_input(proprio, next_ring, SMALL))
            later.square().sum().backward()
        self.assertTrue(bool(torch.isfinite(action.grad).all()))
        self.assertGreater(float(action.grad.abs().sum()), 0.)
        self.assertGreater(float(future.grad.abs().sum()), 0.)
        self.assertTrue(all(p.grad is None for p in model.parameters()))
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)

    def test_flat_hiking_initialization_preserves_actions_and_refuses_other_shapes_before_mutation(self):
        torch.manual_seed(71)
        flat = HikingMoEActor().double()
        actor = PerceptiveActor(SMALL).double()
        vision_before = copy.deepcopy(actor.vision.state_dict())
        actor.copy_from_flat_hiking(flat.state_dict())
        proprio = torch.randn(3, 384, dtype=torch.float64)
        ring = torch.rand(3, *SMALL.ring_shape, dtype=torch.float64)
        torch.testing.assert_close(actor(pack_actor_input(proprio, ring, SMALL)), flat(proprio), rtol=1e-12, atol=1e-12)
        for name, value in actor.vision.state_dict().items():
            torch.testing.assert_close(value, vision_before[name], rtol=0, atol=0)
        before = copy.deepcopy(actor.state_dict())
        with self.assertRaises(ValueError):
            actor.copy_from_flat_hiking(HikingMoEActor(gate_hidden=()).state_dict())
        for name, value in actor.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
