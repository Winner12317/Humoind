"""WM moments follow the saved protocol, including after the old sample cap."""

from __future__ import annotations

import io
import math
import unittest

import torch

from rsl_rl.modules.normalizer import EmpiricalNormalization
from src.wmac.networks import RunningNorm
from src.wmac_perceptive.latent_models import LatentPerceptiveWorldModel
from src.wmac_perceptive.models import TerrainPerceptiveWorldModel
from test_perceptive_models import PlaneOracle, SMALL, state_batch


class ContinuousWorldModelNormalizationTest(unittest.TestCase):
    def make_model(self, *, continuous=False):
        return LatentPerceptiveWorldModel(
            PlaneOracle(), SMALL, hidden=(8,), context_dim=100,
            continuous_normalization=continuous)

    def sample(self, model, count=32):
        state = state_batch(count).float()
        action = torch.linspace(-500., 500., count).unsqueeze(-1).expand(-1, 29)
        context = torch.full((count, model.context_dim), 200.)
        latent = torch.zeros(count, model.latent_dim)
        return state, action, context, latent

    def update(self, model, sample):
        state, action, context, latent = sample
        model.update_normalizers_latent(state, action, context, state, latent)

    def test_continuous_wm_keeps_collecting_at_old_boundary_and_matches_official_moments(self):
        model = self.make_model(continuous=True)
        model.in_norm.count.fill_(200_000_000)
        sample = self.sample(model)
        values = model._physical_inputs(*sample[:3])
        # Official SuperMimic subclasses this normalizer without an until cap.
        official = EmpiricalNormalization(model.in_dim)
        official.count.fill_(200_000_000)
        original_mean = model.in_norm.mean.clone()
        pointers = tuple(value.data_ptr() for value in model.in_norm.buffers())

        for _ in range(3):
            self.update(model, sample)
            official.update(values)

        self.assertTrue(math.isinf(model.in_norm.max_count))
        self.assertEqual(int(model.in_norm.count), 200_000_096)
        self.assertIsNone(official.until)
        self.assertFalse(torch.equal(model.in_norm.mean, original_mean))
        torch.testing.assert_close(model.in_norm.mean, official.mean, rtol=2e-6, atol=1e-10)
        torch.testing.assert_close(model.in_norm.var, official._var.squeeze(0), rtol=2e-6, atol=1e-7)
        torch.testing.assert_close(model.in_norm(values), official.eval()(values), rtol=2e-6, atol=1e-5)
        # Updates keep CUDA-graph-referenced buffers in place.
        self.assertEqual(pointers, tuple(value.data_ptr() for value in model.in_norm.buffers()))

    def test_loaded_moments_over_old_cap_continue_under_new_protocol(self):
        historical = self.make_model()
        historical.in_norm.count.fill_(200_000_032)
        historical.in_norm.mean.fill_(.125)
        historical.in_norm.var.fill_(.75)
        data = io.BytesIO()
        torch.save(historical.state_dict(), data)
        data.seek(0)

        resumed = self.make_model(continuous=True)
        resumed.load_state_dict(torch.load(data, weights_only=True), strict=True)
        for key, value in historical.in_norm.state_dict().items():
            torch.testing.assert_close(resumed.in_norm.state_dict()[key], value, rtol=0, atol=0)
        self.assertTrue(math.isinf(resumed.in_norm.max_count))
        old_mean = resumed.in_norm.mean.clone()
        self.update(resumed, self.sample(resumed))
        self.assertEqual(int(resumed.in_norm.count), 200_000_064)
        self.assertFalse(torch.equal(resumed.in_norm.mean, old_mean))
        # This change adds no checkpoint tensors or physical/latent head dimensions.
        self.assertEqual(set(resumed.state_dict()), set(historical.state_dict()))
        self.assertEqual(resumed.net[-1].out_features, 35 + 2 + 128)

    def test_historical_protocol_and_unrelated_normalizers_keep_existing_cap(self):
        historical = self.make_model()
        pixel = TerrainPerceptiveWorldModel(PlaneOracle(), SMALL, hidden=(8,))
        self.assertEqual(historical.in_norm.max_count, 200_000_000)
        self.assertEqual(pixel.in_norm.max_count, 200_000_000)
        self.assertEqual(RunningNorm(3).max_count, 200_000_000)
        historical.in_norm.count.fill_(200_000_000)
        before = {key: value.clone() for key, value in historical.in_norm.state_dict().items()}
        self.update(historical, self.sample(historical))
        for key, value in historical.in_norm.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_statistics_stay_fixed_during_imagined_forward_with_input_gradients(self):
        model = self.make_model(continuous=True)
        sample = self.sample(model)
        self.update(model, sample)
        before = {key: value.clone() for key, value in model.in_norm.state_dict().items()}
        state, action, context, latent = sample
        action = action.clone().requires_grad_(True)
        model.inputs_latent(state, action, context, latent).square().sum().backward()
        self.assertGreater(float(action.grad.abs().sum()), 0.)
        for key, value in model.in_norm.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
