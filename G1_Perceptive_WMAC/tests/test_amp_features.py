"""CPU checks for legacy compatibility and shared grounded AMP kinematics."""

from __future__ import annotations

from pathlib import Path
import unittest
import xml.etree.ElementTree as ET

import torch

from src.amp_features import (
    G1_FOOT_REFERENCE_OFFSET,
    GROUNDED_SCHEMA,
    LEGACY_SCHEMA,
    amp_feature_dim,
    amp_feature_metadata,
    build_amp_features,
)
from src.wmac.state import quat_apply, quat_inv, quat_mul, quat_normalize, quat_to_matrix


class AmpFeaturesTest(unittest.TestCase):
    body_ids = [0, 2, 3, 5, 7, 8, 10, 11, 12, 13, 14, 15, 16]
    anchor_id = 4
    pelvis_id = 0
    foot_ids = [5, 10]

    def setUp(self):
        generator = torch.Generator().manual_seed(731)
        self.p = torch.randn((2, 3, 17, 3), generator=generator, dtype=torch.float64)
        self.q = quat_normalize(torch.randn((2, 3, 17, 4), generator=generator, dtype=torch.float64))
        self.v = torch.randn((2, 3, 17, 3), generator=generator, dtype=torch.float64)
        self.w = torch.randn((2, 3, 17, 3), generator=generator, dtype=torch.float64)

    def features(self, p=None, q=None, v=None, w=None, **kwargs):
        return build_amp_features(
            self.p if p is None else p,
            self.q if q is None else q,
            self.v if v is None else v,
            self.w if w is None else w,
            body_ids=self.body_ids, anchor_id=self.anchor_id,
            pelvis_id=self.pelvis_id, foot_ids=self.foot_ids,
            schema=kwargs.pop("schema", GROUNDED_SCHEMA), **kwargs,
        )

    def transform(self, rotation, translation):
        rotation = self.q.new_tensor(rotation)
        translation = self.p.new_tensor(translation)
        return (quat_apply(rotation, self.p) + translation,
                quat_mul(rotation, self.q), quat_apply(rotation, self.v), quat_apply(rotation, self.w))

    def test_legacy_exactly_preserves_existing_wmac_columns(self):
        anchor_inv = quat_inv(self.q[..., self.anchor_id:self.anchor_id + 1, :])
        selected = self.q[..., self.body_ids, :]
        terms = (
            quat_apply(anchor_inv, self.p[..., self.body_ids, :] - self.p[..., self.anchor_id:self.anchor_id + 1, :]),
            quat_to_matrix(quat_mul(anchor_inv, selected))[..., :2],
            quat_apply(quat_inv(selected), self.v[..., self.body_ids, :]),
            quat_apply(quat_inv(selected), self.w[..., self.body_ids, :]),
        )
        expected = torch.cat([term.reshape(2, 3, -1) for term in terms], -1)
        actual = self.features(schema=LEGACY_SCHEMA)
        self.assertEqual(actual.shape, (2, 3, 195))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.testing.assert_close(self.features()[..., :195], expected, rtol=0, atol=0)
        # A single body batch must follow precisely the same feature ordering.
        single = self.features(self.p[0, 0], self.q[0, 0], self.v[0, 0], self.w[0, 0])
        torch.testing.assert_close(single, self.features()[0, 0], rtol=0, atol=0)

    def test_yaw_and_horizontal_translation_are_invariant(self):
        angle = torch.tensor(.73, dtype=torch.float64)
        transformed = self.transform([torch.cos(angle / 2), 0., 0., torch.sin(angle / 2)], [3., -4., 0.])
        torch.testing.assert_close(self.features(*transformed), self.features(), rtol=2e-13, atol=2e-13)

    def test_gravity_flip_and_vertical_translation_are_visible(self):
        original = self.features()
        flipped = self.features(*self.transform([0., 1., 0., 0.], [0., 0., 0.]))
        # This was the old blind spot: even a full-body world flip was invisible.
        torch.testing.assert_close(flipped[..., :195], original[..., :195], rtol=2e-13, atol=2e-13)
        self.assertGreater((flipped[..., 195:198] - original[..., 195:198]).abs().max().item(), .5)
        translated = self.features(p=self.p + self.p.new_tensor([0., 0., 1.]))
        torch.testing.assert_close(translated[..., :198], original[..., :198], rtol=2e-13, atol=2e-13)
        torch.testing.assert_close(translated[..., 198:201], original[..., 198:201] + 1., rtol=2e-13, atol=2e-13)
        torch.testing.assert_close(translated[..., 201:], original[..., 201:], rtol=0, atol=0)
        # Heights are relative to the supplied floor, including per-batch floors.
        ground = torch.linspace(.1, .6, 6, dtype=self.p.dtype).reshape(2, 3)
        shifted = self.features(p=self.p + ground[..., None, None] * self.p.new_tensor([0., 0., 1.]),
                                ground_height=ground)
        torch.testing.assert_close(shifted, original, rtol=2e-13, atol=2e-13)

    def test_foot_reference_height_and_rigid_body_point_velocity(self):
        p = torch.zeros((1, 17, 3), dtype=torch.float64)
        p[..., 2] = .035
        q = torch.zeros((1, 17, 4), dtype=torch.float64)
        q[..., 0] = 1.
        v, w = torch.zeros_like(p), torch.zeros_like(p)
        v[:, self.foot_ids, 2] = .7
        w[:, self.foot_ids, 1] = 2.
        features = self.features(p, q, v, w)
        torch.testing.assert_close(features[:, 199:201], torch.zeros((1, 2), dtype=p.dtype))
        # omega=(0,2,0), offset=(.04,0,-.035), so cross-product z=-.08.
        torch.testing.assert_close(features[:, 201:203], torch.full((1, 2), .62, dtype=p.dtype))
        moved = self.features(p, q, v, w, foot_local_offsets=((.1, 0., -.02), (.2, 0., -.01)))
        torch.testing.assert_close(moved[:, 199:201], p.new_tensor([[.015, .025]]))
        torch.testing.assert_close(moved[:, 201:203], p.new_tensor([[.5, .3]]))
        # The defaults are anchored to actual source XML sites, not guessed sole geometry.
        xml = Path(__file__).resolve().parents[1] / "src/assets/robots/unitree_g1/xmls/g1.xml"
        tree = ET.parse(xml)
        for name in ("left_foot", "right_foot"):
            site = tree.find(f".//site[@name='{name}']")
            self.assertIsNotNone(site)
            self.assertEqual(tuple(float(x) for x in site.attrib["pos"].split()), G1_FOOT_REFERENCE_OFFSET)

    def test_backward_is_finite_at_identity_and_inverted_quaternions(self):
        p = self.p[0].detach().clone().requires_grad_()
        q = torch.zeros((3, 17, 4), dtype=torch.float64)
        q[0, :, 0], q[1, :, 1], q[2, :, 2] = 1., 1., 1.
        q.requires_grad_()
        v, w = self.v[0].detach().clone().requires_grad_(), self.w[0].detach().clone().requires_grad_()
        features = self.features(p, q, v, w)
        weights = torch.linspace(.1, 1.3, features.shape[-1], dtype=features.dtype)
        loss = (features * weights).sum()
        gradients = torch.autograd.grad(loss, (p, q, v, w))
        self.assertTrue(torch.isfinite(features).all())
        for gradient in gradients:
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0.)
        # Check the actual Jacobian at arbitrary tilted states, including all
        # quaternion and angular-velocity dependencies of the sole-point cues.
        inputs = tuple(x[0, 0].detach().clone().requires_grad_() for x in (self.p, self.q, self.v, self.w))
        self.assertTrue(torch.autograd.gradcheck(self.features, inputs, fast_mode=True))

    def test_schema_metadata_and_invalid_inputs(self):
        self.assertEqual(amp_feature_dim(LEGACY_SCHEMA), 195)
        self.assertEqual(amp_feature_dim(GROUNDED_SCHEMA), 203)
        metadata = amp_feature_metadata(GROUNDED_SCHEMA)
        self.assertEqual(metadata["fields"][-1], {
            "name": "foot_reference_world_vertical_velocity", "start": 201, "stop": 203})
        self.assertIn("not measured contacts", metadata["support_features"])
        with self.assertRaisesRegex(ValueError, "Unknown AMP feature schema"):
            amp_feature_dim("typo")
        with self.assertRaisesRegex(ValueError, "pelvis_id"):
            build_amp_features(self.p, self.q, self.v, self.w, body_ids=self.body_ids,
                               anchor_id=self.anchor_id, schema=GROUNDED_SCHEMA)
        with self.assertRaisesRegex(ValueError, "shape"):
            self.features(foot_local_offsets=((0., 0., 0.),))


if __name__ == "__main__":
    unittest.main()
