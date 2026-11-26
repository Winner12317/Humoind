"""Kinematic invariants and direct parity with the pinned official GenCoord.

The source-parity tests additionally run when the audited upstream checkout is
present. Set SUPERMIMIC_REFERENCE_CHECKOUT to another checkout of the same
commit; source hashes are checked before extracting only the tested functions.
The normal suite needs no network or external checkout for invariant checks.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import math
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import torch

from src.wmac.gencoord import (
    GENCOORD_NOMINAL_ERRORS,
    gencoord_features,
    gencoord_loss,
    gencoord_loss_terms,
    integrate_velocity_increment,
    velocity_increment_targets,
)
from src.wmac.state import quat_apply, quat_from_rotvec, quat_mul, quat_to_matrix


def physical_state(*prefix: int) -> torch.Tensor:
    state = torch.zeros(*prefix, 71, dtype=torch.float64)
    state[..., 2] = 0.8
    state[..., 3] = 1
    return state


def random_state(*prefix: int) -> torch.Tensor:
    state = torch.randn(*prefix, 71, dtype=torch.float64) * 0.3
    state[..., 2] += 0.8
    state[..., 3:7] = quat_from_rotvec(torch.randn(*prefix, 3, dtype=torch.float64))
    return state


class GenCoordKinematicsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(31)

    def test_representation_layout_matches_root_rotation_matrix(self):
        state = random_state(2, 3)
        rep = gencoord_features(state)
        rot_inv = quat_to_matrix(state[..., 3:7]).transpose(-2, -1)
        expected_gravity = -rot_inv[..., :, 2]
        self.assertEqual(rep.shape, (2, 3, 68))
        torch.testing.assert_close(rep[..., :1], state[..., 2:3])
        torch.testing.assert_close(rep[..., 1:4], expected_gravity)
        torch.testing.assert_close(rep[..., 4:7], (rot_inv @ state[..., 7:10, None]).squeeze(-1))
        torch.testing.assert_close(rep[..., 7:10], (rot_inv @ state[..., 10:13, None]).squeeze(-1))
        torch.testing.assert_close(rep[..., 10:], state[..., 13:])

    def test_representation_is_invariant_to_horizontal_translation_and_world_yaw(self):
        state = random_state(4)
        yaw = quat_from_rotvec(state.new_tensor([0, 0, 2.7])).expand(4, -1)
        transformed = state.clone()
        transformed[..., :3] = quat_apply(yaw, state[..., :3]) + state.new_tensor([4, -8, 0])
        transformed[..., 3:7] = quat_mul(yaw, state[..., 3:7])
        transformed[..., 7:10] = quat_apply(yaw, state[..., 7:10])
        transformed[..., 10:13] = quat_apply(yaw, state[..., 10:13])
        torch.testing.assert_close(gencoord_features(transformed), gencoord_features(state))

    def test_zero_increment_advances_nonzero_velocities_instead_of_freezing_pose(self):
        state = physical_state(2)
        state[..., 7] = 1
        state[..., 42] = 0.5
        predicted = integrate_velocity_increment(state, state.new_zeros(2, 35), .02, 4)
        torch.testing.assert_close(predicted[..., 0], state.new_full((2,), .02))
        torch.testing.assert_close(predicted[..., 13], state.new_full((2,), .01))
        torch.testing.assert_close(predicted[..., 7:13], state[..., 7:13])
        torch.testing.assert_close(velocity_increment_targets(state, predicted), state.new_zeros(2, 35))

    def test_weighted_pose_matches_constant_acceleration_semi_implicit_substeps(self):
        state = random_state(3)
        state[..., 10:13] = 0
        increment = torch.randn(3, 35, dtype=state.dtype) * 0.1
        increment[..., 3:6] = 0
        dt = .02
        delta_v_world = quat_apply(state[..., 3:7], increment[..., :3])
        for decimation in (1, 2, 4, 20):
            p, v = state[..., :3].clone(), state[..., 7:10].clone()
            joints, joint_v = state[..., 13:42].clone(), state[..., 42:71].clone()
            for _ in range(decimation):
                v += delta_v_world / decimation
                joint_v += increment[..., 6:] / decimation
                p += v * (dt / decimation)
                joints += joint_v * (dt / decimation)
            predicted = integrate_velocity_increment(state, increment, dt, decimation)
            torch.testing.assert_close(predicted[..., :3], p)
            torch.testing.assert_close(predicted[..., 7:10], v)
            torch.testing.assert_close(predicted[..., 13:42], joints)
            torch.testing.assert_close(predicted[..., 42:71], joint_v)

    def test_large_turn_keeps_world_velocity_and_uses_same_frame_for_targets(self):
        state = physical_state()
        state[7] = 1
        state[12] = math.pi / 2
        predicted = integrate_velocity_increment(state, state.new_zeros(35), 1., 4)
        torch.testing.assert_close(predicted[:3], state[:3] + state.new_tensor([1, 0, 0]))
        torch.testing.assert_close(predicted[7:13], state[7:13])
        torch.testing.assert_close(gencoord_features(predicted)[4:7], state.new_tensor([0, -1, 0]), atol=2e-9, rtol=1e-9)
        torch.testing.assert_close(velocity_increment_targets(state, predicted), state.new_zeros(35))

    def test_increment_targets_round_trip_current_body_frame(self):
        state = random_state(2, 3)
        increment = torch.randn(2, 3, 35, dtype=state.dtype) * .2
        predicted = integrate_velocity_increment(state, increment, .02, 4)
        torch.testing.assert_close(velocity_increment_targets(state, predicted), increment)

    def test_poses_receive_gradient_through_velocity_increment(self):
        state = physical_state(1)
        increment = state.new_zeros(1, 35, requires_grad=True)
        predicted = integrate_velocity_increment(state, increment, .02, 4)
        root_gradient, = torch.autograd.grad(predicted[..., 0].sum(), increment, retain_graph=True)
        joint_gradient, = torch.autograd.grad(predicted[..., 13].sum(), increment)
        expected_root, expected_joint = torch.zeros_like(increment), torch.zeros_like(increment)
        expected_root[..., 0] = .02 * .625
        expected_joint[..., 6] = .02 * .625
        torch.testing.assert_close(root_gradient, expected_root)
        torch.testing.assert_close(joint_gradient, expected_joint)

    def test_gradients_are_finite_at_identity_and_cover_every_increment_channel(self):
        state = physical_state(2).requires_grad_(True)
        increment = state.new_zeros(2, 35, requires_grad=True)
        predicted = integrate_velocity_increment(state, increment, .02, 4)
        (predicted.sum() + gencoord_features(predicted).sum()).backward()
        self.assertTrue(torch.isfinite(state.grad).all())
        self.assertTrue(torch.isfinite(increment.grad).all())
        self.assertTrue((increment.grad.abs().sum(0) > 0).all())

    def test_losses_use_official_pose_and_joint_channels_with_bryson_scaling(self):
        target = physical_state(3)
        predicted = target.clone()
        predicted[0, :3] += .01
        predicted[1, 13:42] += .1
        predicted[2, 42:71] += 3.14
        self.assertEqual(tuple(gencoord_loss_terms(predicted)), tuple(GENCOORD_NOMINAL_ERRORS))
        torch.testing.assert_close(gencoord_loss(predicted, target), target.new_tensor([3, 29, 29]) / 70)
        # The official objective scores root twist through integrated pose.
        predicted = target.clone()
        predicted[..., 7:13] += 10
        torch.testing.assert_close(gencoord_loss(predicted, target), target.new_zeros(3))

    def test_orientation_loss_is_quaternion_sign_invariant_and_finite_at_identity(self):
        target = physical_state(2)
        predicted = target.clone()
        predicted[..., 3:7] *= -1
        torch.testing.assert_close(gencoord_loss(predicted, target), target.new_zeros(2))
        angle = .1
        predicted[0, 3:7] = quat_from_rotvec(target.new_tensor([0, 0, angle]))
        expected = 2 * (1 - math.cos(angle)) / .02 ** 2 / 70
        torch.testing.assert_close(gencoord_loss(predicted, target), target.new_tensor([expected, 0]))
        predicted = target.clone().requires_grad_(True)
        gencoord_loss(predicted, target).sum().backward()
        self.assertTrue(torch.isfinite(predicted.grad).all())

    def test_control_timing_and_shapes_are_validated_without_batch_broadcasting(self):
        state, increment = physical_state(2), torch.zeros(2, 35, dtype=torch.float64)
        for dt in (0, -1, math.inf, math.nan, True, ".02"):
            with self.subTest(dt=dt), self.assertRaisesRegex(ValueError, "dt"):
                integrate_velocity_increment(state, increment, dt, 4)
        for decimation in (0, -1, 1.5, True):
            with self.subTest(decimation=decimation), self.assertRaisesRegex(ValueError, "decimation"):
                integrate_velocity_increment(state, increment, .02, decimation)
        for bad in (increment[:1], increment[:, :-1], increment.float()):
            with self.assertRaises(ValueError):
                integrate_velocity_increment(state, bad, .02, 4)
        with self.assertRaises(ValueError):
            gencoord_features(state[..., :-1])
        with self.assertRaises(ValueError):
            gencoord_features(torch.zeros(2, 71, dtype=torch.int64))
        with self.assertRaises(ValueError):
            velocity_increment_targets(state, state[:1])
        with self.assertRaises(ValueError):
            gencoord_loss(state, state.float())


def _load_official_reference() -> SimpleNamespace:
    """Execute only hash-verified upstream methods, without its environment."""
    from mjlab.utils.lab_api.math import (
        matrix_from_quat,
        quat_apply as official_quat_apply,
        quat_apply_inverse,
        quat_mul as official_quat_mul,
    )

    root = Path(os.environ.get("SUPERMIMIC_REFERENCE_CHECKOUT", "/tmp/supermimic-official-audit-20261111"))
    files = {
        "src/supermimic/utils/gencoord_helpers.py": "35e6421b41f517cde75a1d339e79cf84e2577fd035a7b99d7477792ad7cd82a0",
        "src/supermimic/utils/helpers.py": "e15aacee141961e9b99fe669f394ae811b054a200bf665079f5953d5ebd0f78a",
        "src/supermimic/representations.py": "b863f79d32acd1486b9e03d0ff6a3f65cbc3bf353ef0d4688ef8dbb7d7c89869",
        "src/supermimic/learners/wm_learner.py": "1e2d489a607677032d9d90de2511c2e8e1750bd845dcc7966026f23a65842665",
    }
    namespace = {"torch": torch, "math": math, "quat_apply": official_quat_apply,
                 "quat_apply_inverse": quat_apply_inverse, "quat_mul": official_quat_mul,
                 "matrix_from_quat": matrix_from_quat, "dataclass": dataclass}
    wanted = {"infer_num_joints_from_state_dim", "full_state_offsets", "decompose_state",
              "decompose_rep_state", "rep_state_offsets", "compute_gencoord_state", "quat_integrate",
              "_root_loss_blocks", "apply_prediction", "reconstruct_full_state", "loss_blocks",
              "_compute_losses"}
    for relative, expected_sha in files.items():
        path = root / relative
        if not path.exists():
            raise unittest.SkipTest("pinned official checkout unavailable; set SUPERMIMIC_REFERENCE_CHECKOUT")
        source = path.read_bytes()
        if hashlib.sha256(source).hexdigest() != expected_sha:
            raise AssertionError(f"official reference source hash mismatch: {path}")
        tree = ast.parse(source, filename=str(path))
        nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted]
        for node in tree.body:
            if isinstance(node, ast.ClassDef) and node.name in ("GenCoordRepresentation", "WorldModelLearner"):
                nodes.extend(method for method in node.body if isinstance(method, ast.FunctionDef) and method.name in wanted)
            elif isinstance(node, ast.ClassDef) and node.name == "WorldModelLearnerCfg":
                nodes.append(node)
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


def _to_official_state(state: torch.Tensor) -> torch.Tensor:
    return torch.cat((state[..., :7], state[..., 13:42], state[..., 7:13], state[..., 42:71]), -1)


def _from_official_state(state: torch.Tensor) -> torch.Tensor:
    return torch.cat((state[..., :7], state[..., 36:42], state[..., 7:36], state[..., 42:71]), -1)


class GenCoordOfficialSourceParityTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.official = _load_official_reference()

    def setUp(self):
        torch.manual_seed(42)
        self.rep = SimpleNamespace(num_joints=29, substep_coeff=.625)

    def _official_step(self, state, increment, decimation=4):
        self.rep.substep_coeff = (decimation + 1) / (2 * decimation)
        official_state = _to_official_state(state)
        curr_rep = self.official.compute_gencoord_state(official_state)
        aux, next_rep = self.official.apply_prediction(self.rep, official_state[..., :7], curr_rep, increment, .02)
        return _from_official_state(self.official.reconstruct_full_state(self.rep, aux, next_rep))

    def test_features_match_pinned_official_function(self):
        state = random_state(2, 3)
        expected = self.official.compute_gencoord_state(_to_official_state(state))
        torch.testing.assert_close(gencoord_features(state), expected, atol=2e-14, rtol=2e-14)

    def test_integrated_state_matches_official_source_at_multiple_decimations(self):
        state = random_state(2, 3)
        increment = torch.randn(2, 3, 35, dtype=state.dtype) * .3
        for decimation in (1, 2, 4, 20):
            actual = integrate_velocity_increment(state, increment, .02, decimation)
            torch.testing.assert_close(actual, self._official_step(state, increment, decimation), atol=2e-14, rtol=2e-14)

    def test_state_and_increment_gradients_match_official_source(self):
        for identity in (False, True):
            state = (physical_state(2) if identity else random_state(2)).requires_grad_(True)
            increment = (torch.zeros(2, 35, dtype=state.dtype) if identity else torch.randn(2, 35, dtype=state.dtype) * .1).requires_grad_(True)
            weight = torch.randn_like(state)
            actual = integrate_velocity_increment(state, increment, .02, 4)
            expected = self._official_step(state, increment)
            actual_grads = torch.autograd.grad((actual * weight).sum(), (state, increment), retain_graph=True)
            expected_grads = torch.autograd.grad((expected * weight).sum(), (state, increment))
            for actual_grad, expected_grad in zip(actual_grads, expected_grads):
                self.assertTrue(torch.isfinite(actual_grad).all())
                torch.testing.assert_close(actual_grad, expected_grad, atol=3e-13, rtol=3e-13)

    def test_multistep_state_and_gradients_match_official_source(self):
        state = random_state(2).requires_grad_(True)
        increments = (torch.randn(8, 2, 35, dtype=state.dtype) * .1).requires_grad_(True)
        actual, expected = state, state
        for increment in increments:
            actual = integrate_velocity_increment(actual, increment, .02, 4)
            expected = self._official_step(expected, increment)
        torch.testing.assert_close(actual, expected, atol=5e-14, rtol=5e-14)
        actual_grads = torch.autograd.grad(actual.square().sum(), (state, increments), retain_graph=True)
        expected_grads = torch.autograd.grad(expected.square().sum(), (state, increments))
        for actual_grad, expected_grad in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual_grad, expected_grad, atol=3e-12, rtol=3e-12)

    def test_loss_channels_and_per_sample_loss_match_official_source(self):
        state, target = random_state(2, 3), random_state(2, 3)
        full = _to_official_state(state)
        rep = self.official.compute_gencoord_state(full)
        expected = self.official.loss_blocks(self.rep, rep, full[..., :7])
        for name, actual in gencoord_loss_terms(state).items():
            torch.testing.assert_close(actual, expected[name], atol=2e-14, rtol=2e-14)
        target_full = _to_official_state(target)
        target_rep = self.official.compute_gencoord_state(target_full)
        target_blocks = self.official.loss_blocks(self.rep, target_rep, target_full[..., :7])
        expected_loss = sum(((expected[name] - target_blocks[name]).square() / nominal ** 2).sum(-1)
                            for name, nominal in GENCOORD_NOMINAL_ERRORS.items()) / 70
        torch.testing.assert_close(gencoord_loss(state, target), expected_loss, atol=2e-11, rtol=2e-14)

    def test_loss_reduction_and_gradients_match_official_learner_function(self):
        state, target = random_state(2, 3).requires_grad_(True), random_state(2, 3)
        full, target_full = _to_official_state(state), _to_official_state(target)
        rep, target_rep = (self.official.compute_gencoord_state(item) for item in (full, target_full))
        cfg = self.official.WorldModelLearnerCfg()
        nominals = {
            "root_pos": cfg.nominal_error_root_pose_pos,
            "root_ori": cfg.nominal_error_root_pose_ori,
            "joint_pos": cfg.nominal_error_joint_pos,
            "joint_vel": cfg.nominal_error_joint_vel,
        }
        learner = SimpleNamespace(
            representation=SimpleNamespace(loss_blocks=lambda rep_state, aux: self.official.loss_blocks(self.rep, rep_state, aux)),
            loss_group_weights={name: 1 / nominal ** 2 for name, nominal in nominals.items()},
        )
        expected, _ = self.official._compute_losses(learner, rep, full[..., :7], target_rep, target_full[..., :7])
        actual = gencoord_loss(state, target).mean()
        torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-14)
        actual_gradient, = torch.autograd.grad(actual, state, retain_graph=True)
        expected_gradient, = torch.autograd.grad(expected, state)
        torch.testing.assert_close(actual_gradient, expected_gradient, atol=2e-12, rtol=2e-13)


if __name__ == "__main__":
    unittest.main()
