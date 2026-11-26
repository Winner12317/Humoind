"""Terrain-relative replay strata and original-project expert data isolation."""
from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.wmac_perceptive.expert import PerceptiveExpertPairs
from src.wmac_perceptive.mirroring import build_g1_mirror_map
from src.wmac_perceptive.motions import prepare_mirrored_dataset, read_motion
from src.wmac_perceptive.perception import PerceptionSpec, unpack_actor_input
from src.wmac_perceptive.replay import PerceptiveReplay


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/"src/assets/motions/g1/amp"
AMP_BODIES = ("pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
              "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
              "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
              "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link")


def tiny_spec():
    return PerceptionSpec(height=2, width=3, ring_length=4, frames=2, stride=2, delay=1,
                          crop_top=0, crop_bottom=0, crop_left=0, crop_right=0)


def transition(step, spec, *, invalid=False):
    count = 4
    state = torch.zeros(count, 71)
    state[:, 2] = torch.tensor([2.2, 2.8, 3.1, .8])
    context = torch.zeros(count, 35); context[2, 34] = 7.
    history = torch.arange(count*384, dtype=torch.float32).reshape(count, 384)*1e-5+step*.1
    frames = torch.empty((count, *spec.frame_shape))
    ring = torch.empty((count, *spec.ring_shape))
    for env in range(count):
        frames[env].fill_(step*.1+env*.01)
        ring[env].fill_(step*.1+env*.01)
    next_frames = frames+.05
    if invalid: next_frames[1, 0, 0, 0] = float("nan")
    boundary = torch.zeros(count, dtype=torch.bool)
    if step == 1: boundary[0] = True
    return {"state": state, "next_state": state+.001, "context": context,
            "action": torch.zeros(count, 29), "actor_history": history,
            "relative_height": torch.tensor([.2, .8, .85, .8]),
            "depth_ring": ring, "depth_frame": frames, "next_depth_frame": next_frames,
            "boundary": boundary, "terminated": boundary.clone(), "timeout": torch.zeros(count, dtype=torch.bool)}


class PerceptiveReplayTest(unittest.TestCase):
    def replay(self):
        spec = tiny_spec()
        replay = PerceptiveReplay(6, 4, validation_env_fraction=.25, spec=spec)
        for step in range(4): replay.add(transition(step, spec, invalid=step == 2))
        return replay, spec

    def test_recovery_groups_use_local_ground_height_or_active_counter(self):
        replay, _ = self.replay()
        # High world z=2.2 is still a fallen robot 0.2m above a raised platform.
        self.assertTrue(bool(replay._recovery[:4, 0].all()))
        self.assertFalse(bool(replay._recovery[:4, 1].any()))
        self.assertTrue(bool(replay._recovery[:4, 2].all()))
        samples = replay.sample_starts(128, recovery_fraction=1., fields=("state",), split="train")
        self.assertEqual(set(samples["env_index"].tolist()), {0, 2})
        self.assertTrue(bool((samples["state"][:, 2] > 2.).all()))

    def test_pack_fetches_identical_indices_and_preserves_fp32_proprioception(self):
        replay, spec = self.replay()
        self.assertEqual(replay.data["actor_history"].dtype, torch.float32)
        for name in ("depth_ring", "depth_frame", "next_depth_frame"):
            self.assertEqual(replay.data[name].dtype, torch.float16)
        samples = replay.sample_starts(17, recovery_fraction=0., fields=("state", "actor_history"))
        self.assertEqual(samples["actor_history"].dtype, torch.float32)
        self.assertNotIn("depth_ring", samples)
        proprio, ring = unpack_actor_input(samples["actor_history"], spec)
        rows, envs = samples["time_index"], samples["env_index"]
        torch.testing.assert_close(proprio, replay.data["actor_history"][rows, envs], rtol=0, atol=0)
        torch.testing.assert_close(ring, replay.data["depth_ring"][rows, envs].float(), rtol=0, atol=0)
        # At least one original value would change if incorrectly stored as fp16.
        self.assertTrue(bool((proprio != proprio.half().float()).any()))
        model_starts = replay.sample_starts(5, fields=("state", "depth_frame", "next_depth_frame"))
        self.assertNotIn("depth_ring", model_starts)
        self.assertNotIn("actor_history", model_starts)
        self.assertEqual(model_starts["depth_frame"].dtype, torch.float16)
        all_fields = replay.sample_starts(3, fields=None)
        self.assertEqual(all_fields["actor_history"].shape[-1], spec.packed_dim)
        self.assertNotIn("depth_ring", all_fields)

    def test_raw_windows_retain_splits_boundaries_and_invalid_quarantine(self):
        replay, spec = self.replay()
        self.assertEqual(replay.num_valid("train"), 11)
        self.assertEqual(replay.num_valid("validation"), 4)
        train = replay.sample_windows(128, 3, recovery_fraction=0.,
                    fields=("state", "actor_history", "depth_frame", "next_depth_frame"), split="train")
        self.assertEqual(train["actor_history"].shape, (128, 3, spec.proprio_dim))
        self.assertNotIn("depth_ring", train)
        self.assertTrue(bool((train["env_index"] < 3).all()))
        self.assertTrue(bool(torch.isfinite(train["next_depth_frame"]).all()))
        found_boundary = found_invalid = False
        for times, envs, mask in zip(train["time_index"], train["env_index"], train["mask"]):
            if int(envs[0]) == 0 and int(times[0]) == 0:
                torch.testing.assert_close(mask, torch.tensor([1., 1., 0.]))
                found_boundary = True
            if int(envs[0]) == 1:
                expected = torch.tensor([1., 1., 0.]) if int(times[0]) == 0 else torch.tensor([1., 0., 0.])
                torch.testing.assert_close(mask, expected)
                found_invalid = True
        self.assertTrue(found_boundary and found_invalid)
        validation = replay.sample_starts(20, fields=("state", "actor_history"), split="validation")
        self.assertTrue(bool((validation["env_index"] == 3).all()))

    def test_ring_wrap_and_checkpoint_restore_preserve_relative_strata_and_schema(self):
        replay, spec = self.replay()
        for step in range(4, 9): replay.add(transition(step, spec))
        payload = copy.deepcopy(replay.state_dict())
        restored = PerceptiveReplay(6, 4, validation_env_fraction=.25, spec=spec)
        restored.load_state_dict(payload)
        self.assertEqual((restored.ptr, restored.size, restored.total_steps), (replay.ptr, replay.size, replay.total_steps))
        for name in replay.data:
            torch.testing.assert_close(restored.data[name], replay.data[name], rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(restored._recovery, replay._recovery, rtol=0, atol=0)
        samples = restored.sample_starts(20, fields=("actor_history",), split="validation")
        self.assertTrue(bool((samples["time_index"] >= 3).all()))
        before = restored.ptr
        bad = copy.deepcopy(payload); bad["perception_spec"]["width"] += 1
        with self.assertRaisesRegex(ValueError, "schema"):
            restored.load_state_dict(bad)
        self.assertEqual(restored.ptr, before)


class PerceptiveExpertTest(unittest.TestCase):
    def prepared(self, directory):
        source, output = directory/"source", directory/"mirrored"
        for group in ("WalkandRun", "Recovery"):
            (source/group).mkdir(parents=True)
            motion = read_motion(sorted((SOURCE/group).glob("*.npz"))[0])
            short = {name: values if name == "fps" else values[:8] for name, values in motion.items()}
            np.savez_compressed(source/group/"clip.npz", **short)
        return prepare_mirrored_dataset(source, output)

    def arguments(self):
        cfg = SimpleNamespace(task="Unitree-G1-AMP-Perceptive-Hiking", amp_feature_schema="grounded_v2",
                              expert_recovery_fraction=.5)
        env = SimpleNamespace(scene={"robot": SimpleNamespace(body_names=build_g1_mirror_map().body_names)}, step_dt=.02)
        policy = SimpleNamespace(amp_body_names=AMP_BODIES, amp_anchor_name="torso_link")
        return cfg, env, policy

    def test_pairs_never_cross_clip_boundaries_and_recovery_sampling_is_preserved(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = self.prepared(Path(temp))
            cfg, env, policy = self.arguments()
            def feature_ids(p, q, v, w, **kwargs):
                return torch.arange(len(p), device=p.device).float()[:, None].expand(len(p), 203)
            with patch("src.wmac_perceptive.expert.load_rl_cfg", return_value=policy), \
                 patch("src.wmac_perceptive.expert.build_amp_features", side_effect=feature_ids):
                expert = PerceptiveExpertPairs(cfg, env, "cpu", paths.root)
            self.assertEqual([len(group) for group in expert.groups], [14, 7])
            for group in expert.groups:
                torch.testing.assert_close(group[:, 1, 0]-group[:, 0, 0], torch.ones(len(group)), rtol=0, atol=0)
            self.assertEqual([g["clip_count"] for g in expert.metadata["groups"]], [2, 1])
            self.assertEqual([g["pair_count"] for g in expert.metadata["groups"]], [14, 7])
            self.assertFalse(expert.metadata["hiking_motion_data_used"])
            first, second = expert.sample(9)
            self.assertEqual(first.shape, (9, 203))
            torch.testing.assert_close(second[:, 0]-first[:, 0], torch.ones(9), rtol=0, atol=0)

    def test_real_grounded_features_have_203_dimensions_and_robot_order_is_checked(self):
        with tempfile.TemporaryDirectory() as temp:
            paths = self.prepared(Path(temp))
            cfg, env, policy = self.arguments()
            with patch("src.wmac_perceptive.expert.load_rl_cfg", return_value=policy):
                expert = PerceptiveExpertPairs(cfg, env, "cpu", paths.root)
                self.assertEqual(expert.groups[0].shape, (14, 2, 203))
                self.assertEqual(expert.groups[1].shape, (7, 2, 203))
                self.assertTrue(all(bool(torch.isfinite(group).all()) for group in expert.groups))
                env.scene["robot"].body_names = tuple(reversed(env.scene["robot"].body_names))
                with self.assertRaisesRegex(ValueError, "body order"):
                    PerceptiveExpertPairs(cfg, env, "cpu", paths.root)


if __name__ == "__main__":
    unittest.main()
