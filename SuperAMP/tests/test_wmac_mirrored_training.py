"""Flat training must use the same mirrored data for resets and AMP experts."""
from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from scripts.train_wmac import configure_motion_dataset
from src.wmac.config import WmacConfig
from src.wmac.mirroring import build_g1_mirror_map
from src.wmac.motions import prepare_mirrored_dataset
from src.wmac.runner import ExpertPairs


ROOT = Path(__file__).resolve().parents[1]


def write_clip(path, marker, frames=4):
    path.parent.mkdir(parents=True, exist_ok=True)
    positions = np.zeros((frames, 30, 3), dtype=np.float32)
    positions[:, :, 0] = marker + np.arange(frames)[:, None]
    quaternion = np.zeros((frames, 30, 4), dtype=np.float32)
    quaternion[:, :, 0] = 1.
    np.savez_compressed(path, fps=np.asarray([50.], dtype=np.float32),
                        joint_pos=np.zeros((frames, 29), dtype=np.float32),
                        joint_vel=np.zeros((frames, 29), dtype=np.float32),
                        body_pos_w=positions, body_quat_w=quaternion,
                        body_lin_vel_w=np.zeros_like(positions),
                        body_ang_vel_w=np.zeros_like(positions))


def source_fixture(root):
    # Deliberately share a stem between groups: its directory determines
    # locomotion/recovery class, not a globally unique motion filename.
    write_clip(root/"WalkandRun/shared.npz", 10.)
    write_clip(root/"WalkandRun/second_walk.npz", 20., frames=5)
    write_clip(root/"Recovery/shared.npz", 100.)
    return root


def env_cfg_fixture(source):
    return SimpleNamespace(events={
        "init_motion_loader": SimpleNamespace(params={
            "motion_dir": str(source/"WalkandRun"),
            "recovery_dir": str(source/"Recovery"),
            "delay_reset_env_ratio": .4,
        }),
        "reset_from_motion": SimpleNamespace(params={
            "motion_dir": str(source/"WalkandRun"), "other_reset_option": 7,
        }),
    })


def frame_marker_features(positions, quaternion, linear, angular, **kwargs):
    # A controlled observable distinguishes source clips and individual frames.
    # Sampling must preserve this observable's within-clip adjacency.
    return positions[:, 0, :1]


class WmacMirroredTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.body_names = build_g1_mirror_map().body_names

    def make_experts(self, source, *, override=None, fraction=.5):
        env = SimpleNamespace(scene={"robot": SimpleNamespace(body_names=self.body_names)})
        if override is not None:
            env._wmac_motion_root = str(override)
        runner_cfg = SimpleNamespace(amp_motion_files=str(source),
                                     amp_body_names=("pelvis",), amp_anchor_name="pelvis")
        cfg = SimpleNamespace(task="Unitree-G1-AMP-Flat-Aligned",
                              amp_feature_schema="grounded_v2",
                              expert_recovery_fraction=fraction)
        with patch("src.wmac.runner.load_rl_cfg", return_value=runner_cfg), \
             patch("src.wmac.runner.build_amp_features", side_effect=frame_marker_features):
            return ExpertPairs(cfg, env, "cpu")

    def test_configure_redirects_both_reset_paths_and_returns_expert_root(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            source = source_fixture(project/"original")
            cfg = WmacConfig.for_profile("supermimic_official")
            cfg.motion_cache_root = "cache/flat_mirrors"
            env_cfg = env_cfg_fixture(source)
            originals = {p: p.read_bytes() for p in source.rglob("*.npz")}
            with patch("scripts.train_wmac.REPO", project):
                paths = configure_motion_dataset(cfg, env_cfg)
            self.assertEqual(paths.root, project/"cache/flat_mirrors")
            loader = env_cfg.events["init_motion_loader"].params
            self.assertEqual(loader["motion_dir"], str(paths.walk))
            self.assertEqual(loader["recovery_dir"], str(paths.recovery))
            self.assertEqual(env_cfg.events["reset_from_motion"].params,
                             {"motion_dir": str(paths.walk), "other_reset_option": 7})
            self.assertEqual(loader["delay_reset_env_ratio"], .4)
            self.assertEqual(cfg.expert_recovery_fraction, .5)
            self.assertEqual(cfg.recovery_fraction, .5)
            for path, contents in originals.items():
                self.assertEqual(path.read_bytes(), contents)
            self.assertEqual((paths.recovery/"shared.npz").read_bytes(),
                             (source/"Recovery/shared.npz").read_bytes())
            experts = self.make_experts(project/"unavailable_default", override=paths.root)
            self.assertEqual(len(experts.groups[0]), 2*((4-1)+(5-1)))
            self.assertEqual(len(experts.groups[1]), 4-1)

    def test_disabled_mirroring_leaves_original_reset_configuration_untouched(self):
        cfg = WmacConfig.for_profile("ppo_aligned")
        self.assertFalse(cfg.mirror_walk_motions)
        env_cfg = env_cfg_fixture(Path("/original/amp"))
        before = copy.deepcopy(env_cfg.events)
        with patch("src.wmac.motions.prepare_mirrored_dataset") as prepare:
            self.assertIsNone(configure_motion_dataset(cfg, env_cfg))
        prepare.assert_not_called()
        for name, term in before.items():
            self.assertEqual(env_cfg.events[name].params, term.params)

    def test_configuration_rejects_overwriting_original_motion_assets(self):
        with tempfile.TemporaryDirectory() as temp:
            source = source_fixture(Path(temp)/"original")
            cfg = WmacConfig.for_profile("supermimic_official")
            cfg.motion_cache_root = str(source)
            env_cfg = env_cfg_fixture(source)
            before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob("*.npz")}
            before_events = copy.deepcopy(env_cfg.events)
            with self.assertRaisesRegex(ValueError, "separate"):
                configure_motion_dataset(cfg, env_cfg)
            for path, digest in before.items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), digest)
            for name, term in before_events.items():
                self.assertEqual(env_cfg.events[name].params, term.params)

    def test_non_native_reset_group_is_rejected_before_cache_preparation(self):
        cfg = WmacConfig.for_profile("supermimic_official")
        env_cfg = env_cfg_fixture(Path("/original/amp"))
        env_cfg.events["init_motion_loader"].params["motion_dir"] = "/other_task/terrain_motions"
        with patch("src.wmac.motions.prepare_mirrored_dataset") as prepare:
            with self.assertRaisesRegex(ValueError, "WalkandRun/Recovery"):
                configure_motion_dataset(cfg, env_cfg)
        prepare.assert_not_called()

    def test_expert_directory_classification_preserves_same_stem_and_adjacency(self):
        with tempfile.TemporaryDirectory() as temp:
            source = source_fixture(Path(temp)/"original")
            experts = self.make_experts(source)
            self.assertEqual(len(experts.groups[0]), (4-1)+(5-1))
            self.assertEqual(len(experts.groups[1]), 4-1)
            self.assertTrue(bool((experts.groups[0] < 50).all()))
            self.assertTrue(bool((experts.groups[1] >= 100).all()))
            for group in experts.groups:
                torch.testing.assert_close(group[:, 1]-group[:, 0], torch.ones_like(group[:, 0]))

    def test_mirroring_does_not_dilute_requested_recovery_sampling_weight(self):
        with tempfile.TemporaryDirectory() as temp:
            source = source_fixture(Path(temp)/"original")
            paths = prepare_mirrored_dataset(source, Path(temp)/"mirrored")
            for fraction in (.5, .25):
                for root in (source, paths.root):
                    experts = self.make_experts(root, fraction=fraction)
                    first, second = experts.sample(128)
                    self.assertEqual(int((first[:, 0] >= 100).sum()), round(128*fraction))
                    torch.testing.assert_close(second-first, torch.ones_like(first))

    def test_experts_reject_other_task_group_even_when_a_file_has_one_frame(self):
        with tempfile.TemporaryDirectory() as temp:
            source = source_fixture(Path(temp)/"original")
            for frames in (2, 1):
                write_clip(source/"OtherTask/not_flat_amp.npz", 500., frames=frames)
                with self.subTest(frames=frames), self.assertRaisesRegex(ValueError, "Unexpected expert motion group"):
                    self.make_experts(source)

    def test_flat_mirror_modules_have_no_perceptive_import_dependency(self):
        for relative in ("src/wmac/mirroring.py", "src/wmac/motions.py",
                         "scripts/prepare_wmac_mirrored_motions.py", "scripts/train_wmac.py",
                         "src/wmac/runner.py"):
            tree = ast.parse((ROOT/relative).read_text())
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imports.append(node.module)
            self.assertFalse(any(name.startswith(("src.wmac_perceptive", "src.tasks.amp_perceptive"))
                                 for name in imports), relative)


if __name__ == "__main__":
    unittest.main()
