"""Official file selection must retain the full walking reference timeline."""
from __future__ import annotations

import hashlib
import ast
from types import SimpleNamespace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.wmac_perceptive.hiking_data import prepare_hiking_dataset, verify_hiking_dataset
from src.wmac_perceptive.mirroring import build_g1_mirror_map
from src.wmac_perceptive.motions import read_motion
from src.wmac_perceptive.config import PerceptiveConfig, saved_training_config


class HikingOfficialDataTest(unittest.TestCase):
    def test_interpolation_matches_author_function_including_exclusive_endpoint(self):
        reference=Path('/tmp/hiking_locomotion_source/source/instinctlab/instinctlab')
        if not reference.is_dir():
            self.skipTest('Pinned Hiking source checkout is not available')
        namespace={'torch':torch,'np':np,'math_utils':SimpleNamespace(
            normalize=lambda x:torch.nn.functional.normalize(x,p=2,dim=-1,eps=1e-9))}
        for path,name in [(reference/'utils/math.py','quat_slerp_batch'),
                          (reference/'motion_reference/utils.py','motion_interpolate_bilinear')]:
            node=next(n for n in ast.parse(path.read_text()).body if isinstance(n,ast.FunctionDef) and n.name==name)
            node.decorator_list=[]
            exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),namespace)
        from src.wmac_perceptive.locomotion_data import interpolate_hiking_motion
        generator=torch.Generator().manual_seed(481)
        p=torch.randn(181,3,generator=generator)
        q=torch.nn.functional.normalize(torch.randn(181,4,generator=generator),dim=-1)
        joints=torch.randn(181,29,generator=generator)
        expected=namespace['motion_interpolate_bilinear'](p,q,joints,50.,50.)
        actual=interpolate_hiking_motion(p.numpy(),q.numpy(),joints.numpy())
        self.assertEqual(len(actual[0]),180)
        for a,e in zip(actual,expected):
            np.testing.assert_array_equal(a,e.numpy())

    def test_full_source_height_and_short_intervals_are_preserved(self):
        # A six-frame walking-source fixture deliberately violates the former
        # minimum-duration, velocity, support-height and continuity heuristics.
        # Official selection is the file list, so conversion must keep it all.
        mapping = build_g1_mirror_map()
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp)/"parkour_motion_without_run_retargetted.npz"
            xyz = np.array([[0, 0, .9], [.01, 0, .9], [.4, .02, 1.2],
                            [.41, 0, 1.2], [.8, .05, 1.5], [.81, 0, 1.5]], dtype=np.float32)
            joints = np.zeros((6, 29), dtype=np.float32)
            quaternion = np.tile(np.array([1., 0, 0, 0], dtype=np.float32), (6, 1))
            np.savez_compressed(p, framerate=np.array(50.), joint_names=np.array(mapping.joint_names),
                joint_pos=joints, base_pos_w=xyz, base_quat_w=quaternion)
            p.with_name("parkour_motion_without_run.yaml").write_text("selected_files:\n- "+p.name+"\n")
            digest = hashlib.sha256(p.read_bytes()).hexdigest()
            with patch("src.wmac_perceptive.hiking_data.ensure_hiking_source", return_value=p), \
                 patch("src.wmac_perceptive.hiking_data.HIKING_MOTION_SHA256", digest):
                paths = prepare_hiking_dataset(Path(tmp)/"unused-local-data", Path(tmp)/"cache")
                manifest = verify_hiking_dataset(paths.root)
                self.assertEqual(manifest["source_frames"], 6)
                self.assertEqual(manifest["walking_original_frames"], 5)
                self.assertEqual(manifest["walking_total_frames"], 10)
                self.assertIsNone(manifest["filters"])
                self.assertFalse(paths.recovery.exists())
                self.assertFalse(manifest["grail_motion_data_used"])
                raw = read_motion(paths.walk/"hiking_official_walk.npz")
                torso = mapping.body_names.index("torso_link")
                np.testing.assert_allclose(raw["body_pos_w"][:, torso], xyz[:-1], rtol=0, atol=2e-7)
                for name in ("joint_vel", "body_lin_vel_w", "body_ang_vel_w"):
                    np.testing.assert_array_equal(raw[name][-1], 0)
                self.assertEqual(prepare_hiking_dataset(None, paths.root), paths)
                target = paths.walk/"hiking_official_walk.npz"
                target.write_bytes(target.read_bytes()+b"changed")
                with self.assertRaisesRegex(ValueError, "modified"):
                    verify_hiking_dataset(paths.root)

    def test_new_default_and_old_saved_protocol_are_distinct(self):
        from dataclasses import asdict
        cfg = PerceptiveConfig.defaults("supermimic")
        cfg.validate()
        self.assertEqual(cfg.motion_data_protocol, "hiking_walk_v1")
        self.assertEqual(cfg.amp_feature_schema, "hiking_v1")
        self.assertEqual(cfg.discriminator_hidden, (1024, 512))
        legacy = asdict(PerceptiveConfig.defaults())
        legacy.pop("motion_data_protocol")
        saved = saved_training_config(legacy)
        self.assertEqual(saved["motion_data_protocol"], "filtered_walk_turn_v2")
        self.assertEqual(saved["amp_feature_schema"], "grounded_v2")

    def test_new_protocol_rejects_filtered_amp_features(self):
        cfg = PerceptiveConfig.defaults("supermimic")
        cfg.amp_feature_schema = "grounded_v2"
        with self.assertRaisesRegex(ValueError, "Hiking AMP"):
            cfg.validate()


if __name__ == "__main__":
    unittest.main()
