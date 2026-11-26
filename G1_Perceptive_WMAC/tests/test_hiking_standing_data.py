"""Existing locked G1 pose supplements, without editing Hiking trajectories."""
from __future__ import annotations

from contextlib import contextmanager
import json
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.wmac_perceptive import hiking_data, standing_data
from src.wmac_perceptive.mirroring import build_g1_mirror_map, mirror_motion
from src.wmac_perceptive.motions import read_motion, file_sha256


class HikingStandingDataTest(unittest.TestCase):
    @contextmanager
    def source_fixture(self):
        mapping=build_g1_mirror_map()
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            source=root/"parkour_motion_without_run_retargetted.npz"
            xyz=np.zeros((15,3),np.float32)
            xyz[:,0]=np.linspace(0,1,15);xyz[:,2]=np.linspace(.9,1.2,15)
            np.savez_compressed(source,framerate=np.array(50.),joint_names=np.array(mapping.joint_names),
                joint_pos=np.zeros((15,29),np.float32),base_pos_w=xyz,
                base_quat_w=np.tile(np.array([1.,0.,0.,0.],np.float32),(15,1)))
            source.with_name('parkour_motion_without_run.yaml').write_text('selected_files:\n- '+source.name+'\n')
            digest=file_sha256(source)
            with patch.object(hiking_data,'ensure_hiking_source',return_value=source), \
                    patch.object(hiking_data,'HIKING_MOTION_SHA256',digest), \
                    patch.object(standing_data,'HIKING_MOTION_SHA256',digest), \
                    patch('urllib.request.urlopen',side_effect=AssertionError('No download expected')):
                yield root

    def test_exact_bundled_source_and_unknown_original_rate_are_explicit(self):
        metadata=standing_data.verify_standing_source()
        self.assertEqual(metadata['sha256'],standing_data.STANDING_CSV_SHA256)
        self.assertEqual(metadata['source_commit'],'fe8db31ef14dc1941459c87bdeeb081de0eb68af')
        self.assertIsNone(metadata['source_fps'])
        self.assertIsNone(metadata['source_recorded_duration_seconds'])
        self.assertEqual(metadata['output_representation_fps'],50.)
        raw=np.loadtxt(standing_data.STANDING_BUNDLE/standing_data.STANDING_CSV_NAME,delimiter=',')
        self.assertEqual(raw.shape,(1201,36))
        np.testing.assert_array_equal(raw,np.broadcast_to(raw[:1],raw.shape))

    def test_converted_pose_has_named_g1_joints_pelvis_root_and_exact_zero_velocity(self):
        motion=standing_data.convert_legacy_g1_standing()
        mapping=build_g1_mirror_map()
        self.assertEqual(motion['joint_pos'].shape,(1201,29))
        self.assertEqual(float(motion['fps'][0]),50.)
        known={'left_hip_pitch_joint':-.312,'right_hip_pitch_joint':-.312,
               'left_knee_joint':.669,'right_knee_joint':.669,
               'left_ankle_pitch_joint':-.363,'right_ankle_pitch_joint':-.363,
               'left_shoulder_pitch_joint':.2,'right_shoulder_pitch_joint':.2,
               'left_shoulder_roll_joint':.2,'right_shoulder_roll_joint':-.2,
               'left_elbow_joint':.6,'right_elbow_joint':.6}
        for name,value in known.items():
            np.testing.assert_allclose(motion['joint_pos'][:,mapping.joint_names.index(name)],value,rtol=0,atol=3e-8)
        pelvis=mapping.body_names.index('pelvis')
        np.testing.assert_array_equal(motion['body_pos_w'][:,pelvis],np.tile(np.array([0.,0.,.78],np.float32),(1201,1)))
        np.testing.assert_array_equal(motion['body_quat_w'][:,pelvis],np.tile(np.array([1.,0.,0.,0.],np.float32),(1201,1)))
        for key in ('joint_vel','body_lin_vel_w','body_ang_vel_w'):
            np.testing.assert_array_equal(motion[key],0.)
        for key in ('joint_pos','body_pos_w','body_quat_w'):
            np.testing.assert_array_equal(motion[key],np.broadcast_to(motion[key][:1],motion[key].shape))
        mirrored=mirror_motion(motion,mapping)
        np.testing.assert_array_equal(mirrored['joint_pos'],motion['joint_pos'])

    def test_appended_pair_keeps_full_official_files_and_reports_natural_window_fraction(self):
        with self.source_fixture() as root:
            official=hiking_data.prepare_hiking_dataset(None,root/'official_existing')
            before={p.name:p.read_bytes() for p in official.walk.glob('*.npz')}
            hiking_frames=len(read_motion(official.walk/'hiking_official_walk.npz')['joint_pos'])
            source_before=file_sha256(standing_data.STANDING_BUNDLE/standing_data.STANDING_CSV_NAME)
            paths=standing_data.prepare_hiking_standing_dataset(None,root/'with_standing')
            manifest=standing_data.verify_hiking_standing_dataset(paths.root)
            self.assertEqual(manifest['standing_expert_source'],'legacy_g1_locked_v1')
            self.assertEqual(manifest['walking_total_clips'],4)
            self.assertEqual(manifest['hiking_retained_frames'],hiking_frames)
            self.assertEqual(manifest['standing_original_frames'],1201)
            self.assertEqual(manifest['walking_original_frames'],hiking_frames+1201)
            self.assertEqual(manifest['walking_total_frames'],2*(hiking_frames+1201))
            self.assertEqual(manifest['standing_history_windows'],2*1192)
            self.assertEqual(manifest['hiking_history_windows'],2*(hiking_frames-9))
            self.assertAlmostEqual(manifest['natural_standing_expert_fraction'],1192/(1192+hiking_frames-9))
            self.assertAlmostEqual(manifest['walking_original_transition_seconds'],(hiking_frames-1)/50+24.)
            self.assertAlmostEqual(manifest['official_walking_original_transition_seconds'],(hiking_frames-1)/50)
            self.assertAlmostEqual(manifest['walking_original_frame_count_seconds'],(hiking_frames+1201)/50)
            self.assertEqual(manifest['turn_statistics']['frames'],2*(hiking_frames+1201))
            self.assertEqual(manifest['turn_statistics']['clips'],4)
            self.assertIsNone(manifest['filters'])
            self.assertFalse(paths.recovery.exists())
            for name,content in before.items():
                self.assertEqual((paths.walk/name).read_bytes(),content)
                self.assertEqual((official.walk/name).read_bytes(),content)
            self.assertEqual(file_sha256(standing_data.STANDING_BUNDLE/standing_data.STANDING_CSV_NAME),source_before)
            self.assertEqual(standing_data.prepare_hiking_standing_dataset(None,paths.root),paths)
            standing=read_motion(paths.walk/standing_data.STANDING_OUTPUT_NAME)
            self.assertEqual(len(standing['joint_pos']),1201)

    def test_standing_is_a_new_cache_protocol_and_never_repairs_old_or_corrupt_files(self):
        with self.source_fixture() as root:
            old=hiking_data.prepare_hiking_dataset(None,root/'old')
            original_manifest=old.manifest.read_bytes()
            with self.assertRaisesRegex(ValueError,'walking plus legacy'):
                standing_data.prepare_hiking_standing_dataset(None,old.root)
            self.assertEqual(old.manifest.read_bytes(),original_manifest)
            paths=standing_data.prepare_hiking_standing_dataset(None,root/'new')
            p=paths.walk/standing_data.STANDING_OUTPUT_NAME
            p.write_bytes(p.read_bytes()+b'changed')
            with self.assertRaisesRegex(ValueError,'modified'):
                standing_data.verify_hiking_standing_dataset(paths.root)
            self.assertTrue(p.read_bytes().endswith(b'changed'))

    def test_verifier_rejects_missing_official_provenance_and_extra_motion_files(self):
        with self.source_fixture() as root:
            paths=standing_data.prepare_hiking_standing_dataset(None,root/'new')
            original=paths.manifest.read_text()
            bad=json.loads(original);bad['official_walking_output_sha256']={}
            paths.manifest.write_text(json.dumps(bad))
            with self.assertRaisesRegex(ValueError,'exact original/mirror'):
                standing_data.verify_hiking_standing_dataset(paths.root)
            paths.manifest.write_text(original)
            shutil.copyfile(paths.walk/standing_data.STANDING_OUTPUT_NAME,paths.walk/'arbitrary.npz')
            with self.assertRaisesRegex(ValueError,'file set changed'):
                standing_data.verify_hiking_standing_dataset(paths.root)

    def test_mutated_csv_is_not_accepted_as_a_different_standing_demonstration(self):
        with tempfile.TemporaryDirectory() as tmp:
            target=Path(tmp)/standing_data.STANDING_CSV_NAME
            target.write_bytes((standing_data.STANDING_BUNDLE/standing_data.STANDING_CSV_NAME).read_bytes()+b'\n')
            with self.assertRaisesRegex(ValueError,'source was modified'):
                standing_data.convert_legacy_g1_standing(target)


if __name__=='__main__':
    unittest.main()
