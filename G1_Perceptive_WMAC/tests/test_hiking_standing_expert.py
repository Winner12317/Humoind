"""Uniform mixed expert loading and read-only reconstruction of both sources."""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import torch

from scripts import audit_wmac_motion_data as audit_cli
from src.wmac_perceptive.config import LEGACY_WALKING_EXPERT
from src.wmac_perceptive.expert import PerceptiveExpertPairs
from src.wmac_perceptive.hiking_data import prepare_hiking_dataset
from src.wmac_perceptive.mirroring import DEFAULT_G1_XML,build_g1_mirror_map
from src.wmac_perceptive.motions import file_sha256
from src.wmac_perceptive.standing_data import (
    STANDING_BUNDLE,STANDING_EXPERT_SOURCE,prepare_hiking_standing_dataset,
)


class HikingStandingExpertTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temp.name)
        cls.stack=contextlib.ExitStack()
        cls.mapping=build_g1_mirror_map()
        cls.source=cls.root/"source";cls.source.mkdir()
        cls.raw=cls.source/"parkour_motion_without_run_retargetted.npz"
        count=21
        joints=np.zeros((count,29),np.float32)
        joints[:,0]=np.linspace(-.02,.02,count)
        root_position=np.zeros((count,3),np.float32)
        root_position[:,0]=np.arange(count)*.005;root_position[:,2]=.9
        quaternion=np.tile(np.array([1.,0.,0.,0.],np.float32),(count,1))
        np.savez_compressed(cls.raw,framerate=np.array(50.),joint_names=np.array(cls.mapping.joint_names),
            joint_pos=joints,base_pos_w=root_position,base_quat_w=quaternion)
        cls.raw.with_name("parkour_motion_without_run.yaml").write_text("selected_files:\n- "+cls.raw.name+"\n")
        cls.raw.with_name("README.md").write_text("Small synthetic walking source for CPU tests.\n")
        digest=file_sha256(cls.raw)
        cls.stack.enter_context(patch("src.wmac_perceptive.hiking_data.ensure_hiking_source",return_value=cls.raw))
        cls.stack.enter_context(patch("src.wmac_perceptive.hiking_data.HIKING_MOTION_SHA256",digest))
        cls.stack.enter_context(patch("src.wmac_perceptive.standing_data.HIKING_MOTION_SHA256",digest))
        cls.stack.enter_context(patch("src.wmac_perceptive.locomotion_data.HIKING_SOURCE_SHA256",
            {path.name:file_sha256(path) for path in cls.source.iterdir()}))
        with contextlib.redirect_stdout(io.StringIO()):
            cls.mixed=prepare_hiking_standing_dataset(None,
                cls.root/".perceptive_cache/motions/g1_hiking_walk_stand_v1")
            cls.official=prepare_hiking_dataset(None,cls.root/".perceptive_cache/motions/g1_hiking_walk_v1")
        model=mujoco.MjModel.from_xml_path(str(DEFAULT_G1_XML))
        cls.env=SimpleNamespace(step_dt=.02,scene={"robot":SimpleNamespace(
            body_names=tuple(cls.mapping.body_names),joint_names=tuple(cls.mapping.joint_names),data=SimpleNamespace(
                default_joint_pos=torch.tensor(model.qpos0[7:],dtype=torch.float32)[None],
                default_joint_vel=torch.zeros(1,29)))})

    @classmethod
    def tearDownClass(cls):
        cls.stack.close();cls.temp.cleanup()

    def config(self,source="none"):
        return SimpleNamespace(amp_feature_schema="hiking_v1",motion_scope="forward_walk_turn_v1",
                               standing_expert_source=source)

    def report(self):
        return dict(issues=[],checks={},files=[],source_files=[],source_reconstruction=[],remote_pinned_urls=[])

    def args(self,root):
        return SimpleNamespace(dataset_root=root,robot_xml=DEFAULT_G1_XML,hiking_source=self.source,
                               training_config=None,check_remote=False)

    def test_mixed_loader_reports_actual_window_share_and_keeps_every_source_clip(self):
        expert=PerceptiveExpertPairs(self.config(STANDING_EXPERT_SOURCE),self.env,"cpu",self.mixed.root)
        clips=expert.metadata["groups"][0]["clips"]
        self.assertEqual(len(clips),4)
        self.assertEqual([row["reference_kind"] for row in clips].count("standing"),2)
        self.assertEqual([row["reference_kind"] for row in clips].count("walking"),2)
        standing=sum(row["history_windows"] for row in clips if row["reference_kind"]=="standing")
        total=sum(row["history_windows"] for row in clips)
        self.assertEqual(standing,2*(1201-9))
        self.assertEqual(expert.metadata["standing_history_window_count"],standing)
        self.assertEqual(expert.metadata["standing_expert_window_fraction"],standing/total)
        self.assertEqual(expert.metadata["total_history_window_count"],total)
        self.assertEqual(expert.groups[0].shape,(sum(row["frames"] for row in clips),67))
        for row in clips:
            self.assertEqual(row["sha256"],file_sha256(self.mixed.root/row["file"]))
            self.assertTrue(row["source_description"])
        # Enumerate the sampler's complete uniform index domain. Every legal
        # end appears once and none of its ten-frame windows crosses a clip.
        with patch("torch.randint",return_value=torch.arange(total)) as draw:
            histories=expert.sample_histories(total)
        draw.assert_called_once_with(total,(total,),device="cpu")
        self.assertEqual(histories.shape,(total,10,67))
        frame_offset=window_offset=0
        for row in clips:
            count,windows=row["frames"],row["history_windows"]
            expected=expert.groups[0][frame_offset:frame_offset+count].unfold(0,10,1).transpose(-1,-2)
            torch.testing.assert_close(histories[window_offset:window_offset+windows],expected,rtol=0,atol=0)
            frame_offset+=count;window_offset+=windows

    def test_old_loader_keeps_two_official_clips_and_rejects_wrong_cache_protocol(self):
        old=PerceptiveExpertPairs(self.config(),self.env,"cpu",self.official.root)
        self.assertEqual(old.metadata["groups"][0]["clip_count"],2)
        self.assertNotIn("standing_expert_window_fraction",old.metadata)
        with self.assertRaisesRegex(ValueError,"complete official Hiking"):
            PerceptiveExpertPairs(self.config(),self.env,"cpu",self.mixed.root)
        with self.assertRaisesRegex(ValueError,"Hiking walking plus legacy"):
            PerceptiveExpertPairs(self.config(STANDING_EXPERT_SOURCE),self.env,"cpu",self.official.root)

    def test_episode_teacher_uses_official_torso_labels_and_original_mirror(self):
        cfg=self.config()
        cfg.hiking_training_version=3
        with patch("src.wmac_perceptive.hiking_data.verify_hiking_dataset",side_effect=AssertionError("Relied on converted cache")), \
             patch("src.wmac_perceptive.expert.read_motion", side_effect=AssertionError("Loaded derived expert labels")):
            expert=PerceptiveExpertPairs(cfg,self.env,"cpu",self.source)
        self.assertEqual(expert.official_reference_frames,20)
        self.assertEqual(expert.metadata["standing_expert_source"],"none")
        normal,mirror=expert.groups[0].split(20)
        expected=torch.tensor([.25,0.,0.]).expand(19,3)
        torch.testing.assert_close(normal[:-1,-6:-3],expected)
        torch.testing.assert_close(mirror[:-1,-6:-3],expected)
        torch.testing.assert_close(normal[-1,-6:],torch.zeros(6))
        torch.testing.assert_close(normal[:,3],mirror[:,9])

    def test_legacy_local_walking_reference_loads_bundled_clips_without_network(self):
        cfg = self.config()
        cfg.hiking_training_version = 3
        official = PerceptiveExpertPairs(cfg, self.env, "cpu", self.source)
        cfg.walking_expert_source = LEGACY_WALKING_EXPERT
        with patch("urllib.request.urlopen", side_effect=AssertionError("Local walking accessed the network")):
            source = Path(__file__).resolve().parents[1] / "src/assets/motions/g1/amp/WalkandRun"
            if not any(source.glob("*.npz")):
                with self.assertRaisesRegex(ValueError, "17 licensed WalkandRun"):
                    PerceptiveExpertPairs(cfg, self.env, "cpu", self.source)
                return
            expert = PerceptiveExpertPairs(cfg, self.env, "cpu", self.source)

        torch.testing.assert_close(expert.groups[0][:len(official.groups[0])], official.groups[0], rtol=0., atol=0.)
        self.assertEqual(official.legacy_walking_clips, ())
        meta = expert.metadata["legacy_walking_reference"]
        self.assertEqual((meta["source_clips"], meta["source_frames"]), (13, 389))
        self.assertEqual((meta["training_clips"], meta["training_frames"]), (26, 778))
        self.assertEqual(meta["history_window_count"], 544)
        self.assertEqual(expert.legacy_walking_sha256, meta["output_sha256"])
        for (original, mirror), clip in zip(expert.legacy_walking_clips, meta["clips"], strict=True):
            self.assertEqual(clip["source"], "local_supplement")
            self.assertEqual(tuple(original.shape), (clip["frames"], 67))
            self.assertEqual(original.shape, mirror.shape)
            self.assertTrue(bool(torch.isfinite(original).all()))
            self.assertTrue(bool(torch.isfinite(mirror).all()))
        self.assertEqual(expert.metadata["official_history_window_count"], 22)
        total = expert.metadata["total_history_window_count"]
        self.assertEqual(total, 566)
        self.assertEqual(expert.metadata["legacy_walking_window_fraction"], 544 / 566)
        group = expert.metadata["groups"][0]
        self.assertEqual(group["clip_count"], 28)
        self.assertEqual(group["history_window_count"], total)
        with patch("torch.randint", return_value=torch.arange(total)) as draw:
            histories = expert.sample_histories(total)
        draw.assert_called_once_with(total, (total,), device="cpu")
        frame_offset = window_offset = 0
        for entry in group["clips"]:
            frames, windows = entry["frames"], entry["history_windows"]
            clip = expert.groups[0][frame_offset:frame_offset + frames]
            expected = clip.unfold(0, 10, 1).transpose(-1, -2)
            torch.testing.assert_close(histories[window_offset:window_offset + windows], expected, rtol=0., atol=0.)
            frame_offset += frames
            window_offset += windows
        self.assertEqual(frame_offset, len(expert.groups[0]))
        self.assertEqual(window_offset, total)

    def test_audit_reconstructs_four_clips_and_does_not_modify_sources_or_cache(self):
        paths=[path for parent in (self.mixed.root,self.source) for path in parent.rglob("*") if path.is_file()]
        before={path:file_sha256(path) for path in paths}
        report=self.report()
        with patch("urllib.request.urlopen",side_effect=AssertionError("offline audit accessed network")):
            audit_cli.audit_official_hiking(self.args(self.mixed.root),report,include_standing=True)
        self.assertEqual(report["issues"],[])
        self.assertEqual(report["summary"]["training_clips"],4)
        self.assertEqual(report["summary"]["hiking_loaded_frames"],20)
        self.assertEqual(report["summary"]["standing_training_frames"],2402)
        self.assertIsNone(report["summary"]["standing_source_fps"])
        self.assertEqual(report["summary"]["standing_representation_transition_seconds"],24.)
        self.assertTrue(report["checks"]["official_walking_npz_bytes_preserved"]["passed"])
        self.assertTrue(all(row["exact_reconstruction"] for row in report["files"]))
        self.assertEqual({path:file_sha256(path) for path in paths},before)
        old_report=self.report()
        audit_cli.audit_official_hiking(self.args(self.official.root),old_report)
        self.assertEqual(old_report["issues"],[])
        self.assertEqual(old_report["summary"]["training_clips"],2)

    def test_default_audit_dispatch_matches_new_training_cache(self):
        output=self.root/"report.json"
        with patch.object(audit_cli,"ROOT",self.root),contextlib.redirect_stdout(io.StringIO()):
            code=audit_cli.main(["--hiking-source",str(self.source),"--robot-xml",str(DEFAULT_G1_XML),
                                 "--output",str(output)])
        self.assertEqual(code,0)
        report=json.loads(output.read_text())
        self.assertEqual(report["dataset_root"],str(self.source))
        self.assertEqual(report["summary"]["source_clips"],1)
        self.assertEqual(report["status"],"pass")

    def test_audit_output_cannot_overwrite_its_bundled_standing_source(self):
        with contextlib.redirect_stderr(io.StringIO()),self.assertRaises(SystemExit) as failure:
            audit_cli.main(["--output",str(STANDING_BUNDLE/"SOURCE_MANIFEST.json")])
        self.assertEqual(failure.exception.code,2)


if __name__=="__main__":unittest.main()
