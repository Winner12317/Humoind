"""Full G1 material conditions retain their identities through WM and Replay."""

from __future__ import annotations

import copy
from dataclasses import asdict
from types import SimpleNamespace as NS
import unittest

import torch

from scripts.train_wmac_perceptive import parse_config
from src.wmac.state import StateAdapter
from src.wmac_perceptive.checkpoints import architecture_metadata, validate_checkpoint
from src.wmac_perceptive.compact_gpu_replay import CompressedLatentReplay
from src.wmac_perceptive.config import FORMAT, config_from_payload, perception_spec_from_config
from src.wmac_perceptive.deduplicated_replay import DeduplicatedHikingReplay
from src.wmac_perceptive.hiking_amp import configure_replay
from src.wmac_perceptive.hiking_context import advance_hiking_context
from src.wmac_perceptive.hiking_materials import (
    configure_hiking_materials, hiking_material_geom_ids, hiking_material_geom_names,
    randomize_hiking_material,
)
from src.wmac_perceptive.state import PerceptiveStateAdapter
from test_deduplicated_replay import SPEC, Stream


class Scene(dict):
    pass


class MaterialConditioningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from mjlab.entity import Entity
        from src.tasks.amp_loco.config.g1.env_cfgs import g1_amp_flat_aligned_env_cfg
        cfg = configure_hiking_materials(g1_amp_flat_aligned_env_cfg())
        cls.model = Entity(cfg.scene.entities["robot"]).spec.compile()

    def make_adapter(self):
        model = self.model
        geom_ids = torch.arange(model.ngeom)
        robot = NS(actuators=[], indexing=NS(geom_ids=geom_ids),
                   data=NS(encoder_bias=torch.zeros(2, 29), joint_pos=torch.zeros(2, 29)))
        sensor = NS(_slots=[NS(primary_name=name) for name in
                    ("left_ankle_roll_link", "right_ankle_roll_link", "torso_link")],
                    data=NS(current_air_time=torch.zeros(2, 3),
                            current_contact_time=torch.zeros(2, 3)))
        scene = Scene(robot=robot, hiking_contact=sensor)
        scene.env_origins = torch.zeros(2, 3)
        friction = torch.as_tensor(model.geom_friction.copy(), dtype=torch.float32)[None].repeat(2, 1, 1)
        body_ipos = torch.as_tensor(model.body_ipos.copy(), dtype=torch.float32)[None].repeat(2, 1, 1)
        env = NS(num_envs=2, device="cpu", common_step_counter=0, scene=scene,
                 sim=NS(mj_model=model, model=NS(geom_friction=friction, body_ipos=body_ipos)),
                 termination_manager=None, action_manager=NS(action=torch.zeros(2, 29)))
        adapter = object.__new__(PerceptiveStateAdapter)
        adapter.env, adapter.robot, adapter.device = env, robot, torch.device("cpu")
        adapter.full_hiking, adapter.hiking_training_version = True, 2
        adapter.material_geom_ids = hiking_material_geom_ids(robot, model, "cpu")
        adapter.context_dim, adapter.camera_capture_steps = 133, 0
        adapter.friction_geom_ids = [i for i in range(model.ngeom)
            if "foot" in model.geom(i).name and "collision" in model.geom(i).name]
        adapter.torso_gid = next(i for i in range(model.nbody)
            if model.body(i).name.split("/")[-1] == "torso_link")
        adapter.default_torso_com = body_ipos[0, adapter.torso_gid].clone()
        return adapter

    def test_actual_collision_names_have_stable_order_independent_of_scene_indices(self):
        adapter = self.make_adapter()
        names = hiking_material_geom_names()
        self.assertEqual(len(names), 33)
        self.assertEqual(names, tuple(sorted(names)))
        self.assertEqual(len(set(names)), len(names))
        self.assertIn("torso_collision", names)
        self.assertIn("left_foot1_collision", names)
        self.assertEqual(tuple(self.model.geom(int(i)).name.split("/")[-1]
                               for i in adapter.material_geom_ids), names)
        adapter.robot.indexing.geom_ids = adapter.robot.indexing.geom_ids.flip(0)
        torch.testing.assert_close(hiking_material_geom_ids(adapter.robot, self.model, "cpu"),
                                   adapter.material_geom_ids, rtol=0, atol=0)
        keep = adapter.robot.indexing.geom_ids != int(adapter.material_geom_ids[0])
        adapter.robot.indexing.geom_ids = adapter.robot.indexing.geom_ids[keep]
        with self.assertRaisesRegex(ValueError, "schema differs"):
            hiking_material_geom_ids(adapter.robot, self.model, "cpu")

    def test_swapped_feet_and_nonfoot_materials_are_distinct_with_unchanged_legacy_prefix(self):
        adapter = self.make_adapter()
        friction = adapter.env.sim.model.geom_friction
        friction[..., 0] = .8
        left = [i for i in adapter.friction_geom_ids if "left_" in self.model.geom(i).name]
        right = [i for i in adapter.friction_geom_ids if "right_" in self.model.geom(i).name]
        self.assertEqual((len(left), len(right)), (7, 7))
        friction[0, left, 0], friction[0, right, 0] = .5, 1.5
        friction[1, left, 0], friction[1, right, 0] = 1.5, .5
        full = adapter.context()
        self.assertEqual(full.shape, (2, 133))
        self.assertTrue(torch.equal(full[0, :100], full[1, :100]))
        self.assertFalse(torch.equal(full[0, 100:], full[1, 100:]))
        torch.testing.assert_close(full[:, 100:], friction[:, adapter.material_geom_ids, 0], rtol=0, atol=0)
        original_ids = adapter.material_geom_ids
        adapter.material_geom_ids = None
        legacy = adapter.context()
        self.assertEqual(legacy.shape, (2, 100))
        torch.testing.assert_close(full[:, :100], legacy, rtol=0, atol=0)
        torch.testing.assert_close(StateAdapter.context(adapter)[:, :33], legacy[:, :33], rtol=0, atol=0)
        adapter.material_geom_ids = original_ids

        # Equal feet cannot hide a changed torso/hand collision material either.
        friction[1].copy_(friction[0])
        torso = int(original_ids[hiking_material_geom_names().index("torso_collision")])
        friction[1, torso, 0] = .35
        full = adapter.context()
        self.assertTrue(torch.equal(full[0, :100], full[1, :100]))
        self.assertEqual(int((full[0, 100:] != full[1, 100:]).sum()), 1)

    def test_conditions_are_the_coefficients_actually_applied_by_randomizer(self):
        adapter = self.make_adapter()
        torch.manual_seed(20261117)
        randomize_hiking_material(adapter.env)
        expected = adapter.env.sim.model.geom_friction[:, adapter.material_geom_ids, 0]
        torch.testing.assert_close(adapter.context()[:, 100:], expected, rtol=0, atol=0)
        self.assertTrue(bool(((expected >= .3) & (expected <= 1.6)).all()))

    def test_context_recurrence_preserves_material_tail_and_legacy_recurrence(self):
        context = torch.randn(2, 3, 133)
        context[..., 100] = -0.
        action, auxiliary = torch.randn(2, 3, 29), torch.rand(2, 3, 13)
        for capture_steps in (0, 5):
            result = advance_hiking_context(context, action, auxiliary,
                                             camera_capture_steps=capture_steps)
            legacy = advance_hiking_context(context[..., :100], action, auxiliary,
                                             camera_capture_steps=capture_steps)
            torch.testing.assert_close(result[..., :100], legacy, rtol=0, atol=0)
            self.assertTrue(torch.equal(result[..., 100:].contiguous().view(torch.int32),
                                        context[..., 100:].contiguous().view(torch.int32)))


class MaterialReplayTest(unittest.TestCase):
    def make_pair(self, capacity=7):
        stream = Stream(unusual=True)
        stream.context = torch.cat((stream.context, stream.rand(stream.n, 33)), -1)
        compact = DeduplicatedHikingReplay(capacity, stream.n, "cpu", .25, spec=SPEC, context_dim=133)
        reference = CompressedLatentReplay(capacity, stream.n, "cpu", .25, spec=SPEC)
        for replay in (compact, reference):
            replay.bootstrap_processed_history(stream.images)
            configure_replay(replay)
        return compact, reference, stream

    def assert_samples_bitwise(self, compact, reference):
        for split in ("train", "validation", "all"):
            for windows in (False, True):
                torch.manual_seed(4401)
                actual = (compact.sample_windows(19, min(3, compact.size), split=split)
                          if windows else compact.sample_starts(19, split=split))
                torch.manual_seed(4401)
                expected = (reference.sample_windows(19, min(3, reference.size), split=split)
                            if windows else reference.sample_starts(19, split=split))
                self.assertEqual(actual.keys(), expected.keys())
                for name, value in actual.items():
                    other = expected[name]
                    if value.dtype == torch.float32:
                        value, other = value.contiguous().view(torch.int32), other.contiguous().view(torch.int32)
                    self.assertTrue(torch.equal(value, other), (split, windows, name))

    def test_133_context_is_bitwise_after_wrap_resets_events_and_restore(self):
        compact, reference, stream = self.make_pair()
        for step in range(26):
            if step == 16:
                compact.mark_boundary(); reference.mark_boundary(); stream.reset_all()
            if step in (3, 9, 15, 20):
                stream.context[:, 100:] = stream.rand(stream.n, 33)
            stream.context[0, 100] = -0. if step % 2 else 0.
            batch = stream.batch()
            if step == 12: batch["next_context"][2, 110] += .125  # Exact exception path.
            torch.testing.assert_close(compact.add(batch), reference.add(batch), rtol=0, atol=0)
            if step in (0, 8, 18, 25): self.assert_samples_bitwise(compact, reference)
        saved = copy.deepcopy(compact.state_dict())
        self.assertEqual(saved["context_dim"], 133)
        self.assertEqual(saved["perceptive_format"], "hiking_lossless_frame_uint8_v3")
        self.assertEqual(saved["layout_version"], 3)
        self.assertEqual(compact._context_events.last.shape[-1], 98)
        self.assertEqual(compact.data["context"].shape[-1], 35)
        restored = DeduplicatedHikingReplay(7, stream.n, "cpu", .25, spec=SPEC, context_dim=133)
        restored.load_state_dict(saved)
        reference.mark_boundary(); reference._pending_reset.fill_(True)
        self.assert_samples_bitwise(restored, reference)
        stream.reset_all()
        for _ in range(10):
            batch = stream.batch(); restored.add(batch); reference.add(batch)
        self.assert_samples_bitwise(restored, reference)

    def test_100_and_133_replay_protocols_reject_cross_loading_before_mutation(self):
        modern, _, stream = self.make_pair()
        modern.add(stream.batch())
        historical = DeduplicatedHikingReplay(7, stream.n, "cpu", .25, spec=SPEC)
        for target, source in ((historical, modern), (modern, historical)):
            before = (target.ptr, target.size, target.total_steps)
            with self.assertRaisesRegex(ValueError, "layout differs"):
                target.load_state_dict(copy.deepcopy(source.state_dict()))
            self.assertEqual((target.ptr, target.size, target.total_steps), before)


class MaterialCheckpointProtocolTest(unittest.TestCase):
    def payload(self, cfg):
        return dict(format=FORMAT, config=asdict(cfg), architecture=architecture_metadata(cfg),
                    visual_spec=asdict(perception_spec_from_config(cfg)),
                    terrain_fingerprint="fixture-terrain", motion_manifest_sha256="fixture-motion")

    def test_v2_protocol_records_every_changed_width_and_material_identity(self):
        cfg = parse_config(["--hiking-training-version", "2"])
        meta = architecture_metadata(cfg)
        self.assertEqual(cfg.hiking_training_version, 2)
        self.assertEqual(meta["architecture_version"], 8)
        self.assertEqual(meta["context_dim"], 133)
        self.assertEqual(meta["world_model"]["input_normalizer_dim"], 311)
        self.assertEqual(meta["world_model"]["input_dim"], 439)
        self.assertEqual(meta["critic"]["input_dim"], 315)
        self.assertEqual(meta["context_layout"]["applied_collision_sliding_friction"], [100, 133])
        self.assertEqual(meta["material_conditioning"]["geom_names"], list(hiking_material_geom_names()))
        self.assertFalse(meta["material_conditioning"]["actor_input"])
        self.assertEqual(meta["world_model"]["normalization"], "continuous_empirical_statistics_no_sample_limit")
        self.assertEqual(validate_checkpoint(self.payload(cfg), cfg), {})

    def test_unshod_v3_records_reference_input_protocol_and_actual_geom_names(self):
        cfg = parse_config([])
        meta = architecture_metadata(cfg)
        names = list(hiking_material_geom_names(3))
        self.assertEqual((cfg.hiking_training_version, meta["architecture_version"]), (3, 9))
        self.assertEqual(len(names), 29)
        self.assertEqual(meta["context_layout"]["applied_collision_sliding_friction"], [100, 129])
        self.assertEqual(meta["material_conditioning"]["geom_names"], names)
        self.assertEqual((meta["context_dim"], meta["world_model"]["input_normalizer_dim"],
                          meta["world_model"]["input_dim"], meta["critic"]["input_dim"]), (129, 97, 435, 281))
        self.assertEqual(validate_checkpoint(self.payload(cfg), cfg), {})

    def test_missing_version_marker_remains_historical_and_cross_resume_is_rejected(self):
        latest = parse_config([])
        historical = parse_config(["--hiking-training-version", "1"])
        payload = self.payload(historical)
        payload["config"].pop("hiking_training_version")
        restored = config_from_payload(payload)
        self.assertEqual(restored.hiking_training_version, 1)
        self.assertEqual(payload["architecture"]["architecture_version"], 7)
        self.assertEqual(payload["architecture"]["context_dim"], 100)
        self.assertEqual(validate_checkpoint(payload, restored), {})
        with self.assertRaisesRegex(ValueError, "architecture"):
            validate_checkpoint(payload, latest)
        with self.assertRaisesRegex(ValueError, "architecture"):
            validate_checkpoint(self.payload(latest), historical)


if __name__ == "__main__":
    unittest.main()
