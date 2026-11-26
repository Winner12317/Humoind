"""Checkpoint preflight rejects incompatible terrain/sensor state atomically."""
from __future__ import annotations

import copy
from dataclasses import asdict, replace
import unittest

import torch
from torch import nn

from src.wmac_perceptive.checkpoints import (
    architecture_metadata, validate_checkpoint, validate_module_states,
)
from src.wmac_perceptive.config import FORMAT, PerceptiveConfig
from src.wmac_perceptive.models import PerceptiveActor
from src.wmac_perceptive.perception import PerceptionSpec


def payload(cfg):
    return {"format": FORMAT, "config": asdict(cfg), "architecture": architecture_metadata(cfg),
            "terrain_fingerprint": {"mesh_sha256": "terrain-a", "height_semantics": cfg.ground_height_semantics},
            "motion_manifest_sha256": "motion-a"}


class PerceptiveCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.cfg = PerceptiveConfig.defaults()
        self.good = payload(self.cfg)

    def validate(self, record, cfg=None):
        return validate_checkpoint(record, self.cfg if cfg is None else cfg,
            terrain_fingerprint=self.good["terrain_fingerprint"], motion_fingerprint="motion-a")

    def test_metadata_covers_actual_actor_sensor_and_grounding_interfaces(self):
        metadata = architecture_metadata(self.cfg)
        spec = PerceptionSpec()
        self.assertEqual(metadata["format"], FORMAT)
        # Version 1 records predate include_valid. Keep their metadata stable,
        # while reconstructing the explicit legacy spec with its true default.
        self.assertEqual(asdict(PerceptionSpec(**metadata["perception"])), asdict(spec))
        self.assertEqual(metadata["actor"]["moe_input_dim"], 512)
        self.assertEqual(metadata["actor"]["gate_hidden"], [256])
        self.assertEqual(metadata["actor"]["expert_hidden"], [256, 128, 64])
        self.assertEqual(metadata["actor"]["packed_input_dim"], spec.packed_dim)
        self.assertEqual(metadata["ground_height_semantics"], "per_point_terrain_v1")
        self.assertEqual(metadata["amp"]["frame_dim"], 203)
        actor = PerceptiveActor(spec)
        self.assertEqual(actor.policy.router[0].in_features, metadata["actor"]["moe_input_dim"])
        self.assertEqual(actor.policy.router[0].out_features, metadata["actor"]["gate_hidden"][0])
        self.assertEqual(actor.vision.conv[0].in_channels, metadata["actor"]["depth_encoder"]["input_channels"])
        self.assertEqual(self.validate(self.good), {})

    def test_flat_ppo_and_missing_provenance_are_never_inferred(self):
        for change in ({"format": "amp-wmac-v1"}, {"format": "ppo"}, {"format": None}):
            bad = {**self.good, **change}
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, "Perceptive resume"):
                self.validate(bad)
        for key in ("architecture", "terrain_fingerprint", "motion_manifest_sha256", "config"):
            bad = copy.deepcopy(self.good); bad.pop(key)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.validate(bad)

    def test_visual_history_crop_and_model_architecture_changes_are_rejected(self):
        for path, value in ((("perception", "stride"), 3), (("perception", "delay"), 0),
                            (("perception", "height"), 18), (("perception", "max_depth"), 5.),
                            (("actor", "gate_hidden"), []), (("actor", "visual_embedding_dim"), 64),
                            (("world_model", "hidden"), [32])):
            bad = copy.deepcopy(self.good); bad["architecture"][path[0]][path[1]] = value
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "architecture"):
                self.validate(bad)
        # New sensor-preprocessing fields are included automatically, rather
        # than silently omitted by a hand-maintained metadata field subset.
        if "crop_top" in self.good["architecture"]["perception"]:
            bad = copy.deepcopy(self.good); bad["architecture"]["perception"]["crop_top"] += 1
            with self.assertRaisesRegex(ValueError, "architecture"):
                self.validate(bad)

    def test_seed_utd_rewards_and_sampling_budgets_are_strict_but_output_paths_are_mutable(self):
        for key, value in (("seed", self.cfg.seed+1), ("utd", .5),
                           ("actor_batch_size", self.cfg.actor_batch_size+1),
                           ("style_weight", self.cfg.style_weight+.01),
                           ("terrain_rows", self.cfg.terrain_rows+1)):
            bad = copy.deepcopy(self.good); bad["config"][key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "strict training"):
                self.validate(bad)
        changed = replace(self.cfg, iterations=self.cfg.iterations+100,
                          log_root="other/logs", run_name="resumed", resume="saved.pt",
                          device="cpu", mirrored_motion_root="other/cache")
        self.assertEqual(self.validate(self.good, changed), {})

    def test_current_terrain_and_motion_fingerprints_must_match(self):
        for key, value in (("terrain_fingerprint", {"mesh_sha256": "other-terrain"}),
                           ("motion_manifest_sha256", "other-motions")):
            bad = copy.deepcopy(self.good); bad[key] = value
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                self.validate(bad)

    def test_full_module_preflight_never_partially_loads_earlier_modules(self):
        modules = {"actor": nn.Sequential(nn.Linear(3, 4), nn.ELU(), nn.Linear(4, 2)),
                   "world_model": nn.Linear(5, 6)}
        initial = {name: copy.deepcopy(module.state_dict()) for name, module in modules.items()}
        good = {name: {key: value.clone()+1 for key, value in state.items()} for name, state in initial.items()}
        self.assertIsNone(validate_module_states(good, modules))
        for problem in ("missing", "shape", "dtype", "nan", "nontensor"):
            bad = copy.deepcopy(good)
            if problem == "missing": bad["world_model"].pop("bias")
            elif problem == "shape": bad["world_model"]["weight"] = torch.zeros(2, 5)
            elif problem == "dtype": bad["world_model"]["weight"] = bad["world_model"]["weight"].double()
            elif problem == "nan": bad["world_model"]["weight"][0, 0] = float("nan")
            else: bad["world_model"]["bias"] = [1.]*6
            with self.subTest(problem=problem), self.assertRaises(ValueError):
                validate_module_states(bad, modules)
            for name, module in modules.items():
                for key, value in module.state_dict().items():
                    torch.testing.assert_close(value, initial[name][key], rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
