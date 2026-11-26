"""Conservative walking terrain profile and historical terrain compatibility."""
from __future__ import annotations

from dataclasses import replace
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from src.tasks.amp_perceptive.env_cfg import g1_amp_perceptive_hiking_env_cfg
from src.tasks.amp_perceptive.terrain import (
    HIKING_PROPORTIONS, WALKING_EXCLUDED_FAMILIES, WALKING_TERRAIN_PROPORTIONS,
    HikingTerrainCfg, hiking_height_array, hiking_terrain_generator,
)


class WalkingTerrainProfileTest(unittest.TestCase):
    def test_walking_profile_has_no_jump_or_recovery_families_and_normalized_weights(self):
        cfg=hiking_terrain_generator(walking_only=True)
        self.assertNotIn("recovery_flat",cfg.sub_terrains)
        self.assertFalse(set(cfg.sub_terrains)&WALKING_EXCLUDED_FAMILIES)
        self.assertEqual(set(cfg.sub_terrains),set(WALKING_TERRAIN_PROPORTIONS))
        self.assertEqual(len(cfg.sub_terrains),7)
        self.assertAlmostEqual(sum(terrain.proportion for terrain in cfg.sub_terrains.values()),1.)
        remaining_weight=sum(weight for name,weight in HIKING_PROPORTIONS.items()
                             if name not in WALKING_EXCLUDED_FAMILIES)
        for name,terrain in cfg.sub_terrains.items():
            self.assertAlmostEqual(terrain.proportion,HIKING_PROPORTIONS[name]/remaining_weight)
            self.assertLessEqual(terrain.maximum_difficulty,.4)
        task=g1_amp_perceptive_hiking_env_cfg()
        self.assertEqual(set(task.scene.terrain.terrain_generator.sub_terrains),set(cfg.sub_terrains))

    def test_curriculum_difficulty_maps_into_the_capped_source_range(self):
        cfg=hiking_terrain_generator(walking_only=True)
        for name in ("pyramid_stairs","boxes","mesh_boxes","hf_pyramid_slope_inv"):
            # TerrainGenerator applies its 8m tile size to each sub-config.
            terrain=replace(cfg.sub_terrains[name],size=cfg.size)
            spec=mujoco.MjSpec();spec.worldbody.add_body(name="terrain")
            with patch("src.tasks.amp_perceptive.terrain.hiking_height_array",wraps=hiking_height_array) as source:
                terrain.function(1.,spec,np.random.default_rng(31))
                source_difficulty=source.call_args.args[1]
            self.assertAlmostEqual(source_difficulty,.2 if name=="mesh_boxes" else .4)
            model=spec.compile()
            self.assertTrue(np.isfinite(model.hfield_data).all())
            self.assertEqual(model.hfield_nrow.tolist(),[81])

    def test_explicit_historical_profile_keeps_original_weights_and_difficulty(self):
        cfg=hiking_terrain_generator(walking_only=False)
        self.assertAlmostEqual(cfg.sub_terrains["recovery_flat"].proportion,.25)
        self.assertEqual(set(cfg.sub_terrains),set(HIKING_PROPORTIONS)|{"recovery_flat"})
        for name,weight in HIKING_PROPORTIONS.items():
            self.assertAlmostEqual(cfg.sub_terrains[name].proportion,.75*weight)
            self.assertEqual(cfg.sub_terrains[name].maximum_difficulty,1.)
        explicit=g1_amp_perceptive_hiking_env_cfg(motion_scope="joint_recovery_v1")
        self.assertEqual(set(explicit.scene.terrain.terrain_generator.sub_terrains),set(cfg.sub_terrains))

    def test_walking_cannot_reintroduce_recovery_and_caps_must_be_valid(self):
        with self.assertRaisesRegex(ValueError,"recovery-flat"):
            hiking_terrain_generator(walking_only=True,recovery_flat_fraction=.25)
        for value in (0.,-1.,1.01,float("nan"),float("inf")):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,"maximum_difficulty"):
                HikingTerrainCfg(maximum_difficulty=value).function(1.,None,None)
        for value in (-.1,1.1,float("nan")):
            with self.subTest(difficulty=value),self.assertRaisesRegex(ValueError,"difficulty"):
                HikingTerrainCfg(maximum_difficulty=.4).function(value,None,None)


if __name__=="__main__":unittest.main()
