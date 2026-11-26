from types import SimpleNamespace
import unittest

import numpy as np
import torch

from src.wmac_perceptive.hiking_materials import (
    configure_hiking_materials,hiking_material_protocol_metadata,randomize_hiking_material,sample_hiking_material_buckets,
)


class HikingMaterialsTest(unittest.TestCase):
    def test_configuration_enables_all_body_sliding_friction_without_mutating_legacy(self):
        from mjlab.entity import Entity
        from src.tasks.amp_loco.config.g1.env_cfgs import g1_amp_flat_aligned_env_cfg
        cfg=g1_amp_flat_aligned_env_cfg()
        previous=cfg.scene.entities["robot"].collisions
        configure_hiking_materials(cfg)
        self.assertIsNot(previous,cfg.scene.entities["robot"].collisions)
        self.assertIsInstance(previous[0].condim,dict)
        model=Entity(cfg.scene.entities["robot"]).spec.compile()
        collisions=(model.geom_contype|model.geom_conaffinity)!=0
        self.assertTrue(bool((model.geom_condim[collisions]==3).all()))
        self.assertTrue(bool((model.geom_priority[collisions]==1).all()))

    def test_buckets_follow_ranges_consistency_and_explicit_backend_boundary(self):
        buckets=sample_hiking_material_buckets()
        self.assertEqual(buckets.shape,(64,3))
        self.assertTrue(bool(((buckets[:,:2]>=.3)&(buckets[:,:2]<=1.6)).all()))
        self.assertTrue(bool(((buckets[:,2]>=.05)&(buckets[:,2]<=.5)).all()))
        self.assertTrue(bool((buckets[:,1]<=buckets[:,0]).all()))
        metadata=hiking_material_protocol_metadata()
        self.assertFalse(metadata["restitution_applied"])
        self.assertFalse(metadata["separate_static_friction_applied"])
        self.assertIsNone(metadata["solref_mapping"])

    def test_all_collision_shapes_receive_buckets_and_other_parameters_are_preserved(self):
        friction=torch.full((5,4,3),.6)
        solref=torch.tensor([.02,1.]).repeat(5,4,1)
        model=SimpleNamespace(geom_contype=np.array([1,0,1,0]),geom_conaffinity=np.zeros(4,dtype=int))
        robot=SimpleNamespace(indexing=SimpleNamespace(geom_ids=torch.arange(4)))
        env=SimpleNamespace(num_envs=5,device="cpu",scene={"robot":robot},
            sim=SimpleNamespace(mj_model=model,model=SimpleNamespace(geom_friction=friction,geom_solref=solref)))
        randomize_hiking_material(env)
        expected=env._hiking_material_buckets[env._hiking_material_assignments][...,1]
        torch.testing.assert_close(friction[:,[0,2],0],expected)
        torch.testing.assert_close(friction[:,[1,3]],torch.full((5,2,3),.6))
        torch.testing.assert_close(friction[:,:,1:],torch.full((5,4,2),.6))
        torch.testing.assert_close(solref,torch.tensor([.02,1.]).repeat(5,4,1))


if __name__=="__main__":unittest.main()
