"""Walking-only scope is enforced through configuration, gating and physics."""
from __future__ import annotations

from dataclasses import asdict,replace
import unittest

import torch

from scripts.check_wmac_perceptive_training import compare_evaluations
from src.wmac_perceptive.checkpoints import architecture_metadata,validate_checkpoint
from src.wmac_perceptive.config import PerceptiveConfig,config_from_payload
from test_perceptive_runner import runner_fixture


def walking_runner():
    runner,batch,noise=runner_fixture(horizon=2)
    runner.cfg=replace(runner.cfg,motion_scope="forward_walk_turn_v1",perception_training_version=2,
        recovery_fraction=0.,expert_recovery_fraction=0.,recovery_flat_fraction=0.,validation_min_recovery_samples=0)
    runner.cfg.validate()
    runner.adapter.motion_scope=runner.cfg.motion_scope
    runner.adapter.max_delay_steps=0
    runner.adapter.reward_cfg.pop("track_root_height",None)
    return runner,batch,noise


class LocomotionTrainingTest(unittest.TestCase):
    def test_new_defaults_cannot_silently_reenable_recovery(self):
        cfg=PerceptiveConfig.defaults();cfg.validate()
        self.assertEqual(cfg.motion_scope,"forward_walk_turn_v1")
        for key in ("recovery_fraction","expert_recovery_fraction","recovery_flat_fraction","validation_min_recovery_samples"):
            self.assertEqual(getattr(cfg,key),0)
            with self.subTest(key=key),self.assertRaises(ValueError):replace(cfg,**{key:1}).validate()

    def test_old_scope_is_preserved_and_cannot_resume_as_new_motion_objective(self):
        old=replace(PerceptiveConfig.defaults(),motion_scope="joint_recovery_v1",recovery_fraction=.5,
            expert_recovery_fraction=.5,recovery_flat_fraction=.25,validation_min_recovery_samples=8)
        saved=asdict(old);saved.pop("motion_scope")
        payload={"format":"amp-wmac-perceptive-v1","config":saved,"architecture":architecture_metadata(old),
                 "terrain_fingerprint":"old-terrain","motion_manifest_sha256":"old-motion"}
        restored=config_from_payload(payload)
        self.assertEqual(restored.motion_scope,"joint_recovery_v1")
        self.assertEqual(validate_checkpoint(payload,restored),{})
        with self.assertRaises(ValueError):validate_checkpoint(payload,PerceptiveConfig.defaults())

    def test_gate_requires_only_the_current_walking_domain(self):
        runner,_,_=walking_runner()
        runner.validation={"validation_walk_h1_samples":16.,"validation_walk_h1_depth_samples":16.,
            "validation_walk_h1_finite_fraction":1.,"validation_walk_h1_model_error":.1,
            "validation_walk_h1_contact_error":.1,"validation_walk_h1_actor_depth_rmse":.4,
            "validation_walk_h1_actor_mask_error":.2}
        row={"wm_h1_model_error":.1,"wm_updates_applied_fraction":1.}
        self.assertTrue(runner._actor_gate(row))
        self.assertEqual(runner._motion_phases(),("walk",))
        self.assertFalse(any("recovery" in key for key in runner._empty_validation()))
        runner.validation["validation_walk_h1_samples"]=0.
        self.assertFalse(runner._actor_gate(row))

    def test_falling_is_not_a_recovery_sampling_domain(self):
        runner,batch,_=walking_runner()
        batch["relative_height"].fill_(.1);batch["context"][:,33:].fill_(99.)
        self.assertFalse(bool(runner._recovery_mask(batch).any()))

    def test_imagined_fall_is_immediate_and_recovery_context_is_zero(self):
        runner,batch,_=walking_runner()
        state=batch["state"].clone();state[:,2]=.2
        context=batch["context"].clone();context[:,33:]=99.
        following,terminated=runner.adapter.advance_context(state,context)
        torch.testing.assert_close(following[:,33:],torch.zeros_like(following[:,33:]),rtol=0,atol=0)
        torch.testing.assert_close(terminated,torch.ones_like(terminated),rtol=0,atol=0)
        state[:,2]=.79
        _,terminated=runner.adapter.advance_context(state,context)
        torch.testing.assert_close(terminated,torch.zeros_like(terminated),rtol=0,atol=0)

    def test_walking_reward_no_longer_requires_the_recovery_height_term(self):
        runner,batch,noise=walking_runner()
        result=runner.imagine(batch,noises=noise)
        self.assertTrue(bool(torch.isfinite(result["objective"]).all()))

    def test_physical_review_accepts_walk_only_results(self):
        baseline={"status":"completed","protocol":{"motion_scope":"forward_walk_turn_v1"},
            "results":{"walk":{"initial_state_sha256":"same","first_episode_survival_fraction":.25}}}
        candidate={**baseline,"results":{"walk":{**baseline["results"]["walk"],"first_episode_survival_fraction":.5}}}
        result=compare_evaluations(candidate,baseline)
        self.assertTrue(result["comparable"])
        self.assertEqual(result["primary_fraction_deltas"],{"walk_survival":.25})


if __name__=="__main__":unittest.main()
