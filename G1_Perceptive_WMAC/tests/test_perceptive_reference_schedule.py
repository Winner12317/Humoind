"""Reference scheduling must not inherit the old policy veto or rollback."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from src.wmac_perceptive.optimizers import build_reference_optimizer
from src.wmac_perceptive.reference_training import build_reference_critic,reference_value_ceiling
from test_perceptive_fast_runner import fast_runner_fixture
from test_perceptive_runner import critic_noop
from test_wmac_optimization import fix_update_noise


def reference_fixture():
    runner,batch,noise=fast_runner_fixture(compiled=False)
    runner.cfg.training_schedule="supermimic_v1"
    runner.cfg.execution_optimizer="muon_adam"
    runner.cfg.actor_microbatch_size=0
    runner.cfg.model_warmup_iterations=runner.cfg.critic_warmup_iterations=0
    runner.cfg.imagination_horizon_start=runner.cfg.imagination_horizon
    runner.critic=build_reference_critic(runner.critic.norm.mean.numel(),(16,8)).double()
    runner._value_fn=runner.critic
    runner._critic_value_ceiling=reference_value_ceiling(runner.adapter,runner.cfg)
    runner.opt_actor=build_reference_optimizer([runner.actor.policy],runner.cfg.actor_lr,
        betas=runner.cfg.actor_betas)
    runner._setup_fast_execution()
    fix_update_noise(runner,noise)
    return runner,batch


class ReferenceScheduleTest(unittest.TestCase):
    def test_physical_visual_errors_and_growth_counter_do_not_veto_or_shorten_policy_update(self):
        runner,_=reference_fixture()
        runner.validation={"validation_model_error":float("inf")}
        row={"wm_loss":float("inf"),"wm_gate_error":float("inf"),
             "wm_updates_applied_fraction":0.,"wm_h1_model_error":float("inf")}
        self.assertEqual(runner.actor_updates_accepted,0)
        self.assertTrue(runner._actor_gate(row))
        self.assertEqual(runner._effective_horizon(),runner.cfg.imagination_horizon)
        self.assertEqual(row["gate_reason"],"supermimic_schedule")
        runner.replay=SimpleNamespace(size=0)
        self.assertIs(runner.validate_model(),runner.validation)

    def test_original_actor_updates_without_parameter_or_optimizer_snapshot_or_drift_rollback(self):
        runner,_=reference_fixture()
        runner.cfg.max_action_shift=1e-30
        before=[parameter.detach().clone() for parameter in runner.actor.policy.parameters()]
        with patch.object(runner,"_fit_critic",return_value=critic_noop()), \
                patch("src.wmac_perceptive.runner.copy.deepcopy",side_effect=AssertionError("unexpected rollback snapshot")):
            result=runner.update_actor_critic()
        self.assertEqual(result["actor_update_rejected"],0.)
        self.assertEqual(runner.actor_updates_accepted,1)
        self.assertTrue(any(not torch.equal(parameter,saved)
                            for parameter,saved in zip(runner.actor.policy.parameters(),before)))
        self.assertTrue(all(parameter.grad is None for parameter in runner.world_model.parameters()))

    def test_collection_events_match_actual_official_ratio_after_replay_fill(self):
        root=Path(os.environ.get("SUPERMIMIC_REFERENCE_CHECKOUT","/tmp/supermimic-official-audit-20261111"))
        path=root/"src/supermimic/runners/supermimic_runner.py"
        if not path.exists():self.skipTest("pinned reference checkout unavailable")
        self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),
            "c19751a691cabfe579da973d9c384b7355c545292bdb7748155978a9d9d78d59")
        source=path.read_text()
        namespace={}
        exec(source[source.index("class Ratio:"):source.index("class BaseRunner[")],namespace)
        runner,_=reference_fixture()
        for ratio in (.1,.2,.4,.8,1.):
            original=namespace["Ratio"](ratio)
            expected=[]
            for step in range(200):
                if original(step) and step+1>8:expected.append(step+1)
            runner.cfg.utd=ratio
            runner.cfg.collect_steps=0
            runner.sim_steps=8*runner.env.num_envs
            actual=[]
            while runner.sim_steps//runner.env.num_envs<200:
                runner.sim_steps+=runner._next_collect_count()*runner.env.num_envs
                if runner.sim_steps//runner.env.num_envs<=200:
                    actual.append(runner.sim_steps//runner.env.num_envs)
            self.assertEqual(actual,expected)


if __name__=="__main__":unittest.main()
