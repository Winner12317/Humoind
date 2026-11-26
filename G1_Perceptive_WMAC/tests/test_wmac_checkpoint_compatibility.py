"""CPU regressions for explicit saved recovery behavior and script bindings."""
from __future__ import annotations

import copy
from dataclasses import asdict
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from src.wmac.checkpoints import (
    checkpoint_height_reward_semantics, checkpoint_recovery_semantics,
    resolve_checkpoint_config,
)
from src.wmac.config import PROFILES, WmacConfig
import src.recovery_rewards  # Load Torch before temporary sys.modules isolation.


def payload(config=None, **metadata):
    return {"format": "amp-wmac-v1", "config": config or {}, **metadata}


class CheckpointRecoveryCompatibilityTest(unittest.TestCase):
    def test_every_old_profile_keeps_old_behavior_and_does_not_inherit_authorization(self):
        for profile in PROFILES:
            with self.subTest(profile=profile):
                source = payload({"profile": profile, "allow_recovery_semantics_migration": True})
                original = copy.deepcopy(source)
                restored = resolve_checkpoint_config(source)
                self.assertEqual(restored.recovery_counter_semantics, "soft_v1")
                self.assertEqual(restored.recovery_height_reward, "height_only_v1")
                self.assertFalse(restored.allow_recovery_semantics_migration)
                self.assertEqual(source, original)
                fresh = WmacConfig.for_profile(profile)
                self.assertEqual(fresh.recovery_counter_semantics, "hard_forward_v2")
                self.assertEqual(fresh.recovery_height_reward, "upright_v2")

    def test_saved_tuple_fields_and_legacy_optimizer_betas_survive_resolution(self):
        source = payload({"profile": "balanced", "policy_hidden": [23, 11]},
                         opt_actor={"param_groups": [{"betas": [0.8, 0.97]}]})
        restored = resolve_checkpoint_config(source)
        self.assertEqual(restored.policy_hidden, (23, 11))
        self.assertEqual(restored.actor_betas, (0.8, 0.97))
        source["config"]["actor_betas"] = [.7, .98]
        self.assertEqual(resolve_checkpoint_config(source).actor_betas, (.7, .98))

    def test_new_version_requires_both_semantics_and_rejects_unknown_or_conflicting_labels(self):
        complete = asdict(WmacConfig())
        source = payload(complete, implementation_version=4)
        self.assertEqual(resolve_checkpoint_config(source).recovery_height_reward, "upright_v2")
        for key, reader in (("recovery_counter_semantics", checkpoint_recovery_semantics),
                            ("recovery_height_reward", checkpoint_height_reward_semantics)):
            with self.subTest(key=key):
                missing = copy.deepcopy(source)
                missing["config"].pop(key)
                with self.assertRaisesRegex(ValueError, "missing"):
                    resolve_checkpoint_config(missing)
                unknown = copy.deepcopy(source)
                unknown["config"][key] = "future_rule"
                with self.assertRaisesRegex(ValueError, "Unknown"):
                    reader(unknown)
                conflict = copy.deepcopy(source)
                conflict[key] = "soft_v1" if key == "recovery_counter_semantics" else "height_only_v1"
                with self.assertRaisesRegex(ValueError, "conflicts"):
                    reader(conflict)

    def test_explicit_top_level_labels_can_restore_config_without_current_defaults(self):
        source = payload({}, implementation_version=4, recovery_counter_semantics="soft_v1",
                         recovery_height_reward="height_only_v1")
        cfg = resolve_checkpoint_config(source)
        self.assertEqual((cfg.recovery_counter_semantics, cfg.recovery_height_reward),
                         ("soft_v1", "height_only_v1"))


class BenchmarkRecoveryBindingTest(unittest.TestCase):
    def test_optimized_and_archived_environment_receive_matching_recovery_modes(self):
        script = Path(__file__).resolve().parents[1] / "scripts/benchmark_wmac.py"
        specification = importlib.util.spec_from_file_location("wmac_benchmark_binding", script)
        benchmark = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(benchmark)
        for implementation in ("optimized", "legacy"):
            with self.subTest(implementation=implementation):
                env_cfg = SimpleNamespace(scene=SimpleNamespace(), rewards={
                    "track_root_height": SimpleNamespace(params={"std": .3})})
                config = SimpleNamespace(task="test-task", num_envs=2, seed=9, device="cpu",
                    recovery_counter_semantics="soft_v1", recovery_height_reward="height_only_v1")
                class ModernAdapter:
                    def __init__(self, env, recovery_counter_semantics):
                        self.recovery_counter_semantics = recovery_counter_semantics
                class ArchivedAdapter:
                    def __init__(self, env):
                        pass
                fake_env = SimpleNamespace()
                def capture(*, cfg, device):
                    fake_env.cfg = cfg
                    return fake_env
                modules = {"env": SimpleNamespace(CaptureEnv=capture), "state": SimpleNamespace(
                    StateAdapter=ArchivedAdapter if implementation == "legacy" else ModernAdapter)}
                registry = SimpleNamespace(load_env_cfg=lambda task: env_cfg)
                with patch.dict(sys.modules, {"mjlab.tasks.registry": registry}), patch.object(
                        benchmark.importlib, "import_module", side_effect=lambda name: modules[name.rsplit(".", 1)[1]]):
                    actual = benchmark.make_environment(config, implementation, "/tmp/archived_test_baseline")
                self.assertEqual(actual.adapter.recovery_counter_semantics, "soft_v1")
                self.assertEqual(actual.cfg.rewards["track_root_height"].params["upright_power"], 0.)
                self.assertEqual(actual.cfg.scene.num_envs, 2)


if __name__ == "__main__":
    unittest.main()
