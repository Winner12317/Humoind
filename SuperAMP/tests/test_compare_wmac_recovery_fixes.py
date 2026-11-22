"""CPU checks for the four-arm diagnostic's budget and stale-label boundary."""
import contextlib
from dataclasses import asdict
import importlib.util
import io
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock

import numpy as np
import torch

from src.wmac.config import WmacConfig

PATH = Path(__file__).resolve().parents[1]/"scripts/compare_wmac_recovery_fixes.py"
SPEC = importlib.util.spec_from_file_location("compare_recovery_fixes_cli", PATH)
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


class RecoveryFixComparisonTest(unittest.TestCase):
    def arguments(self, output):
        return ["--checkpoint", "source.pt", "--dataset", "data.npz", "--output", str(output)]

    def test_saved_multiple_actor_updates_are_respected_and_failed_gate_skips_all(self):
        runner = SimpleNamespace(cfg=SimpleNamespace(actor_updates=2),
            update_actor_critic=Mock(side_effect=[{"actor_loss": 2., "actor_update_rejected": 0.},
                                                {"actor_loss": 4., "actor_update_rejected": 1.}]))
        self.assertEqual(CLI.actor_phase_updates(runner, False), {})
        runner.update_actor_critic.assert_not_called()
        self.assertEqual(CLI.actor_phase_updates(runner, True),
                         {"actor_loss": 3., "actor_update_rejected": .5})
        self.assertEqual(runner.update_actor_critic.call_count, 2)

    def test_default_budget_and_refusal_to_overwrite_or_cross_guard(self):
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory)/"new"
            args = CLI.parse_args(self.arguments(target))
            self.assertEqual((args.data_steps, args.updates, args.batch_size, args.wm_batch_size), (249, 8, 128, 256))
            with contextlib.redirect_stderr(io.StringIO()):
                for extra in (["--data-steps", "250"], ["--updates", "0"]):
                    with self.assertRaises(SystemExit):
                        CLI.parse_args(self.arguments(target)+extra)
                target.mkdir()
                with self.assertRaises(SystemExit):
                    CLI.parse_args(self.arguments(target))
            self.assertEqual(CLI.ARMS["legacy"], ("soft_v1", "height_only_v1"))
            self.assertEqual(CLI.ARMS["both"], ("hard_forward_v2", "upright_v2"))

    def test_stale_real_reward_targets_and_uncalibrated_source_are_refused(self):
        cfg = WmacConfig.for_profile("supermimic")
        payload = {"format": "amp-wmac-v1", "config": asdict(cfg), "iteration": 22000,
                   "critic_calibration_updates": cfg.critic_warmup_iterations,
                   **{key: {} for key in (*CLI.MODULES, *CLI.OPTIMIZERS)}}
        self.assertEqual(CLI.validate_source(payload).real_critic_coef, 0.)
        payload["config"]["real_critic_coef"] = .5
        with self.assertRaisesRegex(ValueError, "real_critic_coef"):
            CLI.validate_source(payload)
        payload["config"]["real_critic_coef"] = 0.
        payload["critic_calibration_updates"] = 0
        with self.assertRaisesRegex(ValueError, "calibration"):
            CLI.validate_source(payload)

    def test_start_zero_249_prefix_is_kept_without_mutating_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"data.npz"
            np.savez_compressed(path, step_index=np.arange(300))
            values = {"state": torch.zeros(300, 2, 71), "context": torch.zeros(300, 2, 35),
                      "next_context": torch.zeros(300, 2, 35), "first_episode_valid": torch.ones(300, 2, dtype=torch.bool)}
            values["context"][:, :, 34] = torch.arange(300)[:, None]
            values["next_context"][:, :, 34] = torch.arange(1, 301)[:, None]
            before = CLI.state_fingerprint(values)
            cropped, metadata = CLI.crop_start_zero_data([values], [{"path": str(path)}], 249)
            self.assertEqual(len(cropped[0]["state"]), 249)
            self.assertEqual(metadata[0]["maximum_used_recovery_counter"]["next_context"], 249.)
            self.assertEqual(CLI.state_fingerprint(values), before)
            values["context"][0, 0, 34] = 1.
            with self.assertRaisesRegex(ValueError, "counter0"):
                CLI.crop_start_zero_data([values], [{"path": str(path)}], 249)
            values["context"][0, 0, 34] = 0.
            np.savez_compressed(path, step_index=np.arange(1, 301))
            with self.assertRaisesRegex(ValueError, "step0"):
                CLI.crop_start_zero_data([values], [{"path": str(path)}], 249)


if __name__ == "__main__":
    unittest.main()
