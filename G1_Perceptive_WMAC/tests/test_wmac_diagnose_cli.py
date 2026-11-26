"""CPU checks for fixed-data diagnostic provenance and intervention controls."""
from __future__ import annotations

from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.wmac.config import WmacConfig


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/diagnose_wmac.py"
SPEC = importlib.util.spec_from_file_location("wmac_diagnose_cli", SCRIPT)
CLI = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLI)


def recording(steps=5, num_envs=3):
    result = {}
    for key, width in CLI.FIELDS.items():
        shape = (steps, num_envs) if width is None else (steps, num_envs, width)
        result[key] = np.zeros(shape, dtype=bool if key in CLI.FLAGS else np.float32)
    result["state"][..., 2] = .7
    result["state"][..., 3] = 1.
    result["next_state"][:] = result["state"]
    result["first_episode_valid"] = np.ones((steps, num_envs), dtype=bool)
    result["recording_protocol_json"] = np.asarray(json.dumps({"kind": "test"}))
    return result


def payload(config=None):
    result = {"format": "amp-wmac-v1", "config": asdict(config or WmacConfig()), "iteration": 30000}
    for name in CLI.MODULES:
        result[name] = {"weight": torch.ones(2, 3)}
    return result


class DiagnosticDatasetTest(unittest.TestCase):
    def test_terminal_step_is_kept_reset_episodes_quarantined_and_chunks_cut(self):
        first, second = recording(), recording()
        first["terminated"][1, 0] = True
        first["boundary"][1, 0] = True
        first["first_episode_valid"][2:, 0] = False
        original = first["state"].copy()
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("walk.npz", "recovery.npz")]
            np.savez_compressed(paths[0], **first)
            np.savez_compressed(paths[1], **second)
            hashes = [CLI.sha256(path) for path in paths]
            chunks, metadata = CLI.read_dataset(paths)
            replay = CLI.build_replay(chunks)
            self.assertTrue(replay._valid[1, 0])
            self.assertFalse(replay._valid[2, 0])
            self.assertTrue(torch.isnan(replay.data["state"][2, 0]).all())
            self.assertTrue(replay._cut_rows[4].all())
            self.assertEqual(replay.num_valid("all"), 27)
            np.testing.assert_array_equal(chunks[0]["state"].numpy(), original)
            self.assertEqual(hashes, [CLI.sha256(path) for path in paths])
            self.assertEqual(metadata[0]["first_episode_valid_count"], 12)

    def test_inconsistent_reset_mask_or_nonboolean_flags_are_rejected(self):
        for fault in ("mask", "flag"):
            arrays = recording()
            if fault == "mask":
                arrays["first_episode_valid"][1, 0] = False
            else:
                arrays["timeout"] = arrays["timeout"].astype(float)
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "bad.npz"
                np.savez_compressed(path, **arrays)
                with self.assertRaises(ValueError):
                    CLI.read_dataset([path])

    def test_nonfinite_real_data_is_quarantined_without_rewriting_input(self):
        arrays = recording()
        arrays["action"][2, 1, 0] = np.nan
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "nonfinite.npz"
            np.savez_compressed(path, **arrays)
            chunks, metadata = CLI.read_dataset([path])
            replay = CLI.build_replay(chunks)
            self.assertFalse(replay._valid[2, 1])
            self.assertTrue(torch.isnan(replay.data["action"][2, 1, 0]))
            self.assertEqual(metadata[0]["nonfinite_values_by_field"]["action"], 1)


class DiagnosticControlTest(unittest.TestCase):
    def test_reused_data_sweep_evaluates_starting_checkpoint_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = SimpleNamespace(updates=2, no_final_evaluation=False,
                                   good_checkpoint=root / "good.pt")
            baseline = {"arguments": {"output": str(root / "evaluations/starting_good")}, "results": {}}
            manifest = {}
            with patch.object(CLI, "evaluate", return_value=baseline) as evaluate:
                self.assertIs(CLI.ensure_starting_baseline(args, root, manifest), baseline)
                self.assertIs(CLI.ensure_starting_baseline(args, root, manifest), baseline)
            evaluate.assert_called_once_with(args.good_checkpoint, root / "evaluations/starting_good", args)
            self.assertIn("physical_starting_baseline", manifest)
            args.no_final_evaluation = True
            with patch.object(CLI, "evaluate") as evaluate:
                self.assertIsNone(CLI.ensure_starting_baseline(args, root, {}))
                evaluate.assert_not_called()

    def test_pair_audit_checks_actual_initial_arrays_and_rejects_protocol_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            arrays = {name: np.zeros((2, 3), dtype=np.float32)
                      for name in ("state", "context", "command", "actor_history")}
            a, b = root / "a.npz", root / "b.npz"
            np.savez_compressed(a, **arrays)
            np.savez_compressed(b, **arrays)
            arguments = {name: 1 for name in (
                "mode", "seed", "num_envs", "steps", "device", "push", "observation_noise",
                "command_vx", "command_vy", "command_yaw", "height_threshold", "upright_cos",
                "stable_steps", "walk_hold_steps", "walk_velocity_tolerance", "walk_yaw_tolerance")}
            protocol = {name: "fixed" for name in (
                "first_episode_only", "task", "policy", "discriminator_used", "stable_recovery", "continued_walk")}
            def evaluation(path, digest):
                return {"arguments": dict(arguments), "protocol": dict(protocol), "results": {
                    mode: {"initial_state_npz": str(path), "initial_state_sha256": digest,
                           "num_envs": 2, "steps_requested": 250, "control_dt": .02}
                    for mode in ("walk", "recovery")}}
            reference, candidate = evaluation(a, "different-a"), evaluation(b, "different-b")
            audit = CLI.audit_evaluation_pair(reference, candidate)
            self.assertTrue(audit["paired_comparison_allowed"])
            self.assertTrue(audit["modes"]["recovery"]["all_arrays_exact"])
            self.assertFalse(audit["modes"]["recovery"]["recorded_initial_hashes_equal"])
            arrays["state"][0, 0] = .01
            np.savez_compressed(b, **arrays)
            self.assertFalse(CLI.audit_evaluation_pair(reference, candidate)["causal_comparison_eligible"])
            np.savez_compressed(b, **{name: np.zeros((2, 3), dtype=np.float32) for name in arrays})
            candidate["arguments"]["seed"] = 999
            self.assertFalse(CLI.audit_evaluation_pair(reference, candidate)["paired_comparison_allowed"])

    def test_compatible_sources_reject_architecture_or_objective_changes(self):
        source = payload()
        self.assertEqual(CLI.validate_compatible(source, payload()).task, "Unitree-G1-AMP-Flat")
        for key, value in (("action_limit", 5.), ("style_weight", .1), ("policy_hidden", (9,))):
            changed = WmacConfig()
            setattr(changed, key, value)
            with self.subTest(key=key), self.assertRaisesRegex(ValueError, "matching architecture"):
                CLI.validate_compatible(source, payload(changed))
        malformed = payload()
        malformed["critic"]["weight"] = torch.ones(3, 3)
        with self.assertRaisesRegex(ValueError, "incompatible shapes"):
            CLI.validate_compatible(source, malformed)

    def test_each_variant_changes_exactly_one_control(self):
        source = WmacConfig.for_profile("supermimic")
        for variant in CLI.VARIANTS:
            intervention = CLI.variant_changes(variant, source)
            count = len(intervention["config_overrides"]) + sum(
                intervention[name] for name in ("skip_model_update", "skip_discriminator_update", "skip_critic_update"))
            self.assertEqual(count, 0 if variant == "control" else 1, variant)
        with self.assertRaisesRegex(ValueError, "strictly shorter"):
            CLI.variant_changes("short_horizon", source, short_horizon=32)
        self.assertEqual(source.imagination_horizon, 32)
        self.assertTrue(source.direct_style_grad)

    def test_legacy_modes_are_explicit_and_cross_mode_component_swaps_are_rejected(self):
        old = payload()
        old["config"].pop("recovery_counter_semantics")
        old["config"].pop("recovery_height_reward")
        explicit = WmacConfig(recovery_counter_semantics="soft_v1", recovery_height_reward="height_only_v1")
        resolved = CLI.validate_compatible(old, payload(explicit))
        self.assertEqual(resolved.recovery_counter_semantics, "soft_v1")
        self.assertEqual(resolved.recovery_height_reward, "height_only_v1")
        for key, value in (("recovery_counter_semantics", "hard_forward_v2"),
                           ("recovery_height_reward", "upright_v2")):
            with self.subTest(key=key):
                changed = payload(explicit)
                changed["config"][key] = value
                with self.assertRaisesRegex(ValueError, "matching architecture"):
                    CLI.validate_compatible(old, changed)

    def test_phase_rng_keeps_actor_draws_identical_when_other_phase_is_skipped(self):
        py_before, np_before, torch_before = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
        with CLI.phase_rng(42, 17, "model"):
            torch.rand(100)
            np.random.rand(30)
            random.random()
        with CLI.phase_rng(42, 17, "actor"):
            first = torch.rand(8)
        with CLI.phase_rng(42, 17, "actor"):
            second = torch.rand(8)
        torch.testing.assert_close(first, second)
        self.assertEqual(random.getstate(), py_before)
        for a, b in zip(np.random.get_state(), np_before):
            if isinstance(a, np.ndarray):
                np.testing.assert_array_equal(a, b)
            else:
                self.assertEqual(a, b)
        torch.testing.assert_close(torch.get_rng_state(), torch_before)

    def test_dry_run_keeps_checkpoint_bytes_and_rejects_output_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            good, bad, output = root / "good.pt", root / "bad.pt", root / "plan"
            torch.save(payload(), good)
            torch.save(payload(), bad)
            before = CLI.sha256(good), CLI.sha256(bad)
            args = ["--good-checkpoint", str(good), "--bad-checkpoint", str(bad),
                    "--output", str(output), "--dry-run"]
            CLI.main(args)
            manifest = json.loads((output / "manifest.json").read_text())
            self.assertEqual(manifest["status"], "planned")
            self.assertEqual(manifest["arguments"]["updates"], 0)
            self.assertEqual(before, (CLI.sha256(good), CLI.sha256(bad)))
            with self.assertRaises(FileExistsError):
                CLI.main(args)


if __name__ == "__main__":
    unittest.main()
