"""PPO checkpoint selection when WMAC runs share the experiment directory."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.train import get_ppo_checkpoint_path


class PpoResumeTest(unittest.TestCase):
  def setUp(self):
    directory = TemporaryDirectory()
    self.addCleanup(directory.cleanup)
    self.log_root = Path(directory.name)

  def add_checkpoint(self, run_name, checkpoint_name):
    run = self.log_root / run_name
    run.mkdir(exist_ok=True)
    checkpoint = run / checkpoint_name
    checkpoint.touch()
    return checkpoint

  def test_latest_run_and_checkpoint_skip_wmac(self):
    self.add_checkpoint("2026-01-01_old", "model_42.pt")
    self.add_checkpoint("2026-01-02_new", "model_9.pt")
    expected = self.add_checkpoint("2026-01-02_new", "model_10.pt")
    self.add_checkpoint("wmac_2026-01-03", "model_99.pt")
    self.add_checkpoint("wandb_checkpoints", "model_100.pt")

    actual = get_ppo_checkpoint_path(self.log_root, ".*", r"model_\d+\.pt")

    self.assertEqual(actual, expected)

  def test_explicit_regex_preserves_inline_flags_and_alternation(self):
    expected = self.add_checkpoint("PPO+[seed]", "model_8.pt")
    self.add_checkpoint("later", "model_9.pt")
    self.add_checkpoint("wmac_PPO+[seed]", "model_10.pt")

    actual = get_ppo_checkpoint_path(
      self.log_root, r"(?i)^ppo\+\[seed\]$|^wmac_", r"model_\d+\.pt"
    )

    self.assertEqual(actual, expected)

  def test_no_matching_ppo_run_reports_original_pattern(self):
    self.add_checkpoint("wmac_2026-01-03", "model_99.pt")

    with self.assertRaisesRegex(ValueError, r"No run directories.*matching '\.\*'"):
      get_ppo_checkpoint_path(self.log_root, ".*", r"model_\d+\.pt")


if __name__ == "__main__":
  unittest.main()
