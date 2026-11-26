"""Resource accounting must survive disabled logging, index zero, and resume."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from rsl_rl.runners.amp_on_policy_runner import AmpOnPolicyRunner
from src.training_cost import WALL_TIME_SCOPE


class TinyPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net = torch.nn.Linear(3, 2)

    def load_state_dict(self, value, strict=True):
        super().load_state_dict(value, strict=strict)
        return True  # ActorCritic's documented resumed-training result.


class TinyEnv:
    num_envs, device = 3, "cpu"

    def __init__(self):
        self.steps = 0

    def get_observations(self):
        return {"actor": torch.zeros(3, 3), "amp": torch.zeros(3, 195)}

    def step(self, actions):
        self.steps += 1
        return self.get_observations(), torch.ones(3), torch.zeros(3, dtype=torch.bool), {}


def make_runner(start=10.):
    runner = AmpOnPolicyRunner.__new__(AmpOnPolicyRunner)
    runner._cost_started, runner._cost_wall_offset = start, 0.
    runner._physical_transitions, runner._cost_history_complete = 0, True
    runner._cost_trajectory_id = "test-new-training-trajectory"
    runner.env, runner.device = TinyEnv(), "cpu"
    runner.cfg = {"amp_body_names": tuple(range(13))}
    runner.amp_feature_schema, runner.amp_feature_dim = "legacy", 195
    runner.gpu_world_size, runner.is_distributed = 1, False
    runner.log_dir, runner.writer, runner.disable_logs = None, None, False
    runner.logger_type, runner.training_type, runner.privileged_obs_type = "tensorboard", "rl", None
    runner.current_learning_iteration, runner.num_steps_per_env = 0, 24
    runner.tot_timesteps, runner.tot_time = 0, 0.
    runner.empirical_normalization = False
    runner.obs_normalizer = torch.nn.Identity()
    policy = TinyPolicy()
    disc = torch.nn.Linear(390, 1)
    disc.predict_amp_reward = lambda first, second, rewards, **kwargs: (rewards, None)
    runner.alg = SimpleNamespace(
        policy=policy, discriminator=disc, rnd=None, amp_normalizer=None,
        optimizer=torch.optim.Adam(policy.parameters()),
        act=lambda *args: torch.zeros(3, 2), process_env_step=lambda *args: None,
        compute_returns=lambda *args: None, update=lambda: {},
    )
    return runner


class PpoTrainingCostTest(unittest.TestCase):
    def test_rollout_counts_without_writer_and_iteration_zero_already_contains_data(self):
        runner = make_runner()
        runner.learn(1)
        self.assertEqual(runner.current_learning_iteration, 0)
        self.assertEqual(runner.env.steps, 24)
        self.assertEqual(runner.tot_timesteps, 3 * 24)
        self.assertEqual(runner.training_cost()["physical_transitions"], 3 * 24)
        runner.learn(2)
        self.assertEqual(runner.env.steps, 72)
        self.assertEqual(runner.training_cost()["physical_transitions"], 3 * 72)

    def test_saved_resources_restore_counts_and_include_new_runner_initialization(self):
        runner = make_runner()
        runner.learn(1)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "model_0.pt")
            with patch("rsl_rl.runners.amp_on_policy_runner.time.perf_counter", return_value=30.):
                runner.save(path)
            payload = torch.load(path, weights_only=False)
            self.assertEqual(payload["training_cost"]["wall_seconds"], 20.)
            self.assertEqual(payload["training_cost"]["wall_time_scope"], WALL_TIME_SCOPE)
            resumed = make_runner(start=100.)
            with patch("rsl_rl.runners.amp_on_policy_runner.time.perf_counter", return_value=115.):
                resumed.load(path)
                cost = resumed.training_cost()
            self.assertEqual(cost["physical_transitions"], 72)
            self.assertEqual(cost["wall_seconds"], 35.)
            self.assertTrue(cost["history_complete"])
            self.assertEqual(cost["trajectory_id"], "test-new-training-trajectory")
            resumed.learn(1)
            self.assertEqual(resumed.training_cost()["physical_transitions"], 144)

    def test_old_checkpoint_never_guesses_cost_from_iteration(self):
        runner = make_runner()
        runner._restore_training_cost({"iter": 60000})
        runner.learn(1)
        cost = runner.training_cost()
        self.assertIsNone(cost["physical_transitions"])
        self.assertIsNone(cost["wall_seconds"])
        self.assertFalse(cost["history_complete"])
        self.assertIsNone(cost["trajectory_id"])

    def test_incomplete_history_and_changed_gpu_count_are_not_promoted_to_complete(self):
        runner = make_runner()
        cost = {"physical_transitions": 123, "wall_seconds": 40., "gpu_count": 0,
                "history_complete": False, "wall_time_scope": WALL_TIME_SCOPE}
        runner._restore_training_cost({"training_cost": cost})
        self.assertFalse(runner.training_cost()["history_complete"])
        self.assertEqual(runner.training_cost()["physical_transitions"], 123)
        cost.update(history_complete=True, gpu_count=2)
        runner._restore_training_cost({"training_cost": cost})
        self.assertFalse(runner.training_cost()["history_complete"])

    def test_evaluator_passes_checkpoint_cost_without_adding_rollout_time(self):
        path = Path(__file__).resolve().parents[1] / "scripts/evaluate_wmac.py"
        spec = importlib.util.spec_from_file_location("evaluation_training_cost_test", path)
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        from src.wmac.networks import Actor

        actor = Actor(hidden=(8, 4))
        cost = {"physical_transitions": 120, "wall_seconds": 90., "gpu_count": 1,
                "wall_time_scope": WALL_TIME_SCOPE, "history_complete": True}
        payload = {"format": "amp-wmac-v1", "config": {"policy_hidden": (8, 4)},
                   "actor": actor.state_dict(), "training_cost": cost}
        with patch("torch.load", return_value=payload):
            loaded, _ = cli.load_actor("unused.pt", "cpu")
        self.assertEqual(loaded.training_cost, cost)
        del payload["training_cost"]
        with patch("torch.load", return_value=payload):
            legacy, _ = cli.load_actor("unused.pt", "cpu")
        self.assertIsNone(legacy.training_cost)


if __name__ == "__main__":
    unittest.main()
