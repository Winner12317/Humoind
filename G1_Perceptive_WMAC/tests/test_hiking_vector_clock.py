import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from src.wmac_perceptive.compact_gpu_replay import NoCompleteReferenceWindow
from src.wmac_perceptive.reference_loop import learn_reference


class VectorStepClockTest(unittest.TestCase):
    def window_runner(self, end, *, start=0, utd=1.):
        runner = SimpleNamespace(
            cfg=SimpleNamespace(collect_steps=0, utd=utd, iterations=end, wm_window=7,
                                actor_updates=1, validation_env_fraction=0., actor_lr=.0002,
                                save_interval=10, log_interval=2),
            sim_steps=start, env=SimpleNamespace(num_envs=1), iteration=3 if start else 0,
            imagination_steps=0, replay=SimpleNamespace(size=min(start, 8), capacity_steps=8),
            _collect_accumulator=max(start-1, 0), wall_time=0., log_dir=Path("run"),
            writer=Mock(), save=Mock(), _log=Mock(),
            update_actor_critic=Mock(return_value={"actor_loss": 1.}),
            update_discriminator=Mock(return_value=dict(
                discriminator_updates_applied=0, amp_rollout_blocks=0, amp_policy_windows=0,
                amp_policy_exposures=0, amp_invalid_windows_dropped=0)),
        )
        def collect(count):
            runner.replay.size = min(runner.replay.size + count, runner.replay.capacity_steps)
            runner.sim_steps += count
            runner.wall_time += 1.
            return {}
        runner.collect = collect
        return runner

    def test_transient_missing_windows_do_not_change_vector_or_update_clock(self):
        runner = self.window_runner(11, utd=2.)
        attempts = []
        def fit():
            attempts.append(runner.sim_steps)
            if runner.sim_steps in (8, 9):
                raise NoCompleteReferenceWindow("terminated=12")
            return {"wm_loss": 1.}
        runner.fit_model = fit
        with patch("builtins.print") as output:
            learn_reference(runner)
        self.assertEqual(attempts, [8, 9, 10, 10, 11, 11])
        self.assertEqual(runner.iteration, 4)
        self.assertEqual(runner.update_actor_critic.call_count, 4)
        self.assertEqual(runner.update_discriminator.call_count, 11)
        self.assertEqual(sum("Continuing collection" in c.args[0] for c in output.call_args_list), 1)

    def test_starvation_fails_at_capacity_or_run_end_even_after_resume(self):
        for start, end, stopped in ((0, 11, 11), (40, 52, 49)):
            with self.subTest(start=start):
                runner = self.window_runner(end, start=start)
                runner.fit_model = Mock(side_effect=NoCompleteReferenceWindow("invalid=7"))
                with patch("builtins.print"), self.assertRaisesRegex(
                    RuntimeError, f"by vector step {stopped}.*invalid=7"
                ):
                    learn_reference(runner)
                runner.update_actor_critic.assert_not_called()
                self.assertEqual(runner.sim_steps, stopped)

    def test_retry_after_more_than_one_replay_capacity_between_updates(self):
        runner = self.window_runner(25)
        runner.cfg.collect_steps = 12
        runner.fit_model = Mock(side_effect=[NoCompleteReferenceWindow("terminated=5"), {"wm_loss": 1.}])
        with patch("builtins.print"):
            learn_reference(runner)
        self.assertEqual(runner.fit_model.call_count, 2)
        runner.update_actor_critic.assert_called_once()
        self.assertEqual(runner.sim_steps, 25)

    def test_short_collection_with_no_scheduled_model_update_still_finishes(self):
        runner = self.window_runner(7)
        runner.fit_model = Mock()
        learn_reference(runner)
        runner.fit_model.assert_not_called()
        self.assertEqual(runner.sim_steps, 7)
        self.assertEqual(runner.save.call_args.args[0].name, "model_0.pt")

    def test_fractional_and_multiple_updates_with_replay_fill(self):
        for ratio in (.4, 2.0):
            with self.subTest(ratio=ratio):
                vector_steps = 26
                updates = []
                saves = []
                runner = SimpleNamespace(
                    cfg=SimpleNamespace(collect_steps=0, utd=ratio, iterations=vector_steps,
                                        wm_window=7, actor_updates=2, validation_env_fraction=0.,
                                        actor_lr=.0002, save_interval=3, log_interval=2),
                    sim_steps=0, env=SimpleNamespace(num_envs=16), iteration=0,
                    imagination_steps=0, replay=SimpleNamespace(size=0), _collect_accumulator=0., wall_time=0.,
                    log_dir=Path("run"), writer=Mock(), _log=Mock())
                def collect(count):
                    self.assertEqual(count, 1)
                    runner.replay.size += 1
                    runner.sim_steps += runner.env.num_envs
                    runner.wall_time += .25
                    return {"real_task_reward": .1}

                def actor(update_actor):
                    self.assertTrue(update_actor)
                    runner.imagination_steps += 128
                    return {"actor_loss": -2.}

                def fit():
                    updates.append(runner.replay.size)
                    runner.wall_time += .75
                    return {"wm_loss": 1.}

                runner.collect = collect
                runner.fit_model = fit
                runner.update_actor_critic = actor
                runner.update_discriminator = Mock(return_value=dict(
                    discriminator_updates_applied=0, amp_rollout_blocks=0, amp_policy_windows=0,
                    amp_policy_exposures=0, amp_invalid_windows_dropped=0))
                runner.save = lambda path: saves.append((path.name, runner.sim_steps // runner.env.num_envs))
                with patch("builtins.print") as progress:
                    learn_reference(runner)
                expected = ([9, 11, 14, 16, 19, 21, 24, 26] if ratio == .4 else
                            [step for step in range(8, 27) for _ in range(2)])
                self.assertEqual(updates, expected)
                self.assertEqual(runner.iteration, len(expected))
                self.assertEqual(runner.imagination_steps, 2 * 128 * len(expected))
                self.assertEqual(runner.update_discriminator.call_count, vector_steps)
                if ratio == .4:
                    self.assertEqual(saves, [("model_0.pt", 1), ("model_3.pt", 14),
                                             ("model_6.pt", 21), ("model_8.pt", 26)])
                else:
                    # Crossing a save boundary inside a vector step writes
                    # the completed step using its actual learner count.
                    saved_iterations = [4, 6, 10, 12, 16, 18, 22, 24, 28, 30, 34, 36, 38]
                    self.assertEqual(saves, [("model_0.pt", 1)] + [
                        (f"model_{iteration}.pt", 7 + iteration // 2) for iteration in saved_iterations
                    ])
                messages = [call.args[0] for call in progress.call_args_list]
                self.assertEqual(len(messages), 1 + len(expected) // 2)
                self.assertTrue(messages[0].startswith("[WMAC] iter=1 "))
                self.assertTrue(messages[1].startswith("[WMAC] iter=2 "))
                if ratio == .4:
                    self.assertIn("sec_per_iter=3.00 vector_step=9", messages[0])
                    self.assertIn("sec_per_iter=1.25 vector_step=11", messages[1])
