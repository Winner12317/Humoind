"""Depth fast paths preserve capture draws, episode boundaries and observations."""
from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import torch

from src.tasks.amp_perceptive.sensors import read_depth_frame
from src.wmac_perceptive.augmentation import augment_depth_frame
from src.wmac_perceptive.hiking_depth_randomization import HikingDepthCapture
from src.wmac_perceptive.latent_training import collect_latent
from src.wmac_perceptive.perception import DepthHistory, PerceptionSpec
from src.wmac_perceptive.runner import PerceptiveRunner


class Scene(dict):
    pass


class CountedRanges:
    def __init__(self, values):
        self.values, self.reads, self.forbid = values, 0, False

    @property
    def distances(self):
        if self.forbid:
            raise AssertionError("Cached read rebuilt the physical packet")
        self.reads += 1
        return self.values


def sensor_env(version=2, *, randomize=True):
    n = 3
    ranges = CountedRanges(torch.full((n, 36*64), .75))
    sensor = NS(data=ranges, optical_cosines=torch.linspace(.8, 1., 36*64)[None],
                cfg=NS(pattern=NS(sensor_protocol="g1_hiking_dr_v1")))
    families = {"stairs": NS(proportion=.5), "stepping_stones": NS(proportion=.5)}
    terrain = NS(cfg=NS(terrain_generator=NS(sub_terrains=families, num_cols=2)),
                 terrain_types=torch.tensor([0, 1, 0]))
    scene = Scene(hiking_depth=sensor)
    scene.terrain = terrain
    env = NS(num_envs=n, device="cpu", common_step_counter=0, scene=scene,
             cfg=NS(_hiking_training_version=version, _hiking_depth_randomization=randomize))
    env._hiking_depth_capture = HikingDepthCapture(n, enabled_randomization=randomize,
        generator=torch.Generator().manual_seed(591))
    return env, ranges


class HikingSensorFastPathTest(unittest.TestCase):
    def test_cached_sensor_skips_range_and_terrain_reads_without_drawing_randomness(self):
        env, ranges = sensor_env()
        first = read_depth_frame(env)
        frozen = first.clone()
        capture = env._hiking_depth_capture
        rng = capture.generator.get_state().clone()
        ranges.forbid = True
        # The cached read should not even need terrain configuration.
        env.scene.terrain = None
        self.assertIs(read_depth_frame(env), first)
        self.assertTrue(torch.equal(first, frozen))
        self.assertTrue(torch.equal(capture.generator.get_state(), rng))
        self.assertEqual((ranges.reads, capture.capture_count), (1, 1))

    def test_latest_sensor_and_original_path_have_identical_packets_counters_and_rng(self):
        modern, modern_ranges = sensor_env(2)
        legacy, legacy_ranges = sensor_env(1)
        for tick, reset in ((0, None), (0, None), (0, [1]), (0, None), (1, None), (2, [0, 2])):
            for env, ranges in ((modern, modern_ranges), (legacy, legacy_ranges)):
                env.common_step_counter = tick
                ranges.values = torch.full_like(ranges.values, .5+.2*tick+(0.1 if reset else 0.))
                if reset is not None:
                    env._hiking_depth_capture.reset(reset)
            actual, expected = read_depth_frame(modern), read_depth_frame(legacy)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            a, b = modern._hiking_depth_capture, legacy._hiking_depth_capture
            self.assertTrue(torch.equal(a.generator.get_state(), b.generator.get_state()))
            torch.testing.assert_close(a.captures_per_env, b.captures_per_env, rtol=0, atol=0)
            self.assertEqual(a.capture_count, b.capture_count)
        self.assertEqual(modern_ranges.reads, 4)
        self.assertEqual(legacy_ranges.reads, 6)

    def test_missing_cache_and_same_step_reset_capture_new_rows_only(self):
        env, ranges = sensor_env(randomize=False)
        del env._hiking_depth_capture
        terminal = read_depth_frame(env).clone()
        capture = env._hiking_depth_capture
        capture.reset([1])
        self.assertIsNone(capture.cached_packet(control_step=0))
        ranges.values.fill_(1.5)
        reset = read_depth_frame(env).clone()
        torch.testing.assert_close(reset[[0, 2]], terminal[[0, 2]], rtol=0, atol=0)
        self.assertFalse(torch.equal(reset[1], terminal[1]))
        self.assertEqual(ranges.reads, 2)
        torch.testing.assert_close(capture.captures_per_env, torch.tensor([1, 2, 1]))
        self.assertTrue(torch.equal(terminal[[0, 2]], reset[[0, 2]]))
        self.assertIs(capture.cached_packet(control_step=0), capture.packet)
        self.assertIsNone(capture.cached_packet(control_step=1))
        with self.assertRaisesRegex(ValueError, "nonnegative integer"):
            capture.cached_packet(control_step=-1)
        env.common_step_counter = 1
        read_depth_frame(env)
        with self.assertRaisesRegex(ValueError, "rewind"):
            capture.cached_packet(control_step=0)


class RecordingActor:
    def __init__(self, *, nonfinite=False):
        self.inputs, self.nonfinite = [], nonfinite

    def encode_processed(self, stack):
        self.inputs.append(stack.clone())
        return stack.flatten(1).mean(-1, keepdim=True).expand(-1, 128)

    def forward_latent(self, proprio, latent):
        result = latent[:, :1].expand(-1, 29)*.1
        return result*float("nan") if self.nonfinite else result


class RecordingReplay:
    def __init__(self, masks):
        self.masks, self.rows = masks, []

    def add(self, row):
        self.rows.append({key: value.clone() for key, value in row.items()})
        return self.masks[len(self.rows)-1].clone()


def collector(version, *, processed=True, local_noise=0., nonfinite=False, delay=1):
    n, spec = 4, PerceptionSpec(include_valid=False, proprio_dim=768, delay=delay)
    cfg = NS(hiking_training_version=version, exploration_noise=.1, action_limit=20.,
             replay_storage="frame_uint8", depth_noise_std_m=local_noise,
             depth_dropout_probability=0.)
    runner = NS(cfg=cfg, spec=spec, _processed_history=processed, sim_steps=0,
                uses_reference_schedule=True, actor=RecordingActor(nonfinite=nonfinite))
    runner._augment_depth = lambda frame: augment_depth_frame(frame, std_m=local_noise,
        dropout_probability=0., max_depth=spec.max_depth)
    preprocessing = []
    def preprocess(frame):
        preprocessing.append(len(frame))
        return PerceptiveRunner._policy_depth_frame(runner, frame)
    runner._policy_depth_frame = preprocess
    history_spec = (replace(spec, height=18, width=32, crop_top=0, crop_bottom=0,
                           crop_left=0, crop_right=0, blur_kernel=1) if processed else spec)
    raw = torch.full((n, 2, 36, 64), .2)
    raw[:, 1] = 1.
    raw[:, 0] += torch.arange(n)[:, None, None]*.025
    runner.depth_history = DepthHistory(history_spec, n)
    runner.depth_history.reset(preprocess(raw))
    preprocessing.clear()
    runner.current_depth = raw
    runner.env = NS(num_envs=n, common_step_counter=0,
                    obs_buf={"actor": torch.zeros(n, spec.proprio_dim)})
    masks = (torch.tensor([True, False, True, True]), torch.zeros(n, dtype=torch.bool),
             torch.ones(n, dtype=torch.bool))
    runner.replay, runner._train_env_mask = RecordingReplay(masks), torch.ones(n, dtype=torch.bool)
    normalizer_inputs = []
    runner.world_model = NS(update_normalizers_latent=lambda *args:
        normalizer_inputs.append(tuple(arg.clone() for arg in args)))
    actions = []
    packets = []
    def step(action):
        t = len(actions)
        actions.append(action.clone())
        terminal = torch.tensor([t == 0, False, False, False])
        timeout = torch.tensor([False, False, t == 1, False])
        end = raw.clone()
        end[:, 0] += .1*(t+1)
        end[:, 0, 22, 24] += .05
        fresh = end.clone()
        fresh[terminal | timeout, 0] = .95
        packets.append(fresh)
        reward = torch.arange(1, n+1).float()
        reward[~masks[t]] = float("nan")
        state = torch.zeros(n, 71)
        state[:, 3], state[:, 2] = 1., .8
        row = dict(state=state, next_state=state.clone(), context=torch.zeros(n, 133),
                   next_context=torch.zeros(n, 133), action=action, previous_action=torch.zeros(n, 29),
                   actor_history=runner.env.obs_buf["actor"].clone(), contact_cost=torch.zeros(n, 13),
                   next_depth_frame=end, relative_height=torch.full((n,), .8), reward=reward,
                   terminated=terminal, timeout=timeout, boundary=terminal | timeout)
        if t != 1:
            row["command_standing"] = torch.tensor([True, False, True, False])
        runner.env.transition = row
        runner.env.common_step_counter += 1
        runner.env.obs_buf["actor"] = runner.env.obs_buf["actor"]+.01
    runner.env.step = step
    return runner, preprocessing, actions, normalizer_inputs, packets


class HikingCollectorFastPathTest(unittest.TestCase):
    def run_fixture(self, version, **kwargs):
        torch.manual_seed(671)
        runner, calls, actions, normals, packets = collector(version, **kwargs)
        with patch("src.wmac_perceptive.latent_training.read_depth_frame", side_effect=lambda env: packets[-1]):
            metrics = collect_latent(runner, 3)
        return runner, calls, actions, normals, metrics, torch.get_rng_state().clone()

    def assert_equal_streams(self, actual, expected):
        a, _, aa, an, am, arng = actual
        b, _, ba, bn, bm, brng = expected
        for tensors, reference in ((aa, ba), (a.actor.inputs, b.actor.inputs)):
            for value, truth in zip(tensors, reference):
                torch.testing.assert_close(value, truth, rtol=0, atol=0)
        for values, truths in zip(an, bn):
            for value, truth in zip(values, truths):
                torch.testing.assert_close(value, truth, rtol=0, atol=0)
        for row, reference in zip(a.replay.rows, b.replay.rows):
            self.assertEqual(row.keys(), reference.keys())
            for key, value in row.items():
                torch.testing.assert_close(value, reference[key], rtol=0, atol=0, equal_nan=True)
        torch.testing.assert_close(a.depth_history.ring, b.depth_history.ring, rtol=0, atol=0)
        torch.testing.assert_close(a.current_depth, b.current_depth, rtol=0, atol=0)
        self.assertTrue(torch.equal(arng, brng))
        self.assertEqual(a.sim_steps, 12)
        self.assertEqual(am, bm)

    def test_reset_subset_preprocessing_preserves_terminal_labels_delay_history_and_rng(self):
        actual, expected = self.run_fixture(2), self.run_fixture(1)
        self.assert_equal_streams(actual, expected)
        self.assertEqual(actual[1], [4, 1, 4, 1, 4])
        self.assertEqual(expected[1], [4, 4, 4, 4, 4, 4])
        # No post-reset far image is inserted into a physical transition target.
        self.assertLess(float(actual[0].replay.rows[0]["next_depth_stack"].max()), .9)
        self.assertEqual(actual[4]["invalid_transition_fraction"], 5/12)
        self.assertEqual(actual[4]["standing_command_fraction"], 1/3)

    def test_legacy_augmentation_or_raw_history_retains_original_preprocessing_and_draws(self):
        for options in ({"local_noise": .01}, {"processed": False}):
            with self.subTest(options=options):
                actual, expected = self.run_fixture(2, **options), self.run_fixture(1, **options)
                self.assert_equal_streams(actual, expected)
                self.assertEqual(actual[1], expected[1])

    def test_latest_delayed_collection_avoids_the_temporary_full_end_ring(self):
        with patch("src.wmac_perceptive.latent_training.advance_depth_ring",
                   side_effect=AssertionError("Collector constructed the complete successor ring")):
            actual = self.run_fixture(2)
        self.assert_equal_streams(actual, self.run_fixture(1))

    def test_immediate_camera_history_keeps_functional_successor_sampling(self):
        actual, expected = self.run_fixture(2, delay=0), self.run_fixture(1, delay=0)
        self.assert_equal_streams(actual, expected)

    def test_history_without_new_sampling_api_keeps_original_collection_contract(self):
        class LegacyHistory:
            def __init__(self, history):
                self.history, self.spec = history, history.spec
            @property
            def ring(self):return self.history.ring
            def append(self, *args, **kwargs):return self.history.append(*args, **kwargs)
        torch.manual_seed(671)
        runner, calls, actions, normals, packets = collector(2)
        runner.depth_history = LegacyHistory(runner.depth_history)
        with patch("src.wmac_perceptive.latent_training.read_depth_frame", side_effect=lambda env: packets[-1]):
            metrics = collect_latent(runner, 3)
        actual = runner, calls, actions, normals, metrics, torch.get_rng_state().clone()
        self.assert_equal_streams(actual, self.run_fixture(1))

    def test_fast_metrics_use_one_explicit_host_copy_per_collect_and_empty_collect_is_zero(self):
        original_cpu, copies = torch.Tensor.cpu, []
        def copy(value, *args, **kwargs):
            copies.append(value.shape)
            return original_cpu(value, *args, **kwargs)
        with patch.object(torch.Tensor, "cpu", copy):
            self.run_fixture(2)
        self.assertEqual(copies, [torch.Size([8])])
        runner, *_ = collector(2)
        self.assertEqual(collect_latent(runner, 0), dict.fromkeys(("real_task_reward", "terminated_fraction",
            "timeout_fraction", "recovery_fraction", "boundary_fraction", "invalid_transition_fraction",
            "sensor_valid_fraction", "standing_command_fraction"), 0.))

    def test_nonfinite_action_is_still_rejected_before_physical_step(self):
        runner, _, actions, _, _ = collector(2, nonfinite=True)
        with self.assertRaisesRegex(RuntimeError, "Non-finite latent policy action"):
            collect_latent(runner, 1)
        self.assertEqual(actions, [])
        self.assertEqual(runner.replay.rows, [])


if __name__ == "__main__":
    unittest.main()
