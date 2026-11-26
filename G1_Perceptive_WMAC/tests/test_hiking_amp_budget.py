"""Fresh Hiking AMP coverage and exact logical batches under WM collection."""
from __future__ import annotations

import ast
from collections import Counter
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.wmac.replay import ReplayBuffer
from src.wmac_perceptive.hiking_amp import (
    add_sample_history, after_replay_add, before_replay_add, configure_replay,
    flatten_history,
)
from src.wmac_perceptive.hiking_amp_budget import (
    HikingAmpBudget, HikingAmpFeatureCache, HikingAmpRollout,HikingAmpPolicyCache,HikingAmpReferenceCache,
    HikingEpisodeReference,
    LEARNING_EPOCHS, MINI_BATCHES, ROLLOUT_STEPS, discriminator_step,
    reference_features, sample_reference_ends,
)


def make_replay(capacity=128, num_envs=5, validation_fraction=.2):
    replay = ReplayBuffer(capacity, num_envs, validation_env_fraction=validation_fraction)
    configure_replay(replay)
    return replay


def add_steps(replay, count, *, invalid=(), resets=None):
    for _ in range(count):
        time = replay.total_steps
        states = torch.arange(71, dtype=torch.float32)[None].expand(replay.num_envs, -1).clone()/100
        states += time/10+torch.arange(replay.num_envs)[:, None]
        terminated = torch.tensor([
            time in (resets or {}).get(env, ()) for env in range(replay.num_envs)])
        batch = dict(state=states, next_state=states+.013, terminated=terminated,
            timeout=torch.zeros(replay.num_envs, dtype=torch.bool),
            boundary=torch.full((replay.num_envs,), time % 7 == 0, dtype=torch.bool),
            context=torch.zeros(replay.num_envs, 35))
        for invalid_time, env in invalid:
            if time == invalid_time:
                batch["context"][env, 0] = float("nan")
        before_replay_add(replay)
        accepted = replay.add(batch)
        after_replay_add(replay, batch, accepted)


class TinyDiscriminator(nn.Module):
    def __init__(self, width=11):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(width, 9), nn.ReLU(), nn.Linear(9, 6), nn.ReLU())
        self.amp_linear = nn.Linear(6, 1)

    def forward(self, value):
        return self.amp_linear(self.trunk(value))


class HikingAmpBudgetTest(unittest.TestCase):
    def test_warmup_and_two_three_step_cadence_expose_every_fresh_window_five_times(self):
        replay = make_replay()
        budget = HikingAmpBudget()
        invalid = {(4, 1), (27, 4), (60, 2)}
        add_steps(replay, 8, invalid=invalid)
        exposures = Counter()
        block_ranges = []
        empty_calls = 0
        for iteration in range(24):
            add_steps(replay, 2+iteration % 2, invalid=invalid)
            rollouts = budget.claim(replay, flush=iteration == 23)
            empty_calls += not rollouts
            for rollout in rollouts:
                block_ranges.append((rollout.start, rollout.end))
                for selection in rollout.minibatches():
                    exposures.update(zip(rollout.time[selection].tolist(), rollout.env[selection].tolist()))
        self.assertGreater(empty_calls, 15)
        self.assertEqual(block_ranges, [(0, 24), (24, 48), (48, 68)])
        self.assertEqual(budget.next_time, 68)
        self.assertEqual(budget.invalid_windows, len(invalid))
        self.assertEqual(budget.windows, 68*5-len(invalid))
        for time in range(68):
            for env in range(5):
                self.assertEqual(exposures[time, env], 0 if (time, env) in invalid else 5)
        # The final env is held out only from WM, not from the source AMP rollout.
        self.assertTrue(replay.validation_env_mask[-1])
        self.assertEqual(exposures[0, 4], 5)
        self.assertEqual(budget.claim(replay, flush=True), [])

    def test_minibatch_order_matches_actual_hiking_storage_and_reuses_one_shuffle(self):
        source_path = Path('/tmp/instinct-rl-hiking-reference/instinct_rl/storage/amp_storage.py')
        if not source_path.exists():
            self.skipTest('Pinned Hiking AmpStorage source unavailable')
        tree = ast.parse(source_path.read_text())
        klass = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'AmpStorage')
        method = next(node for node in klass.body if isinstance(node, ast.FunctionDef) and node.name == 'mini_batch_generator')
        namespace = dict(torch=torch)
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(source_path), 'exec'), namespace)
        source = SimpleNamespace(num_envs=8, num_transitions_per_env=24, device='cpu',
            get_minibatch_from_selection=lambda time, env: torch.stack((time, env), -1))
        time = torch.arange(24).repeat_interleave(8)
        env = torch.arange(8).repeat(24)
        rollout = HikingAmpRollout(0, 24, time, env, 0)
        torch.manual_seed(421)
        expected = list(namespace['mini_batch_generator'](source, MINI_BATCHES, LEARNING_EPOCHS))
        torch.manual_seed(421)
        actual = [torch.stack((time[selection], env[selection]), -1) for selection in rollout.minibatches()]
        self.assertEqual(len(actual), 20)
        for result, reference in zip(actual, expected):
            torch.testing.assert_close(result, reference, rtol=0, atol=0)
        for epoch in range(1, 5):
            for minibatch in range(4):
                torch.testing.assert_close(actual[minibatch], actual[epoch*4+minibatch], rtol=0, atol=0)

    def test_tail_and_non_divisible_small_rollout_do_not_drop_or_repeat_windows(self):
        replay = make_replay(num_envs=1, validation_fraction=0.)
        add_steps(replay, 1)
        budget = HikingAmpBudget()
        self.assertEqual(budget.claim(replay), [])
        tail, = budget.claim(replay, flush=True)
        self.assertEqual(len(list(tail.minibatches())), 5)
        add_steps(replay, 24)
        full, = budget.claim(replay)
        self.assertEqual((full.start, full.end), (1, 25))
        self.assertEqual(sum(map(len, full.minibatches())), 24*5)
        uneven = HikingAmpRollout(0, 1, torch.zeros(7, dtype=torch.long), torch.arange(7), 0)
        counts = Counter(int(index) for part in uneven.minibatches() for index in part)
        self.assertEqual(counts, Counter({index: 5 for index in range(7)}))

    def test_overwritten_fresh_samples_and_stale_timestamps_fail_explicitly(self):
        replay = make_replay(capacity=24)
        add_steps(replay, 25)
        with self.assertRaisesRegex(RuntimeError, 'overwritten'):
            HikingAmpBudget().claim(replay)
        replay = make_replay()
        add_steps(replay, 24)
        replay._timestamps[12] = -1
        with self.assertRaisesRegex(RuntimeError, 'timestamps'):
            HikingAmpBudget().claim(replay)

    def test_indices_alone_are_claimed_and_future_additions_remain_fresh(self):
        replay = make_replay()
        add_steps(replay, 26)
        budget = HikingAmpBudget()
        first, = budget.claim(replay)
        self.assertEqual((first.start, first.end), (0, 24))
        self.assertNotIn('state', vars(first))
        self.assertEqual(sum(value.numel() for value in vars(first).values() if torch.is_tensor(value)), 2*24*5)
        add_steps(replay, 22)
        second, = budget.claim(replay)
        self.assertEqual((second.start, second.end), (24, 48))


class HikingAmpFeatureCacheTest(unittest.TestCase):
    def test_source_playback_advances_across_rollouts_and_fills_reset_histories(self):
        replay = make_replay(num_envs=2, validation_fraction=0.)
        add_steps(replay, 48, resets={0: (23, 35)})
        first, second = HikingAmpBudget().claim(replay)
        frames = torch.arange(20000, dtype=torch.float32)[:, None].expand(-1, 67)
        experts = SimpleNamespace(groups=[frames, None], official_reference_frames=10000)
        torch.manual_seed(91)
        playback = HikingEpisodeReference(experts, 2, .02)
        self.assertTrue(torch.all((playback.start >= 0.) & (playback.start < .9*10000/50)))
        initial = playback.history.clone()
        playback.advance(first, replay)
        table = playback.indices.reshape(24, 2, 10)
        torch.testing.assert_close(table[0, :, :9], initial[:, :9])
        torch.testing.assert_close(table[22, 1, 1:], table[23, 1, :-1])
        torch.testing.assert_close(table[23, 0], table[23, 0, -1].expand(10))
        last = table[-1].clone()
        playback.advance(second, replay)
        table = playback.indices.reshape(24, 2, 10)
        torch.testing.assert_close(table[0, :, :-1], last[:, 1:])
        torch.testing.assert_close(table[11, 0], table[11, 0, -1].expand(10))
        torch.testing.assert_close(table[11, 1, :-1], table[10, 1, 1:])
        samples = torch.tensor([0, 7, 30, 43])
        torch.testing.assert_close(playback.select(samples), flatten_history(frames[playback.indices[samples]]))

    def test_done_discriminator_observes_reset_while_wm_keeps_terminal_physics(self):
        replay = make_replay(num_envs=2, validation_fraction=0.)
        add_steps(replay, 24, resets={0: (3,), 1: (23,)})
        rollout, = HikingAmpBudget().claim(replay)
        live = torch.full((2, 71), 42.)
        adapter = SimpleNamespace(amp_features=lambda x: x[..., :67], state=lambda: live)
        terminal = replay.data["next_state"][:24].clone()
        features = HikingAmpFeatureCache(adapter, replay, rollout, 'cpu', 12, post_step_observations=True)
        policy = HikingAmpPolicyCache(features, rollout)
        done = replay.data["terminated"][:24].reshape(-1)
        expected = torch.stack((replay.data["state"][4, 0, :67], live[1, :67]))
        torch.testing.assert_close(policy.values[done], flatten_history(expected[:, None].expand(-1, 10, -1)))
        # Include the returned non-terminal state as well, e.g. after an interval push.
        collected = torch.cat((replay.data["state"][1:24,:,:67], live[None,:,:67]))
        torch.testing.assert_close(features.successors, collected)
        torch.testing.assert_close(replay.data["next_state"][:24], terminal)

    def test_large_feature_chunks_and_flat_policy_cache_preserve_values_and_do_not_rebuild_each_epoch(self):
        replay=make_replay()
        add_steps(replay,24,resets={0:(5,17),3:(12,)})
        rollout,=HikingAmpBudget().claim(replay)
        calls=[]
        def encode(states):
            calls.append(len(states))
            return states[...,:67].square()
        adapter=SimpleNamespace(amp_features=encode)
        reference=HikingAmpFeatureCache(adapter,replay,rollout,'cpu',7)
        calls.clear()
        actual=HikingAmpFeatureCache(adapter,replay,rollout,'cpu',7,feature_chunk_size=64)
        self.assertEqual(calls,[64,56,64,56])
        cache=HikingAmpPolicyCache(actual,rollout,chunk_size=31)
        expected=reference.policy(rollout.time,rollout.env)
        for selection in rollout.minibatches():
            torch.testing.assert_close(cache.select(selection),expected[selection],rtol=0,atol=0)
        self.assertEqual(calls,[64,56,64,56])
        self.assertFalse(cache.values.requires_grad)

    def test_static_reference_cache_preserves_sampled_ends_pairing_and_rng(self):
        frames=torch.randn(45,67,requires_grad=True)
        experts=SimpleNamespace(groups=[frames,None],_hiking_history_ends=torch.tensor([9,12,14,29,32,44]))
        rng=torch.get_rng_state().clone()
        cache=HikingAmpReferenceCache(experts,'cpu',chunk_size=2)
        torch.testing.assert_close(torch.get_rng_state(),rng,rtol=0,atol=0)
        torch.manual_seed(98)
        ends=sample_reference_ends(experts,29)
        expected=reference_features(experts,ends,'cpu')
        for _ in range(5):
            torch.testing.assert_close(cache.select(ends),expected,rtol=0,atol=0)
        self.assertFalse(cache.values.requires_grad)
        self.assertIsNone(frames.grad)

    def test_cached_features_match_existing_history_at_resets_quarantine_and_navigation_edges(self):
        replay = make_replay(capacity=40)
        budget = HikingAmpBudget()
        resets = {0: (2, 27, 44), 1: (12, 34), 2: (22, 40), 3: (30,), 4: (47,)}
        invalid = {(25, 2), (38, 4)}
        adapter = SimpleNamespace(amp_features=lambda states: states[..., :67].square()+.17*states[..., :67])
        add_steps(replay, 24, invalid=invalid, resets=resets)
        budget.claim(replay)
        # Time24's predecessors now live in the existing nine-state overflow.
        add_steps(replay, 40, invalid=invalid, resets=resets)
        rollouts = budget.claim(replay, flush=True)
        self.assertEqual([(r.start, r.end) for r in rollouts], [(24, 48), (48, 64)])
        for rollout in rollouts:
            cache = HikingAmpFeatureCache(adapter, replay, rollout, 'cpu', 11)
            sample = dict(time_index=rollout.time, env_index=rollout.env)
            history = add_sample_history(replay, sample)['amp_state_history']
            successor = replay.data['next_state'][rollout.time % replay.capacity_steps, rollout.env]
            reference = flatten_history(adapter.amp_features(torch.cat((history[:, 1:], successor[:, None]), 1)))
            torch.testing.assert_close(cache.policy(rollout.time, rollout.env), reference, rtol=0, atol=0)
            self.assertFalse(cache.observed.requires_grad)
            self.assertFalse(cache.successors.requires_grad)
            self.assertLess(cache.observed.numel()+cache.successors.numel(), rollout.size*670)

    def test_feature_and_reference_extraction_cannot_accumulate_encoder_or_expert_gradients(self):
        replay = make_replay(num_envs=2, validation_fraction=0.)
        add_steps(replay, 24)
        rollout, = HikingAmpBudget().claim(replay)
        encoder = nn.Linear(71, 67)
        cache = HikingAmpFeatureCache(SimpleNamespace(amp_features=encoder), replay, rollout, 'cpu', 13)
        expert_frames = torch.randn(43, 67, requires_grad=True)
        experts = SimpleNamespace(groups=[expert_frames, None], _hiking_history_ends=torch.tensor([9, 20, 42]))
        ends = sample_reference_ends(experts, rollout.size)
        self.assertTrue(set(ends.tolist()).issubset({9, 20, 42}))
        paired = []
        for part in rollout.minibatches():
            reference = reference_features(experts, ends[part], 'cpu')
            self.assertFalse(reference.requires_grad)
            paired.append((part, reference))
        for epoch in range(1, 5):
            for minibatch in range(4):
                torch.testing.assert_close(paired[minibatch][1], paired[epoch*4+minibatch][1], rtol=0, atol=0)
        model = TinyDiscriminator(670)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        selection = paired[0][0]
        discriminator_step(model, optimizer, [(cache.policy(rollout.time[selection], rollout.env[selection]), paired[0][1])],
            len(selection), gradient_penalty=5., trunk_weight_decay=3e-4, head_weight_decay=.04)
        self.assertTrue(all(parameter.grad is None for parameter in encoder.parameters()))
        self.assertIsNone(expert_frames.grad)


class HikingAmpLogicalStepTest(unittest.TestCase):
    def test_weighted_microbatches_match_full_objective_gradients_and_adamw_state(self):
        torch.manual_seed(81)
        actual = TinyDiscriminator().double()
        expected = copy.deepcopy(actual)
        policy = torch.randn(13, 11, dtype=torch.float64)
        expert = torch.randn(13, 11, dtype=torch.float64)
        optimizer = torch.optim.AdamW(actual.parameters(), lr=1e-4, weight_decay=.01)
        reference_optimizer = torch.optim.AdamW(expected.parameters(), lr=1e-4, weight_decay=.01)
        examples = torch.cat((policy, expert)).requires_grad_(True)
        scores = expected(examples).squeeze(-1)
        prediction = .5*((scores[:13]+1).square().mean()+(scores[13:]-1).square().mean())
        gradients = torch.autograd.grad(scores.sum(), examples, create_graph=True)[0]
        penalty = 5.*gradients.square().sum(-1).mean()
        loss = prediction+penalty+3e-4*sum(p.square().sum() for p in expected.parameters())
        loss = loss+.04*expected.amp_linear.weight.square().sum()
        loss.backward()
        reference_norm = torch.linalg.vector_norm(torch.stack([p.grad.norm() for p in expected.parameters()]))
        reference_optimizer.step()
        pairs = list(zip(policy.split(4), expert.split(4)))
        result = discriminator_step(actual, optimizer, pairs, 13, gradient_penalty=5.,
            trunk_weight_decay=3e-4, head_weight_decay=.04)
        torch.testing.assert_close(result, torch.stack((loss.detach(), penalty.detach(), reference_norm)), rtol=1e-12, atol=1e-12)
        for parameter, reference in zip(actual.parameters(), expected.parameters()):
            torch.testing.assert_close(parameter.grad, reference.grad, rtol=1e-11, atol=1e-12)
            torch.testing.assert_close(parameter, reference, rtol=1e-12, atol=1e-12)
            for key in ('step', 'exp_avg', 'exp_avg_sq'):
                torch.testing.assert_close(optimizer.state[parameter][key], reference_optimizer.state[reference][key],
                    rtol=1e-11, atol=1e-12)

    def test_incomplete_microbatch_coverage_does_not_take_optimizer_step(self):
        model = TinyDiscriminator()
        optimizer = torch.optim.AdamW(model.parameters())
        original = copy.deepcopy(model.state_dict())
        with self.assertRaisesRegex(ValueError, 'cover'):
            discriminator_step(model, optimizer, [(torch.randn(3, 11), torch.randn(3, 11))], 4,
                gradient_penalty=5., trunk_weight_decay=3e-4, head_weight_decay=.04)
        self.assertFalse(optimizer.state)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, original[name], rtol=0, atol=0)


class HikingAmpRunnerBudgetTest(unittest.TestCase):
    def test_runner_only_updates_complete_fresh_blocks_then_flushes_final_tail(self):
        from src.wmac_perceptive.runner import PerceptiveRunner
        runner = object.__new__(PerceptiveRunner)
        runner.cfg = SimpleNamespace(amp_feature_schema='hiking_v1', locomotion_objective='hiking_full_v1',
            hiking_training_version=2, iterations=4, effective_discriminator_batch_size=7,
            gradient_penalty=5., discriminator_trunk_weight_decay=3e-4, discriminator_head_weight_decay=.04)
        runner.device = torch.device('cpu')
        runner.replay = make_replay(num_envs=2, validation_fraction=.5)
        runner.adapter = SimpleNamespace(amp_features=lambda states: states[..., :67])
        runner.experts = SimpleNamespace(groups=[torch.randn(60, 67), None],
            _hiking_history_ends=torch.arange(9, 60))
        runner.discriminator = TinyDiscriminator(670)
        runner.opt_disc = torch.optim.AdamW(runner.discriminator.parameters(), lr=1e-4, weight_decay=.01)
        exposures = Counter()
        construction = Counter()
        original = HikingAmpFeatureCache.policy
        def capture(cache, time, env):
            construction.update(zip(time.tolist(), env.tolist()))
            return original(cache, time, env)
        original_select=HikingAmpPolicyCache.select
        def select(cache, selection):
            exposures.update(zip(cache.rollout.time[selection].tolist(),cache.rollout.env[selection].tolist()))
            return original_select(cache,selection)
        records = []
        with patch.object(HikingAmpFeatureCache, 'policy', capture),patch.object(HikingAmpPolicyCache,'select',select):
            for iteration, count in enumerate((10, 14, 3, 2)):
                runner.iteration = iteration
                add_steps(runner.replay, count)
                records.append(runner.update_discriminator())
        self.assertEqual([row['discriminator_updates_applied'] for row in records], [0, 20, 0, 20])
        self.assertEqual([row['amp_rollout_pending_steps'] for row in records], [10, 0, 3, 0])
        self.assertEqual([row['amp_policy_windows'] for row in records], [0, 48, 0, 10])
        self.assertEqual([row['amp_policy_exposures'] for row in records], [0, 240, 0, 50])
        self.assertEqual(records[-1]['amp_policy_windows_total'], 58)
        self.assertEqual(records[-1]['amp_policy_exposures_total'], 290)
        self.assertEqual(set(exposures), {(time, env) for time in range(29) for env in range(2)})
        self.assertEqual(set(exposures.values()), {5})
        self.assertEqual(set(construction),set(exposures))
        self.assertEqual(set(construction.values()),{1})
        self.assertTrue(all(parameter.grad is None for parameter in runner.discriminator.parameters()))
        self.assertEqual(set(records[0]), set(records[-1]))
        old_budget = runner._hiking_amp_budget
        runner.replay = make_replay(num_envs=2, validation_fraction=.5)
        runner.iteration = 0
        add_steps(runner.replay, 10)
        self.assertEqual(runner.update_discriminator()['amp_rollout_pending_steps'], 10)
        self.assertIsNot(old_budget, runner._hiking_amp_budget)


if __name__ == '__main__':
    unittest.main()
