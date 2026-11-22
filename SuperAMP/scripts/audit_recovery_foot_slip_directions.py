"""Frozen one-step action perturbation audit for real and imagined G1 foot slip."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('MUJOCO_GL', 'egl')
os.environ.setdefault('OMP_NUM_THREADS', '1')

import numpy as np
import torch

from scripts.compare_recovery_effort import (
    capture_feet_contact_at_reward_clock, clean, content_hash, file_hash, physical_model_arrays,
)


VERSION = 'recovery-foot-slip-action-direction-audit-v1'
SNAPSHOT_STEPS = (6, 13, 21, 30, 42, 56)
AMPLITUDES = (.05, .10)
DIRECTION_KINDS = ('leg', 'leg', 'leg', 'all', 'all', 'all')
MIN_COST_DELTA = 1e-4


def load_source(path):
    from src.wmac.checkpoints import resolve_checkpoint_config

    payload = torch.load(path, map_location='cpu', weights_only=False, mmap=True)
    config = resolve_checkpoint_config(payload)
    if config.task != 'Unitree-G1-AMP-Flat-Aligned' or config.recovery_counter_semantics != 'hard_forward_v2':
        raise ValueError('Expected flat aligned Wmac source with hard recovery clock')
    return payload, config


def build_physics(seed, count, device, config):
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg
    from scripts.evaluate_wmac import keep_fallen_recovery_frames, seed_all
    from src.recovery_rewards import configure_recovery_height_reward
    from src.wmac.env import CaptureEnv
    from src.wmac.state import StateAdapter

    cfg = load_env_cfg(config.task)
    configure_recovery_height_reward(cfg, config.recovery_height_reward)
    cfg.scene.num_envs, cfg.seed, cfg.curriculum = count, seed, {}
    cfg.events.pop('push_robot', None)
    for observation in cfg.observations.values():
        observation.enable_corruption = False
    cfg.events['init_motion_loader'].params.update(delay_reset_env_ratio=1., max_delay_steps=250)
    command = cfg.commands['twist']
    command.rel_standing_envs = command.rel_heading_envs = 0.
    command.heading_command = False
    command.ranges.heading = None
    command.resampling_time_range = (1e9, 1e9)
    command.ranges.lin_vel_x = (.5, .5)
    command.ranges.lin_vel_y = command.ranges.ang_vel_z = (0., 0.)
    seed_all(seed)
    env = CaptureEnv(cfg=cfg, device=device)
    env.adapter = StateAdapter(env, recovery_counter_semantics=config.recovery_counter_semantics)
    sensor = env.scene['feet_ground_contact']
    capture_feet_contact_at_reward_clock(env)
    starts = keep_fallen_recovery_frames(env)
    seed_all(seed)
    env.reset(seed=seed)
    if sensor.data.found.shape != (count, 2):
        raise RuntimeError(f'Expected exactly two foot contacts: {sensor.data.found.shape}')
    return env, starts


def probe_fields(transition, actual_action):
    if not torch.equal(transition['action'], actual_action):
        raise RuntimeError('Simulator executed a different outer action than the paired intervention')
    return {
        'cost': transition['contact_cost'][..., 0].clone(),
        'contacts': transition['feet_contact'].clone(),
        'valid': ~(transition['boundary'] | transition['terminated'] | transition['timeout']),
    }


def snapshot_sensor_bindings(env):
    """Remember caches and history bindings that mjlab can reassign during a step."""
    return [
        (sensor, sensor._cached_data, sensor._cache_valid,
         dict(sensor._history_state) if getattr(sensor, '_history_state', None) is not None else None)
        for sensor in env.scene.sensors.values()
    ]


def physical_suite(path, seed, args):
    from scripts.check_wmac_physics import PhysicsSnapshot
    from scripts.evaluate_wmac import load_actor
    from src.wmac.smooth_rewards import SmoothRewardTerms

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    payload, config = load_source(path)
    env, starts = build_physics(seed, args.num_envs, args.device, config)
    try:
        actor, _ = load_actor(path, args.device)
        actor.precision = 'float32'
        actor.requires_grad_(False)
        action_limit = float(actor.source_action_limit)
        motor = SmoothRewardTerms.from_env(env)
        if motor.metadata() != payload['smooth_reward_model']:
            raise RuntimeError('Motor context differs from saved training checkpoint')
        action_order = motor.action_to_joint.cpu().numpy()
        leg_slots = action_order[np.array([any(part in name for part in ('hip', 'knee', 'ankle'))
                                           for name in env.scene['robot'].joint_names])]
        if len(leg_slots) != 12:
            raise RuntimeError(f'Expected G1 12 leg joints, got {len(leg_slots)}')
        scenario_rng = np.random.default_rng(seed + 20000)
        sensor = env.scene['feet_ground_contact']
        records = []
        live = torch.ones(args.num_envs, dtype=torch.bool, device=args.device)
        model_sha = content_hash(physical_model_arrays(env))
        with torch.no_grad():
            for step in range(SNAPSHOT_STEPS[-1] + 1):
                action = actor(env.obs_buf['actor']).clamp(-action_limit, action_limit)
                if step not in SNAPSHOT_STEPS:
                    env.step(action)
                    live &= ~env.transition['boundary']
                    continue
                snapshot = PhysicsSnapshot(env)
                if env.adapter.dt != .02 or not torch.equal(snapshot.command, torch.tensor([.5, 0., 0.], device=args.device).expand_as(snapshot.command)):
                    raise RuntimeError('Unexpected control interval or physical command')
                if not torch.equal(snapshot.obs['actor'], env.obs_buf['actor']):
                    raise RuntimeError('Snapshot observability differs before audit')
                pre_contact = (sensor._extract_sensor_data().found > 0).clone()
                sensor_bindings = snapshot_sensor_bindings(env)
                direction = scenario_rng.choice((-1., 1.), (args.num_envs, len(DIRECTION_KINDS), action.shape[-1])).astype(np.float32)
                not_leg = np.setdiff1d(np.arange(action.shape[-1]), leg_slots)
                direction[:, :DIRECTION_KINDS.count('leg'), not_leg] = 0.
                direction = torch.from_numpy(direction).to(args.device)
                actuations = action[:, None, None] + direction[:, None] * torch.tensor(AMPLITUDES, device=args.device)[None, :, None, None]
                low_actuations = action[:, None, None] - direction[:, None] * torch.tensor(AMPLITUDES, device=args.device)[None, :, None, None]
                symmetric = (actuations.abs() <= action_limit).all(-1) & (low_actuations.abs() <= action_limit).all(-1)
                actuations = actuations.clamp(-action_limit, action_limit)
                low_actuations = low_actuations.clamp(-action_limit, action_limit)

                def probe(actuation):
                    snapshot.restore()
                    for detector, cached, valid, history in sensor_bindings:
                        detector._cached_data, detector._cache_valid = cached, valid
                        if history is not None:
                            detector._history_state = dict(history)
                    if not torch.equal(snapshot.obs['actor'], env.obs_buf['actor']):
                        raise RuntimeError('Snapshot actor history failed exact restore')
                    if not torch.equal(sensor._extract_sensor_data().found > 0, pre_contact):
                        raise RuntimeError('Snapshot did not restore earlier foot contact flags')
                    env.step(actuation)
                    return probe_fields(env.transition, actuation)

                base_first = probe(action)
                probes = []
                for scale_index in range(len(AMPLITUDES)):
                    per_scale = []
                    for direction_index in range(len(DIRECTION_KINDS)):
                        pair = []
                        for actuation in (low_actuations[:, scale_index, direction_index], actuations[:, scale_index, direction_index]):
                            pair.append(probe(actuation))
                        per_scale.append(pair)
                    probes.append(per_scale)
                base_second = probe(action)
                if content_hash(physical_model_arrays(env)) != model_sha:
                    raise RuntimeError('Physical parameters changed within a scaffolded snapshot')

                def stack_probes(key):
                    return torch.stack([torch.stack([torch.stack([result[key] for result in pair], 1)
                                                    for pair in per_scale], 1) for per_scale in probes], 1)

                pre_recovery = (snapshot.context[:, 33] > .5) & ((snapshot.context[:, 34] > 0) | (snapshot.state[:, 2] < .55))
                records.append({
                    'state': snapshot.state, 'context': snapshot.context, 'command': snapshot.command,
                    'action': action, 'direction': direction, 'initial_recovery': live & pre_recovery,
                    'pre_contact': pre_contact, 'symmetric_action': symmetric,
                    'first_cost': base_first['cost'], 'second_cost': base_second['cost'],
                    'first_contact': base_first['contacts'], 'second_contact': base_second['contacts'],
                    'base_valid': base_first['valid'] & base_second['valid'],
                    'probe_cost': stack_probes('cost'), 'probe_contact': stack_probes('contacts'),
                    'probe_valid': stack_probes('valid'),
                })
                live &= base_second['valid']
                print(f'{path.name} seed {seed} step {step}: {int((pre_recovery & live).sum())} recovering', flush=True)
        arrays = {key: torch.stack([record[key] for record in records]).cpu().numpy() for key in records[0]}
        return arrays, {'checkpoint': str(path), 'sha256': file_hash(path), 'iteration': int(payload['iteration']),
                        'seed': seed, 'num_envs': args.num_envs, 'motion_frames_low_or_tilted': starts,
                        'physical_model_unchanged_sha256': model_sha, 'snapshot_steps': SNAPSHOT_STEPS,
                        'restore_method': 'PhysicsSnapshot: MJWarp arrays, managers, commands, history, clocks and RNG',
                        'repeat_control': 'One unperturbed step before and after all symmetric probes; each independently restored.'}
    finally:
        env.close()


def score_arrays(arrays, path, args):
    from scripts.analyze_amp_recovery_style import make_cpu_adapter
    from src.wmac.networks import WorldModel
    from src.wmac.foot_slip import G1FootSlipGeometry

    payload, config = load_source(path)
    torch.backends.cuda.matmul.allow_tf32 = (config.precision == 'tf32')
    adapter = make_cpu_adapter(ROOT / 'src/assets/robots/unitree_g1/xmls/g1.xml')
    if args.device != 'cpu':
        from scripts.analyze_recovery_gap_model import move_adapter
        move_adapter(adapter, args.device)
    geometry_args = (adapter.env.sim.mj_model, adapter.body_ids, adapter.foot_ids)
    geometry_kwargs = dict(
        contact_height_m=args.contact_height_m, softness_m=args.softness_m,
        command_threshold=.1, device=args.device)
    geometry = G1FootSlipGeometry.from_model(*geometry_args, **geometry_kwargs)
    model = WorldModel(context_dim=35, hidden=config.model_hidden, activation='gelu', output_zero_init=False)
    model.load_state_dict(payload['world_model'], strict=True)
    model = model.to(args.device).eval().requires_grad_(False)
    model.precision = config.precision
    inputs = {key: torch.from_numpy(arrays[key]).flatten(0, 1).to(args.device)
              for key in ('state', 'action', 'context', 'command', 'direction')}
    action = inputs['action'].detach().requires_grad_(True)
    state_next, original_contact_cost = model.step_with_contact(inputs['state'], action, inputs['context'])
    analysis = adapter.analyze_state(state_next)
    geometric_cost = geometry(analysis, inputs['command'])
    full_cost = None
    if args.compare_instantamp:
        instantamp = G1FootSlipGeometry.from_model(*geometry_args, **geometry_kwargs, differentiate_support=True)
        full_cost = instantamp(analysis, inputs['command'])
        if not torch.equal(geometric_cost, full_cost):
            raise RuntimeError('Detached and full support gradients must use identical forward costs')
    learned, = torch.autograd.grad(original_contact_cost[..., 0].sum(), action, retain_graph=True)
    candidate, = torch.autograd.grad(geometric_cost.sum(), action, retain_graph=full_cost is not None)
    gradients = {'learned': learned, 'geometry': candidate}
    if full_cost is not None:
        gradients['instantamp'], = torch.autograd.grad(full_cost.sum(), action)
    data = {}
    shape = (*arrays['initial_recovery'].shape, len(DIRECTION_KINDS))
    for key, gradient in gradients.items():
        if not gradient.isfinite().all():
            raise RuntimeError(f'Nonfinite {key} action derivative')
        data[key] = (gradient[:, None] * inputs['direction']).sum(-1).reshape(shape).detach().cpu().numpy()
    return data


def phase_masks(arrays):
    before = arrays['pre_contact'][:, :, None, None, :]
    base = arrays['first_contact'][:, :, None, None, :]
    same_pair = np.all(arrays['probe_contact'] == base[..., None, :], axis=(-1, -2))
    same_before = np.all(before == base, axis=-1)
    repeat_contact_ok = np.all(arrays['first_contact'] == arrays['second_contact'], axis=-1)
    base_noise = np.abs(arrays['second_cost'] - arrays['first_cost'])
    true_delta = np.diff(arrays['probe_cost'], axis=-1)[..., 0]
    eligible = (arrays['initial_recovery'] & arrays['base_valid'] & repeat_contact_ok)[:, :, None, None]
    eligible = eligible & arrays['symmetric_action'] & arrays['probe_valid'].all(-1)
    resolved = np.abs(true_delta) > np.maximum(MIN_COST_DELTA, 3 * base_noise)[:, :, None, None]
    stable = same_pair & same_before
    contact = base.any(-1)
    return true_delta, eligible, resolved, {'all': np.ones_like(eligible), 'stable_support': stable & contact,
                                             'switching': ~stable, 'stable_airborne': stable & ~contact,
                                             'action_changes_contact': ~same_pair}


def bootstrap_contrast(first, second):
    if len(first) < 2:
        return None
    rng = np.random.default_rng(20261118)
    selected = rng.integers(len(first), size=(10000, len(first)))
    return np.quantile((second - first)[selected].mean(1), [.025, .975]).tolist()


def stats_for(cases, scale_index, name):
    per_start = {key: [] for key in cases[0][1]}
    resolved_count, eligible_count = 0, 0
    sign_inactive = {key: 0 for key in per_start}
    for arrays, derivatives in cases:
        delta, eligible, resolved, masks = phase_masks(arrays)
        eligible = (eligible & masks[name])[:, :, scale_index]
        mask = eligible & resolved[:, :, scale_index]
        eligible_count += np.count_nonzero(eligible)
        resolved_count += np.count_nonzero(mask)
        counts = mask.sum((0, 2))
        sign_true = np.sign(delta[:, :, scale_index])
        for key, signed_slope in derivatives.items():
            prediction = np.sign(np.where(np.abs(signed_slope) > 1e-8, signed_slope, 0.))
            sign_inactive[key] += np.count_nonzero(mask & (prediction == 0))
            correct = (prediction == sign_true) & mask
            per_start[key].extend((correct.sum((0, 2))[counts > 0] / counts[counts > 0]).tolist())
    counts = len(per_start['learned'])
    if not counts:
        return {'eligible_probes': int(eligible_count), 'resolved_probes': 0, 'paired_starts': 0}
    scores = {key: np.asarray(values) for key, values in per_start.items()}
    report = {'eligible_probes': int(eligible_count), 'resolved_probes': int(resolved_count), 'paired_starts': counts,
              **{f'{key}_agreement': float(values.mean()) for key, values in scores.items()},
              'tie_predictions_scored_incorrect': {key: int(value) for key, value in sign_inactive.items()}}
    comparisons = [('geometry', 'learned')]
    if 'instantamp' in scores:
        comparisons.extend([('instantamp', 'learned'), ('instantamp', 'geometry')])
    for candidate, baseline in comparisons:
        key = f'{candidate}_minus_{baseline}'
        report[key] = float((scores[candidate] - scores[baseline]).mean())
        report[f'{key}_bootstrap95'] = bootstrap_contrast(scores[baseline], scores[candidate])
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, nargs='+')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--seeds', type=int, nargs='+', default=[2028, 2029])
    parser.add_argument('--num-envs', type=int, default=32)
    parser.add_argument('--physics-only', action='store_true')
    parser.add_argument('--score-existing', action='store_true')
    parser.add_argument('--contact-height-m', type=float)
    parser.add_argument('--softness-m', type=float)
    parser.add_argument('--compare-instantamp', action='store_true',
                        help='Compare full and detached support gradients at the upstream 1 mm / 1 mm setting')
    args = parser.parse_args()
    if args.physics_only and args.score_existing:
        parser.error('--physics-only and --score-existing are mutually exclusive')
    if not args.score_existing and not args.checkpoint:
        parser.error('--checkpoint is required for physical collection')
    if args.compare_instantamp:
        if args.contact_height_m not in (None, .001) or args.softness_m not in (None, .001):
            parser.error('--compare-instantamp uses 1 mm for both contact height and softness')
        args.contact_height_m = args.softness_m = .001
    if not args.physics_only and (args.contact_height_m is None or args.softness_m is None):
        parser.error('Scoring requires explicit calibrated --contact-height-m and --softness-m')
    if args.num_envs < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error('Require positive environments and distinct seeds')
    torch.set_num_threads(1)

    output = args.output.expanduser().resolve()
    index_path = output / 'physical_manifest.json'
    if not args.score_existing:
        output.mkdir(parents=True, exist_ok=False)
        index = []
        for checkpoint in args.checkpoint:
            checkpoint = checkpoint.expanduser().resolve(strict=True)
            for seed in args.seeds:
                arrays, item = physical_suite(checkpoint, seed, args)
                filename = f"physical_{item['iteration']}_seed{seed}.npz"
                artifact = output / filename
                np.savez_compressed(artifact, **arrays)
                index.append({**item, 'file': filename, 'file_sha256': file_hash(artifact)})
                index_path.write_text(json.dumps(clean(index), indent=2, allow_nan=False) + '\n')
    else:
        index = json.loads(index_path.read_text())
    if args.physics_only:
        return

    results = {}
    cases_by_iteration = {}
    for item in index:
        path = Path(item['checkpoint'])
        artifact = output / item['file']
        if file_hash(path) != item['sha256'] or file_hash(artifact) != item['file_sha256']:
            raise RuntimeError('Source checkpoint or physical packet changed')
        with np.load(artifact, allow_pickle=False) as packet:
            arrays = dict(packet)
        scores = score_arrays(arrays, path, args)
        cases_by_iteration.setdefault(item['iteration'], []).append((arrays, scores))
    labels = ('all', 'stable_support', 'switching', 'stable_airborne', 'action_changes_contact')
    for iteration, cases in cases_by_iteration.items():
        results[str(iteration)] = {str(amplitude): {name: stats_for(cases, scale_index, name) for name in labels}
                                   for scale_index, amplitude in enumerate(AMPLITUDES)}
    report = {'schema': VERSION, 'physical_manifest': str(index_path), 'position': 'one real step from identical snapshots',
              'phase': 'Only valid initial recovery (delayed env with current clock>0 or current pelvis z<.55); '
                       'stable support: pre, both unperturbed and plus/minus contacts agree and at least one support; '
                       'switching: contact state differs between pre/base/plus/minus; '
                       'action_changes_contact additionally requires plus or minus to change the base footprint.',
              'observation_noise': False, 'push': False, 'command': [.5, 0., 0.],
              'geometry_height_m': args.contact_height_m, 'geometry_softness_m': args.softness_m,
              'geometry_contact_gradient_detached': True, 'action_amplitudes': AMPLITUDES,
              'direction_kinds': DIRECTION_KINDS, 'direction_source': 'fixed seed Rademacher ±1; per-coordinate scale, symmetric unclipped pairs only',
              'true_delta_threshold': f'abs raw slip plus-minus > max({MIN_COST_DELTA}, 3 times absolute repeat jitter)',
              'score': 'sign(dcost_model/da at base dot direction) versus sign of physical raw cost difference between plus/minus',
              'aggregation': 'Equal initial states within each checkpoint after averaging eligible probes; paired bootstrap over initial states.',
              'limitations': ['One-step conditional direction, not long-horizon BPTT; different perturbations from the same initial state are not independent.',
                              'Switches are nonsmooth and can have a finite jump; the two perturbation radii are reported separately.',
                              'A deterministic phase boundary or modeled no-gradient can still produce a wrong prediction.'],
              'results': results}
    if args.compare_instantamp:
        report['instantamp_contact_gradient_detached'] = False
        report['detached_full_forward_costs_identical'] = True
    report_path = output / ('direction_results_instantamp.json' if args.compare_instantamp else 'direction_results.json')
    if report_path.exists():
        report_path = report_path.with_stem(f'{report_path.stem}_h{args.contact_height_m:g}_tau{args.softness_m:g}')
    with report_path.open('x') as stream:
        json.dump(clean(report), stream, indent=2, allow_nan=False)
        stream.write('\n')
    print(report_path)


if __name__ == '__main__':
    main()
