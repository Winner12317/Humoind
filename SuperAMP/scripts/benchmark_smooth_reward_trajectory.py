"""Paired timing of the actual reward/FK/D path and its input backward.

Uses frozen source D and recorded state tensors, not policy optimization. This
does not time a full WMAC iteration or establish training/behavior improvement.
"""
from pathlib import Path
import argparse
import copy
import json
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.torch import configure_torch_backends
from scripts.analyze_amp_recovery_style import make_cpu_adapter, load_scorer
from scripts.analyze_recovery_gap_model import move_adapter
from scripts.benchmark_centered_amp import paired, gpu_processes
from src.recovery_rewards import configure_recovery_height_reward
from src.wmac.amp_rewards import CenteredAmpReward
from src.wmac.checkpoints import resolve_checkpoint_config
from src.wmac.runner import WmacRunner
from src.wmac.smooth_rewards import SmoothRewardTerms


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--recording', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, default=4096)
    parser.add_argument('--horizon', type=int, default=32)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--precision', choices=('float32', 'tf32', 'bf16'),
                        help='Defaults to the checkpoint saved training precision')
    parser.add_argument('--compile', action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument('--geometry-contact-height-m', type=float)
    parser.add_argument('--geometry-softness-m', type=float)
    parser.add_argument('--repeats', type=int, default=8)
    parser.add_argument('--warmup', type=int, default=2)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    geometry_enabled = args.geometry_contact_height_m is not None
    if geometry_enabled != (args.geometry_softness_m is not None):
        parser.error('Specify both geometry contact height and softness')
    torch.set_num_threads(1)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = resolve_checkpoint_config(payload)
    source_precision = cfg.precision
    cfg.device = str(device)
    if args.precision is not None:
        cfg.precision = args.precision
    configure_torch_backends(allow_tf32=cfg.precision == 'tf32')
    cfg.actor_batch_size, cfg.imagination_horizon = args.batch_size, args.horizon
    cfg.compile = False  # Compilation of the exact trajectory method is below.
    env_cfg = load_env_cfg(cfg.task)
    configure_recovery_height_reward(env_cfg, cfg.recovery_height_reward)
    adapter = make_cpu_adapter(REPO/'src/assets/robots/unitree_g1/xmls/g1.xml')
    adapter.reward_cfg = env_cfg.rewards
    adapter.desired_height = float(env_cfg.scene.entities['robot'].init_state.pos[2])
    move_adapter(adapter, device)
    mj_model = adapter.env.sim.mj_model
    joint_names = [mj_model.joint(int(j)).name for j in adapter.robot.indexing.joint_ids]
    module = SmoothRewardTerms.from_config(env_cfg, joint_names).to(device)
    discriminator, norm, _, _ = load_scorer(args.checkpoint)
    discriminator, norm = discriminator.to(device).eval(), norm.to(device).eval()
    discriminator.requires_grad_(False); norm.requires_grad_(False)
    amp = CenteredAmpReward(cfg.amp_reward_coef, cfg.amp_score_scale, cfg.amp_score_ema_decay).to(device)
    amp.load_state_dict(payload['amp_reward_state'])
    with np.load(args.recording) as archive:
        arrays = {key: archive[key].copy() for key in ('state', 'context', 'action', 'previous_action',
                                                      'command', 'terminated', 'contact_cost', 'actor_history')}
    total_steps, environments = arrays['action'].shape[:2]
    if total_steps <= args.horizon:
        raise ValueError('Recording is shorter than the benchmark horizon')
    generator = np.random.default_rng(49351)
    starts = generator.integers(0, total_steps-args.horizon, size=args.batch_size)
    ids = np.arange(args.batch_size) % environments
    t = starts[None, :] + np.arange(args.horizon+1)[:, None]
    def tensor(array, index=t):
        return torch.from_numpy(array[index, ids[None, :]]).to(device)
    states = tensor(arrays['state']).requires_grad_(True)
    contexts = tensor(arrays['context'])
    actions = tensor(arrays['action'], t[:-1]).requires_grad_(True)
    previous = tensor(arrays['previous_action'], t[:-1]).requires_grad_(True)
    history = arrays['actor_history'].reshape(total_steps, environments, 4, 96)
    older = tensor(history[..., -2, 67:96], t[:-1]).requires_grad_(True)
    command = torch.from_numpy(arrays['command'][starts, ids]).to(device)
    terminated = tensor(arrays['terminated'], t[:-1]).float()
    contact = tensor(arrays['contact_cost'], t[:-1])
    inputs = (states, contexts, actions, previous, command, terminated, contact, older)
    leaves = (states, actions, previous, older)
    functions, runners = {}, {}
    candidate_label = 'geometry' if geometry_enabled else 'smooth'
    compared = (('baseline', 'host_humanup_v1' if geometry_enabled else 'none'),
                (candidate_label, 'host_humanup_v1'))
    for label, mode in compared:
        runner = object.__new__(WmacRunner)
        runner.cfg = copy.deepcopy(cfg); runner.cfg.smooth_reward_mode = mode
        runner.adapter, runner.device = copy.copy(adapter), device
        if label == 'geometry':
            runner.adapter.env.cfg.scene = env_cfg.scene
            runner.adapter.configure_recovery_foot_slip(
                contact_height_m=args.geometry_contact_height_m, softness_m=args.geometry_softness_m)
        runner.discriminator, runner.amp_norm, runner.amp_reward_stats = discriminator, norm, amp
        runner.smooth_reward_terms = module if mode != 'none' else None
        forward = runner._trajectory_rewards
        if args.compile:
            forward = torch.compile(forward, fullgraph=True)
        def backward(forward=forward):
            for value in leaves:
                value.grad = None
            task, style, _ = forward(*inputs)
            (.75*task + .25*style).sum().backward()
        runners[label] = runner
        functions[label] = backward
    concurrency = gpu_processes() if device.type == 'cuda' else None
    started = time.perf_counter()
    timing = paired(functions, device=device, repeats=args.repeats, iterations=1, warmup=args.warmup)
    differences = [new-old for old, new in zip(timing['baseline']['samples_ms'], timing[candidate_label]['samples_ms'])]
    from scripts.benchmark_centered_amp import summarize
    timing[f'paired_{candidate_label}_minus_baseline'] = summarize(differences)
    timing['ratio_of_medians'] = timing[candidate_label]['median_ms']/timing['baseline']['median_ms']
    # Alternating order can finish on baseline, which intentionally has no
    # gradient for the second-previous action. Validate the enabled path.
    functions[candidate_label]()
    scope = {}
    with torch.no_grad():
        base = runners['baseline']._trajectory_rewards(*inputs)
        candidate = runners[candidate_label]._trajectory_rewards(*inputs)
        torch.testing.assert_close(base[1], candidate[1], rtol=0, atol=0)
        torch.testing.assert_close(base[2], candidate[2], rtol=0, atol=0)
        if geometry_enabled:
            from src.wmac.state import recovery_foot_slip_mask
            recovery = recovery_foot_slip_mask(states[:-1], contexts[:-1])
            if not recovery.any() or recovery.all():
                raise ValueError('Timing recording must cover recovery and ordinary walking')
            assert torch.equal(base[0][~recovery], candidate[0][~recovery])
            assert (base[0][recovery] != candidate[0][recovery]).any()
            scope.update(geometry_contact_height_m=args.geometry_contact_height_m,
                         geometry_softness_m=args.geometry_softness_m,
                         geometry_steps_fraction=float(recovery.float().mean()),
                         nonrecovery_task_unchanged=True)
        else:
            assert (candidate[0] <= base[0]).all()
            scope['new_task_reward_no_larger'] = True
    result = dict(schema=('imagined-foot-slip-trajectory-reward-benchmark-v1' if geometry_enabled
                          else 'smooth-reward-full-trajectory-component-benchmark-v1'), **scope,
                  device=str(device), torch=torch.__version__, compiled=args.compile,
                  actor_batch_size=args.batch_size, imagination_horizon=args.horizon,
                  reward_chunk_size=cfg.reward_chunk_size, amp_feature_dim=203,
                  precision=cfg.precision, saved_training_precision=source_precision,
                  precision_matches_saved_training=cfg.precision == source_precision,
                  allow_tf32=bool(torch.backends.cuda.matmul.allow_tf32),
                  dtype=('fp32 state/rewards; bf16 frozen D'
                         if device.type == 'cuda' and cfg.precision == 'bf16'
                         else 'fp32 state/rewards/D; TF32 matmul allowed'
                         if cfg.precision == 'tf32' else 'fp32 state/rewards/D; TF32 disabled'),
                  checkpoint=str(args.checkpoint), recording=str(args.recording),
                  extra_network_forward_passes=0, extra_optimizer_steps=0, physical_sampling=0,
                  wall_seconds=time.perf_counter()-started, timings=timing,
                  finite_input_gradients=all(value.grad is not None and bool(torch.isfinite(value.grad).all()) for value in leaves),
                  discriminator_parameter_gradients_absent=all(p.grad is None for p in discriminator.parameters()),
                  style_and_score_unchanged=True,
                  concurrency_at_start=concurrency,
                  concurrency_at_end=gpu_processes() if device.type == 'cuda' else None,
                  limitations=[
                      'Production trajectory reward/FK/D forward and input backward only; not a whole training iteration.',
                      'No actor/world-model forward, critic fit, simulation or optimizer update is timed.',
                      'Recorded tensors stand in for imagined states; this is a timing and gradient-finiteness check.',
                      'First calls and optional torch.compile costs are reported separately and excluded from steady timing.',
                  ])
    if device.type == 'cuda':
        result['device_name'] = torch.cuda.get_device_name(device)
        result['peak_process_allocated_bytes'] = torch.cuda.max_memory_allocated(device)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(args.output), 'timings': timing,
                      'finite_input_gradients': result['finite_input_gradients']}, indent=2))


if __name__ == '__main__':
    main()
