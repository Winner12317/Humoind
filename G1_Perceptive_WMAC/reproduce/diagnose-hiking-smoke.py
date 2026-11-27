"""Trace the G1 v5 smoke run before and during the first few learner updates.

Run from G1_Perceptive_WMAC after sourcing reproduce/activate.sh.
This command is for small diagnostic runs only. Host comparisons synchronize CUDA.
"""
import os
import sys
from pathlib import Path

root = Path.cwd()
if not (root / 'scripts/train_wmac_perceptive.py').is_file():
    raise SystemExit('Run this script from G1_Perceptive_WMAC.')
sys.path.insert(0, str(root))

import torch
from scripts import train_wmac_perceptive as train
from src.wmac_perceptive import latent_training
from src.wmac_perceptive.compact_gpu_replay import NoCompleteReferenceWindow
from src.wmac_perceptive.hiking_observations import HIKING_PROPRIO_TERM_WIDTHS
from src.wmac_perceptive.latent_perception import unpack_processed_actor_input
from src.wmac_perceptive.runner import PerceptiveRunner


def report(message):
    print('[trace] ' + message, flush=True)


def check_equal(stage, src, dst):
    if src.is_cuda and not dst.is_cuda:
        torch.cuda.synchronize(src.device)
    a, b = src.detach().cpu(), dst.detach().cpu()
    if a.shape != b.shape or a.dtype != b.dtype:
        raise RuntimeError(f'{stage}: shape or dtype mismatch')
    if a.numel() and not torch.equal(a.contiguous().reshape(-1).view(torch.uint8),
                                     b.contiguous().reshape(-1).view(torch.uint8)):
        raise RuntimeError(f'{stage}: tensor values differ ({src.device} -> {dst.device})')


def check_magnitude(stage, tensor):
    # Deliberately loose limits for scaled network inputs. This reports a fault;
    # it never clips, filters or changes a training sample.
    a = tensor.detach().cpu()
    if not bool(torch.isfinite(a).all()) or a.abs().amax().item() > 1e6:
        index = (torch.isnan(a) | torch.isinf(a) | (a.abs() > 1e6)).nonzero()[0].tolist()
        raise RuntimeError(f'{stage}: sample/index {index} has value {a[tuple(index)].item():.6g}')


def check_proprio(stage, proprio):
    widths = [8 * n for n in HIKING_PROPRIO_TERM_WIDTHS]
    terms = ['angular_velocity', 'gravity', 'command', 'joint_position',
             'joint_velocity', 'previous_action']
    for name, value in zip(terms, proprio.detach().cpu().split(widths, -1)):
        check_magnitude(f'{stage}.{name}', value)


def probe_transfers(sim, learner):
    if sim == learner:
        return
    for src, dst in ((sim, learner), (learner, sim)):
        for width in (29, 768, 4096):
            # Source and expected occupy independent memory, same pattern.
            ref = (torch.arange(64 * width, dtype=torch.int32).reshape(64, width) % 997).float()
            x = ref.to(src)
            for v, expected in ((x, ref), (x[:, ::2], ref[:, ::2])):
                out = v.to(dst, non_blocking=dst.type == 'cuda')
                check_equal(f'copy probe {src} -> {dst}', expected, out)
        report(f'copy probe {src} -> {dst} passed')


def report_windows(replay, length, stage):
    rows = replay._row(torch.arange(replay.size, device=replay.device))
    train_envs = replay.train_env_mask
    invalid = ~replay._valid[rows][:, train_envs].cpu()
    terminated = replay.data['terminated'][rows][:, train_envs].cpu()
    timeout = replay.data['timeout'][rows][:, train_envs].cpu()
    cut = replay._cut_rows[rows][:, train_envs].cpu()
    # The strict sampler needs the next state frame already in replay. The
    # newest transition cannot yet be the final action of a complete window.
    usable = ~(invalid | terminated | timeout | cut)
    streak = torch.zeros(usable.shape[1], dtype=torch.int64)
    longest = windows = 0
    for valid in usable[:-1]:
        streak = torch.where(valid, streak + 1, 0)
        longest = max(longest, int(streak.max()))
        windows += int((streak >= length).sum())
    report(f'{stage}: last row invalid={int(invalid[-1].sum())}, '
           f'terminated={int(terminated[-1].sum())}, timeout={int(timeout[-1].sum())}, '
           f'explicit_cut={int(cut[-1].sum())}; replay totals '
           f'invalid={int(invalid.sum())}, terminated={int(terminated.sum())}, '
           f'timeout={int(timeout.sum())}, explicit_cut={int(cut.sum())}, '
           f'longest usable streak={longest}/{length}, complete windows={windows}')


def report_invalid_fields(batch, replay, accepted):
    rejected = ~accepted
    fields = []
    for name, value in batch.items():
        if not value.is_floating_point():
            continue
        if name in ('depth_stack', 'next_depth_stack'):
            valid = replay._visual_valid(value, replay.spec.include_valid)
        else:
            valid = torch.isfinite(value).reshape(len(accepted), -1).all(-1)
        count = int((~valid & rejected).sum())
        if count:
            fields.append(f'{name}={count}')
    report('rejected environments ' + str(rejected.nonzero().flatten().tolist()) +
           ', affected fields ' + (', '.join(fields) or 'unknown'))


def trace_boundary(runner):
    from_sim, to_sim = original_boundary(runner)

    def receive(source):
        result = from_sim(source)
        step = runner.sim_steps // runner.env.num_envs + 1
        if isinstance(source, dict):
            if step <= 9:
                report(f'step {step} simulator/learner flags: ' +
                       ', '.join(f'{key}={int(source[key].sum())}/{int(result[key].sum())}'
                                 for key in ('terminated', 'timeout')))
            for key, value in source.items():
                if isinstance(value, torch.Tensor):
                    check_equal(f'step {step} transfer.{key}', value, result[key])
        else:
            if source.ndim == 2 and source.shape[-1] == runner.spec.proprio_dim:
                check_proprio(f'step {step} simulator', source)
            check_equal(f'step {step} transfer.observation', source, result)
        return result

    def send(source):
        result = to_sim(source)
        check_equal('action to simulator', source, result)
        check_magnitude('action to simulator', source)
        return result

    return receive, send


def trace_init(runner, *args, **kwargs):
    original_init(runner, *args, **kwargs)
    snapshots = {}
    replay = runner.replay
    add, sample = replay.add, replay.sample_starts
    first_rejection_reported = first_block_reported = False

    def add_checked(batch, *args, **kwargs):
        nonlocal first_rejection_reported
        t = replay.total_steps
        values = batch['actor_history'].detach().cpu().clone()
        check_proprio(f'transition {t} incoming actor_history', values)
        snapshots[t] = values
        snapshots.pop(t - replay.capacity_steps, None)
        accepted = add(batch, *args, **kwargs)
        if t < 9:
            report_windows(replay, runner.cfg.wm_window, f'after vector step {t + 1}')
        if not bool(accepted.all()) and (t < 9 or not first_rejection_reported):
            report_invalid_fields(batch, replay, accepted)
            first_rejection_reported = True
        return accepted

    sample_wm = replay.sample_reference_model_windows

    def sample_wm_checked(*args, **kwargs):
        nonlocal first_block_reported
        try:
            return sample_wm(*args, **kwargs)
        except NoCompleteReferenceWindow:
            if not first_block_reported:
                report_windows(replay, runner.cfg.wm_window,
                               f'first failed WM sample at vector step {replay.total_steps}')
                first_block_reported = True
            raise

    @torch.no_grad()
    def sample_checked(*args, **kwargs):
        batch = sample(*args, **kwargs)
        if 'actor_history' in batch:
            proprio, depth = unpack_processed_actor_input(batch['actor_history'], runner.spec)
            times, envs = batch['time_index'].tolist(), batch['env_index'].tolist()
            expected = torch.stack([snapshots[t][e] for t, e in zip(times, envs)])
            check_equal('sampled proprio history vs. original input', expected, proprio)
            check_proprio('sampled proprio history', proprio)
            check_magnitude('sampled depth', depth)
            encoded = runner.actor.encode_processed(depth)
            check_magnitude('encoded vision', encoded)
            logits = runner.actor.route_latent(proprio, encoded).logits
            check_magnitude('real-sample MoE router logits', logits)
            report(f'before update {runner.iteration+1}: real history matches; '
                   f'max abs proprio={proprio.abs().max().item():.3g}, '
                   f'depth={depth.abs().max().item():.3g}, latent={encoded.abs().max().item():.3g}, '
                   f'router logits={logits.abs().max().item():.3g}')
        return batch

    replay.add, replay.sample_starts = add_checked, sample_checked
    replay.sample_reference_model_windows = sample_wm_checked
    original_step = runner._optimizer_step

    def checked_step(loss, optimizer, module, *args, **kwargs):
        if optimizer is runner.opt_actor:
            if any(p.grad is not None and not torch.isfinite(p.grad).all().item()
                   for p in module.parameters()):
                raise RuntimeError('actor backward produced a non-finite gradient before optimizer.step')
        return original_step(loss, optimizer, module, *args, **kwargs)

    runner._optimizer_step = checked_step
    learn = runner.learn

    def learn_checked():
        start = runner.iteration
        learn()
        if runner.iteration == start:
            raise RuntimeError('Diagnostic completed collection without a single learner update')
        report(f'{runner.iteration - start} learner updates completed')

    runner.learn = learn_checked


if __name__ == '__main__':
    smoke = ['--execution-profile', 'supermimic', '--num-envs', '16', '--iterations', '25',
             '--model-hidden', '128', '128', '--wm-batch-size', '32', '--wm-updates', '1',
             '--actor-batch-size', '16', '--replay-steps', '80', '--latent-recon-images', '16',
             '--collect-steps', '8', '--log-interval', '1', '--save-interval', '5',
             '--no-compile', '--no-cuda-graphs', '--run-name', 'diagnose_nonfinite']
    cfg = train.parse_config(smoke + sys.argv[1:])
    if (cfg.num_envs != 16 or cfg.compile or cfg.cuda_graphs or cfg.resume or
            cfg.hiking_training_version != 5 or not 1 <= cfg.iterations <= 200):
        raise SystemExit('Use a fresh, eager v5 run with 16 environments and up to 200 steps.')
    train.assign_devices(cfg, os.environ)
    sim, learner = torch.device(cfg.sim_device or cfg.device), torch.device(cfg.device)
    probe_transfers(sim, learner)
    original_init, original_boundary = PerceptiveRunner.__init__, latent_training.boundary
    PerceptiveRunner.__init__, latent_training.boundary = trace_init, trace_boundary
    train.main(cfg)
    report(f'completed {cfg.iterations} vector steps')
