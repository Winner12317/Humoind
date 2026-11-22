"""Read-only time/term input-layout forensics from an old PPO normalizer.

This infers checkpoint input semantics, not the identity of a historical
site-packages file or whether every historical training stage used that file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

TERMS = [('base_ang_vel', 3), ('projected_gravity', 3), ('command', 3),
         ('joint_pos', 29), ('joint_vel', 29), ('actions', 29)]
CRITIC_TERMS = TERMS + [('base_lin_vel', 3), ('body_pos_b', 39), ('body_ori_b', 78)]
HISTORY = 4
FRAME = sum(width for _, width in TERMS)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def decode(array, ordering, terms=TERMS):
    array = np.asarray(array).reshape(-1)
    frame = sum(width for _, width in terms)
    if array.size != HISTORY * frame:
        raise ValueError(f'Expected four {frame}-dimensional history frames')
    if ordering == 'time':
        return array.reshape(HISTORY, frame)
    pieces = []
    start = 0
    for _, width in terms:
        pieces.append(array[start:start + HISTORY * width].reshape(HISTORY, width))
        start += HISTORY * width
    return np.concatenate(pieces, axis=-1)


def layout_metrics(mean, var, ordering, terms=TERMS):
    m, v = decode(mean, ordering, terms), decode(var, ordering, terms)
    normalized_lag_deviation = m.std(axis=0) / np.sqrt(v.mean(axis=0) + 1e-12)
    log_v = np.log(v + 1e-12)
    result = {
        'mean_lag_dispersion_normalized_rms': float(np.sqrt(np.mean(normalized_lag_deviation ** 2))),
        'log_variance_lag_dispersion_rms': float(np.sqrt(np.mean((log_v-log_v.mean(axis=0))**2))),
        'projected_gravity_mean_by_history': m[:, 3:6].tolist(),
        'projected_gravity_second_norm_moment_by_history': (v[:, 3:6] + m[:, 3:6]**2).sum(axis=-1).tolist(),
        'command_mean_by_history': m[:, 6:9].tolist(),
        'term_mean_variance_across_history': {},
    }
    start = 0
    for name, width in terms:
        result['term_mean_variance_across_history'][name] = float(v[:, start:start+width].mean())
        start += width
    return result


def time_to_term(values):
    frames = values.reshape(-1, HISTORY, FRAME)
    pieces = []
    start = 0
    for _, width in TERMS:
        pieces.append(frames[..., start:start+width].reshape(len(frames), -1))
        start += width
    return np.concatenate(pieces, axis=-1)


def functional_check(checkpoint, recording, size):
    """Replay-consistency only: the recording itself used a time-major wrapper."""
    from scripts.evaluate_wmac import load_actor
    actor, kind = load_actor(checkpoint, 'cpu')
    actor.precision = 'float32'
    actor.eval()
    with np.load(recording, allow_pickle=False) as z:
        history = np.asarray(z['actor_history']).reshape(-1, HISTORY * FRAME)
        raw_action = np.asarray(z['raw_action']).reshape(-1, 29)
        valid = np.asarray(z['valid']).reshape(-1).astype(bool)
    available = np.flatnonzero(valid)
    rng = np.random.default_rng(17311)
    ids = np.sort(rng.choice(available, min(size, len(available)), replace=False))
    values = history[ids]
    with torch.no_grad():
        time_action = actor(torch.from_numpy(values).float()).numpy()
        term_action = actor(torch.from_numpy(time_to_term(values)).float()).numpy()
    delta = time_action - raw_action[ids]
    counterfactual = term_action - time_action
    return {
        'recording': str(recording), 'recording_sha256': sha256(recording),
        'policy_kind': kind, 'sampled_rows': len(ids),
        'time_input_vs_recorded_action_max_abs': float(np.abs(delta).max()),
        'time_input_vs_recorded_action_rms': float(np.sqrt(np.mean(delta.astype(np.float64)**2))),
        'permuted_term_input_vs_time_action_rms': float(np.sqrt(np.mean(counterfactual.astype(np.float64)**2))),
        'limitation': 'This confirms current replay/input consistency only. The recording was itself generated using a time-major wrapper, so it is not independent historical training provenance.',
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, default=Path('/home/autel/amp_60000.pt'))
    p.add_argument('--output', type=Path, default=ROOT/'docs/validation/ppo60000_history_stats_20261117.json')
    p.add_argument('--recording', type=Path)
    p.add_argument('--samples', type=int, default=1024)
    args = p.parse_args()
    torch.set_num_threads(4)
    payload = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    norm = payload['obs_norm_state_dict']
    mean = norm['_mean'].detach().numpy().reshape(-1).astype(np.float64)
    var = norm['_var'].detach().numpy().reshape(-1).astype(np.float64)
    std = norm['_std'].detach().numpy().reshape(-1).astype(np.float64)
    if not np.isfinite(mean).all() or not np.isfinite(var).all() or not (var > 0).all():
        raise ValueError('Normalizer has unusable statistics')
    result = {
        'schema': 'ppo-history-order-checkpoint-forensics-v1',
        'checkpoint': str(args.checkpoint.resolve()), 'checkpoint_sha256': sha256(args.checkpoint),
        'checkpoint_iteration': payload.get('iter'),
        'normalizer': {'shape': list(norm['_mean'].shape), 'count': int(norm['count']),
                       'std_vs_sqrt_variance_max_error': float(np.abs(std-np.sqrt(var)).max())},
        'assumed_terms': [{'name': name, 'width': width} for name, width in TERMS],
        'history_length': HISTORY, 'frame_width': FRAME,
        'unit_projected_gravity_plus_uniform_005_noise_expected_second_norm_moment': 1.0025,
        'layouts': {order: layout_metrics(mean, var, order) for order in ('time', 'term')},
        'conclusion': 'Stored actor normalizer statistics strongly support time-major input semantics. They do not identify the exact historical patch file or prove all 60000 iterations used one unchanged environment.',
        'limitations': [
            'The checkpoint has no explicit history_ordering or full environment config.',
            'Inference assumes the original six actor terms, units and dimensions. A different custom observation transform would require a different interpretation.',
            'Saved count=98304 is only the normalizer count. Do not identify it with the whole 60000-iteration training history.',
            'Statistics could have been imported or frozen during a resume; no historical file identity follows from them.',
            'Lag-consistency scores are descriptive, not a calibrated hypothesis probability.',
        ],
    }
    if 'privileged_obs_norm_state_dict' in payload:
        critic_norm = payload['privileged_obs_norm_state_dict']
        critic_mean = critic_norm['_mean'].detach().numpy().reshape(-1).astype(np.float64)
        critic_var = critic_norm['_var'].detach().numpy().reshape(-1).astype(np.float64)
        critic_std = critic_norm['_std'].detach().numpy().reshape(-1).astype(np.float64)
        result['critic_normalizer'] = {
            'shape': list(critic_norm['_mean'].shape), 'count': int(critic_norm['count']),
            'assumed_terms': [{'name': name, 'width': width} for name, width in CRITIC_TERMS],
            'frame_width': sum(width for _, width in CRITIC_TERMS),
            'std_vs_sqrt_variance_max_error': float(np.abs(critic_std-np.sqrt(critic_var)).max()),
            'unit_projected_gravity_without_noise_expected_second_norm_moment': 1.0,
            'layouts': {order: layout_metrics(critic_mean, critic_var, order, CRITIC_TERMS)
                        for order in ('time', 'term')},
        }
    if args.recording:
        result['replay_consistency_check'] = functional_check(args.checkpoint, args.recording, args.samples)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    md = args.output.with_suffix('.md')
    t, u = result['layouts']['time'], result['layouts']['term']
    lines = ['# PPO60000 观测历史：权重统计取证', '',
             '本报告只读取 `/home/autel/amp_60000.pt`，不修改权重、训练或环境。', '',
             '## 可确认的统计事实', '',
             f"- 权重 iter={result['checkpoint_iteration']}，Actor输入384维。保存有观测归一化均值、方差、标准差，count={int(norm['count'])}。", 
             '- 原平地Actor每帧为96维：角速度3、重力3、指令3、关节位置29、关节速度29、上一动作29；4帧合计384。',
             '- 将保存统计按四个连续96维帧解释时，每帧都恢复出相同物理类型的分组。', '',
             '| 解释方式 | 标准化均值跨历史差异RMS | log方差跨历史差异RMS | 重力二阶范数矩（四帧） |',
             '| --- | ---: | ---: | --- |',
             f"| time | {t['mean_lag_dispersion_normalized_rms']:.6f} | {t['log_variance_lag_dispersion_rms']:.6f} | {', '.join(f'{x:.6f}' for x in t['projected_gravity_second_norm_moment_by_history'])} |",
             f"| term | {u['mean_lag_dispersion_normalized_rms']:.6f} | {u['log_variance_lag_dispersion_rms']:.6f} | {', '.join(f'{x:.6f}' for x in u['projected_gravity_second_norm_moment_by_history'])} |", '',
             '原配置的projected_gravity是单位重力投影，逐分量加入U(−0.05,0.05)噪声时，理想E[||g||²]为1.0025。time解释与此非常吻合，term解释不吻合。',
             '均值也在索引5、101、197、293重复出现重力z约−0.8，在索引6、102、198、294重复出现前向指令约0.7。', '',
             '## 结论与边界', '',
             '**权重本身强烈支持它使用时间优先、每帧完整96维的观测输入语义。**这比只看到当前仓库配置更直接。',
             '但checkpoint没有保存history_ordering、环境源码摘要或安装补丁记录，因此不能仅凭这些统计证明历史site-packages文件就是仓库某个补丁字节版本，或全部60000轮从未换过环境。',
             'count=98304只是保存的归一化计数，不应把它当成完整训练长度，不能由它推算60000轮历史。',
             '当前回放在time输入下成功，仅是额外运行一致性支持；如果回放本身由time布局产生，动作复现并不是独立历史证据。', '',
             f"JSON记录：`{args.output.name}`；脚本：`scripts/audit_ppo_history_stats.py`。"]
    if 'critic_normalizer' in result:
        c = result['critic_normalizer']
        ct, cu = c['layouts']['time'], c['layouts']['term']
        lines += ['', '## Critic 的独立统计交叉核对', '',
                  f"Critic归一化形状为{c['shape']}，即4×216，count={c['count']}。原Critic关闭观测噪声，重力二阶范数矩应接近1。", '',
                  '| 解释方式 | log方差跨历史差异RMS | 重力二阶范数矩（四帧） |',
                  '| --- | ---: | --- |',
                  f"| time | {ct['log_variance_lag_dispersion_rms']:.6f} | {', '.join(f'{x:.9f}' for x in ct['projected_gravity_second_norm_moment_by_history'])} |",
                  f"| term | {cu['log_variance_lag_dispersion_rms']:.6f} | {', '.join(f'{x:.9f}' for x in cu['projected_gravity_second_norm_moment_by_history'])} |", '',
                  'Actor与Critic两套保存统计均指向time布局。两种归一化器count不同是保存事实；不能据此自行编造重置、冻结或训练阶段历史。']
    if 'replay_consistency_check' in result:
        q = result['replay_consistency_check']
        lines += ['', '## 当前记录的CPU动作一致性（非历史直接证明）', '',
                  f"使用{q['sampled_rows']}个保存的真实观测，time输入相对记录动作最大误差{q['time_input_vs_recorded_action_max_abs']:.6g}、RMS {q['time_input_vs_recorded_action_rms']:.6g}；把同一观测重新排列为term输入后，动作相对time输入RMS变化为{q['permuted_term_input_vs_time_action_rms']:.6g}。"]
    md.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'json': str(args.output), 'md': str(md), 'layouts': result['layouts'],
                      'critic_normalizer': result.get('critic_normalizer'),
                      'replay_consistency_check': result.get('replay_consistency_check')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
