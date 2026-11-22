"""Describe reward magnitudes on fixed real traces; no fitting or tuning loop.

The result supports explicit initial coefficient units. It does not demonstrate
better policy behavior and must not be used as an optimization-effect claim.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import numpy as np
import torch
import mjlab.tasks  # noqa: F401
import src.tasks  # noqa: F401
from mjlab.tasks.registry import load_env_cfg
from scripts.analyze_amp_recovery_style import make_cpu_adapter
from src.wmac.config import WmacConfig
from src.wmac.smooth_rewards import SmoothRewardTerms


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, default=REPO/'docs/validation/smooth_reward_scale_calibration.json')
    args = parser.parse_args()
    torch.set_num_threads(1)
    adapter = make_cpu_adapter(REPO/'src/assets/robots/unitree_g1/xmls/g1.xml')
    mj_model = adapter.env.sim.mj_model
    names = [mj_model.joint(int(j)).name for j in adapter.robot.indexing.joint_ids]
    kernel = SmoothRewardTerms.from_config(load_env_cfg('Unitree-G1-AMP-Flat-Aligned'), names)
    cfg = WmacConfig()
    coefficients = {
        'action_second_difference': cfg.smooth_action_second_coef,
        'joint_velocity': cfg.smooth_joint_velocity_coef,
        'requested_torque': cfg.smooth_torque_coef,
        'commanded_power': cfg.smooth_power_coef,
    }
    report = dict(schema='fixed-trace-smooth-reward-scale-calibration-v1', training_updates=0,
                  new_physical_rollouts=0, dtype='float32', control_dt=.02,
                  coefficients=coefficients, kernel=kernel.metadata(), scenarios={},
                  limitations=[
                      'Magnitude calibration only, not an optimal-coefficient search or improved-policy trial.',
                      'PD request and command power are analytic proxies, not all-physics-substep actuator force.',
                      'The same fixed trajectories are scored before and after adding costs.',
                      'Raw physical task rewards include existing task coefficients and dt, before outer task weight .75.',
                  ])
    for scenario in ('recovery_seed2026', 'recovery_seed2027', 'walk_seed2026'):
        directory = args.data_root/scenario
        rows, provenance = {}, {}
        for trial in ('a', 'a_repeat', 'b', 'b_repeat'):
            path = directory/f'transitions_{trial}.npz'
            with np.load(path) as archive:
                rows[trial] = {key: archive[key].copy() for key in
                               ('state', 'next_state', 'action', 'previous_action', 'context',
                                'actor_history', 'valid', 'boundary', 'reward')}
            provenance[trial] = dict(path=str(path.resolve()), sha256=sha256(path))
        common = np.logical_and.reduce([row['valid'] & ~row['boundary'] for row in rows.values()])
        time_axis = np.arange(len(common))[:, None]*.02
        mask = common & (time_axis >= .1) & (time_axis < 2.)
        estimates = {}
        for trial, row in rows.items():
            tensor = {k: torch.from_numpy(v) for k, v in row.items() if k not in ('valid', 'boundary')}
            older = tensor['actor_history'].reshape(*mask.shape, 4, 96)[..., -2, 67:96]
            with torch.no_grad():
                raw = kernel(tensor['state'], tensor['next_state'], tensor['action'],
                             tensor['previous_action'], older, tensor['context'])
            values = {key: value.numpy().astype(np.float64) for key, value in raw.items()}
            values['physical_task_reward'] = row['reward'].astype(np.float64)
            for key in raw:
                values['penalty_'+key] = values[key]*coefficients[key]*.02
            values['added_task_penalty'] = sum(values['penalty_'+key] for key in raw)
            # Equal-start means. The four-repeat common mask also fixes exposure.
            denom = mask.sum(0)
            estimates[trial] = {key: np.sum(np.where(mask, value, 0), 0)/denom
                                for key, value in values.items()}
        summary = {}
        for policy in ('a', 'b'):
            summary[policy] = {key: float((estimates[policy][key]+estimates[policy+'_repeat'][key]).mean()*.5)
                               for key in estimates[policy]}
            summary[policy]['added_penalty_over_physical_task_reward'] = (
                summary[policy]['added_task_penalty']/summary[policy]['physical_task_reward'])
            summary[policy]['added_outer_mixed_reward_penalty'] = .75*summary[policy]['added_task_penalty']
        report['scenarios'][scenario] = dict(window_seconds=[.1, 2.], starts=mask.shape[1],
                                            provenance=provenance, summary=summary)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    print(json.dumps({'output': str(args.output), 'coefficients': coefficients,
                      'summary': {key: value['summary'] for key, value in report['scenarios'].items()}}, indent=2))


if __name__ == '__main__':
    main()
