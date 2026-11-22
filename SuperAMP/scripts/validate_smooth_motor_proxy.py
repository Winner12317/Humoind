"""Validate the parameter-free PD/motor proxy against passive physical records.

This is offline CPU arithmetic only. First-substep requested/clipped torque
should match exactly; commanded effort is deliberately not physical force after
integration. No simulator, policy, world-model update or optimizer is created.
"""
from pathlib import Path
import argparse
import hashlib
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch
import mjlab.tasks
import src.tasks
from mjlab.tasks.registry import load_env_cfg
from scripts.analyze_amp_recovery_style import make_cpu_adapter
from src.wmac.smooth_rewards import SmoothRewardTerms


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--recordings', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cfg = load_env_cfg('Unitree-G1-AMP-Flat-Aligned')
    adapter = make_cpu_adapter(ROOT/'src/assets/robots/unitree_g1/xmls/g1.xml')
    model = adapter.env.sim.mj_model
    names = [model.joint(int(index)).name for index in adapter.robot.indexing.joint_ids]
    module = SmoothRewardTerms.from_config(cfg, names)
    result = dict(schema='smooth-reward-motor-proxy-validation-v1', device='cpu',
                  simulator_steps=0, optimizer_steps=0, new_networks=0,
                  motor_metadata=module.metadata(), recordings={})
    total = 0
    with torch.no_grad():
        for path in args.recordings:
            with np.load(path, allow_pickle=False) as data:
                state, action, context = (torch.from_numpy(data[key]) for key in ('state', 'action', 'context'))
                count = data['qacc'].shape[0] // data['state'].shape[0]
                if count != 4:
                    raise ValueError('This validation expects the retained four physical substeps')
                expected_request = torch.from_numpy(data['request'][::count])
                expected_clip = torch.from_numpy(data['clipped'][::count])
                physical_force = torch.from_numpy(data['force'][::count])
                request, clipped, command = module.motor_effort(state, action, context)
                request_error = request - expected_request
                clip_error = clipped - expected_clip
                request_max = float(request_error.abs().max())
                clip_max = float(clip_error.abs().max())
                if request_max > 2e-5 or clip_max > 2e-5:
                    raise RuntimeError(f'Motor proxy first-substep mismatch: {path}: {request_max}, {clip_max}')
                result['recordings'][str(path)] = dict(
                    sha256=digest(path), control_transitions=int(state.shape[0] * state.shape[1]),
                    requested_torque_max_abs_error_Nm=request_max,
                    requested_torque_rmse_Nm=float(request_error.square().mean().sqrt()),
                    clipped_torque_max_abs_error_Nm=clip_max,
                    clipped_torque_rmse_Nm=float(clip_error.square().mean().sqrt()),
                    commanded_proxy_minus_after_integration_force_rmse_Nm=float((command-physical_force).square().mean().sqrt()),
                    command_is_not_actual_actuator_force=True,
                )
                total += state.shape[0] * state.shape[1]
    result['validated_control_transitions'] = total
    result['status'] = 'pass'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+'\n')
    print(json.dumps(dict(status='pass', files=len(args.recordings), control_transitions=total,
                          output=str(args.output)), ensure_ascii=False))


if __name__ == '__main__':
    main()
