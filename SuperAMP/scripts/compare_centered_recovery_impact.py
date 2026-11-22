"""Passive extension of the paired effort probe; never changes control/physics.

Records physical substep root velocity/generalized acceleration, joint velocity,
constraint generalized torque and exact reward-manager control-step terms.
Constraint torque combines contacts/joint constraints and is not a contact-force
sensor. Root angular generalized accelerations are in free-joint coordinates.
"""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts import compare_recovery_effort as paired


class ImpactRecorder(paired.EffortRecorder):
    def __init__(self, env):
        super().__init__(env)
        import torch
        self._reward_rows = []
        self._reward_names = list(env.reward_manager.active_terms)
        self._original_metrics = env.metrics_manager.compute
        original_step = env.sim.step
        joint_v = self.robot.indexing.joint_v_adr.long()
        root_v = self.robot.indexing.free_joint_v_adr.long()

        def step():
            original_step()
            if self.enabled:
                raw = env.sim.data
                self.rows[-1].update(
                    physical_joint_velocity=raw.qvel[:, joint_v].detach().clone(),
                    physical_root_qvel=raw.qvel[:, root_v].detach().clone(),
                    physical_root_qacc=raw.qacc[:, root_v].detach().clone(),
                    constraint_torque=raw.qfrc_constraint[:, joint_v].detach().clone(),
                )

        def metrics(*args, **kwargs):
            result = self._original_metrics(*args, **kwargs)
            if self.enabled:
                # RewardManager stores weighted rates here; CaptureEnv's real
                # reward has already been captured, before reset in this hook.
                terms = env.reward_manager._step_reward.detach().clone() * env.step_dt
                if not torch.allclose(terms.sum(-1), env.reward_buf, rtol=1e-5, atol=1e-6):
                    raise RuntimeError('Reward terms do not reconstruct physical task reward')
                self._reward_rows.append(terms)
            return result

        env.sim.step = step
        env.metrics_manager.compute = metrics

    def begin(self):
        self._reward_rows.clear()
        super().begin()

    def close(self):
        self.env.metrics_manager.compute = self._original_metrics
        super().close()


_original_rollout = paired.rollout
_original_compare = paired.compare


def rollout(env, actor, recorder, steps):
    import numpy as np
    import torch
    arrays, summary = _original_rollout(env, actor, recorder, steps)
    if len(recorder._reward_rows) != steps:
        raise RuntimeError('Reward recording count differs from control steps')
    arrays['reward_terms'] = torch.stack(recorder._reward_rows).cpu().numpy()
    arrays['reward_term_names'] = np.asarray(recorder._reward_names)
    error = float(np.abs(arrays['reward_terms'].sum(-1) - arrays['reward']).max())
    if error > 1e-5:
        raise RuntimeError('Recorded transition task rewards differ from recorded terms')
    summary['reward_reconstruction_max_abs_error'] = error
    return arrays, summary


def compare(args, output):
    import json
    report = _original_compare(args, output)
    report['impact_recorder'] = dict(
        passive=True, network_updates=0, changed_actions=False,
        added_substep_fields=['physical_joint_velocity', 'physical_root_qvel',
                              'physical_root_qacc', 'constraint_torque'],
        added_control_fields=['reward_terms', 'reward_term_names'],
        constraint_torque_is_contact_force=False,
        physical_root_qacc='Free-joint generalized acceleration; first 3 translational, last 3 rotational coordinates',
    )
    (output/'metrics.json').write_text(json.dumps(paired.clean(report), ensure_ascii=False,
                                                 indent=2, allow_nan=False)+'\n')
    return report


if __name__ == '__main__':
    paired.EffortRecorder = ImpactRecorder
    paired.rollout = rollout
    paired.compare = compare
    paired.main()
