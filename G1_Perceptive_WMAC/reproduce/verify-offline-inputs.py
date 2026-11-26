#!/usr/bin/env python3
"""Verify the complete current default expert pool with network access forbidden."""
from __future__ import annotations
from dataclasses import asdict
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import mujoco
import numpy as np
import torch
from mjlab.entity import Entity

from scripts.train_wmac_perceptive import parse_config
from scripts.prepare_mirrored_motions import main as prepare
from scripts.audit_wmac_motion_data import main as audit
from src.tasks.amp_perceptive.env_cfg import g1_amp_perceptive_hiking_env_cfg
from src.tasks.amp_perceptive.hiking_robot import HIKING_HEAD, HIKING_URDF
from src.wmac_perceptive.expert import PerceptiveExpertPairs
from src.wmac_perceptive.motions import file_sha256

root = Path(__file__).resolve().parents[1]
cfg = parse_config([])
assert cfg.hiking_training_version == 3
assert cfg.standing_expert_source == 'none'
assert cfg.walking_expert_source == 'pre_hiking_local_walk_turn_v2'
assert cfg.action_limit is None and cfg.validation_env_fraction == 0.
output = root/'reproduce'
output.mkdir(exist_ok=True)
(output/'reference-config.json').write_text(json.dumps(asdict(cfg), indent=2)+'\n')
with patch('urllib.request.urlopen', side_effect=AssertionError('Motion preparation must work offline')):
    prepare([])
    env_cfg = g1_amp_perceptive_hiking_env_cfg(
        sensor_protocol=cfg.sensor_protocol, motion_data_protocol=cfg.motion_data_protocol,
        locomotion_objective=cfg.locomotion_objective, hiking_training_version=cfg.hiking_training_version)
    entity = Entity(env_cfg.scene.entities['robot'])
    model = entity.spec.compile()
    hinge = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    default = model.key_qpos[model.key('init_state').id, model.jnt_qposadr[hinge]]
    robot = SimpleNamespace(body_names=entity.body_names, joint_names=entity.joint_names,
        data=SimpleNamespace(default_joint_pos=torch.as_tensor(default, dtype=torch.float32)[None],
                             default_joint_vel=torch.zeros(1, len(hinge))))
    env = SimpleNamespace(step_dt=cfg.dynamics_dt, scene={'robot': robot})
    expert = PerceptiveExpertPairs(cfg, env, 'cpu', cfg.motion_root(root))
    histories = expert.sample_histories(256)
    assert histories.shape == (256, 10, 67) and bool(torch.isfinite(histories).all())
    metadata = expert.metadata
    local = metadata['legacy_walking_reference']
    assert local['source_clips'] == 13 and local['training_clips'] == 26
    assert local['source_frames'] == 389 and local['training_frames'] == 778
    assert metadata['groups'][0]['clip_count'] == 28
    assert metadata['official_history_window_count'] == 37944
    assert metadata['total_history_window_count'] == 38488
    assert metadata['standing_expert_source'] == 'none'
    code = audit(['--training-config',str(output/'reference-config.json'),
                  '--output',str(output/'motion-audit.json')])
    assert code == 0, 'Motion audit failed'
report = dict(snapshot_date='2026-11-20', network_forbidden=True,
    hiking_training_version=cfg.hiking_training_version, sensor_protocol=cfg.sensor_protocol,
    walking_expert_source=cfg.walking_expert_source, standing_expert_source=cfg.standing_expert_source,
    official_reference_frames=expert.official_reference_frames,
    sampled_histories=list(histories.shape), default_expert_metadata=metadata,
    asset_sha256={str(p.relative_to(root)):file_sha256(p) for p in (HIKING_URDF,HIKING_HEAD)})
(output/'offline-input-audit.json').write_text(json.dumps(report, indent=2)+'\n')
print('Offline current default expert pool verified: 28 clips, 38488 valid histories; no standing.')
