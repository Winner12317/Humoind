#!/usr/bin/env python3
"""Verify the default Hiking reference and the optional pool with network access forbidden."""
from __future__ import annotations
import argparse
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
from src.tasks.amp_perceptive.hiking_robot import HIKING_HEAD, HIKING_SHOE_URDF, HIKING_URDF
from src.wmac_perceptive.expert import PerceptiveExpertPairs
from src.wmac_perceptive.motions import file_sha256

root = Path(__file__).resolve().parents[1]
cfg = parse_config([])
assert cfg.hiking_training_version == 5
assert (cfg.collision_horizontal_scale, cfg.depth_geometry, cfg.robot_dynamics_randomization) == (
    .025, 'source_mesh', 'unitree_g1_v1')
assert cfg.standing_expert_source == 'none' and cfg.walking_expert_source == 'none'
assert cfg.sensor_protocol == 'hiking_depth_only_v1' and cfg.robot_variant == 'unshod'
assert cfg.action_limit is None and cfg.validation_env_fraction == 0.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--with-legacy-walking', action='store_true',
                    help='Also check the v3 expert pool using your separately supplied licensed NPZ files')
args = parser.parse_args()
pool_report = {'verified': False, 'reason': 'Optional v3 pool was not requested'}
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
    # Default: Hiking's single reference, played back per environment.
    expert = PerceptiveExpertPairs(cfg, env, 'cpu', cfg.motion_root(root))
    metadata = expert.metadata
    assert expert.official_reference_frames == 18981
    assert tuple(expert.groups[0].shape) == (2*18981, 67) and bool(torch.isfinite(expert.groups[0]).all())
    assert metadata['reference_start_fraction'] == [0., .9] and metadata['mirror_probability'] == .5
    assert metadata['standing_expert_source'] == 'none'
    if args.with_legacy_walking:
        pool_cfg = parse_config(['--hiking-training-version', '3'])
        pool = PerceptiveExpertPairs(pool_cfg, env, 'cpu', pool_cfg.motion_root(root))
        histories = pool.sample_histories(256)
        assert histories.shape == (256, 10, 67) and bool(torch.isfinite(histories).all())
        pool_metadata = pool.metadata
        local = pool_metadata['legacy_walking_reference']
        assert local['source_clips'] == 13 and local['training_clips'] == 26
        assert local['source_frames'] == 389 and local['training_frames'] == 778
        assert pool_metadata['groups'][0]['clip_count'] == 28
        assert pool_metadata['official_history_window_count'] == 37944
        assert pool_metadata['total_history_window_count'] == 38488
        pool_report = dict(verified=True, walking_expert_source=pool_cfg.walking_expert_source,
                           sampled_histories=list(histories.shape), metadata=pool_metadata)
    code = audit(['--training-config',str(output/'reference-config.json'),
                  '--output',str(output/'motion-audit.json')])
    assert code == 0, 'Motion audit failed'
report = dict(snapshot_date='2026-11-26', network_forbidden=True,
    hiking_training_version=cfg.hiking_training_version, sensor_protocol=cfg.sensor_protocol,
    walking_expert_source=cfg.walking_expert_source, standing_expert_source=cfg.standing_expert_source,
    robot_variant=cfg.robot_variant,
    official_reference_frames=expert.official_reference_frames,
    default_expert_metadata=metadata,
    optional_reference_pool=pool_report,
    asset_sha256={str(p.relative_to(root)):file_sha256(p) for p in (HIKING_URDF,HIKING_SHOE_URDF,HIKING_HEAD)})
(output/'offline-input-audit.json').write_text(json.dumps(report, indent=2)+'\n')
print('Offline inputs verified: Hiking reference (18981 frames + mirror); no standing.')
print('Optional v3 walking pool:', 'verified' if args.with_legacy_walking else 'not requested')
