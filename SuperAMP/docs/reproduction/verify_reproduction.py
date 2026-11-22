"""Verify the portable flat WMAC source snapshot and active environment."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
from importlib import import_module
from importlib.metadata import distribution, version
import json
from pathlib import Path
import subprocess
import sys


def reproduction_config():
    from src.wmac.config import WmacConfig

    cfg = WmacConfig.for_profile('supermimic_official')
    cfg.amp_reward_mapping = 'instantamp_centered_tanh_v1'
    cfg.smooth_reward_mode = 'host_humanup_v1'
    cfg.smooth_action_second_coef = 0.0175
    cfg.smooth_joint_velocity_coef = 0.000625
    cfg.smooth_torque_coef = 0.000625
    cfg.smooth_power_coef = 0.00003125
    cfg.imagined_recovery_foot_slip = 'learned_v1'
    cfg.discriminator_updates = 2
    cfg.discriminator_batch_size = 25600
    cfg.run_name = 'local_4096_ppo_amp_budget'
    cfg.validate()
    return cfg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--environment', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads((root/'docs/reproduction/source_manifest.json').read_text())
    head = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    if head != manifest['base_commit']:
        raise SystemExit(f'Expected public base HEAD {manifest["base_commit"]}, found {head}')
    mismatches = []
    for relative, expected in manifest['files'].items():
        path = root/relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() and not path.is_symlink() else 'MISSING_OR_SYMLINK'
        if actual != expected:
            mismatches.append((relative, actual, expected))
    if mismatches:
        raise SystemExit(json.dumps(mismatches, indent=2))
    log_root = root/'logs'
    if log_root.is_symlink() or (log_root/'wmac').is_symlink():
        raise SystemExit('Reproduction must use local output directories, not shared-disk symlinks')
    print(f'Source verified: {len(manifest["files"])} files, {len(manifest["motion_assets"])} original motion clips; base={head}')
    if not args.environment:
        return
    if sys.version_info[:3] != (3, 11, 17):
        raise SystemExit(sys.version)
    sys.path.insert(0, str(root))
    from packaging.requirements import Requirement
    checked = 0
    for line in (root/'docs/reproduction/requirements-linux-py311-cu126.txt').read_text().splitlines():
        line = line.strip()
        if not line or line.startswith(('#', '--')):
            continue
        requirement = Requirement(line)
        installed = version(requirement.name)
        if not requirement.specifier.contains(installed):
            raise SystemExit(f'Dependency mismatch: {requirement.name}: {installed} != {requirement.specifier}')
        checked += 1
    if version('pip') != '26.2.1':
        raise SystemExit(f'Expected pip26.2.1, found {version("pip")}')
    import torch
    from mjlab.managers.observation_manager import ObservationGroupCfg
    from src.wmac.reference import config_protocol, imagined_foot_slip_protocol, smooth_reward_protocol
    from src.wmac.smooth_rewards import SmoothRewardTerms
    for name in ('src.wmac.runner', 'src.wmac.reference', 'src.wmac.reward_integration', 'rsl_rl', 'scripts.train_wmac'):
        loaded = Path(import_module(name).__file__).resolve()
        if not loaded.is_relative_to(root):
            raise SystemExit(f'Wrong imported source: {name}: {loaded}')
        print(name, loaded)
    if version('rsl-rl-lib') != '5.0.1' or str(torch.__version__) != '2.7.0+cu126' or torch.version.cuda != '12.6':
        raise SystemExit('Unexpected PyTorch/CUDA/RSL package versions')
    if 'history_ordering' not in ObservationGroupCfg.__dataclass_fields__:
        raise SystemExit('Missing mjlab history-ordering patch')
    installed = Path(distribution('mjlab').locate_file('mjlab/managers/observation_manager.py'))
    if installed.read_bytes() != (root/'mjlab_patch/mjlab/managers/observation_manager.py').read_bytes():
        raise SystemExit('Installed history patch differs from pinned source')
    cfg = reproduction_config()
    actual = json.loads(json.dumps(asdict(cfg)))
    expected = json.loads((root/'docs/reproduction/default_config.json').read_text())
    if actual != expected:
        raise SystemExit('Current resolved fresh-training configuration differs from the manual snapshot')
    if (cfg.num_envs, cfg.wm_window, cfg.optimizer_algorithm, cfg.mirror_walk_motions, cfg.precision) != (4096, 7, 'muon_adam', True, 'tf32'):
        raise SystemExit('Unexpected production training configuration')
    print(f'Environment pins verified: {checked} dependencies plus pip26.2.1')
    print('Production protocol:', json.dumps(config_protocol(cfg), sort_keys=True))
    print('Smooth reward formulas:', json.dumps(smooth_reward_protocol(cfg), sort_keys=True))
    print('Imagined recovery foot slip:', json.dumps(imagined_foot_slip_protocol(cfg), sort_keys=True))
    print('CUDA availability:', torch.cuda.is_available(), '(not a GPU simulation test)')


if __name__ == '__main__':
    main()
