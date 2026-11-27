"""Train the isolated world-model-dominant AMP actor-critic version."""

from __future__ import annotations

from dataclasses import asdict
from datetime import datetime
import argparse
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import tyro

from src.wmac.config import PROFILES, WmacConfig
from src.wmac.checkpoints import resolve_checkpoint_config
from src.wmac.actors import actor_architecture
from src.wmac.reference import (OFFICIAL_REPOSITORY, OFFICIAL_COMMIT, MIRROR_REFERENCE_COMMIT,
                                checkpoint_protocol, config_protocol,
                                validate_training_protocol_resume, amp_reward_protocol,
                                INSTANTAMP_REPOSITORY, INSTANTAMP_COMMIT, INSTANTAMP_SOURCE,
                                smooth_reward_protocol, validate_smooth_reward_resume,
                                validate_smooth_reward_retune,
                                imagined_foot_slip_protocol, IMAGINED_FOOT_SLIP_GEOMETRY,
                                IMAGINED_FOOT_SLIP_GEOMETRIC_MODES,
                                HOST_COMMIT, HUMANUP_COMMIT)


def parse_config(argv: list[str] | None = None) -> WmacConfig:
    """Choose profile/checkpoint defaults before applying explicit CLI values.

    Older WMAC checkpoints have no profile. Their known configuration fields
    overlay the legacy defaults; a resume never implicitly becomes a new-sized
    balanced experiment. The previous initialization source is provenance,
    rather than a request to warmstart the restored run a second time.
    """
    arguments = list(sys.argv[1:] if argv is None else argv)
    preliminary = argparse.ArgumentParser(add_help=False)
    preliminary.add_argument("--profile", choices=PROFILES)
    preliminary.add_argument("--resume")
    selected, _ = preliminary.parse_known_args(arguments)
    source_modes = None
    source_protocol = None
    source_actor_architecture = None
    if selected.resume and selected.resume != "None":
        import torch

        checkpoint = Path(selected.resume).expanduser().resolve()
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if payload.get("format") != "amp-wmac-v1":
            raise ValueError("--resume requires a WMAC checkpoint; use --init-policy for PPO weights.")
        saved = payload.get("config")
        if not isinstance(saved, dict):
            raise ValueError("The WMAC checkpoint has no saved configuration.")
        saved_profile = saved.get("profile", "legacy")
        if selected.profile is not None and selected.profile != saved_profile:
            raise ValueError(
                f"This checkpoint uses profile {saved_profile!r}; --resume cannot relabel it "
                f"as {selected.profile!r}. Omit --profile to restore the saved configuration. "
                "Use --init-policy for a new profile experiment, or override individual "
                "compatible parameters explicitly while retaining the saved profile."
            )
        default = resolve_checkpoint_config(payload)
        source_protocol = checkpoint_protocol(payload)
        source_actor_architecture = actor_architecture(default)
        source_modes = {key: getattr(default, key) for key in
                        ("recovery_counter_semantics", "recovery_height_reward")}
        default.resume = str(checkpoint)
        default.init_policy = None
    else:
        default = WmacConfig.for_profile(selected.profile or "balanced")
    cfg = tyro.cli(WmacConfig, default=default, args=arguments)
    for key, value in imagined_foot_slip_protocol(cfg).items():
        setattr(cfg, key, value)
    if source_protocol is None and cfg.allow_imagined_foot_slip_migration:
        raise ValueError("--allow-imagined-foot-slip-migration requires --resume")
    smooth_migrated = False
    smooth_retuned = False
    if source_protocol is not None:
        validate_training_protocol_resume(payload, cfg)
        smooth_migrated = validate_smooth_reward_resume(payload, cfg)
        smooth_retuned = validate_smooth_reward_retune(payload, cfg)
    elif cfg.allow_smooth_reward_retune:
        raise ValueError("--allow-smooth-reward-retune requires --resume from an active smooth checkpoint")
    if source_actor_architecture is not None and actor_architecture(cfg) != source_actor_architecture:
        raise ValueError("Resume Actor architecture mismatch; use --init-policy for a separate MLP-to-MoE warmstart experiment")
    changed_modes = {key: [value, getattr(cfg, key)] for key, value in (source_modes or {}).items()
                     if value != getattr(cfg, key)}
    if smooth_migrated and "recovery_height_reward" in changed_modes:
        raise ValueError("Smooth reward migration cannot also change the recovery height objective; "
                         "use separate controlled migrations with consistent replay labels")
    if smooth_retuned and changed_modes:
        raise ValueError("Smooth reward retune cannot also change recovery semantics")
    if changed_modes and not cfg.allow_recovery_semantics_migration:
        raise ValueError("Resume changes recovery semantics; explicitly pass "
                         f"--allow-recovery-semantics-migration for this experiment: {changed_modes}")
    return cfg


def source_snapshot(log_dir: Path) -> None:
    """Record tracked changes and complete experiment sources for uncommitted runs."""
    import shutil

    def git(*args: str) -> str:
        result = subprocess.run(
            ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False,
        )
        return result.stdout.strip() if result.returncode == 0 else "unavailable"

    metadata = {
        "repository": str(REPO), "commit": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"), "status": git("status", "--short"),
        "python": sys.executable, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "paper": "https://arxiv.org/abs/2610.28480",
        "reference_implementation": "https://github.com/zita-ch/supermimic_vibed",
    }
    config = json.loads((log_dir / "config.json").read_text())
    if config.get("reference_alignment") == "official_details_v1":
        metadata["reference_implementation"] = OFFICIAL_REPOSITORY
        metadata["reference_commit"] = OFFICIAL_COMMIT
        metadata["alignment_scope"] = "Official compatible details; validated AMP local-residual dynamics, BPTT and recovery retained"
    if config.get("mirror_walk_motions"):
        metadata["motion_mirror_reference_commit"] = MIRROR_REFERENCE_COMMIT
    metadata["actor_architecture"] = actor_architecture(config)
    metadata["amp_reward_protocol"] = amp_reward_protocol(config)
    metadata["smooth_reward_protocol"] = smooth_reward_protocol(config)
    foot_slip = imagined_foot_slip_protocol(config)
    foot_slip_mode = foot_slip['imagined_recovery_foot_slip']
    if foot_slip_mode in IMAGINED_FOOT_SLIP_GEOMETRIC_MODES:
        metadata['imagined_foot_slip_protocol'] = foot_slip
        reference_root = INSTANTAMP_REPOSITORY + '/blob/' + INSTANTAMP_COMMIT
        metadata['imagined_foot_slip_reference'] = {
            'geometry': reference_root + '/instantamp/sim.py#L248-L264',
            'analytic_reward': reference_root + '/instantamp/env.py#L274-L285',
            'imagined_rollout': reference_root + '/instantamp/train.py#L361-L373',
            'scope': 'Flat delayed recovery only; physical foot slip, self collisions and AMP unchanged.',
        }
        if foot_slip_mode == IMAGINED_FOOT_SLIP_GEOMETRY:
            metadata['imagined_foot_slip_reference']['detached_support'] = (
                'https://github.com/Renforce-Dynamics/trackerlab-supermimic/blob/'
                '5d1af044f1943b7f6347f14ea1d1f9dd1a58a458/source/trackerLab/trackerLab/algorithms/'
                'supermimic/locomotion_quality.py#L149-L169')
    if config.get("smooth_reward_mode", "none") != "none":
        metadata["smooth_reward_reference"] = {
            "HoST": {"repository": "https://github.com/InternRobotics/HoST", "commit": HOST_COMMIT},
            "HumanUP": {"repository": "https://github.com/RunpeiDong/humanup", "commit": HUMANUP_COMMIT},
            "scope": "Reward terms only; no action amplitude curriculum, new world-model head or extra training branch",
            "torque_note": "Analytic existing-PD request before clipping; power uses limited/frictional command, not physical 5 ms actuator_force",
        }
    if config.get("amp_reward_mapping") == "instantamp_centered_tanh_v1":
        metadata["amp_reward_reference"] = {
            "repository": INSTANTAMP_REPOSITORY, "commit": INSTANTAMP_COMMIT,
            "source": INSTANTAMP_SOURCE,
            "scope": "Robot score EMA centering and continuous bounded tanh reward only; existing single LSGAN discriminator and world model retained",
            "scale_note": "Fixed score scale in LSGAN +/-1 target units; InstantAMP WGAN scale 5 is not copied",
        }
    if config.get("actor_type") == "moe" and config.get("moe_variant") == "hiking":
        metadata["moe_reference"] = {
            "scope": "Hiking MoE Actor with paper Table6 gate [256]; full Hiking PPO/vision framework is not replicated",
            "paper": "https://arxiv.org/abs/2601.07718",
            "network_commit": "574c8d50cdd16618cf8f4e0df553f84759c1d837",
            "task_commit": "de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057",
            "gate_hidden": list(config.get("moe_gate_hidden", [256])),
            "gate_note": ("Paper Table6 gate [256], selected for this experiment."
                          if config.get("moe_gate_hidden") == [256] else
                          "Historical author-code linear gate [], restored explicitly."),
        }
    (log_dir / "source.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (log_dir / "source.diff").write_text(git("diff", "HEAD") + "\n")
    target = log_dir / "source"
    shutil.copytree(REPO / "src" / "wmac", target / "wmac", ignore=shutil.ignore_patterns("__pycache__"))
    for name in ("amp_features.py", "training_cost.py", "recovery_rewards.py"):
        shutil.copy2(REPO / "src" / name, target / name)
    reward_source = Path("src/tasks/amp_loco/mdp/rewards.py")
    destination = target / "tasks/amp_loco/mdp/rewards.py"
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(REPO / reward_source, destination)
    for name in ("train_wmac.py", "run_wmac_ablations.py"):
        shutil.copy2(REPO / "scripts" / name, target / name)


def configure_motion_dataset(cfg, env_cfg):
    """Use one verified mirrored flat dataset for both resets and AMP experts."""
    if not cfg.mirror_walk_motions:
        return None
    from src.wmac.motions import prepare_mirrored_dataset
    loader = env_cfg.events["init_motion_loader"].params
    original_walk = Path(loader["motion_dir"]).resolve()
    if original_walk.name != "WalkandRun":
        raise ValueError("Flat mirroring requires the original WalkandRun/Recovery motion groups")
    cache = Path(cfg.motion_cache_root).expanduser()
    cache = cache if cache.is_absolute() else REPO / cache
    paths = prepare_mirrored_dataset(original_walk.parent, cache)
    loader.update(motion_dir=str(paths.walk), recovery_dir=str(paths.recovery))
    env_cfg.events["reset_from_motion"].params["motion_dir"] = str(paths.walk)
    return paths


def main(cfg: WmacConfig) -> None:
    started_at = time.perf_counter()
    cfg.apply_preset()
    cfg.validate()
    os.environ["MUJOCO_GL"] = "egl"
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    if cfg.device.startswith("cuda") and "CUDA_VISIBLE_DEVICES" not in os.environ:
        # Without an existing visibility mask, --device selects the physical GPU.
        os.environ["CUDA_VISIBLE_DEVICES"] = cfg.device.partition(":")[2] or "0"
        cfg.device = "cuda:0"
    if cfg.device.startswith("cuda"):
        os.environ["MUJOCO_EGL_DEVICE_ID"] = cfg.device.partition(":")[2] or "0"

    import numpy as np
    import torch
    from mjlab.tasks.registry import load_env_cfg
    from mjlab.utils.torch import configure_torch_backends
    import mjlab.tasks  # noqa: F401
    import src.tasks  # noqa: F401
    from src.wmac.env import CaptureEnv
    from src.wmac.state import StateAdapter
    from src.wmac.runner import WmacRunner
    from src.recovery_rewards import configure_recovery_height_reward
    from src.wmac.reward_integration import configure_smooth_reward_env

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    if cfg.device.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the selected existing environment.")
        torch.cuda.manual_seed_all(cfg.seed)
    configure_torch_backends(allow_tf32=cfg.precision == "tf32")

    log_root = Path(cfg.log_root).expanduser()
    if not log_root.is_absolute():
        log_root = REPO / log_root
    suffix = f"_{cfg.run_name}" if cfg.run_name else ""
    if Path(suffix).name != suffix or "/" in suffix or "\\" in suffix:
        raise ValueError("run_name must not contain path separators.")
    log_dir = log_root / f"{datetime.now():%Y-%m-%d_%H-%M-%S_%f}_seed{cfg.seed}_{cfg.preset}{suffix}"
    log_dir.mkdir(parents=True, exist_ok=False)
    (log_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2) + "\n")
    source_snapshot(log_dir)
    print(f"[WMAC] log_dir={log_dir}", flush=True)
    print(f"[WMAC] resolved_config={json.dumps(asdict(cfg), sort_keys=True)}", flush=True)

    env_cfg = load_env_cfg(cfg.task)
    configure_recovery_height_reward(env_cfg, cfg.recovery_height_reward)
    configure_smooth_reward_env(cfg, env_cfg)
    motion_paths = configure_motion_dataset(cfg, env_cfg)
    if motion_paths is not None:
        import shutil
        shutil.copy2(motion_paths.manifest, log_dir / "motion_dataset.json")
    env_cfg.scene.num_envs = cfg.num_envs
    env_cfg.seed = cfg.seed
    env = None
    completed = False
    try:
        env = CaptureEnv(cfg=env_cfg, device=cfg.device)
        if motion_paths is not None:
            env._wmac_motion_root = str(motion_paths.root)
        env.adapter = StateAdapter(env, recovery_counter_semantics=cfg.recovery_counter_semantics)
        runner = WmacRunner(env, cfg, log_dir)
        if cfg.resume:
            runner.load(str(Path(cfg.resume).expanduser().resolve()))
        runner.learn()
        completed = True
    finally:
        if env is not None:
            env.close()
        (log_dir / "runtime.json").write_text(json.dumps({
            "wall_seconds_including_environment_setup": time.perf_counter() - started_at,
            "completed": completed,
        }, indent=2) + "\n")


if __name__ == "__main__":
    main(parse_config())
