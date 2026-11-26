"""Inspect a WMAC integration assumption without running physics or training.

The numeric reference is zero-acceleration kinematic integration. It is not a
ground-truth physics rollout or a trained-policy assessment. With no --output,
the script only prints JSON. An optional official checkout verifies the pinned
upstream commit and records source hashes without importing its dependencies.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OFFICIAL_COMMIT = "abc124c1bd5976341034668af3d12c087f2f1f55"
OFFICIAL_ROOT = "https://github.com/HybridRobotics/supermimic"
OFFICIAL_PATHS = (
    "src/supermimic/models.py",
    "src/supermimic/representations.py",
    "src/supermimic/envs/world_model_env.py",
    "src/supermimic/learners/wm_learner.py",
    "src/supermimic/learners/policy_learner.py",
    "src/supermimic/runners/supermimic_runner.py",
    "src/supermimic/tasks/task_spec.py",
)
LOCAL_PATHS = (
    "src/wmac/networks.py", "src/wmac/state.py",
    "src/wmac_perceptive/models.py", "src/wmac_perceptive/runner.py",
    "src/wmac_perceptive/config.py", "src/wmac_perceptive/visual_metrics.py",
    "src/tasks/amp_perceptive/sensors.py", "src/tasks/amp_perceptive/terrain.py",
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def collect_evidence(dt: float, official_checkout: Path | None = None) -> dict:
    import torch
    from src.wmac.networks import WorldModel

    if not math.isfinite(dt) or dt <= 0:
        raise ValueError("dt must be positive and finite")
    upstream_hashes = {}
    if official_checkout is not None:
        official_checkout = official_checkout.expanduser().resolve()
        actual = subprocess.check_output(
            ["git", "-C", str(official_checkout), "rev-parse", "HEAD"], text=True
        ).strip()
        if actual != OFFICIAL_COMMIT:
            raise ValueError(f"Official checkout is {actual}, expected {OFFICIAL_COMMIT}")
        for name in OFFICIAL_PATHS:
            committed = subprocess.check_output(
                ["git", "-C", str(official_checkout), "show", f"{OFFICIAL_COMMIT}:{name}"]
            )
            actual_source = (official_checkout/name).read_bytes()
            if actual_source != committed:
                raise ValueError(f"Official source has uncommitted changes: {name}")
            upstream_hashes[name] = hashlib.sha256(actual_source).hexdigest()

    with torch.random.fork_rng(devices=[]), torch.no_grad():
        torch.manual_seed(0)
        model = WorldModel(hidden=(8,)).cpu()
        assert model.out_norm.mean.numel() == 70, "This audit targets the 70D residual model"
        state = torch.zeros(1, 71)
        state[:, 2], state[:, 3] = .75, 1.
        state[:, 7], state[:, 42] = 1., .5
        following = model.integrate(state, torch.zeros(1, 70))
        assert following.shape == state.shape
        assert bool(torch.isfinite(following).all())
        observed = {
            "root_dx_m": float(following[0, 0]-state[0, 0]),
            "joint0_dq_rad": float(following[0, 13]-state[0, 13]),
            "root_velocity_unchanged": bool(torch.equal(following[:, 7:13], state[:, 7:13])),
            "joint_velocity_unchanged": bool(torch.equal(following[:, 42:71], state[:, 42:71])),
        }

    previous = ROOT/"docs/validation/perceptive_training_v2.json"
    snapshot = json.loads(previous.read_text()) if previous.is_file() else {}
    baseline_hashes = snapshot.get("changed_python_source_sha256", {})
    local = {}
    for name in LOCAL_PATHS:
        actual = file_sha256(ROOT/name)
        baseline = baseline_hashes.get(name)
        local[name] = {"current_sha256": actual, "recorded_v2_sha256": baseline,
                       "matches_recorded_v2": actual == baseline if baseline is not None else None}

    return {
        "format": "supermimic-wmac-assumption-audit-v1",
        "verified_at": datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        "official": {
            "repository": OFFICIAL_ROOT, "commit": OFFICIAL_COMMIT,
            "checkout_verified": official_checkout is not None,
            "source_urls": {name: f"{OFFICIAL_ROOT}/blob/{OFFICIAL_COMMIT}/{name}" for name in OFFICIAL_PATHS},
            "verified_source_sha256": upstream_hashes,
        },
        "local_sources": local,
        "recorded_v2_snapshot": str(previous.relative_to(ROOT)) if previous.is_file() else None,
        "findings_from_source_inspection": {
            "official_depth_prediction_in_audited_main_model": False,
            "official_future_proprioception": "Synthesized analytically from predicted physical state",
            "official_gencoord_dimensions_for_29_joints": {"representation": 68, "velocity_increment_prediction": 35},
            "local_dimensions": {"raw_state": 71, "independent_increment_prediction": 70},
            "local_future_depth": "Additional deterministic image decoder conditioned on current cropped image, physical state and local terrain",
            "third_party_parameterization_reference": "https://github.com/zita-ch/supermimic_vibed/tree/0525a23a2dcfa61f8543f98b84c075fe66d9d286",
        },
        "numeric_probe": {
            "description": "Local WorldModel.integrate with zero normalized output, unfitted normalization mean=0",
            "device": "cpu", "control_dt_seconds": dt,
            "initial_root_forward_velocity_m_s": 1., "initial_joint0_velocity_rad_s": .5,
            "local_observed": observed,
            "zero_acceleration_kinematic_reference": {"root_dx_m": dt, "joint0_dq_rad": .5*dt},
            "interpretation": "Independent pose and velocity residuals do not enforce zero-acceleration kinematic integration by construction.",
        },
        "future_image_ambiguity": {
            "evidence_kind": "Input-information counterexample; no paired-scene physics experiment performed",
            "construction": "Keep the cropped image, robot state, action and local terrain patch identical; vary a side obstacle outside the crop/patch that enters the next view after turning.",
            "consequence": "Identical deterministic inputs can correspond to different true future images; memory or additional scene information is needed to reduce this ambiguity.",
        },
        "limitations": [
            "The kinematic reference is not a simulated physical transition and is not executed from official code.",
            "Zero normalized output after fitted normalization can have a different meaning because the learned output mean changes.",
            "The diagnostic does not prove a trained model violates physics or explain the unique cause of falling.",
            "No policy convergence, terrain success rate, future-camera accuracy or gradient correctness is established.",
            "The new locomotion-only task scope does not itself change these model architectures.",
        ],
        "audit_tool": {"path": "scripts/audit_wmac_model_assumptions.py", "sha256": file_sha256(Path(__file__))},
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dt", type=float, default=.02)
    parser.add_argument("--official-checkout", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if not math.isfinite(args.dt) or args.dt <= 0:
        parser.error("dt must be positive and finite")
    result = collect_evidence(args.dt, args.official_checkout)
    rendered = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    if args.output:
        output = args.output.expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
        print(output)
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
