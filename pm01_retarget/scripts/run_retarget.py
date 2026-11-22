"""Retarget the SuperAMP G1 references onto EngineAI PM01 with hhtools (robot -> robot).

Drives hhtools' maintained R2R library path in-process, in the same order its
Web / Agent runtime does (``hhtools/application/agent_services.py``):

    load_source_trajectory -> source_trajectory_to_motion (G1 FK keypoints)
    -> load_r2r_calibration -> retarget_robot_to_robot (Newton IK)
    -> scaled preview + build_result_diagnostics -> write_r2r_export_bundle

``Human-Humanoid-Tools/scripts/batch_r2r_retarget.py`` wraps the same calls but,
at the checked-out commit (13b6dd3), passes ``source_model=`` /
``calibrated_joint_q=`` to ``write_r2r_export_bundle``, which now takes
``scene_scale_ratio=`` instead, so every clip fails at export.  This driver
leaves the clone untouched.

Stride scale
------------
hhtools moves the pelvis by ``target stature / source stature`` times the
source displacement (1.074 for G1 -> PM01) but places each foot relative to the
pelvis with the calibrated leg ratio (1.025: PM01's legs are only 2.5 % longer;
most of its extra height is torso and head).  A foot the source keeps planted
therefore drifts by the difference, 4.9 % of the pelvis speed.
``ScalerConfig.root_trajectory_scale`` exists for exactly this case ("match it
to the leg scale"), but the R2R entry point never applies the ``robot.yaml``
override.  ``--stride-scale leg`` (default) therefore builds the same
``NewtonBasicPipeline`` that ``retarget_robot_to_robot`` builds and sets that
one field to the calibrated ankle scale; ``--stride-scale stature`` calls
``retarget_robot_to_robot`` unchanged.

Inputs : pm01_retarget/g1_source/<group>/<clip>.csv   (prepare_g1_sources.py)
Outputs: pm01_retarget/pm01_hhtools/<group>/<clip>.csv (hhtools robot CSV, source fps)
         pm01_retarget/qa/hhtools_diagnostics/<group>/<clip>.json

Run with the hhtools environment:  python run_retarget.py [--clip NAME ...]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from common import G1_JOINT_NAMES, PM01_JOINT_NAMES, SOURCE_ROBOT, TARGET_ROBOT, WORK

SOURCE_DIR = WORK / "g1_source"
DEFAULT_OUT = WORK / "pm01_hhtools"
DEFAULT_DIAG = WORK / "qa" / "hhtools_diagnostics"


def _display_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(WORK))
    except ValueError:
        return str(path)


def _diagnostics_summary(diagnostics: dict) -> dict:
    """hhtools' result diagnostics without the per-frame arrays."""

    def compact(value):
        if isinstance(value, dict):
            return {k: compact(v) for k, v in value.items() if not _is_frame_array(v)}
        if isinstance(value, list):
            return [compact(v) for v in value]
        return value

    return {
        key: compact(diagnostics[key])
        for key in ("available", "reason", "frame_count", "mapped_effectors", "requested_effectors",
                    "quality", "tracking", "contact")
        if key in diagnostics
    }


def _is_frame_array(value) -> bool:
    """Per-frame series (long lists of numbers / per-frame records) are dropped."""
    return isinstance(value, list) and len(value) > 32


class LegScaledR2R:
    """``retarget_robot_to_robot`` (Newton backend) with the stride scale tied to the legs.

    Mirrors the Newton branch of ``hhtools.retarget.robot_to_robot.retarget_robot_to_robot``
    line for line; the only difference is ``ScalerConfig.root_trajectory_scale``.
    The pipeline is built once and reused for every clip.
    """

    def __init__(self, source_model, target_model, calibration: dict[str, float], *, ik_iterations: int) -> None:
        from hhtools.retarget import robot_to_robot as r2r
        from hhtools.retarget.newton_basic import NewtonBasicPipeline
        from hhtools.retarget.newton_basic._warp_config import configure as configure_warp
        from hhtools.robot.retarget_profile import (
            build_feet_stabilizer_config,
            build_pipeline_config_for_preset,
        )

        self._r2r = r2r
        self._source_model = source_model
        self._target_model = target_model
        self._calibration = calibration
        config, reference = r2r._build_scaler_config(source_model, target_model, calibration)  # noqa: SLF001
        ankle_scales = [config.joint_scales[name] for name in ("left_ankle", "right_ankle")]
        self.stature_scale = float(config.model_height) / float(reference.height_m)
        self.stride_scale = float(np.mean(ankle_scales))
        config.root_trajectory_scale = self.stride_scale
        reference_key = f"robot_{source_model.preset.name}"
        configure_warp()
        self._pipeline = NewtonBasicPipeline(
            target_model,
            scaler_config=config,
            pipeline_config=build_pipeline_config_for_preset(
                target_model.preset, reference_key, ik_iterations=ik_iterations
            ),
            feet_stabilizer_config=build_feet_stabilizer_config(
                target_model.preset, reference_key, model=target_model
            ),
            human_height=float(reference.height_m),
            source_to_canonical={name: name for name in reference.joint_names},
            configure_warp=False,
        )

    def __call__(self, motion):
        from dataclasses import replace

        retargeted = self._pipeline.run(motion)
        retargeted = self._r2r.align_retargeted_ankles_to_scaled_source(
            self._target_model, self._source_model, motion, retargeted, self._calibration
        )
        # Recorded in the exported CSV header next to hhtools' own metadata.
        meta = dict(getattr(retargeted, "meta", {}) or {})
        meta["root_trajectory_scale"] = f"{self.stride_scale:.6f}"
        return replace(retargeted, meta=meta)


def retarget_clip(
    source_path: Path,
    group: str,
    *,
    source_model,
    target_model,
    calibration: dict[str, float],
    out_root: Path,
    diag_root: Path,
    limit_frames: int | None,
    ik_iterations: int,
    leg_scaled: LegScaledR2R | None,
) -> dict:
    from hhtools.analysis.result_diagnostics import build_result_diagnostics
    from hhtools.application.previews import (
        _anchor_scaled_preview_to_ik_root,
        _compute_r2r_scaled_preview,
    )
    from hhtools.io.export_bundle import identity_resample
    from hhtools.io.r2r_export_bundle import write_r2r_export_bundle
    from hhtools.io.scene_serialize import _scaled_overlay_foot_z, serialize_robot_trajectory
    from hhtools.retarget import robot_to_robot as r2r

    stem = source_path.stem
    t0 = time.perf_counter()
    trajectory = r2r.load_source_trajectory(source_path, source_model=source_model)
    if tuple(trajectory.dof_names) != G1_JOINT_NAMES:
        raise ValueError(f"{source_path}: unexpected DOF columns {trajectory.dof_names[:3]}...")
    if trajectory.meta.get("robot") != SOURCE_ROBOT:
        raise ValueError(f"{source_path}: declared robot {trajectory.meta.get('robot')!r} != {SOURCE_ROBOT!r}")
    joint_q = np.asarray(trajectory.joint_q, dtype=np.float32)
    if limit_frames is not None:
        joint_q = joint_q[:limit_frames].copy()
    fps = float(trajectory.framerate)

    motion = r2r.source_trajectory_to_motion(
        source_model, joint_q, trajectory.dof_names, framerate=fps, name=stem
    )
    if leg_scaled is not None:
        retargeted = leg_scaled(motion)
    else:
        retargeted = r2r.retarget_robot_to_robot(
            source_model,
            target_model,
            calibrated_joint_q=calibration,
            source_motion=motion,
            backend="newton",
            ik_iterations=ik_iterations,
        )
    if tuple(retargeted.dof_names) != PM01_JOINT_NAMES:
        raise ValueError(f"{stem}: retargeted DOF order is not the PM01 SDK order")
    if int(retargeted.num_frames) != joint_q.shape[0]:
        raise ValueError(f"{stem}: {retargeted.num_frames} retargeted frames for {joint_q.shape[0]} source frames")

    # hhtools' own tracking diagnostics, built the way the Agent runtime does.
    scaled = _anchor_scaled_preview_to_ik_root(
        _compute_r2r_scaled_preview(source_model, target_model, motion, calibration),
        retargeted,
        motion,
    )
    target_trajectory = serialize_robot_trajectory(
        target_model,
        retargeted,
        scaled_preview=scaled,
        max_frames=600,
        ground_follow=False,
        yellow_align="ankle",
    )
    diagnostics = build_result_diagnostics(
        target_trajectory,
        scaled,
        ik_map=target_model.preset.ik_map,
        feet=target_model.preset.feet,
        ik_targets=getattr(retargeted, "ik_targets", None),
    )
    yellow_foot_z = _scaled_overlay_foot_z(scaled, 0)

    scale_ratio = r2r.r2r_scene_scale_ratio(source_model, target_model, motion, calibration)
    exported = write_r2r_export_bundle(
        retargeted,
        target_model,
        motion,
        scene_scale_ratio=scale_ratio,
        entry={
            "source_path": str(source_path),
            "clip_dir": str(source_path.parent),
            "stem": stem,
            "has_scene": False,
            "upload_profile": "mimic",
        },
        out_root=out_root / group,
        stem=stem,
        fps=None,
        fmt="csv",
        resample_fn=identity_resample,
        csv_header=True,
        yellow_foot_z=yellow_foot_z,
        pack_scene=False,
    )

    summary = {
        "clip": stem,
        "group": group,
        "source": _display_path(source_path),
        "output": _display_path(Path(exported)),
        "frames": int(retargeted.num_frames),
        "fps": float(retargeted.sample_rate),
        "scale_ratio": float(scale_ratio),
        "stride_scale": float(leg_scaled.stride_scale) if leg_scaled is not None else float(scale_ratio),
        "seconds": round(time.perf_counter() - t0, 2),
        "retarget_meta": {k: (v if isinstance(v, (int, float, str, bool)) else str(v))
                          for k, v in dict(getattr(retargeted, "meta", {}) or {}).items()},
        "hhtools_diagnostics": _diagnostics_summary(diagnostics),
    }
    diag_path = diag_root / group / f"{stem}.json"
    diag_path.parent.mkdir(parents=True, exist_ok=True)
    diag_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip", action="append", default=None, help="Only this clip stem (repeatable).")
    parser.add_argument("--limit-frames", type=int, default=None, help="Smoke test: first N frames only.")
    parser.add_argument("--ik-iterations", type=int, default=24, help="hhtools default: 24.")
    parser.add_argument(
        "--stride-scale",
        choices=("leg", "stature"),
        default="leg",
        help="Pelvis travel scale: calibrated leg ratio (default) or hhtools' stature ratio.",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--diagnostics", type=Path, default=DEFAULT_DIAG)
    args = parser.parse_args()

    from hhtools.retarget import robot_to_robot as r2r
    from hhtools.robot.loader import load_robot
    from hhtools.robot.registry import get as get_preset
    from hhtools.robot.registry import refresh

    refresh()
    source_model = load_robot(get_preset(SOURCE_ROBOT), compile_mjcf=False)
    target_model = load_robot(get_preset(TARGET_ROBOT), compile_mjcf=True)
    calibration = r2r.load_r2r_calibration(
        target_model.preset.urdf_path.parent, SOURCE_ROBOT, target_robot=TARGET_ROBOT
    )
    if not calibration:
        raise SystemExit("no R2R calibration for g1_29dof -> pm01; run calibrate_pair.py first")

    sources = sorted(SOURCE_DIR.glob("*/*.csv"))
    if args.clip:
        wanted = set(args.clip)
        sources = [p for p in sources if p.stem in wanted]
        missing = wanted - {p.stem for p in sources}
        if missing:
            raise SystemExit(f"unknown clip(s): {sorted(missing)}")
    if not sources:
        raise SystemExit(f"no source trajectories under {SOURCE_DIR}; run prepare_g1_sources.py first")

    leg_scaled = None
    if args.stride_scale == "leg":
        leg_scaled = LegScaledR2R(source_model, target_model, calibration, ik_iterations=args.ik_iterations)
        print(
            f"stride scale: leg ratio {leg_scaled.stride_scale:.4f} "
            f"(hhtools stature default would be {leg_scaled.stature_scale:.4f})"
        )
    else:
        print("stride scale: hhtools default (target stature / source stature)")

    failures = []
    print(f"{'group':11s} {'clip':40s} {'frames':>6s} {'fps':>5s} {'stride':>6s} {'p95 err':>8s} {'band':>10s} {'sec':>6s}")
    for path in sources:
        group = path.parent.name
        try:
            s = retarget_clip(
                path,
                group,
                source_model=source_model,
                target_model=target_model,
                calibration=calibration,
                out_root=args.out,
                diag_root=args.diagnostics,
                limit_frames=args.limit_frames,
                ik_iterations=args.ik_iterations,
                leg_scaled=leg_scaled,
            )
        except Exception as error:  # noqa: BLE001 - report every clip, fail at the end
            failures.append((path.stem, repr(error)))
            print(f"{group:11s} {path.stem:40s} FAILED: {error!r}")
            continue
        tracking = s["hhtools_diagnostics"].get("tracking", {})
        quality = s["hhtools_diagnostics"].get("quality", {})
        p95 = tracking.get("p95_error_m")
        print(
            f"{group:11s} {s['clip']:40s} {s['frames']:6d} {s['fps']:5.0f} {s['stride_scale']:6.3f} "
            f"{(f'{p95 * 1e3:6.1f}mm' if isinstance(p95, (int, float)) else '     n/a'):>8s} "
            f"{str(quality.get('band', quality.get('level', '?'))):>10s} {s['seconds']:6.1f}"
        )
    if failures:
        print(f"\n{len(failures)} clip(s) failed:")
        for stem, error in failures:
            print(f"  {stem}: {error}")
        return 1
    print(f"\nretargeted {len(sources)} clip(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
