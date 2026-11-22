"""Write the retargeted PM01 motions in the two layouts the downstream code reads.

Input : pm01_retarget/pm01_refined/<group>/<clip>.csv   (refine_motions.py) where it exists,
        pm01_retarget/pm01_hhtools/<group>/<clip>.csv   (hhtools robot CSV) otherwise

Output:
* pm01_csv/<group>/<clip>.csv
    Headerless ``x y z qx qy qz qw`` + 24 joints (J00..J23, EngineAI SDK order),
    at the source frame rate (120 Hz walk/run, 50 Hz recovery).  Same layout as
    SuperAMP's ``motion_data_csv/amp/*.csv`` and engineai_rl_lab's
    ``datasets/tracking/pm01/*.csv``, so both ``csv_to_npz.py`` scripts take it
    with ``--input-fps`` / ``--input_fps`` set to that rate.
* pm01_npz_<fps>hz/<group>/<clip>.npz
    engineai_rl_lab's compact AMP layout (``datasets/amp/pm01/data``):
    ``fps``, ``root_pos`` (T, 3), ``root_rot`` (T, 4, xyzw), ``dof_pos`` (T, 24),
    plus ``dof_names`` for self-description.  Every clip is resampled to one
    rate (default 50 Hz) with the same interpolation SuperAMP's csv_to_npz.py
    applies to G1, so each file has the frame count and timing of the G1 NPZ
    it corresponds to.

Values are the input with the root quaternion re-normalised and kept
sign-continuous.  Refined clips are written as they are (refine_motions.py has
grounded them on PM01's official foot collision boxes).  Clips taken straight
from hhtools get one vertical correction:

Grounding
---------
hhtools places every clip so that the single lowest point a sole ever reaches
is on the floor.  Walking and jogging feet pitch at heel strike and toe-off, so
that lowest point is a toe or heel edge and the feet hover 1-3 cm during
ordinary flat stance.  For the groups in ``--ground-groups`` (default:
WalkandRun) the root height of each clip is shifted by one constant so that the
median sole height of *planted* feet is zero, which is how the G1 source sits
on its floor.  A foot counts as planted when it moves slower than 0.15 m/s
horizontally and its sole is within 3 cm of the clip's low sole level; the sole
is the hull of the foot shell in the PM01 URDF.  Joint angles, root orientation
and horizontal motion are not touched; the shift of each clip and which input
it came from are written to qa/ground_shift.json.

The Recovery clip is left as hhtools placed it: lying phases rest on the floor
there, and no single constant can also plant the standing phases.

Run with the hhtools environment:  python export_formats.py [--npz-fps 50] [--no-refined]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from common import G1_NPZ_DIR, PM01_JOINT_NAMES, WORK, resample_like_csv_to_npz
from validate import PM01_URDF, Robot, load_hhtools_csv

PLANTED_SPEED_M_S = 0.15
PLANTED_HEIGHT_M = 0.03
SOLE_BODIES = ("LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R")  # carry the foot shell in the URDF model


def continuous_unit_quaternions(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    flip = np.cumprod(np.where(np.sum(q[1:] * q[:-1], axis=1) < 0.0, -1.0, 1.0))
    q[1:] *= flip[:, None]
    return q


def planted_sole_level(robot: Robot, joint_q: np.ndarray, fps: float) -> tuple[float, float]:
    """(median sole height of planted feet, share of foot-frames that are planted)."""
    frames = joint_q.shape[0]
    bodies = [robot.body_id(name) for name in SOLE_BODIES]
    geoms = [robot.body_collision_geoms(b) for b in bodies]
    sole = np.zeros((frames, len(bodies)))
    xy = np.zeros((frames, len(bodies), 2))
    for f in range(frames):
        rot, pos = robot.set_pose(joint_q[f])
        for i, body in enumerate(bodies):
            sole[f, i] = min(robot.geom_lowest_z(g, rot, pos) for g in geoms[i])
            xy[f, i] = (rot @ robot.data.xpos[body] + pos)[:2]
    speed = np.linalg.norm(np.gradient(xy, 1.0 / fps, axis=0), axis=2)
    low_level = np.percentile(sole.min(axis=1), 5)
    planted = (speed < PLANTED_SPEED_M_S) & (sole < low_level + PLANTED_HEIGHT_M)
    if planted.sum() < 10:
        raise ValueError("too few planted frames to ground the clip")
    return float(np.median(sole[planted])), float(planted.mean())


def check_compact_motion(path: Path) -> tuple[int, float]:
    """The checks engineai_rl_lab's AMPDataLoader runs on a compact clip."""
    with np.load(path, allow_pickle=False) as data:
        missing = {"fps", "root_pos", "root_rot", "dof_pos"} - set(data.files)
        assert not missing, (path, missing)
        fps = float(data["fps"])
        root_pos, root_rot, dof_pos = data["root_pos"], data["root_rot"], data["dof_pos"]
        names = tuple(str(n) for n in data["dof_names"])
    assert math.isfinite(fps) and fps > 0.0
    assert root_pos.ndim == 2 and root_pos.shape[1] == 3
    assert root_rot.ndim == 2 and root_rot.shape[1] == 4
    assert dof_pos.ndim == 2 and dof_pos.shape[1] == len(PM01_JOINT_NAMES)
    assert dof_pos.shape[0] >= 2 and root_pos.shape[0] == root_rot.shape[0] == dof_pos.shape[0]
    assert all(np.isfinite(a).all() for a in (root_pos, root_rot, dof_pos))
    assert np.abs(np.linalg.norm(root_rot, axis=1) - 1.0).max() < 1e-9
    assert names == PM01_JOINT_NAMES
    return int(dof_pos.shape[0]), fps


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pm01", type=Path, default=WORK / "pm01_hhtools")
    parser.add_argument("--refined", type=Path, default=WORK / "pm01_refined", help="Clips found here replace the hhtools ones.")
    parser.add_argument("--no-refined", action="store_true", help="Export the hhtools clips only.")
    parser.add_argument("--npz-fps", type=float, default=50.0, help="Common frame rate of the NPZ set.")
    parser.add_argument(
        "--ground-groups",
        nargs="*",
        default=["WalkandRun"],
        help="Groups whose clips are shifted so planted soles sit at z = 0 (none: keep hhtools' heights).",
    )
    args = parser.parse_args()
    robot = Robot(PM01_URDF, PM01_JOINT_NAMES)
    shifts: dict[str, dict] = {}

    csv_root = WORK / "pm01_csv"
    npz_root = WORK / f"pm01_npz_{args.npz_fps:g}hz"
    sources = sorted(args.pm01.glob("*/*.csv"))
    if not sources:
        raise SystemExit(f"no hhtools exports under {args.pm01}; run run_retarget.py first")

    print(
        f"{'group':11s} {'clip':40s} {'csv frames':>10s} {'fps':>5s} {'npz frames':>10s} {'fps':>5s} {'G1 npz':>7s} "
        f"{'planted sole':>12s} {'z shift':>8s}  input"
    )
    ok = True
    for path in sources:
        group, stem = path.parent.name, path.stem
        refined = args.refined / group / path.name
        is_refined = refined.is_file() and not args.no_refined
        if is_refined:
            path = refined
        joint_q, fps, dof_names, meta = load_hhtools_csv(path)
        assert dof_names == PM01_JOINT_NAMES and meta.get("robot") == "pm01", path
        joint_q = joint_q.copy()
        joint_q[:, 3:7] = continuous_unit_quaternions(joint_q[:, 3:7])

        level, planted_share = planted_sole_level(robot, joint_q, fps)
        shift = -level if group in args.ground_groups and not is_refined else 0.0
        joint_q[:, 2] += shift
        shifts[f"{group}/{stem}"] = {
            "input": str(path.relative_to(WORK)) if path.is_relative_to(WORK) else str(path),
            "planted_sole_height_before_m": round(level, 5),
            "root_z_shift_m": round(shift, 5),
            "planted_share": round(planted_share, 4),
        }

        csv_path = csv_root / group / f"{stem}.csv"
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        np.savetxt(csv_path, joint_q, delimiter=",", fmt="%.6f")
        reread = np.loadtxt(csv_path, delimiter=",")
        assert reread.shape == (joint_q.shape[0], 7 + len(PM01_JOINT_NAMES)), csv_path
        assert np.abs(reread - joint_q).max() < 1e-6, csv_path

        if abs(fps - args.npz_fps) < 1e-9:
            resampled = joint_q
        else:
            resampled = resample_like_csv_to_npz(joint_q, fps, args.npz_fps)
            resampled[:, 3:7] = continuous_unit_quaternions(resampled[:, 3:7])
        npz_path = npz_root / group / f"{stem}.npz"
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            npz_path,
            fps=np.float64(args.npz_fps),
            root_pos=resampled[:, :3].astype(np.float64),
            root_rot=resampled[:, 3:7].astype(np.float64),
            dof_pos=resampled[:, 7:].astype(np.float64),
            dof_names=np.asarray(PM01_JOINT_NAMES),
        )
        frames, npz_fps = check_compact_motion(npz_path)

        g1_frames = "-"
        g1_npz = G1_NPZ_DIR / group / f"{stem}.npz"
        if g1_npz.is_file() and abs(args.npz_fps - 50.0) < 1e-9:
            with np.load(g1_npz, allow_pickle=False) as data:
                g1_count = int(data["joint_pos"].shape[0])
            g1_frames = str(g1_count)
            ok &= g1_count == frames
        print(
            f"{group:11s} {stem:40s} {joint_q.shape[0]:10d} {fps:5.0f} {frames:10d} {npz_fps:5.0f} {g1_frames:>7s} "
            f"{level * 100:9.2f} cm {shift * 100:5.2f} cm  {'refined' if is_refined else 'hhtools'}"
        )

    qa = WORK / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    (qa / "ground_shift.json").write_text(json.dumps(shifts, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {len(sources)} CSVs under {csv_root}")
    print(f"wrote {len(sources)} NPZs under {npz_root}")
    if not ok:
        print("frame counts differ from the SuperAMP G1 NPZs", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
