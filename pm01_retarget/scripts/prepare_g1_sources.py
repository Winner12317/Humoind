"""Convert SuperAMP's G1 reference motions into hhtools robot-trajectory CSVs.

SuperAMP stores the G1 references in two layouts that hhtools cannot ingest
safely as they are:

* ``motion_data_csv/amp/*.csv`` (17 walk/run clips, 120 Hz): headerless rows of
  ``x y z qx qy qz qw`` + 29 joints.  hhtools' fallback for headerless numeric
  CSVs assumes the first column is ``time``, which would silently shift every
  column by one.
* ``src/assets/motions/g1/amp/Recovery/fallAndGetUp1_subject1.npz`` (50 Hz,
  mjlab export): the raw CSV is not in the repository, and the NPZ keys
  (``body_pos_w`` / ``body_quat_w`` / ``joint_pos``) are not a layout hhtools
  recognises.

Both are rewritten with hhtools' own writer into its canonical schema
(``# robot`` / ``# sample_rate`` comments, ``time,root_*,dof_<joint>`` header),
so the source robot, frame rate and joint of every column are explicit.

Run with the hhtools environment:  python prepare_g1_sources.py
"""

from __future__ import annotations

import sys

import numpy as np

from common import (
    G1_CSV_DIR,
    G1_CSV_FPS,
    G1_JOINT_NAMES,
    G1_NPZ_DIR,
    SOURCE_ROBOT,
    SUPERAMP,
    WORK,
    load_g1_csv,
    load_g1_npz_as_joint_q,
)

OUT_DIR = WORK / "g1_source"


def main() -> int:
    from hhtools.io.robot_csv import load_robot_csv, save_robot_csv
    from hhtools.robot.loader import load_robot
    from hhtools.robot.registry import get as get_preset
    from hhtools.robot.registry import refresh

    refresh()
    g1 = load_robot(get_preset(SOURCE_ROBOT), compile_mjcf=False)
    if tuple(g1.dof_names()) != G1_JOINT_NAMES:
        raise SystemExit("hhtools g1_29dof dof order differs from the SuperAMP CSV column order")

    jobs: list[tuple[str, np.ndarray, float, str]] = []
    for csv_path in sorted(G1_CSV_DIR.glob("*.csv")):
        jobs.append(("WalkandRun", load_g1_csv(csv_path), G1_CSV_FPS, str(csv_path.relative_to(SUPERAMP.parent))))
    recovery = G1_NPZ_DIR / "Recovery" / "fallAndGetUp1_subject1.npz"
    recovery_q, recovery_fps = load_g1_npz_as_joint_q(recovery)
    jobs.append(("Recovery", recovery_q, recovery_fps, str(recovery.relative_to(SUPERAMP.parent))))

    print(f"{'group':11s} {'clip':40s} {'frames':>6s} {'fps':>5s} {'sec':>6s}  max|quat norm-1|")
    for group, joint_q, fps, source in jobs:
        stem = source.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        norm_dev = float(np.abs(np.linalg.norm(joint_q[:, 3:7], axis=1) - 1.0).max())
        if not np.isfinite(joint_q).all() or norm_dev > 1e-3:
            raise SystemExit(f"{source}: non-finite values or non-unit root quaternion ({norm_dev:.2e})")
        out_path = OUT_DIR / group / f"{stem}.csv"
        save_robot_csv(
            out_path,
            robot=g1,
            joint_q=joint_q,
            sample_rate=fps,
            meta={"source_file": source, "root_quat_format": "xyzw"},
        )
        # Read back through hhtools to prove the file parses as intended.
        parsed = load_robot_csv(out_path)
        assert parsed.dof_names == G1_JOINT_NAMES, out_path
        assert parsed.meta.get("robot") == SOURCE_ROBOT, out_path
        assert abs(parsed.sample_rate - fps) < 1e-9, out_path
        assert parsed.joint_q.shape == joint_q.shape, out_path
        assert float(np.abs(parsed.joint_q - joint_q).max()) < 1e-6, out_path
        print(
            f"{group:11s} {stem:40s} {joint_q.shape[0]:6d} {fps:5.0f} "
            f"{(joint_q.shape[0] - 1) / fps:6.2f}  {norm_dev:.1e}"
        )
    print(f"\nwrote {len(jobs)} trajectories under {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
