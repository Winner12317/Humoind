"""Verify the assumptions made about SuperAMP's G1 reference data.

1. CSV conventions: resampling ``motion_data_csv/amp/<clip>.csv`` at 120 Hz with
   SuperAMP's own csv_to_npz interpolation must reproduce the shipped 50 Hz NPZ
   (root position, root quaternion, joint order).

   Root position and joints agree to float precision.  The root quaternion
   agrees only to ~0.2 deg: mjlab's float32 ``quat_slerp`` returns the first
   quaternion unchanged whenever ``|dot| - 1`` is within 4 eps, which at 120 Hz
   is true for a large share of frame pairs, so the shipped NPZs hold the
   previous sample's orientation on those frames.  Exact slerp is used here.
2. Source model: forward kinematics of hhtools' ``g1_29dof`` preset must land on
   the body poses SuperAMP's own G1 MJCF produced (``body_pos_w`` /
   ``body_quat_w``), so the keypoints hhtools derives are the ones SuperAMP
   trained against.

   Every body orientation and the pelvis + leg positions agree exactly.  The
   two G1 descriptions place the waist roll / pitch axes differently (Unitree
   URDF: z = 0.035 and 0.054 above the pelvis; SuperAMP MJCF: both at 0.044),
   so ``waist_roll_link`` / ``torso_link`` origins sit 9-10 mm apart and the
   arms shift by a few millimetres depending on the waist angles.

Run with the hhtools environment:  python check_g1_source.py
"""

from __future__ import annotations

import sys

import numpy as np

from common import (
    G1_CSV_DIR,
    G1_CSV_FPS,
    G1_JOINT_NAMES,
    G1_NPZ_DIR,
    G1_NPZ_FPS,
    SOURCE_ROBOT,
    g1_mjcf_body_names,
    load_g1_csv,
    load_g1_npz_as_joint_q,
    quat_xyzw_to_matrix,
    resample_like_csv_to_npz,
)


def quat_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dot = np.abs(np.sum(a * b, axis=-1) / (np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)))
    return np.degrees(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))


def check_csv_against_npz() -> bool:
    print("== 1. CSV @120 Hz resampled to 50 Hz vs shipped NPZ ==")
    print(f"{'clip':40s} {'csv':>5s} {'npz':>5s} {'resamp':>6s} {'|dpos| mm':>10s} {'drot deg':>9s} {'|dq| rad':>9s}")
    ok = True
    for csv_path in sorted(G1_CSV_DIR.glob("*.csv")):
        npz_path = G1_NPZ_DIR / "WalkandRun" / f"{csv_path.stem}.npz"
        csv_q = load_g1_csv(csv_path)
        npz_q, fps = load_g1_npz_as_joint_q(npz_path)
        assert fps == G1_NPZ_FPS, (npz_path, fps)
        resampled = resample_like_csv_to_npz(csv_q, G1_CSV_FPS, G1_NPZ_FPS)
        same_len = resampled.shape[0] == npz_q.shape[0]
        n = min(resampled.shape[0], npz_q.shape[0])
        dpos = np.abs(resampled[:n, :3] - npz_q[:n, :3]).max() * 1e3
        drot = quat_angle_deg(resampled[:n, 3:7], npz_q[:n, 3:7]).max()
        dq = np.abs(resampled[:n, 7:] - npz_q[:n, 7:]).max()
        good = same_len and dpos < 0.01 and drot < 0.25 and dq < 2e-5
        ok &= good
        print(
            f"{csv_path.stem:40s} {csv_q.shape[0]:5d} {npz_q.shape[0]:5d} {resampled.shape[0]:6d} "
            f"{dpos:10.4f} {drot:9.5f} {dq:9.2e} {'ok' if good else 'MISMATCH'}"
        )
    return ok


def check_hhtools_g1_fk() -> bool:
    from hhtools.retarget.calibration.calibration import _collect_link_transforms_at_q
    from hhtools.robot.loader import load_robot
    from hhtools.robot.registry import get as get_preset
    from hhtools.robot.registry import refresh

    print("\n== 2. hhtools g1_29dof FK vs SuperAMP body_pos_w ==")
    refresh()
    model = load_robot(get_preset(SOURCE_ROBOT), compile_mjcf=False)
    assert tuple(model.dof_names()) == G1_JOINT_NAMES, "hhtools G1 dof order differs from SuperAMP CSV order"
    print(f"hhtools {SOURCE_ROBOT} dof order == SuperAMP CSV joint order: True ({len(G1_JOINT_NAMES)} joints)")

    body_names = g1_mjcf_body_names()
    links = set(model.link_names())
    missing = [b for b in body_names if b not in links]
    print(f"SuperAMP MJCF bodies: {len(body_names)}; missing from hhtools URDF: {missing or 'none'}")

    worst_overall = 0.0
    worst_rot_overall = 0.0
    ok = True
    saved = model.zero_configuration()
    try:
        for npz_path in sorted(G1_NPZ_DIR.glob("*/*.npz")):
            with np.load(npz_path, allow_pickle=False) as data:
                body_pos = np.asarray(data["body_pos_w"], dtype=np.float64)
                body_quat_wxyz = np.asarray(data["body_quat_w"], dtype=np.float64)
                joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
            assert body_pos.shape[1] == len(body_names)
            frames = np.unique(np.linspace(0, body_pos.shape[0] - 1, 25).astype(int))
            worst = 0.0
            worst_rot = 0.0
            worst_body = ""
            worst_lower = 0.0
            worst_arm = 0.0
            for f in frames:
                cfg = {name: float(joint_pos[f, i]) for i, name in enumerate(G1_JOINT_NAMES)}
                link_tx = _collect_link_transforms_at_q(model, cfg)
                root = np.eye(4)
                root[:3, :3] = quat_xyzw_to_matrix(body_quat_wxyz[f, 0, [1, 2, 3, 0]])
                root[:3, 3] = body_pos[f, 0]
                for b, name in enumerate(body_names):
                    world = root @ link_tx[name]
                    err = float(np.linalg.norm(world[:3, 3] - body_pos[f, b]))
                    ref_rot = quat_xyzw_to_matrix(body_quat_wxyz[f, b, [1, 2, 3, 0]])
                    cos = (np.trace(world[:3, :3].T @ ref_rot) - 1.0) / 2.0
                    rot_err = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
                    if err > worst:
                        worst, worst_body = err, name
                    worst_rot = max(worst_rot, rot_err)
                    if name == "pelvis" or name == "waist_yaw_link" or "hip" in name or "knee" in name or "ankle" in name:
                        worst_lower = max(worst_lower, err)
                    elif "shoulder" in name or "elbow" in name or "wrist" in name:
                        worst_arm = max(worst_arm, err)
            worst_overall = max(worst_overall, worst)
            worst_rot_overall = max(worst_rot_overall, worst_rot)
            # Legs must be exact; the waist-axis placement difference bounds the rest.
            good = worst_lower < 1e-5 and worst_rot < 0.01 and worst < 0.012 and worst_arm < 0.008
            ok &= good
            print(
                f"{npz_path.parent.name + '/' + npz_path.stem:52s} frames={len(frames):2d} "
                f"pelvis+legs {worst_lower * 1e3:6.4f} mm | arms {worst_arm * 1e3:5.2f} mm | "
                f"worst {worst * 1e3:6.3f} mm ({worst_body}) | rot {worst_rot:6.4f} deg "
                f"{'ok' if good else 'MISMATCH'}"
            )
    finally:
        model.apply_configuration(saved)
    print(f"worst over all clips: {worst_overall * 1e3:.4f} mm, {worst_rot_overall:.4f} deg")
    return ok


def main() -> int:
    ok_csv = check_csv_against_npz()
    ok_fk = check_hhtools_g1_fk()
    print(f"\nCSV conventions verified: {ok_csv}\nhhtools G1 model matches SuperAMP G1: {ok_fk}")
    return 0 if (ok_csv and ok_fk) else 1


if __name__ == "__main__":
    sys.exit(main())
