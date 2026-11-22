"""R2R pair calibration: the PM01 pose that matches the G1 zero-configuration skeleton.

hhtools retargets robot -> robot by closing every IK target on the *target*
robot at one calibrated pose that corresponds to the *source* robot's zero
configuration.  Any limb misalignment at that pose is carried into every frame,
so it should be anatomically exact rather than merely "valid".

``hhtools.retarget.calibration.assistant.propose_calibration_pose`` produces a
valid seed but stops as soon as each limb edge is within 15 degrees.  This
script starts from the same reference and solves the arms exactly:

* upper-arm long axis   G1 shoulder_yaw axis   <->  PM01 J15 / J20 axis
* forearm long axis     G1 wrist_roll axis     <->  PM01 J17 / J22 axis

Four unknowns per arm (shoulder pitch / roll / yaw, elbow pitch) and two unit
directions to match; with the G1 elbow bent 90 degrees at zero the forearm
direction also pins the upper-arm twist.  Legs, waist yaw, forearm twist and
head stay at zero: both robots stand straight with flat feet at zero.

The result is checked with hhtools' own ``assess_calibration_pose`` and saved
with ``save_r2r_calibration`` beside the target URDF.

Run with the hhtools environment:  python calibrate_pair.py [--dry-run]
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np
from scipy.optimize import least_squares

from common import PM01_JOINT_NAMES, SOURCE_ROBOT, TARGET_ROBOT, WORK

# (upper-arm twist joint, forearm twist joint, elbow link, hand link) per robot/side.
G1_ARM = {
    "left": ("left_shoulder_yaw_joint", "left_wrist_roll_joint", "left_elbow_link", "left_rubber_hand"),
    "right": ("right_shoulder_yaw_joint", "right_wrist_roll_joint", "right_elbow_link", "right_rubber_hand"),
}
PM01_ARM = {
    "left": ("J15_SHOULDER_YAW_L", "J17_ELBOW_YAW_L", "LINK_ELBOW_PITCH_L", "LINK_ELBOW_END_L"),
    "right": ("J20_SHOULDER_YAW_R", "J22_ELBOW_YAW_R", "LINK_ELBOW_PITCH_R", "LINK_ELBOW_END_R"),
}
PM01_ARM_JOINTS = {
    "left": ("J13_SHOULDER_PITCH_L", "J14_SHOULDER_ROLL_L", "J15_SHOULDER_YAW_L", "J16_ELBOW_PITCH_L"),
    "right": ("J18_SHOULDER_PITCH_R", "J19_SHOULDER_ROLL_R", "J20_SHOULDER_YAW_R", "J21_ELBOW_PITCH_R"),
}


def _unit(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(_unit(a), _unit(b)), -1.0, 1.0))))


def _link_transform(model, link: str) -> np.ndarray:
    return np.asarray(model.urdf.get_transform(link), dtype=np.float64)


def _joint_axis_world(model, joint_name: str) -> tuple[np.ndarray, np.ndarray]:
    """World direction of a revolute axis and the origin of its child link."""
    joint = model.urdf.joint_map[joint_name]
    child = _link_transform(model, joint.child)
    return _unit(child[:3, :3] @ np.asarray(joint.axis, dtype=np.float64)), child[:3, 3].copy()


def arm_directions(model, spec: tuple[str, str, str, str], joint_q: dict[str, float]) -> tuple[np.ndarray, np.ndarray]:
    """(upper-arm, forearm) unit long axes, oriented proximal -> distal."""
    upper_joint, fore_joint, elbow_link, hand_link = spec
    model.apply_configuration(dict(joint_q))
    upper_axis, upper_origin = _joint_axis_world(model, upper_joint)
    fore_axis, _ = _joint_axis_world(model, fore_joint)
    elbow = _link_transform(model, elbow_link)[:3, 3]
    hand = _link_transform(model, hand_link)[:3, 3]
    shoulder = _link_transform(model, model.urdf.joint_map[upper_joint].parent)[:3, 3]
    del upper_origin
    if np.dot(upper_axis, elbow - shoulder) < 0.0:
        upper_axis = -upper_axis
    if np.dot(fore_axis, hand - elbow) < 0.0:
        fore_axis = -fore_axis
    return upper_axis, fore_axis


def solve_arm(pm01, side: str, target_upper: np.ndarray, target_fore: np.ndarray) -> dict[str, float]:
    names = PM01_ARM_JOINTS[side]
    limits = {j.name: (float(j.limit_lower), float(j.limit_upper)) for j in pm01.actuated_joints}
    lower = np.array([limits[n][0] for n in names])
    upper = np.array([limits[n][1] for n in names])
    zero = pm01.zero_configuration()

    def residual(x: np.ndarray) -> np.ndarray:
        q = dict(zero)
        q.update({n: float(v) for n, v in zip(names, x, strict=True)})
        up, fore = arm_directions(pm01, PM01_ARM[side], q)
        return np.concatenate([up - target_upper, fore - target_fore])

    best = None
    # The zero pose plus a flexed-elbow start; keep the solution nearest zero.
    for start in (np.zeros(4), np.array([0.0, 0.0, 0.0, -1.3]), np.array([0.0, 0.0, 0.0, -0.7])):
        start = np.clip(start, lower + 1e-6, upper - 1e-6)
        sol = least_squares(residual, start, bounds=(lower, upper), xtol=1e-14, ftol=1e-14, gtol=1e-14)
        cost = float(np.linalg.norm(sol.fun))
        key = (round(cost, 9), float(np.linalg.norm(sol.x)))
        if best is None or key < best[0]:
            best = (key, sol.x)
    assert best is not None
    return {n: float(v) for n, v in zip(names, best[1], strict=True)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true", help="Solve and report without saving.")
    args = parser.parse_args()

    from hhtools.retarget.calibration.assistant import (
        assess_calibration_pose,
        propose_calibration_pose,
        render_calibration_preview_png,
    )
    from hhtools.retarget.robot_to_robot import build_source_reference_pose, save_r2r_calibration
    from hhtools.robot.loader import load_robot
    from hhtools.robot.registry import get as get_preset
    from hhtools.robot.registry import refresh

    refresh()
    g1 = load_robot(get_preset(SOURCE_ROBOT), compile_mjcf=False)
    pm01 = load_robot(get_preset(TARGET_ROBOT), compile_mjcf=False)
    assert tuple(pm01.dof_names()) == PM01_JOINT_NAMES

    reference = build_source_reference_pose(g1)
    label = f"robot:{SOURCE_ROBOT}"
    qa_dir = WORK / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)

    proposed, proposed_assessment = propose_calibration_pose(pm01, label, None, reference_pose=reference)

    g1_zero = g1.zero_configuration()
    calibrated = dict(pm01.zero_configuration())
    report: dict[str, dict] = {}
    for side in ("left", "right"):
        target_upper, target_fore = arm_directions(g1, G1_ARM[side], g1_zero)
        calibrated.update(solve_arm(pm01, side, target_upper, target_fore))
        report[side] = {
            "g1_upper_arm_axis": np.round(target_upper, 5).tolist(),
            "g1_forearm_axis": np.round(target_fore, 5).tolist(),
        }
    g1.apply_configuration(g1_zero)

    print(f"{'candidate':28s} {'side':6s} {'upper-arm axis err':>19s} {'forearm axis err':>17s}")
    for name, joint_q in (("PM01 zero", pm01.zero_configuration()), ("hhtools proposal", proposed), ("axis-matched", calibrated)):
        for side in ("left", "right"):
            target_upper, target_fore = arm_directions(g1, G1_ARM[side], g1_zero)
            up, fore = arm_directions(pm01, PM01_ARM[side], joint_q)
            print(f"{name:28s} {side:6s} {_angle_deg(up, target_upper):17.3f} ° {_angle_deg(fore, target_fore):15.3f} °")
            if name == "axis-matched":
                report[side]["upper_arm_axis_error_deg"] = _angle_deg(up, target_upper)
                report[side]["forearm_axis_error_deg"] = _angle_deg(fore, target_fore)
    g1.apply_configuration(g1_zero)
    pm01.apply_configuration(pm01.zero_configuration())

    assessment = assess_calibration_pose(pm01, label, calibrated, reference_pose=reference)
    print("\nhhtools assess_calibration_pose:")
    for name, a in (("hhtools proposal", proposed_assessment), ("axis-matched", assessment)):
        print(
            f"  {name:18s} valid={a.valid} score={a.score:.4f} "
            f"edge_err_deg={ {k: round(v, 2) for k, v in a.edge_errors_deg.items()} }"
        )
        print(
            f"  {'':18s} alignment_errors={a.alignment_errors} warnings={a.alignment_warnings} "
            f"limit_violations={a.limit_violations} near_limits={a.near_limit_joints} "
            f"symmetry_deg={a.symmetry_errors_deg} foot_dz={a.foot_height_delta_m}"
        )
    print("\ncalibrated_joint_q (non-zero):")
    for name in PM01_JOINT_NAMES:
        if abs(calibrated[name]) > 1e-9:
            print(f"  {name:22s} {calibrated[name]:+.6f} rad  ({np.degrees(calibrated[name]):+7.2f} deg)")

    (qa_dir / "calibration_proposed.png").write_bytes(
        render_calibration_preview_png(pm01, label, proposed, proposed_assessment, reference_pose=reference)
    )
    (qa_dir / "calibration_axis_matched.png").write_bytes(
        render_calibration_preview_png(pm01, label, calibrated, assessment, reference_pose=reference)
    )
    (qa_dir / "calibration_report.json").write_text(
        json.dumps(
            {
                "source_robot": SOURCE_ROBOT,
                "target_robot": TARGET_ROBOT,
                "calibrated_joint_q": calibrated,
                "hhtools_proposal_joint_q": proposed,
                "arms": report,
                "assessment": {
                    "valid": assessment.valid,
                    "score": assessment.score,
                    "edge_errors_deg": assessment.edge_errors_deg,
                    "alignment_warnings": list(assessment.alignment_warnings),
                    "symmetry_errors_deg": assessment.symmetry_errors_deg,
                },
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if not assessment.valid:
        print("\nrefined pose rejected by hhtools' validator; not saving", file=sys.stderr)
        return 1
    if args.dry_run:
        print("\n--dry-run: calibration not saved")
        return 0

    assert pm01.preset.urdf_path is not None
    path = save_r2r_calibration(
        pm01.preset.urdf_path.parent,
        target_robot=TARGET_ROBOT,
        source_robot=SOURCE_ROBOT,
        calibrated_joint_q=calibrated,
        notes=(
            "PM01 pose matching the Unitree G1 zero configuration. Arms solved so the "
            "upper-arm (shoulder-yaw) and forearm (G1 wrist-roll / PM01 elbow-yaw) long axes "
            "coincide with G1's; legs, waist, forearm twist and head at zero. "
            "Generated by pm01_retarget/scripts/calibrate_pair.py."
        ),
    )
    print(f"\nsaved R2R calibration -> {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
