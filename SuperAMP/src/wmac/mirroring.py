"""Sagittal motion reflection audited against the actual robot MJCF (G1 or PM01).

The reflection plane is y=constant in the source motion's world frame. Linear
vectors are polar (M v); angular velocities and hinge axes are axial (-M w).
Quaternions use MuJoCo's wxyz convention and represent M R M, a proper rotation.
Flat WMAC ports only this geometry/data utility; no perceptive task is imported.
Original motion files and the robot model are never changed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import hashlib

import mujoco
import numpy as np

from src.robot_profiles import G1, RobotProfile


POLAR_SIGN = np.asarray([1., -1., 1.])
AXIAL_SIGN = np.asarray([-1., 1., -1.])
QUATERNION_SIGN = np.asarray([1., -1., 1., -1.])
MOTION_KEYS = frozenset(("fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w",
                         "body_lin_vel_w", "body_ang_vel_w"))
DEFAULT_G1_XML = Path(__file__).resolve().parents[1]/"assets/robots/unitree_g1/xmls/g1.xml"


def opposite_name(name: str) -> str:
    if name.startswith("left_"):
        return "right_" + name[5:]
    if name.startswith("right_"):
        return "left_" + name[6:]
    return name


@dataclass(frozen=True)
class G1MirrorMap:
    """Left/right reflection of one robot model (first written for the G1)."""

    joint_names: tuple[str, ...]
    body_names: tuple[str, ...]
    joint_permutation: np.ndarray
    joint_sign: np.ndarray
    joint_reference: np.ndarray
    body_permutation: np.ndarray
    xml_sha256: str
    maximum_rest_position_asymmetry: float
    robot: str = "g1"
    # Largest difference between a hinge axis and its counterpart's reflection.
    maximum_axis_asymmetry: float = 0.
    # Joint ranges to check mirrored clips against, for a robot whose limits are
    # not mirror-symmetric; None when the model's limits are symmetric.
    joint_range: np.ndarray | None = None
    # Measured per joint: the angle (rad) between its hinge axis and the reflection of
    # its counterpart's axis, and the reflection of its counterpart's range.
    axis_asymmetry: np.ndarray | None = None
    reflected_joint_range: np.ndarray | None = None

    def metadata(self) -> dict:
        result = {"joint_names": list(self.joint_names), "body_names": list(self.body_names),
                  "joint_permutation": self.joint_permutation.tolist(),
                  "joint_sign": self.joint_sign.tolist(),
                  "joint_reference": self.joint_reference.tolist(),
                  "body_permutation": self.body_permutation.tolist(),
                  "g1_xml_sha256": self.xml_sha256,
                  "maximum_rest_position_asymmetry_m": self.maximum_rest_position_asymmetry,
                  "convention": "MuJoCo native hinge/body order; quaternions wxyz; sagittal world reflection"}
        if self.robot != "g1":
            # Only the G1 keeps its historical key; its manifests stay byte-identical.
            result["robot_xml_sha256"] = result.pop("g1_xml_sha256")
            result.update(robot=self.robot, maximum_axis_asymmetry=self.maximum_axis_asymmetry,
                          symmetric_joint_limits=self.joint_range is None)
            result.update(self.measured_asymmetry())
        return result

    def measured_asymmetry(self) -> dict:
        """What the model's left and right sides differ by, as measured from the MJCF."""
        angles = np.degrees(self.axis_asymmetry)
        ranges = {}
        if self.joint_range is not None:
            for index, name in enumerate(self.joint_names):
                own, reflected = self.joint_range[index], self.reflected_joint_range[index]
                if not np.allclose(own, reflected, rtol=0, atol=1e-6):
                    ranges[name] = {"counterpart": self.joint_names[self.joint_permutation[index]],
                                    "range_rad": own.tolist(),
                                    "reflected_counterpart_range_rad": reflected.tolist()}
        return {
            "hinge_axis_asymmetry_definition": "angle between a hinge axis and the sagittal reflection "
                                               "of its left/right counterpart's axis, in degrees",
            "maximum_hinge_axis_asymmetry_deg": float(angles.max()),
            "hinge_axis_asymmetry_deg": {name: float(angle) for name, angle in zip(self.joint_names, angles)},
            "asymmetric_joint_ranges": ranges,
        }


MirrorMap = G1MirrorMap


def build_g1_mirror_map(xml_path: Path | str = DEFAULT_G1_XML) -> G1MirrorMap:
    """Derive signs from actual hinge axes, and reject incompatible topology.

    The supplied G1 has a 10 micrometre shoulder-offset asymmetry. It is
    retained in the audit instead of silently changing the robot model.
    """
    return build_mirror_map(G1, xml_path)


def build_mirror_map(profile: RobotProfile, xml_path: Path | str | None = None) -> G1MirrorMap:
    """The same audit for any supported robot, with that robot's naming and tolerances.

    Asymmetries of the supplied model (for the PM01: elbow-pitch axes that differ
    from a mirror pair by 0.31 degrees, and a one-sided waist-yaw range) are
    measured and recorded, never corrected in the model.
    """
    path = Path(profile.xml_path if xml_path is None else xml_path).resolve()
    model = mujoco.MjModel.from_xml_path(str(path))
    hinge_ids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    joints, bodies = profile.num_joints, profile.num_bodies
    if model.nq != 7 + joints or len(hinge_ids) != joints or model.nbody != bodies + 1:
        raise ValueError(f"Expected the native {profile.name.upper()} {joints}-hinge / "
                         f"{bodies}-nonworld-body motion layout")
    joint_names = tuple(model.joint(int(i)).name for i in hinge_ids)
    body_names = tuple(model.body(i).name for i in range(1, model.nbody))
    if any(not name for name in (*joint_names, *body_names)):
        raise ValueError("Every joint and nonworld body must have an unambiguous name")
    try:
        jp = np.asarray([joint_names.index(profile.counterpart(n, joint_names)) for n in joint_names], dtype=np.int64)
        bp = np.asarray([body_names.index(profile.counterpart(n, body_names)) for n in body_names], dtype=np.int64)
    except ValueError as error:
        raise ValueError("A left/right counterpart is missing in the robot model") from error
    if not np.array_equal(jp[jp], np.arange(joints)) or not np.array_equal(bp[bp], np.arange(bodies)):
        raise ValueError("Robot left/right permutations are not involutions")
    axes = model.jnt_axis[hinge_ids]
    reflected_axes = axes[jp] * AXIAL_SIGN
    signs = np.sum(reflected_axes * axes, axis=-1)
    axis_tolerance = profile.mirror_axis_tolerance
    # Unit vectors that differ by d have a dot product of 1 - d^2 / 2.
    if not np.allclose(np.abs(signs), 1., rtol=0, atol=max(1e-9, axis_tolerance**2)):
        raise ValueError("Hinge axes do not support the expected sagittal reflection")
    signs = np.where(signs > 0, 1., -1.)
    if not np.allclose(reflected_axes, axes * signs[:, None], rtol=0, atol=axis_tolerance):
        raise ValueError("Reflected hinge axes are not collinear with counterpart axes")
    axis_difference = np.linalg.norm(reflected_axes - axes * signs[:, None], axis=-1)
    axis_error = float(np.max(axis_difference))
    # Unit vectors that differ by d enclose the angle 2 asin(d / 2).
    axis_angles = 2 * np.arcsin(np.clip(axis_difference / 2, 0., 1.))
    limits = model.jnt_range[hinge_ids]
    mapped_limits = np.sort(limits[jp] * signs[:, None], axis=-1)
    if profile.mirror_requires_symmetric_limits:
        if not np.allclose(mapped_limits, limits, rtol=0, atol=1e-6):
            raise ValueError("Mirrored joint limits differ from the counterpart robot limits")
        joint_range = None
    else:
        joint_range = limits.copy()
        joint_range.setflags(write=False)
    body_ids = np.arange(1, model.nbody)
    mirrored_body_ids = bp + 1
    lookup = {int(body): i for i, body in enumerate(body_ids)}
    for i, body_id in enumerate(body_ids):
        parent = int(model.body_parentid[body_id])
        reflected_parent = 0 if parent == 0 else int(mirrored_body_ids[lookup[parent]])
        if int(model.body_parentid[mirrored_body_ids[i]]) != reflected_parent:
            raise ValueError("Left/right body trees do not have mirrored parent relationships")
    rest_error = float(np.max(np.abs(model.body_pos[mirrored_body_ids] - model.body_pos[body_ids]*POLAR_SIGN)))
    if rest_error > 2e-5:
        raise ValueError(f"Robot rest offsets are too asymmetric for reflection: {rest_error}m")
    reflected_q = model.body_quat[body_ids] * QUATERNION_SIGN
    orientation_agreement = np.abs(np.sum(reflected_q * model.body_quat[mirrored_body_ids], axis=-1))
    if not np.allclose(orientation_agreement, 1., rtol=0, atol=1e-8):
        raise ValueError("Robot rest-frame rotations do not obey sagittal reflection")
    pivots = model.jnt_pos[hinge_ids]
    if not np.allclose(pivots[jp]*POLAR_SIGN, pivots, rtol=0, atol=1e-9):
        raise ValueError("Joint pivots do not obey sagittal reflection")
    qref = model.qpos0[model.jnt_qposadr[hinge_ids]].copy()
    for array in (jp, bp, signs, qref, axis_angles, mapped_limits):
        array.setflags(write=False)
    return G1MirrorMap(joint_names, body_names, jp, signs, qref, bp,
                       hashlib.sha256(path.read_bytes()).hexdigest(), rest_error,
                       robot=profile.name, maximum_axis_asymmetry=axis_error, joint_range=joint_range,
                       axis_asymmetry=axis_angles, reflected_joint_range=mapped_limits)


def reflect_polar(values: np.ndarray) -> np.ndarray:
    if values.shape[-1] != 3:
        raise ValueError("A polar vector must have three components")
    return values * POLAR_SIGN.astype(values.dtype)


def reflect_axial(values: np.ndarray) -> np.ndarray:
    if values.shape[-1] != 3:
        raise ValueError("An axial vector must have three components")
    return values * AXIAL_SIGN.astype(values.dtype)


def reflect_quaternion_wxyz(values: np.ndarray) -> np.ndarray:
    if values.shape[-1] != 4:
        raise ValueError("A wxyz quaternion must have four components")
    return values * QUATERNION_SIGN.astype(values.dtype)


def validate_motion(motion: dict[str, np.ndarray], mapping: G1MirrorMap) -> int:
    if set(motion) != MOTION_KEYS:
        raise ValueError(f"Motion schema differs: missing={sorted(MOTION_KEYS-set(motion))}, "
                         f"unsupported={sorted(set(motion)-MOTION_KEYS)}")
    frames = len(motion["joint_pos"])
    if frames < 2:
        raise ValueError("A motion clip must contain at least two consecutive frames")
    expected = {"joint_pos": (frames, len(mapping.joint_names)),
                "joint_vel": (frames, len(mapping.joint_names)),
                "body_pos_w": (frames, len(mapping.body_names), 3),
                "body_quat_w": (frames, len(mapping.body_names), 4),
                "body_lin_vel_w": (frames, len(mapping.body_names), 3),
                "body_ang_vel_w": (frames, len(mapping.body_names), 3)}
    for name, shape in expected.items():
        values = motion[name]
        if values.shape != shape or not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
            raise ValueError(f"Invalid motion tensor {name}: expected finite floating {shape}")
    fps = np.asarray(motion["fps"])
    if fps.size != 1 or not np.isfinite(fps).all() or float(fps.reshape(-1)[0]) <= 0:
        raise ValueError("Motion fps must be one finite positive scalar")
    if not np.allclose(np.linalg.norm(motion["body_quat_w"], axis=-1), 1., rtol=0, atol=2e-4):
        raise ValueError("Motion quaternions must be normalized wxyz rotations")
    return frames


def mirror_motion(motion: dict[str, np.ndarray], mapping: G1MirrorMap | None = None,
                  *, plane_y: float | None = None) -> dict[str, np.ndarray]:
    """Return a fresh reflected clip, keeping its fps, duration and frame order.

    The default plane passes through the first pelvis position, preserving the
    initial world anchor. No temporal reversal, resampling or clip concatenation
    is performed. Original arrays are never modified.
    """
    mapping = build_g1_mirror_map() if mapping is None else mapping
    validate_motion(motion, mapping)
    if plane_y is None:
        plane_y = float(motion["body_pos_w"][0, 0, 1])
    if not np.isfinite(plane_y):
        raise ValueError("Reflection plane must be finite")
    jp, bp = mapping.joint_permutation, mapping.body_permutation
    q = motion["joint_pos"]
    sign = mapping.joint_sign.astype(q.dtype)
    ref = mapping.joint_reference.astype(q.dtype)
    pos = reflect_polar(motion["body_pos_w"][:, bp])
    pos[..., 1] += np.asarray(2*plane_y, dtype=pos.dtype)
    mirrored = {
        "fps": np.array(motion["fps"], copy=True),
        "joint_pos": (q[:, jp]-ref[jp])*sign+ref,
        "joint_vel": motion["joint_vel"][:, jp]*mapping.joint_sign.astype(motion["joint_vel"].dtype),
        "body_pos_w": pos,
        "body_quat_w": reflect_quaternion_wxyz(motion["body_quat_w"][:, bp]),
        "body_lin_vel_w": reflect_polar(motion["body_lin_vel_w"][:, bp]),
        "body_ang_vel_w": reflect_axial(motion["body_ang_vel_w"][:, bp]),
    }
    validate_motion(mirrored, mapping)
    if mapping.joint_range is not None:
        low, high = mapping.joint_range[:, 0], mapping.joint_range[:, 1]
        if ((mirrored["joint_pos"] < low - 1e-6) | (mirrored["joint_pos"] > high + 1e-6)).any():
            raise ValueError("A mirrored clip leaves the robot's joint limits, which are not mirror-symmetric")
    return mirrored


def turn_statistics(motions: list[dict[str, np.ndarray]], threshold: float = .1) -> dict:
    """Frame/duration statistics on pelvis world yaw angular velocity, rad/s."""
    if threshold < 0 or not np.isfinite(threshold):
        raise ValueError("Turn threshold must be finite and nonnegative")
    left = right = neutral = frames = 0
    duration = yaw_integral = left_seconds = right_seconds = 0.
    for motion in motions:
        wz = np.asarray(motion["body_ang_vel_w"][:, 0, 2], dtype=np.float64)
        fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
        nleft, nright = int((wz > threshold).sum()), int((wz < -threshold).sum())
        left += nleft; right += nright; neutral += len(wz)-nleft-nright; frames += len(wz)
        duration += len(wz)/fps; yaw_integral += float(wz.sum())/fps
        left_seconds += nleft/fps; right_seconds += nright/fps
    return {"clips": len(motions), "frames": frames, "duration_seconds": duration,
            "positive_yaw_frames": left, "negative_yaw_frames": right, "neutral_yaw_frames": neutral,
            "positive_yaw_seconds": left_seconds, "negative_yaw_seconds": right_seconds,
            "yaw_angular_velocity_integral_rad": yaw_integral,
            "mean_yaw_angular_velocity_rad_s": yaw_integral/duration if duration else 0.,
            "threshold_rad_s": threshold,
            "definition": "Pelvis world angular velocity z; +yaw is left/CCW on upright z-up walking. Not a path-planning success metric."}
