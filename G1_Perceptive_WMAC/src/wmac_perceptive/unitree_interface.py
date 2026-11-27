"""Run a Hiking-convention perceptive policy on a Unitree-convention G1.

The training robot uses Hiking's torso-rooted URDF. Its joint order is the
Unitree order, but its three waist hinges run from the torso to the pelvis, so

* every waist angle (yaw, roll, pitch) has the opposite sign to Unitree's
  MJCF/SDK convention: ``q_policy = -q_unitree`` (exact for orientation, because
  both chains compose as Rz(yaw) Rx(roll) Ry(pitch));
* angular velocity and projected gravity are measured in the **torso_link**
  frame, not the pelvis IMU and not Unitree MJCF's ``secondary_imu`` site (that
  site sits on waist_roll_link, so it does not follow waist pitch).

In the training simulator, feeding unconverted waist angles or a pelvis IMU makes
the policy fall in 100% of episodes; using ``secondary_imu`` raises falls from
17% to 68%. This module performs the conversions and builds the exact policy
input. It depends only on NumPy so it can be copied into a deployment harness.

Typical MuJoCo sim-to-sim loop at 50 Hz (physics 200 Hz, decimation 4)::

    iface = UnitreeG1PolicyInterface()
    iface.reset(**iface.mujoco_measurements(model, data), command=cmd,
                depth_frame=iface.depth_frame(depth_m))
    while True:
        packed = iface.policy_input()                   # [1, 768 + 37*2*36*64]
        action = onnx_session.run(None, {"proprio_and_depth_ring": packed})[0][0]
        targets = iface.unitree_targets(action)         # Unitree joint convention
        for _ in range(4):
            data.ctrl[:] = iface.pd_torques(targets, *iface.unitree_joint_state(model, data))
            mujoco.mj_step(model, data)
        iface.step(**iface.mujoco_measurements(model, data), command=cmd, action=action,
                   depth_frame=iface.depth_frame(render_depth()))

On the robot the depth frame comes from the RealSense stream instead::

    camera = RealSenseDepthAdapter()                    # G1 480x270 @ 60 Hz -> training camera
    depth_frame = camera.frame(raw_z16)                 # newest image, once per 50 Hz step
"""
from __future__ import annotations

import json
import math
from pathlib import Path
import re

import numpy as np

POLICY_JOINT_NAMES = (
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint", "left_knee_joint",
    "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint", "right_knee_joint",
    "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint", "left_elbow_joint",
    "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint", "right_elbow_joint",
    "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
)
WAIST_JOINT_NAMES = ("waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint")
# Unitree joint value = sign * policy joint value (and the same for velocities/targets).
UNITREE_JOINT_SIGN = np.array([-1. if name in WAIST_JOINT_NAMES else 1. for name in POLICY_JOINT_NAMES])
# Hiking/InstinctLab G1 default pose (identical to mjlab's KNEES_BENT keyframe).
_DEFAULT_POSE = {
    r".*_hip_pitch_joint": -.312, r".*_knee_joint": .669, r".*_ankle_pitch_joint": -.363,
    r".*_elbow_joint": .6, "left_shoulder_roll_joint": .2, "left_shoulder_pitch_joint": .2,
    "right_shoulder_roll_joint": -.2, "right_shoulder_pitch_joint": .2,
}
PROPRIO_HISTORY = 8
PROPRIO_TERM_WIDTHS = (3, 3, 3, 29, 29, 29)  # gyro*.25, gravity, command, q-q0, dq*.05, last action
ANGULAR_VELOCITY_SCALE, JOINT_VELOCITY_SCALE = .25, .05
DEPTH_RING, DEPTH_HEIGHT, DEPTH_WIDTH = 37, 36, 64
DEPTH_MAX_M, DEPTH_MIN_RAY_M, DEPTH_MISSING_M = 2.5, .1, 1e6
# The policy crops each frame to rows 18-35, columns 16-47 before its blur.
DEPTH_CROP_ROWS, DEPTH_CROP_COLS = slice(18, 36), slice(16, 48)
# Hiking's training camera (version 4 default, "hiking_depth_only_v1"): a
# pinhole with the principal point at the image centre, mounted in torso_link
# with +X forward, +Y left, +Z up; image-plane depth along +X.
HIKING_CAMERA_FOV_DEGREES = (89.51, 58.29)
HIKING_CAMERA_POSITION = (.0487988662332928, .01, .4378029937970051)
HIKING_CAMERA_ROTATION_WXYZ = (.9135367613482678, .004363309284746571, .4067366430758002, 0.)
# Hiking's robot code replaces every pixel nearer than this (which includes
# RealSense's invalid zeros) from its neighbours before blurring.
REALSENSE_HOLE_DEPTH_M = .2
G1_DEPLOYMENT_DEPTH_CALIBRATION = (Path(__file__).resolve().parents[1]
                                   / "assets/cameras/g1_depth_calibration_480x270_60hz.json")
CONTROL_DT, DECIMATION = .02, 4
# Hiking's torso_link origin is 10 mm lower in the torso shell than Unitree's: the
# head joint is at z=-0.044 m (Hiking URDF) vs -0.054 m (unitree_mujoco). A point
# fixed to the torso therefore has Unitree torso coordinates = Hiking ones + this.
HIKING_TO_UNITREE_TORSO_OFFSET = np.array([0., 0., -.010])


def default_joint_positions() -> np.ndarray:
    """Default pose in the policy (Hiking) convention; waist entries are zero."""
    out = np.zeros(len(POLICY_JOINT_NAMES))
    for i, name in enumerate(POLICY_JOINT_NAMES):
        for pattern, value in _DEFAULT_POSE.items():
            if re.fullmatch(pattern, name):
                out[i] = value
    return out


def joint_parameters() -> dict[str, np.ndarray]:
    """Training PD gains, effort limits, armature and action scale per policy joint."""
    from .hiking_actuators import _joint_parameters
    rows = [_joint_parameters(name) for name in POLICY_JOINT_NAMES]
    params = {key: np.array([row[key] for row in rows], dtype=float)
              for key in ("kp", "kd", "effort_limit", "velocity_limit", "armature")}
    params["action_scale"] = .25 * params["effort_limit"] / params["kp"]
    return params


# Quaternions are (w, x, y, z).
def quat_multiply(a, b):
    aw, ax, ay, az = np.moveaxis(np.asarray(a, float), -1, 0)
    bw, bx, by, bz = np.moveaxis(np.asarray(b, float), -1, 0)
    return np.stack((aw*bw - ax*bx - ay*by - az*bz, aw*bx + ax*bw + ay*bz - az*by,
                     aw*by - ax*bz + ay*bw + az*bx, aw*bz + ax*by - ay*bx + az*bw), -1)


def quat_rotate_inverse(q, v):
    """Express world vector v in the frame whose world orientation is q."""
    q = np.asarray(q, float)
    conj = q * np.array([1., -1., -1., -1.])
    pure = np.concatenate((np.zeros(np.shape(v)[:-1] + (1,)), np.asarray(v, float)), -1)
    return quat_multiply(quat_multiply(conj, pure), q)[..., 1:]


def _axis_quat(axis, angle):
    half = .5 * np.asarray(angle, float)
    out = np.zeros(np.shape(half) + (4,))
    out[..., 0] = np.cos(half)
    out[..., 1 + axis] = np.sin(half)
    return out


def torso_from_pelvis(pelvis_quat, pelvis_gyro, waist_q_unitree, waist_dq_unitree):
    """Torso orientation and torso-frame angular velocity from a pelvis IMU.

    Uses Unitree's waist chain pelvis -> Rz(yaw) -> Rx(roll) -> Ry(pitch) -> torso
    with waist angles/rates in the **Unitree** convention (yaw, roll, pitch).
    ``pelvis_gyro`` is in the pelvis frame. Returns (torso_quat_world, torso_gyro).
    """
    yaw, roll, pitch = np.moveaxis(np.asarray(waist_q_unitree, float), -1, 0)
    dyaw, droll, dpitch = np.moveaxis(np.asarray(waist_dq_unitree, float), -1, 0)
    rz, rx, ry = _axis_quat(2, yaw), _axis_quat(0, roll), _axis_quat(1, pitch)
    relative = quat_multiply(quat_multiply(rz, rx), ry)
    torso = quat_multiply(np.asarray(pelvis_quat, float), relative)
    zeros = np.zeros_like(np.asarray(dyaw, float))
    # Joint-rate contributions, each expressed in the torso frame.
    yaw_axis = quat_rotate_inverse(quat_multiply(rx, ry), np.stack((zeros, zeros, dyaw), -1))
    roll_axis = quat_rotate_inverse(ry, np.stack((droll, zeros, zeros), -1))
    pitch_axis = np.stack((zeros, dpitch, zeros), -1)
    gyro = quat_rotate_inverse(relative, np.asarray(pelvis_gyro, float)) + yaw_axis + roll_axis + pitch_axis
    return torso, gyro


def projected_gravity(quat):
    return quat_rotate_inverse(quat, np.broadcast_to(np.array([0., 0., -1.]), np.shape(quat)[:-1] + (3,)))


def depth_frame(image_plane_depth_m, ray_distance_m=None):
    """One [2, 36, 64] policy frame from a 64x36 image-plane depth image in metres.

    Pixels without a hit (inf/nan/<=0), or whose *ray* distance is at most 0.1 m,
    become 1e6 m. Do not clip before the policy: its own blur runs on the
    unclipped values, then clips to [0, 2.5] m. Channel 1 (validity) is not
    used by the policy and is provided for diagnostics only.
    """
    depth = np.asarray(image_plane_depth_m, dtype=np.float32)
    if depth.shape != (DEPTH_HEIGHT, DEPTH_WIDTH):
        raise ValueError(f"Expected a {DEPTH_HEIGHT}x{DEPTH_WIDTH} depth image")
    ray = depth if ray_distance_m is None else np.asarray(ray_distance_m, dtype=np.float32)
    hit = np.isfinite(depth) & (depth > 0) & np.isfinite(ray) & (ray > DEPTH_MIN_RAY_M)
    depth = np.where(hit, np.minimum(depth, DEPTH_MISSING_M), DEPTH_MISSING_M)
    valid = hit & (depth <= DEPTH_MAX_M)
    return np.stack((depth / DEPTH_MAX_M, valid.astype(np.float32))).astype(np.float32)


class UnitreeG1PolicyInterface:
    """Exact policy input (8-step term-major proprioception + 37-frame depth ring)."""

    def __init__(self):
        self.default_q = default_joint_positions()
        self.params = joint_parameters()
        self.sign = UNITREE_JOINT_SIGN.copy()
        self._proprio = None
        self._ring = None

    # Conversions -------------------------------------------------------------
    def policy_joint_state(self, q_unitree, dq_unitree):
        return self.sign * np.asarray(q_unitree, float), self.sign * np.asarray(dq_unitree, float)

    def unitree_targets(self, action):
        """Joint position targets for the Unitree-convention robot."""
        return self.sign * (self.default_q + self.params["action_scale"] * np.asarray(action, float))

    def pd_torques(self, targets_unitree, q_unitree, dq_unitree):
        """Training PD law (sign-invariant), clipped to the training effort limits."""
        torque = self.params["kp"] * (targets_unitree - q_unitree) - self.params["kd"] * dq_unitree
        return np.clip(torque, -self.params["effort_limit"], self.params["effort_limit"])

    # MuJoCo helpers (Unitree G1 MJCF) ---------------------------------------
    @staticmethod
    def unitree_joint_state(model, data):
        import mujoco
        ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) for name in POLICY_JOINT_NAMES]
        if min(ids) < 0:
            raise ValueError("MuJoCo model lacks a policy joint")
        q = np.array([data.qpos[model.jnt_qposadr[i]] for i in ids])
        dq = np.array([data.qvel[model.jnt_dofadr[i]] for i in ids])
        return q, dq

    @classmethod
    def mujoco_measurements(cls, model, data, *, torso_body="torso_link"):
        """Torso orientation/gyro read directly from the simulated torso_link body."""
        import mujoco
        body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, torso_body)
        velocity = np.zeros(6)
        mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, body, velocity, 1)
        q, dq = cls.unitree_joint_state(model, data)
        return dict(torso_quat=data.xquat[body].copy(), torso_gyro=velocity[:3].copy(),
                    q_unitree=q, dq_unitree=dq)

    # Observation ---------------------------------------------------------------
    def proprio_frame(self, torso_quat, torso_gyro, q_unitree, dq_unitree, command, last_action):
        """One 96-D sample; leading batch dimensions are allowed."""
        q, dq = self.policy_joint_state(q_unitree, dq_unitree)
        return np.concatenate((np.asarray(torso_gyro, float) * ANGULAR_VELOCITY_SCALE,
                               projected_gravity(torso_quat), np.asarray(command, float),
                               q - self.default_q, dq * JOINT_VELOCITY_SCALE,
                               np.asarray(last_action, float)), axis=-1).astype(np.float32)

    def reset(self, *, torso_quat, torso_gyro, q_unitree, dq_unitree, command, depth_frame):
        """Start an episode: last action is zero, both histories backfill with this sample."""
        frame = self.proprio_frame(torso_quat, torso_gyro, q_unitree, dq_unitree, command, np.zeros(29))
        self._proprio = np.repeat(frame[None], PROPRIO_HISTORY, 0)
        self._ring = np.repeat(np.asarray(depth_frame, np.float32)[None], DEPTH_RING, 0)

    def step(self, *, torso_quat, torso_gyro, q_unitree, dq_unitree, command, action, depth_frame):
        """Append the post-step sample. ``action`` is the raw policy output just applied."""
        if self._proprio is None:
            raise RuntimeError("Call reset() first")
        frame = self.proprio_frame(torso_quat, torso_gyro, q_unitree, dq_unitree, command, action)
        self._proprio = np.concatenate((self._proprio[1:], frame[None]))
        self._ring = np.concatenate((self._ring[1:], np.asarray(depth_frame, np.float32)[None]))

    def proprio_history(self):
        """[768] in the training layout: each term's 8 samples oldest->newest, term by term."""
        terms = np.split(self._proprio, np.cumsum(PROPRIO_TERM_WIDTHS)[:-1], axis=-1)
        return np.concatenate([term.reshape(-1) for term in terms]).astype(np.float32)

    def policy_input(self):
        """[1, 768 + 37*2*36*64] float32 for the exported ONNX ``proprio_and_depth_ring``."""
        return np.concatenate((self.proprio_history(), self._ring.reshape(-1)))[None].astype(np.float32)

    depth_frame = staticmethod(depth_frame)


def _rotation_matrix(wxyz):
    w, x, y, z = np.asarray(wxyz, dtype=np.float64) / np.linalg.norm(wxyz)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def fill_depth_holes(depth_m, hole_depth_m=REALSENSE_HOLE_DEPTH_M):
    """Replace unmeasured or too-near pixels from their measured neighbours.

    A hole takes the mean of the measured pixels among its eight neighbours;
    the fill then advances inwards one ring at a time. This is the NumPy
    counterpart of the ``cv2.inpaint`` call in Hiking's robot code (same hole
    rule, depth below 0.2 m; not bit-identical to its Navier-Stokes fill).
    Returns the filled image and the hole mask. An image without any
    measurement is returned unchanged.
    """
    depth = np.array(depth_m, dtype=np.float32)
    holes = ~(np.isfinite(depth) & (depth >= hole_depth_m))
    known = ~holes
    if not known.any():
        return depth, holes
    depth[holes] = 0.
    while not known.all():
        values, counts = np.pad(depth * known, 1), np.pad(known, 1).astype(np.float32)
        rows, cols = depth.shape
        total = sum(values[i:i + rows, j:j + cols] for i in range(3) for j in range(3))
        count = sum(counts[i:i + rows, j:j + cols] for i in range(3) for j in range(3))
        ring = ~known & (count > 0)
        depth[ring] = total[ring] / count[ring]
        known = known | ring
    return depth, holes


class RealSenseDepthAdapter:
    """Turn a raw RealSense depth image into the frame the policy was trained on.

    The policy saw Hiking's pinhole camera (64 x 36, 89.51 x 58.29 degrees,
    principal point at the centre). A RealSense image has its own focal length
    and an off-centre principal point, so resizing it is not the same picture.
    Both cameras share one optical centre, so the exact conversion is a fixed
    lookup: every training pixel reads the one RealSense pixel on its ray
    (nearest pixel, as the simulated camera samples one ray per pixel).

    ``calibration`` is the depth stream's intrinsics: a mapping or JSON file with
    ``width, height, fx, fy, cx, cy, depth_scale_m_per_raw_unit`` and zero
    distortion (RealSense depth is rectified). It defaults to the G1's
    480 x 270, 60 Hz profile.

    ``mount_rotation_wxyz`` is the measured camera orientation in torso_link
    (+X forward, +Y left, +Z up), if it differs from Hiking's nominal mount
    ``HIKING_CAMERA_ROTATION_WXYZ``. The lookup then re-aims every ray and
    converts depth to the training camera's image plane, which is exact for a
    rotation. A position offset cannot be removed by a lookup; retrain with the
    measured position if it differs by more than about a centimetre.

    At deployment call ``frame(raw)`` on the newest camera image once per
    50 Hz control step. A 60 Hz stream always has a new image for every step.
    """

    def __init__(self, calibration=None, *, mount_rotation_wxyz=None, fill_holes=True):
        if calibration is None:
            calibration = G1_DEPLOYMENT_DEPTH_CALIBRATION
        if not isinstance(calibration, dict):
            calibration = json.loads(Path(calibration).read_text())
        if any(float(c) != 0. for c in calibration.get("distortion_coefficients", ())):
            raise ValueError("Rectify the depth image first: the lookup assumes an undistorted pinhole")
        self.width, self.height = int(calibration["width"]), int(calibration["height"])
        self.depth_scale = float(calibration["depth_scale_m_per_raw_unit"])
        self.fill_holes = bool(fill_holes)
        fx, fy, cx, cy = (float(calibration[name]) for name in ("fx", "fy", "cx", "cy"))
        # Training rays, one per pixel centre, in the training camera frame.
        u = (np.arange(DEPTH_WIDTH) + .5) * (2. / DEPTH_WIDTH) - 1.
        v = (np.arange(DEPTH_HEIGHT) + .5) * (2. / DEPTH_HEIGHT) - 1.
        x, y = np.meshgrid(u * math.tan(math.radians(HIKING_CAMERA_FOV_DEGREES[0]) / 2),
                           v * math.tan(math.radians(HIKING_CAMERA_FOV_DEGREES[1]) / 2), indexing="xy")
        training = np.stack((np.ones_like(x), -x, -y), -1)
        training /= np.linalg.norm(training, axis=-1, keepdims=True)
        real = training
        if mount_rotation_wxyz is not None:
            real = training @ (_rotation_matrix(mount_rotation_wxyz).T
                               @ _rotation_matrix(HIKING_CAMERA_ROTATION_WXYZ)).T
        forward = real[..., 0]
        with np.errstate(divide="ignore", invalid="ignore"):
            column = np.rint(fx * (-real[..., 1] / forward) + cx)
            row = np.rint(fy * (-real[..., 2] / forward) + cy)
            # optical Z of the real camera -> distance along the ray -> training image-plane depth
            self.ray_per_z = (1. / forward).astype(np.float32)
            self.depth_per_z = (training[..., 0] / forward).astype(np.float32)
        self.covered = (forward > 1e-6) & (column >= 0) & (column < self.width) & (row >= 0) & (row < self.height)
        self.source_index = np.where(self.covered, row * self.width + column, -1).astype(np.int64)
        self.crop_coverage = float(self.covered[DEPTH_CROP_ROWS, DEPTH_CROP_COLS].mean())
        if self.crop_coverage < 1.:
            raise ValueError(
                f"The camera sees only {100 * self.crop_coverage:.1f}% of the image region the policy uses; "
                "this stream or mount cannot be converted by a lookup")

    def image_plane_depth(self, raw):
        """[36, 64] training-camera image-plane depth in metres; 0 where not measured."""
        raw = np.asarray(raw)
        if raw.shape != (self.height, self.width):
            raise ValueError(f"Expected a {self.height}x{self.width} raw depth image")
        z = raw.reshape(-1)[np.clip(self.source_index, 0, None)].astype(np.float32) * self.depth_scale
        measured = self.covered & (z > 0)
        return np.where(measured, z * self.depth_per_z, 0.).astype(np.float32)

    def frame(self, raw):
        """One [2, 36, 64] policy frame for ``UnitreeG1PolicyInterface`` from a raw Z16 image."""
        depth = self.image_plane_depth(raw)
        if self.fill_holes:
            # As on Hiking's robot, holes are filled inside the region the policy sees.
            depth[DEPTH_CROP_ROWS, DEPTH_CROP_COLS], _ = fill_depth_holes(depth[DEPTH_CROP_ROWS, DEPTH_CROP_COLS])
            depth[depth < REALSENSE_HOLE_DEPTH_M] = 0.
        return depth_frame(depth)

    def description(self) -> dict:
        """Everything another harness (for example C++) needs to repeat the conversion."""
        return {
            "source_image": [self.height, self.width], "depth_scale_m_per_raw_unit": self.depth_scale,
            "lookup": ("training pixel (row, col) reads raw.flat[source_index[row][col]] * depth_scale * "
                       "depth_per_z[row][col]; raw 0 or source_index -1 means no measurement"),
            "source_index": self.source_index.tolist(), "depth_per_z": self.depth_per_z.tolist(),
            "hole_fill": ({"region": "rows 18-35, cols 16-47", "hole_if_depth_below_m": REALSENSE_HOLE_DEPTH_M,
                           "rule": "mean of measured 8-neighbours, repeated ring by ring; see fill_depth_holes"}
                          if self.fill_holes else None),
            "unmeasured_outside_region": "treated as no hit (1e6 m)",
            "timing": "convert the newest camera image once per 50 Hz control step",
            "uncovered_training_pixels": int((~self.covered).sum()),
        }


def deployment_contract(*, camera: dict | None = None, checkpoint: str | None = None) -> dict:
    """Machine-readable description of everything a harness must reproduce."""
    params = joint_parameters()
    contract = {
        "format": "g1-hiking-policy-deployment-v1",
        "checkpoint": checkpoint,
        "onnx_input": {"name": "proprio_and_depth_ring",
                       "layout": "[768 proprioception] + [37 ring frames x 2 channels x 36 x 64]"},
        "timing": {"control_dt_s": CONTROL_DT, "physics_dt_s": CONTROL_DT / DECIMATION, "decimation": DECIMATION,
                   "pd_update": "every physics substep; the training actuators also lag 0-2 substeps"},
        "joints": {
            "order": list(POLICY_JOINT_NAMES),
            "unitree_sign": UNITREE_JOINT_SIGN.tolist(),
            "convention": ("Policy angles use Hiking's torso-rooted URDF. Unitree MJCF/SDK waist angles, "
                           "velocities and targets are the negatives: policy = unitree_sign * unitree."),
            "default_position_policy": default_joint_positions().tolist(),
            "action_scale": params["action_scale"].tolist(),
            "target_policy": "default_position_policy + action_scale * action (then multiply by unitree_sign)",
            "kp": params["kp"].tolist(), "kd": params["kd"].tolist(),
            "effort_limit": params["effort_limit"].tolist(),
            "training_armature": params["armature"].tolist(),
        },
        "proprioception": {
            "history": PROPRIO_HISTORY, "order": "term-major; within a term oldest -> newest",
            "terms": [
                {"name": "torso_angular_velocity", "width": 3, "scale": ANGULAR_VELOCITY_SCALE,
                 "frame": "torso_link body frame (NOT pelvis imu, NOT unitree secondary_imu)"},
                {"name": "projected_gravity", "width": 3, "frame": "torso_link body frame"},
                {"name": "command", "width": 3, "units": "[vx m/s, vy m/s, yaw rate rad/s]",
                 "training_range": "vx 0 or 0.45-1.0, vy 0, yaw -1..1"},
                {"name": "joint_position_minus_default", "width": 29, "convention": "policy"},
                {"name": "joint_velocity", "width": 29, "scale": JOINT_VELOCITY_SCALE, "convention": "policy"},
                {"name": "last_action", "width": 29, "note": "raw policy output; zero after reset"}],
            "reset": "every history slot is filled with the first post-reset sample",
            "torso_from_pelvis_imu": ("torso = pelvis * Rz(yaw_u) * Rx(roll_u) * Ry(pitch_u) with Unitree waist "
                                      "angles; see unitree_interface.torso_from_pelvis"),
        },
        "depth": {
            "frame": "[depth / 2.5, valid] at 36 x 64, image-plane (z) depth in metres",
            "missing": "no hit, or ray distance <= 0.1 m (self-occluding head geometry): 1e6 m, unclipped",
            "ring": f"{DEPTH_RING} frames, index 36 newest, one appended per control step after physics",
            "policy_frames": "ring indices 0,5,...,35 (newest frame is never visible: one-step delay)",
            "reset": "every ring slot is filled with the first post-reset frame",
            "processing_inside_onnx": "crop rows 18-35, cols 16-47; 3x3 Gaussian (sigma 1, reflect); clip 0-1",
        },
    }
    if camera is not None:
        contract["depth"]["camera"] = camera
        if "position" in camera:
            contract["depth"]["camera_mount_unitree_torso_link"] = {
                "position": (np.asarray(camera["position"], float) + HIKING_TO_UNITREE_TORSO_OFFSET).tolist(),
                "rotation_wxyz": camera.get("rotation_wxyz"),
                "note": ("camera.position is in the training (Hiking) torso_link frame; on unitree_mujoco's G1 "
                         "the same physical mount is 10 mm lower in torso_link coordinates"),
                "convention": camera.get("coordinate_convention"),
            }
    return contract
