"""Shared paths and conventions for the SuperAMP G1 -> EngineAI PM01 retarget."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SUPERAMP = PROJECT_ROOT / "SuperAMP"
HHTOOLS = PROJECT_ROOT / "Human-Humanoid-Tools"
WORK = PROJECT_ROOT / "pm01_retarget"

G1_CSV_DIR = SUPERAMP / "motion_data_csv" / "amp"
G1_NPZ_DIR = SUPERAMP / "src" / "assets" / "motions" / "g1" / "amp"
G1_MJCF = SUPERAMP / "src" / "assets" / "robots" / "unitree_g1" / "xmls" / "g1.xml"

# PM01's official collision model: the collision primitives of EngineAI's MJCF
# (engineai_robotics_native_sdk, assets/resource/robot/pm01_edu/xml), floating base,
# no meshes.  Written by make_collision_model.py.
PM01_SDK_COLLISION = WORK / "models" / "pm01_sdk_collision.xml"

SOURCE_ROBOT = "g1_29dof"
TARGET_ROBOT = "pm01"

# Frame rate of SuperAMP/motion_data_csv/amp/*.csv (headerless Unitree layout).
G1_CSV_FPS = 120.0
# Frame rate of every SuperAMP reference NPZ.
G1_NPZ_FPS = 50.0

# Column order of the SuperAMP G1 CSVs after the 7 root columns
# (x y z qx qy qz qw).  Copied from SuperAMP/scripts/csv_to_npz.py.
G1_JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# EngineAI SDK joint order (joint_sdk_names in
# engineai_rl_lab/tasks/tracking/robots/pm01.py).
PM01_JOINT_NAMES = (
    "J00_HIP_PITCH_L",
    "J01_HIP_ROLL_L",
    "J02_HIP_YAW_L",
    "J03_KNEE_PITCH_L",
    "J04_ANKLE_PITCH_L",
    "J05_ANKLE_ROLL_L",
    "J06_HIP_PITCH_R",
    "J07_HIP_ROLL_R",
    "J08_HIP_YAW_R",
    "J09_KNEE_PITCH_R",
    "J10_ANKLE_PITCH_R",
    "J11_ANKLE_ROLL_R",
    "J12_WAIST_YAW",
    "J13_SHOULDER_PITCH_L",
    "J14_SHOULDER_ROLL_L",
    "J15_SHOULDER_YAW_L",
    "J16_ELBOW_PITCH_L",
    "J17_ELBOW_YAW_L",
    "J18_SHOULDER_PITCH_R",
    "J19_SHOULDER_ROLL_R",
    "J20_SHOULDER_YAW_R",
    "J21_ELBOW_PITCH_R",
    "J22_ELBOW_YAW_R",
    "J23_HEAD_YAW",
)


def g1_mjcf_body_names() -> list[str]:
    """Body names of SuperAMP's G1 MJCF in document order (mjlab body order)."""
    text = G1_MJCF.read_text(encoding="utf-8")
    worldbody = text[text.index("<worldbody>") : text.index("</worldbody>")]
    return re.findall(r'<body name="([^"]+)"', worldbody)


def load_g1_csv(path: Path) -> np.ndarray:
    """SuperAMP headerless G1 CSV -> (F, 36) [x y z qx qy qz qw | 29 joints]."""
    data = np.loadtxt(path, delimiter=",", dtype=np.float64)
    if data.ndim != 2 or data.shape[1] != 7 + len(G1_JOINT_NAMES):
        raise ValueError(f"{path}: expected {7 + len(G1_JOINT_NAMES)} columns, got {data.shape}")
    return data


def load_g1_npz_as_joint_q(path: Path) -> tuple[np.ndarray, float]:
    """SuperAMP reference NPZ -> ((F, 36) joint_q with xyzw root quat, fps).

    body 0 of the mjlab export is the pelvis (floating base) and its
    quaternion is stored wxyz; joint_pos follows the MJCF joint order, which
    is the same order as ``G1_JOINT_NAMES``.
    """
    with np.load(path, allow_pickle=False) as data:
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        root_pos = np.asarray(data["body_pos_w"][:, 0], dtype=np.float64)
        root_wxyz = np.asarray(data["body_quat_w"][:, 0], dtype=np.float64)
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
    root_xyzw = root_wxyz[:, [1, 2, 3, 0]]
    return np.concatenate([root_pos, root_xyzw, joint_pos], axis=1), fps


def quat_slerp_xyzw(q0: np.ndarray, q1: np.ndarray, blend: np.ndarray) -> np.ndarray:
    """Batched shortest-path slerp for xyzw quaternions."""
    q0 = q0 / np.linalg.norm(q0, axis=-1, keepdims=True)
    q1 = q1 / np.linalg.norm(q1, axis=-1, keepdims=True)
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    blend = blend.reshape(-1, 1)
    small = sin_theta < 1e-6
    safe = np.where(small, 1.0, sin_theta)
    w0 = np.where(small, 1.0 - blend, np.sin((1.0 - blend) * theta) / safe)
    w1 = np.where(small, blend, np.sin(blend * theta) / safe)
    out = w0 * q0 + w1 * q1
    return out / np.linalg.norm(out, axis=-1, keepdims=True)


def resample_like_csv_to_npz(joint_q: np.ndarray, input_fps: float, output_fps: float) -> np.ndarray:
    """Resample (F, 7+N) exactly as SuperAMP/scripts/csv_to_npz.py does.

    Output frame k is sampled at t = k / output_fps for t in
    ``arange(0, (F - 1) / input_fps, 1 / output_fps)``: linear interpolation
    for root position and joints, slerp for the root quaternion.
    """
    frames = joint_q.shape[0]
    duration = (frames - 1) / input_fps
    times = np.arange(0.0, duration, 1.0 / output_fps, dtype=np.float32).astype(np.float64)
    phase = times / duration
    index_0 = np.floor(phase * (frames - 1)).astype(np.int64)
    index_1 = np.minimum(index_0 + 1, frames - 1)
    blend = phase * (frames - 1) - index_0
    out = np.empty((times.shape[0], joint_q.shape[1]), dtype=np.float64)
    lerp = lambda a, b: a * (1.0 - blend[:, None]) + b * blend[:, None]  # noqa: E731
    out[:, :3] = lerp(joint_q[index_0, :3], joint_q[index_1, :3])
    out[:, 3:7] = quat_slerp_xyzw(joint_q[index_0, 3:7], joint_q[index_1, 3:7], blend)
    out[:, 7:] = lerp(joint_q[index_0, 7:], joint_q[index_1, 7:])
    return out


def quat_xyzw_to_matrix(q: np.ndarray) -> np.ndarray:
    """(..., 4) xyzw -> (..., 3, 3)."""
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    m[..., 0, 0] = 1 - 2 * (y * y + z * z)
    m[..., 0, 1] = 2 * (x * y - z * w)
    m[..., 0, 2] = 2 * (x * z + y * w)
    m[..., 1, 0] = 2 * (x * y + z * w)
    m[..., 1, 1] = 1 - 2 * (x * x + z * z)
    m[..., 1, 2] = 2 * (y * z - x * w)
    m[..., 2, 0] = 2 * (x * z - y * w)
    m[..., 2, 1] = 2 * (y * z + x * w)
    m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def write_hhtools_csv(path: Path, joint_q: np.ndarray, fps: float, dof_names: tuple[str, ...], meta: dict[str, str]) -> None:
    """Write (F, 7+N) rows in hhtools' robot CSV layout (``# key: value`` lines, then named columns)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    header = ["time", "root_x", "root_y", "root_z", "root_qx", "root_qy", "root_qz", "root_qw"] + [f"dof_{n}" for n in dof_names]
    table = np.column_stack([np.arange(joint_q.shape[0]) / fps, joint_q])
    with path.open("w", encoding="utf-8") as fp:
        for key in sorted({**meta, "sample_rate": f"{fps:.6f}"}):
            fp.write(f"# {key}: {({**meta, 'sample_rate': f'{fps:.6f}'})[key]}\n")
        fp.write(",".join(header) + "\n")
        np.savetxt(fp, table, delimiter=",", fmt="%.6f")
