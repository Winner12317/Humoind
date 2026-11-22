"""Independent kinematic validation of the G1 -> PM01 retarget.

Replays the G1 source and the PM01 result through MuJoCo forward kinematics and
compares them without using any hhtools retargeting code.  G1 is SuperAMP's own
MJCF.  PM01 is the official collision model of EngineAI's MJCF
(models/pm01_sdk_collision.xml: boxes, cylinders and spheres; the feet are a
20 cm x 11 cm sole box plus a toe box), which is what a simulator collides and
is larger than the foot shell of the URDF.

Per clip it reports

* bookkeeping     frame count / fps match, finite values, unit root quaternions
* pose fidelity   for every matched body segment, the angle between the
                  rotation each robot has undergone since the calibration pose
                  (G1 zero configuration <-> PM01 calibrated pose).  This is the
                  quantity hhtools' rotation objectives minimise, evaluated on
                  all segments, including ones it does not target directly.
                  For limbs the angle between the long axes is listed as well
                  (the same comparison with twist about the limb ignored).
* root            PM01/G1 horizontal travel ratio, root height range
* joint limits    smallest distance to a URDF limit, frames outside limits,
                  J23_HEAD_YAW excursion, peak joint speed vs the URDF limit
* smoothness      peak joint acceleration next to the source's
* feet            stance frames are taken from the *source*: a G1 foot is
                  planted when its sole is within 3 cm of the floor (z = 0) and
                  the foot moves slower than 0.15 m/s.  On exactly those
                  frames the PM01 foot's horizontal speed and sole height are
                  reported next to G1's, with the share of samples more than
                  2 cm above the floor.  The speed is that of the foot frame,
                  so it includes the foot rolling over its toe or heel; slip
                  of the point in contact is in qa/refine/ for refined clips.
                  Also: share of frames with a sole more than 1 cm below the
                  floor, and frames in which the two feet's collision shapes
                  overlap by more than 1 cm.
* ground / self   deepest ground penetration of any collision geom and share
                  of frames with a self-penetration deeper than 1 cm

Run with the hhtools environment:
    python validate.py                                   # final clips in pm01_csv/
    python validate.py --pm01 ../pm01_npz_50hz --tag validation_50hz
    python validate.py --pm01 ../pm01_hhtools --tag validation_hhtools_raw
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import yaml

from common import (
    G1_JOINT_NAMES,
    G1_MJCF,
    HHTOOLS,
    PM01_JOINT_NAMES,
    PM01_SDK_COLLISION,
    WORK,
    quat_xyzw_to_matrix,
    resample_like_csv_to_npz,
)

PM01_URDF = HHTOOLS / "configs" / "robots" / "pm01" / "serial_pm01_edu.urdf"  # visual meshes; export_formats.py
PM01_CALIBRATION = HHTOOLS / "configs" / "robots" / "pm01" / "r2r_calibration_g1_29dof.yaml"

# segment -> (G1 body, PM01 body)
SEGMENTS: dict[str, tuple[str, str]] = {
    "pelvis": ("pelvis", "LINK_BASE"),
    "torso": ("torso_link", "LINK_TORSO_YAW"),
    "thigh_L": ("left_hip_yaw_link", "LINK_HIP_YAW_L"),
    "thigh_R": ("right_hip_yaw_link", "LINK_HIP_YAW_R"),
    "shank_L": ("left_knee_link", "LINK_KNEE_PITCH_L"),
    "shank_R": ("right_knee_link", "LINK_KNEE_PITCH_R"),
    "foot_L": ("left_ankle_roll_link", "LINK_ANKLE_ROLL_L"),
    "foot_R": ("right_ankle_roll_link", "LINK_ANKLE_ROLL_R"),
    "upper_arm_L": ("left_shoulder_yaw_link", "LINK_SHOULDER_YAW_L"),
    "upper_arm_R": ("right_shoulder_yaw_link", "LINK_SHOULDER_YAW_R"),
    "forearm_L": ("left_elbow_link", "LINK_ELBOW_PITCH_L"),
    "forearm_R": ("right_elbow_link", "LINK_ELBOW_PITCH_R"),
    "hand_L": ("left_wrist_yaw_link", "LINK_ELBOW_YAW_L"),
    "hand_R": ("right_wrist_yaw_link", "LINK_ELBOW_YAW_R"),
}
SEGMENT_GROUPS = ("pelvis", "torso", "thigh", "shank", "foot", "upper_arm", "forearm", "hand")
# foot -> (G1 body, PM01 body) carrying the foot's collision shapes
FEET = {"L": ("left_ankle_roll_link", "LINK_FOOT_L"), "R": ("right_ankle_roll_link", "LINK_FOOT_R")}

STANCE_HEIGHT_M = 0.03  # G1 sole within 3 cm of the floor ...
STANCE_SPEED_M_S = 0.15  # ... and foot slower than this: planted
HOVER_M = 0.02  # a planted foot's sole higher than this floats
SOLE_PENETRATION_M = 0.01
SELF_PENETRATION_M = 0.01

# Long axis of each limb segment in the world frame at the calibration pose
# (both robots: upper arm / thigh / shank hang straight down, forearm forward).
LIMB_AXES = {
    "thigh": np.array([0.0, 0.0, -1.0]),
    "shank": np.array([0.0, 0.0, -1.0]),
    "upper_arm": np.array([0.0, 0.0, -1.0]),
    "forearm": np.array([1.0, 0.0, 0.0]),
}


class Robot:
    """MuJoCo FK wrapper that accepts hhtools joint_q rows (xyz + xyzw + dofs)."""

    def __init__(self, xml: Path, joint_names: tuple[str, ...]) -> None:
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        m = self.model
        self.floating = m.njnt > 0 and m.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
        self.qadr = np.array([m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in joint_names])
        jid = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in joint_names]
        if min(jid) < 0:
            raise ValueError(f"{xml}: missing joints")
        self.limits = m.jnt_range[jid].copy()
        self.collision_geoms = [g for g in range(m.ngeom) if m.geom_contype[g] or m.geom_conaffinity[g]]
        self._hulls: dict[int, np.ndarray] = {}
        for g in self.collision_geoms:
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = m.geom_dataid[g]
                verts = m.mesh_vert[m.mesh_vertadr[mesh] : m.mesh_vertadr[mesh] + m.mesh_vertnum[mesh]].astype(np.float64)
                try:
                    from scipy.spatial import ConvexHull

                    verts = verts[ConvexHull(verts).vertices]
                except Exception:  # noqa: BLE001 - degenerate mesh: keep all vertices
                    pass
                self._hulls[g] = verts

    def body_id(self, name: str) -> int:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise KeyError(name)
        return bid

    def set_pose(self, row: np.ndarray, *, root_identity: bool = False) -> tuple[np.ndarray, np.ndarray]:
        """Run FK; return the root rotation / translation to apply on top (identity if floating)."""
        d = self.data
        d.qpos[:] = 0.0
        root_r = np.eye(3) if root_identity else quat_xyzw_to_matrix(row[3:7])
        root_p = np.zeros(3) if root_identity else np.asarray(row[:3], dtype=np.float64)
        if self.floating:
            d.qpos[:3] = root_p
            if root_identity:
                d.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
            else:
                q = row[3:7] / np.linalg.norm(row[3:7])
                d.qpos[3:7] = (q[3], q[0], q[1], q[2])
        d.qpos[self.qadr] = row[7:]
        mujoco.mj_kinematics(self.model, d)
        if self.floating:
            return np.eye(3), np.zeros(3)
        return root_r, root_p

    def body_rotation(self, bid: int, root_r: np.ndarray) -> np.ndarray:
        return root_r @ self.data.xmat[bid].reshape(3, 3)

    def geom_lowest_z(self, g: int, root_r: np.ndarray, root_p: np.ndarray) -> float:
        m, d = self.model, self.data
        rot = root_r @ d.geom_xmat[g].reshape(3, 3)
        pos = root_r @ d.geom_xpos[g] + root_p
        size = m.geom_size[g]
        kind = m.geom_type[g]
        if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            return float(pos[2] - size[0])
        if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
            return float(pos[2] - abs(rot[2, 2]) * size[1] - size[0])
        if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
            az = abs(rot[2, 2])
            return float(pos[2] - az * size[1] - size[0] * np.sqrt(max(0.0, 1.0 - az * az)))
        if kind == mujoco.mjtGeom.mjGEOM_BOX:
            return float(pos[2] - np.abs(rot[2, :]) @ size)
        if kind == mujoco.mjtGeom.mjGEOM_MESH:
            return float(pos[2] + (self._hulls[g] @ rot[2, :]).min())
        return float(pos[2])

    def body_collision_geoms(self, bid: int) -> list[int]:
        return [g for g in self.collision_geoms if self.model.geom_bodyid[g] == bid]

    def self_penetration(self) -> tuple[float, tuple[str, str] | None]:
        """Deepest penetration between two robot geoms (>= 0) and its body pair."""
        m, d = self.model, self.data
        mujoco.mj_collision(m, d)
        deepest, pair = 0.0, None
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = int(m.geom_bodyid[c.geom1]), int(m.geom_bodyid[c.geom2])
            if -float(c.dist) > deepest:
                deepest = -float(c.dist)
                pair = tuple(sorted((m.body(b1).name, m.body(b2).name)))
        return deepest, pair

    def penetration_between(self, geoms_a: list[int], geoms_b: list[int]) -> float:
        """Deepest penetration between two sets of geoms; call after ``self_penetration``."""
        a, b, deepest = set(geoms_a), set(geoms_b), 0.0
        for c in self.data.contact[: self.data.ncon]:
            if (c.geom1 in a and c.geom2 in b) or (c.geom1 in b and c.geom2 in a):
                deepest = max(deepest, -float(c.dist))
        return deepest


def load_pm01_motion(path: Path, source_fps: float) -> tuple[np.ndarray, float, tuple[str, ...], dict]:
    """hhtools robot CSV, the headerless export (rate then comes from the source clip), or a compact NPZ."""
    if path.suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            joint_q = np.concatenate([data["root_pos"], data["root_rot"], data["dof_pos"]], axis=1).astype(np.float64)
            names = tuple(str(n) for n in data["dof_names"]) if "dof_names" in data.files else PM01_JOINT_NAMES
            return joint_q, float(data["fps"]), names, {"robot": "pm01"}
    with path.open(encoding="utf-8") as fp:
        first = fp.readline()
    if first.startswith("#") or first.startswith("time"):
        return load_hhtools_csv(path)
    data = np.loadtxt(path, delimiter=",", dtype=np.float64, ndmin=2)
    if data.shape[1] != 7 + len(PM01_JOINT_NAMES):
        raise ValueError(f"{path}: expected {7 + len(PM01_JOINT_NAMES)} columns, got {data.shape[1]}")
    return data, source_fps, PM01_JOINT_NAMES, {"robot": "pm01"}


def load_hhtools_csv(path: Path) -> tuple[np.ndarray, float, tuple[str, ...], dict]:
    meta: dict[str, str] = {}
    header: list[str] | None = None
    with path.open(encoding="utf-8") as fp:
        skip = 0
        for line in fp:
            if line.startswith("#"):
                key, _, value = line[1:].partition(":")
                meta[key.strip()] = value.strip()
                skip += 1
                continue
            header = line.strip().split(",")
            skip += 1
            break
    if header is None or header[0] != "time":
        raise ValueError(f"{path}: not an hhtools robot CSV")
    data = np.loadtxt(path, delimiter=",", skiprows=skip, dtype=np.float64, ndmin=2)
    dof_names = tuple(c[len("dof_") :] for c in header[8:])
    return data[:, 1:], float(meta["sample_rate"]), dof_names, meta


def rotation_angle_deg(r: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(r) - 1.0) / 2.0, -1.0, 1.0))))


def calibration_rotations(g1: Robot, pm: Robot) -> tuple[dict, dict]:
    cal = yaml.safe_load(PM01_CALIBRATION.read_text(encoding="utf-8"))["calibrated_joint_q"]
    g1_row = np.zeros(7 + len(G1_JOINT_NAMES))
    g1_row[6] = 1.0
    pm_row = np.zeros(7 + len(PM01_JOINT_NAMES))
    pm_row[6] = 1.0
    pm_row[7:] = [float(cal[n]) for n in PM01_JOINT_NAMES]
    rg, _ = g1.set_pose(g1_row, root_identity=True)
    c_g1 = {s: g1.body_rotation(g1.body_id(b[0]), rg).copy() for s, b in SEGMENTS.items()}
    rp, _ = pm.set_pose(pm_row, root_identity=True)
    c_pm = {s: pm.body_rotation(pm.body_id(b[1]), rp).copy() for s, b in SEGMENTS.items()}
    return c_g1, c_pm


def _pct(values: np.ndarray, q: float) -> float:
    return float(np.percentile(values, q)) if values.size else float("nan")


# URDF velocity limits of PM01's actuators, rad/s, in PM01_JOINT_NAMES order.
PM01_SPEED_LIMITS = np.array(
    [26.3 if any(t in name for t in ("HIP_PITCH", "HIP_ROLL", "KNEE")) else 35.2 for name in PM01_JOINT_NAMES]
)


def frame_metrics(g1: Robot, pm: Robot, c_g1: dict, c_pm: dict, src: np.ndarray, out: np.ndarray) -> dict:
    """Per-frame comparison of a G1 clip and its PM01 retarget (same frame index = same instant).

    Returns arrays over the first ``min(len(src), len(out))`` frames:
    ``seg_err`` / ``dir_err`` (degrees, per segment), ``sole`` (sole height, m) and
    ``foot_xy`` per robot and foot, ``body_low`` (lowest collision point, m),
    ``self_pen`` (deepest self-penetration, m) and ``feet_pen`` (deepest overlap of
    the two feet, m) per robot, and ``self_pair`` (the body pair of the deepest
    self-penetration, or None).
    """
    n = min(src.shape[0], out.shape[0])
    seg_err = {s: np.zeros(n) for s in SEGMENTS}
    g1_seg_ids = {s: g1.body_id(b[0]) for s, b in SEGMENTS.items()}
    pm_seg_ids = {s: pm.body_id(b[1]) for s, b in SEGMENTS.items()}
    g1_foot_geoms = {k: g1.body_collision_geoms(g1.body_id(v[0])) for k, v in FEET.items()}
    pm_foot_geoms = {k: pm.body_collision_geoms(pm.body_id(v[1])) for k, v in FEET.items()}
    g1_foot_ids = {k: g1.body_id(v[0]) for k, v in FEET.items()}
    pm_foot_ids = {k: pm.body_id(v[1]) for k, v in FEET.items()}

    dir_err = {s: np.zeros(n) for s in SEGMENTS if s.rsplit("_", 1)[0] in LIMB_AXES}
    sole = {r: {k: np.zeros(n) for k in FEET} for r in ("g1", "pm")}
    foot_xy = {r: {k: np.zeros((n, 2)) for k in FEET} for r in ("g1", "pm")}
    body_low = {"g1": np.zeros(n), "pm": np.zeros(n)}
    self_pen = {"g1": np.zeros(n), "pm": np.zeros(n)}
    feet_pen = {"g1": np.zeros(n), "pm": np.zeros(n)}
    self_pair: dict[str, list[tuple[str, str] | None]] = {"g1": [None] * n, "pm": [None] * n}

    for f in range(n):
        rg, pg = g1.set_pose(src[f])
        g1_rot = {s: g1.body_rotation(g1_seg_ids[s], rg) @ c_g1[s].T for s in SEGMENTS}
        for k in FEET:
            sole["g1"][k][f] = min(g1.geom_lowest_z(g, rg, pg) for g in g1_foot_geoms[k])
            foot_xy["g1"][k][f] = (rg @ g1.data.xpos[g1_foot_ids[k]] + pg)[:2]
        body_low["g1"][f] = min(g1.geom_lowest_z(g, rg, pg) for g in g1.collision_geoms)
        self_pen["g1"][f], self_pair["g1"][f] = g1.self_penetration()
        feet_pen["g1"][f] = g1.penetration_between(g1_foot_geoms["L"], g1_foot_geoms["R"])

        rp, pp = pm.set_pose(out[f])
        for s in SEGMENTS:
            pm_rot = pm.body_rotation(pm_seg_ids[s], rp) @ c_pm[s].T
            seg_err[s][f] = rotation_angle_deg(pm_rot @ g1_rot[s].T)
            if s in dir_err:
                axis = LIMB_AXES[s.rsplit("_", 1)[0]]
                cos = float(np.clip((pm_rot @ axis) @ (g1_rot[s] @ axis), -1.0, 1.0))
                dir_err[s][f] = np.degrees(np.arccos(cos))
        for k in FEET:
            sole["pm"][k][f] = min(pm.geom_lowest_z(g, rp, pp) for g in pm_foot_geoms[k])
            foot_xy["pm"][k][f] = (rp @ pm.data.xpos[pm_foot_ids[k]] + pp)[:2]
        body_low["pm"][f] = min(pm.geom_lowest_z(g, rp, pp) for g in pm.collision_geoms)
        self_pen["pm"][f], self_pair["pm"][f] = pm.self_penetration()
        feet_pen["pm"][f] = pm.penetration_between(pm_foot_geoms["L"], pm_foot_geoms["R"])

    return {
        "frames": n,
        "seg_err": seg_err,
        "dir_err": dir_err,
        "sole": sole,
        "foot_xy": foot_xy,
        "body_low": body_low,
        "self_pen": self_pen,
        "feet_pen": feet_pen,
        "self_pair": self_pair,
    }


def stance_frames(metrics: dict, dt: float) -> tuple[float, dict, dict]:
    """(G1's low sole level, horizontal foot speed per robot and foot, G1-planted mask per foot).

    Planted is judged against the floor at z = 0.  The low sole level (5th percentile of the
    lower sole) is returned for information: it tells how far the source sits below its floor.
    """
    sole, foot_xy = metrics["sole"], metrics["foot_xy"]
    g1_floor = float(np.percentile(np.minimum(sole["g1"]["L"], sole["g1"]["R"]), 5))
    speed = {r: {k: np.linalg.norm(np.gradient(foot_xy[r][k], dt, axis=0), axis=1) for k in FEET} for r in ("g1", "pm")}
    planted = {k: (sole["g1"][k] < STANCE_HEIGHT_M) & (speed["g1"][k] < STANCE_SPEED_M_S) for k in FEET}
    return g1_floor, speed, planted


def validate_clip(g1: Robot, pm: Robot, c_g1: dict, c_pm: dict, src_path: Path, out_path: Path) -> dict:
    src, src_fps, src_names, _ = load_hhtools_csv(src_path)
    out, out_fps, out_names, out_meta = load_pm01_motion(out_path, src_fps)
    if out_path.suffix == ".npz" and abs(out_fps - src_fps) > 1e-9:  # bring the source to the NPZ rate the way the NPZ was made
        src, src_fps = resample_like_csv_to_npz(src, src_fps, out_fps), out_fps
    result: dict = {"clip": src_path.stem, "group": src_path.parent.name}
    result["frames"] = int(out.shape[0])
    result["fps"] = out_fps
    result["bookkeeping_ok"] = bool(
        src_names == G1_JOINT_NAMES
        and out_names == PM01_JOINT_NAMES
        and out.shape[0] == src.shape[0]
        and abs(out_fps - src_fps) < 1e-9
        and np.isfinite(out).all()
        and out_meta.get("robot") == "pm01"
        and np.abs(np.linalg.norm(out[:, 3:7], axis=1) - 1.0).max() < 1e-4
    )
    dt = 1.0 / out_fps

    metrics = frame_metrics(g1, pm, c_g1, c_pm, src, out)
    n = metrics["frames"]
    seg_err, dir_err, sole = metrics["seg_err"], metrics["dir_err"], metrics["sole"]
    body_low, self_pen = metrics["body_low"], metrics["self_pen"]
    self_pairs: dict[str, dict[tuple[str, str], int]] = {"g1": {}, "pm": {}}
    for r in ("g1", "pm"):
        for depth, pair in zip(self_pen[r], metrics["self_pair"][r]):
            if pair is not None and depth > SELF_PENETRATION_M:
                self_pairs[r][pair] = self_pairs[r].get(pair, 0) + 1

    # --- pose fidelity ---------------------------------------------------
    pose: dict = {}
    for group in SEGMENT_GROUPS:
        errs = np.concatenate([v for s, v in seg_err.items() if s == group or s.startswith(group + "_")])
        pose[group] = {"mean_deg": float(errs.mean()), "p95_deg": _pct(errs, 95), "max_deg": float(errs.max())}
    result["segment_rotation_error"] = pose
    direction: dict = {}
    for group in LIMB_AXES:
        errs = np.concatenate([v for s, v in dir_err.items() if s.startswith(group + "_")])
        direction[group] = {"mean_deg": float(errs.mean()), "p95_deg": _pct(errs, 95), "max_deg": float(errs.max())}
    result["limb_direction_error"] = direction

    # --- root --------------------------------------------------------------
    d_src = np.linalg.norm(np.diff(src[:n, :2], axis=0), axis=1).sum()
    d_out = np.linalg.norm(np.diff(out[:n, :2], axis=0), axis=1).sum()
    result["root"] = {
        "g1_path_m": float(d_src),
        "pm01_path_m": float(d_out),
        "path_ratio": float(d_out / d_src) if d_src > 1e-6 else float("nan"),
        "g1_height_m": [float(src[:n, 2].min()), float(src[:n, 2].mean()), float(src[:n, 2].max())],
        "pm01_height_m": [float(out[:n, 2].min()), float(out[:n, 2].mean()), float(out[:n, 2].max())],
    }

    # --- joints ------------------------------------------------------------
    q = out[:n, 7:]
    lower, upper = pm.limits[:, 0], pm.limits[:, 1]
    margin = np.minimum(q - lower, upper - q)
    worst_joint = int(np.argmin(margin.min(axis=0)))
    speed = np.abs(np.diff(q, axis=0)) / dt
    accel = np.abs(np.diff(q, n=2, axis=0)) / dt**2
    src_accel = np.abs(np.diff(src[:n, 7:], n=2, axis=0)) / dt**2
    result["joints"] = {
        "min_limit_margin_rad": float(margin.min()),
        "min_limit_margin_joint": PM01_JOINT_NAMES[worst_joint],
        "frames_outside_limits": int((margin.min(axis=1) < -1e-6).sum()),
        "frames_within_0.01rad_of_limit": int((margin.min(axis=1) < 0.01).sum()),
        "head_yaw_max_abs_rad": float(np.abs(q[:, PM01_JOINT_NAMES.index("J23_HEAD_YAW")]).max()),
        "max_speed_rad_s": float(speed.max()) if speed.size else 0.0,
        "max_speed_over_limit_ratio": float((speed / PM01_SPEED_LIMITS).max()) if speed.size else 0.0,
        "max_accel_rad_s2": float(accel.max()) if accel.size else 0.0,
        "g1_max_accel_rad_s2": float(src_accel.max()) if src_accel.size else 0.0,
        "p99_accel_rad_s2": _pct(accel, 99),
        "g1_p99_accel_rad_s2": _pct(src_accel, 99),
    }

    # --- feet --------------------------------------------------------------
    g1_floor, foot_speed, planted = stance_frames(metrics, dt)
    stance_speed = {r: [] for r in ("g1", "pm")}
    stance_height = {r: [] for r in ("g1", "pm")}
    stance_count = 0
    for k in FEET:
        stance_count += int(planted[k].sum())
        for r in ("g1", "pm"):
            stance_speed[r].append(foot_speed[r][k][planted[k]])
            stance_height[r].append(sole[r][k][planted[k]])
    feet: dict = {"g1_floor_m": g1_floor, "stance_ratio": stance_count / (2 * n)}
    for r in ("g1", "pm"):
        v = np.concatenate(stance_speed[r])
        h = np.concatenate(stance_height[r])
        lower = np.minimum(sole[r]["L"], sole[r]["R"])
        feet[r] = {
            "sole_min_m": float(lower.min()),
            "lower_sole_p05_m": _pct(lower, 5),
            "lower_sole_p50_m": _pct(lower, 50),
            "stance_slide_mean_m_s": float(v.mean()) if v.size else float("nan"),
            "stance_slide_p95_m_s": _pct(v, 95),
            "stance_sole_mean_m": float(h.mean()) if h.size else float("nan"),
            "stance_sole_p05_m": _pct(h, 5),
            "stance_sole_p95_m": _pct(h, 95),
            "stance_hover_share": float((h > HOVER_M).mean()) if h.size else float("nan"),
            "frames_sole_below_1cm_share": float((lower < -SOLE_PENETRATION_M).mean()),
            "frames_feet_overlap_gt_1cm": int((metrics["feet_pen"][r] > SELF_PENETRATION_M).sum()),
            "feet_overlap_max_m": float(metrics["feet_pen"][r].max()),
        }
    result["feet"] = feet

    # --- ground / self collisions -----------------------------------------
    result["ground"] = {
        "g1_deepest_penetration_m": float(max(0.0, -body_low["g1"].min())),
        "pm01_deepest_penetration_m": float(max(0.0, -body_low["pm"].min())),
        "pm01_frames_penetrating_gt_1cm": float((body_low["pm"] < -0.01).mean()),
        "g1_frames_penetrating_gt_1cm": float((body_low["g1"] < -0.01).mean()),
    }
    def top_pairs(counts: dict[tuple[str, str], int]) -> list[list]:
        ranked = sorted(counts.items(), key=lambda kv: -kv[1])[:3]
        return [[a, b, round(c / n, 4)] for (a, b), c in ranked]

    result["self_collision"] = {
        "g1_frames_gt_1cm": float((self_pen["g1"] > SELF_PENETRATION_M).mean()),
        "pm01_frames_gt_1cm": float((self_pen["pm"] > SELF_PENETRATION_M).mean()),
        "g1_max_m": float(self_pen["g1"].max()),
        "pm01_max_m": float(self_pen["pm"].max()),
        "g1_top_pairs": top_pairs(self_pairs["g1"]),
        "pm01_top_pairs": top_pairs(self_pairs["pm"]),
    }
    return result


def print_tables(results: list[dict]) -> None:
    print("\nSegment rotation error vs G1, degrees (mean / p95 / max)")
    print(f"{'clip':38s} " + " ".join(f"{g:>17s}" for g in SEGMENT_GROUPS))
    for r in results:
        cells = [
            f"{r['segment_rotation_error'][g]['mean_deg']:5.1f}/{r['segment_rotation_error'][g]['p95_deg']:5.1f}/{r['segment_rotation_error'][g]['max_deg']:5.1f}"
            for g in SEGMENT_GROUPS
        ]
        print(f"{r['clip']:38s} " + " ".join(f"{c:>17s}" for c in cells))

    print("\nRoot, joints")
    print(
        f"{'clip':38s} {'ok':>3s} {'frames':>6s} {'travel':>7s} {'z min/mean/max':>18s} {'lim margin':>10s} "
        f"{'joint':>20s} {'out':>4s} {'head':>6s} {'v/vmax':>6s} {'acc p99':>8s} {'G1 p99':>7s} {'acc max':>8s} {'G1 max':>7s}"
    )
    for r in results:
        j, root = r["joints"], r["root"]
        z = root["pm01_height_m"]
        print(
            f"{r['clip']:38s} {'y' if r['bookkeeping_ok'] else 'N':>3s} {r['frames']:6d} {root['path_ratio']:7.3f} "
            f"{z[0]:5.3f}/{z[1]:5.3f}/{z[2]:5.3f} {j['min_limit_margin_rad']:10.3f} {j['min_limit_margin_joint']:>20s} "
            f"{j['frames_outside_limits']:4d} {j['head_yaw_max_abs_rad']:6.3f} {j['max_speed_over_limit_ratio']:6.2f} "
            f"{j['p99_accel_rad_s2']:8.0f} {j['g1_p99_accel_rad_s2']:7.0f} {j['max_accel_rad_s2']:8.0f} {j['g1_max_accel_rad_s2']:7.0f}"
        )

    print("\nLimb long-axis direction error vs G1, degrees (mean / p95 / max)")
    print(f"{'clip':38s} " + " ".join(f"{g:>17s}" for g in LIMB_AXES))
    for r in results:
        cells = [
            f"{r['limb_direction_error'][g]['mean_deg']:5.1f}/{r['limb_direction_error'][g]['p95_deg']:5.1f}/{r['limb_direction_error'][g]['max_deg']:5.1f}"
            for g in LIMB_AXES
        ]
        print(f"{r['clip']:38s} " + " ".join(f"{c:>17s}" for c in cells))

    print("\nFeet on G1 stance frames: speed = mean horizontal speed of the foot frame (cm/s), sole = sole height (cm),")
    print("hover = stance samples > 2 cm up, below = frames with a sole > 1 cm under the floor, overlap = frames with the feet > 1 cm into each other")
    print(
        f"{'clip':38s} {'stance':>6s} {'speed PM':>8s} {'speed G1':>8s} {'p95 PM':>7s} {'p95 G1':>7s} "
        f"{'sole PM mean':>12s} {'p05':>6s} {'p95':>6s} {'sole G1 mean':>12s} {'G1 low':>8s} {'PM min':>7s} {'G1 min':>7s} "
        f"{'hover PM':>8s} {'below PM':>8s} {'G1':>6s} {'overlap PM':>10s} {'G1':>4s} {'max cm':>6s}"
    )
    for r in results:
        f = r["feet"]
        print(
            f"{r['clip']:38s} {f['stance_ratio'] * 100:5.1f}% {f['pm']['stance_slide_mean_m_s'] * 100:8.2f} "
            f"{f['g1']['stance_slide_mean_m_s'] * 100:8.2f} {f['pm']['stance_slide_p95_m_s'] * 100:7.2f} "
            f"{f['g1']['stance_slide_p95_m_s'] * 100:7.2f} {f['pm']['stance_sole_mean_m'] * 100:12.2f} "
            f"{f['pm']['stance_sole_p05_m'] * 100:6.2f} {f['pm']['stance_sole_p95_m'] * 100:6.2f} "
            f"{f['g1']['stance_sole_mean_m'] * 100:12.2f} {f['g1_floor_m'] * 100:8.2f} "
            f"{f['pm']['sole_min_m'] * 100:7.2f} {f['g1']['sole_min_m'] * 100:7.2f} "
            f"{f['pm']['stance_hover_share'] * 100:7.1f}% {f['pm']['frames_sole_below_1cm_share'] * 100:7.1f}% "
            f"{f['g1']['frames_sole_below_1cm_share'] * 100:5.1f}% {f['pm']['frames_feet_overlap_gt_1cm']:10d} "
            f"{f['g1']['frames_feet_overlap_gt_1cm']:4d} {f['pm']['feet_overlap_max_m'] * 100:6.2f}"
        )

    print("\nGround penetration (deepest, cm; share of frames > 1 cm) and self-penetration > 1 cm")
    print(
        f"{'clip':38s} {'pen PM':>7s} {'pen G1':>7s} {'fr PM':>6s} {'fr G1':>6s} {'self PM':>8s} {'self G1':>8s} "
        f"{'max PM/G1 cm':>13s}  most frequent PM01 pair"
    )
    for r in results:
        g, c = r["ground"], r["self_collision"]
        pair = c["pm01_top_pairs"][0] if c["pm01_top_pairs"] else None
        print(
            f"{r['clip']:38s} {g['pm01_deepest_penetration_m'] * 100:7.2f} {g['g1_deepest_penetration_m'] * 100:7.2f} "
            f"{g['pm01_frames_penetrating_gt_1cm'] * 100:5.1f}% {g['g1_frames_penetrating_gt_1cm'] * 100:5.1f}% "
            f"{c['pm01_frames_gt_1cm'] * 100:7.1f}% {c['g1_frames_gt_1cm'] * 100:7.1f}% "
            f"{c['pm01_max_m'] * 100:6.1f}/{c['g1_max_m'] * 100:5.1f}  "
            + (f"{pair[0]} x {pair[1]} ({pair[2] * 100:.1f}%)" if pair else "-")
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, default=WORK / "g1_source")
    parser.add_argument(
        "--pm01",
        type=Path,
        default=WORK / "pm01_csv",
        help="PM01 clips to check: pm01_csv (final, default), an NPZ set such as pm01_npz_50hz, or pm01_hhtools (raw hhtools export).",
    )
    parser.add_argument("--tag", default="validation", help="Basename of the JSON written to qa/.")
    parser.add_argument("--clip", action="append", default=None)
    args = parser.parse_args()

    g1 = Robot(G1_MJCF, G1_JOINT_NAMES)
    pm = Robot(PM01_SDK_COLLISION, PM01_JOINT_NAMES)
    c_g1, c_pm = calibration_rotations(g1, pm)

    results = []
    for out_path in sorted(p for p in args.pm01.glob("*/*") if p.suffix in (".csv", ".npz")):
        if args.clip and out_path.stem not in args.clip:
            continue
        src_path = args.src / out_path.parent.name / f"{out_path.stem}.csv"
        if not src_path.is_file():
            print(f"skip {out_path}: no matching source", file=sys.stderr)
            continue
        results.append(validate_clip(g1, pm, c_g1, c_pm, src_path, out_path))
    if not results:
        raise SystemExit(f"no retargeted clips under {args.pm01}")

    print_tables(results)
    qa = WORK / "qa"
    qa.mkdir(parents=True, exist_ok=True)
    (qa / f"{args.tag}.json").write_text(json.dumps(results, indent=1) + "\n", encoding="utf-8")
    print(f"\nwrote {qa / (args.tag + '.json')}")
    return 0 if all(r["bookkeeping_ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
