"""Contact-aware refinement of hhtools' PM01 retargets.

hhtools places each PM01 foot relative to the pelvis and applies one vertical
shift per clip.  On PM01's official collision model (the MJCF of
engineai_robotics_native_sdk: a 20 cm x 11 cm sole box plus a toe box, larger
than G1's foot) that leaves planted feet sliding, toes and heels below the
floor, feet that overlap when G1 brings its own narrower feet together, and,
in the fall-and-get-up clip, a robot that floats while standing.  This script
repairs those in a second pass over every clip, the way contact-aware
retargeters do (foot sticking and non-penetration constraints as in
OmniRetarget, footskate clean-up followed by leg IK as in Kovar et al. 2002,
per-frame IK against a posture reference with collision avoidance as in GMR /
mink):

1. Soft joint range.  The elbow pitch is mapped smoothly into the range RL
   frameworks treat as usable (centre +/- 0.9 x half range, engineai_rl_lab's
   soft limit) instead of resting on the hard stop.
2. Root height.  hhtools raises the whole robot when a source toe or heel is
   below the floor, which lifts the stance foot too.  The pelvis height is
   rebuilt from the source instead: G1's pelvis height times the leg ratio plus
   one constant per clip, chosen so that planted feet reach the floor with the
   legs as hhtools solved them.  While G1 rests on something other than its
   feet (the fall-and-get-up clip), the height is the one that puts PM01's
   lowest point where G1's lowest point is.
3. Foot height.  Near the ground each PM01 sole is given the height G1's sole
   has at that instant (never below the floor), measured at the lowest point of
   the sole.  Contact timing and swing clearance therefore equal the source's.
   Where the tilted sole would reach lower than that (PM01's sole is longer
   than G1's, and G1's own toes and heels are below the floor in places), the
   foot's pitch and roll are reduced first, which leaves the leg as it is; the
   foot is raised only for what remains.  ``--keep-foot-tilt`` raises the foot
   instead, which keeps G1's foot angle and bends the knee.
4. No slip, no overlap.  A horizontal correction per foot and frame is solved
   for the whole clip at once (a QP): while G1's foot is planted, the point of
   PM01's sole that touches the floor must not move, so the foot rolls and
   pivots about its own contact point; corrections blend smoothly through the
   swing; the two feet's collision shapes must stay ``--foot-gap`` apart.  The
   separation constraint is re-linearised a few times.
5. Legs.  Inverse kinematics on the twelve leg joints moves each foot to its
   corrected pose.  Where a foot on the floor would be out of reach the pelvis
   drops by what is missing (smoothed over time); a swing foot that is out of
   reach stays where the leg can put it.  Knees are not straightened past
   0.1 rad.
6. Arms.  Where a forearm or hand penetrates the torso, head or the other arm,
   the arm joints are moved the least amount that clears it.

The pelvis path and orientation, the waist and the head are not changed.

Input : pm01_hhtools/<group>/<clip>.csv, g1_source/<group>/<clip>.csv
Output: pm01_refined/<group>/<clip>.csv   (hhtools robot CSV, same rate)
        qa/refine/<clip>.json             (what changed, before / after)

export_formats.py uses a refined clip in place of the hhtools one when it exists.

Run with the hhtools environment:
    python refine_motions.py                       # every clip in pm01_hhtools/ (about 75 s)
    python refine_motions.py --clip NAME [--clip NAME ...]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
import scipy.sparse as sp
from scipy.ndimage import binary_closing, gaussian_filter1d, maximum_filter1d, minimum_filter1d
from scipy.sparse.linalg import spsolve

from common import G1_JOINT_NAMES, G1_MJCF, PM01_JOINT_NAMES, PM01_SDK_COLLISION, WORK, write_hhtools_csv
from validate import FEET, PM01_SPEED_LIMITS, Robot, load_hhtools_csv

PLANTED_HEIGHT_M = 0.03  # G1 sole within 3 cm of the floor ...
PLANTED_SPEED_M_S = 0.15  # ... and slower than this: planted (validate.py's definition)
BODY_CONTACT_M = (0.02, 0.04)  # G1's lowest non-foot point below / above which it can / cannot be resting on the floor
BODY_ABOVE_SOLE_M = (0.005, 0.02)  # ... and its height above the lower sole below / above which it is / is not
SOLE_DOWN_DEG = (35.0, 55.0)  # G1 foot tilt below / above which the foot can / cannot be standing on its sole
NEAR_GROUND_M = (0.04, 0.12)  # G1 sole height below / above which foot height is fully / not corrected
PELVIS_DROP_HOLD_S = 0.1  # a pelvis drop needed for reach is held this long around the frames that need it
KNEE_FLOOR_RAD = 0.1  # the leg IK does not straighten a knee past this (or past hhtools' value, if that is straighter)
SOFT_RANGE_FACTOR = 0.9  # engineai_rl_lab: soft_joint_pos_limit_factor
SOFT_RANGE_KNEE_RAD = 0.15  # the mapping starts this far inside the soft range
SOFT_RANGE_JOINTS = ("J16_ELBOW_PITCH_L", "J21_ELBOW_PITCH_R")
LEG_JOINTS = {
    "L": ("J00_HIP_PITCH_L", "J01_HIP_ROLL_L", "J02_HIP_YAW_L", "J03_KNEE_PITCH_L", "J04_ANKLE_PITCH_L", "J05_ANKLE_ROLL_L"),
    "R": ("J06_HIP_PITCH_R", "J07_HIP_ROLL_R", "J08_HIP_YAW_R", "J09_KNEE_PITCH_R", "J10_ANKLE_PITCH_R", "J11_ANKLE_ROLL_R"),
}
ARM_JOINTS = PM01_JOINT_NAMES[13:23]
FOOT_BODY = {"L": "LINK_FOOT_L", "R": "LINK_FOOT_R"}
# Arm parts that can be moved clear, and what they must not penetrate.
ARM_BODIES = {
    "L": ("LINK_ELBOW_PITCH_L", "LINK_ELBOW_YAW_L", "LINK_ELBOW_END_L"),
    "R": ("LINK_ELBOW_PITCH_R", "LINK_ELBOW_YAW_R", "LINK_ELBOW_END_R"),
}
ARM_OBSTACLES = ("LINK_TORSO_YAW", "LINK_HEAD_YAW", "LINK_BASE")


# ------------------------------------------------------------- kinematics ----

class Pm01(Robot):
    """PM01 on the official collision model, with what the refinement needs on top of FK."""

    def __init__(self) -> None:
        super().__init__(PM01_SDK_COLLISION, PM01_JOINT_NAMES)
        m = self.model
        self.dof = np.array([m.jnt_dofadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in PM01_JOINT_NAMES])
        self.foot_body = {k: self.body_id(v) for k, v in FOOT_BODY.items()}
        self.foot_geoms = {k: self.body_collision_geoms(b) for k, b in self.foot_body.items()}
        # Corners of the sole boxes in the foot frame: their lowest one is the sole's lowest point.
        self.sole_corners, self.sole_boxes = {}, {}
        for side, geoms in self.foot_geoms.items():
            corners, boxes = [], []
            for g in geoms:
                if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_BOX:
                    raise ValueError("foot collision geoms are expected to be boxes")
                rot = np.zeros(9)
                mujoco.mju_quat2Mat(rot, m.geom_quat[g])
                signs = np.array([[x, y, z] for x in (-1, 1) for y in (-1, 1) for z in (-1, 1)], dtype=float)
                corners.append(m.geom_pos[g] + (signs * m.geom_size[g]) @ rot.reshape(3, 3).T)
                boxes.append((m.geom_pos[g].copy(), rot.reshape(3, 3).copy(), m.geom_size[g].copy()))
            self.sole_corners[side] = np.concatenate(corners)
            self.sole_boxes[side] = boxes  # (centre, axes as columns, half sizes) in the foot frame
        feet = {g for geoms in self.foot_geoms.values() for g in geoms}
        self.body_geoms = [g for g in self.collision_geoms if g not in feet]
        self.root_z_dof = 2  # free joint: x y z, then rotation
        self._jacp = np.zeros((3, m.nv))
        self._jacr = np.zeros((3, m.nv))

    def fk(self, row: np.ndarray) -> None:
        self.set_pose(row)
        mujoco.mj_comPos(self.model, self.data)  # needed by the Jacobians

    def foot_pose(self, side: str) -> tuple[np.ndarray, np.ndarray]:
        b = self.foot_body[side]
        return self.data.xpos[b].copy(), self.data.xmat[b].reshape(3, 3).copy()

    def foot_jacobian(self, side: str, columns: np.ndarray) -> np.ndarray:
        mujoco.mj_jacBody(self.model, self.data, self._jacp, self._jacr, self.foot_body[side])
        return np.vstack([self._jacp[:, columns], self._jacr[:, columns]])

    def lowest_body_point(self) -> float:
        """Lowest point of everything but the feet."""
        eye, zero = np.eye(3), np.zeros(3)
        return min(self.geom_lowest_z(g, eye, zero) for g in self.body_geoms)


def rotation_vector(r: np.ndarray) -> np.ndarray:
    """Axis * angle of a rotation matrix (angle < pi)."""
    w = np.array([r[2, 1] - r[1, 2], r[0, 2] - r[2, 0], r[1, 0] - r[0, 1]])
    s = np.linalg.norm(w) / 2.0
    c = (np.trace(r) - 1.0) / 2.0
    angle = np.arctan2(s, c)
    return w * (0.5 if angle < 1e-9 else angle / (2.0 * s))


def smooth(values: np.ndarray, seconds: float, fps: float) -> np.ndarray:
    sigma = seconds * fps
    return gaussian_filter1d(values, sigma, axis=0, mode="nearest") if sigma > 0.2 else values


def runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, stop) of every run of True."""
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return list(zip(edges[::2], edges[1::2]))


# ----------------------------------------------------------------- stages ----

def soft_range(q: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Identity inside the soft range minus a knee; beyond, approach the soft bound smoothly (C1)."""
    mid, half = (lo + hi) / 2.0, (hi - lo) / 2.0
    soft_lo, soft_hi = mid - SOFT_RANGE_FACTOR * half, mid + SOFT_RANGE_FACTOR * half
    k = min(SOFT_RANGE_KNEE_RAD, SOFT_RANGE_FACTOR * half / 2.0)
    out = q.copy()
    below, above = q < soft_lo + k, q > soft_hi - k
    out[below] = soft_lo + k * np.exp((q[below] - (soft_lo + k)) / k)
    out[above] = soft_hi - k * np.exp(((soft_hi - k) - q[above]) / k)
    return out


def source_contacts(g1: Robot, src: np.ndarray, fps: float) -> dict:
    """Per frame, from the G1 source: sole height and planted flag per foot, lowest non-foot point."""
    n = src.shape[0]
    bodies = {k: g1.body_id(v[0]) for k, v in FEET.items()}
    geoms = {k: g1.body_collision_geoms(b) for k, b in bodies.items()}
    sole = {k: np.zeros(n) for k in FEET}
    xy = {k: np.zeros((n, 2)) for k in FEET}
    upright = {k: np.zeros(n) for k in FEET}  # cosine of the foot's tilt from sole-down
    feet = {k: {"pos": np.zeros((n, 3)), "rot": np.zeros((n, 3, 3))} for k in FEET}
    body = np.zeros(n)
    other = [g for g in g1.collision_geoms if g not in geoms["L"] + geoms["R"]]
    for f in range(n):
        rot, pos = g1.set_pose(src[f])
        for k in FEET:
            sole[k][f] = min(g1.geom_lowest_z(g, rot, pos) for g in geoms[k])
            xy[k][f] = (rot @ g1.data.xpos[bodies[k]] + pos)[:2]
            feet[k]["rot"][f] = rot @ g1.data.xmat[bodies[k]].reshape(3, 3)
            feet[k]["pos"][f] = rot @ g1.data.xpos[bodies[k]] + pos
            upright[k][f] = feet[k]["rot"][f][2, 2]
        body[f] = min(g1.geom_lowest_z(g, rot, pos) for g in other)
    step = lambda x, lo, hi: (lambda t: t * t * (3.0 - 2.0 * t))(np.clip((x - lo) / (hi - lo), 0.0, 1.0))  # noqa: E731
    planted, sole_down = {}, {}
    for k in FEET:
        tilt = np.degrees(np.arccos(np.clip(upright[k], -1.0, 1.0)))
        sole_down[k] = 1.0 - step(tilt, *SOLE_DOWN_DEG)  # 0: the foot is on its toes, side or back (kneeling, lying)
        speed = np.linalg.norm(np.gradient(xy[k], 1.0 / fps, axis=0), axis=1)
        mask = (sole[k] < PLANTED_HEIGHT_M) & (speed < PLANTED_SPEED_M_S) & (sole_down[k] > 0.5)
        mask = binary_closing(mask, structure=np.ones(max(3, int(round(0.03 * fps)) | 1)))  # bridge flicker
        for a, b in runs(mask):
            if b - a < max(2, int(round(0.04 * fps))):
                mask[a:b] = False
        planted[k] = mask
    # G1 rests on its body when some non-foot part is near the floor and not clearly above the
    # lower sole (G1's shin capsule ends 3-4 cm above the floor while it simply stands).
    above_sole = step(body - np.maximum(np.minimum(sole["L"], sole["R"]), 0.0), *BODY_ABOVE_SOLE_M)
    above_floor = step(body, *BODY_CONTACT_M)
    on_feet = smooth(1.0 - (1.0 - above_sole) * (1.0 - above_floor), 0.1, fps)  # 1: only the feet carry the robot
    for k in FEET:
        planted[k] &= on_feet > 0.5
    # sole points of G1's feet: the two ends of each collision capsule, at the bottom of its round section
    m, corners = g1.model, {}
    for k in FEET:
        points = []
        for g in geoms[k]:
            axes = np.zeros(9)
            mujoco.mju_quat2Mat(axes, m.geom_quat[g])
            for end in (-1.0, 1.0):
                points.append(m.geom_pos[g] + end * m.geom_size[g][1] * axes.reshape(3, 3)[:, 2] - np.array([0.0, 0.0, m.geom_size[g][0]]))
        corners[k] = np.asarray(points)
    return {
        "sole": sole,
        "planted": planted,
        "sole_down": sole_down,
        "body": body,
        "on_feet": on_feet,
        "root_z": src[:, 2].copy(),
        "stance_slip_m_s": stance_slip(feet, corners, planted, fps),
    }


def foot_tracks(pm: Pm01, q: np.ndarray) -> dict:
    """Foot pose, sole lowest point and lowest non-foot point of a PM01 trajectory."""
    n = q.shape[0]
    out = {k: {"pos": np.zeros((n, 3)), "rot": np.zeros((n, 3, 3)), "low": np.zeros(n)} for k in FEET}
    body = np.zeros(n)
    for f in range(n):
        pm.set_pose(q[f])
        for k in FEET:
            pos, rot = pm.foot_pose(k)
            out[k]["pos"][f], out[k]["rot"][f] = pos, rot
            out[k]["low"][f] = pos[2] + (pm.sole_corners[k] @ rot[2]).min()
        body[f] = pm.lowest_body_point()
    out["body"] = body
    return out


def root_height_shift(base_z: np.ndarray, tracks: dict, contacts: dict, leg_scale: float, fps: float) -> tuple[np.ndarray, dict]:
    """Vertical shift of the pelvis per frame (see the module docstring, step 2)."""
    on_feet = contacts["on_feet"]

    # Height at which a planted foot reaches G1's sole height with the leg unchanged, as an
    # offset from the scaled G1 pelvis height.  One robust constant per clip.
    samples = []
    for k in FEET:
        use = contacts["planted"][k] & (on_feet > 0.999)
        needed = base_z[use] - (tracks[k]["low"][use] - np.maximum(contacts["sole"][k][use], 0.0))
        samples.append(needed - leg_scale * contacts["root_z"][use])
    samples = np.concatenate(samples)
    if samples.size < 10:
        raise ValueError("too few planted frames to place the pelvis")
    offset = float(np.median(samples))
    shift_feet = leg_scale * contacts["root_z"] + offset - base_z

    # Shift that puts PM01's lowest point (feet included) where G1's is, not below the floor.  The
    # upper envelope over a short window keeps it smooth without dipping below what a frame needs.
    lowest_g1 = np.minimum(contacts["body"], np.minimum(contacts["sole"]["L"], contacts["sole"]["R"]))
    lowest_pm = np.minimum(tracks["body"], np.minimum(tracks["L"]["low"], tracks["R"]["low"]))
    shift_body = np.maximum(lowest_g1, 0.0) - lowest_pm
    window = max(3, int(round(0.2 * fps)) | 1)
    shift_body = smooth(maximum_filter1d(shift_body, window, mode="nearest"), 0.06, fps)

    shift = smooth(on_feet * shift_feet + (1.0 - on_feet) * shift_body, 0.03, fps)
    info = {
        "pelvis_offset_from_scaled_g1_m": offset,
        "frames_resting_on_body": int((on_feet < 0.5).sum()),
        "min_m": float(shift.min()),
        "max_m": float(shift.max()),
    }
    return shift, info


def near_ground_weight(contacts: dict, side: str) -> np.ndarray:
    """1 where G1's sole is near the floor, sole down, and G1 stands on its feet; fading to 0 away from that."""
    t = np.clip((contacts["sole"][side] - NEAR_GROUND_M[0]) / (NEAR_GROUND_M[1] - NEAR_GROUND_M[0]), 0.0, 1.0)
    return contacts["on_feet"] * contacts["sole_down"][side] * (1.0 - t * t * (3.0 - 2.0 * t))  # else: leave the foot to hhtools


def flatten_feet(pm: Pm01, q: np.ndarray, tracks: dict, contacts: dict, shift: np.ndarray, fps: float) -> dict:
    """Where a tilted sole would reach below G1's sole height, reduce the tilt instead of raising the foot.

    The foot keeps its heading and its position; pitch and roll are scaled by one factor per
    frame, just enough for the lowest point of the sole to come up to the height G1's sole has
    (not below the floor), and no more than the ankle joints have room for.  This keeps the leg
    as hhtools solved it.  ``tracks`` is updated.
    """
    from scipy.spatial.transform import Rotation

    info = {}
    for k in FEET:
        rot, height = tracks[k]["rot"], tracks[k]["pos"][:, 2] + shift
        yaw = np.arctan2(rot[:, 1, 0], rot[:, 0, 0])
        heading = Rotation.from_rotvec(np.stack([np.zeros_like(yaw), np.zeros_like(yaw), yaw], axis=1))
        tilt = (heading.inv() * Rotation.from_matrix(rot)).as_rotvec()
        corners = pm.sole_corners[k]

        def lowest(scale: np.ndarray) -> np.ndarray:
            matrices = Rotation.from_rotvec(tilt * scale[:, None]).as_matrix()
            return height + (matrices[:, 2, :] @ corners.T).min(axis=1)

        weight = near_ground_weight(contacts, k)
        target = np.maximum(contacts["sole"][k], 0.0)
        ones = np.ones_like(height)
        too_low = (lowest(ones) < target - 1e-4) & (weight > 0.0)
        lo, hi = np.zeros_like(height), ones.copy()  # bisect the tilt factor: lowest(lo) >= target > lowest(hi)
        reachable = lowest(lo) >= target
        for _ in range(24):
            mid = 0.5 * (lo + hi)
            above = lowest(mid) >= target
            lo, hi = np.where(above, mid, lo), np.where(above, hi, mid)
        factor = np.where(too_low, np.where(reachable, lo, 0.0), 1.0)
        # Untilting turns the ankle by about the removed roll (foot x) and pitch (foot y): stay inside its range.
        for axis, name in ((0, LEG_JOINTS[k][5]), (1, LEG_JOINTS[k][4])):
            j = PM01_JOINT_NAMES.index(name)
            angle = q[:, 7 + j]
            room = np.where(tilt[:, axis] > 0.0, angle - pm.limits[j, 0], pm.limits[j, 1] - angle) - 0.01
            factor = np.maximum(factor, 1.0 - np.clip(room, 0.0, None) / np.maximum(np.abs(tilt[:, axis]), 1e-9))
        factor = np.clip(smooth(1.0 - weight * (1.0 - factor), 0.015, fps), 0.0, 1.0)
        new = (heading * Rotation.from_rotvec(tilt * factor[:, None])).as_matrix()
        changed = factor < 0.999
        info[k] = {
            "frames_flattened": int(changed.sum()),
            "largest_tilt_reduction_deg": float(np.degrees((np.linalg.norm(tilt, axis=1) * (1.0 - factor)).max())),
        }
        tracks[k]["rot"] = np.where(changed[:, None, None], new, rot)
        tracks[k]["low"] = tracks[k]["pos"][:, 2] + np.einsum("nj,cj->nc", tracks[k]["rot"][:, 2, :], corners).min(axis=1)
    return info


def foot_height_correction(tracks: dict, contacts: dict, shift: np.ndarray, fps: float) -> dict[str, np.ndarray]:
    out = {}
    for k in FEET:
        low = tracks[k]["low"] + shift
        source = contacts["sole"][k]
        dz = near_ground_weight(contacts, k) * (np.maximum(source, 0.0) - low)
        dz = smooth(dz, 0.008, fps)
        out[k] = np.maximum(dz, -low)  # never below the floor, also far from it
    return out


def box_separations(box_a: tuple, box_b: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Separating-axis test of two boxes (centre, axes as columns, half sizes).

    Returns the candidate axes (k, 3), the separation along each (negative: overlap) and the
    sign of b's offset along each.  The largest separation is the boxes' clearance: a lower
    bound on their distance when positive, the penetration depth when negative.
    """
    (ca, ra, ha), (cb, rb, hb) = box_a, box_b
    axes = [ra[:, i] for i in range(3)] + [rb[:, i] for i in range(3)]
    for i in range(3):
        for j in range(3):
            cross = np.cross(ra[:, i], rb[:, j])
            length = np.linalg.norm(cross)
            if length > 1e-6:
                axes.append(cross / length)
    axes = np.asarray(axes)
    offset = axes @ (cb - ca)
    reach = np.abs(axes @ ra) @ ha + np.abs(axes @ rb) @ hb
    return axes, np.abs(offset) - reach, np.where(offset >= 0.0, 1.0, -1.0)


def feet_separation(pm: Pm01, pose: dict[str, tuple[np.ndarray, np.ndarray]], near: float) -> tuple[float, list]:
    """Clearance between the feet's collision boxes at the given poses (position, rotation), and
    for every pair of boxes closer than ``near``: its best mostly-horizontal separating axis as
    (gradient of that separation with respect to moving the left foot horizontally, separation),
    and its best mostly-vertical separation.
    """
    world = {
        side: [(pos + rot @ centre, rot @ axes, half) for centre, axes, half in pm.sole_boxes[side]]
        for side, (pos, rot) in pose.items()
    }
    clearance, horizontal = np.inf, []
    for box_l in world["L"]:
        for box_r in world["R"]:
            axes, separation, sign = box_separations(box_r, box_l)
            clearance = min(clearance, float(separation.max()))
            if separation.max() < near:
                level = np.abs(axes[:, 2]) < 0.5
                k = np.flatnonzero(level)[np.argmax(separation[level])]
                horizontal.append((sign[k] * axes[k, :2], float(separation[k]), float(separation[~level].max())))
    return clearance, horizontal


def contact_point_slip(pos: np.ndarray, rot: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Horizontal motion, frame to frame, of the sole point that is on the floor (n-1, 2).

    ``corners`` are sole points in the foot frame.  The point in contact is their centroid
    weighted towards the lowest ones; between two frames it is treated as fixed to the foot.
    """
    heights = np.einsum("nij,cj->nci", rot, corners)[..., 2]  # (n, corners) height of each corner above the foot origin
    weights = np.exp(-(heights - heights.min(axis=1, keepdims=True)) / 0.004)
    weights /= weights.sum(axis=1, keepdims=True)
    contact = weights @ corners  # (n, 3) contact point in the foot frame
    local = 0.5 * (contact[1:] + contact[:-1])
    here = pos[:-1, :2] + np.einsum("nij,nj->ni", rot[:-1], local)[:, :2]
    there = pos[1:, :2] + np.einsum("nij,nj->ni", rot[1:], local)[:, :2]
    return there - here


def stance_slip(tracks: dict, corners: dict, planted: dict, fps: float) -> float:
    """Mean speed of the sole's contact point over the frames G1 has that foot planted, m/s."""
    speeds = []
    for k in FEET:
        pair = planted[k][:-1] & planted[k][1:]
        speeds.append(np.linalg.norm(contact_point_slip(tracks[k]["pos"], tracks[k]["rot"], corners[k]), axis=1)[pair] * fps)
    speeds = np.concatenate(speeds)
    return float(speeds.mean()) if speeds.size else float("nan")


def solve_foot_corrections(
    pm: Pm01, tracks: dict, dz: dict, shift: np.ndarray, contacts: dict, fps: float, foot_gap: float
) -> tuple[dict[str, np.ndarray], dict]:
    """Horizontal correction e[side] (n, 2): no slip while planted, smooth elsewhere, feet apart."""
    n = shift.shape[0]
    sides = list(FEET)
    planted, on_feet = contacts["planted"], contacts["on_feet"] > 0.5
    col = lambda s, t, a: (sides.index(s) * n + t) * 2 + a  # noqa: E731
    rows, cols, vals, rhs = [], [], [], []

    def add(entries: list[tuple[int, float]], target: float, weight: float) -> None:
        r = len(rhs)
        for c, v in entries:
            rows.append(r); cols.append(c); vals.append(v * weight)  # noqa: E702
        rhs.append(target * weight)

    w_anchor = 1.0
    w_smooth = np.sqrt(4e-5) * fps**2  # second difference; with w_anchor: transitions of about 0.08 s
    w_slip = 10.0 * fps  # first difference; 100x the anchor weight in velocity units
    slip_frames = {}
    for s in sides:
        slip = contact_point_slip(tracks[s]["pos"], tracks[s]["rot"], pm.sole_corners[s])
        grounded = planted[s] & (tracks[s]["low"] + shift + dz[s] < 0.005)
        pair = grounded[:-1] & grounded[1:]
        slip_frames[s] = pair
        for t in range(n):
            for a in range(2):
                add([(col(s, t, a), 1.0)], 0.0, w_anchor)
                if 0 < t < n - 1 and not (pair[t - 1] and pair[t]):
                    add([(col(s, t - 1, a), 1.0), (col(s, t, a), -2.0), (col(s, t + 1, a), 1.0)], 0.0, w_smooth)
                if t < n - 1 and pair[t]:
                    add([(col(s, t + 1, a), 1.0), (col(s, t, a), -1.0)], -slip[t, a], w_slip)
    jac = sp.csr_matrix((vals, (rows, cols)), shape=(len(rhs), 4 * n))
    target = np.asarray(rhs)
    hessian = (jac.T @ jac).tocsc()
    gradient = jac.T @ target
    e = spsolve(hessian, gradient)

    # Feet apart.  Where the two feet's boxes come closer than foot_gap while the robot stands on
    # its feet, they are separated horizontally along the axis that needs the smallest push (a
    # box's side or end, or an edge pair).  The separation along a fixed axis is linear in the
    # corrections and never more than the true clearance, so each inequality is safe; the axes
    # are taken again at the new foot positions until no frame is too close.
    def poses(t: int, e_now: np.ndarray) -> dict:
        return {
            s: (tracks[s]["pos"][t] + np.array([e_now[col(s, t, 0)], e_now[col(s, t, 1)], shift[t] + dz[s][t]]), tracks[s]["rot"][t])
            for s in sides
        }

    def clearances(e_now: np.ndarray) -> np.ndarray:
        out = np.full(n, np.inf)
        for t in standing:
            out[t] = feet_separation(pm, poses(int(t), e_now), 0.0)[0]
        return out

    standing = np.flatnonzero(on_feet)
    before = clearances(np.zeros_like(e))
    info = {"foot_clearance_min_before_m": float(before.min()), "frames_feet_closer_than_gap_before": int((before < foot_gap - 5e-4).sum())}

    # A swinging foot that passes just over the other foot is cleared by lifting it, when that
    # takes far less than moving it aside.
    lift = {s: np.zeros(n) for s in sides}
    for t in np.flatnonzero(clearances(e) < foot_gap - 5e-4):
        pose = poses(int(t), e)
        pairs = feet_separation(pm, pose, foot_gap)[1]
        aside = max(foot_gap - separation for _, separation, _ in pairs)
        up = max(foot_gap - vertical for _, _, vertical in pairs)
        upper = max(sides, key=lambda s: pose[s][0][2])
        if not planted[upper][t] and up < min(0.03, 0.5 * aside):
            lift[upper][t] = up + 1e-3
    for s in sides:
        if lift[s].any():
            window = max(3, int(round(0.06 * fps)) | 1)
            dz[s] += np.maximum(lift[s], smooth(maximum_filter1d(lift[s], window, mode="nearest"), 0.015, fps))
    info["frames_swing_foot_lifted_over_other"] = int(sum((lift[s] > 0).sum() for s in sides))
    constrained: set[int] = set()
    clearance, passes = clearances(e), 0
    while passes < 12:
        close = set(np.flatnonzero(clearance < foot_gap - 5e-4).tolist()) - constrained
        if not close and (passes == 0 or clearance[sorted(constrained)].min() > foot_gap - 5e-4):
            break
        import osqp  # only needed when feet come too close

        passes += 1
        constrained |= close
        c_rows, c_cols, c_vals, lower = [], [], [], []
        for t in sorted(constrained):
            rel = np.array([e[col("L", t, a)] - e[col("R", t, a)] for a in range(2)])
            for grad, separation, _ in feet_separation(pm, poses(t, e), foot_gap + 0.03)[1]:
                # separation + grad . (rel_new - rel) >= foot_gap
                r = len(lower)
                for a in range(2):
                    c_rows += [r, r]; c_cols += [col("L", t, a), col("R", t, a)]; c_vals += [grad[a], -grad[a]]  # noqa: E702
                lower.append(foot_gap + 2e-4 - separation + grad @ rel)
        a_mat = sp.csc_matrix((c_vals, (c_rows, c_cols)), shape=(len(lower), 4 * n))
        solver = osqp.OSQP()
        solver.setup(P=sp.triu(hessian, format="csc"), q=-gradient, A=a_mat, l=np.asarray(lower), u=np.full(len(lower), np.inf),
                     verbose=False, eps_abs=1e-7, eps_rel=1e-7, max_iter=50000, polish=True)
        result = solver.solve()
        if result.x is None or not np.isfinite(result.x).all() or result.info.status not in ("solved", "solved inaccurate"):
            raise RuntimeError(f"foot separation QP failed: {result.info.status}")
        e = result.x
        clearance = clearances(e)
    info["foot_clearance_min_after_m"] = float(clearance.min())
    info["frames_feet_closer_than_gap_after"] = int((clearance < foot_gap - 5e-4).sum())
    info["separation_passes"] = passes
    info["frames_with_separation_constraint"] = len(constrained)
    out = {s: np.stack([e[[col(s, t, 0) for t in range(n)]], e[[col(s, t, 1) for t in range(n)]]], axis=1) for s in sides}
    info["no_slip_frames"] = {s: int(slip_frames[s].sum()) for s in sides}
    return out, info


def solve_legs(pm: Pm01, q: np.ndarray, targets: dict, joint_index: dict[str, np.ndarray], knee_floor: np.ndarray, give: bool) -> dict:
    """Move both feet to their target poses with the twelve leg joints.  ``q`` is modified in place.

    A foot below its target height always counts fully.  A foot above it counts fully where it
    is to touch the floor and little while it swings (``targets[side]["height_weight"]``), so a
    swing foot the leg cannot quite reach stays where the leg can put it.  With ``give`` the pelvis height is a thirteenth variable on a spring: it
    drops where a foot on the floor would otherwise be out of reach.  Knees do not straighten
    past ``knee_floor`` (per frame and side).
    """
    rot_weight = 0.3  # metres per radian: 1 degree of foot tilt weighs like 5 mm
    spring = 0.3 if give else 1e3  # pelvis height against foot position
    sides = list(FEET)
    joints = np.concatenate([joint_index[k] for k in sides])
    knees = [int(np.flatnonzero(joints == joint_index[k][3])[0]) + 1 for k in sides]
    columns = np.concatenate([[pm.root_z_dof], pm.dof[joints]])
    worst_pos = worst_rot = 0.0
    for f in range(q.shape[0]):
        row = q[f]
        z_ref = row[2]
        lower = np.concatenate([[z_ref - 0.08], pm.limits[joints, 0]])
        upper = np.concatenate([[z_ref], pm.limits[joints, 1]])  # the pelvis only ever drops
        for i, knee in enumerate(knees):
            lower[knee] = knee_floor[f, i]
        x = np.clip(np.concatenate([[row[2]], row[7 + joints]]), lower, upper)
        row[7 + joints] = x[1:]
        base_scale = np.concatenate([[1.0, 1.0, 1.0, rot_weight, rot_weight, rot_weight] for k in sides])
        above_weight = [targets[k]["height_weight"][f] for k in sides]
        pull = np.zeros(13)
        pull[0] = spring**2

        def residual(x_now: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
            row[2], row[7 + joints] = x_now[0], x_now[1:]
            pm.fk(row)
            parts = []
            for side in sides:
                pos, rot = pm.foot_pose(side)
                parts.append(np.concatenate([targets[side]["pos"][f] - pos, rotation_vector(targets[side]["rot"][f] @ rot.T)]))
            err_now = np.concatenate(parts)
            scale_now = base_scale.copy()
            for i in range(len(sides)):
                if err_now[6 * i + 2] < 0.0:  # foot above its target
                    scale_now[6 * i + 2] = above_weight[i]
            err_now = err_now * scale_now
            return err_now, scale_now, float(err_now @ err_now + pull[0] * (x_now[0] - z_ref) ** 2)

        # Levenberg-Marquardt with box limits: a step is kept only if it lowers the cost
        err, scale, cost = residual(x)
        damping = 1e-4
        for _ in range(60):
            jac = np.vstack([pm.foot_jacobian(side, columns) for side in sides]) * scale[:, None]
            g = jac.T @ err - pull * (x - z_ref)
            free = ~(((x <= lower) & (g < 0)) | ((x >= upper) & (g > 0)))  # joints on a limit and pushed into it stay
            if np.abs(g[free]).max(initial=0.0) < 1e-7:
                break
            h = jac.T @ jac + np.diag(pull)
            improved = False
            while damping < 1e3:
                step = np.zeros(13)
                step[free] = np.linalg.solve(h[np.ix_(free, free)] + damping * np.eye(int(free.sum())), g[free])
                trial = np.clip(x + step, lower, upper)
                trial_err, trial_scale, trial_cost = residual(trial)
                if trial_cost < cost:
                    improved = True
                    break
                damping *= 10.0
            if not improved:
                residual(x)  # leave the model at the accepted point
                break
            gain = cost - trial_cost
            x, err, scale, cost = trial, trial_err, trial_scale, trial_cost
            damping = max(damping / 5.0, 1e-6)
            if gain < 1e-12:
                break
        row[2], row[7 + joints] = x[0], x[1:]
        pm.fk(row)
        for side in sides:
            pos, rot = pm.foot_pose(side)
            if targets[side]["height_weight"][f] > 0.9:  # report the feet that are meant to be exactly placed
                worst_pos = max(worst_pos, float(np.linalg.norm(targets[side]["pos"][f] - pos)))
            worst_rot = max(worst_rot, float(np.linalg.norm(rotation_vector(targets[side]["rot"][f] @ rot.T))))
    return {"worst_grounded_foot_position_error_m": worst_pos, "worst_foot_rotation_error_deg": float(np.degrees(worst_rot))}


def clear_arms(pm: Pm01, q: np.ndarray, fps: float, clearance: float) -> dict:
    """Move the arm joints the least amount that keeps forearms and hands out of torso, head and each other."""
    m, d = pm.model, pm.data
    arm_idx = np.array([PM01_JOINT_NAMES.index(n) for n in ARM_JOINTS])
    arm_geoms = {s: [g for b in bodies for g in pm.body_collision_geoms(pm.body_id(b))] for s, bodies in ARM_BODIES.items()}
    obstacles = [g for b in ARM_OBSTACLES for g in pm.body_collision_geoms(pm.body_id(b))]
    pairs = [(a, o) for s in arm_geoms for a in arm_geoms[s] for o in obstacles]
    pairs += [(a, b) for a in arm_geoms["L"] for b in arm_geoms["R"]]
    fromto = np.zeros(6)

    def gaps(row: np.ndarray, reach: float) -> np.ndarray:
        pm.set_pose(row)
        return np.array([mujoco.mj_geomDistance(m, d, a, b, reach, fromto) for a, b in pairs])

    def separate(row: np.ndarray, reference: np.ndarray, solve_to: float) -> None:
        """Gauss-Newton on: stay at ``reference``, with a stiff penalty on every gap below ``solve_to``."""
        reach = solve_to + 0.02
        gap = gaps(row, reach)
        for _ in range(30):
            active = np.flatnonzero(gap < solve_to + 0.003)
            if not active.size or gap.min() > solve_to - 2e-4:
                break
            grad = np.zeros((active.size, arm_idx.size))
            for j, joint in enumerate(arm_idx):  # numeric gradient of the gaps: robust to how the shapes touch
                probe = row.copy()
                probe[7 + joint] += 1e-3
                grad[:, j] = (gaps(probe, reach)[active] - gap[active]) / 1e-3
            weight = 100.0
            h = np.eye(arm_idx.size) + weight**2 * grad.T @ grad
            g = -(row[7 + arm_idx] - reference) + weight**2 * grad.T @ np.maximum(solve_to - gap[active], 0.0)
            step = np.linalg.solve(h, g)
            norm = np.linalg.norm(step)
            if norm > 0.15:
                step *= 0.15 / norm
            row[7 + arm_idx] = np.clip(row[7 + arm_idx] + step, pm.limits[arm_idx, 0], pm.limits[arm_idx, 1])
            gap = gaps(row, reach)

    n = q.shape[0]
    original = q[:, 7 + arm_idx].copy()
    before = np.array([gaps(q[f], clearance + 0.03).min() for f in range(n)])
    # 1. every frame on its own, a little further than needed; 2. smooth the changes over time;
    # 3. touch up the frames the smoothing left short.
    for f in np.flatnonzero(before < clearance + 0.004):
        separate(q[f], original[f], clearance + 0.004)
    delta = q[:, 7 + arm_idx] - original
    moved = np.abs(delta).max(axis=1) > 1e-6
    if moved.any():
        q[:, 7 + arm_idx] = np.clip(original + smooth(delta, 0.03, fps), pm.limits[arm_idx, 0], pm.limits[arm_idx, 1])
        for f in range(n):
            if gaps(q[f], clearance + 0.03).min() < clearance:
                separate(q[f], q[f, 7 + arm_idx].copy(), clearance)
    after = np.array([gaps(q[f], clearance + 0.03).min() for f in range(n)]) if moved.any() else before
    delta = q[:, 7 + arm_idx] - original
    return {
        "frames_adjusted": int((np.abs(delta).max(axis=1) > 1e-6).sum()),
        "frames_penetrating_before": int((before < 0.0).sum()),
        "frames_penetrating_after": int((after < 0.0).sum()),
        "deepest_before_m": float(max(0.0, -before.min())),
        "deepest_after_m": float(max(0.0, -after.min())),
        "largest_joint_change_deg": float(np.degrees(np.abs(delta).max())),
    }


# ------------------------------------------------------------------ driver ----

def refine_clip(pm: Pm01, g1: Robot, src_path: Path, base_path: Path, out_path: Path, args: argparse.Namespace) -> dict:
    src, fps, src_names, _ = load_hhtools_csv(src_path)
    base, base_fps, names, meta = load_hhtools_csv(base_path)
    if src_names != G1_JOINT_NAMES or names != PM01_JOINT_NAMES or base.shape[0] != src.shape[0] or abs(fps - base_fps) > 1e-9:
        raise ValueError(f"{base_path}: does not match its G1 source")
    group = base_path.parent.name
    leg_scale = float(meta["root_trajectory_scale"])  # the calibrated leg ratio run_retarget.py scaled the pelvis path with
    q = base.copy()
    q[:, 3:7] /= np.linalg.norm(q[:, 3:7], axis=1, keepdims=True)
    report: dict = {"clip": base_path.stem, "group": group, "frames": int(q.shape[0]), "fps": fps}

    # 1. soft joint range
    soft = {}
    for name in SOFT_RANGE_JOINTS:
        j = PM01_JOINT_NAMES.index(name)
        lo, hi = pm.limits[j]
        before = q[:, 7 + j].copy()
        q[:, 7 + j] = soft_range(before, lo, hi)
        margin = lambda v: np.minimum(v - lo, hi - v)  # noqa: E731
        soft[name] = {
            "frames_within_0.01rad_of_limit_before": int((margin(before) < 0.01).sum()),
            "smallest_margin_before_rad": float(margin(before).min()),
            "smallest_margin_after_rad": float(margin(q[:, 7 + j]).min()),
            "largest_change_deg": float(np.degrees(np.abs(q[:, 7 + j] - before).max())),
        }
    report["soft_range"] = soft

    # 2-4. root height, foot height, horizontal corrections
    contacts = source_contacts(g1, src, fps)
    tracks = foot_tracks(pm, q)
    slip_before = stance_slip(tracks, pm.sole_corners, contacts["planted"], fps)
    shift, shift_info = root_height_shift(q[:, 2], tracks, contacts, leg_scale, fps)
    flattened = None if args.keep_foot_tilt else flatten_feet(pm, q, tracks, contacts, shift, fps)
    dz = foot_height_correction(tracks, contacts, shift, fps)
    e, info = solve_foot_corrections(pm, tracks, dz, shift, contacts, fps, args.foot_gap)
    report["root_height"] = shift_info
    report["feet"] = info
    report["feet"]["flattened"] = flattened
    report["feet"]["largest_horizontal_correction_m"] = float(max(np.linalg.norm(e[k], axis=1).max() for k in FEET))
    report["feet"]["largest_height_correction_m"] = float(max(np.abs(dz[k]).max() for k in FEET))

    # 5. legs: once with the pelvis height on a spring, then again with what it gave, smoothed
    q[:, 2] += shift
    targets = {}
    for k in FEET:
        pos = tracks[k]["pos"].copy()
        pos[:, :2] += e[k]
        pos[:, 2] += shift + dz[k]
        low = tracks[k]["low"] + shift + dz[k]  # sole height the foot is asked to have
        t = np.clip((low - 0.005) / 0.025, 0.0, 1.0)
        targets[k] = {"pos": pos, "rot": tracks[k]["rot"], "height_weight": 1.0 - 0.9 * t * t * (3.0 - 2.0 * t)}
    leg_index = {k: np.array([PM01_JOINT_NAMES.index(n) for n in LEG_JOINTS[k]]) for k in FEET}
    knee_floor = np.stack([np.minimum(base[:, 7 + leg_index[k][3]], KNEE_FLOOR_RAD) for k in FEET], axis=1)
    reference_z = q[:, 2].copy()
    start = q[:, 7:19].copy()
    solve_legs(pm, q, targets, leg_index, knee_floor, give=True)
    after = foot_tracks(pm, q)
    # The spring leaves about a tenth of the reach missing; take all of it plus a margin, hold it
    # over neighbouring frames (lower envelope) and smooth, so that the second solve finds every
    # grounded foot within reach and the pelvis moves gently.
    gave = q[:, 2] - reference_z
    gave = np.where(gave < -5e-4, gave / 0.9 - 0.001, 0.0)
    window = max(3, int(round(PELVIS_DROP_HOLD_S * fps)) | 1)
    gave = np.minimum(smooth(minimum_filter1d(gave, window, mode="nearest"), PELVIS_DROP_HOLD_S / 4.0, fps), 0.0)
    q[:, 2] = reference_z + gave
    # Where joint limits kept a foot more tilted than asked, its sole is lower than planned: raise it.
    for k in FEET:
        wanted = np.where(near_ground_weight(contacts, k) > 0.0, tracks[k]["low"] + shift + dz[k], 0.0)
        targets[k]["pos"][:, 2] += np.maximum(wanted - after[k]["low"], 0.0)
    q[:, 7:19] = start
    report["legs"] = solve_legs(pm, q, targets, leg_index, knee_floor, give=False)
    report["legs"]["largest_pelvis_drop_for_reach_m"] = float(-gave.min())
    final = foot_tracks(pm, q)
    clearance = np.array([
        feet_separation(pm, {k: (final[k]["pos"][t], final[k]["rot"][t]) for k in FEET}, 0.0)[0] if contacts["on_feet"][t] > 0.5 else np.inf
        for t in range(q.shape[0])
    ])
    report["feet"]["foot_clearance_min_final_m"] = float(clearance.min())
    report["feet"]["frames_feet_closer_than_gap_final"] = int((clearance < args.foot_gap - 1e-3).sum())
    report["feet"]["frames_feet_overlapping_final"] = int((clearance < 0.0).sum())
    report["feet"]["lowest_sole_final_m"] = float(min(final[k]["low"].min() for k in FEET))
    report["feet"]["stance_slip_at_contact_point_m_s"] = {
        "g1": contacts["stance_slip_m_s"],
        "hhtools": slip_before,
        "refined": stance_slip(final, pm.sole_corners, contacts["planted"], fps),
    }
    leg_cols = 7 + np.concatenate(list(leg_index.values()))
    report["legs"]["largest_joint_change_deg"] = float(np.degrees(np.abs(q[:, leg_cols] - base[:, leg_cols]).max()))
    report["legs"]["mean_joint_change_deg"] = float(np.degrees(np.abs(q[:, leg_cols] - base[:, leg_cols]).mean()))

    # 6. arms
    report["arms"] = clear_arms(pm, q, fps, args.arm_clearance)

    lower, upper = pm.limits[:, 0], pm.limits[:, 1]
    if (q[:, 7:] < lower - 1e-9).any() or (q[:, 7:] > upper + 1e-9).any() or not np.isfinite(q).all():
        raise RuntimeError("refined trajectory left the joint limits")
    speed = np.abs(np.diff(q[:, 7:], axis=0)) * fps
    report["joint_speed"] = {
        "max_rad_s": float(speed.max()),
        "max_over_actuator_limit": float((speed / PM01_SPEED_LIMITS).max()),
        "hhtools_max_over_actuator_limit": float((np.abs(np.diff(base[:, 7:], axis=0)) * fps / PM01_SPEED_LIMITS).max()),
    }
    accel = lambda x: np.percentile(np.abs(np.diff(x[:, 7:], n=2, axis=0)) * fps**2, 99)  # noqa: E731
    report["joint_accel_p99_rad_s2"] = {"hhtools": float(accel(base)), "refined": float(accel(q)), "g1": float(accel(src))}

    meta = {k: v for k, v in meta.items() if k != "sample_rate"}
    meta.update(
        {
            "refined_from": str(base_path.relative_to(WORK)),
            "refined_by": "refine_motions.py (soft elbow range, grounding, no-slip feet, foot gap, leg IK, arm clearance)",
            "refined_foot_gap_m": f"{args.foot_gap:g}",
        }
    )
    write_hhtools_csv(out_path, q, fps, PM01_JOINT_NAMES, meta)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--clip", action="append", default=None, help="Only this clip stem (repeatable). Default: every clip.")
    parser.add_argument("--keep-foot-tilt", action="store_true",
                        help="Clear a sole that would be below the floor by raising the foot (bending the leg) instead of reducing its tilt.")
    parser.add_argument("--foot-gap", type=float, default=0.01, help="Smallest distance kept between the feet's collision shapes, m.")
    parser.add_argument("--arm-clearance", type=float, default=0.002, help="Distance kept between forearm / hand and torso, head, other arm, m.")
    parser.add_argument("--base", type=Path, default=WORK / "pm01_hhtools")
    parser.add_argument("--out", type=Path, default=WORK / "pm01_refined")
    parser.add_argument("--report", type=Path, default=WORK / "qa" / "refine")
    args = parser.parse_args()

    wanted = set(args.clip) if args.clip else None
    bases = [p for p in sorted(args.base.glob("*/*.csv")) if wanted is None or p.stem in wanted]
    missing = (wanted or set()) - {p.stem for p in bases}
    if missing:
        raise SystemExit(f"no hhtools export for: {sorted(missing)}")

    pm, g1 = Pm01(), Robot(G1_MJCF, G1_JOINT_NAMES)
    args.report.mkdir(parents=True, exist_ok=True)
    print(
        f"{'clip':38s} {'body fr':>7s} {'pelvis dz cm':>13s} {'feet xy cm':>10s} {'feet gap cm':>13s} {'slip cm/s':>12s} "
        f"{'leg deg':>8s} {'arm fr':>6s} {'acc p99':>13s}"
    )
    for base_path in bases:
        group, stem = base_path.parent.name, base_path.stem
        report = refine_clip(pm, g1, WORK / "g1_source" / group / f"{stem}.csv", base_path, args.out / group / f"{stem}.csv", args)
        (args.report / f"{stem}.json").write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
        feet, legs, shift, acc = report["feet"], report["legs"], report["root_height"], report["joint_accel_p99_rad_s2"]
        print(
            f"{stem:38s} {shift['frames_resting_on_body']:7d} {shift['min_m'] * 100:6.2f}/{shift['max_m'] * 100:6.2f} "
            f"{feet['largest_horizontal_correction_m'] * 100:10.2f} {min(feet['foot_clearance_min_before_m'], 9.99) * 100:6.1f}>{min(feet['foot_clearance_min_final_m'], 9.99) * 100:6.1f} "
            f"{feet['stance_slip_at_contact_point_m_s']['hhtools'] * 100:5.2f}>{feet['stance_slip_at_contact_point_m_s']['refined'] * 100:5.2f} "
            f"{legs['largest_joint_change_deg']:8.1f} {report['arms']['frames_adjusted']:6d} "
            f"{acc['hhtools']:6.0f}>{acc['refined']:6.0f}"
        )
    print(f"\nrefined {len(bases)} clip(s) -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
