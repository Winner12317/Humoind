"""Remove floor and self penetration from the Recovery reference clips.

    python scripts/clean_recovery_reference.py --robot g1
    python scripts/clean_recovery_reference.py --robot g1 --report-only

The fall-and-get-up clip was retargeted without contact constraints: while the
robot is on the ground, thighs, knees and hands are centimetres below the floor
and the arms pass through the torso. Such frames are used twice in training, as
reset states (the simulator ejects the robot from the floor) and as AMP experts
(a pose no physical robot can reach scores as the most expert-like).

Each frame is repaired on its own, against the collision model of the training
simulator (same geoms, same excluded pairs). The repaired pose is the one closest
to the original in which

* no geom is below the floor and no two links overlap (hard constraints,
  linearised and solved as a small quadratic programme, repeated to convergence);
* a geom that rested on or in the floor in the original stays about as close to
  it (a penalty), so the robot is not simply lifted off the ground.

"Closest" weighs joints lightly, root height more, and root orientation and
horizontal position heavily; joints stay inside their limits.

The per-frame corrections are then smoothed over time and the repair is run once
more from the smoothed poses. Velocities and body states are recomputed the way
``scripts/csv_to_npz.py`` computes them.

Input : ``src/assets/motions/<robot>/amp/{WalkandRun,Recovery}/*.npz`` (not modified)
Output: ``src/assets/motions/<robot>/amp_clean/`` with the same layout: cleaned
        Recovery clips, byte-identical copies of the walking clips, and
        ``clean_manifest.json`` (settings, source hashes, before/after numbers).

Train on it with ``train_wmac.py --reference-motion-root src/assets/motions/<robot>/amp_clean``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import mujoco
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import nnls

MOTION_KEYS = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
CLEARANCE_WEIGHT = 100.0  # per m^2 of an original floor contact floating; joint motion costs 1 per rad^2
ROOT_COST = (1e3, 1e3, 30.0, 100.0, 100.0, 100.0)  # x, y, z (per m^2); rotation (per rad^2)
JOINT_COST = 1.0
MAX_STEP = 0.1  # largest change of any coordinate in one iteration (m or rad)
NEAR = 0.03  # constraints are written for geoms this close to the floor or to each other (m)


def closest_feasible_step(hessian, gradient, constraints, bounds):
    """argmin 1/2 x'Hx - g'x subject to constraints @ x >= bounds.

    Least-distance form solved with non-negative least squares (Lawson and Hanson,
    Solving Least Squares Problems, ch. 23). Returns the unconstrained minimiser
    when the linearised constraints cannot all hold.
    """
    factor = np.linalg.cholesky(hessian)
    free = np.linalg.solve(hessian, gradient)
    if not len(constraints):
        return free
    scaled = np.linalg.solve(factor, constraints.T).T  # constraints @ factor^-T
    slack = bounds - constraints @ free
    if np.all(slack <= 0):
        return free
    system = np.vstack((scaled.T, slack[None]))
    target = np.zeros(len(system)); target[-1] = 1.0
    multipliers, _ = nnls(system, target, maxiter=20 * system.shape[1] + 100)
    residual = system @ multipliers - target
    if abs(residual[-1]) < 1e-12:
        return free
    return free + np.linalg.solve(factor.T, -residual[:-1] / residual[-1])


def build_model(robot: str) -> mujoco.MjModel:
    """The training scene (robot with its collision settings on a plane), compiled on the CPU."""
    from mjlab.scene import Scene, SceneCfg
    from mjlab.terrains import TerrainEntityCfg
    from src.assets.robots import get_g1_robot_cfg, get_pm01_robot_cfg

    entity = {"g1": get_g1_robot_cfg, "pm01": get_pm01_robot_cfg}[robot]()
    scene = Scene(SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1,
                           entities={"robot": entity}), device="cpu")
    return scene.compile()


class Robot:
    """Kinematics, penetration measurement and per-frame repair on one MuJoCo model."""

    def __init__(self, model: mujoco.MjModel, *, contact_band: float, floor_clearance: float,
                 self_clearance: float, iterations: int):
        self.model, self.data = model, mujoco.MjData(model)
        self.contact_band, self.floor_clearance = contact_band, floor_clearance
        self.self_clearance, self.iterations = self_clearance, iterations
        name = lambda kind, index: mujoco.mj_id2name(model, kind, index) or ""
        self.floor = next(g for g in range(model.ngeom) if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_PLANE)
        self.geoms = [g for g in range(model.ngeom) if g != self.floor
                      and (model.geom_contype[g] or model.geom_conaffinity[g])]
        self.geom_names = [name(mujoco.mjtObj.mjOBJ_GEOM, g).split("/")[-1] for g in self.geoms]
        self.bodies = [b for b in range(model.nbody) if name(mujoco.mjtObj.mjOBJ_BODY, b).startswith("robot/")]
        self.body_names = [name(mujoco.mjtObj.mjOBJ_BODY, b).split("/")[-1] for b in self.bodies]
        hinges = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
        free = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        if len(free) != 1 or model.jnt_qposadr[free[0]] != 0 or len(hinges) != model.nv - 6:
            raise ValueError("Expected one floating base followed by hinge joints")
        self.joint_names = [name(mujoco.mjtObj.mjOBJ_JOINT, j).split("/")[-1] for j in hinges]
        self.qadr = model.jnt_qposadr[hinges]
        self.vadr = model.jnt_dofadr[hinges]
        self.limits = model.jnt_range[hinges].copy()
        self.cost = np.full(model.nv, JOINT_COST)
        self.cost[:6] = ROOT_COST
        # The collision pass reports robot-robot pairs within this distance.
        model.geom_margin[self.geoms] = max(self_clearance, 0.0) + NEAR

    # -- kinematics --------------------------------------------------------------
    def qpos(self, root_pos, root_quat, joint_pos) -> np.ndarray:
        q = np.zeros(self.model.nq)
        q[:3], q[3:7] = root_pos, root_quat / np.linalg.norm(root_quat)
        q[self.qadr] = joint_pos
        return q

    def forward(self, q) -> None:
        self.data.qpos[:] = q
        mujoco.mj_kinematics(self.model, self.data)
        mujoco.mj_comPos(self.model, self.data)

    def floor_points(self) -> list[list[tuple[float, np.ndarray]]]:
        """Per collision geom, (height above the floor, point) of each of its low points.

        A capsule has two (one per end) and a box eight (its corners): constraining
        only the lowest one lets a flat-lying shape rock onto its other end.
        """
        model, data, result = self.model, self.data, []
        floor_point, floor_normal = data.geom_xpos[self.floor], data.geom_xmat[self.floor].reshape(3, 3)[:, 2]
        segment = np.zeros(6)
        for geom in self.geoms:
            centre, frame, size = data.geom_xpos[geom], data.geom_xmat[geom].reshape(3, 3), model.geom_size[geom]
            kind = model.geom_type[geom]
            if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                centres, radius = [centre], size[0]
            elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                centres, radius = [centre + sign * size[1] * frame[:, 2] for sign in (1.0, -1.0)], size[0]
            elif kind == mujoco.mjtGeom.mjGEOM_BOX:
                centres, radius = [centre + frame @ (size * np.array(signs)) for signs in
                                   np.ndindex(2, 2, 2) for signs in [np.array(signs) * 2.0 - 1.0]], 0.0
            else:
                distance = mujoco.mj_geomDistance(model, data, geom, self.floor, 10.0, segment)
                result.append([(float(distance), segment[:3].copy())])
                continue
            result.append([(float((c - floor_point) @ floor_normal - radius), c - radius * floor_normal)
                           for c in centres])
        return result

    def floor_distances(self) -> np.ndarray:
        """Signed distance of every collision geom to the floor."""
        return np.array([min(distance for distance, _ in points) for points in self.floor_points()])

    def self_contacts(self) -> list[tuple[float, int, int, np.ndarray, np.ndarray]]:
        """(distance, geom, geom, point, normal from the first to the second) of robot-robot pairs."""
        mujoco.mj_collision(self.model, self.data)
        result = []
        for contact in self.data.contact[: self.data.ncon]:
            first, second = int(contact.geom[0]), int(contact.geom[1])
            if self.floor in (first, second):
                continue
            result.append((float(contact.dist), first, second, contact.pos.copy(), contact.frame[:3].copy()))
        return result

    def point_jacobian(self, point, geom) -> np.ndarray:
        jacobian = np.zeros((3, self.model.nv))
        mujoco.mj_jac(self.model, self.data, jacobian, None, point, int(self.model.geom_bodyid[geom]))
        return jacobian

    # -- measurement ---------------------------------------------------------------
    def penetration(self, q) -> dict:
        self.forward(q)
        floor = self.floor_distances()
        contacts = self.self_contacts()
        worst = min(contacts, key=lambda item: item[0], default=None)
        return {"floor_depth": float(max(0.0, -floor.min())), "floor_geom": self.geom_names[int(floor.argmin())],
                "self_depth": float(max(0.0, -worst[0])) if worst else 0.0,
                "self_pair": (self.geom_names[self.geoms.index(worst[1])],
                              self.geom_names[self.geoms.index(worst[2])]) if worst and worst[0] < 0 else None}

    # -- repair ----------------------------------------------------------------------
    def repair(self, q_start, q_reference, targets) -> np.ndarray:
        """Closest pose to ``q_reference`` without penetration, iterating from ``q_start``.

        ``targets`` holds, per collision geom, the floor clearance to keep (NaN for a
        geom that was not near the floor in the original frame).
        """
        model, q = self.model, q_start.copy()
        deviation = np.zeros(model.nv)
        for _ in range(self.iterations):
            self.forward(q)
            mujoco.mj_differentiatePos(model, deviation, 1.0, q_reference, q)
            # Objective: cost * (deviation from the original)^2 + floating penalties.
            hessian, gradient = np.diag(self.cost), -self.cost * deviation
            constraints, bounds = [], []
            for index, (geom, points) in enumerate(zip(self.geoms, self.floor_points())):
                target = targets[index]
                lowest = min(range(len(points)), key=lambda item: points[item][0])
                for item, (distance, point) in enumerate(points):
                    # Only a geom's lowest point is held to the clearance it had.
                    floating = item == lowest and not np.isnan(target) and distance > target
                    if not floating and distance > self.floor_clearance + NEAR:
                        continue
                    row = self.point_jacobian(point, geom)[2]
                    if floating:
                        hessian += CLEARANCE_WEIGHT * np.outer(row, row)
                        gradient += CLEARANCE_WEIGHT * (target - distance) * row
                    if distance <= self.floor_clearance + NEAR:
                        constraints.append(row); bounds.append(self.floor_clearance - distance)
            for distance, first, second, point, normal in self.self_contacts():
                constraints.append(normal @ (self.point_jacobian(point, second) - self.point_jacobian(point, first)))
                bounds.append(self.self_clearance - distance)
            # Joint limits are constraints too, so a joint at its stop is not relied on.
            joints = q[self.qadr]
            for column, low, high in zip(self.vadr, self.limits[:, 0] - joints, self.limits[:, 1] - joints):
                for sign, bound in ((1.0, low), (-1.0, -high)):
                    if bound > -MAX_STEP:
                        row = np.zeros(model.nv); row[column] = sign
                        constraints.append(row); bounds.append(min(bound, 0.0))
            step = closest_feasible_step(hessian, gradient, np.asarray(constraints).reshape(-1, model.nv),
                                         np.asarray(bounds))
            step *= min(1.0, MAX_STEP / max(np.abs(step).max(), 1e-12))
            mujoco.mj_integratePos(model, q, step, 1.0)
            q[self.qadr] = np.clip(q[self.qadr], self.limits[:, 0], self.limits[:, 1])
            if np.abs(step).max() < 1e-5:
                break
        return q

    def floor_targets(self, q_reference) -> np.ndarray:
        self.forward(q_reference)
        floor = self.floor_distances()
        return np.where(floor < self.contact_band, np.maximum(floor, self.floor_clearance), np.nan)

    # -- export -------------------------------------------------------------------------
    def motion(self, q: np.ndarray, fps: float) -> dict[str, np.ndarray]:
        """Motion arrays of a pose sequence, with csv_to_npz.py's finite differences."""
        dt = 1.0 / fps
        root_pos, root_quat, joint_pos = q[:, :3], q[:, 3:7], q[:, self.qadr]
        linear = np.gradient(root_pos, dt, axis=0)
        joint_vel = np.gradient(joint_pos, dt, axis=0)
        angular = so3_derivative(root_quat, dt)
        result = {key: [] for key in MOTION_KEYS}
        velocity, rotation = np.zeros(6), np.zeros(9)
        for frame in range(len(q)):
            self.data.qpos[:] = q[frame]
            mujoco.mju_quat2Mat(rotation, q[frame, 3:7])
            self.data.qvel[:3] = linear[frame]
            self.data.qvel[3:6] = rotation.reshape(3, 3).T @ angular[frame]  # MuJoCo: body-frame angular velocity
            self.data.qvel[self.vadr] = joint_vel[frame]
            mujoco.mj_kinematics(self.model, self.data)
            mujoco.mj_comPos(self.model, self.data)
            mujoco.mj_comVel(self.model, self.data)
            body_linear, body_angular = [], []
            for body in self.bodies:
                mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_XBODY, body, velocity, 0)
                body_angular.append(velocity[:3].copy()); body_linear.append(velocity[3:].copy())
            result["joint_pos"].append(joint_pos[frame]); result["joint_vel"].append(joint_vel[frame])
            result["body_pos_w"].append(self.data.xpos[self.bodies].copy())
            result["body_quat_w"].append(self.data.xquat[self.bodies].copy())
            result["body_lin_vel_w"].append(np.asarray(body_linear)); result["body_ang_vel_w"].append(np.asarray(body_angular))
        return {key: np.asarray(value, dtype=np.float32) for key, value in result.items()}


def so3_derivative(quaternions: np.ndarray, dt: float) -> np.ndarray:
    """World-frame angular velocity by central differences (csv_to_npz.MotionLoader._so3_derivative)."""
    relative = np.zeros((len(quaternions) - 2, 4))
    conjugate, velocity = np.zeros(4), np.zeros(3)
    result = np.zeros((len(quaternions) - 2, 3))
    for index in range(len(relative)):
        mujoco.mju_negQuat(conjugate, quaternions[index])
        mujoco.mju_mulQuat(relative[index], quaternions[index + 2], conjugate)
        mujoco.mju_quat2Vel(velocity, relative[index], 2.0 * dt)
        result[index] = velocity
    return np.concatenate((result[:1], result, result[-1:]))


def summarize(robot: Robot, q: np.ndarray, fps: float) -> dict:
    rows = [robot.penetration(frame) for frame in q]
    floor = np.array([row["floor_depth"] for row in rows])
    own = np.array([row["self_depth"] for row in rows])
    height = q[:, 2]
    w, x, y, z = q[:, 3:7].T
    lying = (height < 0.45 * height.max()) | (1 - 2 * (x * x + y * y) < 0.5)
    speed = np.abs(np.gradient(q[:, robot.qadr], 1.0 / fps, axis=0))
    worst_floor, worst_self = int(floor.argmax()), int(own.argmax())
    return {
        "frames": len(q), "frames_on_ground": int(lying.sum()),
        "floor_over_1cm_share_on_ground": round(float((floor[lying] > 0.01).mean()) if lying.any() else 0.0, 4),
        "floor_over_1cm_share_upright": round(float((floor[~lying] > 0.01).mean()) if (~lying).any() else 0.0, 4),
        "floor_over_1mm_share": round(float((floor > 0.001).mean()), 4),
        "floor_max_m": round(float(floor.max()), 4), "floor_worst": [worst_floor, rows[worst_floor]["floor_geom"]],
        "self_over_1cm_share": round(float((own > 0.01).mean()), 4),
        "self_over_1mm_share": round(float((own > 0.001).mean()), 4),
        "self_max_m": round(float(own.max()), 4), "self_worst": [worst_self, rows[worst_self]["self_pair"]],
        "joint_speed_p99_rad_s": round(float(np.percentile(speed, 99)), 3),
        "joint_speed_max_rad_s": round(float(speed.max()), 3),
    }


def clean(robot: Robot, q_reference: np.ndarray, *, smooth_frames: float) -> np.ndarray:
    model = robot.model
    targets = [robot.floor_targets(frame) for frame in q_reference]
    q = np.stack([robot.repair(frame, frame, target) for frame, target in zip(q_reference, targets)])
    if smooth_frames > 0:
        # Smooth the correction, not the motion: untouched frames stay untouched.
        deviation = np.zeros((len(q), model.nv))
        for index in range(len(q)):
            mujoco.mj_differentiatePos(model, deviation[index], 1.0, q_reference[index], q[index])
        deviation = gaussian_filter1d(deviation, smooth_frames, axis=0, mode="nearest")
        for index in range(len(q)):
            q[index] = q_reference[index]
            mujoco.mj_integratePos(model, q[index], deviation[index], 1.0)
            # From the smoothed pose, stay close to it: only what still penetrates moves.
            q[index] = robot.repair(q[index], q[index], targets[index])
    return q


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--robot", choices=("g1", "pm01"), default="g1")
    parser.add_argument("--source", type=Path, help="motion directory (default src/assets/motions/<robot>/amp)")
    parser.add_argument("--output", type=Path, help="new directory (default <source>_clean)")
    parser.add_argument("--report-only", action="store_true", help="measure the source clips; write nothing")
    parser.add_argument("--contact-band", type=float, default=0.01,
                        help="geoms within this height of the floor in the original keep their clearance (m)")
    parser.add_argument("--floor-clearance", type=float, default=0.0, help="smallest height above the floor (m)")
    parser.add_argument("--self-clearance", type=float, default=0.0, help="smallest distance between links (m)")
    parser.add_argument("--smooth-frames", type=float, default=2.0,
                        help="Gaussian sigma for the corrections, in frames (0: none)")
    parser.add_argument("--iterations", type=int, default=30)
    args = parser.parse_args()
    source = (args.source or REPO / "src" / "assets" / "motions" / args.robot / "amp").resolve()
    output = (args.output or source.with_name(source.name + "_clean")).resolve()
    clips = sorted((source / "Recovery").glob("*.npz"))
    if not clips:
        raise SystemExit(f"No Recovery clips under {source}")
    if not args.report_only and output.exists():
        raise SystemExit(f"{output} exists; choose another --output or remove it")

    robot = Robot(build_model(args.robot), contact_band=args.contact_band, floor_clearance=args.floor_clearance,
                  self_clearance=args.self_clearance, iterations=args.iterations)
    manifest = {"format": "recovery-reference-clean-v1", "robot": args.robot, "source_root": str(source),
                "settings": {key: getattr(args, key) for key in
                             ("contact_band", "floor_clearance", "self_clearance", "smooth_frames", "iterations")},
                "costs": {"kept_clearance": CLEARANCE_WEIGHT, "root_xyz_rotation": ROOT_COST, "joint": JOINT_COST},
                "source_sha256": {}, "output_sha256": {}, "clips": {}}
    staging = output.with_name(output.name + ".partial")
    if not args.report_only:
        shutil.rmtree(staging, ignore_errors=True)
        (staging / "Recovery").mkdir(parents=True)
    for path in clips:
        with np.load(path, allow_pickle=False) as data:
            motion = {key: np.asarray(data[key]) for key in data.files}
        fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
        if motion["joint_pos"].shape[1] != len(robot.joint_names) or motion["body_pos_w"].shape[1] != len(robot.bodies):
            raise SystemExit(f"{path.name} does not match the {args.robot} model")
        q_reference = np.stack([robot.qpos(p, r, j) for p, r, j in
                                zip(motion["body_pos_w"][:, 0], motion["body_quat_w"][:, 0], motion["joint_pos"])])
        # The exporter must reproduce the stored arrays of the untouched clip.
        rebuilt = robot.motion(q_reference, fps)
        mismatch = {key: float(np.abs(rebuilt[key] - motion[key]).max()) for key in MOTION_KEYS}
        before = summarize(robot, q_reference, fps)
        print(f"{path.name}: {before['frames']} frames at {fps:g} Hz; re-export of the original differs by "
              f"{max(mismatch.values()):.1e} at most ({max(mismatch, key=mismatch.get)})")
        print(f"  before: {json.dumps(before)}")
        entry = {"fps": fps, "reexport_max_abs_difference": mismatch, "before": before}
        if not args.report_only:
            if max(mismatch.values()) > 1e-2:
                raise SystemExit("Re-export of the original clip does not match its stored arrays; not writing")
            q = clean(robot, q_reference, smooth_frames=args.smooth_frames)
            after = summarize(robot, q, fps)
            change = np.zeros((len(q), robot.model.nv))
            for index in range(len(q)):
                mujoco.mj_differentiatePos(robot.model, change[index], 1.0, q_reference[index], q[index])
            moved = np.abs(change).max(1) > 1e-4
            entry.update(after=after, change={
                "frames_changed_share": round(float(moved.mean()), 4),
                "root_height_max_m": round(float(np.abs(change[:, 2]).max()), 4),
                "root_horizontal_max_m": round(float(np.abs(change[:, :2]).max()), 4),
                "root_rotation_max_rad": round(float(np.abs(change[:, 3:6]).max()), 4),
                "joint_max_rad": round(float(np.abs(change[:, 6:]).max()), 4),
                "joint_mean_abs_rad_changed_frames": round(float(np.abs(change[moved][:, 6:]).mean()), 4) if moved.any() else 0.0,
                "largest_joint": robot.joint_names[int(np.abs(change[:, 6:]).max(0).argmax())],
            })
            print(f"  after : {json.dumps(after)}")
            print(f"  change: {json.dumps(entry['change'])}")
            np.savez(staging / "Recovery" / path.name, fps=motion["fps"], **robot.motion(q, fps))
        manifest["clips"][path.name] = entry
        manifest["source_sha256"][f"Recovery/{path.name}"] = sha256(path)
    if args.report_only:
        return 0
    shutil.copytree(source / "WalkandRun", staging / "WalkandRun")
    for path in sorted((source / "WalkandRun").glob("*.npz")):
        manifest["source_sha256"][f"WalkandRun/{path.name}"] = sha256(path)
    for path in sorted(staging.rglob("*.npz")):
        manifest["output_sha256"][str(path.relative_to(staging))] = sha256(path)
    (staging / "clean_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    staging.rename(output)
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
