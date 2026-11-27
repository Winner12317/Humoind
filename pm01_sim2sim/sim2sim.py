"""Sim-to-sim validation of SuperAMP's PM01 policies in MuJoCo, with keyboard control.

    python sim2sim.py                         # interactive viewer, EngineAI SDK physics
    python sim2sim.py --policy-dir policy_wamp_v2   # SuperMimic + AMP v2 (default: policy = v1)
    python sim2sim.py --policy-dir policy_wamp_v3_70400   # SuperMimic + AMP v3, 70,400 iterations
    python sim2sim.py --policy-dir policy_ppo_amp   # the PPO + AMP policy
    python sim2sim.py --dynamics training     # the training environment's joint/actuator model
    python sim2sim.py --headless              # scripted walk / get-up / push test, prints a report
    python sim2sim.py --headless --record run.mp4

The policy runs as it would on the robot: an ONNX graph fed from the IMU and joint
encoders at 50 Hz, its joint targets tracked by a PD loop at the physics rate.

Physics (``--dynamics``):
  sdk       EngineAI's own sim-to-sim setup (engineai_robotics_native_sdk, run_mujoco.sh):
            the official PM01 MJCF as published, 2 ms steps, implicitfast; the PD law
            tau = kp (q* - q) - kd qd evaluated every physics step and sent to the torque
            motors, which clamp it at their ±164 / ±61 N m ranges; official joint
            armature (0.045 / 0.039), damping 0.6 and friction loss 0.8; head yaw held by
            the SDK controller's PD (kp 50, kd 0.3); no self-collision exclusions.
  training  the same geometry with the training environment's dynamics: 5 ms steps,
            armature 0.0453 / 0.0067, PD as an implicit position actuator, torque limits
            164 / 52 N m with no torque above the motor speed limit, actuator friction
            0.8 tanh(qd / 0.01) + 0.6 qd, head welded, foot friction 0.6, the four
            hand/thigh contact exclusions. A policy that fails here but not in the
            training simulator points at the observation pipeline; one that fails only
            with ``sdk`` points at the dynamics gap.

Keyboard (terminal window, or arrow keys in the viewer window):
  W / S, Up / Down       forward / backward speed  +- 0.1 m/s
  A / D                  sideways speed (left / right)  +- 0.1 m/s
  Q / E, Left / Right    turn left / right  +- 0.1 rad/s
  Space or X             stop (zero command)
  R                      reset: standing in the default pose
  F                      reset: lying on the floor (cycles through 8 poses), to test getting up
  P                      push: random velocity kick with the training push ranges
  - / =                  slower / faster playback (1/8x ... 1x)
  H                      print this help
  Ctrl+C or close the viewer   quit
In the viewer, Ctrl + right-drag on a body pulls it (MuJoCo's perturbation).
"""

from __future__ import annotations

import argparse
import atexit
import copy
import hashlib
import json
import math
import os
import queue
import select
import sys
import threading
import time
from dataclasses import dataclass, field
from functools import cached_property
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL = HERE / "model" / "pm01_edu.xml"

# Contact pairs the training model excludes: the official hand sphere sits inside the
# thigh and hip-roll cylinders of its own side with every joint at zero.
TRAINING_EXCLUDES = (
    ("LINK_ELBOW_END_L", "LINK_HIP_YAW_L"), ("LINK_ELBOW_END_L", "LINK_HIP_ROLL_L"),
    ("LINK_ELBOW_END_R", "LINK_HIP_YAW_R"), ("LINK_ELBOW_END_R", "LINK_HIP_ROLL_R"),
)
SDK_HEAD_GAINS = (50.0, 0.3)  # engineai_robotics_native_sdk assets/config/pm01_edu/rl_lab/default.yaml
STEP = {"vx": 0.1, "vy": 0.1, "wz": 0.1}
CLEAR_LINE = "\r\x1b[K" if sys.stdout.isatty() else ""
HELP = __doc__[__doc__.index("Keyboard"):].rstrip()


# --------------------------------------------------------------------------- policy

class Policy:
    """The exported actor: (312,) observation -> (23,) action (clipped inside, where trained so)."""

    def __init__(self, directory: Path):
        import onnxruntime as ort

        if not directory.is_dir() and (HERE / directory).is_dir():
            directory = HERE / directory  # a name relative to this package, from any working directory
        self.directory = directory
        self.meta = json.loads((directory / "policy.json").read_text())
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(str(directory / "policy.onnx"), options,
                                            providers=["CPUExecutionProvider"])
        self.obs_dim = self.meta["observation"]["dim"]

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return self.session.run(None, {"obs": obs.reshape(1, -1).astype(np.float32)})[0][0].astype(np.float64)


# ---------------------------------------------------------------------------- model

def build_model(dynamics: str, meta: dict):
    import mujoco

    spec = mujoco.MjSpec.from_file(str(MODEL))
    control = meta["control"]
    if dynamics == "training":
        # The training model has no head joint (and so no use for the official keyframe,
        # whose qpos includes it).
        spec.delete(spec.actuator("motor_J23_HEAD_YAW"))
        spec.delete(spec.joint("J23_HEAD_YAW"))
        for key in list(spec.keys):
            spec.delete(key)
        for body1, body2 in TRAINING_EXCLUDES:
            spec.add_exclude(bodyname1=body1, bodyname2=body2)
    model = spec.compile()
    if dynamics == "training":
        opt = model.opt
        opt.timestep = control["training_physics_dt"]
        opt.iterations, opt.ls_iterations, opt.tolerance, opt.ccd_iterations = 10, 20, 1e-8, 50
        for i, name in enumerate(meta["joint_names"]):
            joint = model.joint(name)
            dof = model.jnt_dofadr[joint.id]
            model.dof_armature[dof] = control["armature"][i]
            model.dof_damping[dof] = 0.0
            model.dof_frictionloss[dof] = 0.0
            act = model.actuator(f"motor_{name}").id
            kp, kd, effort = control["kp"][i], control["kd"][i], control["effort_limit"][i]
            model.actuator_gainprm[act, 0] = kp
            model.actuator_biastype[act] = mujoco.mjtBias.mjBIAS_AFFINE
            model.actuator_biasprm[act, :3] = (0.0, -kp, -kd)
            model.actuator_ctrllimited[act] = 0
            model.actuator_forcelimited[act] = 1
            model.actuator_forcerange[act] = (-effort, effort)
        feet = {model.body("LINK_FOOT_L").id, model.body("LINK_FOOT_R").id}
        for geom in range(model.ngeom):
            if model.geom_bodyid[geom] == 0 or not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
                continue
            if model.geom_bodyid[geom] in feet:
                model.geom_friction[geom, 0], model.geom_priority[geom], model.geom_condim[geom] = 0.6, 1, 3
            else:
                model.geom_condim[geom] = 1
    elif dynamics != "sdk":
        raise ValueError(f"unknown dynamics {dynamics!r}")
    return model


def lowest_collision_point(model, data) -> float:
    """Lowest world z of the robot's collision primitives (spheres, boxes, cylinders)."""
    import mujoco

    lowest = math.inf
    for geom in range(model.ngeom):
        if model.geom_bodyid[geom] == 0 or not (model.geom_contype[geom] or model.geom_conaffinity[geom]):
            continue
        kind, size = model.geom_type[geom], model.geom_size[geom]
        rotation, z = data.geom_xmat[geom].reshape(3, 3), data.geom_xpos[geom][2]
        if kind == mujoco.mjtGeom.mjGEOM_SPHERE:
            reach = size[0]
        elif kind == mujoco.mjtGeom.mjGEOM_BOX:
            reach = float(np.abs(rotation[2]) @ size)
        elif kind in (mujoco.mjtGeom.mjGEOM_CYLINDER, mujoco.mjtGeom.mjGEOM_CAPSULE):
            axial = abs(rotation[2, 2]) * size[1]
            radial = size[0] * (1.0 if kind == mujoco.mjtGeom.mjGEOM_CAPSULE
                                else math.sqrt(max(0.0, 1.0 - rotation[2, 2] ** 2)))
            reach = axial + radial
        else:
            continue
        lowest = min(lowest, z - reach)
    return lowest


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


# ----------------------------------------------------------------------- controller

@dataclass
class Status:
    base_height: float = 0.0
    up_z: float = 1.0
    fallen: bool = False
    upright: bool = True
    measured: np.ndarray = field(default_factory=lambda: np.zeros(3))
    torque_saturated: np.ndarray | None = None


class Controller:
    """Observation history, policy inference and the PD loop, as on the robot."""

    def __init__(self, policy: Policy, dynamics: str, seed: int = 0, recovery_poses: str = "auto"):
        import mujoco

        self.mj = mujoco
        self.policy, self.meta, self.dynamics = policy, policy.meta, dynamics
        self.model = build_model(dynamics, self.meta)
        self.data = mujoco.MjData(self.model)
        model, control = self.model, self.meta["control"]
        names = self.meta["joint_names"]
        self.joint_qpos = np.array([model.jnt_qposadr[model.joint(n).id] for n in names])
        self.joint_dof = np.array([model.jnt_dofadr[model.joint(n).id] for n in names])
        self.joint_act = np.array([model.actuator(f"motor_{n}").id for n in names])
        self.head = None
        if dynamics == "sdk":
            head = model.joint("J23_HEAD_YAW").id
            self.head = (model.jnt_qposadr[head], model.jnt_dofadr[head], model.actuator("motor_J23_HEAD_YAW").id)
        self.default = np.array(control["default_joint_pos"])
        self.scale = np.array(control["action_scale"])
        self.kp, self.kd = np.array(control["kp"]), np.array(control["kd"])
        self.effort = np.array(control["effort_limit"])
        self.speed_limit = np.array(control["velocity_limit"])
        self.fs, self.fd = np.array(control["friction_static"]), np.array(control["friction_viscous"])
        self.fv = np.array(control["friction_activation_velocity"])
        self.ctrl_limit = model.actuator_ctrlrange[self.joint_act, 1] if dynamics == "sdk" else self.effort
        self.control_dt = control["control_dt"]
        self.decimation = round(self.control_dt / model.opt.timestep)
        if abs(self.decimation * model.opt.timestep - self.control_dt) > 1e-9:
            raise ValueError("the control period is not a whole number of physics steps")
        self.quat_sensor = model.sensor("imu_quaternion").id
        self.gyro_sensor = model.sensor("imu_angular_velocity").id
        self.base = model.body("LINK_BASE").id
        self.torso = model.body("LINK_TORSO_YAW").id
        self.frame_dim = self.meta["observation"]["frame_dim"]
        self.history_length = self.meta["observation"]["history_length"]
        self.thresholds = self.meta["thresholds_m"]
        self.recovery_poses = recovery_poses
        self.rng = np.random.default_rng(seed)
        self.command = np.zeros(3)
        self.status = Status()
        self.reset("stand")

    @cached_property
    def recovery_starts(self) -> tuple[str, np.ndarray]:
        """Load on the first fallen reset. Standing, walking and pushes need no motion clip."""
        archive = self.policy.directory / "start_poses.npz"
        reference = self.policy.directory / "start_poses.reference.json"
        if self.recovery_poses != "generic":
            if archive.is_file():
                with np.load(archive, allow_pickle=False) as poses:
                    return "reference_motion", poses["qpos"]
            if reference.is_file():
                spec = json.loads(reference.read_text())
                root = Path(os.environ.get("PM01_MOTION_ROOT", HERE.parent / "SuperAMP/src/assets/motions/pm01"))
                clip = root / spec["clip"]
                if clip.is_file():
                    with clip.open("rb") as stream:
                        if hashlib.sha256(stream.read()).hexdigest() != spec["clip_sha256"]:
                            raise ValueError(f"{clip} differs from the clip specified in {reference}.")
                        stream.seek(0)
                        with np.load(stream, allow_pickle=False) as motion:
                            frames = spec["frames"]
                            poses = np.concatenate((motion["body_pos_w"][frames, 0], motion["body_quat_w"][frames, 0],
                                                    motion["joint_pos"][frames]), axis=1).astype(np.float64)
                    poses[:, :2] = 0.0
                    return "reference_motion", poses
            if self.recovery_poses == "reference":
                raise FileNotFoundError(f"Reference resets require {archive} or the clip named by {reference}. "
                                        "Set PM01_MOTION_ROOT to your local PM01 motion directory.")
        return "generic_simulated", self._simulate_fallen_starts()

    def _simulate_fallen_starts(self) -> np.ndarray:
        # Generate falls in a separate passive model so the active controller and viewer are unaffected.
        model = copy.copy(self.model)
        model.opt.disableflags |= int(self.mj.mjtDisableBit.mjDSBL_ACTUATION)
        data = self.mj.MjData(model)
        poses = []
        half_angle = math.radians(80.0) / 2
        for degrees in range(0, 360, 45):
            axis = math.radians(degrees)
            self.mj.mj_resetData(model, data)
            data.qpos[self.joint_qpos] = self.default
            data.qpos[3:7] = (math.cos(half_angle), math.sin(half_angle) * math.cos(axis),
                              math.sin(half_angle) * math.sin(axis), 0.0)
            self.mj.mj_forward(model, data)
            data.qpos[2] += 0.02 - lowest_collision_point(model, data)
            self.mj.mj_step(model, data, nstep=round(1.5 / model.opt.timestep))
            poses.append(np.concatenate((data.qpos[:7], data.qpos[self.joint_qpos])))
        return np.stack(poses)

    @property
    def start_poses(self) -> np.ndarray:
        return self.recovery_starts[1]

    # -- resets -------------------------------------------------------------------

    def reset(self, start: str = "stand", pose_index: int = 0) -> None:
        model, data, mj = self.model, self.data, self.mj
        mj.mj_resetData(model, data)
        data.qpos[:] = model.qpos0
        if start == "stand":
            data.qpos[3:7] = (1.0, 0.0, 0.0, 0.0)
            data.qpos[self.joint_qpos] = self.default
            clearance = 0.001
        elif start == "fallen":
            pose = self.start_poses[pose_index % len(self.start_poses)]
            data.qpos[3:7] = pose[3:7]
            data.qpos[self.joint_qpos] = pose[7:]
            clearance = 0.002
        else:
            raise ValueError(start)
        data.qpos[:3] = (0.0, 0.0, 0.0)
        mj.mj_forward(model, data)
        data.qpos[2] = clearance - lowest_collision_point(model, data)
        mj.mj_forward(model, data)
        self.history = None
        self.last_action = np.zeros(len(self.default))
        self.target = data.qpos[self.joint_qpos].copy() if start == "fallen" else self.default.copy()
        self.time = 0.0
        self._update_status()

    def push(self) -> np.ndarray:
        """Add a random root velocity drawn from the training push ranges (world frame)."""
        ranges = self.meta["push"]["velocity_range"]
        kick = np.array([self.rng.uniform(*ranges[k]) for k in ("x", "y", "z", "roll", "pitch", "yaw")])
        data = self.data
        data.qvel[0:3] += kick[:3]
        rotation = quat_to_matrix(data.qpos[3:7])
        data.qvel[3:6] += rotation.T @ kick[3:]  # free-joint angular velocity is in the body frame
        return kick

    # -- one control period ----------------------------------------------------------

    def observe(self) -> np.ndarray:
        data = self.data
        quat = data.sensordata[self.model.sensor_adr[self.quat_sensor]:][:4]
        gyro = data.sensordata[self.model.sensor_adr[self.gyro_sensor]:][:3]
        gravity = quat_to_matrix(quat).T @ np.array([0.0, 0.0, -1.0])
        frame = np.concatenate((gyro, gravity, self.command, data.qpos[self.joint_qpos] - self.default,
                                data.qvel[self.joint_dof], self.last_action))
        if self.history is None:
            self.history = np.tile(frame, (self.history_length, 1))
        else:
            self.history = np.roll(self.history, -1, axis=0)
            self.history[-1] = frame
        return self.history.reshape(-1)

    def step(self) -> None:
        action = self.policy(self.observe())
        self.last_action = action
        self.target = self.default + self.scale * action
        saturated = np.zeros(len(self.default), dtype=bool)
        for _ in range(self.decimation):
            saturated |= self._apply_pd()
            self.mj.mj_step(self.model, self.data)
        self.status.torque_saturated = saturated
        self.time += self.control_dt
        self._update_status()

    def _apply_pd(self) -> np.ndarray:
        data = self.data
        q, qd = data.qpos[self.joint_qpos], data.qvel[self.joint_dof]
        torque = self.kp * (self.target - q) - self.kd * qd
        if self.dynamics == "sdk":
            # EngineAI's simulator: the motor actuator clamps the PD torque at its ctrlrange.
            data.ctrl[self.joint_act] = torque
            if self.head is not None:
                qpos, dof, act = self.head
                kp, kd = SDK_HEAD_GAINS
                data.ctrl[act] = kp * (0.0 - data.qpos[qpos]) - kd * data.qvel[dof]
            return np.abs(torque) >= self.ctrl_limit
        # Training actuator: torque limit, no torque above the speed limit, friction; the
        # PD is a position actuator, so ctrl encodes the resulting effort.
        limit = np.where(np.abs(qd) < self.speed_limit, self.effort, 0.0)
        effort = np.clip(torque, -limit, limit)
        effort -= self.fs * np.tanh(qd / self.fv) + self.fd * qd
        data.ctrl[self.joint_act] = q + (effort + self.kd * qd) / self.kp
        return np.abs(torque) >= limit

    def _update_status(self) -> None:
        data, status, thresholds = self.data, self.status, self.thresholds
        rotation = quat_to_matrix(data.qpos[3:7])
        status.base_height = float(data.qpos[2])
        status.up_z = float(rotation[2, 2])
        tilt = math.acos(max(-1.0, min(1.0, status.up_z)))
        status.fallen = status.base_height < thresholds["termination_height"] or tilt > thresholds["termination_tilt_rad"]
        status.upright = status.base_height > thresholds["upright_height"] and status.up_z > 0.8
        # Torso velocity in its heading frame (what training rewards), and base yaw rate.
        velocity = np.zeros(6)
        self.mj.mj_objectVelocity(self.model, data, self.mj.mjtObj.mjOBJ_XBODY, self.torso, velocity, 0)
        torso_rotation = data.xmat[self.torso].reshape(3, 3)
        yaw = math.atan2(torso_rotation[1, 0], torso_rotation[0, 0])
        c, s = math.cos(yaw), math.sin(yaw)
        status.measured = np.array([c * velocity[3] + s * velocity[4], -s * velocity[3] + c * velocity[4], velocity[2]])

    def set_command(self, vx=None, vy=None, wz=None) -> None:
        ranges = self.meta["command_ranges"]
        for index, (value, key) in enumerate(((vx, "lin_vel_x"), (vy, "lin_vel_y"), (wz, "ang_vel_z"))):
            if value is not None:
                low, high = ranges[key]
                self.command[index] = round(min(high, max(low, value)), 3)


# ------------------------------------------------------------------------- keyboard

class Keyboard:
    """Keys from the terminal (raw mode) and from the viewer window (arrow keys)."""

    GLFW_ARROWS = {265: "up", 264: "down", 263: "left", 262: "right"}

    def __init__(self):
        self.events: queue.Queue[str] = queue.Queue()
        self._saved = None
        self._stop = threading.Event()

    def start_terminal(self) -> bool:
        if not sys.stdin.isatty():
            return False
        import termios
        import tty

        fd = sys.stdin.fileno()
        self._saved = termios.tcgetattr(fd)
        tty.setcbreak(fd)  # keys arrive one by one, unechoed; Ctrl+C still interrupts
        atexit.register(self.stop)
        threading.Thread(target=self._read, args=(fd,), daemon=True).start()
        return True

    def stop(self) -> None:
        self._stop.set()
        if self._saved is not None:
            import termios

            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._saved)
            self._saved = None

    def _read(self, fd: int) -> None:
        while not self._stop.is_set():
            ready, _, _ = select.select([fd], [], [], 0.1)
            if not ready:
                continue
            text = os.read(fd, 64).decode(errors="ignore")
            i = 0
            while i < len(text):
                if text.startswith("\x1b[", i) and i + 2 < len(text):
                    arrow = {"A": "up", "B": "down", "C": "right", "D": "left"}.get(text[i + 2])
                    if arrow:
                        self.events.put(arrow)
                    i += 3
                    continue
                self.events.put(text[i].lower())
                i += 1

    def viewer_key(self, keycode: int) -> None:
        if keycode in self.GLFW_ARROWS:
            self.events.put(self.GLFW_ARROWS[keycode])


def apply_key(key: str, controller: Controller, state: dict) -> str | None:
    """Act on one key; returns a message for the terminal, if any."""
    vx, vy, wz = controller.command
    if key in ("w", "up"):
        controller.set_command(vx=vx + STEP["vx"])
    elif key in ("s", "down"):
        controller.set_command(vx=vx - STEP["vx"])
    elif key == "a":
        controller.set_command(vy=vy + STEP["vy"])
    elif key == "d":
        controller.set_command(vy=vy - STEP["vy"])
    elif key in ("q", "left"):
        controller.set_command(wz=wz + STEP["wz"])
    elif key in ("e", "right"):
        controller.set_command(wz=wz - STEP["wz"])
    elif key in (" ", "x"):
        controller.set_command(0.0, 0.0, 0.0)
    elif key == "r":
        controller.reset("stand")
        return "reset: standing"
    elif key == "f":
        index = state.setdefault("pose", 0)
        controller.reset("fallen", index)
        state["pose"] = index + 1
        return (f"reset: {controller.recovery_starts[0]} lying pose "
                f"{index % len(controller.start_poses) + 1}/{len(controller.start_poses)}")
    elif key == "p":
        kick = controller.push()
        return "push: v += ({:+.2f}, {:+.2f}, {:+.2f}) m/s, w += ({:+.2f}, {:+.2f}, {:+.2f}) rad/s".format(*kick)
    elif key == "-":
        state["speed"] = max(0.125, state["speed"] / 2)
        return f"playback {state['speed']:g}x"
    elif key == "=":
        state["speed"] = min(1.0, state["speed"] * 2)
        return f"playback {state['speed']:g}x"
    elif key == "h":
        return HELP
    return None


# ---------------------------------------------------------------------- interactive

def status_line(controller: Controller) -> str:
    s, c = controller.status, controller.command
    state = "FALLEN" if s.fallen else ("upright" if s.upright else "getting up")
    return (f"cmd vx {c[0]:+.1f} vy {c[1]:+.1f} wz {c[2]:+.1f} | measured vx {s.measured[0]:+.2f} "
            f"vy {s.measured[1]:+.2f} wz {s.measured[2]:+.2f} | base {s.base_height:.2f} m {state}")


def draw_arrows(scene, controller: Controller, clear: bool) -> None:
    """Commanded (green) and measured (blue) planar velocity, drawn at the robot's feet.

    ``clear`` empties the scene first: right for the viewer's user scene, which holds
    only these arrows; wrong for a renderer's scene, which holds the robot too.
    """
    mj, data = controller.mj, controller.data
    rotation = data.xmat[controller.torso].reshape(3, 3)
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    c, s = math.cos(yaw), math.sin(yaw)
    start = np.array([data.qpos[0], data.qpos[1], 0.02])
    if clear:
        scene.ngeom = 0
    for velocity, rgba in ((controller.command[:2], (0.2, 0.85, 0.3, 0.9)),
                           (controller.status.measured[:2], (0.2, 0.45, 1.0, 0.9))):
        world = np.array([c * velocity[0] - s * velocity[1], s * velocity[0] + c * velocity[1], 0.0])
        if np.linalg.norm(world) < 0.02 or scene.ngeom >= scene.maxgeom:
            continue
        geom = scene.geoms[scene.ngeom]
        mj.mjv_initGeom(geom, mj.mjtGeom.mjGEOM_ARROW, np.zeros(3), np.zeros(3), np.eye(3).ravel(),
                        np.array(rgba, dtype=np.float32))
        mj.mjv_connector(geom, mj.mjtGeom.mjGEOM_ARROW, 0.025, start, start + world)
        scene.ngeom += 1


def run_interactive(controller: Controller, start: str) -> None:
    import mujoco
    import mujoco.viewer

    keyboard = Keyboard()
    terminal = keyboard.start_terminal()
    state = {"speed": 1.0}
    if start == "fallen":
        print(apply_key("f", controller, state))
    print(f"PM01 sim2sim, dynamics '{controller.dynamics}', policy {controller.meta['source']['task']} "
          f"iteration {controller.meta['source']['iteration']}")
    print(HELP if terminal else "stdin is not a terminal: use the arrow keys in the viewer window")
    model, data = controller.model, controller.data
    with mujoco.viewer.launch_passive(model, data, key_callback=keyboard.viewer_key,
                                      show_left_ui=False, show_right_ui=False) as viewer:
        with viewer.lock():
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
            viewer.cam.trackbodyid = controller.base
            viewer.cam.distance, viewer.cam.elevation, viewer.cam.azimuth = 3.2, -15.0, 135.0
        next_tick = time.perf_counter()
        last_print = 0.0
        last_time = data.time
        while viewer.is_running():
            while not keyboard.events.empty():
                message = apply_key(keyboard.events.get(), controller, state)
                if message:
                    sys.stdout.write(CLEAR_LINE + message + "\n")
            if data.time < last_time:  # reset from the viewer's own UI
                controller.history, controller.last_action = None, np.zeros_like(controller.last_action)
            controller.step()
            last_time = data.time
            with viewer.lock():
                draw_arrows(viewer.user_scn, controller, clear=True)
            # MuJoCo takes the viewer lock inside set_texts.
            s, c = controller.status, controller.command
            viewer.set_texts([
                (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_TOPLEFT,
                 "command\nmeasured\nbase\nplayback",
                 f"vx {c[0]:+.1f}  vy {c[1]:+.1f}  wz {c[2]:+.1f}\n"
                 f"vx {s.measured[0]:+.2f}  vy {s.measured[1]:+.2f}  wz {s.measured[2]:+.2f}\n"
                 f"{s.base_height:.2f} m  {'FALLEN' if s.fallen else ('upright' if s.upright else 'getting up')}\n"
                 f"{state['speed']:g}x   ({controller.dynamics} physics)"),
                (mujoco.mjtFontScale.mjFONTSCALE_100, mujoco.mjtGridPos.mjGRID_BOTTOMLEFT,
                 "W/S A/D Q/E", "speed / sideways / turn (terminal; arrows also in this window)\n"
                 "Space stop   R stand   F lying   P push   -/= playback speed"),
            ])
            viewer.sync()
            now = time.perf_counter()
            if now - last_print > 0.2 and terminal:
                sys.stdout.write(CLEAR_LINE + status_line(controller))
                sys.stdout.flush()
                last_print = now
            next_tick += controller.control_dt / state["speed"]
            delay = next_tick - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.25:  # too far behind (window dragged, breakpoint...): resynchronise
                next_tick = time.perf_counter()
    keyboard.stop()
    print()


# ------------------------------------------------------------------------- headless

WALK_SEGMENTS = (  # seconds, vx, vy, wz
    ("stand", 3, 0.0, 0.0, 0.0), ("forward 0.5", 6, 0.5, 0.0, 0.0), ("forward 1.0", 6, 1.0, 0.0, 0.0),
    ("forward 1.5", 5, 1.5, 0.0, 0.0), ("stop", 3, 0.0, 0.0, 0.0), ("backward 0.5", 6, -0.5, 0.0, 0.0),
    ("left 0.4", 5, 0.0, 0.4, 0.0), ("right 0.4", 5, 0.0, -0.4, 0.0), ("turn left 0.8", 5, 0.0, 0.0, 0.8),
    ("turn right 0.8", 5, 0.0, 0.0, -0.8), ("arc 0.6 / 0.5", 6, 0.6, 0.0, 0.5), ("stand", 3, 0.0, 0.0, 0.0),
)


class Recorder:
    def __init__(self, controller: Controller, path: Path | None, every: int = 2):
        self.writer = None
        if path is None:
            return
        import imageio
        import mujoco

        self.mj = mujoco
        self.controller, self.every, self.count = controller, every, 0
        visual = controller.model.vis.global_
        visual.offwidth, visual.offheight = max(visual.offwidth, 854), max(visual.offheight, 480)
        self.renderer = mujoco.Renderer(controller.model, height=480, width=854)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.camera.trackbodyid = controller.base
        self.camera.distance, self.camera.elevation, self.camera.azimuth = 3.2, -15.0, 135.0
        self.writer = imageio.get_writer(str(path), fps=round(1 / (controller.control_dt * every)),
                                         codec="libx264", quality=7, macro_block_size=2)

    def frame(self, caption: str) -> None:
        if self.writer is None:
            return
        self.count += 1
        if self.count % self.every:
            return
        self.renderer.update_scene(self.controller.data, self.camera)
        draw_arrows(self.renderer.scene, self.controller, clear=False)
        self.writer.append_data(self.renderer.render())

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.renderer.close()  # release the EGL context before interpreter shutdown


def scenario_walk(controller: Controller, recorder: Recorder) -> dict:
    controller.reset("stand")
    rows, falls = [], 0
    for name, seconds, vx, vy, wz in WALK_SEGMENTS:
        controller.set_command(vx, vy, wz)
        steps = round(seconds / controller.control_dt)
        measured, fell, saturated = [], False, 0
        for k in range(steps):
            controller.step()
            recorder.frame(name)
            fell |= controller.status.fallen
            saturated += int(controller.status.torque_saturated.any())
            if k >= steps // 3:  # skip the transition into the new command
                measured.append(controller.status.measured.copy())
        measured = np.array(measured)
        command = np.array([vx, vy, wz])
        rows.append({"segment": name, "command": command.tolist(), "measured_mean": measured.mean(0).round(3).tolist(),
                     "abs_error_mean": np.abs(measured - command).mean(0).round(3).tolist(), "fell": fell,
                     "torque_limit_steps": saturated / steps})
        falls += fell
    return {"segments": rows, "falls": falls}


def scenario_recovery(controller: Controller, recorder: Recorder, seconds: float = 8.0) -> dict:
    rows, samples = [], []
    stable_needed = round(0.5 / controller.control_dt)
    for index in range(len(controller.start_poses)):
        controller.reset("fallen", index)
        controller.set_command(0.0, 0.0, 0.0)
        stable, stood_at, rise, speeds = 0, None, [], []
        for k in range(round(seconds / controller.control_dt)):
            controller.step()
            recorder.frame(f"get up {index + 1}")
            rise.append(float(controller.data.qvel[2]))
            speeds.append(np.abs(controller.data.qvel[controller.joint_dof]))
            stable = stable + 1 if controller.status.upright else 0
            if stood_at is None and stable >= stable_needed:
                stood_at = (k + 1 - stable_needed) * controller.control_dt
        # Tempo of the get-up itself: from the start until the robot first stands.
        end = len(rise) if stood_at is None else round(stood_at / controller.control_dt) + 1
        samples.append(np.array(speeds[:end]))
        rows.append({"pose": index + 1, "stood_up": stood_at is not None,
                     "seconds_to_stand": None if stood_at is None else round(stood_at, 2),
                     "peak_base_rise_m_s": round(max(rise[:end]), 2),
                     "peak_joint_speed_rad_s": round(float(samples[-1].max()), 1),
                     "upright_at_end": controller.status.upright})
    stood = [row["stood_up"] for row in rows]
    tempo = None
    if any(stood):
        speeds = np.concatenate([sample for sample, ok in zip(samples, stood) if ok])
        peaks = [row["peak_base_rise_m_s"] for row in rows if row["stood_up"]]
        tempo = {"peak_base_rise_m_s_mean": round(float(np.mean(peaks)), 2), "peak_base_rise_m_s_max": max(peaks),
                 "joint_speed_p95_rad_s": round(float(np.percentile(speeds, 95)), 1),
                 "joint_speed_max_rad_s": round(float(speeds.max()), 1)}
    return {"pose_source": controller.recovery_starts[0], "poses": rows, "success": sum(stood),
            "total": len(rows), "tempo_until_standing": tempo}


def scenario_push(controller: Controller, recorder: Recorder, seconds: float = 30.0, every: float = 2.5) -> dict:
    controller.reset("stand")
    controller.set_command(0.5, 0.0, 0.0)
    pushes = falls = stood_again = 0
    was_fallen = False
    for k in range(round(seconds / controller.control_dt)):
        if k * controller.control_dt >= 3.0 and k % round(every / controller.control_dt) == 0:
            controller.push()
            pushes += 1
        controller.step()
        recorder.frame("push")
        if controller.status.fallen and not was_fallen:
            falls += 1
        if was_fallen and controller.status.upright:
            stood_again += 1
        was_fallen = controller.status.fallen if controller.status.fallen else (was_fallen and not controller.status.upright)
    return {"pushes": pushes, "falls": falls, "stood_up_again": stood_again,
            "upright_at_end": controller.status.upright}


def run_headless(controller: Controller, scenarios: list[str], record: Path | None) -> dict:
    recorder = Recorder(controller, record)
    report = {"dynamics": controller.dynamics, "physics_dt": controller.model.opt.timestep,
              "policy": controller.meta["source"]}
    started = time.perf_counter()
    try:
        if "walk" in scenarios:
            report["walk"] = scenario_walk(controller, recorder)
        if "recovery" in scenarios:
            report["recovery"] = scenario_recovery(controller, recorder)
        if "push" in scenarios:
            report["push"] = scenario_push(controller, recorder)
    finally:
        recorder.close()
    report["wall_seconds"] = round(time.perf_counter() - started, 1)
    return report


def print_report(report: dict) -> None:
    print(f"\nPM01 sim2sim, {report['dynamics']} physics ({report['physics_dt'] * 1000:g} ms steps)")
    if "walk" in report:
        print(f"\n{'walk segment':18s} {'command vx vy wz':>20s} {'measured vx vy wz':>22s} {'|error|':>20s}  fell  torque limit")
        for row in report["walk"]["segments"]:
            print(f"{row['segment']:18s} {' '.join(f'{v:+.2f}' for v in row['command']):>20s} "
                  f"{' '.join(f'{v:+.2f}' for v in row['measured_mean']):>22s} "
                  f"{' '.join(f'{v:.2f}' for v in row['abs_error_mean']):>20s}  {'YES ' if row['fell'] else 'no  '}  "
                  f"{100 * row['torque_limit_steps']:5.1f} %")
    if "recovery" in report:
        r = report["recovery"]
        times = [p["seconds_to_stand"] for p in r["poses"] if p["stood_up"]]
        print(f"\nget up from lying ({r.get('pose_source', 'unspecified')}): {r['success']}/{r['total']} poses"
              + (f", {np.mean(times):.2f} s on average (max {max(times):.2f} s)" if times else ""))
        tempo = r.get("tempo_until_standing")
        if tempo:
            print(f"  tempo until standing: base rises at up to {tempo['peak_base_rise_m_s_mean']:.2f} m/s on average "
                  f"(max {tempo['peak_base_rise_m_s_max']:.2f}); joint speed p95 {tempo['joint_speed_p95_rad_s']:.1f}, "
                  f"max {tempo['joint_speed_max_rad_s']:.1f} rad/s")
        for p in r["poses"]:
            print(f"  pose {p['pose']}: " + (f"stood up after {p['seconds_to_stand']:.2f} s, peak rise "
                                              f"{p['peak_base_rise_m_s']:.2f} m/s, peak joint speed "
                                              f"{p['peak_joint_speed_rad_s']:.1f} rad/s" if p["stood_up"] else "did NOT stand up"))
    if "push" in report:
        p = report["push"]
        print(f"\npushes while walking at 0.5 m/s: {p['pushes']} pushes, {p['falls']} falls, "
              f"stood up again {p['stood_up_again']} times, upright at the end: {p['upright_at_end']}")
    print(f"\n({report['wall_seconds']} s wall time)")


# ----------------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dynamics", choices=("sdk", "training"), default="sdk")
    parser.add_argument("--policy-dir", type=Path, default=HERE / "policy",
                        help="exported policy: policy (SuperMimic + AMP v1, default), policy_wamp_v2 "
                             "(SuperMimic + AMP v2), policy_wamp_v3 (50,400), policy_wamp_v3_70400 (70,400), or "
                             "policy_ppo_amp (PPO + AMP)")
    parser.add_argument("--recovery-poses", choices=("auto", "reference", "generic"), default="auto",
                        help="auto uses the exported or local reference poses if available, otherwise simulated falls")
    parser.add_argument("--start", choices=("stand", "fallen"), default="stand")
    parser.add_argument("--command", type=float, nargs=3, default=(0.0, 0.0, 0.0), metavar=("VX", "VY", "WZ"))
    parser.add_argument("--headless", action="store_true", help="run the scripted test instead of the viewer")
    parser.add_argument("--scenario", nargs="+", choices=("walk", "recovery", "push"),
                        default=["walk", "recovery", "push"])
    parser.add_argument("--record", type=Path, help="headless: write an MP4 of the run")
    parser.add_argument("--json", type=Path, help="headless: write the report as JSON")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.headless:
        os.environ.setdefault("MUJOCO_GL", "egl")

    controller = Controller(Policy(args.policy_dir), args.dynamics, seed=args.seed, recovery_poses=args.recovery_poses)
    controller.set_command(*args.command)
    if args.headless:
        report = run_headless(controller, args.scenario, args.record)
        print_report(report)
        if args.json:
            args.json.write_text(json.dumps(report, indent=2) + "\n")
        return 0
    try:
        run_interactive(controller, args.start)
    except KeyboardInterrupt:
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
