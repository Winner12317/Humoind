"""MuJoCo sim-to-sim for perceptive Hiking policies running in unitree_rl_lab's g1_ctrl.

Based on unitree_mujoco's simulate_python (same DDS topics, LowState/LowCmd
handling, elastic band and joystick encoding), with three additions:

* Hiking training terrains (``terrains/*.npz``, exported by
  G1_Perceptive_WMAC/scripts/export_sim2sim_terrains.py) as heightfields.
* The policy's depth camera: the exact training ray directions and mount (from
  the policy's ``params/depth_camera.npz``), image-plane depth in metres,
  published as an organized 64x36 PointCloud2 on ``rt/hiking/depth`` at 50 Hz.
  The robot's own torso/head geometry is excluded, which reproduces training's
  rule that geometry closer than 0.1 m along a ray is ignored.
* A scripted remote (``--auto``) that stands the robot up, starts the policy,
  walks it forward and records falls/progress to a JSON file.

The PD law tau = tau_ff + kp (q* - q) + kd (dq* - dq) is applied at every physics
step with the latest LowCmd, as the motor drivers do (unitree_mujoco's Python
bridge only recomputes it when a command arrives).

    # terminal 1
    python hiking_mujoco.py --terrain stairs_up_0.3 --viewer
    # terminal 2
    cd ../build && ./g1_ctrl --network lo
    # remote: [L2 + right] stand, [R1 + Y] start the policy, left stick forward to walk
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import struct
import threading
import time

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_UNITREE_MUJOCO = HERE.parents[4] / "unitree_mujoco"
DEFAULT_POLICY_ROOT = HERE.parent / "config/policy/hiking"
DT = 0.005
CONTROL_DECIMATION = 4
# Hiking default pose (Unitree joint convention; waist zero).
DEFAULT_POSE = np.array([-.312, 0, 0, .669, -.363, 0, -.312, 0, 0, .669, -.363, 0, 0, 0, 0,
                         .2, .2, 0, .6, 0, 0, 0, .2, -.2, 0, .6, 0, 0, 0])
KEY_BITS = {"R1": 0, "L1": 1, "start": 2, "select": 3, "R2": 4, "L2": 5, "F1": 6, "F2": 7,
            "A": 8, "B": 9, "X": 10, "Y": 11, "up": 12, "right": 13, "down": 14, "left": 15}


def latest_policy_dir(root: Path) -> Path:
    if (root / "params/depth_camera.npz").exists():
        return root
    candidates = sorted(p for p in root.iterdir() if (p / "params/depth_camera.npz").exists())
    if not candidates:
        raise SystemExit(f"No exported Hiking policy with params/depth_camera.npz under {root}")
    return candidates[-1]


def build_model(unitree_mujoco: Path, terrain: str, feet: str = "unitree"):
    spec = mujoco.MjSpec.from_file(str(unitree_mujoco / "unitree_robots/g1/scene_29dof.xml"))
    spec.option.timestep = DT
    if feet == "training":
        # Replace Unitree's four 5 mm foot spheres with the training robot's seven sole
        # capsules (Hiking URDF, same ankle_roll_link frame); see training_feet.json.
        shapes = json.loads((HERE / "training_feet.json").read_text())
        for side in ("left", "right"):
            body = spec.body(f"{side}_ankle_roll_link")
            for geom in list(body.geoms):
                if geom.type == mujoco.mjtGeom.mjGEOM_SPHERE:
                    spec.delete(geom)
            for i, shape in enumerate(shapes[side]):
                body.add_geom(name=f"{side}_foot_training_{i}", type=mujoco.mjtGeom(shape["type"]),
                              pos=shape["pos"], quat=shape["quat"], size=shape["size"], rgba=[.2, .2, .2, 1])
    elif feet != "unitree":
        raise ValueError("feet must be unitree or training")
    origin = np.zeros(3)
    target = np.array([3.7, 0., 0.])
    heights = None
    if terrain != "flat":
        path = Path(terrain)
        if not path.exists():
            path = HERE / "terrains" / f"{terrain}.npz"
        tile = np.load(path)
        heights = tile["heights"].astype(np.float64)
        size, tile_origin = tile["size"], tile["origin"]
        low, high = float(heights.min()), float(heights.max())
        span = max(high - low, .005)
        hfield = spec.add_hfield(name="terrain", size=[size[0] / 2, size[1] / 2, span, .1],
                                 nrow=heights.shape[0], ncol=heights.shape[1],
                                 userdata=((heights - low) / span).flatten().tolist())
        # Put the spawn origin (tile centre) at the world origin.
        spec.worldbody.add_geom(name="terrain", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname=hfield.name,
                                pos=[size[0] / 2 - tile_origin[0], size[1] / 2 - tile_origin[1], low],
                                rgba=[.45, .5, .48, 1.])
        spec.geom("floor").pos = [0, 0, low - .05]   # safety floor below the tile
        origin = np.array([0., 0., float(tile_origin[2])])
        target = np.array([tile["target"][0] - tile_origin[0], tile["target"][1] - tile_origin[1],
                           float(tile["target"][2])])
        heights = (heights, size, tile_origin, low)
    model = spec.compile()
    return model, origin, target, heights


def _geom_bottom(model, data, geom):
    """Lowest world z of a sphere or capsule geom."""
    radius = model.geom_size[geom][0]
    if model.geom_type[geom] == mujoco.mjtGeom.mjGEOM_CAPSULE:
        axis = data.geom_xmat[geom].reshape(3, 3)[:, 2] * model.geom_size[geom][1]
        return min(data.geom_xpos[geom][2] + axis[2], data.geom_xpos[geom][2] - axis[2]) - radius
    return data.geom_xpos[geom][2] - radius


class TerrainHeight:
    def __init__(self, heights):
        self.heights = heights

    def __call__(self, x, y):
        if self.heights is None:
            return 0.
        array, size, origin, _ = self.heights
        column = int(np.clip(round((x + origin[0]) / .05), 0, array.shape[1] - 1))
        row = int(np.clip(round((y + origin[1]) / .05), 0, array.shape[0] - 1))
        return float(array[row, column])


class DepthCamera:
    def __init__(self, model, policy_dir: Path):
        camera = np.load(policy_dir / "params/depth_camera.npz")
        self.width, self.height = int(camera["width"]), int(camera["height"])
        quat = camera["mount_quat_wxyz"] / np.linalg.norm(camera["mount_quat_wxyz"])
        mount = np.zeros(9)
        mujoco.mju_quat2Mat(mount, quat)
        self.directions = camera["directions_camera"] @ mount.reshape(3, 3).T   # torso_link frame
        self.position = camera["mount_position_unitree_torso"]
        self.cosine = camera["image_plane_cosine"]
        self.body = model.body(str(camera["camera_body"])).id
        self.groups = np.array([1, 1, 0, 0, 0, 0], np.uint8)       # terrain/collision (0) + visual meshes (1)
        self.geomid = np.zeros(len(self.directions), np.int32)
        self.distance = np.zeros(len(self.directions))

    def render(self, model, data):
        rotation = data.xmat[self.body].reshape(3, 3)
        origin = data.xpos[self.body] + rotation @ self.position
        vectors = (self.directions @ rotation.T).reshape(-1)
        mujoco.mj_multiRay(model, data, origin, vectors, self.groups, 1, self.body,
                           self.geomid, self.distance, len(self.directions), mujoco.mjMAXVAL)
        depth = np.where(self.distance >= 0, self.distance * self.cosine, np.inf)
        return depth.astype(np.float32).reshape(self.height, self.width)


class ElasticBand:
    """unitree_mujoco's virtual band, attached to torso_link."""

    def __init__(self, point, length, stiffness=1000., damping=200.):
        self.point, self.length = np.asarray(point, float), length
        self.stiffness, self.damping, self.enable = stiffness, damping, True

    def force(self, position, velocity):
        delta = self.point - position
        distance = np.linalg.norm(delta)
        direction = delta / distance
        return (self.stiffness * (distance - self.length) - self.damping * velocity @ direction) * direction


class Remote:
    """Wireless-remote state: timed button combos (hold + press) and sticks."""

    COMBOS = {"stand": ("L2", "right"), "policy": ("R1", "Y"), "passive": ("L2", "B")}

    def __init__(self):
        self.events, self.vx, self.yaw, self.lock = [], 0., 0., threading.Lock()

    def press(self, t, combo):
        """Hold the first key for 0.8 s and tap the second in the middle (FSM ``A + B.on_pressed``)."""
        hold, tap = self.COMBOS[combo]
        with self.lock:
            self.events += [(t, t + .8, (hold,)), (t + .4, t + .6, (tap,))]

    def state(self, t):
        buttons = 0
        with self.lock:
            for start, end, keys in self.events:
                if start <= t < end:
                    for key in keys:
                        buttons |= 1 << KEY_BITS[key]
            return buttons, 0., -self.yaw, 0., self.vx   # velocity_commands: vx = ly, yaw = -rx


class ScriptedRemote(Remote):
    """--auto: stand, start the policy, stand 1 s, then walk."""

    def __init__(self, vx, yaw, stand_at=1., policy_at=5., walk_at=6.5):
        super().__init__()
        self.press(stand_at, "stand")
        self.press(policy_at, "policy")
        self.policy_at, self.walk_at, self.walk_vx, self.walk_yaw = policy_at + .5, walk_at, vx, yaw

    def state(self, t):
        self.vx, self.yaw = (self.walk_vx, self.walk_yaw) if t >= self.walk_at else (0., 0.)
        return super().state(t)


class KeyboardRemote(Remote):
    """Viewer keys: 1 stand, 2 start policy, 0 passive, arrows vx/yaw, space stop, 9 band."""

    def __init__(self, band, clock):
        super().__init__()
        self.band, self.clock = band, clock

    def __call__(self, key):
        t = self.clock()
        if key == ord("1"):
            self.press(t, "stand")
        elif key == ord("2"):
            self.press(t, "policy")
        elif key == ord("0"):
            self.press(t, "passive")
        elif key == ord("9"):
            self.band.enable = not self.band.enable
        elif key == 32:                                   # space
            self.vx = self.yaw = 0.
        elif key in (265, 264):                           # up / down
            self.vx = float(np.clip(self.vx + (.1 if key == 265 else -.1), 0., 1.))
        elif key in (263, 262):                           # left / right
            self.yaw = float(np.clip(self.yaw + (.2 if key == 263 else -.2), -1., 1.))
        print(f"[remote] vx {self.vx:.1f} m/s, yaw {self.yaw:+.1f} rad/s, band {'on' if self.band.enable else 'off'}",
              flush=True)


def remote_bytes(buttons, lx, rx, ry, ly):
    raw = bytearray(40)
    raw[2:4] = struct.pack("<H", buttons)
    raw[4:8], raw[8:12], raw[12:16], raw[20:24] = (struct.pack("<f", v) for v in (lx, rx, ry, ly))
    return list(raw)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--terrain", default="flat", help="flat, a name in terrains/, or an .npz path")
    parser.add_argument("--unitree-mujoco", type=Path, default=DEFAULT_UNITREE_MUJOCO)
    parser.add_argument("--policy-dir", type=Path, default=DEFAULT_POLICY_ROOT)
    parser.add_argument("--interface", default="lo")
    parser.add_argument("--domain", type=int, default=0)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--auto", action="store_true", help="scripted stand -> policy -> walk test")
    parser.add_argument("--vx", type=float, default=.6, help="--auto forward command (m/s)")
    parser.add_argument("--yaw", type=float, default=0., help="--auto yaw-rate command (rad/s)")
    parser.add_argument("--walk-seconds", type=float, default=12.)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--feet", choices=("unitree", "training"), default="unitree",
                        help="foot collision: Unitree's 4 spheres, or the training robot's 7 capsules")
    parser.add_argument("--seed", type=int, help="random spawn offset as in Hiking resets: xy and yaw within +-0.1")
    parser.add_argument("--depth-mode", choices=("normal", "blind"), default="normal",
                        help="blind publishes no depth returns (ablation)")
    args = parser.parse_args()

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    from unitree_sdk2py.idl.default import (unitree_hg_msg_dds__LowState_, std_msgs_msg_dds__Header_)
    from unitree_sdk2py.idl.sensor_msgs.msg.dds_ import PointCloud2_, PointField_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_

    policy_dir = latest_policy_dir(args.policy_dir)
    model, origin, target, heights = build_model(args.unitree_mujoco, args.terrain, args.feet)
    data = mujoco.MjData(model)
    terrain_height = TerrainHeight(heights)
    nu = model.nu
    sensor = {model.sensor(i).name: (model.sensor_adr[i], model.sensor_dim[i]) for i in range(model.nsensor)}

    def reading(name):
        start, dim = sensor[name]
        return data.sensordata[start:start + dim]

    # Spawn standing in the Hiking default pose with the feet on the terrain.
    data.qpos[7:7 + nu] = DEFAULT_POSE
    data.qpos[3:7] = [1, 0, 0, 0]
    mujoco.mj_kinematics(model, data)
    feet = [g for g in range(model.ngeom) if model.body(model.geom_bodyid[g]).name.endswith("ankle_roll_link")
            and model.geom_contype[g]]
    lowest = min(_geom_bottom(model, data, g) for g in feet)
    data.qpos[:2] = origin[:2]
    data.qpos[2] += origin[2] - lowest + .002    # lowest foot point 2 mm above the spawn ground
    if args.seed is not None:
        rng = np.random.default_rng(args.seed)
        data.qpos[:2] += rng.uniform(-.1, .1, 2)
        yaw = rng.uniform(-.1, .1)
        data.qpos[3:7] = [np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
    mujoco.mj_forward(model, data)
    torso = model.body("torso_link").id
    band = ElasticBand(data.xpos[torso] + [0, 0, 1.], 1. - .5 * model.body_subtreemass[1] * 9.81 / 1000.)
    camera = DepthCamera(model, policy_dir)
    # The Hiking state is recognisable by its PD gains (FixStand/Passive use others).
    policy_kp = np.array(json.loads((policy_dir / "params/deployment_contract.json").read_text())["joints"]["kp"])
    print(f"[hiking_mujoco] terrain={args.terrain} policy={policy_dir} spawn z={data.qpos[2]:.3f} "
          f"target=({target[0]:.2f}, {target[1]:.2f}, {target[2]:.2f})", flush=True)

    ChannelFactoryInitialize(args.domain, args.interface)
    state_pub = ChannelPublisher("rt/lowstate", LowState_); state_pub.Init()
    depth_pub = ChannelPublisher("rt/hiking/depth", PointCloud2_); depth_pub.Init()
    command = {"q": np.zeros(nu), "dq": np.zeros(nu), "kp": np.zeros(nu), "kd": np.zeros(nu),
               "tau": np.zeros(nu), "received": False}
    lock = threading.Lock()

    def on_command(msg):
        with lock:
            for i in range(nu):
                motor = msg.motor_cmd[i]
                command["q"][i], command["dq"][i] = motor.q, motor.dq
                command["kp"][i], command["kd"][i], command["tau"][i] = motor.kp, motor.kd, motor.tau
            command["received"] = True

    cmd_sub = ChannelSubscriber("rt/lowcmd", LowCmd_); cmd_sub.Init(on_command, 10)
    low_state = unitree_hg_msg_dds__LowState_()
    cloud = PointCloud2_(header=std_msgs_msg_dds__Header_(), height=camera.height, width=camera.width,
                         fields=[PointField_(name="z", offset=0, datatype=7, count=1)], is_bigendian=False,
                         point_step=4, row_step=4 * camera.width, data=[], is_dense=False)

    viewer = None
    step = 0
    remote = ScriptedRemote(args.vx, args.yaw) if args.auto else KeyboardRemote(band, lambda: step * DT)
    if args.viewer:
        from mujoco import viewer as mujoco_viewer
        viewer = mujoco_viewer.launch_passive(model, data, key_callback=None if args.auto else remote)
        if not args.auto:
            print("[hiking_mujoco] keys: 1 stand, 2 start policy, 0 passive, up/down vx, left/right yaw, "
                  "space stop, 9 elastic band", flush=True)
    log, events, fell, reached = [], {}, None, None
    started = schedule = time.perf_counter()
    connected = None          # the remote timeline starts once g1_ctrl publishes LowCmd
    auto = args.auto
    try:
        while (viewer is None or viewer.is_running()):
            t = step * DT
            if connected is None and command["received"]:
                connected = t
                events["controller_connected"] = round(t, 3)
            if auto and connected is not None and t - connected >= remote.walk_at + args.walk_seconds:
                break
            if auto and connected is None and t > 60.:
                raise SystemExit("g1_ctrl did not connect within 60 s (start it with --network lo)")
            with lock:
                joint_q = data.qpos[7:7 + nu]
                joint_dq = data.qvel[6:6 + nu]
                data.ctrl[:] = (command["tau"] + command["kp"] * (command["q"] - joint_q)
                                + command["kd"] * (command["dq"] - joint_dq))
                policy_on = bool(np.allclose(command["kp"], policy_kp, atol=1e-3))
            remote_t = (t - connected if connected is not None else -1.) if auto else t
            if band.enable and "policy_started" in events and "band_released" not in events:
                band.enable = False              # release once g1_ctrl runs the Hiking policy
                events["band_released"] = round(t, 3)
            data.xfrc_applied[torso, :3] = band.force(data.xpos[torso], data.cvel[torso, 3:]) if band.enable else 0.
            mujoco.mj_step(model, data)
            step += 1
            # LowState (pelvis IMU, joints, remote) every physics step.
            for i in range(nu):
                low_state.motor_state[i].q = float(data.qpos[7 + i])
                low_state.motor_state[i].dq = float(data.qvel[6 + i])
                low_state.motor_state[i].tau_est = float(data.actuator_force[i])
            low_state.imu_state.quaternion = [float(v) for v in reading("imu_quat")]
            low_state.imu_state.gyroscope = [float(v) for v in reading("imu_gyro")]
            low_state.imu_state.accelerometer = [float(v) for v in reading("imu_acc")]
            low_state.wireless_remote = remote_bytes(*remote.state(remote_t))
            low_state.tick = step
            state_pub.Write(low_state)
            if step % CONTROL_DECIMATION == 0:
                depth = camera.render(model, data)
                if args.depth_mode == "blind":
                    depth[:] = np.inf
                cloud.data = list(depth.tobytes())
                cloud.header.stamp.sec, cloud.header.stamp.nanosec = int(t), int((t % 1) * 1e9)
                depth_pub.Write(cloud)
                pelvis = data.qpos[:3]
                up_z = data.xmat[1].reshape(3, 3)[2, 2]
                ground = terrain_height(pelvis[0], pelvis[1])
                log.append([round(t, 3), *map(float, pelvis), ground, float(up_z), float(policy_on)])
                if policy_on and "policy_started" not in events:
                    events["policy_started"] = round(t, 3)
                if fell is None and "policy_started" in events and (pelvis[2] - ground < .45 or up_z < .5):
                    fell = round(t, 3)
                if reached is None and fell is None and pelvis[0] >= target[0] - .3:
                    reached = round(t, 3)
                if viewer is not None:
                    viewer.sync()
            if auto and fell is not None and t > fell + 1.:
                break
            if auto and reached is not None:
                break   # at the Hiking target; the terrain tile ends 0.6 m further
            schedule += DT
            delay = schedule - time.perf_counter()
            if delay > 0:
                time.sleep(delay)
            elif delay < -.1:
                schedule = time.perf_counter()   # after a long stall, do not burst to catch up
    finally:
        wall = time.perf_counter() - started
        if viewer is not None:
            viewer.close()
    summary = {"terrain": args.terrain, "seed": args.seed, "depth_mode": args.depth_mode, "feet": args.feet,
               "policy_dir": str(policy_dir), "vx": args.vx, "yaw": args.yaw,
               "target": target.tolist(), "events": events, "fell_at": fell, "reached_target_at": reached,
               "final_pelvis": log[-1][1:4] if log else None, "sim_seconds": step * DT,
               "real_time_factor": step * DT / wall if wall > 0 else None,
               "command_received": command["received"]}
    if log and "policy_started" in events:
        start = next(i for i, row in enumerate(log) if row[0] >= events["policy_started"])
        summary["progress_x_m"] = log[-1][1] - log[start][1]
    print(json.dumps(summary), flush=True)
    if args.results:
        args.results.parent.mkdir(parents=True, exist_ok=True)
        args.results.write_text(json.dumps({**summary, "log_columns": ["t", "x", "y", "z", "ground", "up_z",
                                                                        "policy_gains"], "log": log}) + "\n")


if __name__ == "__main__":
    main()
