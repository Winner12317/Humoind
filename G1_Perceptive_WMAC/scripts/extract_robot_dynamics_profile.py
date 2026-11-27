"""Extract a robot dynamics profile from another G1 MJCF for this task's robot.

The profile stores, for every body and joint of the training robot, the mass,
inertia and joint parameters of a *different* G1 model (for example
unitreerobotics/unitree_mujoco ``unitree_robots/g1/g1_29dof.xml``), expressed
in the training robot's own body frames. Evaluation can then run a policy in
the training simulator with the other model's dynamics, and training can
randomize between the two.

Body frames are matched in the zero joint configuration with the pelvis at the
origin. The training robot keeps Hiking's torso-rooted waist (each waist angle
has the opposite sign to Unitree's convention, and the joint pivots differ by
up to 19 mm), so inertial poses are converted through world coordinates rather
than copied.

Only numerical physical parameters are written; no meshes are copied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _zero_pose(model):
    data = mujoco.MjData(model)
    data.qpos[:] = 0.
    data.qpos[3] = 1.
    mujoco.mj_kinematics(model, data)
    return data


def _matrix_to_quat(matrix):
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, np.ascontiguousarray(matrix).reshape(9))
    return quat


def extract_profile(source_xml: Path, *, name: str, source_description: str) -> dict:
    from src.tasks.amp_perceptive.hiking_robot import get_hiking_robot_spec

    train = get_hiking_robot_spec().compile()
    other = mujoco.MjModel.from_xml_path(str(source_xml))
    train_bodies = {train.body(i).name for i in range(1, train.nbody)}
    other_bodies = {other.body(i).name for i in range(1, other.nbody)}
    if train_bodies != other_bodies:
        raise ValueError(f"Body sets differ: only training {sorted(train_bodies - other_bodies)}, "
                         f"only source {sorted(other_bodies - train_bodies)}")
    train_joints = [train.joint(i).name for i in range(train.njnt) if train.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
    other_joints = [other.joint(i).name for i in range(other.njnt) if other.jnt_type[i] != mujoco.mjtJoint.mjJNT_FREE]
    if sorted(train_joints) != sorted(other_joints):
        raise ValueError("Joint sets differ between the training robot and the source model")
    td, od = _zero_pose(train), _zero_pose(other)
    bodies = {}
    for body in sorted(train_bodies):
        t, o = train.body(body).id, other.body(body).id
        rotation = td.xmat[t].reshape(3, 3)
        inertial_position = rotation.T @ (od.xipos[o] - td.xpos[t])
        inertial_rotation = rotation.T @ od.ximat[o].reshape(3, 3)
        bodies[body] = {
            "mass": float(other.body_mass[o]),
            "ipos": [float(x) for x in inertial_position],
            "iquat": [float(x) for x in _matrix_to_quat(inertial_rotation)],
            "inertia": [float(x) for x in other.body_inertia[o]],
            "training_mass": float(train.body_mass[t]),
        }
    joints = {}
    for joint in sorted(train_joints):
        dof = int(other.jnt_dofadr[other.joint(joint).id])
        joints[joint] = {
            "armature": float(other.dof_armature[dof]),
            "damping": float(other.dof_damping[dof]),
            "frictionloss": float(other.dof_frictionloss[dof]),
        }
    source_bytes = Path(source_xml).read_bytes()
    return {
        "format": "robot-dynamics-profile-v1",
        "name": name,
        "source": source_description,
        "source_file_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "frames": "training robot body frames at zero joint angles, pelvis at origin",
        "total_mass": float(sum(b["mass"] for b in bodies.values())),
        "training_total_mass": float(sum(b["training_mass"] for b in bodies.values())),
        "bodies": bodies,
        "joints": joints,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-xml", type=Path, required=True)
    parser.add_argument("--name", default="unitree_mujoco_g1_29dof")
    parser.add_argument("--source-description",
                        default="unitreerobotics/unitree_mujoco unitree_robots/g1/g1_29dof.xml (BSD-3-Clause)")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    profile = extract_profile(args.source_xml, name=args.name, source_description=args.source_description)
    args.output.write_text(json.dumps(profile, indent=1) + "\n")
    print(f"{args.output}: total mass {profile['total_mass']:.3f} kg "
          f"(training robot {profile['training_total_mass']:.3f} kg)")


if __name__ == "__main__":
    main()
