"""Derive src/assets/robots/engineai_pm01/xmls/pm01.xml from EngineAI's official PM01 MJCF.

    python scripts/pm01/build_pm01_mjcf.py <engineai_robotics_native_sdk>/assets/resource/robot/pm01_edu

The official model (``xml/serial_pm01_edu.xml``, SDK commit 6065bb2) is written for
EngineAI's own torque-controlled simulator.  mjlab needs a model without actuators
(it adds its own from the actuator configs) and with named collision geoms, an IMU
and foot sites.  What this script keeps, changes and adds:

kept     body tree, body frames, masses and inertias, hinge axes and ranges, and every
         collision primitive (shape, size, pose) exactly as in the official file
removed  motors, sensors, the keyframe, textures/materials, and the per-joint
         armature / damping / frictionloss / actuatorfrcrange (the actuator config in
         pm01_constants.py is the single source for those)
changed  J23_HEAD_YAW is welded: the head stays a body, but has no joint.  EngineAI's
         own AMP locomotion task drives the 23 joints J00..J22 and leaves the head
         alone, and the G1 source motions carry no head motion
added    names for the collision geoms (``*_collision``), sites ``left_foot`` /
         ``right_foot`` at the centre of the underside of each sole box, the sensors
         mjlab's observation terms read (``imu_ang_vel``, ``imu_lin_vel``,
         ``imu_lin_acc`` on the official ``imu`` site), contact excludes for the body
         pairs whose official primitives already overlap in the neutral pose, and
         light-weight visual meshes (``--meshes``; vertex-clustered copies of the
         official OBJ files, visual only)

The script ends by compiling the result and comparing it body by body with the
official model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "src" / "assets" / "robots" / "engineai_pm01" / "xmls"
WELDED_JOINTS = ("J23_HEAD_YAW",)

# Official default class of each collision primitive -> geom name in pm01.xml.
# "{side}" is filled from the body name (_L -> left, _R -> right).
GEOM_NAMES = {
    "collision_base_lower_box": "base_collision",
    "collision_hip_left": "left_hip_pitch_collision",
    "collision_hip_right": "right_hip_pitch_collision",
    "collision_hiproll_left": "left_hip_roll_collision",
    "collision_hiproll_right": "right_hip_roll_collision",
    "collision_hip_yaw_left": "left_thigh_collision",
    "collision_hip_yaw_right": "right_thigh_collision",
    "collision_knee": "{side}_knee_collision",
    "collision_knee1": "{side}_shin_collision",
    "collision_left_foot": "left_foot_sole_collision",
    "collision_right_foot": "right_foot_sole_collision",
    "collision_left_foot_toe": "left_foot_toe_collision",
    "collision_right_foot_toe": "right_foot_toe_collision",
    "torso_yaw": "torso_lower_collision",
    "torso_yaw1": "torso_upper_collision",
    "shoulder_roll": "{side}_shoulder_collision",
    "shoulder_roll1_left": "left_shoulder_lower_collision",
    "shoulder_roll1_right": "right_shoulder_lower_collision",
    "shoulder_yaw": "{side}_upper_arm_collision",
    "elbow_pitch_left": "left_elbow_pitch_collision",
    "elbow_pitch_right": "right_elbow_pitch_collision",
    "collision_elbow": "{side}_elbow_collision",
    "elbow_yaw_left": "left_forearm_collision",
    "elbow_yaw_right": "right_forearm_collision",
    "collision_left_elbow_end": "left_hand_collision",
    "collision_right_elbow_end": "right_hand_collision",
    "collision_head": "head_collision",
}

# Body pairs whose official primitives intersect with every joint at zero: the hand
# sphere sits 2.37 cm inside the thigh cylinder and 1.01 cm inside the hip-roll
# cylinder on its own side.  Left enabled, the pairs would push the hanging arms
# outward for good (and the retargeted arm swing passes through them in about 10 %
# of the reference frames).  No other pair that MuJoCo checks overlaps at zero or in
# the default pose; scripts/pm01/audit_pm01_collisions.py reproduces the numbers.
CONTACT_EXCLUDES = (
    ("LINK_ELBOW_END_L", "LINK_HIP_YAW_L"),
    ("LINK_ELBOW_END_L", "LINK_HIP_ROLL_L"),
    ("LINK_ELBOW_END_R", "LINK_HIP_YAW_R"),
    ("LINK_ELBOW_END_R", "LINK_HIP_ROLL_R"),
)

GEOM_TYPES = {
    mujoco.mjtGeom.mjGEOM_SPHERE: ("sphere", 1),
    mujoco.mjtGeom.mjGEOM_CAPSULE: ("capsule", 2),
    mujoco.mjtGeom.mjGEOM_CYLINDER: ("cylinder", 2),
    mujoco.mjtGeom.mjGEOM_BOX: ("box", 3),
}


def num(values) -> str:
    return " ".join(f"{float(v):.10g}" for v in np.atleast_1d(values))


def side_of(body_name: str) -> str:
    return {"_L": "left", "_R": "right"}.get(body_name[-2:], "")


def decimate(vertices: np.ndarray, faces: np.ndarray, max_faces: int) -> tuple[np.ndarray, np.ndarray]:
    """Vertex clustering on the coarsest grid that keeps at most ``max_faces`` triangles."""

    def cluster(cell: float) -> tuple[np.ndarray, np.ndarray]:
        keys = np.floor((vertices - vertices.min(axis=0)) / cell).astype(np.int64)
        _, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
        inverse = inverse.reshape(-1)
        merged = np.zeros((counts.size, 3))
        np.add.at(merged, inverse, vertices)
        merged /= counts[:, None]
        tri = inverse[faces]
        tri = tri[(tri[:, 0] != tri[:, 1]) & (tri[:, 1] != tri[:, 2]) & (tri[:, 0] != tri[:, 2])]
        tri = tri[np.unique(np.sort(tri, axis=1), axis=0, return_index=True)[1]]
        return merged, tri

    if len(faces) <= max_faces:
        return vertices, faces
    low, high = 1e-4, float(np.ptp(vertices, axis=0).max())
    for _ in range(30):
        mid = 0.5 * (low + high)
        if len(cluster(mid)[1]) > max_faces:
            low = mid
        else:
            high = mid
    return cluster(high)


def write_binary_stl(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    tri = vertices[faces].astype(np.float32)
    normal = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1e-20)
    record = np.zeros(len(tri), dtype=[("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
    record["n"], record["v"] = normal, tri
    with path.open("wb") as stream:
        stream.write(b"PM01 visual mesh, vertex-clustered from the official OBJ".ljust(80, b" "))
        stream.write(np.uint32(len(tri)).tobytes())
        stream.write(record.tobytes())


def build_meshes(source: Path, names: list[str], max_faces: int) -> None:
    import trimesh

    target = OUT_DIR / "assets"
    target.mkdir(parents=True, exist_ok=True)
    for name in names:
        mesh = trimesh.load(source / "meshes" / f"{name}.obj", force="mesh", process=False)
        vertices, faces = decimate(np.asarray(mesh.vertices, dtype=np.float64), np.asarray(mesh.faces), max_faces)
        write_binary_stl(target / f"{name}.stl", vertices, faces)
        print(f"  {name}: {len(mesh.faces)} -> {len(faces)} triangles")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("pm01_edu", type=Path, help="…/assets/resource/robot/pm01_edu of the EngineAI SDK")
    parser.add_argument("--meshes", action="store_true", help="Also rebuild the visual meshes in xmls/assets/")
    parser.add_argument("--max-faces", type=int, default=8000, help="Triangle budget per visual mesh")
    args = parser.parse_args()

    source_xml = args.pm01_edu / "xml" / "serial_pm01_edu.xml"
    spec = mujoco.MjSpec.from_file(str(source_xml))
    model = spec.compile()
    classes = [geom.classname.name for geom in spec.geoms]
    assert len(classes) == model.ngeom

    mesh_names: list[str] = []
    lines: list[str] = []

    def emit_body(body_id: int, depth: int) -> None:
        pad = "  " * depth
        name = model.body(body_id).name
        attrs = f'name="{name}" pos="{num(model.body_pos[body_id])}"'
        if not np.allclose(model.body_quat[body_id], (1, 0, 0, 0)):
            attrs += f' quat="{num(model.body_quat[body_id])}"'
        if body_id == 1:
            attrs += ' childclass="pm01"'
        lines.append(f"{pad}<body {attrs}>")
        inner = pad + "  "
        if body_id == 1:
            lines.append(f'{inner}<light pos="0 0 2" mode="trackcom"/>')
            lines.append(
                f'{inner}<camera name="tracking" pos="1.734 -1.135 .35" '
                'xyaxes="0.552 0.834 -0.000 -0.170 0.112 0.979" mode="trackcom"/>'
            )
        lines.append(
            f'{inner}<inertial pos="{num(model.body_ipos[body_id])}" quat="{num(model.body_iquat[body_id])}" '
            f'mass="{num(model.body_mass[body_id])}" diaginertia="{num(model.body_inertia[body_id])}"/>'
        )
        for joint_id in range(model.body_jntadr[body_id], model.body_jntadr[body_id] + model.body_jntnum[body_id]):
            joint = model.joint(joint_id)
            if model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_FREE:
                lines.append(f'{inner}<freejoint name="floating_base_joint"/>')
            elif joint.name in WELDED_JOINTS:
                lines.append(f"{inner}<!-- {joint.name} is welded: see scripts/pm01/build_pm01_mjcf.py -->")
            else:
                assert model.jnt_type[joint_id] == mujoco.mjtJoint.mjJNT_HINGE and not model.jnt_pos[joint_id].any()
                lines.append(
                    f'{inner}<joint name="{joint.name}" axis="{num(model.jnt_axis[joint_id])}" '
                    f'range="{num(model.jnt_range[joint_id])}"/>'
                )
        for geom_id in np.flatnonzero(model.geom_bodyid == body_id):
            if model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = model.mesh(model.geom_dataid[geom_id]).name
                mesh_names.append(mesh)
                lines.append(f'{inner}<geom class="visual" mesh="{mesh}"/>')
                continue
            kind, size_count = GEOM_TYPES[int(model.geom_type[geom_id])]
            geom_name = GEOM_NAMES[classes[geom_id]].format(side=side_of(name))
            attrs = f'name="{geom_name}" class="collision" type="{kind}" size="{num(model.geom_size[geom_id][:size_count])}"'
            if model.geom_pos[geom_id].any():
                attrs += f' pos="{num(model.geom_pos[geom_id])}"'
            if not np.allclose(model.geom_quat[geom_id], (1, 0, 0, 0)):
                attrs += f' quat="{num(model.geom_quat[geom_id])}"'
            lines.append(f"{inner}<geom {attrs}/>")
            if geom_name.endswith("_foot_sole_collision"):
                # Underside centre of the sole box, in the frame of the ankle-roll link
                # (LINK_FOOT_* has the same frame; the site goes on its parent so that
                # the foot's reference point belongs to a jointed body).
                sole = model.geom_pos[geom_id] - np.array([0.0, 0.0, model.geom_size[geom_id][2]])
                foot_sites[name] = sole
        for site_id in np.flatnonzero(model.site_bodyid == body_id):
            site = model.site(site_id).name
            if site == "imu":
                lines.append(f'{inner}<site name="imu" size="0.01" pos="{num(model.site_pos[site_id])}"/>')
        for child in range(1, model.nbody):
            if model.body_parentid[child] == body_id and child != body_id:
                emit_body(child, depth + 1)
        if name.startswith("LINK_ANKLE_ROLL_"):
            foot = f"LINK_FOOT_{name[-1]}"
            lines.append(f'{inner}<site name="{side_of(name)}_foot" pos="{num(foot_sites[foot])}"/>')
        lines.append(f"{pad}</body>")

    foot_sites: dict[str, np.ndarray] = {}
    emit_body(1, 2)
    body_lines = lines

    mesh_names = list(dict.fromkeys(mesh_names))
    out = [
        "<!-- EngineAI PM01 for mjlab. Generated by scripts/pm01/build_pm01_mjcf.py from",
        "     engineai_robotics_native_sdk assets/resource/robot/pm01_edu/xml/serial_pm01_edu.xml",
        "     (commit 6065bb2). Edit the script, not this file. Licence: see LICENSE next to it. -->",
        '<mujoco model="engineai_pm01_23dof">',
        '  <compiler angle="radian" meshdir="assets" autolimits="true"/>',
        "",
        "  <default>",
        '    <default class="pm01">',
        '      <default class="visual">',
        '        <geom type="mesh" density="0" material="silver" contype="0" conaffinity="0" group="2"/>',
        "      </default>",
        '      <default class="collision">',
        '        <geom priority="1" condim="6" group="3"/>',
        "      </default>",
        '      <site rgba="1 0 0 1" group="5"/>',
        "    </default>",
        "  </default>",
        "",
        "  <asset>",
        '    <material name="silver" rgba="0.7 0.7 0.7 1"/>',
        *[f'    <mesh name="{name}" file="{name}.stl"/>' for name in mesh_names],
        "  </asset>",
        "",
        "  <worldbody>",
        *body_lines,
        "  </worldbody>",
        "",
        "  <contact>",
        *[f'    <exclude body1="{a}" body2="{b}"/>' for a, b in CONTACT_EXCLUDES],
        "  </contact>",
        "",
        "  <sensor>",
        '    <gyro name="imu_ang_vel" site="imu"/>',
        '    <velocimeter name="imu_lin_vel" site="imu"/>',
        '    <accelerometer name="imu_lin_acc" site="imu"/>',
        '    <subtreeangmom name="root_angmom" body="LINK_BASE"/>',
        "  </sensor>",
        "</mujoco>",
        "",
    ]
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.meshes:
        build_meshes(args.pm01_edu, mesh_names, args.max_faces)
    target = OUT_DIR / "pm01.xml"
    target.write_text("\n".join(out), encoding="utf-8")

    # Compare with the official model.
    new = mujoco.MjModel.from_xml_path(str(target))
    assert new.nbody == model.nbody and [new.body(i).name for i in range(new.nbody)] == [
        model.body(i).name for i in range(model.nbody)
    ]
    for field in ("body_parentid", "body_pos", "body_quat", "body_mass", "body_ipos", "body_inertia"):
        assert np.allclose(getattr(new, field), getattr(model, field), rtol=0, atol=1e-9), field
    assert np.allclose(np.abs(np.sum(new.body_iquat * model.body_iquat, axis=1)), 1.0, atol=1e-9)
    old_hinges = [j for j in range(model.njnt) if model.joint(j).name not in WELDED_JOINTS and model.jnt_type[j] == 3]
    new_hinges = [j for j in range(new.njnt) if new.jnt_type[j] == 3]
    assert [new.joint(j).name for j in new_hinges] == [model.joint(j).name for j in old_hinges]
    for field in ("jnt_axis", "jnt_range", "jnt_pos", "jnt_bodyid"):
        assert np.allclose(getattr(new, field)[new_hinges], getattr(model, field)[old_hinges], rtol=0, atol=1e-9), field
    old_collision = [g for g in range(model.ngeom) if model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH]
    new_collision = [g for g in range(new.ngeom) if new.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH]
    for field in ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_bodyid"):
        assert np.allclose(
            getattr(new, field)[new_collision], getattr(model, field)[old_collision], rtol=0, atol=1e-9
        ), field
    assert all(new.geom(g).name.endswith("_collision") for g in new_collision)
    assert len({new.geom(g).name for g in new_collision}) == len(new_collision)
    print(
        f"wrote {target}: {new.nbody - 1} bodies, {len(new_hinges)} hinges, {len(new_collision)} collision geoms, "
        f"{new.nmesh} visual meshes, total mass {new.body_mass.sum():.3f} kg; matches the official model"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
