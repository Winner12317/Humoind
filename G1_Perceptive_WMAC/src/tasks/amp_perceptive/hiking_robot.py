"""Hiking's G1, unshod or shod, expressed in the pelvis-root GenCoord layout.

InstinctLab de6b92f ships two variants of the task robot. ``unshod`` is
``G1ParkourRoughEnvCfg``'s ``g1_29dof_torsobase_popsicle.urdf``. ``shoe`` is
``G1ParkourEnvCfg`` (the registered ``Instinct-Parkour-Target-Amp-G1-v0``), whose
``ShoeConfigMixin`` swaps in ``g1_29dof_torsoBase_popsicle_with_shoe.urdf`` and
lowers the foot measurements with it. The two URDFs differ only in the height
of the seven collision capsules under each foot (z -0.025 -> -0.048 m).
"""
from __future__ import annotations

import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from src.assets.robots.unitree_g1.g1_constants import G1_XML, get_spec

HIKING_URDF = G1_XML.parent.parent / "g1_29dof_torsobase_popsicle.urdf"
HIKING_SHOE_URDF = HIKING_URDF.with_name("g1_29dof_torsobase_popsicle_with_shoe.urdf")
HIKING_SHOE_URDF_SHA256 = "ef88a3094280b966624f60f7179aa59b7b2536fc282cf9fefa0c71c75a0a40fb"
HIKING_HEAD = HIKING_URDF.with_name("hiking_head_link.STL")
# Foot measurements that ShoeConfigMixin changes together with the URDF:
# z range of the 10 x 5 x 2 volume-point grid in the ankle-roll frame, and
# feet_at_plane's height offset between that frame and the sole.
HIKING_SHOE_SOLE_OFFSET = .023
HIKING_FOOT_GEOMETRY = {
    "unshod": {"urdf": HIKING_URDF, "volume_points_z": (-.04, 0.), "plane_height_offset": .035},
    "shoe": {"urdf": HIKING_SHOE_URDF, "volume_points_z": (-.063, -.023), "plane_height_offset": .058},
}


def hiking_foot_geometry(variant="unshod"):
    if variant not in HIKING_FOOT_GEOMETRY:
        raise ValueError(f"Unknown Hiking robot variant: {variant!r}")
    return HIKING_FOOT_GEOMETRY[variant]


def source_hiking_model(variant="unshod"):
    """Load pinned URDF physics and merge fixed-link inertias in MuJoCo."""
    robot = ET.parse(hiking_foot_geometry(variant)["urdf"]).getroot()
    for link in robot.findall("link"):
        for visual in link.findall("visual"):
            link.remove(visual)  # The MJCF supplies the Unitree render meshes.
    ET.SubElement(robot, "link", name="world")
    floating = ET.SubElement(robot, "joint", name="floating_base_joint", type="floating")
    ET.SubElement(floating, "parent", link="world")
    ET.SubElement(floating, "child", link="torso_link")
    return mujoco.MjModel.from_xml_string(ET.tostring(robot, encoding="unicode"))


def get_hiking_shoe_robot_spec():
    """The shod variant; a named function because the scene stores a spec callable."""
    return get_hiking_robot_spec("shoe")


def get_hiking_robot_spec(variant="unshod"):
    source = source_hiking_model(variant)
    spec = get_spec()
    # The other 34 visual mesh files and their transforms agree with the raw
    # URDF. Override the historical head mesh only on the v3 asset.
    assets = spec.assets
    assets["assets/head_link.STL"] = HIKING_HEAD.read_bytes()
    spec.assets = assets
    for exclusion in list(spec.excludes):
        spec.delete(exclusion)
    for side in ("left", "right"):
        spec.site(f"{side}_foot").pos = (0.04, 0., -0.037)  # LL_FOOT/LR_FOOT frames
    for geom in list(spec.geoms):
        if geom.name.endswith("_collision"):
            spec.delete(geom)
    for i in range(1, source.nbody):
        source_body = source.body(i)
        body = spec.body(source_body.name)
        body.mass = source.body_mass[i]
        body.ipos = source.body_ipos[i]
        body.iquat = source.body_iquat[i]
        body.inertia = source.body_inertia[i]
        first = source.body_geomadr[i]
        for k, geom_id in enumerate(range(first, first + source.body_geomnum[i]), 1):
            if source_body.name.endswith("ankle_roll_link"):
                name = f"{source_body.name.split('_')[0]}_foot{k}_collision"
            else:
                name = f"{source_body.name}_{k}_collision"
            kind = source.geom_type[geom_id]
            # InstinctLab sets replace_cylinders_with_capsules=True at import.
            if kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                kind = mujoco.mjtGeom.mjGEOM_CAPSULE
            body.add_geom(name=name, type=mujoco.mjtGeom(kind), size=source.geom_size[geom_id],
                          pos=source.geom_pos[geom_id], quat=source.geom_quat[geom_id], group=3)
    for i in range(1, source.njnt):
        joint = spec.joint(source.joint(i).name)
        child = int(source.jnt_bodyid[i])
        parent = int(source.body_parentid[child])
        offset, rotation = source.body_pos[child], source.body_quat[child]
        joint.range = source.jnt_range[i]
        if joint.parent.name == source.body(child).name:
            joint.parent.pos = offset
            joint.parent.quat = rotation
            joint.axis = source.jnt_axis[i]
            joint.pos = source.jnt_pos[i]
        else:
            # Reversing each waist hinge preserves the source anatomical q.
            # The joint centre moves into the new child frame as well.
            if (joint.parent.name != source.body(parent).name or
                    not np.allclose(rotation, (1., 0., 0., 0.)) or np.any(source.jnt_pos[i])):
                raise ValueError(f"Unexpected reversed Hiking hinge transform for {joint.name}")
            joint.parent.pos = -offset
            joint.parent.quat = rotation
            joint.axis = -source.jnt_axis[i]
            joint.pos = offset
    if variant == "shoe":
        # The model's default pose stands on the soles. Raise it by the sole
        # offset so the compiled rest state does not start inside the ground
        # (MuJoCo-Warp sizes its constraint buffers from that state).
        pelvis = spec.body("pelvis")
        pelvis.pos = (pelvis.pos[0], pelvis.pos[1], pelvis.pos[2] + HIKING_SHOE_SOLE_OFFSET)
    return spec
