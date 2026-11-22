"""Write models/pm01_sdk_collision.xml from EngineAI's official PM01 MJCF.

The official model (engineai_robotics_native_sdk,
``assets/resource/robot/pm01_edu/xml/serial_pm01_edu.xml``) pulls in visual
meshes, actuators and sensors from sibling files.  This keeps what the motion
checks need and nothing else: the floating-base kinematic tree, joint limits and
the collision primitives (the boxes, cylinders and spheres a simulator collides).
The result is one self-contained file with no mesh dependencies.

    python make_collision_model.py <path to serial_pm01_edu.xml>

The file in models/ was made from commit 6065bb2 of the SDK; its licence is next
to it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco

from common import PM01_JOINT_NAMES, PM01_SDK_COLLISION


def main() -> int:
    if len(sys.argv) != 2:
        print(__doc__)
        return 2
    spec = mujoco.MjSpec.from_file(sys.argv[1])
    for geom in list(spec.geoms):
        if (geom.contype == 0 and geom.conaffinity == 0) or geom.type == mujoco.mjtGeom.mjGEOM_MESH:
            spec.delete(geom)
    for group in (spec.meshes, spec.materials, spec.textures, spec.actuators, spec.sensors, spec.keys, spec.lights):
        for item in list(group):
            spec.delete(item)
    spec.compile()
    PM01_SDK_COLLISION.parent.mkdir(parents=True, exist_ok=True)
    PM01_SDK_COLLISION.write_text(spec.to_xml(), encoding="utf-8")

    model = mujoco.MjModel.from_xml_path(str(PM01_SDK_COLLISION))
    joints = [model.joint(j).name for j in range(model.njnt) if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE]
    if tuple(joints) != PM01_JOINT_NAMES or model.nmesh:
        raise SystemExit("unexpected model: joints or meshes differ from what the scripts assume")
    print(f"wrote {PM01_SDK_COLLISION}: {model.nbody} bodies, {len(joints)} joints, {model.ngeom} collision geoms")
    return 0


if __name__ == "__main__":
    sys.exit(main())
