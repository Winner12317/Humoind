"""Which PM01 collision primitives intersect, in neutral poses and in the reference motions.

    python scripts/pm01/audit_pm01_collisions.py

Uses MuJoCo's own collision pass on ``pm01.xml`` with its ``<exclude>`` pairs
removed, so the pairs listed are the ones the simulator would act on (parent and
child bodies are never checked against each other).  The result is the basis for
the four excludes written by ``build_pm01_mjcf.py``.
"""

from __future__ import annotations

import collections
import re
import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

from src.assets.robots.engineai_pm01 import pm01_constants as pm01  # noqa: E402

MOTIONS = REPO / "src" / "assets" / "motions" / "pm01" / "amp"


def main() -> int:
    spec = mujoco.MjSpec.from_file(str(pm01.PM01_XML))
    excluded = [(pair.bodyname1, pair.bodyname2) for pair in spec.excludes]
    for pair in list(spec.excludes):
        spec.delete(pair)
    model = spec.compile()
    data = mujoco.MjData(model)

    def deepest(qpos: np.ndarray) -> dict[tuple[str, str], float]:
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        pairs: dict[tuple[str, str], float] = {}
        for contact in data.contact[: data.ncon]:
            bodies = tuple(sorted(model.body(model.geom_bodyid[g]).name for g in (contact.geom1, contact.geom2)))
            pairs[bodies] = min(pairs.get(bodies, 0.0), contact.dist)
        return pairs

    zero = model.qpos0.copy()
    zero[2] = 2.0
    default = zero.copy()
    for joint in range(1, model.njnt):
        for pattern, value in pm01.HOME_KEYFRAME.joint_pos.items():
            if re.fullmatch(pattern, model.joint(joint).name):
                default[model.jnt_qposadr[joint]] = value
    for label, qpos in (("all joints at zero", zero), ("default pose", default)):
        print(f"{label}:")
        pairs = deepest(qpos)
        for bodies, depth in sorted(pairs.items(), key=lambda item: item[1]):
            print(f"  {bodies[0]:20s} {bodies[1]:20s} {-100 * depth:5.2f} cm")
        if not pairs:
            print("  no intersecting pair")

    counts: dict[tuple[str, str], list] = collections.defaultdict(lambda: [0, 0.0])
    frames = 0
    for path in sorted(MOTIONS.glob("*/*.npz")):
        with np.load(path) as motion:
            root_pos, root_quat, joint_pos = motion["body_pos_w"][:, 0], motion["body_quat_w"][:, 0], motion["joint_pos"]
        for frame in range(len(joint_pos)):
            frames += 1
            for bodies, depth in deepest(np.concatenate((root_pos[frame], root_quat[frame], joint_pos[frame]))).items():
                if "world" in bodies:
                    continue
                entry = counts[bodies]
                entry[0] += depth < -0.01
                entry[1] = min(entry[1], depth)
    print(f"reference motions ({frames} frames): share of frames deeper than 1 cm, deepest")
    for bodies, (count, depth) in sorted(counts.items(), key=lambda item: -item[1][0]):
        if count:
            mark = "  excluded in pm01.xml" if bodies in {tuple(sorted(pair)) for pair in excluded} else ""
            print(f"  {bodies[0]:20s} {bodies[1]:20s} {100 * count / frames:6.2f} %  {-100 * depth:5.2f} cm{mark}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
