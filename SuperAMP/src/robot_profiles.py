"""What the shared training code needs to know about each supported robot.

The AMP feature builder, the WMAC state adapter, the motion mirroring and the
evaluation scripts were written for the Unitree G1 and named its bodies inline.
A ``RobotProfile`` holds those names and the few robot-specific numbers, so the
same code runs the EngineAI PM01.  The G1 profile reproduces the previous
constants exactly.

A profile is found from a task id (``profile_for_task``) or from the body names of
a live or compiled model (``profile_for_bodies``).  No simulator is imported here.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_ROBOTS = Path(__file__).resolve().parent / "assets" / "robots"

Vector3 = tuple[float, float, float]


@dataclass(frozen=True)
class RobotProfile:
    name: str
    xml_path: Path
    num_joints: int
    num_bodies: int  # without the world body
    root_body: str
    torso_body: str  # AMP anchor; also the body whose velocity follows the command
    foot_bodies: tuple[str, str]  # left, right; each carries its own hinge
    foot_sites: tuple[str, str]
    imu_site: str
    amp_body_names: tuple[str, ...]
    # Sole reference point in each foot body's frame (left, right): the position of
    # the foot site.  grounded_v2 AMP features use its height and vertical velocity.
    foot_reference_offsets: tuple[Vector3, Vector3]
    # Root height (m) under which a state is treated as fallen / getting up when
    # replay, statistics and evaluation are split into locomotion and recovery.
    recovery_height: float
    # Root height (m) above which an upright robot counts as standing again when
    # recovery success is evaluated.
    upright_height: float
    # (legacy AMP features, grounded_v2 AMP features) flat tasks.
    flat_tasks: tuple[str, str]
    mirrored_motion_cache: str
    mirrored_motion_format: str
    # Left/right naming: a pattern with the groups (prefix)(side)(suffix); the
    # counterpart has the same prefix and suffix and the other side token.
    side_pattern: str
    side_tokens: tuple[str, str]
    # A numbering prefix that differs between counterparts (PM01: "J00_" / "J06_").
    index_pattern: str | None
    # Largest accepted difference between a hinge axis and the reflection of its
    # counterpart, as the norm of the difference of unit vectors.
    mirror_axis_tolerance: float
    # Whether a joint's limits must equal the reflected limits of its counterpart.
    # When False, each mirrored clip is checked against the actual limits instead.
    mirror_requires_symmetric_limits: bool

    @property
    def amp_foot_offsets(self) -> tuple[Vector3, Vector3] | None:
        """``foot_local_offsets`` argument for ``build_amp_features``.

        None for the G1: its offset is that function's built-in default, and
        leaving it implicit keeps the metadata of existing G1 checkpoints unchanged.
        """
        return None if self.name == "g1" else self.foot_reference_offsets

    def task_schema(self, task: str) -> str:
        if task not in self.flat_tasks:
            raise ValueError(f"{task!r} is not a flat {self.name} task")
        return "legacy" if task == self.flat_tasks[0] else "grounded_v2"

    def counterpart(self, name: str, names: tuple[str, ...] | list[str]) -> str:
        """The left/right partner of ``name`` among ``names`` (itself when it has no side)."""
        key = self._side_key(name)
        if key is None:
            return name
        base, side = key
        other = self.side_tokens[1 - self.side_tokens.index(side)]
        matches = [candidate for candidate in names if self._side_key(candidate) == (base, other)]
        if len(matches) != 1:
            raise ValueError(f"{name!r} has {len(matches)} left/right counterparts in the {self.name} model")
        return matches[0]

    def _side_key(self, name: str) -> tuple[str, str] | None:
        stripped = re.sub(self.index_pattern, "", name) if self.index_pattern else name
        match = re.fullmatch(self.side_pattern, stripped)
        if match is None:
            return None
        return match.group(1) + "\0" + match.group(3), match.group(2)


G1 = RobotProfile(
    name="g1",
    xml_path=_ROBOTS / "unitree_g1" / "xmls" / "g1.xml",
    num_joints=29,
    num_bodies=30,
    root_body="pelvis",
    torso_body="torso_link",
    foot_bodies=("left_ankle_roll_link", "right_ankle_roll_link"),
    foot_sites=("left_foot", "right_foot"),
    imu_site="imu_in_pelvis",
    amp_body_names=(
        "pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
        "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
        "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
        "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link",
    ),
    foot_reference_offsets=((0.04, 0.0, -0.035), (0.04, 0.0, -0.035)),
    recovery_height=0.55,
    upright_height=0.65,
    flat_tasks=("Unitree-G1-AMP-Flat", "Unitree-G1-AMP-Flat-Aligned"),
    mirrored_motion_cache=".wmac_cache/motions/g1_mirrored",
    mirrored_motion_format="g1-sagittal-mirrored-motions-v1",
    side_pattern=r"()(left|right)(_.*)",
    side_tokens=("left", "right"),
    index_pattern=None,
    mirror_axis_tolerance=1e-9,
    mirror_requires_symmetric_limits=True,
)

PM01 = RobotProfile(
    name="pm01",
    xml_path=_ROBOTS / "engineai_pm01" / "xmls" / "pm01.xml",
    num_joints=23,
    num_bodies=29,
    root_body="LINK_BASE",
    torso_body="LINK_TORSO_YAW",
    foot_bodies=("LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"),
    foot_sites=("left_foot", "right_foot"),
    imu_site="imu",
    amp_body_names=(
        "LINK_BASE", "LINK_HIP_ROLL_L", "LINK_KNEE_PITCH_L", "LINK_ANKLE_ROLL_L",
        "LINK_HIP_ROLL_R", "LINK_KNEE_PITCH_R", "LINK_ANKLE_ROLL_R",
        "LINK_SHOULDER_ROLL_L", "LINK_ELBOW_PITCH_L", "LINK_ELBOW_YAW_L",
        "LINK_SHOULDER_ROLL_R", "LINK_ELBOW_PITCH_R", "LINK_ELBOW_YAW_R",
    ),
    # Centre of the underside of the official sole box (the foot sites in pm01.xml).
    foot_reference_offsets=((0.017, 0.00166, -0.04631), (0.017, -0.00166, -0.04631)),
    # The G1 values times the ratio of the two robots' walking base heights (1.035).
    recovery_height=0.57,
    upright_height=0.67,
    flat_tasks=("EngineAI-PM01-AMP-Flat", "EngineAI-PM01-AMP-Flat-Aligned"),
    mirrored_motion_cache=".wmac_cache/motions/pm01_mirrored",
    mirrored_motion_format="pm01-sagittal-mirrored-motions-v1",
    side_pattern=r"(.*_)(L|R)()",
    side_tokens=("L", "R"),
    index_pattern=r"^J\d+_",
    # The official model is not exactly symmetric: the elbow-pitch axes have the
    # same small forward lean (x = +0.0027) on both arms instead of mirrored ones,
    # 5.4e-3 as a unit-vector difference (0.31 deg); the other axes agree to 2e-5.
    mirror_axis_tolerance=6e-3,
    # The waist yaw range is (-4.014, 1.57) rad, so its own reflection differs.
    mirror_requires_symmetric_limits=False,
)

PROFILES: tuple[RobotProfile, ...] = (G1, PM01)


def profile_for_task(task: str) -> RobotProfile:
    for profile in PROFILES:
        if task in profile.flat_tasks:
            return profile
    raise ValueError(
        f"No robot profile for task {task!r}; flat tasks are "
        f"{[name for profile in PROFILES for name in profile.flat_tasks]}"
    )


def profile_for_bodies(body_names) -> RobotProfile:
    """Identify the robot from its body names (as mjlab or MuJoCo report them)."""
    names = {str(name).rsplit("/", 1)[-1] for name in body_names}
    matches = [profile for profile in PROFILES
               if profile.root_body in names and profile.torso_body in names
               and set(profile.foot_bodies) <= names]
    if len(matches) != 1:
        raise ValueError(f"Cannot identify the robot from its bodies: {sorted(names)[:6]}...")
    return matches[0]


def profile_for_joint_count(num_joints: int) -> RobotProfile:
    matches = [profile for profile in PROFILES if profile.num_joints == num_joints]
    if len(matches) != 1:
        raise ValueError(f"No single robot profile has {num_joints} joints")
    return matches[0]


def profile_by_name(name: str) -> RobotProfile:
    for profile in PROFILES:
        if profile.name == name:
            return profile
    raise ValueError(f"Unknown robot {name!r}; expected one of {[profile.name for profile in PROFILES]}")
