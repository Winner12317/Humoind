"""Audited AMP experts for locomotion-only and historical joint training."""
from __future__ import annotations

from pathlib import Path

import torch

from mjlab.tasks.registry import load_rl_cfg
from src.amp_features import amp_feature_dim, build_amp_features
from src.wmac.runner import ExpertPairs

from .config import LEGACY_WALKING_EXPERT
from .motions import (
    file_sha256, prepare_locomotion_dataset, read_motion, verify_locomotion_dataset, verify_mirrored_dataset,
)


class PerceptiveExpertPairs(ExpertPairs):
    """Sample adjacent frames within each audited interval, never across cuts.

    The Hiking task consumes its official walking timeline. Historical
    protocols may contain additional local motion data.
    """

    def __init__(self, cfg, env, device, dataset_root):
        root = Path(dataset_root).expanduser().resolve()
        if getattr(cfg, "amp_feature_schema", None) == "hiking_v1":
            self._init_hiking(cfg, env, device, root)
            return
        locomotion_only=getattr(cfg,"motion_scope","joint_recovery_v1")=="forward_walk_turn_v1"
        manifest = verify_locomotion_dataset(root) if locomotion_only else verify_mirrored_dataset(root)
        runner_cfg = load_rl_cfg(cfg.task)
        all_names = list(env.scene["robot"].body_names)
        if all_names != manifest["robot_mapping"]["body_names"]:
            raise ValueError("Expert NPZ body order differs from the current robot; an explicit conversion is required")
        body_ids = [all_names.index(name) for name in runner_cfg.amp_body_names]
        anchor_id = all_names.index(runner_cfg.amp_anchor_name)
        pelvis_id = all_names.index("pelvis")
        foot_ids = tuple(all_names.index(name) for name in ("left_ankle_roll_link", "right_ankle_roll_link"))
        schema = cfg.amp_feature_schema
        if schema != "grounded_v2":
            raise ValueError("Perceptive experts require the 203-dimensional grounded_v2 AMP schema")
        self.device = device
        self.fraction = cfg.expert_recovery_fraction
        if not 0 <= self.fraction <= 1:
            raise ValueError("Expert recovery fraction must lie in [0, 1]")
        if locomotion_only and self.fraction!=0:
            raise ValueError("Locomotion-only experts cannot sample recovery motions")
        groups, metadata = [], []
        for group_name in (("WalkandRun",) if locomotion_only else ("WalkandRun", "Recovery")):
            chunks, clips = [], []
            paths = sorted((root/group_name).glob("*.npz"))
            for path in paths:
                data = read_motion(path)
                frames = len(data["body_pos_w"])
                fps = float(data["fps"].reshape(-1)[0])
                expected_dt = getattr(env, "step_dt", None)
                if expected_dt is not None and abs(1/fps-float(expected_dt)) > 1e-6:
                    raise ValueError(f"Expert frame interval differs from control step: {path.name}; resample explicitly")
                p, q, v, w = [torch.as_tensor(data[key], device=device, dtype=torch.float32)
                               for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")]
                features = build_amp_features(p, q, v, w, body_ids=body_ids,
                    anchor_id=anchor_id, pelvis_id=pelvis_id, foot_ids=foot_ids, schema=schema)
                if features.shape != (frames, amp_feature_dim(schema)) or not bool(torch.isfinite(features).all()):
                    raise ValueError(f"Invalid expert AMP features: {path.name}")
                if frames > 1:
                    chunks.append(torch.stack((features[:-1], features[1:]), dim=1))
                clips.append({"file": str(path.relative_to(root)), "frames": frames,
                              "pairs": max(0, frames-1), "fps": fps,
                              "sha256": manifest["output_sha256"][str(path.relative_to(root))]})
            group = torch.cat(chunks) if chunks else None
            if group is None:
                raise ValueError(f"The perceptive experiment requires nonempty {group_name} adjacent pairs")
            groups.append(group)
            metadata.append({"group": group_name, "clips": clips, "clip_count": len(clips),
                             "pair_count": sum(row["pairs"] for row in clips)})
        self.groups = [groups[0],None] if locomotion_only else groups
        self.metadata = {"dataset_root": str(root), "manifest": str(root/"mirror_manifest.json"),
                         "manifest_sha256": file_sha256(root/"mirror_manifest.json"),
                         "schema": schema, "feature_dimension": amp_feature_dim(schema),
                         "expert_recovery_fraction": self.fraction, "groups": metadata,
                         "reference_height": (manifest["height_normalization"] if locomotion_only else
                                              "Original-project flat expert ground z=0; no terrain elevation added to experts."),
                         "pair_protocol": "Adjacent frames within each individual NPZ; no cross-clip pair and no duplicated final frame.",
                         "hiking_motion_data_used": manifest.get("hiking_motion_data_used",False),
                         "motion_scope":getattr(cfg,"motion_scope","joint_recovery_v1"),
                         "source_metadata":manifest.get("source_metadata",[]),
                         "terrain_reference_categories":manifest.get("terrain_reference_categories",{}),
                         "reference_limitations":manifest.get("limitations",[])}

    def _init_hiking(self, cfg, env, device, root):
        from .hiking_data import verify_hiking_dataset
        from .hiking_amp import frame, HISTORY_LENGTH
        standing_source=getattr(cfg,"standing_expert_source","none")
        if getattr(cfg,"hiking_training_version",1)>=3:
            if standing_source!="none":
                raise ValueError("Hiking episode playback requires the official walking data")
            self._init_episode_reference(cfg, env, device, root)
            return
        if standing_source=="none":
            manifest=verify_hiking_dataset(root)
        elif standing_source=="legacy_g1_locked_v1":
            from .standing_data import verify_hiking_standing_dataset
            manifest=verify_hiking_standing_dataset(root)
        else:
            raise ValueError("Unsupported Hiking standing expert source")
        clip_sources={output:(clip,mirrored)
            for clip in manifest["clips"]
            for output,mirrored in ((clip["original_output"],False),(clip["mirrored_output"],True))}
        if standing_source!="none" and any(
                clip["source"] not in ("hiking_public",standing_source) for clip in manifest["clips"]):
            raise ValueError("Hiking standing expert contains an unsupported clip source")
        names = list(env.scene["robot"].body_names)
        if names != manifest["robot_mapping"]["body_names"]:
            raise ValueError("Hiking expert body mapping differs from the simulated G1")
        torso = names.index("torso_link")
        robot = env.scene["robot"]
        default_pos = robot.data.default_joint_pos[0].to(device)
        default_vel = robot.data.default_joint_vel[0].to(device)
        chunks, ends, clips, offset = [], [], [], 0
        standing_windows=0
        for path in sorted((root/"WalkandRun").glob("*.npz")):
            data = read_motion(path)
            count = len(data["joint_pos"])
            if count < HISTORY_LENGTH:
                raise ValueError("Hiking references must contain a complete ten-frame history")
            fps = float(data["fps"].reshape(-1)[0])
            if abs(1/fps-float(env.step_dt)) > 1e-6:
                raise ValueError("Hiking reference frequency differs from the physical control rate")
            values = {k:torch.as_tensor(v, device=device, dtype=torch.float32) for k,v in data.items()}
            features = frame(values["joint_pos"], values["joint_vel"],
                values["body_quat_w"][:,torso], values["body_lin_vel_w"][:,torso],
                values["body_ang_vel_w"][:,torso], default_pos, default_vel)
            if not bool(torch.isfinite(features).all()):
                raise ValueError("Hiking AMP reference contains non-finite features")
            chunks.append(features)
            ends.append(torch.arange(offset+HISTORY_LENGTH-1, offset+count, device=device))
            relative=str(path.relative_to(root))
            clip_record={"file":relative,"frames":count,
                         "history_windows":count-HISTORY_LENGTH+1,"fps":fps}
            if standing_source!="none":
                source,mirrored=clip_sources[relative]
                is_standing=source["source"]==standing_source
                if is_standing:standing_windows+=count-HISTORY_LENGTH+1
                clip_record.update(source=source["source"],source_file=source["file"],
                    reference_kind="standing" if is_standing else "walking",mirrored=mirrored,
                    sha256=manifest["output_sha256"][relative],
                    source_description=(
                        "Static locked pose from the user's earlier G1 AMP framework; its 1201 source rows are re-expressed at 50 Hz, not a claim about the source recording duration."
                        if is_standing else
                        "Complete official Hiking YAML-selected walking trajectory with author interpolation; no local frame filtering."))
            clips.append(clip_record)
            offset += count
        if not chunks:
            raise ValueError("No complete Hiking AMP references were found")
        self.device, self.fraction = device, 0.
        self.groups = [torch.cat(chunks), None]
        self._hiking_history_ends = torch.cat(ends)
        self.metadata = {"dataset_root":str(root),"manifest":str(root/"mirror_manifest.json"),
            "manifest_sha256":file_sha256(root/"mirror_manifest.json"),
            "schema":"hiking_v1","feature_dimension":67,"history_length":HISTORY_LENGTH,
            "expert_recovery_fraction":0.,"groups":[{"group":"WalkandRun","clips":clips,
                "clip_count":len(clips),"history_window_count":len(self._hiking_history_ends)}],
            "reference_height":None,"pair_protocol":"Ten consecutive reference frames inside each complete source NPZ",
            "motion_scope":cfg.motion_scope,"hiking_motion_data_used":True,
            "source_metadata":manifest.get("source_metadata",[]),
            "reference_limitations":manifest.get("limitations",[])}
        if standing_source!="none":
            self.metadata.update(standing_expert_source=standing_source,
                standing_history_window_count=standing_windows,
                total_history_window_count=len(self._hiking_history_ends),
                standing_expert_window_fraction=standing_windows/len(self._hiking_history_ends),
                expert_sampling="Uniform over all valid ten-frame windows in all four complete clips; no additional mixture weights or command conditioning")

    def _init_episode_reference(self, cfg, env, device, root):
        import numpy as np
        from .hiking_amp import HISTORY_LENGTH, frame
        from .hiking_data import official_hiking_reference
        from .locomotion_data import HIKING_SOURCE_SHA256, interpolate_hiking_motion
        from mjlab.utils.lab_api.math import axis_angle_from_quat, quat_mul, quat_conjugate
        from .mirroring import opposite_name

        # The official AMP terms need only the raw torso and joint trajectory.
        # Do not reconstruct expert velocities through a MuJoCo body tree.
        path = official_hiking_reference(root)
        robot = env.scene["robot"]
        names = list(robot.joint_names)
        with np.load(path, allow_pickle=False) as raw:
            fps = float(raw["framerate"])
            source_names = raw["joint_names"].tolist()
            if set(names) != set(source_names) or abs(1/fps - env.step_dt) > 1e-6:
                raise ValueError("Official Hiking reference and G1 control layout disagree")
            joints = raw["joint_pos"][:, [source_names.index(name) for name in names]]
            position, quaternion, joints = interpolate_hiking_motion(
                raw["base_pos_w"], raw["base_quat_w"], joints, fps)
        # Hiking's "frontward" velocity pads the final next frame with itself.
        # Its angular increment is q_next * conjugate(q), using the short arc.
        dt = 1/fps
        pos, rot, q = (torch.as_tensor(value, device=device, dtype=torch.float32)
                       for value in (position, quaternion, joints))
        v = torch.diff(pos, dim=0, append=pos[-1:]) / dt
        dq = torch.diff(q, dim=0, append=q[-1:]) / dt
        next_rot = torch.cat((rot[1:], rot[-1:]))
        next_rot = torch.where((rot*next_rot).sum(-1, keepdim=True) < 0, -next_rot, next_rot)
        omega = axis_angle_from_quat(quat_mul(next_rot, quat_conjugate(rot))) / dt
        default = robot.data.default_joint_pos[0].to(device)
        default_vel = robot.data.default_joint_vel[0].to(device)
        normal = frame(q, dq, rot, v, omega, default, default_vel)
        # Apply the original G1 sagittal permutation/signs in runtime joint order.
        # In the release's G1 table each left/right joint is exchanged and
        # roll/yaw angles change sign. Pitch, knee and elbow retain their sign.
        perm = [names.index(opposite_name(name)) for name in names]
        sign = q.new_tensor([-1 if name.endswith(("_roll_joint", "_yaw_joint")) else 1 for name in names])
        mirror = frame(q[:, perm]*sign, dq[:, perm]*sign, rot*rot.new_tensor([1,-1,1,-1]),
            v*v.new_tensor([1,-1,1]), omega*omega.new_tensor([-1,1,-1]), default, default_vel)
        self.device, self.fraction = device, 0.
        self.groups = [torch.cat((normal, mirror)), None]
        self.official_reference_frames = len(normal)
        if not torch.isfinite(self.groups[0]).all():
            raise ValueError("Nonfinite official Hiking expert frame")
        self.metadata = dict(
            dataset_root=str(path.parent), selection=str(path.with_name("parkour_motion_without_run.yaml")),
            source_sha256=dict(HIKING_SOURCE_SHA256), schema="hiking_v1",
            feature_dimension=normal.shape[-1], history_length=HISTORY_LENGTH, expert_recovery_fraction=0.,
            groups=[dict(group="WalkandRun", clip_count=2, clips=[
                dict(file=path.name, mirrored=mirrored, frames=len(normal), fps=fps,
                     source_sha256=HIKING_SOURCE_SHA256[path.name]) for mirrored in (False, True)])],
            standing_expert_source="none", motion_scope=cfg.motion_scope, hiking_motion_data_used=True,
            expert_sampling="Per-environment 50 Hz episode playback, round-to-nearest frame and ten-frame observation history",
            reference_start_fraction=[0., .9], mirror_probability=.5,
            reference_observation_on_robot_reset="New start and mirror, first frame fills history")

        self.legacy_walking_clips = ()
        self.legacy_walking_sha256 = {}
        if getattr(cfg, "walking_expert_source", None) == LEGACY_WALKING_EXPERT:
            self._load_legacy_walking_reference(root, names, env.step_dt, default, default_vel)
            self._pool_walking_references(normal, mirror, fps)

    def _pool_walking_references(self, normal, mirror, fps):
        from .hiking_amp import HISTORY_LENGTH

        clips = (normal, mirror, *(clip for pair in self.legacy_walking_clips for clip in pair))
        offset = 0
        ends = []
        for clip in clips:
            if len(clip) < HISTORY_LENGTH:
                raise ValueError("Pooled walking reference requires complete ten-frame clips")
            ends.append(torch.arange(offset + HISTORY_LENGTH - 1, offset + len(clip), device=self.device))
            offset += len(clip)
        self.groups[0] = torch.cat(clips)
        self._hiking_history_ends = torch.cat(ends)

        group = self.metadata["groups"][0]
        legacy = self.metadata["legacy_walking_reference"]
        local_windows = sum(len(indices) for indices in ends[2:])
        official_windows = sum(len(indices) for indices in ends[:2])
        for entry in group["clips"]:
            entry["history_windows"] = entry["frames"] - HISTORY_LENGTH + 1
        for clip in legacy["clips"]:
            for key, mirrored in (("original_output", False), ("mirrored_output", True)):
                name = clip[key]
                group["clips"].append(dict(file=name, source_file=clip["file"], frames=clip["frames"],
                    history_windows=clip["frames"] - HISTORY_LENGTH + 1, fps=fps,
                    source=clip["source"], mirrored=mirrored, sha256=self.legacy_walking_sha256[name]))
        group.update(clip_count=len(clips), history_window_count=len(self._hiking_history_ends))
        legacy["history_window_count"] = local_windows
        self.metadata.update(
            expert_sampling="Uniform over valid ten-frame windows pooled across official Hiking and local clips",
            pair_protocol="Each sampled history stays within one original or mirrored clip",
            official_history_window_count=official_windows,
            total_history_window_count=len(self._hiking_history_ends),
            legacy_walking_window_fraction=local_windows / len(self._hiking_history_ends),
        )
        for key in ("reference_start_fraction", "mirror_probability", "reference_observation_on_robot_reset"):
            self.metadata.pop(key)

    def _load_legacy_walking_reference(self, root, joint_names, dt, default, default_vel):
        """Load the historical repository clips in the current Hiking AMP format."""
        from .hiking_amp import FRAME_DIM, HISTORY_LENGTH, frame

        source = Path(__file__).resolve().parents[2] / "src/assets/motions/g1/amp"
        paths = prepare_locomotion_dataset(
            source, root / "legacy_local_walk_turn_v2", include_hiking=False, include_grail=False,
        )
        manifest = verify_locomotion_dataset(paths.root, source_root=source)
        mapping = manifest["robot_mapping"]
        source_joints = mapping["joint_names"]
        if len(joint_names) != len(source_joints) or set(joint_names) != set(source_joints):
            raise ValueError("Legacy walking reference and G1 joint layout disagree")
        joint_indices = [source_joints.index(name) for name in joint_names]
        torso = mapping["body_names"].index("torso_link")
        clips = manifest["clips"]
        filenames = [clip[key] for clip in clips for key in ("original_output", "mirrored_output")]
        if sorted(filenames) != sorted(manifest["output_sha256"]) or any(
            clip["source"] != "local_supplement" for clip in clips
        ):
            raise ValueError("Legacy walking reference must contain only original and mirrored repository clips")

        def load_features(name, length):
            data = read_motion(paths.root / name)
            fps = float(data["fps"].reshape(-1)[0])
            if fps <= 0. or abs(1 / fps - dt) > 1e-6:
                raise ValueError(f"Legacy walking reference frequency differs from the control rate: {name}")
            values = (data["joint_pos"][:, joint_indices], data["joint_vel"][:, joint_indices],
                      data["body_quat_w"][:, torso], data["body_lin_vel_w"][:, torso],
                      data["body_ang_vel_w"][:, torso])
            features = frame(*(torch.as_tensor(value, device=self.device, dtype=torch.float32)
                               for value in values), default, default_vel)
            if (length < HISTORY_LENGTH or features.shape != (length, FRAME_DIM)
                    or not bool(torch.isfinite(features).all())):
                raise ValueError(f"Invalid legacy walking AMP reference: {name}")
            return features

        self.legacy_walking_clips = tuple(
            (load_features(clip["original_output"], clip["frames"]),
             load_features(clip["mirrored_output"], clip["frames"]))
            for clip in clips
        )
        self.legacy_walking_sha256 = dict(manifest["output_sha256"])
        self.metadata["legacy_walking_reference"] = {
            "source": LEGACY_WALKING_EXPERT,
            "dataset_root": str(paths.root),
            "source_clips": len(clips),
            "training_clips": 2 * len(clips),
            "source_frames": sum(clip["frames"] for clip in clips),
            "training_frames": 2 * sum(clip["frames"] for clip in clips),
            "output_sha256": self.legacy_walking_sha256,
            "clips": clips,
        }

    def sample_histories(self, size):
        from .hiking_amp import HISTORY_LENGTH
        indices = torch.randint(len(self._hiking_history_ends), (size,), device=self.device)
        ends = self._hiking_history_ends[indices]
        return self.groups[0][ends[:,None]+torch.arange(1-HISTORY_LENGTH,1,device=self.device)]
