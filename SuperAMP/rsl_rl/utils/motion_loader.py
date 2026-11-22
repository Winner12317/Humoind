from __future__ import annotations
import math
import numpy as np
import os
import torch
from collections.abc import Sequence
from tqdm import tqdm

import mjlab.utils.lab_api.math as math_utils
from src.amp_features import LEGACY_SCHEMA, amp_feature_dim, amp_feature_metadata, build_amp_features


class AMPLoader:
    def __init__(self, motion_file: str,
                 body_names: Sequence[str],
                 anchor_name: str,
                 all_body_names: Sequence[str],
                 device: str = "cuda:0",
                 feature_schema: str = LEGACY_SCHEMA,
                 expert_recovery_fraction: float | None = None,
                 root_name: str = "pelvis",
                 foot_names: Sequence[str] = ("left_ankle_roll_link", "right_ankle_roll_link"),
                 foot_reference_offsets: Sequence[Sequence[float]] | None = None):
        """Load AMP motion data.

        Args:
            motion_file: Path to a single .npz file or a directory of .npz files.
            body_names: Names of the target bodies to track.
            anchor_name: Name of the anchor (root) body.
            all_body_names: Ordered list of *all* body names in the model.
                The index of each name in this list must match the body
                dimension in the .npz arrays.
            device: Torch device.
            feature_schema: Versioned features; the default preserves old data.
            expert_recovery_fraction: Recovery pair proportion for the new schema.
            root_name: Root body, for the grounded schema's gravity and height.
            foot_names: Left and right foot bodies, for the grounded schema.
            foot_reference_offsets: Sole reference point in each foot frame;
                None uses the G1 offset built into ``build_amp_features``.
        """
        assert os.path.exists(motion_file), f"Invalid path: {motion_file}"

        # resolve name -> index
        all_names_list = list(all_body_names)
        self._body_indexes = [all_names_list.index(n) for n in body_names]
        self._anchor_indexes = all_names_list.index(anchor_name)
        self._num_bodies = len(self._body_indexes)
        self.feature_schema = feature_schema
        self._root_name = root_name
        self._foot_names = tuple(foot_names)
        self._foot_reference_offsets = foot_reference_offsets
        self.feature_metadata = amp_feature_metadata(feature_schema, self._num_bodies, foot_reference_offsets)
        if expert_recovery_fraction is not None and not 0 <= expert_recovery_fraction <= 1:
            raise ValueError("expert_recovery_fraction must be between zero and one")
        self.expert_recovery_fraction = expert_recovery_fraction

        # 检查是文件还是文件夹
        if os.path.isfile(motion_file):
            # 单个文件的情况（保持向后兼容）
            motion_files = [motion_file]
            motion_names = [os.path.splitext(os.path.basename(motion_file))[0]]
        elif os.path.isdir(motion_file):
            # 文件夹的情况：递归查找所有子目录下的 .npz 文件
            motion_names = []
            motion_files = []
            for root, _dirs, files in os.walk(motion_file):
                for filename in sorted(files):
                    if filename.endswith('.npz'):
                        motion_names.append(os.path.splitext(filename)[0])
                        motion_files.append(os.path.join(root, filename))
            motion_files, motion_names = zip(*sorted(zip(motion_files, motion_names))) if motion_files else ([], [])
            motion_files, motion_names = list(motion_files), list(motion_names)
            assert len(motion_files) > 0, f"No npz files found in directory: {motion_file}"
        else:
            raise ValueError(f"Path is neither a file nor a directory: {motion_file}")
        
        # 存储所有motion的数据列表
        self.motion_names = motion_names
        if feature_schema != LEGACY_SCHEMA:
            self._load_grounded_pairs(motion_files, all_names_list, device)
            return
        self._body_pos_b_list = []
        self._body_quat_b_list = []
        self._body_ori_b_list = []
        self._body_lin_vel_b_list = []
        self._body_ang_vel_b_list = []
        
        # 处理每个motion文件
        for motion_idx, (motion_name, motion_path) in enumerate(zip(motion_names, motion_files)):
            print(f"Processing motion {motion_idx+1}/{len(motion_files)}: {motion_name}")
            data = np.load(motion_path)
            
            if motion_idx == 0:
                self.fps = data["fps"]
            
            _dof_pos = torch.tensor(data["joint_pos"], dtype=torch.float32, device=device)
            _dof_vel = torch.tensor(data["joint_vel"], dtype=torch.float32, device=device)
            _body_pos_w = torch.tensor(data["body_pos_w"], dtype=torch.float32, device=device)
            _body_quat_w = torch.tensor(data["body_quat_w"], dtype=torch.float32, device=device)
            _body_lin_vel_w = torch.tensor(data["body_lin_vel_w"], dtype=torch.float32, device=device)
            _body_ang_vel_w = torch.tensor(data["body_ang_vel_w"], dtype=torch.float32, device=device)
            
            time_step_total = _dof_pos.shape[0]
            
            # 为当前motion初始化存储
            _body_pos_b = torch.zeros((time_step_total, self._num_bodies, 3), dtype=torch.float32, device=device)
            _body_quat_b = torch.zeros((time_step_total, self._num_bodies, 4), dtype=torch.float32, device=device)
            _body_ori_b = torch.zeros((time_step_total, self._num_bodies, 6), dtype=torch.float32, device=device)
            _body_lin_vel_b = torch.zeros((time_step_total, self._num_bodies, 3), dtype=torch.float32, device=device)
            _body_ang_vel_b = torch.zeros((time_step_total, self._num_bodies, 3), dtype=torch.float32, device=device)
            
            # 处理所有帧
            for frame_idx in tqdm(range(time_step_total), desc=f"Preloading AMP data for {motion_name}"):
                # 获取当前帧的anchor和body数据
                tgt_anchor_pos_w = _body_pos_w[frame_idx, self._anchor_indexes, :].squeeze().unsqueeze(0).repeat(self._num_bodies, 1)
                tgt_anchor_quat_w = _body_quat_w[frame_idx, self._anchor_indexes, :].squeeze().unsqueeze(0).repeat(self._num_bodies, 1)
                tgt_body_pos_w = _body_pos_w[frame_idx, self._body_indexes, :]
                tgt_body_quat_w = _body_quat_w[frame_idx, self._body_indexes, :]
                tgt_body_lin_vel_w = _body_lin_vel_w[frame_idx, self._body_indexes, :]
                tgt_body_ang_vel_w = _body_ang_vel_w[frame_idx, self._body_indexes, :]

                # 计算body相对于anchor的位置和姿态 (局部坐标系)
                tgt_robot_body_pos_b, tgt_robot_body_quat_b = (
                    math_utils.subtract_frame_transforms(
                        tgt_anchor_pos_w,
                        tgt_anchor_quat_w,
                        tgt_body_pos_w,
                        tgt_body_quat_w,
                    )
                )

                # 将姿态四元数转换为旋转矩阵的前两列
                mat = math_utils.matrix_from_quat(tgt_robot_body_quat_b)
                tgt_robot_body_ori_b = mat[..., :, :2].reshape(self._num_bodies, 6)

                # 将速度转换到每个body自己的局部坐标系
                tgt_body_lin_vel_b = math_utils.quat_apply_inverse(
                    tgt_body_quat_w,
                    tgt_body_lin_vel_w,
                )

                tgt_body_ang_vel_b = math_utils.quat_apply_inverse(
                    tgt_body_quat_w,
                    tgt_body_ang_vel_w,
                )

                # 存储当前帧的局部坐标系数据
                _body_pos_b[frame_idx] = tgt_robot_body_pos_b
                _body_quat_b[frame_idx] = tgt_robot_body_quat_b
                _body_ori_b[frame_idx] = tgt_robot_body_ori_b
                _body_lin_vel_b[frame_idx] = tgt_body_lin_vel_b
                _body_ang_vel_b[frame_idx] = tgt_body_ang_vel_b
            
            # 将当前motion的数据添加到列表
            self._body_pos_b_list.append(_body_pos_b)
            self._body_quat_b_list.append(_body_quat_b)
            self._body_ori_b_list.append(_body_ori_b)
            self._body_lin_vel_b_list.append(_body_lin_vel_b)
            self._body_ang_vel_b_list.append(_body_ang_vel_b)
        
        # 为了向后兼容，使用第一个motion的数据作为默认值
        self.time_step_total = self._body_pos_b_list[0].shape[0]
        self.motion_total_time = self.time_step_total / self.fps
        self._body_pos_b = self._body_pos_b_list[0]
        self._body_quat_b = self._body_quat_b_list[0]
        self._body_ori_b = self._body_ori_b_list[0]
        self._body_lin_vel_b = self._body_lin_vel_b_list[0]
        self._body_ang_vel_b = self._body_ang_vel_b_list[0]

    @property
    def observation_dim(self) -> int:
        return amp_feature_dim(self.feature_schema, self._num_bodies)

    def _load_grounded_pairs(self, motion_files, all_names, device):
        """Use exactly the shared kinematics and pair pools used by WMAC."""
        groups = [[], []]
        self.device = torch.device(device)
        for motion_idx, path in enumerate(motion_files):
            with np.load(path) as data:
                if motion_idx == 0:
                    self.fps = data["fps"].copy()
                arrays = [torch.as_tensor(data[key], dtype=torch.float32, device=device)
                          for key in ("body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")]
            features = build_amp_features(
                *arrays, body_ids=self._body_indexes, anchor_id=self._anchor_indexes,
                pelvis_id=all_names.index(self._root_name),
                foot_ids=[all_names.index(name) for name in self._foot_names],
                schema=self.feature_schema, foot_local_offsets=self._foot_reference_offsets,
            )
            if motion_idx == 0:
                self.time_step_total = len(features)
                self.motion_total_time = self.time_step_total / self.fps
            if len(features) > 1:
                recovery = "recovery" in {part.lower() for part in os.path.normpath(path).split(os.sep)}
                groups[int(recovery)].append(torch.stack((features[:-1], features[1:]), 1))
        self._pair_groups = [torch.cat(group) if group else None for group in groups]
        if not any(group is not None for group in self._pair_groups):
            raise ValueError("No adjacent expert motion pairs were found")

    def _sample_grounded_pairs(self, size):
        fraction = self.expert_recovery_fraction
        if fraction is None or fraction == 0:
            if not hasattr(self, "_all_pairs"):
                self._all_pairs = torch.cat([group for group in self._pair_groups if group is not None])
            pool = self._all_pairs
            pairs = pool[torch.randint(len(pool), (size,), device=self.device)]
        else:
            recovery_count = int(round(size * fraction))
            parts = []
            for group_id, count in enumerate((size - recovery_count, recovery_count)):
                if count:
                    pool = self._pair_groups[group_id]
                    if pool is None:
                        pool = self._pair_groups[1 - group_id]
                    parts.append(pool[torch.randint(len(pool), (count,), device=self.device)])
            pairs = torch.cat(parts)[torch.randperm(size, device=self.device)]
        return pairs[:, 0], pairs[:, 1]

    def feed_forward_generator(self, num_mini_batch, mini_batch_size):
        if self.feature_schema != LEGACY_SCHEMA:
            for _ in range(num_mini_batch):
                yield self._sample_grounded_pairs(mini_batch_size)
            return
        num_motions = len(self._body_pos_b_list)
        
        for batch_idx in range(num_mini_batch):
            # 按顺序循环选择motion文件
            motion_idx = batch_idx % num_motions
            
            # 获取当前motion的数据
            current_body_pos_b = self._body_pos_b_list[motion_idx]
            current_body_ori_b = self._body_ori_b_list[motion_idx]
            current_body_lin_vel_b = self._body_lin_vel_b_list[motion_idx]
            current_body_ang_vel_b = self._body_ang_vel_b_list[motion_idx]
            current_time_step_total = current_body_pos_b.shape[0]
            
            # 从当前motion中随机采样
            idxs = torch.randint(0, current_time_step_total, (mini_batch_size,), device=current_body_pos_b.device)
            idxs = torch.clamp(idxs, max=current_time_step_total - 1)
            
            batch_body_pos_b = current_body_pos_b[idxs]  # (mini_batch_size, num_bodies, 3)
            batch_body_ori_b = current_body_ori_b[idxs]  # (mini_batch_size, num_bodies, 6)
            batch_body_lin_vel_b = current_body_lin_vel_b[idxs]  # (mini_batch_size, num_bodies, 3)
            batch_body_ang_vel_b = current_body_ang_vel_b[idxs]  # (mini_batch_size, num_bodies, 3)
            s = torch.cat(
                [
                    batch_body_pos_b.reshape(mini_batch_size, -1),
                    batch_body_ori_b.reshape(mini_batch_size, -1),
                    batch_body_lin_vel_b.reshape(mini_batch_size, -1),
                    batch_body_ang_vel_b.reshape(mini_batch_size, -1),
                ],
                dim=-1,
            )  # (mini_batch_size, obs_dim)

            next_idxs = (idxs + 1)
            next_idxs = torch.clamp(next_idxs, max=current_time_step_total - 1)
            batch_next_body_pos_b = current_body_pos_b[next_idxs]  # (mini_batch_size, num_bodies, 3)
            batch_next_body_ori_b = current_body_ori_b[next_idxs]  # (mini_batch_size, num_bodies, 6)
            batch_next_body_lin_vel_b = current_body_lin_vel_b[next_idxs]  # (mini_batch_size, num_bodies, 3)
            batch_next_body_ang_vel_b = current_body_ang_vel_b[next_idxs]  # (mini_batch_size, num_bodies, 3)
            s_next = torch.cat(
                [
                    batch_next_body_pos_b.reshape(mini_batch_size, -1),
                    batch_next_body_ori_b.reshape(mini_batch_size, -1),
                    batch_next_body_lin_vel_b.reshape(mini_batch_size, -1),
                    batch_next_body_ang_vel_b.reshape(mini_batch_size, -1),
                ],
                dim=-1,
            )  # (mini_batch_size, obs_dim)
            yield s, s_next
