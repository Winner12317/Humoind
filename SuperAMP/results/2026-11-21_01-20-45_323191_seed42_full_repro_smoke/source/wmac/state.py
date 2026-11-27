"""Differentiable state/observation/reward adapters for the isolated WMAC experiment.

Quaternions use MuJoCo's wxyz convention. The model state contains the *true*
generalized state, rather than the one-substep-stale derived simulator buffers.
Task rewards involving contact are supplied separately; joint acceleration and
soft termination are explicitly differentiable proxies for the simulator terms.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

from src.amp_features import amp_feature_dim, build_amp_features
from src.recovery_rewards import recovery_height_reward, recovery_root_up_z
from .config import RECOVERY_COUNTER_SEMANTICS

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv

STATE_DIM = 71
CONTEXT_DIM = 35
ACTOR_FRAME_DIM = 96
CRITIC_FEATURE_DIM = 136
AMP_DIM = 195


@dataclass(frozen=True)
class _KinematicsLevel:
    """State-independent topology and constants for one depth of the body tree."""

    parent_slots: torch.Tensor
    joint_indices: torch.Tensor
    joint_mask: torch.Tensor
    offsets: torch.Tensor
    orientations: torch.Tensor
    axes: torch.Tensor
    pivots: torch.Tensor
    qrefs: torch.Tensor
    has_pivots: bool


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / q.norm(dim=-1, keepdim=True).clamp_min(1e-8)


def quat_inv(q: torch.Tensor) -> torch.Tensor:
    return torch.cat((q[..., :1], -q[..., 1:]), dim=-1) / q.square().sum(-1, keepdim=True).clamp_min(1e-8)


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, v1 = q1[..., :1], q1[..., 1:]
    w2, v2 = q2[..., :1], q2[..., 1:]
    v1, v2 = torch.broadcast_tensors(v1, v2)
    return torch.cat((w1 * w2 - (v1 * v2).sum(-1, keepdim=True), w1 * v2 + w2 * v1 + torch.cross(v1, v2, dim=-1)), -1)


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q = quat_normalize(q)
    xyz, v = torch.broadcast_tensors(q[..., 1:], v)
    uv = torch.cross(xyz, v, dim=-1)
    return v + 2 * (q[..., :1] * uv + torch.cross(xyz, uv, dim=-1))


def quat_from_rotvec(rotvec: torch.Tensor) -> torch.Tensor:
    angle = rotvec.norm(dim=-1, keepdim=True)
    # sinc avoids the undefined derivative of sin(x)/x at identity.
    xyz = 0.5 * torch.sinc(angle / (2 * math.pi)) * rotvec
    return quat_normalize(torch.cat((torch.cos(0.5 * angle), xyz), -1))


def quat_to_matrix(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat_normalize(q).unbind(-1)
    return torch.stack((1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                        2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                        2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)), -1).reshape(*q.shape[:-1], 3, 3)


def heading_quat(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat_normalize(q).unbind(-1)
    sin_yaw = 2 * (w*z+x*y)
    cos_yaw = 1-2*(y*y+z*z)
    # A vertical forward axis has no defined heading. Feed atan2 a safe pair
    # before selecting a conventional zero heading: masking its output alone
    # still lets atan2(0, 0)'s undefined derivative contaminate backward.
    degenerate = sin_yaw.square()+cos_yaw.square() < 1e-8
    sin_yaw = torch.where(degenerate, torch.zeros_like(sin_yaw), sin_yaw)
    cos_yaw = torch.where(degenerate, torch.ones_like(cos_yaw), cos_yaw)
    yaw = torch.atan2(sin_yaw, cos_yaw)
    zero = torch.zeros_like(yaw)
    return torch.stack((torch.cos(yaw/2), zero, zero, torch.sin(yaw/2)), -1)


def recovery_foot_slip_mask(state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
    return (context[..., 33] > .5) & ((state[..., 2] < .55) | (context[..., 34] > 0))


class StateAdapter:
    """Build Torch FK from the actual compiled model and expose learning signals."""

    def __init__(self, env: ManagerBasedRlEnv, recovery_counter_semantics="hard_forward_v2"):
        if recovery_counter_semantics not in RECOVERY_COUNTER_SEMANTICS:
            raise ValueError(f"Unknown recovery counter semantics: {recovery_counter_semantics!r}")
        self.recovery_counter_semantics = recovery_counter_semantics
        self.env = env
        self.robot = env.scene["robot"]
        self.device = torch.device(env.device)
        self.dt = float(env.step_dt)
        model = env.sim.mj_model
        self.body_names = tuple(self.robot.body_names)
        self.body_ids = self.robot.indexing.body_ids.long()
        self.anchor_id = self.body_names.index("torso_link")
        self.amp_body_names = ("pelvis", "left_hip_roll_link", "left_knee_link", "left_ankle_roll_link",
                               "right_hip_roll_link", "right_knee_link", "right_ankle_roll_link",
                               "left_shoulder_roll_link", "left_elbow_link", "left_wrist_yaw_link",
                               "right_shoulder_roll_link", "right_elbow_link", "right_wrist_yaw_link")
        self.amp_ids = [self.body_names.index(n) for n in self.amp_body_names]
        observations = getattr(env.cfg, "observations", {})
        amp_group = observations.get("amp")
        grounded_term = getattr(amp_group, "terms", {}).get("grounded_features")
        self.amp_feature_schema = (grounded_term.params.get("schema", "legacy")
                                   if grounded_term is not None else "legacy")
        self.amp_dim = amp_feature_dim(self.amp_feature_schema, len(self.amp_ids))
        self.pelvis_id = self.body_names.index("pelvis")
        self.foot_ids = tuple(self.body_names.index(name) for name in
                              ("left_ankle_roll_link", "right_ankle_roll_link"))
        self._joint_global_to_local = {int(j): i for i, j in enumerate(self.robot.indexing.joint_ids.tolist())}
        if len(self._joint_global_to_local) != 29:
            raise ValueError("WMAC state adapter requires the 29-hinge G1 model")
        gids = self.body_ids.tolist()
        to_local = {gid: i for i, gid in enumerate(gids)}
        self.parents, self.joint_indices = [], []
        axes, offsets, orientations, pivots, qrefs = [], [], [], [], []
        for i, gid in enumerate(gids):
            self.parents.append(to_local.get(int(model.body_parentid[gid]), -1))
            jnum = int(model.body_jntnum[gid])
            jid = int(model.body_jntadr[gid]) if jnum else -1
            if jnum > 1:
                raise ValueError("WMAC FK supports at most one joint per body")
            ji = self._joint_global_to_local.get(jid, -1)
            if ji >= 0 and model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                raise ValueError("WMAC FK supports hinge joints only")
            self.joint_indices.append(ji)
            offsets.append(model.body_pos[gid].copy())
            orientations.append(model.body_quat[gid].copy())
            axes.append(model.jnt_axis[jid].copy() if ji >= 0 else np.zeros(3))
            pivots.append(model.jnt_pos[jid].copy() if ji >= 0 else np.zeros(3))
            qrefs.append(float(model.qpos0[model.jnt_qposadr[jid]]) if ji >= 0 else 0.)
        def tensor(x):
            return torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)
        self.offsets, self.orientations = tensor(offsets), tensor(orientations)
        self.axes, self.pivots, self.qrefs = tensor(axes), tensor(pivots), tensor(qrefs)
        # MuJoCo stores parents before children. Group independent branches at
        # equal depth so each Torch operation handles a whole level rather than
        # launching a separate sequence of small kernels for every body.
        depths = []
        for i, parent in enumerate(self.parents):
            if parent >= i:
                raise ValueError("WMAC FK requires parents before their children")
            depths.append(0 if parent < 0 else depths[parent]+1)
        levels = [[i for i, depth in enumerate(depths) if depth == d]
                  for d in range(max(depths)+1)]
        self.kinematics_level_sizes = tuple(len(level) for level in levels)
        self._fk_root_count = len(levels[0])
        self._fk_levels = []
        for previous, current in zip(levels, levels[1:]):
            slots = {body: i for i, body in enumerate(previous)}
            ids = torch.tensor(current, dtype=torch.long, device=self.device)
            joints = torch.tensor([self.joint_indices[i] for i in current], dtype=torch.long, device=self.device)
            self._fk_levels.append(_KinematicsLevel(
                parent_slots=torch.tensor([slots[self.parents[i]] for i in current], dtype=torch.long, device=self.device),
                joint_indices=joints.clamp_min(0),
                joint_mask=(joints >= 0),
                offsets=self.offsets.index_select(0, ids),
                orientations=self.orientations.index_select(0, ids),
                axes=self.axes.index_select(0, ids),
                pivots=self.pivots.index_select(0, ids),
                qrefs=self.qrefs.index_select(0, ids),
                has_pivots=bool(np.any(np.asarray(pivots)[current] != 0)),
            ))
        self._fk_levels = tuple(self._fk_levels)
        depth_order = [i for level in levels for i in level]
        self._fk_body_order = torch.tensor(np.argsort(depth_order), dtype=torch.long, device=self.device)
        self.default_joint_pos = self.robot.data.default_joint_pos[0].detach().clone()
        self.default_joint_vel = self.robot.data.default_joint_vel[0].detach().clone()
        self.joint_limits = self.robot.data.soft_joint_pos_limits[0].detach().clone()
        self.desired_height = float(self.robot.data.default_root_state[0, 2])
        self.torso_gid = gids[self.anchor_id]
        self.default_torso_com = tensor(model.body_ipos[self.torso_gid].copy())
        foot_ids = [int(g.id) for g in self.robot.indexing.geoms if "foot" in g.name and "collision" in g.name]
        self.friction_geom_ids = foot_ids or self.robot.indexing.geom_ids.tolist()
        # The source MDP uses the pelvis IMU site for linear velocity.
        imu = [s for s in self.robot.indexing.sites if s.name.split("/")[-1] == "imu_in_pelvis"]
        self.imu_offset = tensor(model.site_pos[imu[0].id].copy() if imu else np.zeros(3))
        self.imu_rotation = tensor(model.site_quat[imu[0].id].copy() if imu else [1., 0., 0., 0.])
        self.reward_cfg = env.cfg.rewards
        self.recovery_foot_slip = None
        self.reward_scale = self.dt if env.cfg.scale_rewards_by_dt else 1.
        self.max_delay_steps = int(getattr(getattr(env, "termination_manager", None), "_max_delay_steps", 250))
        self.height_threshold = float(env.cfg.terminations["bad_base_height"].params["minimum_height"])
        self.tilt_angle = float(env.cfg.terminations["bad_orientation"].params["limit_angle"])
        self.tilt_cos = math.cos(self.tilt_angle)

    def state(self) -> torch.Tensor:
        idx, data = self.robot.indexing, self.robot.data.data
        root = data.qpos[:, idx.free_joint_q_adr.long()]
        vel = data.qvel[:, idx.free_joint_v_adr.long()]
        q = root[:, 3:7]
        return torch.cat((root[:, :3], q, vel[:, :3], quat_apply(q, vel[:, 3:6]),
                          data.qpos[:, idx.joint_q_adr.long()], data.qvel[:, idx.joint_v_adr.long()]), -1)

    def context(self) -> torch.Tensor:
        n = self.env.num_envs
        friction = self.env.sim.model.geom_friction
        com = self.env.sim.model.body_ipos
        if friction.ndim == 2:
            friction = friction.unsqueeze(0)
        if com.ndim == 2:
            com = com.unsqueeze(0)
        friction = friction.expand(n, -1, -1)[:, self.friction_geom_ids, 0].mean(-1, keepdim=True)
        com = com.expand(n, -1, -1)[:, self.torso_gid] - self.default_torso_com
        tm = getattr(self.env, "termination_manager", None)
        flag = getattr(tm, "_delay_env_mask", torch.zeros(n, device=self.device))
        counter = getattr(tm, "_delay_counters", torch.zeros(n, device=self.device))
        return torch.cat((friction, self.robot.data.encoder_bias, com, flag[:, None].float(), counter[:, None].float()), -1)

    def command(self) -> torch.Tensor:
        return self.env.command_manager.get_command("twist")

    def kinematics(self, state: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Body poses/velocities in the established order, with all input gradients."""
        return self.analyze_state(state)

    def analyze_state(self, state: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Compute FK once for reuse by AMP and task reward within an update.

        The returned ``(positions, quaternions, linear_velocities,
        angular_velocities)`` tuple belongs to the caller's current graph. No
        state-dependent tensor is retained on the adapter or in a global cache.
        Arbitrary leading dimensions, including ``[T, B, 71]``, are supported.
        """
        prefix = state.shape[:-1]
        root_count = self._fk_root_count
        p = state[..., None, :3].expand(*prefix, root_count, 3)
        q = state[..., None, 3:7].expand(*prefix, root_count, 4)
        v = state[..., None, 7:10].expand(*prefix, root_count, 3)
        w = state[..., None, 10:13].expand(*prefix, root_count, 3)
        joints, speeds = state[..., 13:42], state[..., 42:71]
        ps, qs, vs, ws = [p], [q], [v], [w]
        for level in self._fk_levels:
            pp = p.index_select(-2, level.parent_slots)
            pq = q.index_select(-2, level.parent_slots)
            pv = v.index_select(-2, level.parent_slots)
            pw = w.index_select(-2, level.parent_slots)
            base = pp + quat_apply(pq, level.offsets)
            base_q = quat_mul(pq, level.orientations)
            angle = (joints.index_select(-1, level.joint_indices)-level.qrefs)*level.joint_mask
            speed = speeds.index_select(-1, level.joint_indices)*level.joint_mask
            q = quat_mul(base_q, quat_from_rotvec(angle[..., None]*level.axes))
            axis_speed = quat_apply(base_q, level.axes)*speed[..., None]
            w = pw+axis_speed
            if level.has_pivots:
                pivot = base+quat_apply(base_q, level.pivots)
                p = pivot-quat_apply(q, level.pivots)
                v = pv+torch.cross(pw, p-pp, dim=-1)+torch.cross(axis_speed, p-pivot, dim=-1)
            else:
                # G1 hinges are placed at their body origins. Both pivot
                # rotations and the hinge-origin velocity term are identically
                # zero, so avoiding them preserves values and derivatives.
                p = base
                v = pv+torch.cross(pw, p-pp, dim=-1)
            ps.append(p); qs.append(q); vs.append(v); ws.append(w)
        return tuple(torch.cat(xs, dim=-2).index_select(-2, self._fk_body_order)
                     for xs in (ps, qs, vs, ws))

    def actor_frame(self, state, previous_action, command, context, noise=False):
        q, w = state[..., 3:7], state[..., 10:13]
        sensor_q = quat_mul(q, self.imu_rotation)
        omega = quat_apply(quat_inv(sensor_q), w)
        gravity = quat_apply(quat_inv(q), state.new_tensor([0., 0., -1.]))
        joints = state[..., 13:42] - self.default_joint_pos
        speed = state[..., 42:71] - self.default_joint_vel
        if noise:
            omega = omega + torch.empty_like(omega).uniform_(-.2, .2)
            gravity = gravity + torch.empty_like(gravity).uniform_(-.05, .05)
            joints = joints + torch.empty_like(joints).uniform_(-.01, .01)
            speed = speed + torch.empty_like(speed).uniform_(-.5, .5)
        return torch.cat((omega, gravity, command, joints, speed, previous_action), -1)

    def amp_features(self, state, kinematics=None):
        """AMP features, optionally reusing FK from ``analyze_state(state)``."""
        p, q, v, w = self.kinematics(state) if kinematics is None else kinematics
        if self.amp_feature_schema != "legacy":
            return build_amp_features(p, q, v, w, body_ids=self.amp_ids,
                anchor_id=self.anchor_id, pelvis_id=self.pelvis_id,
                foot_ids=self.foot_ids, schema=self.amp_feature_schema)
        aq = q[..., self.anchor_id:self.anchor_id+1, :]
        ap = p[..., self.anchor_id:self.anchor_id+1, :]
        sel = self.amp_ids
        pos = quat_apply(quat_inv(aq), p[..., sel, :]-ap)
        ori = quat_to_matrix(quat_mul(quat_inv(aq), q[..., sel, :]))[..., :2]
        vel = quat_apply(quat_inv(q[..., sel, :]), v[..., sel, :])
        ang = quat_apply(quat_inv(q[..., sel, :]), w[..., sel, :])
        prefix = state.shape[:-1]
        return torch.cat((pos.reshape(*prefix, -1), ori.reshape(*prefix, -1), vel.reshape(*prefix, -1), ang.reshape(*prefix, -1)), -1)

    def critic_features(self, state, previous_action, command, context):
        h_inv = quat_inv(heading_quat(state[..., 3:7]))
        orientation = quat_mul(h_inv, state[..., 3:7])
        return torch.cat((state[..., 2:3], orientation, quat_apply(h_inv, state[..., 7:10]),
                          quat_apply(h_inv, state[..., 10:13]), state[..., 13:], previous_action, command, context), -1)

    def advance_context(self, next_state, context):
        """Advance the permanent recovery flag and consecutive bad-pose clock.

        V2 uses the real geometric predicate and integer-valued clock in forward.
        Only delayed termination uses a straight-through smooth hazard derivative;
        this is a biased optimization surrogate, not a physical Jacobian. The
        discrete clock is detached, so its value cannot shrink through soft bad
        probabilities or accumulate gradients over hundreds of steps. Ordinary
        non-delay termination retains its existing soft proxy. Timeouts remain
        separately captured real-data boundaries, outside this pose-only proxy.
        """
        gravity = quat_apply(quat_inv(next_state[..., 3:7]), next_state.new_tensor([0., 0., -1.]))
        bad_height = torch.sigmoid((self.height_threshold-next_state[..., 2])/.025)
        bad_tilt = torch.sigmoid((gravity[..., 2]+self.tilt_cos)/.025)
        bad = 1-(1-bad_height)*(1-bad_tilt)
        delay = context[..., 33]
        if self.recovery_counter_semantics == "hard_forward_v2":
            if self.max_delay_steps <= 0:
                # The physical manager leaves counters untouched when disabled.
                return context, bad
            with torch.no_grad():
                # Match strict simulator comparisons, including its acos form at
                # the orientation boundary; cos-threshold rounding can differ.
                hard_bad = ((next_state[..., 2] < self.height_threshold) |
                            (torch.acos(-gravity[..., 2]).abs() > self.tilt_angle))
                delayed = delay > .5
                candidate = torch.where(delayed & hard_bad, context[..., 34]+1,
                    torch.where(delayed, torch.zeros_like(context[..., 34]), context[..., 34]))
                ready = delayed & hard_bad & (candidate >= self.max_delay_steps)
                counter = torch.where(ready, torch.zeros_like(candidate), candidate)
            # Forward is exactly 0/1. The candidate clock is intentionally
            # detached in the smooth backward hazard to avoid counter*d(bad).
            hazard = bad * torch.sigmoid((candidate-self.max_delay_steps+.5)/.25)
            delayed_done = ready.to(bad.dtype) + (hazard-hazard.detach())
            terminated = (1-delay)*bad + delay*delayed_done
            return torch.cat((context[..., :34], counter[..., None]), -1), terminated
        # Preserve the full historical expression for explicit v1 resumes/A-Bs.
        counter = delay * (context[..., 34]+1) * bad
        delayed_done = bad * torch.sigmoid((counter-self.max_delay_steps+.5)/.25)
        terminated = (1-delay)*bad + delay*delayed_done
        counter = counter * (1-delayed_done)
        next_context = torch.cat((context[..., :34], counter[..., None]), -1)
        return next_context, terminated

    def configure_recovery_foot_slip(
        self, *, contact_height_m: float, softness_m: float, differentiate_support: bool = False,
    ) -> None:
        """Enable the imagined-only geometric foot-slip proxy for a flat task."""
        from .foot_slip import G1FootSlipGeometry

        terrain = getattr(getattr(self.env.cfg, "scene", None), "terrain", None)
        if getattr(terrain, "terrain_type", None) != "plane":
            raise ValueError("Imagined recovery foot-slip geometry requires a flat plane task")
        term = self.reward_cfg.get("foot_slip")
        if (term is None or float(term.weight) != -.25
                or term.params.get("command_name") != "twist"
                or float(term.params.get("command_threshold", -1)) != .1
                or tuple(getattr(term.params.get("asset_cfg"), "site_names", ())) != ("left_foot", "right_foot")):
            raise ValueError("Imagined recovery foot-slip geometry requires the original G1 foot_slip reward")
        self.recovery_foot_slip = G1FootSlipGeometry.from_model(
            self.env.sim.mj_model, self.body_ids, self.foot_ids, device=self.device,
            contact_height_m=contact_height_m, softness_m=softness_m,
            command_threshold=term.params["command_threshold"], differentiate_support=differentiate_support,
        )

    def task_reward(self, state, next_state, action, previous_action, command,
                    next_context, terminated, contact_cost, kinematics=None, *, current_context=None):
        """Differentiable task reward; finite-difference qacc is a model proxy.

        The same expression is used by imagined targets and imagined actor
        updates. The optional flat recovery slip geometry changes only that
        reward component, leaving the collision and locomotion costs learned.
        """
        pos, quats, velocities, angular = self.kinematics(next_state) if kinematics is None else kinematics
        torso = self.anchor_id
        tq, tv, tw = quats[..., torso, :], velocities[..., torso, :], angular[..., torso, :]
        target_v = quat_apply(heading_quat(tq), torch.cat((command[..., :2], torch.zeros_like(command[..., :1])), -1))
        tw_body = quat_apply(quat_inv(tq), tw)
        root_w = quat_apply(quat_inv(next_state[..., 3:7]), next_state[..., 10:13])
        active_delay = next_context[..., 33] * next_context[..., 34].clamp(0., 1.)
        value = torch.zeros_like(next_state[..., 0])
        def add(name, raw):
            nonlocal value
            cfg = self.reward_cfg.get(name)
            if cfg is not None:
                value = value + float(cfg.weight)*raw
        def scaled_delay(name, raw, only=False):
            cfg = self.reward_cfg.get(name)
            if cfg is None:
                return torch.zeros_like(raw)
            mask = bool(cfg.params.get("mask_delay", False))
            ratio = float(cfg.params.get("delay_env_rew_ratio", 1.))
            if only:
                return raw*active_delay*ratio if mask else torch.zeros_like(raw)
            return raw*(1-active_delay+ratio*active_delay) if mask else raw
        cfg = self.reward_cfg["track_anchor_linear_velocity"]
        add("track_anchor_linear_velocity", scaled_delay("track_anchor_linear_velocity", torch.exp(-(target_v-tv).square().sum(-1)/float(cfg.params["std"])**2)))
        cfg = self.reward_cfg["track_anchor_angular_velocity"]
        err = (command[..., 2]-tw[..., 2]).square()+tw_body[..., :2].square().sum(-1)
        add("track_anchor_angular_velocity", scaled_delay("track_anchor_angular_velocity", torch.exp(-err/float(cfg.params["std"])**2)))
        cfg = self.reward_cfg["track_root_height"]
        upright_power = float(cfg.params.get("upright_power", 0.))
        height_reward = recovery_height_reward(next_state[..., 2], self.desired_height,
            float(cfg.params["std"]), upright_power=upright_power,
            up_z=recovery_root_up_z(next_state[..., 3:7]) if upright_power else None)
        add("track_root_height", scaled_delay("track_root_height", height_reward, only=True))
        cfg = self.reward_cfg["body_ang_vel_xy_l2"]
        add("body_ang_vel_xy_l2", scaled_delay("body_ang_vel_xy_l2", torch.exp(-root_w[..., :2].square().sum(-1)/float(cfg.params["std"])**2)))
        add("is_terminated", terminated.float())
        add("joint_acc_l2", ((next_state[..., 42:71]-state[..., 42:71])/self.dt).square().sum(-1))
        joints = next_state[..., 13:42]
        excess = (self.joint_limits[:, 0]-joints).clamp_min(0)+(joints-self.joint_limits[:, 1]).clamp_min(0)
        add("joint_pos_limits", excess.sum(-1))
        add("action_rate_l2", (action-previous_action).square().sum(-1))
        foot_slip = contact_cost[..., 0]
        geometry = getattr(self, "recovery_foot_slip", None)
        if geometry is not None:
            if current_context is None:
                raise ValueError("Geometric recovery foot slip requires the pre-transition context")
            foot_slip = torch.where(recovery_foot_slip_mask(state, current_context),
                                    geometry((pos, quats, velocities, angular), command), foot_slip)
        add("foot_slip", foot_slip)
        add("self_collisions", contact_cost[..., 1])
        return value*self.reward_scale
