"""CPU correctness and operation-count checks for depth-batched differentiable FK.

Run with the existing training environment, without allocating a GPU:
    python -m unittest discover -s tests -p test_wmac_state_fast.py -v
Optional repeatable CPU timing:
    python tests/test_wmac_state_fast.py --benchmark
"""

from __future__ import annotations

import gc
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import unittest
import weakref

import mujoco
import numpy as np
import torch

from src.wmac.state import (
    StateAdapter, quat_apply, quat_from_rotvec, quat_inv, quat_mul, quat_normalize,
)


def make_adapter(nonzero_pivot=False, dtype=torch.float64):
    """Construct the actual G1 MuJoCo model and only the CPU env metadata needed."""
    xml = Path(__file__).resolve().parents[1]/"src/assets/robots/unitree_g1/xmls/g1.xml"
    model = mujoco.MjModel.from_xml_path(str(xml))
    if nonzero_pivot:
        # Exercise the general pivot branch as well as G1's zero-pivot fast path.
        model.jnt_pos[3] = [.003, -.002, .005]
    body_ids = np.arange(1, model.nbody)
    joint_ids = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE)
    default_q = torch.tensor(model.qpos0[model.jnt_qposadr[joint_ids]], dtype=torch.float32)[None]
    limits = torch.tensor(model.jnt_range[joint_ids], dtype=torch.float32)[None]
    root = torch.zeros(1, 13)
    root[0, :7] = torch.tensor(model.qpos0[:7], dtype=torch.float32)
    indexing = SimpleNamespace(
        body_ids=torch.tensor(body_ids), joint_ids=torch.tensor(joint_ids),
        geom_ids=torch.arange(model.ngeom),
        geoms=[SimpleNamespace(id=i, name=model.geom(i).name) for i in range(model.ngeom)],
        sites=[SimpleNamespace(id=i, name=model.site(i).name) for i in range(model.nsite)],
    )
    robot = SimpleNamespace(
        body_names=tuple(model.body(i).name for i in body_ids), indexing=indexing,
        data=SimpleNamespace(default_joint_pos=default_q, default_joint_vel=torch.zeros(1, 29),
                             soft_joint_pos_limits=limits, default_root_state=root),
    )
    def term(weight, **params):
        return SimpleNamespace(weight=weight, params=params)
    rewards = {
        "track_anchor_linear_velocity": term(1., std=1., mask_delay=True, delay_env_rew_ratio=0.),
        "track_anchor_angular_velocity": term(1., std=3.14, mask_delay=True, delay_env_rew_ratio=0.),
        "track_root_height": term(1., std=.3, mask_delay=True, delay_env_rew_ratio=3.5),
        "body_ang_vel_xy_l2": term(.5, std=3.14, mask_delay=True, delay_env_rew_ratio=0.),
        "is_terminated": term(-200.), "joint_acc_l2": term(-2.5e-7),
        "joint_pos_limits": term(-10.), "action_rate_l2": term(-.01),
        "foot_slip": term(-.25), "self_collisions": term(-.1),
    }
    env = SimpleNamespace(
        scene={"robot": robot}, device="cpu", step_dt=.02,
        sim=SimpleNamespace(mj_model=model),
        cfg=SimpleNamespace(rewards=rewards, scale_rewards_by_dt=True, terminations={
            "bad_base_height": term(0., minimum_height=.5),
            "bad_orientation": term(0., limit_angle=np.deg2rad(70.)),
        }), termination_manager=SimpleNamespace(_max_delay_steps=250),
    )
    adapter = StateAdapter(env)
    # Production physics is float32; double constants make the CPU value and
    # derivative comparison more sensitive to algorithmic differences.
    for name, value in vars(adapter).items():
        if isinstance(value, torch.Tensor) and value.is_floating_point():
            setattr(adapter, name, value.to(dtype))
    adapter._fk_levels = tuple(replace(level, **{
        name: getattr(level, name).to(dtype)
        for name in ("offsets", "orientations", "axes", "pivots", "qrefs")
    }) for level in adapter._fk_levels)
    return adapter, model


def legacy_kinematics(adapter, state):
    """Independent body-by-body oracle retained from the pre-optimization code."""
    p, q, v, w = state[..., :3], state[..., 3:7], state[..., 7:10], state[..., 10:13]
    joints, speeds = state[..., 13:42], state[..., 42:71]
    ps, qs, vs, ws = [], [], [], []
    for i, parent in enumerate(adapter.parents):
        if parent < 0:
            pi, qi, vi, wi = p, q, v, w
        else:
            pp, pq, pv, pw = ps[parent], qs[parent], vs[parent], ws[parent]
            base = pp+quat_apply(pq, adapter.offsets[i])
            base_q = quat_mul(pq, adapter.orientations[i])
            ji = adapter.joint_indices[i]
            if ji >= 0:
                angle = joints[..., ji]-adapter.qrefs[i]
                qi = quat_mul(base_q, quat_from_rotvec(angle[..., None]*adapter.axes[i]))
                pivot = base+quat_apply(base_q, adapter.pivots[i])
                pi = pivot-quat_apply(qi, adapter.pivots[i])
                axis = quat_apply(base_q, adapter.axes[i])
                wi = pw+axis*speeds[..., ji:ji+1]
                vi = pv+torch.cross(pw, pi-pp, dim=-1)+torch.cross(axis*speeds[..., ji:ji+1], pi-pivot, dim=-1)
            else:
                pi, qi, wi = base, base_q, pw
                vi = pv+torch.cross(pw, pi-pp, dim=-1)
        ps.append(pi);qs.append(qi);vs.append(vi);ws.append(wi)
    return tuple(torch.stack(xs, dim=-2) for xs in (ps, qs, vs, ws))


def sample_state(prefix=(8,), dtype=torch.float64):
    generator = torch.Generator().manual_seed(612)
    state = torch.randn(*prefix, 71, generator=generator, dtype=dtype)*.2
    state[..., 2] += .65
    state[..., 3:7] = quat_normalize(torch.randn(*prefix, 4, generator=generator, dtype=dtype))
    # Explicitly cover the previously problematic vertical-forward recovery pose,
    # identity, and a 180-degree side/back-facing pose.
    flat = state.reshape(-1, 71)
    flat[0, 3:7] = state.new_tensor([.5, .5, .5, -.5])
    if len(flat) > 1:
        flat[1, 3:7] = state.new_tensor([1., 0., 0., 0.])
    if len(flat) > 2:
        flat[2, 3:7] = state.new_tensor([0., 0., 1., 0.])
    return state


def check_depth_fk_values_and_gradients_match_body_oracle(prefix, nonzero_pivot):
    adapter, _ = make_adapter(nonzero_pivot)
    state = sample_state(prefix).requires_grad_(True)
    expected = legacy_kinematics(adapter, state)
    actual = adapter.kinematics(state)
    generator = torch.Generator().manual_seed(14)
    weights = [torch.randn(x.shape, generator=generator, dtype=x.dtype) for x in actual]
    for new, old in zip(actual, expected):
        torch.testing.assert_close(new, old, rtol=1e-10, atol=1e-11)
    old_loss = sum((x*weight).mean() for x, weight in zip(expected, weights))
    new_loss = sum((x*weight).mean() for x, weight in zip(actual, weights))
    old_grad = torch.autograd.grad(old_loss, state)[0]
    new_grad = torch.autograd.grad(new_loss, state)[0]
    assert torch.isfinite(new_grad).all()
    torch.testing.assert_close(new_grad, old_grad, rtol=1e-9, atol=1e-10)


def check_shared_amp_and_reward_preserve_values_and_all_input_gradients(prefix):
    adapter, _ = make_adapter()
    state = sample_state(prefix).requires_grad_(True)
    next_state = sample_state(prefix).add(.01).requires_grad_(True)
    # Keep quaternion normalized in the sampled state, without detaching its graph.
    next_state = torch.cat((next_state[..., :3], quat_normalize(next_state[..., 3:7]), next_state[..., 7:]), -1)
    action = torch.full((*prefix, 29), .03, dtype=torch.float64, requires_grad=True)
    previous = torch.full_like(action, .01, requires_grad=True)
    command = torch.full((*prefix, 3), .05, dtype=torch.float64, requires_grad=True)
    context = torch.zeros(*prefix, 35, dtype=torch.float64)
    context[..., 33] = 1.;context[..., 34] = .6
    context.requires_grad_(True)
    terminated = torch.full(prefix, .1, dtype=torch.float64, requires_grad=True)
    contact = torch.full((*prefix, 2), .02, dtype=torch.float64, requires_grad=True)
    inputs = (state, next_state, action, previous, command, context, terminated, contact)
    old_fk = legacy_kinematics(adapter, next_state)
    new_fk = adapter.analyze_state(next_state)
    old_amp = adapter.amp_features(next_state, kinematics=old_fk)
    old_reward = adapter.task_reward(*inputs, kinematics=old_fk)
    # Passing cached FK must bypass both compatibility and computation entrypoints.
    with patch.object(adapter, "kinematics", side_effect=AssertionError("duplicate FK")), \
         patch.object(adapter, "analyze_state", side_effect=AssertionError("duplicate FK")):
        new_amp = adapter.amp_features(next_state, kinematics=new_fk)
        new_reward = adapter.task_reward(*inputs, kinematics=new_fk)
    torch.testing.assert_close(new_amp, old_amp, rtol=1e-10, atol=1e-11)
    torch.testing.assert_close(new_reward, old_reward, rtol=1e-10, atol=1e-11)
    old_loss = old_reward.sum()+old_amp.square().mean()
    new_loss = new_reward.sum()+new_amp.square().mean()
    old_grads = torch.autograd.grad(old_loss, inputs, retain_graph=True)
    new_grads = torch.autograd.grad(new_loss, inputs)
    for new, old in zip(new_grads, old_grads):
        assert torch.isfinite(new).all()
        torch.testing.assert_close(new, old, rtol=1e-8, atol=1e-9)


def check_fk_matches_real_mujoco_on_cpu(nonzero_pivot):
    adapter, model = make_adapter(nonzero_pivot)
    states = sample_state((5,))
    p, q, v, w = adapter.analyze_state(states)
    data = mujoco.MjData(model)
    ids = adapter.body_ids.numpy()
    for i, state in enumerate(states):
        data.qpos[:7] = state[:7].numpy()
        data.qpos[7:] = state[13:42].numpy()
        data.qvel[:3] = state[7:10].numpy()
        data.qvel[3:6] = quat_apply(quat_inv(state[3:7]), state[10:13]).numpy()
        data.qvel[6:] = state[42:71].numpy()
        mujoco.mj_forward(model, data)
        root_id = int(ids[0])
        angular = data.cvel[ids, :3]
        linear = data.cvel[ids, 3:]-np.cross(angular, data.subtree_com[root_id]-data.xpos[ids])
        np.testing.assert_allclose(p[i].numpy(), data.xpos[ids], rtol=3e-6, atol=3e-6)
        # Quaternions q and -q describe the same orientation.
        difference = np.minimum(np.max(np.abs(q[i].numpy()-data.xquat[ids]), -1),
                                np.max(np.abs(q[i].numpy()+data.xquat[ids]), -1))
        assert difference.max() < 3e-6
        np.testing.assert_allclose(v[i].numpy(), linear, rtol=3e-6, atol=3e-6)
        np.testing.assert_allclose(w[i].numpy(), angular, rtol=3e-6, atol=3e-6)


def check_analysis_has_no_persistent_state_graph_cache():
    adapter, _ = make_adapter()
    state = sample_state((2,)).requires_grad_(True)
    first = adapter.analyze_state(state)
    second = adapter.analyze_state(state)
    assert all(a is not b for a, b in zip(first, second))
    reference = weakref.ref(state)
    del state, first, second
    gc.collect()
    assert reference() is None


class FastStateTest(unittest.TestCase):
    def test_depth_fk_values_and_gradients_match_body_oracle(self):
        for prefix in [(), (7,), (3, 4)]:
            for nonzero_pivot in [False, True]:
                with self.subTest(prefix=prefix, nonzero_pivot=nonzero_pivot):
                    check_depth_fk_values_and_gradients_match_body_oracle(prefix, nonzero_pivot)

    def test_shared_amp_and_reward_preserve_values_and_all_input_gradients(self):
        for prefix in [(6,), (2, 3)]:
            with self.subTest(prefix=prefix):
                check_shared_amp_and_reward_preserve_values_and_all_input_gradients(prefix)

    def test_fk_matches_real_mujoco_on_cpu(self):
        for nonzero_pivot in [False, True]:
            with self.subTest(nonzero_pivot=nonzero_pivot):
                check_fk_matches_real_mujoco_on_cpu(nonzero_pivot)

    def test_analysis_has_no_persistent_state_graph_cache(self):
        check_analysis_has_no_persistent_state_graph_cache()


def benchmark():
    """Report CPU latency and dispatcher counts without initializing CUDA."""
    from torch.utils.benchmark import Timer
    from torch.utils._python_dispatch import TorchDispatchMode

    class CountOperations(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.count = 0

        def __torch_dispatch__(self, operation, types, args=(), kwargs=None):
            self.count += 1
            return operation(*args, **(kwargs or {}))

    adapter, _ = make_adapter(dtype=torch.float32)
    state = sample_state((256,), torch.float32)
    print("Body count:", len(adapter.parents), "levels:", adapter.kinematics_level_sizes)
    def backward(legacy):
        inputs = state.detach().requires_grad_(True)
        outputs = legacy_kinematics(adapter, inputs) if legacy else adapter.analyze_state(inputs)
        torch.autograd.grad(sum(value.square().mean() for value in outputs), inputs)

    variants = {
        "body_loop_fk": lambda: legacy_kinematics(adapter, state),
        "depth_batched_fk": lambda: adapter.analyze_state(state),
        "legacy_two_fk": lambda: (legacy_kinematics(adapter, state), legacy_kinematics(adapter, state)),
        "shared_one_fk": lambda: adapter.analyze_state(state),
        "body_loop_fk_forward_backward": lambda: backward(True),
        "depth_batched_fk_forward_backward": lambda: backward(False),
    }
    for name, run in variants.items():
        timing = Timer(stmt="run()", globals={"run": run}, num_threads=1).blocked_autorange(min_run_time=.2)
        with CountOperations() as operations:
            run()
        print(f"{name}: {timing.median*1000:.3f} ms, {operations.count} CPU ATen dispatches")


if __name__ == "__main__":
    import sys
    if "--benchmark" in sys.argv:
        benchmark()
    else:
        unittest.main()
