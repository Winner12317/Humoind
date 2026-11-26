"""Visual gradient boundaries against hash-verified Instinct Parkour source.

The source oracle executes upstream functions or statements, rather than a
second local implementation. All fixtures are small CPU models; no simulator,
renderer or training process is created. Reference checks skip if the optional
pinned checkout is absent (INSTINCTMJ_REFERENCE_CHECKOUT overrides its path).
"""
from __future__ import annotations

import ast
import copy
from dataclasses import replace
import hashlib
import os
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.wmac.networks import frozen_parameters
from src.wmac_perceptive.compiled_imagination import LatentImaginedStep
from src.wmac_perceptive.latent_execution import LatentWorldModelKernel
from src.wmac_perceptive.latent_models import LatentPerceptiveActor, LatentPerceptiveWorldModel
from src.wmac_perceptive.latent_perception import processed_stack_shape
from test_perceptive_latent_execution import fixture
from test_perceptive_latent_runner import latent_runner_fixture
from test_perceptive_models import PlaneOracle, SMALL, state_batch


REFERENCE_COMMIT = "03563ee02665f327425106372264070315673748"
PARKOUR_RUNNER_SHA256 = "1855c0dbc32005cd7087fb16e52d09cb1c4da2b9f512c838741c50181f07c8bb"


def reference_class():
    root = Path(os.environ.get("INSTINCTMJ_REFERENCE_CHECKOUT", "/tmp/instinctmj-supermimic-20261111"))
    path = root / "src/instinct_mj/rl/supermimic/parkour/runner.py"
    if not path.is_file():
        raise unittest.SkipTest("Optional pinned InstinctMJ Parkour checkout is absent")
    source = path.read_bytes()
    if hashlib.sha256(source).hexdigest() != PARKOUR_RUNNER_SHA256:
        raise AssertionError("Pinned InstinctMJ Parkour runner.py source hash differs")
    return next(node for node in ast.parse(source, filename=str(path)).body
                if isinstance(node, ast.ClassDef) and node.name == "ParkourSuperMimicRunner")


def method(name):
    return copy.deepcopy(next(node for node in reference_class().body
                              if isinstance(node, ast.FunctionDef) and node.name == name))


def execute(nodes, **globals_):
    namespace = {"torch": torch, "nn": nn, **globals_}
    tree = ast.fix_missing_locations(ast.Module(body=copy.deepcopy(nodes), type_ignores=[]))
    exec(compile(tree, "<hash-verified InstinctMJ Parkour source>", "exec"), namespace)
    return namespace


def assignment_to(node, name):
    if not isinstance(node, ast.Assign):
        return False
    return any((isinstance(target, ast.Name) and target.id == name)
               or (isinstance(target, ast.Attribute) and target.attr == name)
               for target in node.targets)


def source_reconstruction(runner, batch):
    names = ("Br", "stack", "recon")
    statements = [node for node in method("wm_update").body
                  if any(assignment_to(node, name) for name in names)]
    if len(statements) != 3:
        raise AssertionError("Pinned upstream reconstruction statements changed")
    oracle = SimpleNamespace(encode_depth=runner.actor.encode_processed, latent_dec=runner.latent_decoder)
    return execute(statements, self=oracle, w={"depth1": batch["next_depth_stack"] * 255.},
                   c={"latent_recon_images": runner.cfg.latent_recon_images},
                   Bz=len(batch["mask"]), k=batch["mask"].shape[1], DEPTH_SCALE=255.)["recon"]


class ReferenceVisualFitGradientTest(unittest.TestCase):
    def test_wm_optimizer_owns_encoder_and_only_reconstruction_trains_it(self):
        runner, batch = fixture()
        batch["mask"].fill_(1.)  # Upstream and local image-selection rules coincide here.
        nodes = [node for node in method("setup_training").body
                 if assignment_to(node, "wm_params") or assignment_to(node, "opt_wm")]
        self.assertEqual(len(nodes), 2)
        holder = SimpleNamespace(wm=runner.world_model, enc=runner.actor.vision,
                                 latent_dec=runner.latent_decoder)
        execute(nodes, self=holder, c={"lr": .001})
        expected = {id(p) for group in holder.opt_wm.param_groups for p in group["params"]}
        actual = {id(p) for group in runner.opt_model.param_groups for p in group["params"]}
        policy = {id(p) for group in runner.opt_actor.param_groups for p in group["params"]}
        encoder = {id(p) for p in runner.actor.vision.parameters()}
        self.assertEqual(actual, expected)
        self.assertTrue(encoder <= actual)
        self.assertFalse(encoder & policy)

        # Re-use representable uint8 depths to avoid introducing a scale-rounding
        # difference between upstream's /255 and already-normalized local input.
        for key in ("depth_stack", "next_depth_stack"):
            batch[key] = (batch[key] * 255).round().float() / 255
        kernel = LatentWorldModelKernel(runner)
        prepared = kernel.prepare(batch, update_statistics=False)
        total, _ = kernel(prepared)
        expected_recon = source_reconstruction(runner, batch)
        parameters = tuple(runner.actor.vision.parameters()) + tuple(runner.latent_decoder.parameters())
        actual_grad = torch.autograd.grad(total, parameters)
        expected_grad = torch.autograd.grad(expected_recon, parameters)
        for actual, expected in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual, expected, rtol=3e-5, atol=2e-7)
        self.assertGreater(sum(float(grad.abs().sum()) for grad in actual_grad), 0.)

    def test_wm_start_and_future_targets_match_upstream_no_grad_encoding(self):
        runner, batch = fixture()
        for key in ("depth_stack", "next_depth_stack"):
            batch[key] = ((batch[key] * 255).round().float() / 255).requires_grad_()
        block = next(node for node in method("wm_update").body if isinstance(node, ast.With))
        holder = SimpleNamespace(encode_depth=runner.actor.encode_processed, wm=runner.world_model)
        modes = []
        with runner.actor.vision.conv[0].register_forward_pre_hook(
                lambda *args: modes.append(torch.is_grad_enabled())):
            actual = LatentWorldModelKernel(runner).prepare(batch, update_statistics=False)
            oracle = execute([block], self=holder,
                w={"depth": batch["depth_stack"] * 255., "depth1": batch["next_depth_stack"] * 255.,
                   "mask": batch["mask"]}, Bz=len(batch["mask"]), k=batch["mask"].shape[1], DEPTH_SCALE=255.)
        self.assertEqual(modes, [False] * 4)
        for key, reference_key in (("initial_latent", "z0"), ("target_latent", "z_tgt")):
            self.assertFalse(actual[key].requires_grad)
            self.assertIsNone(actual[key].grad_fn)
            self.assertFalse(oracle[reference_key].requires_grad)
            torch.testing.assert_close(actual[key], oracle[reference_key], rtol=2e-6, atol=1e-7)


def dynamics_fixture():
    """Make the ONLY action-to-task path go through predicted z then physics."""
    torch.manual_seed(882)
    spec = replace(SMALL, include_valid=False)
    actor = LatentPerceptiveActor(spec)
    actor.policy = nn.Linear(spec.proprio_dim + 128, 29)
    world = LatentPerceptiveWorldModel(PlaneOracle(slope=0.), spec, hidden=(8,))
    world.net = nn.Linear(world.in_dim + 128, 35 + 2 + 128)
    with torch.no_grad():
        actor.policy.weight.fill_(.001)
        actor.policy.bias.fill_(.02)
        world.net.weight.zero_()
        world.net.bias.zero_()
        world.net.weight[0, -128] = .4  # z[0] -> next root velocity, via GenCoord integration.
        world.net.weight[37, 68] = .2  # first action -> predicted z[0]; no direct physical action path.
    count = 2
    state = state_batch(count).float()
    stack = torch.rand(count, *processed_stack_shape(spec))
    history = torch.zeros(count, 4, 96)
    context = torch.zeros(count, 35)
    target = state[:, :2].clone() + state.new_tensor((1., 0.))
    speed = torch.full((count,), .3)
    frame = lambda s, a, *_: torch.cat((s[..., :67], a), -1)
    adapter = SimpleNamespace(advance_context=lambda s, ctx: (ctx, s.new_zeros(s.shape[0])), actor_frame=frame)
    runner = SimpleNamespace(actor=actor, world_model=world, adapter=adapter,
                             cfg=SimpleNamespace(action_limit=20., locomotion_objective="legacy_walk_turn_v1"))
    return runner, state, stack, history, context, target, speed


def source_rollout(runner, state, stack, history, context, target, speed):
    shape = processed_stack_shape(runner.actor.spec)
    mdp = SimpleNamespace(scan=lambda s: runner.world_model.terrain_oracle.local_patch(s),
        policy_obs=lambda h, image: torch.cat((h.flatten(1), image.flatten(1)), -1),
        advance_feet_timers=lambda timers, contacts: timers,
        add_proprio_noise=lambda value: value,
        proprio=lambda s, a, cmd: runner.adapter.actor_frame(s, a, cmd, context),
        amp_frame=lambda s: s[:, :1])
    holder = SimpleNamespace(cfg={"latent_policy_grad": 0., "checkpoint_imagination": False},
        horizon=2, encode_depth=runner.actor.encode_processed, mdp=mdp,
        fr_n=torch.zeros(len(state), 1), depth_dim=shape[0] * shape[1] * shape[2],
        history=lambda _: (history, torch.zeros(len(state), 2, 1)),
        episode_context=lambda lags, fr: lags)
    start = execute([method("start_latent")])["start_latent"]
    holder.start_latent = MethodType(start, holder)
    core = next(node for node in method("_build_fns").body
                if isinstance(node, ast.FunctionDef) and node.name == "_imag_core")
    wm = SimpleNamespace(step=lambda s, a, ctx, scan, z:
                         runner.world_model.step_with_latent(s, a, ctx, z))
    namespace = execute([core, method("_imagine")], self=holder, mdp=mdp, wm=wm,
        ac=SimpleNamespace(actor=runner.actor.policy), normalize=lambda value: value,
        policy_autocast=False, d0=runner.actor.spec.proprio_dim, DEPTH_SHAPE=shape,
        DEPTH_SCALE=255., unflatten_state=lambda value: value)
    holder.imag_core = namespace["_imag_core"]
    starts = {"s": state, "cmd": torch.zeros(len(state), 1), "env": torch.arange(len(state)),
              "lags": context[:, :6], "depth": stack * 255., "timers": torch.zeros(len(state), 4),
              "a_prev": torch.zeros(len(state), 29)}
    return namespace["_imagine"](holder, starts, {})


class ReferenceImaginedVisualGradientTest(unittest.TestCase):
    def test_actor_detaches_z_but_live_latent_prediction_carries_future_physics_gradient(self):
        runner, state, stack, history, context, target, speed = dynamics_fixture()
        original = runner.world_model.step_with_latent
        source_predictions = []

        def observe(*args):
            result = original(*args)
            source_predictions.append(result)
            return result

        with frozen_parameters(runner.world_model), patch.object(runner.world_model, "step_with_latent", side_effect=observe):
            expected = source_rollout(runner, state, stack, history, context, target, speed)
        self.assertEqual(len(source_predictions), 2)
        with torch.no_grad():
            latent = runner.actor.encode_processed(stack)
        step = LatentImaginedStep(runner)
        predictions, actions = [], []
        with frozen_parameters(runner.world_model):
            for _ in range(2):
                state, context, history, latent, action, _, _, _ = step(
                    state, context, history, latent, target, speed, torch.zeros(len(state), 96))
                predictions.append((state, latent))
                actions.append(action)
        for i in range(2):
            torch.testing.assert_close(predictions[i][0], expected["Ss"][i], rtol=2e-6, atol=1e-7)
            torch.testing.assert_close(predictions[i][1], source_predictions[i][2], rtol=2e-6, atol=1e-7)
            torch.testing.assert_close(actions[i], expected["As"][i], rtol=2e-6, atol=1e-7)
        for final, first_z, first_action, second_action in (
            (predictions[-1][0], predictions[0][1], actions[0], actions[1]),
            (expected["Ss"][-1], source_predictions[0][2], expected["As"][0], expected["As"][1]),
        ):
            task = final[:, 0].sum()
            dz, da = torch.autograd.grad(task, (first_z, first_action), retain_graph=True)
            self.assertGreater(float(dz.abs().sum()), 0.)
            self.assertGreater(float(da.abs().sum()), 0.)
            blocked, = torch.autograd.grad(second_action.sum(), first_z, allow_unused=True, retain_graph=True)
            self.assertIsNone(blocked)
            encoder_grad = torch.autograd.grad(task, tuple(runner.actor.vision.parameters()),
                                               allow_unused=True, retain_graph=True)
            self.assertTrue(all(grad is None for grad in encoder_grad))
        actual_grad = torch.autograd.grad(predictions[-1][0][:, 0].sum(), tuple(runner.actor.policy.parameters()))
        expected_grad = torch.autograd.grad(expected["Ss"][-1][:, 0].sum(), tuple(runner.actor.policy.parameters()))
        for actual, reference in zip(actual_grad, expected_grad):
            torch.testing.assert_close(actual, reference, rtol=2e-6, atol=1e-8)
        self.assertTrue(all(parameter.grad is None for parameter in runner.actor.vision.parameters()))
        self.assertGreater(sum(float(grad.abs().sum()) for grad in actual_grad), 0.)

    def test_terminal_bootstrap_keeps_same_state_gradient_as_upstream(self):
        runner, batch, noise = latent_runner_fixture(batch_size=2, horizon=2)
        runner.cfg.task_weight = runner.cfg.style_weight = 0.
        runner.cfg.critic_bootstrap = True
        runner._value = lambda features: .01 * features.square().sum(-1)
        features, original = [], runner.adapter.critic_features

        def observe(*args, **kwargs):
            value = original(*args, **kwargs)
            if torch.is_grad_enabled() and value.requires_grad:
                features.append(value)
            return value

        with frozen_parameters(runner.world_model, runner.discriminator, runner.critic), \
                patch.object(runner.adapter, "critic_features", side_effect=observe):
            actual = runner.imagine(batch, noises=noise)
        feature_graph = features[-1]
        terminal = feature_graph[-1] if feature_graph.ndim == 3 else feature_graph
        self.assertTrue(terminal.requires_grad)
        update = method("policy_update")
        block = next(node for node in update.body if isinstance(node, ast.If)
                     and any(assignment_to(child, "J") for child in node.body))
        self.assertEqual(len(block.body), 1)
        holder = SimpleNamespace(critic=runner._value, critic_input=lambda *args: terminal)
        expected = execute(block.body, self=holder, J=torch.zeros(len(batch["state"])),
            c={"gamma": runner.cfg.gamma}, HZ=2, cT=actual["survival"],
            im={"S": None, "timers": None}, cmd=None, xi_n=None)["J"]
        torch.testing.assert_close(actual["objective"], expected)
        actual_grad, = torch.autograd.grad(actual["objective"].sum(), feature_graph, retain_graph=True)
        expected_grad, = torch.autograd.grad(expected.sum(), feature_graph)
        torch.testing.assert_close(actual_grad, expected_grad)
        self.assertGreater(float(actual_grad.abs().sum()), 0.)


if __name__ == "__main__":
    unittest.main()
