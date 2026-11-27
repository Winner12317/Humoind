"""Simulator and learner on different devices (``--sim-device``).

The learner imagines with a copy of the environment adapter's constants. The
copy must compute exactly what the environment's adapter computes, share
nothing mutable with it, and leave live readings to the environment.
"""
from __future__ import annotations

from collections import namedtuple
from dataclasses import asdict, dataclass
import unittest

import torch

from scripts.train_wmac_perceptive import assign_devices, make_environment
from src.wmac.device_placement import LIVE_ADAPTER_READS, learner_adapter, relocate, to_device
from src.wmac_perceptive.config import FORMAT, PerceptiveConfig, config_from_payload
from src.wmac_perceptive.checkpoints import MUTABLE_RESUME_FIELDS

Pair = namedtuple("Pair", "first second")


@dataclass(frozen=True)
class Level:
    ids: torch.Tensor
    scale: float


class Helper:
    def __init__(self):
        self.table = torch.arange(6.).reshape(2, 3)
        self.levels = [Level(torch.tensor([1, 2]), 2.), Level(torch.tensor([3]), 3.)]


class Owner:
    def __init__(self, environment):
        self.env = environment
        self.device = torch.device("cpu")
        self.weights = torch.ones(3)
        self.alias = self.weights
        self.helper = Helper()
        self.pair = Pair(torch.zeros(2), "label")
        self.options = {"gain": torch.tensor(2.), "names": ("a", "b")}
        self.myself = self


# relocate() descends only into classes of this project.
for _cls in (Level, Helper, Owner):
    _cls.__module__ = "src.tests.fake_adapter"


class Foreign:
    """Stands for an mjlab object: it must be shared, never copied."""

    def __init__(self):
        self.buffer = torch.zeros(1)


def all_tensors(value, seen=None):
    seen = set() if seen is None else seen
    if id(value) in seen:
        return []
    seen.add(id(value))
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, dict):
        return [t for item in value.values() for t in all_tensors(item, seen)]
    if isinstance(value, (list, tuple)):
        return [t for item in value for t in all_tensors(item, seen)]
    if type(value).__module__.startswith("src.") and hasattr(value, "__dict__") and not isinstance(value, type):
        return [t for name, item in vars(value).items() if name not in ("env", "robot") for t in all_tensors(item, seen)]
    return []


class RelocateTest(unittest.TestCase):
    def test_structure_is_copied_and_everything_else_shared(self):
        environment, foreign = Owner(None), Foreign()
        owner = Owner(environment)
        owner.foreign = foreign
        moved = relocate(owner, torch.device("cpu"), source="cpu", shared=(environment,))
        self.assertIsNot(moved, owner)
        self.assertIs(moved.env, environment)              # explicitly shared, although a project class
        self.assertIs(moved.foreign, foreign)              # not a project class
        self.assertIsNot(moved.helper, owner.helper)
        self.assertIsNot(moved.helper.levels, owner.helper.levels)
        self.assertIsNot(moved.helper.levels[0], owner.helper.levels[0])
        self.assertIsInstance(moved.pair, Pair)
        self.assertEqual(moved.pair.second, "label")
        self.assertIsNot(moved.options, owner.options)
        self.assertEqual(moved.options["names"], ("a", "b"))
        self.assertIs(moved.myself, moved)                 # cycles keep their shape
        self.assertIs(moved.alias, moved.weights)          # and so does aliasing
        self.assertEqual(moved.helper.levels[1].scale, 3.)
        torch.testing.assert_close(moved.helper.table, owner.helper.table)
        # New attributes on the copy do not appear on the original.
        moved.helper.extra = 1
        self.assertFalse(hasattr(owner.helper, "extra"))

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
    def test_every_tensor_and_device_attribute_moves(self):
        owner = Owner(Owner(None))
        moved = relocate(owner, torch.device("cuda:0"), source="cpu", shared=(owner.env,))
        self.assertEqual(moved.device, torch.device("cuda:0"))
        self.assertTrue(all(t.device.type == "cuda" for t in all_tensors(moved)))
        self.assertTrue(all(t.device.type == "cpu" for t in all_tensors(owner)))
        self.assertEqual(len(all_tensors(moved)), len(all_tensors(owner)))
        self.assertTrue(all(t.device.type == "cpu" for t in all_tensors(moved.env)))

    def test_same_device_costs_nothing(self):
        owner = Owner(None)
        self.assertIs(learner_adapter(owner, "cpu"), owner)
        tensor, batch = torch.zeros(2), {"state": torch.zeros(2), "step": 3}
        self.assertIs(to_device(tensor, torch.device("cpu")), tensor)
        self.assertIs(to_device(batch, torch.device("cpu")), batch)

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
    def test_transitions_cross_as_new_dictionaries(self):
        batch = {"state": torch.ones(2, 3), "done": torch.tensor([True, False]), "step": 3}
        moved = to_device(batch, torch.device("cuda:0"))
        self.assertIsNot(moved, batch)
        self.assertEqual((moved["state"].device.type, moved["done"].device.type, moved["step"]), ("cuda", "cuda", 3))
        self.assertEqual(batch["state"].device.type, "cpu")


class DeviceAssignmentTest(unittest.TestCase):
    @staticmethod
    def assigned(device, sim_device="", environment=None):
        cfg = PerceptiveConfig.defaults("supermimic")
        cfg.device, cfg.sim_device = device, sim_device
        environment = {} if environment is None else environment
        assign_devices(cfg, environment)
        return cfg.device, cfg.sim_device, environment.get("CUDA_VISIBLE_DEVICES")

    def test_learner_is_card_zero_and_simulator_card_one(self):
        self.assertEqual(self.assigned("cuda:0"), ("cuda:0", "", "0"))
        self.assertEqual(self.assigned("cuda:2"), ("cuda:0", "", "2"))
        self.assertEqual(self.assigned("cuda:0", "cuda:1"), ("cuda:0", "cuda:1", "0,1"))
        self.assertEqual(self.assigned("cuda:3", "cuda:1"), ("cuda:0", "cuda:1", "3,1"))
        self.assertEqual(self.assigned("cuda:1", "cuda:1"), ("cuda:0", "", "1"))
        self.assertEqual(self.assigned("cuda:0", "cpu"), ("cuda:0", "cpu", "0"))

    def test_an_existing_assignment_is_respected(self):
        environment = {"CUDA_VISIBLE_DEVICES": "2,3"}
        self.assertEqual(self.assigned("cuda:0", "cuda:1", environment), ("cuda:0", "cuda:1", "2,3"))

    def test_placement_is_validated_and_never_restored_from_a_checkpoint(self):
        cfg = PerceptiveConfig.defaults("supermimic")
        cfg.sim_device = "cuda:1"
        cfg.validate()
        self.assertIn("sim_device", MUTABLE_RESUME_FIELDS)
        self.assertEqual(config_from_payload({"format": FORMAT, "config": asdict(cfg)}).sim_device, "")
        cfg.sim_device = "gpu1"
        with self.assertRaisesRegex(ValueError, "sim_device"):
            cfg.validate()
        old = PerceptiveConfig.defaults("supermimic", hiking_training_version=2)
        old.sim_device = "cuda:1"
        with self.assertRaisesRegex(ValueError, "separate simulator device"):
            old.validate()


class LearnerAdapterTest(unittest.TestCase):
    """The learner's adapter copy against the environment's own adapter, on a real environment."""

    @classmethod
    def setUpClass(cls):
        cfg = PerceptiveConfig.defaults("supermimic")
        # Only the simulator is built, on the CPU; the learner's device is not touched here.
        cfg.num_envs, cfg.sim_device, cfg.terrain_rows, cfg.terrain_cols, cfg.seed = 6, "cpu", 2, 10, 3
        cls.env = make_environment(cfg)
        cls.env.reset()
        generator = torch.Generator().manual_seed(0)
        for _ in range(3):
            cls.env.step(.1 * torch.randn(6, 29, generator=generator))
        cls.adapter = cls.env.adapter
        cls.inputs = dict(
            state=cls.adapter.state().clone(), context=cls.adapter.context().clone(),
            command=cls.adapter.command().clone(), transition={k: v.clone() if torch.is_tensor(v) else v
                                                               for k, v in cls.env.transition.items()},
            action=.3 * torch.randn(6, 29, generator=generator),
            previous_action=.3 * torch.randn(6, 29, generator=generator))

    @classmethod
    def tearDownClass(cls):
        cls.env.close()

    def outputs(self, adapter, device):
        """Every adapter function imagination calls, on the same physical inputs."""
        i = {name: to_device(value, torch.device(device)) for name, value in self.inputs.items()}
        t = i["transition"]
        state, next_state, context, command = t["state"], t["next_state"], t["context"], t["command"]
        fk = adapter.kinematics(next_state)
        next_context, terminal = adapter.advance_context(next_state, context, i["action"], t["contact_cost"], kinematics=fk)
        result = dict(
            kinematics=torch.cat([part.reshape(6, -1) for part in fk], -1),
            critic=adapter.critic_features(state, i["previous_action"], command, context),
            amp=adapter.amp_features(state),
            relative_height=adapter.relative_height(state),
            actor_frame=adapter.actor_frame(state, i["previous_action"], command, context),
            next_context=next_context, terminal=terminal,
            reward=adapter.task_reward(state, next_state, i["action"], i["previous_action"], command, next_context,
                                       terminal, t["contact_cost"], kinematics=fk),
            command=adapter.navigation_command(next_state, t["navigation_target"], t["desired_speed"],
                                               standing_mask=t["command_standing"],
                                               random_yaw_rate=t.get("random_yaw_rate"),
                                               random_velocity_mask=t.get("random_velocity_mask")),
            terrain_height=adapter.terrain.height_at(state[:, :2]),
            terrain_patch=adapter.terrain.local_patch(state),
        )
        return {name: value.detach().cpu() for name, value in result.items()}

    def test_copy_on_the_same_device_is_exact_and_independent(self):
        copied = relocate(self.adapter, torch.device("cpu"), source="cpu", shared=(self.env,))
        self.assertIsNot(copied, self.adapter)
        self.assertIsNot(copied.terrain, self.adapter.terrain)
        self.assertIs(copied.env, self.env)
        self.assertIs(copied.robot, self.adapter.robot)
        expected, actual = self.outputs(self.adapter, "cpu"), self.outputs(copied, "cpu")
        self.assertGreaterEqual(len(expected), 11)
        for name in expected:
            with self.subTest(function=name):
                self.assertTrue(torch.isfinite(expected[name].float()).all())
                torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)

    @unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
    def test_copy_on_another_device_computes_the_same_values(self):
        moved = learner_adapter(self.adapter, "cuda:0")
        self.assertEqual(moved.device, torch.device("cuda:0"))
        self.assertEqual(self.adapter.device, torch.device("cpu"))
        on_learner, on_simulator = all_tensors(moved), all_tensors(self.adapter)
        self.assertGreater(len(on_learner), 40)
        self.assertEqual({t.device.type for t in on_learner}, {"cuda"})
        self.assertEqual({t.device.type for t in on_simulator}, {"cpu"})
        expected, actual = self.outputs(self.adapter, "cpu"), self.outputs(moved, "cuda:0")
        for name in expected:
            with self.subTest(function=name):
                torch.testing.assert_close(actual[name].float(), expected[name].float(), rtol=2e-5, atol=2e-5)
        # Live readings still come from the simulator, delivered on the learner's device.
        for name in LIVE_ADAPTER_READS:
            with self.subTest(live=name):
                value = getattr(moved, name)()
                self.assertEqual(value.device.type, "cuda")
                torch.testing.assert_close(value.cpu(), getattr(self.adapter, name)())


if __name__ == "__main__":
    unittest.main()
