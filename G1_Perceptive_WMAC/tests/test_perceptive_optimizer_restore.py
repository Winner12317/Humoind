"""Pure preflight of actual optimizer states before perceptive restore."""
from __future__ import annotations

import copy
import unittest

import torch
from torch import nn

from src.wmac_perceptive.checkpoints import validate_optimizer_states


def fixture():
    torch.manual_seed(765)
    actor, world = nn.Linear(3, 2), nn.Linear(3, 2)
    optimizers = {"opt_actor": torch.optim.Adam(actor.parameters(), lr=.001, amsgrad=True),
                  "opt_model": torch.optim.AdamW(world.parameters(), lr=.002, weight_decay=.01)}
    for name, module in (("opt_actor", actor), ("opt_model", world)):
        optimizer = optimizers[name]
        optimizer.zero_grad(set_to_none=True)
        module(torch.randn(4, 3)).square().mean().backward()
        optimizer.step()
    saved = {name: copy.deepcopy(optimizer.state_dict()) for name, optimizer in optimizers.items()}
    return (actor, world), optimizers, saved


class PerceptiveOptimizerRestoreTest(unittest.TestCase):
    def assert_tree_equal(self, first, second):
        if isinstance(first, torch.Tensor):
            torch.testing.assert_close(first, second, rtol=0, atol=0)
        elif isinstance(first, dict):
            self.assertEqual(first.keys(), second.keys())
            for key in first: self.assert_tree_equal(first[key], second[key])
        elif isinstance(first, (list, tuple)):
            self.assertEqual(len(first), len(second))
            for left, right in zip(first, second): self.assert_tree_equal(left, right)
        else: self.assertEqual(first, second)

    def assert_pure_rejection(self, modules, optimizers, bad):
        states = {name: copy.deepcopy(opt.state_dict()) for name, opt in optimizers.items()}
        weights = [copy.deepcopy(module.state_dict()) for module in modules]
        rng = torch.get_rng_state().clone()
        with self.assertRaises(ValueError): validate_optimizer_states(bad, optimizers)
        for name, opt in optimizers.items(): self.assert_tree_equal(opt.state_dict(), states[name])
        for module, expected in zip(modules, weights): self.assert_tree_equal(module.state_dict(), expected)
        torch.testing.assert_close(torch.get_rng_state(), rng, rtol=0, atol=0)

    def test_actual_adam_adamw_and_fresh_states_pass_without_mutation(self):
        modules, optimizers, saved = fixture()
        before = copy.deepcopy(saved)
        self.assertIsNone(validate_optimizer_states(saved, optimizers))
        self.assert_tree_equal(saved, before)
        for name, optimizer in optimizers.items(): self.assert_tree_equal(optimizer.state_dict(), before[name])
        fresh = {name: {"state": {}, "param_groups": copy.deepcopy(state["param_groups"])} for name, state in saved.items()}
        self.assertIsNone(validate_optimizer_states(fresh, optimizers))

    def test_late_optimizer_shape_nan_dtype_and_second_moment_errors_are_atomic(self):
        modules, optimizers, good = fixture()
        for problem in ("shape", "nan", "dtype", "negative_variance", "missing_moment", "unknown_scalar"):
            bad = copy.deepcopy(good); state = bad["opt_model"]["state"][0]
            if problem == "shape": state["exp_avg"] = torch.zeros(3, 2)
            elif problem == "nan": state["exp_avg"][0, 0] = float("nan")
            elif problem == "dtype": state["exp_avg"] = state["exp_avg"].double()
            elif problem == "negative_variance": state["exp_avg_sq"][0, 0] = -.1
            elif problem == "missing_moment": state.pop("exp_avg_sq")
            else: state["unknown"] = 2.
            with self.subTest(problem=problem): self.assert_pure_rejection(modules, optimizers, bad)
        bad = copy.deepcopy(good); bad["opt_actor"]["state"][0].pop("max_exp_avg_sq")
        self.assert_pure_rejection(modules, optimizers, bad)

    def test_steps_must_be_single_finite_nonnegative_real_values(self):
        modules, optimizers, good = fixture()
        for step in (torch.tensor(float("inf")), torch.tensor(float("nan")),
                     torch.tensor(-1.), torch.zeros(2), "one", True):
            bad = copy.deepcopy(good); bad["opt_model"]["state"][0]["step"] = step
            with self.subTest(step=step): self.assert_pure_rejection(modules, optimizers, bad)
        for step in (0., 4, torch.tensor(7.), torch.tensor([2.])):
            saved = copy.deepcopy(good); saved["opt_model"]["state"][0]["step"] = step
            self.assertIsNone(validate_optimizer_states(saved, optimizers))

    def test_ordered_ids_groups_duplicates_and_orphan_states_are_checked(self):
        modules, optimizers, good = fixture()
        for problem in ("reordered", "duplicate", "bad_id", "orphan_state", "missing_group", "too_many_groups"):
            bad = copy.deepcopy(good); model = bad["opt_model"]
            if problem == "reordered": model["param_groups"][0]["params"].reverse()
            elif problem == "duplicate": model["param_groups"][0]["params"] = [0, 0]
            elif problem == "bad_id": model["param_groups"][0]["params"][0] = "zero"
            elif problem == "orphan_state": model["state"][999] = copy.deepcopy(model["state"][0])
            elif problem == "missing_group": model["param_groups"] = []
            else: model["param_groups"].append(copy.deepcopy(model["param_groups"][0]))
            with self.subTest(problem=problem): self.assert_pure_rejection(modules, optimizers, bad)

    def test_hyperparameters_are_finite_and_legal(self):
        modules, optimizers, good = fixture()
        for key, value in (("lr", float("nan")), ("lr", -1.), ("eps", -1.),
                           ("weight_decay", float("inf")), ("betas", (.9, 1.)),
                           ("betas", (-.1, .99)), ("betas", (.9,)), ("amsgrad", 1),
                           ("fused", "yes")):
            bad = copy.deepcopy(good); bad["opt_model"]["param_groups"][0][key] = value
            with self.subTest(key=key, value=value): self.assert_pure_rejection(modules, optimizers, bad)
        bad = copy.deepcopy(good); bad["opt_model"]["param_groups"][0].pop("eps")
        self.assert_pure_rejection(modules, optimizers, bad)

    def test_sgd_momentum_and_named_group_mapping_are_supported(self):
        first, second = nn.Parameter(torch.ones(3)), nn.Parameter(torch.ones(3))
        optimizer = torch.optim.SGD([{"params": [first], "name": "first"},
                                     {"params": [second], "name": "second"}], lr=.1, momentum=.9)
        (first.square().sum()+second.square().sum()).backward(); optimizer.step()
        saved = {"opt": copy.deepcopy(optimizer.state_dict())}
        self.assertIsNone(validate_optimizer_states(saved, {"opt": optimizer}))
        bad = copy.deepcopy(saved); bad["opt"]["param_groups"][1]["name"] = "first"
        with self.assertRaisesRegex(ValueError, "named parameter"):
            validate_optimizer_states(bad, {"opt": optimizer})
        for changes in ({"momentum": -.1}, {"nesterov": True, "momentum": 0.},
                        {"nesterov": True, "dampening": .2}):
            bad = copy.deepcopy(saved); bad["opt"]["param_groups"][0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_optimizer_states(bad, {"opt": optimizer})


if __name__ == "__main__":
    unittest.main()
