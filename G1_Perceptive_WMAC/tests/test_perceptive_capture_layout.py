"""Regression for replay-view layout changes at the first Actor graph capture.

CPU checks exercise the real capture preparation with a no-op graph. Optional
CUDA tests use only 29 trainable scalars and no environment or world model.
Run CUDA checks explicitly with PERCEPTIVE_TEST_CAPTURE_LAYOUT=1.
"""
from __future__ import annotations

from contextlib import nullcontext
import os
import unittest
from unittest.mock import patch

import torch
from torch import nn

from src.wmac_perceptive.captured_updates import CapturedTensorUpdate
from src.wmac_perceptive.optimizers import build_reference_optimizer


CUDA = os.environ.get("PERCEPTIVE_TEST_CAPTURE_LAYOUT") == "1" and torch.cuda.is_available()


def replay_view_batch(device="cpu", offset=0.):
    # The replay frame concatenates state71 and action29. Its action slice is
    # [B,29] with a 100-element leading stride and a nonzero storage offset.
    frame = torch.arange(4*100, device=device, dtype=torch.float32).reshape(4, 100)/400 + offset
    return {"previous_action": frame[:, 71:100],
            "target": torch.full((4, 29), .17+offset, device=device)}


def tensor_layout(value):
    return tuple(value.shape), tuple(value.stride()), value.dtype, value.device, value.requires_grad


def small_loss(gain, previous_action, target):
    prediction = gain*previous_action
    loss = (prediction-target).square().mean()
    return loss, {"loss": loss.detach(), "prediction": prediction.detach()}


class NestedCompiledLoss(nn.Module):
    metric_names = ()

    def __init__(self, function=small_loss, device="cpu"):
        super().__init__()
        self.gain = nn.Parameter(torch.linspace(.5, .9, 29, device=device))
        self.function = function
        self.observed_layouts = []
        self.observed_inputs = []

    @staticmethod
    def prepare(batch, **unused):
        return batch

    def forward(self, batch):
        self.observed_layouts.append({key: tensor_layout(value) for key, value in batch.items()})
        self.observed_inputs.append((batch, dict(batch)))
        return self.function(self.gain, batch["previous_action"], batch["target"])


class NoOpGraph:
    def replay(self):
        pass

    def reset(self):
        pass


class CaptureLayoutCpuTest(unittest.TestCase):
    def test_fixture_reproduces_replay_action_view_that_clone_densifies(self):
        value = replay_view_batch()["previous_action"]
        self.assertEqual(value.shape, (4, 29))
        self.assertEqual(value.stride(), (100, 1))
        self.assertFalse(value.is_contiguous())
        self.assertEqual(value.clone().stride(), (29, 1))

    def test_real_capture_preparation_matches_warmup_layout_and_numerical_update(self):
        kernel, reference = NestedCompiledLoss(), NestedCompiledLoss()
        optimizer = torch.optim.SGD(kernel.parameters(), lr=.01)
        expected_optimizer = torch.optim.SGD(reference.parameters(), lr=.01)
        execution = CapturedTensorUpdate(kernel, optimizer, mode="eager", warmup=3,
            compile_loss=False, guard_nonfinite=False)
        expected = CapturedTensorUpdate(reference, expected_optimizer, mode="eager", warmup=3,
            compile_loss=False, guard_nonfinite=False)
        # Bypass only GPU allocation/capture, not production input preparation.
        # Loss/backward execute once inside the fake recording context; its
        # no-op replay prevents a duplicate CPU update.
        execution.mode = "graph"
        with patch("torch.cuda.CUDAGraph", NoOpGraph), \
             patch("torch.cuda.graph", lambda *args, **kwargs: nullcontext()):
            source_strides = []
            for index in range(4):
                batch = replay_view_batch(offset=.01*index)
                if index % 2: batch["previous_action"] = batch["previous_action"].contiguous()
                source_strides.append(batch["previous_action"].stride())
                original = batch["previous_action"].clone()
                actual_outputs, _, _ = execution.step(batch)
                expected_outputs, _, _ = expected.step(batch)
                torch.testing.assert_close(actual_outputs["loss"], expected_outputs["loss"], rtol=0, atol=0)
                torch.testing.assert_close(kernel.gain, reference.gain, rtol=0, atol=0)
                torch.testing.assert_close(batch["previous_action"], original, rtol=0, atol=0)
                if index < 3: self.assertIsNone(execution._graph)
        self.assertIsNotNone(execution._graph)
        self.assertEqual(execution._calls, 4)
        self.assertEqual(source_strides, [(100, 1), (29, 1), (100, 1), (29, 1)])
        self.assertEqual(len(kernel.observed_layouts), 4)
        first_batch, first_inputs = kernel.observed_inputs[0]
        for index, (seen_batch, seen_inputs) in enumerate(kernel.observed_inputs):
            self.assertIs(seen_batch, first_batch, f"macro {index+1} rebound its persistent input dictionary")
            self.assertEqual(kernel.observed_layouts[index], kernel.observed_layouts[0])
            for name, tensor in seen_inputs.items():
                self.assertIs(tensor, first_inputs[name])
                self.assertEqual(tensor.data_ptr(), first_inputs[name].data_ptr())
        execution.invalidate()
        self.assertEqual(execution._calls, 0)
        self.assertIsNone(execution._graph)
        execution.step({"previous_action": torch.ones(6, 29), "target": torch.zeros(6, 29)})
        new_batch, new_inputs = kernel.observed_inputs[-1]
        self.assertIsNot(new_batch, first_batch)
        for name in new_inputs:
            self.assertIsNot(new_inputs[name], first_inputs[name])
            self.assertNotEqual(new_inputs[name].data_ptr(), first_inputs[name].data_ptr())

    def test_schema_changes_are_rejected_during_warmup_and_invalidate_accepts_new_shape(self):
        kernel = NestedCompiledLoss()
        execution = CapturedTensorUpdate(kernel, torch.optim.SGD(kernel.parameters(), lr=.01),
            mode="eager", warmup=3, compile_loss=False, guard_nonfinite=False)
        execution.mode = "graph"
        execution.step(replay_view_batch())
        first_batch, first_inputs = kernel.observed_inputs[0]
        snapshots = {name: value.clone() for name, value in first_inputs.items()}
        invalid = (
            {"previous_action": torch.ones(4, 29)},
            {"previous_action": torch.ones(6, 29), "target": torch.zeros(6, 29)},
            {"previous_action": torch.ones(4, 29), "target": torch.zeros(4, 29, dtype=torch.float64)},
            {"previous_action": torch.ones(4, 29), "target": torch.empty(4, 29, device="meta")},
        )
        for batch in invalid:
            with self.assertRaisesRegex(ValueError, "fields|schema"):
                execution.step(batch)
            self.assertEqual(execution._calls, 1)
            self.assertEqual(len(kernel.observed_inputs), 1)
            for name, value in first_inputs.items():
                torch.testing.assert_close(value, snapshots[name], rtol=0, atol=0)
        execution.invalidate()
        self.assertEqual(execution._calls, 0)
        execution.step({"previous_action": torch.ones(6, 29), "target": torch.zeros(6, 29)})
        self.assertEqual(execution._calls, 1)
        self.assertIsNone(execution._graph)
        self.assertIsNot(kernel.observed_inputs[-1][0], first_batch)
        self.assertEqual(kernel.observed_inputs[-1][1]["previous_action"].shape, (6, 29))


@unittest.skipUnless(CUDA, "set PERCEPTIVE_TEST_CAPTURE_LAYOUT=1 for the small CUDA graph regression")
class CaptureLayoutCudaTest(unittest.TestCase):
    def _capture_case(self, backend_name):
        torch.cuda.set_device(0)
        torch._dynamo.reset()
        compilations = []

        def backend(graph, example_inputs):
            capturing = torch.cuda.is_current_stream_capturing()
            compilations.append(capturing)
            if capturing:
                raise AssertionError("Nested Dynamo compilation occurred during CUDA graph capture")
            if backend_name == "inductor":
                return torch._inductor.compile(graph, example_inputs)
            return graph.forward

        function = torch.compile(small_loss, backend=backend, fullgraph=True, dynamic=False)
        kernel, reference = NestedCompiledLoss(function, "cuda"), NestedCompiledLoss(device="cuda")
        optimizer = build_reference_optimizer([kernel], lr=2e-4, device="cuda")
        expected_optimizer = build_reference_optimizer([reference], lr=2e-4, device="cuda")
        execution = CapturedTensorUpdate(kernel, optimizer, mode="graph_full", warmup=3,
            compile_loss=False, guard_nonfinite=False)
        expected = CapturedTensorUpdate(reference, expected_optimizer, mode="eager", warmup=3,
            compile_loss=False, guard_nonfinite=False)
        captured_on = None
        layouts = []
        for macro in range(1, 7):
            batch = replay_view_batch("cuda", offset=.003*macro)
            if macro == 6: batch["previous_action"] = batch["previous_action"].contiguous()
            layouts.append(tensor_layout(batch["previous_action"]))
            was_captured = execution._graph is not None
            actual, norm, accepted = execution.step(batch)
            expected_outputs, expected_norm, expected_accepted = expected.step(batch)
            if not was_captured and execution._graph is not None:
                captured_on = macro
            self.assertEqual(bool(accepted), bool(expected_accepted))
            torch.testing.assert_close(actual["loss"], expected_outputs["loss"], rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(norm, expected_norm, rtol=2e-5, atol=2e-6)
            torch.testing.assert_close(kernel.gain, reference.gain, rtol=2e-5, atol=2e-6)
        torch.cuda.synchronize()
        self.assertEqual(captured_on, 4)
        self.assertEqual(execution._calls, 6)
        self.assertEqual([layout[1] for layout in layouts[:3]], [(100, 1)]*3)
        self.assertEqual(layouts[-1][1], (29, 1))
        self.assertTrue(compilations)
        self.assertFalse(any(compilations))
        self.assertEqual(len(compilations), 1, "No second layout specialization should be needed at capture or replay")
        self.assertEqual(len(kernel.observed_layouts), 4)
        self.assertTrue(all(layout == kernel.observed_layouts[0] for layout in kernel.observed_layouts))
        first_batch, first_inputs = kernel.observed_inputs[0]
        for seen_batch, seen_inputs in kernel.observed_inputs:
            self.assertIs(seen_batch, first_batch)
            for name, tensor in seen_inputs.items(): self.assertIs(tensor, first_inputs[name])
        for state in optimizer.state.values():
            if "step" in state: self.assertEqual(int(state["step"]), 6)

    def test_nested_compiled_fx_kernel_captures_after_strided_warmup(self):
        self._capture_case("fx")

    def test_nested_inductor_kernel_captures_and_replays_after_strided_warmup(self):
        self._capture_case("inductor")


if __name__ == "__main__":
    unittest.main()
