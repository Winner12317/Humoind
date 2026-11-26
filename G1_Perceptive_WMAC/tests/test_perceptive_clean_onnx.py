"""No shape-tracing/Slice-fold warnings, with real ONNX Runtime parity."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
import warnings

import numpy as np
import torch

from src.wmac_perceptive.latent_models import LatentPerceptiveActor
from src.wmac_perceptive.models import PerceptiveActor
from src.wmac_perceptive.onnx_export import RawPackedExportActor, export_raw_packed_actor
from src.wmac_perceptive.perception import PerceptionSpec, pack_actor_input


AVAILABLE = all(importlib.util.find_spec(name) for name in ("onnx", "onnxruntime"))


def attention_actor(spec, *, gate_hidden):
    return LatentPerceptiveActor(spec, gate_hidden=gate_hidden, actor_type="transformer")


def variants():
    return (
        ("hiking_latent_depth_only", LatentPerceptiveActor,
         PerceptionSpec(include_valid=False, proprio_dim=768), ()),
        ("hiking_attention_depth_only", attention_actor,
         PerceptionSpec(include_valid=False, proprio_dim=768), ()),
        ("historical_latent_depth_valid", LatentPerceptiveActor, PerceptionSpec(), (256,)),
        ("historical_pixel_depth_valid", PerceptiveActor, PerceptionSpec(), (256,)),
        ("historical_rpl_centered_latent", LatentPerceptiveActor,
         PerceptionSpec(height=27, width=48, ring_length=36, delay=0, max_depth=2., proprio_dim=768,
             crop_top=0, crop_bottom=0, crop_left=0, crop_right=0, blur_kernel=1,
             include_valid=False, depth_protocol="g1_rpl_v1"), ()),
    )


def packed_batch(spec, batch):
    proprio = torch.randn(batch, spec.proprio_dim)
    ring = torch.rand(batch, *spec.ring_shape)
    if spec.depth_protocol == "g1_rpl_v1":
        ring[:, :, 0] = ring[:, :, 0] * 1.4 - .7
    elif not spec.include_valid:
        ring[:, :, 0] = ring[:, :, 0] * 1.4 - .2
        # Preserve the important blur-before-clip behavior on no-hit pixels.
        ring[:, :, 0, spec.crop_top + 1, spec.crop_left + 1] = 400000.
    ring[:, :, 1] = (ring[:, :, 1] > .2).float()
    return pack_actor_input(proprio, ring, spec)


class RawPackedAdapterTest(unittest.TestCase):
    def test_private_copy_matches_all_saved_raw_protocols_and_leaves_actor_unchanged(self):
        torch.manual_seed(20261118)
        for name, kind, spec, gate in variants():
            with self.subTest(protocol=name):
                actor = kind(spec, gate_hidden=gate).train()
                actor.precision = "tf32"
                next(actor.parameters()).requires_grad_(False)
                state = copy.deepcopy(actor.state_dict())
                modes = tuple(module.training for module in actor.modules())
                flags = tuple(parameter.requires_grad for parameter in actor.parameters())
                adapter = RawPackedExportActor(actor)
                for size in (1, 3, 7):
                    packed = packed_batch(spec, size)
                    with torch.no_grad():
                        torch.testing.assert_close(adapter(packed), actor(packed), rtol=0., atol=0.)
                self.assertEqual(tuple(module.training for module in actor.modules()), modes)
                self.assertEqual(tuple(parameter.requires_grad for parameter in actor.parameters()), flags)
                self.assertEqual(actor.precision, "tf32")
                for key, value in actor.state_dict().items():
                    torch.testing.assert_close(value, state[key], rtol=0., atol=0.)


@unittest.skipUnless(AVAILABLE, "ONNX and ONNX Runtime are required")
class CleanOnnxTest(unittest.TestCase):
    def test_export_has_no_tracer_or_slice_folding_warnings_and_dynamic_batch_matches(self):
        import onnx
        import onnxruntime as ort
        torch.manual_seed(20261118)
        for name, kind, spec, gate in variants():
            with self.subTest(protocol=name), tempfile.TemporaryDirectory() as directory:
                actor = kind(spec, gate_hidden=gate).train()
                actor.precision = "tf32"
                state = copy.deepcopy(actor.state_dict())
                modes = tuple(module.training for module in actor.modules())
                path = Path(directory)/"policy.onnx"
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter("always")
                    export_raw_packed_actor(actor, path)
                offending = [str(item.message) for item in captured
                    if issubclass(item.category, torch.jit.TracerWarning)
                    or ("Constant folding" in str(item.message) and "Slice" in str(item.message))]
                self.assertEqual(offending, [])
                model = onnx.load(str(path))
                onnx.checker.check_model(model)
                self.assertEqual(model.graph.input[0].type.tensor_type.shape.dim[0].dim_param, "batch")
                self.assertEqual(model.graph.input[0].type.tensor_type.shape.dim[1].dim_value, spec.packed_dim)
                self.assertEqual(model.graph.input[0].type.tensor_type.elem_type, onnx.TensorProto.FLOAT)
                self.assertEqual(model.graph.output[0].type.tensor_type.shape.dim[0].dim_param, "batch")
                # Contiguous input regions may use unit-step Slice. Reverse
                # or strided Slice is unnecessary for this fixed interface.
                constants = {value.name: onnx.numpy_helper.to_array(value)
                             for value in model.graph.initializer}
                for node in model.graph.node:
                    if node.op_type == "Constant":
                        tensor = next((attribute.t for attribute in node.attribute
                                       if attribute.name == "value"), None)
                        if tensor is not None:
                            constants[node.output[0]] = onnx.numpy_helper.to_array(tensor)
                for node in model.graph.node:
                    if node.op_type == "Slice" and len(node.input) >= 5 and node.input[4]:
                        self.assertIn(node.input[4], constants)
                        self.assertTrue(np.all(constants[node.input[4]] == 1))
                self.assertNotIn("TopK", [node.op_type for node in model.graph.node])
                options = ort.SessionOptions()
                options.intra_op_num_threads = options.inter_op_num_threads = 1
                session = ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])
                for size in (1, 3, 7):
                    packed = packed_batch(spec, size)
                    with torch.no_grad():
                        expected = actor(packed).numpy()
                    actual = session.run(None, {session.get_inputs()[0].name: packed.numpy()})[0]
                    self.assertEqual(actual.shape, (size, 29))
                    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=5e-7)
                self.assertEqual(tuple(module.training for module in actor.modules()), modes)
                self.assertEqual(actor.precision, "tf32")
                for key, value in actor.state_dict().items():
                    torch.testing.assert_close(value, state[key], rtol=0., atol=0.)


if __name__ == "__main__":
    unittest.main()
