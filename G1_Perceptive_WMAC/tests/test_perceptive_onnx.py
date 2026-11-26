"""Deployment parity for the complete default crop/blur/depth-history Actor."""
from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from src.wmac_perceptive.models import PerceptiveActor
from src.wmac_perceptive.perception import PerceptionSpec, pack_actor_input


class PerceptiveOnnxTest(unittest.TestCase):
    def test_full_export_stores_runtime_pd_parameters_in_natural_joint_order(self):
        """Motor gain=1/bias=0 must never masquerade as wrapper PD gains."""
        if importlib.util.find_spec("onnx") is None:
            self.skipTest("ONNX is required for deployment metadata acceptance")
        import mujoco
        import onnx
        from src.assets.robots.unitree_g1.g1_constants import G1_XML
        from src.wmac_perceptive.config import PerceptiveConfig,perception_spec_from_config
        from src.wmac_perceptive.hiking_actuators import (
            HIKING_ACTUATOR_GROUPS,HikingDelayedPdActuatorCfg,get_hiking_joint_parameters,
        )
        from src.wmac_perceptive.latent_models import LatentPerceptiveActor
        from src.wmac_perceptive.runner import PerceptiveRunner

        model=mujoco.MjModel.from_xml_path(str(G1_XML))
        names=tuple(model.joint(i).name for i in range(1,model.njnt))
        source=get_hiking_joint_parameters(names)
        # Deliberately distinguish runtime gains from their nominal config,
        # and environment 0 from environment 1. Export must use runtime row 0.
        factors=1+torch.arange(29,dtype=torch.float32)/100
        expected={"kp":source["kp"]*factors,"kd":source["kd"]*factors,
                  "effort_limit":source["effort_limit"],"velocity_limit":source["velocity_limit"]}
        robot=SimpleNamespace(joint_names=names,data=SimpleNamespace(joint_pos=torch.zeros(2,29)),actuators=[])
        for group_id,group in enumerate(HIKING_ACTUATOR_GROUPS):
            # Reverse each real actuator group's storage order to detect an
            # exporter that concatenates group arrays instead of mapping IDs.
            ids=(source["delay_group"]==group_id).nonzero().flatten().flip(0)
            group_names=[names[i] for i in ids.tolist()]
            cfg=HikingDelayedPdActuatorCfg(group=group,target_names_expr=tuple(group_names))
            actuator=cfg.build(robot,ids.tolist(),group_names)
            actuator._target_ids=ids
            for attribute,key in (("stiffness","kp"),("damping","kd"),
                                  ("force_limit","effort_limit"),("velocity_limit","velocity_limit")):
                values=expected[key][ids]
                setattr(actuator,attribute,torch.stack((values,values*2)))
            robot.actuators.append(actuator)

        runner=object.__new__(PerceptiveRunner)
        runner.cfg=PerceptiveConfig.defaults("supermimic")
        runner.spec=perception_spec_from_config(runner.cfg)
        runner.actor=LatentPerceptiveActor(runner.spec,gate_hidden=runner.cfg.moe_gate_hidden).eval()
        runner.env=SimpleNamespace(scene={"robot":robot})
        base={"joint_names":list(names),"joint_stiffness":[1.]*29,"joint_damping":[0.]*29,
              "action_scale":[.123]*29,"default_joint_pos":[0.]*29}
        with tempfile.TemporaryDirectory() as directory:
            runner.log_dir=Path(directory)/"runtime_pd_metadata"
            path=Path(directory)/"policy.onnx"
            with patch("mjlab.rl.exporter_utils.get_base_metadata",return_value=copy.deepcopy(base)) as base_metadata:
                runner.export_policy(path)
            base_metadata.assert_called_once_with(runner.env,"runtime_pd_metadata")
            graph=onnx.load(str(path));onnx.checker.check_model(graph)
            metadata={entry.key:entry.value for entry in graph.metadata_props}
        self.assertEqual(metadata["joint_names"].split(","),list(names))
        for name,key in (("joint_stiffness","kp"),("joint_damping","kd"),
                         ("joint_effort_limits","effort_limit"),("joint_velocity_limits","velocity_limit")):
            values=np.fromstring(metadata[name],sep=",")
            self.assertEqual(values.shape,(29,))
            # mjlab's public metadata writer serializes list values to 3 decimals.
            np.testing.assert_allclose(values,expected[key].numpy(),rtol=0,atol=.000501)
        np.testing.assert_allclose(np.fromstring(metadata["action_scale"],sep=","),.123,rtol=0,atol=0)
        self.assertEqual(metadata["observation_history_order"],"term")
        self.assertEqual(metadata["observation_history_length"],"8")
        self.assertEqual(metadata["proprioception_width"],"768")
        self.assertEqual(metadata["proprioception_frame"],"torso_link")
        self.assertEqual(metadata["angular_velocity_scale"],"0.25")
        self.assertEqual(metadata["joint_velocity_scale"],"0.05")
        self.assertEqual(metadata["locomotion_objective"],"hiking_full_v1")

    def test_complete_default_actor_exports_dynamic_batches_with_reflect_gaussian_blur(self):
        if any(importlib.util.find_spec(name) is None for name in ("onnx", "onnxruntime")):
            self.skipTest("ONNX and ONNX Runtime are required for deployment acceptance")
        import onnx
        import onnxruntime as ort
        torch.manual_seed(941)
        spec = PerceptionSpec()
        self.assertEqual((spec.crop_top,spec.crop_left,spec.crop_right,spec.blur_kernel),(18,16,16,3))
        actor = PerceptiveActor(spec).cpu().eval()
        before = copy.deepcopy(actor.state_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/"perceptive.onnx"
            torch.onnx.export(actor,torch.zeros(1,spec.packed_dim),str(path),
                input_names=["obs"],output_names=["actions"],
                dynamic_axes={"obs":{0:"batch"},"actions":{0:"batch"}},
                opset_version=18,dynamo=False)
            graph = onnx.load(str(path))
            onnx.checker.check_model(graph)
            self.assertEqual(graph.graph.input[0].type.tensor_type.shape.dim[0].dim_param,"batch")
            self.assertEqual(graph.graph.output[0].type.tensor_type.shape.dim[0].dim_param,"batch")
            self.assertNotIn("TopK",[node.op_type for node in graph.graph.node])
            blur_convolutions=[]
            for node in graph.graph.node:
                if node.op_type!="Conv":continue
                attributes={attribute.name:onnx.helper.get_attribute_value(attribute) for attribute in node.attribute}
                if attributes.get("group")==spec.frames:
                    blur_convolutions.append(attributes)
            self.assertTrue(blur_convolutions)
            self.assertTrue(all(tuple(node["kernel_shape"])==(3,3) for node in blur_convolutions))
            options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
            session=ort.InferenceSession(str(path),sess_options=options,providers=["CPUExecutionProvider"])
            self.assertEqual(session.get_inputs()[0].shape,["batch",spec.packed_dim])
            for batch in (1,3,7):
                with self.subTest(batch=batch):
                    proprio=torch.randn(batch,384)
                    ring=torch.rand(batch,*spec.ring_shape)
                    ring[:,:,1]=(ring[:,:,1]>.2).float()
                    packed=pack_actor_input(proprio,ring,spec)
                    with torch.no_grad():expected=actor(packed).numpy()
                    actual=session.run(None,{"obs":packed.numpy()})[0]
                    self.assertEqual(actual.shape,(batch,29))
                    np.testing.assert_allclose(actual,expected,rtol=2e-5,atol=5e-7)
        self.assertEqual(set(actor.state_dict()),set(before))
        for name,value in actor.state_dict().items():
            torch.testing.assert_close(value,before[name],rtol=0,atol=0)


if __name__=="__main__":
    unittest.main()
