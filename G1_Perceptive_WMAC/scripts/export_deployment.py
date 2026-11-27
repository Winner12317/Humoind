"""Export a perceptive checkpoint for a MuJoCo / Unitree-convention harness.

Writes ``policy.onnx`` (input ``proprio_and_depth_ring``, see the contract) and
``deployment_contract.json`` (joint order and signs, defaults, action scale, PD
gains, torso-frame proprioception, depth ring rules, camera). Build the policy
input with ``src/wmac_perceptive/unitree_interface.py``; it performs the waist
sign conversion and the torso-frame IMU computation.

For a policy trained on Hiking's pinhole camera (training version 4) it also
writes ``realsense_lookup.json``: the fixed table that converts the robot's raw
RealSense depth image into the 64 x 36 frame the policy was trained on
(``RealSenseDepthAdapter`` applies the same table in Python).

    python scripts/export_deployment.py --checkpoint /path/model.pt --output-dir deploy/
    # another depth stream, or a measured camera orientation in torso_link (w x y z):
    python scripts/export_deployment.py ... --realsense-calibration stream.json --camera-mount-rotation .91 0 .41 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--realsense-calibration", type=Path, default=None,
                        help="Depth-stream intrinsics JSON (default: the G1's 480x270, 60 Hz profile)")
    parser.add_argument("--camera-mount-rotation", type=float, nargs=4, metavar=("W", "X", "Y", "Z"), default=None,
                        help="Measured camera orientation in torso_link (+X forward, +Y left, +Z up); "
                             "default: Hiking's nominal mount")
    args = parser.parse_args(argv)
    import numpy as np
    import torch
    from scripts.evaluate_wmac_perceptive import load_actor
    from src.wmac_perceptive.onnx_export import export_raw_packed_actor
    from src.wmac_perceptive.config import HIKING_SENSOR_PROTOCOL
    from src.wmac_perceptive.unitree_interface import (
        G1_DEPLOYMENT_DEPTH_CALIBRATION, HIKING_CAMERA_ROTATION_WXYZ, RealSenseDepthAdapter, deployment_contract)

    actor, cfg, payload = load_actor(args.checkpoint.expanduser().resolve(), "cpu")
    if cfg.hiking_training_version < 3:
        parser.error("The deployment contract describes the Hiking v3 task only")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = args.output_dir / "policy.onnx"
    export_raw_packed_actor(actor, onnx_path)
    contract = deployment_contract(camera=payload["architecture"].get("camera"), checkpoint=str(args.checkpoint))
    contract["training_robot_dynamics_randomization"] = getattr(cfg, "robot_dynamics_randomization", "none")
    calibration_path = args.realsense_calibration or G1_DEPLOYMENT_DEPTH_CALIBRATION
    if cfg.sensor_protocol == HIKING_SENSOR_PROTOCOL:
        adapter = RealSenseDepthAdapter(calibration_path, mount_rotation_wxyz=args.camera_mount_rotation)
        (args.output_dir / "realsense_lookup.json").write_text(json.dumps(adapter.description()) + "\n")
        calibration = json.loads(Path(calibration_path).read_text())
        contract["depth"]["real_camera"] = {
            "lookup_file": "realsense_lookup.json",
            "stream": {name: calibration[name] for name in ("width", "height", "fps", "fx", "fy", "cx", "cy",
                                                           "depth_scale_m_per_raw_unit") if name in calibration},
            "mount_rotation_wxyz": list(args.camera_mount_rotation or HIKING_CAMERA_ROTATION_WXYZ),
            "mount_measured": args.camera_mount_rotation is not None,
            "mount_note": ("The camera position is assumed to be the training position (depth.camera.position); "
                           "a lookup cannot correct a position offset."),
            "steps": ["raw Z16 image -> 64x36 training frame through the lookup",
                      "fill holes (depth < 0.2 m) from neighbours inside rows 18-35, cols 16-47",
                      "remaining unmeasured pixels -> 1e6 m; divide by 2.5; append once per 50 Hz control step"],
        }
    else:
        contract["depth"]["real_camera"] = {
            "note": (f"This policy was trained with sensor protocol {cfg.sensor_protocol}, not Hiking's pinhole; "
                     "the RealSense lookup is written only for hiking_depth_only_v1 policies.")}
    (args.output_dir / "deployment_contract.json").write_text(json.dumps(contract, indent=1) + "\n")
    try:
        import onnxruntime
    except ImportError:
        print(f"Wrote {onnx_path} and deployment_contract.json (onnxruntime unavailable: output not cross-checked)")
        return 0
    sample = torch.zeros(2, actor.spec.packed_dim)
    sample[:, :actor.spec.proprio_dim].normal_()
    sample[:, actor.spec.proprio_dim:] = torch.rand(2, actor.spec.packed_dim - actor.spec.proprio_dim)
    with torch.inference_mode():
        expected = actor(sample).numpy()
    session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    got = session.run(None, {"proprio_and_depth_ring": sample.numpy()})[0]
    error = float(np.abs(got - expected).max())
    print(f"Wrote {onnx_path} and deployment_contract.json; ONNX vs PyTorch max |difference| {error:.2e}")
    if error > 1e-4:
        raise SystemExit("ONNX output differs from the PyTorch Actor")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
