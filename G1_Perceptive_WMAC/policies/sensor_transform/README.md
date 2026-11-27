# G1 sensor_transform policy

This is the trained, unshod 29-DoF Unitree G1 policy for the Hiking task. The checkpoint records training version 3, learner iteration 27,997 and camera protocol `g1_hiking_calibrated_v1`. New training runs in this repository default to version 5.

| File | Use |
| --- | --- |
| [sensor_transform.pt](sensor_transform.pt) | Original PyTorch training checkpoint, stored in Git LFS (105,075,150 bytes). |
| [exported/policy.onnx](exported/policy.onnx) | Actor for the patched Unitree `g1_ctrl` controller. Float32 input `obs[1,171264]` produces 29 actions at 50 Hz. |
| [params/deploy.yaml](params/deploy.yaml) | Controller configuration, joint order, gains, scales and observation history. |
| [params/deployment_contract.json](params/deployment_contract.json) | Torso and waist conventions, timing, depth processing and camera calibration. |
| [params/depth_camera.npz](params/depth_camera.npz) | Calibrated ray directions, depth projection factors and camera mount. |

## Download and verify

From the repository root, with Git LFS installed, run the following commands. All five entries in [SHA256SUMS](SHA256SUMS) should report `OK`.

```bash
git lfs pull --include="G1_Perceptive_WMAC/policies/sensor_transform/sensor_transform.pt"
(cd G1_Perceptive_WMAC/policies/sensor_transform && sha256sum -c SHA256SUMS)
```

## Controller and camera

Follow the [Unitree controller integration guide](../../deploy/unitree_rl_lab/README.md) to build the controller and run its offline observation check. The exported ONNX actor reproduced the PyTorch actions to a maximum absolute difference of 0.0000062 across 16 test inputs.

On hardware, the controller needs a publisher providing 64 × 36 image-plane depth in metres on DDS topic `rt/hiking/depth`. Map the actual stream to the v3 rays in this export. The 480 × 270 lookup described for new v4/v5 training targets Hiking's pinhole projection. The hardware camera publisher and end-to-end camera latency remain unverified.

In the recorded MuJoCo evaluation with Unitree's stock G1 feet, this policy walked on flat ground in five of five trials. It fell in all five trials at each tested rough-ground, slope and stair setting. Physical robot performance has not been tested.
