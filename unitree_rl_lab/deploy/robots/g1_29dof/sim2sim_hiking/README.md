# Sim-to-sim for perceptive Hiking policies (G1_Perceptive_WMAC)

Runs a depth-perceptive policy trained in `G1_Perceptive_WMAC` with this repository's
unchanged `g1_ctrl` controller loop. The simulator is MuJoCo with Unitree's G1 model
from `unitree_mujoco`, and the two processes talk over DDS as on the robot.

## What was added

| File | Purpose |
|---|---|
| `deploy/include/isaaclab/envs/mdp/observations/hiking_observations.h` | Three new observation terms. `hiking_torso_ang_vel` and `hiking_torso_projected_gravity` compute the torso frame from the pelvis IMU and waist encoders. `hiking_depth_ring` keeps a 37-frame depth ring read from `rt/hiking/depth` (a PointCloud2). |
| `deploy/include/FSM/State_FixStand.h` | A FixStand-type state can carry its own `qs`/`kp`/`kd`. Plain `FixStand` is unchanged. |
| `config/config.yaml` | Adds `HikingStand` (id 4: the policy's knees-bent default pose) and `Hiking` (id 5: RLBase on `config/policy/hiking`). |
| `config/policy/hiking/<name>/` | The exported policy: `policy.onnx`, `deploy.yaml`, the camera rays and the deployment contract. |
| `sim2sim_hiking/hiking_mujoco.py` | The simulator. It is unitree_mujoco's Python bridge plus Hiking terrains, the policy's depth camera, PD applied at every physics step, a scripted remote and keyboard control. |
| `sim2sim_hiking/terrains/*.npz` | 8 × 8 m tiles from the training terrain generator. |
| `sim2sim_hiking/test/` | A C++ check that `g1_ctrl` builds exactly the training policy input. |

The policy uses Hiking's torso-rooted URDF, which matters in three places:

- **Waist joints have the opposite sign** to the Unitree SDK. `deploy.yaml` folds the conversion into negative observation and action scales.
- **The IMU frame is the torso**, not the pelvis. The new terms derive it from the pelvis IMU and the waist encoders.
- **The camera sits at `(0.0488, 0.01, 0.4278)` in Unitree's `torso_link`**. All of this is recorded in `params/deployment_contract.json`.

## Setup

```bash
# 1. Controller (needs unitree_sdk2, as for the stock policies)
cd deploy/robots/g1_29dof && mkdir -p build && cd build && cmake .. && make -j8

# 2. Simulator environment (mujoco, cyclonedds 0.10.2 and an editable unitree_sdk2py)
uv venv --python 3.10 ~/.local/share/g1-hiking-sim2sim/venv
uv pip install --python ~/.local/share/g1-hiking-sim2sim/venv/bin/python mujoco==3.3.7 "numpy<2.3" cyclonedds==0.10.2
uv pip install --python ~/.local/share/g1-hiking-sim2sim/venv/bin/python --no-deps \
  "git+https://github.com/unitreerobotics/unitree_sdk2_python.git@ab0d8ae2fee8e3ec337ba941d5e7fd49de5fd30a"

# 3. unitree_mujoco must sit next to unitree_rl_lab (or pass --unitree-mujoco)

# 4. Export a checkpoint (from G1_Perceptive_WMAC, in its own environment)
python scripts/export_unitree_rl_lab.py --checkpoint /path/model.pt \
  --policy-dir ../unitree_rl_lab/deploy/robots/g1_29dof/config/policy/hiking/<name>
```

The `Hiking` state loads the alphabetically last folder under `config/policy/hiking/`.

## Running

Interactive, with keyboard control in the viewer:

```bash
python hiking_mujoco.py --terrain stairs_up_0.3 --viewer        # terminal 1
cd ../build && ./g1_ctrl --network lo                           # terminal 2
```

| Key | Action |
|---|---|
| `1` | stand (HikingStand) |
| `2` | start the policy (Hiking); the band releases |
| `↑` / `↓` | forward speed |
| `←` / `→` | yaw rate |
| `space` | stop |
| `0` | Passive |
| `9` | elastic band |

On a gamepad the same states are L2 + right and R1 + Y.

The policy was trained on forward speeds of 0, or 0.45–1.0 m/s; no sideways motion.

Unattended (stand, start the policy, stand for 1 s, walk at 0.6 m/s to the target 3.7 m ahead):

```bash
bash run_auto.sh stairs_down_0.3                               # results/stairs_down_0.3.json
bash run_suite.sh results/suite 5                              # all terrains, 5 seeds each
SUITE_ARGS="--feet training" bash run_suite.sh results/suite_training_feet 5
```

Terrains: `flat`, `rough_0.5`, `rough_1.0`, `slope_0.5`, `stairs_down_0.3`, `stairs_down_0.6`, `stairs_up_0.3`, `stairs_up_0.6`. The number is the training difficulty. A stair rise is 0.05 + 0.18 × difficulty m on 0.3 m treads.

## Verified

- `test/`: across 40 steps × 171,264 values, `g1_ctrl`'s observation vector matches the training-side reference. Proprioception agrees to 3×10⁻⁷ and the depth ring is identical. This covers the waist signs, the torso-frame IMU, history order, scales and the depth encoding.
- The fixed-batch ONNX matches the PyTorch actor to 3×10⁻⁶.
- The simulator keeps 1.00× real time on every terrain.

## Results for `sensor_transform.pt`

The [archived trial reports and controller logs](reference_results/) include the per-step measurements, outcomes and loopback controller events. The logs use repository-relative paths for policy directories. Fresh runs write to `results/`.

Each cell is falls/5 seeds (reached the target/5). The path is `g1_ctrl` over DDS.

| Terrain | Unitree G1 model (default) | `--feet training` |
|---|---|---|
| flat | 0 (5) | 0 (5) |
| rough_0.5 / rough_1.0 | 5 (0) / 5 (0) | 0 (5) / 0 (5) |
| slope_0.5 (uphill) | 5 (0) | 1 (3); one stalled |
| stairs_down_0.3 / 0.6 | 5 (0) / 5 (0) | 2 (3) / 0 (5) |
| stairs_up_0.3 / 0.6 | 5 (0) / 5 (0) | 4 (0) / 5 (0) |

- **Feet.** Unitree's MJCF models each foot as four 5 mm spheres at the sole corners. The training robot uses seven capsules covering the sole. On uneven ground the four-point feet sink up to 20 mm and slip, and the policy falls within a few seconds. Swapping only the foot collision shapes (`--feet training`) removes the rough-terrain falls. A policy meant for Unitree's model, or one robust to foot contact modelling, needs that geometry or its randomization in training.
- **Stair ascent** fails with either foot model, as it does in the training simulator (61–82%). That is a training problem.
- **Depth** is load-bearing. With `--depth-mode blind` (no returns), the policy falls on flat ground in 5/5 runs.
