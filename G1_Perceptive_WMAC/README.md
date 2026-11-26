# AMP_mjlab

## Perceptive walking on rough terrain

Fresh runs use the **unshod G1** from Hiking’s `G1ParkourRoughEnvCfg`, ported to MuJoCo/mjlab. The official torso-base URDF supplies the collision geometry, axes and inertias. The task uses Hiking’s command distributions, terrain curriculum, foot measurements, 26 reward terms, delayed actuators and eight-frame proprioceptive history. Training and playback record the robot version. Contact solving, heightfield collision and material behavior depend on the simulator. [Hiking task source](https://github.com/project-instinct/InstinctLab/blob/de6b92f01eb155d51f30e8ed1ea7aecd2eb1f057/source/instinctlab/instinctlab/tasks/parkour/config/g1/g1_parkour_target_amp_cfg.py)

The fixed front camera uses the measured G1 848×480 depth intrinsics scaled to 64×36, with Hiking’s nominal torso mount. Simulation captures at 50 Hz. Each frame goes through Hiking’s 32×18 crop, blur and 0–2.5 m normalization. The policy uses eight depth frames and a one-control-step delay. Masking and camera perturbations are off. The measured hardware profile runs at 30 Hz. Existing checkpoints load their saved camera protocol. [Measured intrinsics](src/assets/cameras/g1_depth_calibration.json)

The shared AMP pool includes the complete Hiking trajectory specified by `parkour_motion_without_run.yaml` and the 13 local walking and turn clips from the pre-alignment protocol. It samples valid ten-frame windows uniformly across all clips and keeps mirror partners separate. AMP trains on 24 collected control steps with five epochs and four minibatches per epoch. For a new run, `--walking-expert-source none` selects the Hiking-only episode reference. [Bundled Hiking source and hashes](src/assets/motions/g1/hiking_official/SOURCE_MANIFEST.json)

Training uses SuperMimic’s GenCoord velocity prediction, analytic integration, 7-transition supervision and H32 imagination. Fresh defaults use 4096 environments, a 2048×3 world-model MLP, 10 WM batches of 2048, an actor batch of 4096 and 1000 vector steps of GPU Replay. The depth encoder learns from online reconstruction and the WM predicts the next 128-dimensional latent directly. GPU frame sharing, batched kinematics, compilation and CUDA Graphs remain available. [SuperMimic training loop](https://github.com/HybridRobotics/supermimic/blob/abc124c1bd5976341034668af3d12c087f2f1f55/src/supermimic/runners/supermimic_runner.py) · [InstinctMJ visual learner](https://github.com/project-instinct/InstinctMJ/tree/03563ee02665f327425106372264070315673748/src/instinct_mj/rl/supermimic)

```bash
python scripts/train_wmac_perceptive.py --run-name hiking_unshod
```

Add `--actor-type transformer` to a fresh run to compare Actor architectures. The latest proprioception and command form the action query over eight proprioception tokens, the single-camera depth latent, the current command and the readout token. The policy outputs 29 actions. Resume uses the Actor architecture recorded in the checkpoint.

In training version 3 (architecture 9), `--iterations` counts **vector control steps**. The console reports learner updates as `iter` and shows cumulative `wall` time and average `sec_per_iter` every 50 updates by default. `--save-interval` counts learner updates and defaults to 1000. Checkpoints use `model_<iter>.pt`. `metrics.csv` records both `vector_step` and learner `iteration`. Earlier `model_step_<vector_step>.pt` checkpoints still use their saved protocol on resume or playback.

## Historical flat task

The `feature/robotics_yao` branch adds world-model AMP Actor-Critic training via
`scripts/train_wmac.py`, with policy playback, physical evaluation, benchmarks,
and ablations. See the [WMAC guide](docs/WMAC.md) for configuration and commands.
The PPO-AMP training entry point remains `scripts/train.py`.

The new WMAC `ppo_aligned` profile and PPO task `Unitree-G1-AMP-Flat-Aligned`
share 203D AMP features with gravity, ground height and foot motion cues, plus
matched common parameters. See the [alignment guide](docs/AMP_GROUNDED_ALIGNMENT.md)
for commands, checkpoint compatibility and the retained algorithm differences.
WMAC retains UTD=0.4 and PPO retains 24-step rollouts; compare cumulative
resources at the same measured physical performance rather than outer iterations.

Deployment integration code is in [ccrpRepo/wbc_fsm](https://github.com/ccrpRepo/wbc_fsm), under `MJAmp State`.

G1 AMP motion control project built on top of mjlab + rsl_rl.

Key features of the historical flat task:

- A single policy learns both locomotion (walk/run) and recovery (fall-and-get-up)
- AMP discriminator regularizes motion style and priors
- Training and deployment pipelines are consistent, with direct ONNX policy export support

## Core Idea

Instead of training separate policies for locomotion and recovery and switching between them, this project learns both capabilities in one unified policy.

Implementation highlights:

- Motion data split:
  - Walk/Run data: `src/assets/motions/g1/amp/WalkandRun`
  - Recovery data: `src/assets/motions/g1/amp/Recovery`
- Delayed termination/reset:
  - A subset of environments does not reset immediately after termination
  - These environments receive a recovery window and reset states sampled from recovery clips
- Unified AMP training:
  - One actor-critic + One AMP discriminator
  - Velocity tracking, perturbation robustness, and recovery are learned together

This reduces discontinuities caused by policy switching and yields more consistent behavior.

## Requirements

- Linux
- Python 3.11 (recommended)
- Working MuJoCo and GPU driver setup

## Quick Start

### 1. Install

```bash
conda activate mjlab
cd AMP_mjlab
python -m pip install -e .
```

### 2. Apply mjlab Patch (Optional)

If you do not apply this patch, remove `history_ordering` configuration from the code.

What this patch does:

- It adds an option for how observation history is flattened: by time (`time`) or by term (`term`).
- Default mjlab behavior supports only `term` ordering.

Patch file:

- `mjlab_patch/mjlab/managers/observation_manager.py`

Example command:

```bash
cp mjlab_patch/mjlab/managers/observation_manager.py \
  /home/crp/miniconda3/envs/mjlab/lib/python3.11/site-packages/mjlab/managers/observation_manager.py
```

### 3. List Available Tasks

```bash
python scripts/list_envs.py --keyword AMP
```

Main tasks:

- `Unitree-G1-AMP-Rough`
- `Unitree-G1-AMP-Flat`

## Training

```bash
python scripts/train.py Unitree-G1-AMP-Flat --env.scene.num-envs=4096
```

Logs are saved by default to:

- `logs/rsl_rl/g1_amp_locomotion/<time_stamp_run>/`

## Training Curve Note (Important)

- Around `2w` iterations (about 20k), the policy often suddenly learns fall-recovery behavior.
- As a result, multiple metrics in `logs` may show abrupt jumps. This is expected and not necessarily a training failure.

![Training log transition example](logs.png)

## Evaluation and Visualization

Replay with a trained checkpoint:

```bash
python scripts/play.py Unitree-G1-AMP-Rough \
  --checkpoint-file logs/rsl_rl/g1_amp_locomotion/<run_dir>/model_<iter>.pt
```

Note: ONNX export is enabled by default in both training and play workflows.

## Motion Data Preparation

CSV-to-NPZ conversion script:

```bash
python scripts/csv_to_npz.py --help
```

Recommended data layout:

- Raw CSV: `motion_data_csv/amp`
- Converted NPZ: `src/assets/motions/g1/amp/WalkandRun` and `src/assets/motions/g1/amp/Recovery`

If valid NPZ files exist in these folders, training config loads them automatically.

## Repository Structure

- `src/tasks/amp_loco`: AMP locomotion/recovery task implementation
- `src/tasks/amp_loco/config/g1`: G1 task registration, env configs, RL configs
- `src/tasks/amp_loco/mdp`: rewards, observations, events, termination logic
- `scripts/train.py`: training entry point
- `scripts/play.py`: playback entry point
- `scripts/csv_to_npz.py`: motion data conversion tool
- `mjlab_patch`: required local patch for mjlab

## Highlights

- One policy unifies walk/run and recovery skills
- AMP + velocity objective jointly optimize style and task performance
- Delayed reset with recovery sampling explicitly improves recovery ability
- End-to-end pipeline supports ONNX export for deployment

## Acknowledgements

- Thanks to [unitreerobotics/unitree_rl_mjlab](https://github.com/unitreerobotics/unitree_rl_mjlab) for open-sourcing their work and inspiration.
- Thanks to [Open-X-Humanoid/TienKung-Lab](https://github.com/Open-X-Humanoid/TienKung-Lab); the rsl_rl AMP part in this project references their implementation.
