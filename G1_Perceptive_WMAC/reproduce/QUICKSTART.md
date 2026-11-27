# G1 Perceptive Locomotion on Rough Terrain

This project trains `Unitree-G1-AMP-Perceptive-Hiking`. Training version 5, the default, pairs the SuperMimic world model with the original Hiking task (its camera, rewards, foot measurements, terrain edges and reference motion) and adds three settings for the physical robot: vertical stair risers in the depth image, a 2.5 cm collision grid, and robot dynamics sampled around Unitree's G1. `--hiking-training-version 4` selects the Hiking task without those three. The policy uses a four-expert mixture of experts (MoE) and eight depth frames. `--hiking-training-version 3` selects the protocol before that. The repository includes the code, robot model, source motions, dependency locks, and file checksums.

## Installation and GPU smoke test

Clone or download the project to a directory without spaces on a Linux x86_64 machine with an NVIDIA driver. From the project directory containing `src/` and `reproduce/`, run:

```bash
bash reproduce/install.sh
bash reproduce/smoke-local.sh
```

The installer downloads the pinned Miniforge and Python dependencies, prepares the expert motions, and runs the CPU regression tests. The smoke test starts a fresh run with 16 parallel environments and advances them for 200 vector control steps. On an RTX 4080 with 16 GB of VRAM, it produced `model_24.pt`. The checkpoint number counts learner updates. Training logs are saved in `logs/wmac_perceptive/`.

## Full training run

The reference configuration needs either one GPU with 80 GB of memory or two with 48 GB each. For two cards, add `--device cuda:0 --sim-device cuda:1`: the simulator (about 28 GB) then runs on the second card and the learner with its replay (33–41 GB) on the first. See “Training on two GPUs” in the [README](../README.md#training-on-two-gpus). On one card:

| Resource | Estimate for 4,096 robots | Basis |
| --- | --- | --- |
| GPU memory | 60–70 GB peak, so an 80 GB card | Simulator about 28 GB (6.4 MB per robot plus 2.5 GB, measured at 256–1,536 robots). Learner about 24 GB at the default actor batch of 4,096 (7.9, 9.3 and 14.2 GB measured at 512, 1,024 and 2,048). Replay and other per-robot tensors 10–17 GB. |
| Time for 1,000,000 vector steps | About 8 days at RTX 4080 speed | 400,000 learner updates at about 0.9 s (world model 0.39 s measured; actor–critic 0.17 s and 0.28 s measured at batches of 1,024 and 2,048), plus 1,000,000 simulator steps at about 0.36 s (0.069–0.165 s measured at 256–1,536 robots). |
| Host memory | 12 GB used by the process; 32 GB recommended | Measured; it does not grow with the robot count. |
| CPU | One fast core is the limit; the process uses about 1.5 cores | Measured. About 0.05 s of every simulator step is fixed Python overhead. |
| Disk | About 45 GB per run | A checkpoint of 111 MB every 1,000 learner updates. |

None of these was run at 4,096 robots: the card used for the measurements has 16 GB, and the default learner alone exceeds it. Each figure is a straight-line extrapolation from the sizes listed.

On the 2.5 cm collision grid MuJoCo-Warp prints `height field collision overflow` whenever a fallen robot lies on the ground, a few lines per vector step at this scale. Filter them out of the console log if they get in the way, for example with `| grep -v -e "collision overflow" -e "decrease the number of hfield"`.

From the project directory, run:

```bash
source reproduce/activate.sh
python scripts/train_wmac_perceptive.py \
  --execution-profile supermimic --num-envs 4096 --seed 42 \
  --iterations 1000000 --run-name hiking_v5_from_zero
```

These defaults train a policy for the physical robot. The [README](../README.md#real-robot-deployment) covers the three real-robot settings, the RealSense image conversion and the camera mount.

Chapter 13 of the [reproduction handbook](FRAMEWORK_REPRODUCTION.md) covers training parameters, checkpoint resumption, and evaluation in the physical simulator. The stock MuJoCo-Warp 3.8.1 heightfield collision code emitted warnings during the smoke test. Section 12.6 describes an optional fix. Sources and terms for third-party code and data are listed in the [third-party notices](../THIRD_PARTY_NOTICES.md).
