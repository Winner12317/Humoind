#!/usr/bin/env bash
# 手册第 13.2 节的真实 GPU 短测。每次执行都会创建一次新的训练记录。
set -eo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
source reproduce/activate.sh
exec python scripts/train_wmac_perceptive.py \
  --execution-profile supermimic \
  --num-envs 16 --iterations 200 \
  --model-hidden 128 128 \
  --wm-batch-size 32 --wm-updates 1 \
  --actor-batch-size 16 --replay-steps 80 \
  --latent-recon-images 16 \
  --collect-steps 8 --log-interval 1 --save-interval 5 \
  --no-compile --no-cuda-graphs \
  --run-name repro_v3_gpu_smoke
