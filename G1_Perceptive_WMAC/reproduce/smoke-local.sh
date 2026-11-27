#!/usr/bin/env bash
# Run the GPU smoke test from Section 13.2. Each invocation starts a new training run.
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
  --run-name repro_gpu_smoke "$@"
