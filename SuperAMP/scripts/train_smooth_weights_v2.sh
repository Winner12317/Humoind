#!/usr/bin/env bash
# Existing WMAC method with the gentler, short-tested coefficient candidate.
# This reduces some impact metrics; recovery tempo and all tails are not solved.
# Defaults start from scratch; resume requires an explicit checkpoint and,
# when coefficients change, --allow-smooth-reward-retune.
set -eo pipefail
amp_tuned_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd -- "$amp_tuned_root"
source scripts/activate_env.sh
exec python scripts/train_wmac.py \
  --profile supermimic_official \
  --amp-reward-mapping instantamp_centered_tanh_v1 \
  --smooth-reward-mode host_humanup_v1 \
  --smooth-action-second-coef 0.015 \
  --smooth-joint-velocity-coef 0.0005 \
  --smooth-torque-coef 0.0005 \
  --smooth-power-coef 0.000025 \
  --run-name smooth_weights_v2 "$@"
