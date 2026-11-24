#!/usr/bin/env bash
# WMAC with the recovery-style fixes (src/wmac/recovery_style.py) on top of the
# train_smooth_weights_v3.sh setup. The values are the ones validated in the PM01
# fine-tuning pilots; they are starting points, not tuned optima.
#
# Add the robot's task and its cleaned reference clips (scripts/clean_recovery_reference.py):
#   --task EngineAI-PM01-AMP-Flat-Aligned --reference-motion-root src/assets/motions/pm01/amp_clean
#   --task Unitree-G1-AMP-Flat-Aligned    --reference-motion-root src/assets/motions/g1/amp_clean
# To continue an existing checkpoint that was trained without the fixes, add
#   --resume <checkpoint> --allow-recovery-style-migration
set -eo pipefail
amp_recovery_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$amp_recovery_root/scripts/train_smooth_weights_v3.sh" \
  --recovery-value-grad-scale 0.1 \
  --recovery-height-reward-ratio 2.0 \
  --recovery-tempo-mode reference_limits_v1 \
  --discriminator-imagined-fraction 0.5 \
  --discriminator-penalty-points interpolated \
  --recovery-refresh-seconds 2.0 \
  --recovery-max-action-shift 0.05 \
  --run-name recovery_style_v1 "$@"
