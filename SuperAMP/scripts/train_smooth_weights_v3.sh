#!/usr/bin/env bash
# Intermediate coefficient experiment between gentle and balanced_v2.
# Reuses the same WMAC setup and output selection as the gentle launcher.
set -eo pipefail
amp_moderate_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$amp_moderate_root/scripts/train_smooth_weights_v2.sh" \
  --smooth-action-second-coef 0.0175 \
  --smooth-joint-velocity-coef 0.000625 \
  --smooth-torque-coef 0.000625 \
  --smooth-power-coef 0.00003125 \
  --run-name smooth_weights_v3_moderate "$@"
