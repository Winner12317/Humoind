#!/usr/bin/env bash
# train_recovery_style_v1.sh plus a larger, centred style weight on getting-up
# transitions (recovery_style_weight, src/wmac/recovery_style.py): 1.0 instead of 0.25
# for the part of the style reward above the robot's average.
#
# Of the settings tried in the PM01 pilots (docs/pm01/README.md) this one moved the
# get-up toward the reference clip (longer kneeling) without making it less reliable.
# recovery_patience_seconds (no reward for rushing up) is available too but made
# get-ups slower in those pilots; add it explicitly to try it, e.g.
#   --recovery-patience-seconds 3.5 --recovery-patience-reward 1.5
#
# Same arguments as train_recovery_style_v1.sh. To continue a checkpoint trained
# without this option, add --resume <checkpoint> --allow-recovery-style-migration.
set -eo pipefail
amp_recovery_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$amp_recovery_root/scripts/train_recovery_style_v1.sh" \
  --recovery-style-weight 1.0 \
  --run-name recovery_style_v2 "$@"
