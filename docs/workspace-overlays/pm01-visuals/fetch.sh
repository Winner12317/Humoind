#!/usr/bin/env bash
# Restore optional EngineAI Collada meshes directly from their pinned upstream source.
# The Git LFS checkout of pm01_retarget/hhtools_presets/pm01 already contains these files.
# Upstream BSD-3-Clause license: pm01_retarget/hhtools_presets/pm01/licenses/.
# Usage: bash docs/workspace-overlays/pm01-visuals/fetch.sh [mesh-directory]
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
destination="${1:-$here/../../../Human-Humanoid-Tools/configs/robots/pm01/meshes}"
source_url="https://raw.githubusercontent.com/engineai-robotics/engineai_robotics_native_sdk/6065bb219e3c15cbf93ab240fcfd737d6f007816/assets/resource/robot/pm01_edu/meshes"
mkdir -p -- "$destination"
cd -- "$destination"
tmp="$(mktemp -d .pm01-visuals.XXXXXX)"
trap 'rm -rf -- "$tmp"' EXIT
while read -r digest name; do
    if [[ -e "$name" ]]; then
        printf '%s  %s\n' "$digest" "$name" | sha256sum --check --status || {
            echo "Existing file differs from the pinned upstream mesh: $name" >&2
            exit 1
        }
        continue
    fi
    curl --fail --silent --show-error --location --retry 3 "$source_url/$name" --output "$tmp/$name"
    printf '%s  %s\n' "$digest" "$tmp/$name" | sha256sum --check --status
    mv -- "$tmp/$name" "$name"
done < "$here/SHA256SUMS"
sha256sum --check "$here/SHA256SUMS"
