"""Prepare original+mirrored flat walking motions, retaining original Recovery.

This utility never imports perceptive tasks or downloads Hiking assets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.wmac.motions import prepare_mirrored_dataset, verify_mirrored_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT/"src/assets/motions/g1/amp")
    parser.add_argument("--output", type=Path, default=ROOT/".wmac_cache/motions/g1_mirrored")
    args = parser.parse_args(argv)
    paths = prepare_mirrored_dataset(args.source, args.output)
    manifest = verify_mirrored_dataset(paths.root, source_root=args.source)
    print(json.dumps({"root": str(paths.root), "manifest": str(paths.manifest),
                      "walking_original_clips": manifest["walking_original_clips"],
                      "walking_total_clips": manifest["walking_total_clips"],
                      "recovery_original_clips": manifest["recovery_original_clips"],
                      "recovery_copied_byte_for_byte": manifest["recovery_copied_byte_for_byte"],
                      "turn_statistics": manifest["turn_statistics"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
