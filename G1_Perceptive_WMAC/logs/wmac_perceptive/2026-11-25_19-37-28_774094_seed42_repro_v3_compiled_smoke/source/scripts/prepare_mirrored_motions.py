"""Prepare the official source for online Hiking reference playback."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from src.wmac_perceptive.motions import (prepare_mirrored_dataset,verify_mirrored_dataset,
                                        prepare_locomotion_dataset,verify_locomotion_dataset)
from src.wmac_perceptive.mirroring import DEFAULT_G1_XML
from src.wmac_perceptive.hiking_data import prepare_hiking_dataset,verify_hiking_dataset
from src.wmac_perceptive.standing_data import prepare_hiking_standing_dataset,verify_hiking_standing_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=REPO/"src/assets/motions/g1/amp")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--legacy-joint-recovery",action='store_true',help="Explicit historical walk/run+recovery dataset")
    parser.add_argument("--legacy-filtered",action='store_true',help="Historical local frame-filtered Hiking/GRAIL/local dataset")
    parser.add_argument("--official-walking-only",action='store_true',help="Build the historical FK-converted walking-only cache for older checkpoints")
    parser.add_argument("--legacy-standing",action='store_true',help="Recreate the historical Hiking plus local standing dataset")
    parser.add_argument("--robot-xml", type=Path, default=DEFAULT_G1_XML)
    args = parser.parse_args(argv)
    if sum((args.legacy_joint_recovery,args.legacy_filtered,args.official_walking_only,args.legacy_standing))>1:
        parser.error('Select only one data protocol')
    if not any((args.legacy_joint_recovery, args.legacy_filtered, args.official_walking_only, args.legacy_standing)):
        from src.wmac_perceptive.hiking_data import official_hiking_reference
        from src.wmac_perceptive.locomotion_data import HIKING_SOURCE_SHA256
        source = official_hiking_reference(args.output)
        print(json.dumps({"source": str(source), "source_sha256": HIKING_SOURCE_SHA256,
                          "source_clips": 1, "mirror_probability": .5}, indent=2))
        return
    suffix=('g1_mirrored' if args.legacy_joint_recovery else
            'g1_forward_walk_turn_v2' if args.legacy_filtered else
            'g1_hiking_walk_stand_v1' if args.legacy_standing else 'g1_hiking_walk_v1')
    output=args.output or REPO/'.perceptive_cache/motions'/suffix
    prepare=(prepare_mirrored_dataset if args.legacy_joint_recovery else
             prepare_locomotion_dataset if args.legacy_filtered else
             prepare_hiking_standing_dataset if args.legacy_standing else prepare_hiking_dataset)
    verify=(verify_mirrored_dataset if args.legacy_joint_recovery else
            verify_locomotion_dataset if args.legacy_filtered else
            verify_hiking_standing_dataset if args.legacy_standing else verify_hiking_dataset)
    paths = prepare(args.source, output, xml_path=args.robot_xml)
    manifest = verify(paths.root, source_root=args.source)
    print(json.dumps({"root": str(paths.root), "walk": str(paths.walk), "recovery": str(paths.recovery),
                      "manifest": str(paths.manifest), "walking_original_clips": manifest["walking_original_clips"],
                      "walking_total_clips": manifest["walking_total_clips"],
                      "recovery_clips": manifest["recovery_original_clips"],
                      "motion_scope":manifest.get('motion_scope','joint_recovery_v1'),
                      "motion_data_protocol":manifest.get('motion_data_protocol','filtered_walk_turn_v2'),
                      "standing_expert_source":manifest.get('standing_expert_source','none'),
                      "hiking_motion_data_used":manifest.get('hiking_motion_data_used',False),
                      "grail_motion_data_used":manifest.get('grail_motion_data_used',False),
                      "terrain_reference_categories":manifest.get('terrain_reference_categories',{}),
                      "turn_statistics": manifest["turn_statistics"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
