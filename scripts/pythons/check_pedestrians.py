"""
Phase 1 pedestrian extension: verification script.

Reports per-class object counts in annotation.json and per-class PLY counts in
aggregate_lidar/dynamic_objects/. Exits non-zero if no pedestrians made it
through the pipeline, so this can be used as a CI-style smoke check after
running generate_annotations.py.

Usage:
    python scripts/pythons/check_pedestrians.py \
        --sgn_data_root waymo-dataset/sgn-data/002

The sgn_data_root must contain:
    annotation.json
    aggregate_lidar/dynamic_objects/<gid>.ply  (per surviving object)
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sgn_data_root", type=str, required=True,
                        help="Path to sgn-data scene root (containing annotation.json)")
    parser.add_argument("--require_pedestrians", action="store_true", default=True,
                        help="Exit non-zero if no pedestrian PLYs found (default: on)")
    parser.add_argument("--no_require_pedestrians", action="store_false",
                        dest="require_pedestrians",
                        help="Disable the pedestrian-required exit code")
    args = parser.parse_args()

    root = Path(args.sgn_data_root).resolve()
    anno_path = root / "annotation.json"
    ply_dir = root / "aggregate_lidar" / "dynamic_objects"

    if not anno_path.exists():
        print(f"ERROR: annotation.json not found at {anno_path}")
        sys.exit(2)

    # Count annotation entries per class.
    with open(anno_path) as f:
        annotation = json.load(f)

    # objects-per-class: total entries (per-frame), and unique gids.
    entries_per_class = defaultdict(int)
    moving_entries_per_class = defaultdict(int)
    gids_per_class = defaultdict(set)
    moving_gids_per_class = defaultdict(set)

    for frame in annotation["frames"]:
        for obj in frame["objects"]:
            cls = obj["type"]
            entries_per_class[cls] += 1
            gids_per_class[cls].add(obj["gid"])
            if obj.get("is_moving", False):
                moving_entries_per_class[cls] += 1
                moving_gids_per_class[cls].add(obj["gid"])

    # Count PLY files. A PLY file's class is recovered by cross-referencing its
    # gid against the annotation. (PLYs are filename-keyed by gid only.)
    gid_to_class = {}
    for cls, gids in gids_per_class.items():
        for gid in gids:
            gid_to_class[gid] = cls

    plys_per_class = defaultdict(int)
    plys_unmatched = []
    if ply_dir.exists():
        for ply_path in sorted(ply_dir.glob("*.ply")):
            gid = ply_path.stem
            cls = gid_to_class.get(gid)
            if cls is None:
                plys_unmatched.append(gid)
            else:
                plys_per_class[cls] += 1
    else:
        print(f"WARNING: PLY directory not found at {ply_dir}")

    # Report.
    print(f"=== Phase 1 pedestrian-extension verification ===")
    print(f"Scene root: {root}")
    print(f"Frames in annotation.json: {len(annotation['frames'])}")
    print()
    print(f"{'Class':<14} {'Entries':>8} {'Unique':>8} {'Moving':>8} "
          f"{'MovGids':>8} {'PLYs':>6}")
    all_classes = sorted(set(list(entries_per_class.keys()) + list(plys_per_class.keys())))
    for cls in all_classes:
        print(f"{cls:<14} {entries_per_class[cls]:>8d} "
              f"{len(gids_per_class[cls]):>8d} {moving_entries_per_class[cls]:>8d} "
              f"{len(moving_gids_per_class[cls]):>8d} {plys_per_class[cls]:>6d}")

    if plys_unmatched:
        print(f"\nWARNING: {len(plys_unmatched)} PLY file(s) have gids not present in "
              f"annotation.json (orphaned). First few: {plys_unmatched[:5]}")

    # Exit code logic.
    n_pedestrian_plys = plys_per_class.get("pedestrian", 0)
    n_pedestrian_moving = len(moving_gids_per_class.get("pedestrian", set()))

    print()
    if n_pedestrian_plys == 0:
        if n_pedestrian_moving == 0:
            print("INFO: No moving pedestrians in this scene. Phase 1 cannot exercise "
                  "pedestrian rendering on this scene; pick a different scene if you "
                  "want B0 vs M to have signal.")
        else:
            print(f"WARNING: {n_pedestrian_moving} moving pedestrian gid(s) in "
                  f"annotation.json, but no pedestrian PLYs were saved. "
                  f"Check MIN_POINTS_PER_CLASS in generate_annotations.py.")
        if args.require_pedestrians:
            sys.exit(1)
    else:
        print(f"OK: {n_pedestrian_plys} pedestrian PLY(s) saved out of "
              f"{n_pedestrian_moving} moving pedestrian gid(s).")

    sys.exit(0)


if __name__ == "__main__":
    main()
