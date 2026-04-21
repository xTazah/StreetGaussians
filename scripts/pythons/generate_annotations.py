"""
Generate annotation.json and per-object PLY files from processed Waymo data.

Reads track_info.txt (ego-local bounding boxes), ego_pose (ego-to-world transforms),
timestamps.json, and pointcloud.npz, then:
  1. Transforms bounding boxes from ego to world coordinates
  2. Writes annotation.json
  3. Extracts per-object lidar points and saves as .ply files

Usage:
    python scripts/pythons/generate_annotations.py \
        --processed_root waymo-dataset/processed/002 \
        --output_root waymo-dataset/sgn-data/002
"""

import argparse
import json
import os
from pathlib import Path
from collections import defaultdict

import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    import open3d as o3d
except ImportError:
    raise ImportError("open3d is required. Install with: pip install open3d")


# Constants matching the codebase
MIN_MOVING_SPEED = 0.2  # from extract_waymo.py
EXP_RATE = np.array([1.3, 1.3, 1.1])  # from dynamic_annotation.py
MIN_POINTS_PER_OBJECT = 10000  # from dynamic_annotation.py load_object_3D_points
FILTER_LABEL = ["car"]  # from dynamic_annotation.py

# Mapping from track_info.txt class names to annotation.json type names
# (matching extract_waymo.py _box_type_to_str)
CLASS_MAP = {
    "vehicle": "car",
    "pedestrian": "pedestrian",
    "cyclist": "cyclist",
    "sign": "sign",
    "unknown": "unknown",
}


def load_timestamps(processed_root: Path) -> dict:
    """Load frame timestamps. Returns dict: frame_id_str -> timestamp_float."""
    with open(processed_root / "timestamps.json") as f:
        data = json.load(f)
    return data["FRAME"]


def load_ego_pose(processed_root: Path, frame_idx: int) -> np.ndarray:
    """Load 4x4 ego-to-world transform for a frame."""
    pose_file = processed_root / "ego_pose" / f"{frame_idx:06d}.txt"
    return np.loadtxt(pose_file).reshape(4, 4)


def load_track_ids(processed_root: Path) -> dict:
    """Load track_ids.json mapping string GIDs to integer track IDs.
    Returns both forward and reverse mappings."""
    with open(processed_root / "track" / "track_ids.json") as f:
        gid_to_int = json.load(f)
    int_to_gid = {v: k for k, v in gid_to_int.items()}
    return gid_to_int, int_to_gid


def parse_track_info(processed_root: Path):
    """Parse track_info.txt. Returns list of dicts per row."""
    track_file = processed_root / "track" / "track_info.txt"
    rows = []
    with open(track_file) as f:
        header = f.readline()  # skip header
        for line in f:
            parts = line.strip().split()
            if len(parts) < 12:
                continue
            rows.append({
                "frame_id": int(parts[0]),
                "track_id": int(parts[1]),
                "object_class": parts[2],
                "alpha": float(parts[3]),
                "box_height": float(parts[4]),
                "box_width": float(parts[5]),
                "box_length": float(parts[6]),
                "box_center_x": float(parts[7]),
                "box_center_y": float(parts[8]),
                "box_center_z": float(parts[9]),
                "box_heading": float(parts[10]),
                "speed": float(parts[11]),
            })
    return rows


def load_pointcloud(processed_root: Path) -> dict:
    """Load pointcloud.npz. Returns dict: frame_index(int) -> (N,3) array."""
    data = np.load(processed_root / "pointcloud.npz", allow_pickle=True)
    return data["pointcloud"].item()


def ego_to_world_box(track_row: dict, ego_pose: np.ndarray):
    """Transform a bounding box from ego-vehicle frame to world frame.

    Returns (center_world, quat_wxyz, size_lwh, obj_type, is_moving).
    """
    cx = track_row["box_center_x"]
    cy = track_row["box_center_y"]
    cz = track_row["box_center_z"]
    heading = track_row["box_heading"]
    speed = track_row["speed"]

    # Center: ego -> world
    center_ego = np.array([cx, cy, cz, 1.0])
    center_world = ego_pose @ center_ego

    # Rotation: heading is yaw around Z in ego frame
    rot_ego = R.from_euler("xyz", [0, 0, heading], degrees=False).as_matrix()
    rot_world = ego_pose[:3, :3] @ rot_ego
    quat_xyzw = R.from_matrix(rot_world).as_quat()  # scipy returns [x, y, z, w]
    quat_wxyz = [float(quat_xyzw[3]), float(quat_xyzw[0]),
                 float(quat_xyzw[1]), float(quat_xyzw[2])]

    # Size: [length, width, height] for annotation.json
    size_lwh = [track_row["box_length"], track_row["box_width"], track_row["box_height"]]

    obj_type = CLASS_MAP.get(track_row["object_class"], track_row["object_class"])
    is_moving = bool(speed > MIN_MOVING_SPEED)

    return center_world[:3].tolist(), quat_wxyz, size_lwh, obj_type, is_moving


def build_annotation_json(processed_root: Path, int_to_gid: dict):
    """Build the annotation.json structure.

    Returns the annotation dict and a per-frame/per-object world-frame box lookup
    for point extraction.
    """
    timestamps = load_timestamps(processed_root)
    track_rows = parse_track_info(processed_root)

    # Group track rows by frame_id
    rows_by_frame = defaultdict(list)
    for row in track_rows:
        rows_by_frame[row["frame_id"]].append(row)

    # Sorted frame IDs from timestamps
    frame_ids_sorted = sorted(timestamps.keys(), key=lambda x: int(x))

    frames = []
    # world_boxes[frame_idx] = list of (gid, center_world, rot_world_3x3, size_lwh, obj_type, is_moving)
    world_boxes_by_frame = {}

    for frame_id_str in frame_ids_sorted:
        frame_idx = int(frame_id_str)
        timestamp = timestamps[frame_id_str]
        ego_pose = load_ego_pose(processed_root, frame_idx)

        objects_list = []
        world_boxes = []

        for row in rows_by_frame.get(frame_idx, []):
            track_int_id = row["track_id"]
            gid = int_to_gid.get(track_int_id)
            if gid is None:
                # Track ID not in track_ids.json, skip
                continue

            center_world, quat_wxyz, size_lwh, obj_type, is_moving = ego_to_world_box(row, ego_pose)

            objects_list.append({
                "type": obj_type,
                "gid": gid,
                "translation": center_world,
                "size": size_lwh,
                "rotation": quat_wxyz,
                "is_moving": is_moving,
            })

            # Store world-frame box info for point extraction
            heading = row["box_heading"]
            rot_ego = R.from_euler("xyz", [0, 0, heading], degrees=False).as_matrix()
            rot_world = ego_pose[:3, :3] @ rot_ego
            world_boxes.append({
                "gid": gid,
                "center": np.array(center_world),
                "rot": rot_world,
                "size": np.array(size_lwh),  # [length, width, height]
                "type": obj_type,
                "is_moving": is_moving,
            })

        frames.append({
            "timestamp": timestamp,
            "objects": objects_list,
        })
        world_boxes_by_frame[frame_idx] = world_boxes

    annotation = {"frames": frames}
    return annotation, world_boxes_by_frame


def get_obb_corners(center, size, rot):
    """Get 8 corners of an oriented bounding box (expanded by EXP_RATE).

    Args:
        center: (3,) world center
        size: [length, width, height]
        rot: (3,3) rotation matrix (world frame)
    """
    exp_size = np.array(size) * EXP_RATE
    l, w, h = exp_size
    dx, dy, dz = l / 2.0, w / 2.0, h / 2.0

    corners_local = np.array([
        [ dx,  dy,  dz],
        [-dx,  dy,  dz],
        [-dx, -dy,  dz],
        [ dx, -dy,  dz],
        [ dx,  dy, -dz],
        [-dx,  dy, -dz],
        [-dx, -dy, -dz],
        [ dx, -dy, -dz],
    ])

    corners_world = (rot @ corners_local.T).T + center
    return corners_world


def extract_object_points(processed_root: Path, world_boxes_by_frame: dict):
    """Extract per-object lidar points from pointcloud data.

    For each frame, transforms ego-frame points to world, then finds
    points inside each object's bounding box (expanded by EXP_RATE),
    and transforms them to the object's local coordinate frame.

    Returns dict: gid -> {"xyz": list of (3,) arrays}.
    """
    print("Loading point cloud data...")
    pointcloud = load_pointcloud(processed_root)

    obj_points = defaultdict(list)  # gid -> list of (3,) local points

    frame_indices = sorted(world_boxes_by_frame.keys())
    total_frames = len(frame_indices)

    for count, frame_idx in enumerate(frame_indices):
        if (count + 1) % 20 == 0 or count == 0:
            print(f"  Processing frame {count + 1}/{total_frames}...")

        boxes = world_boxes_by_frame[frame_idx]
        # Filter to only moving objects with types in FILTER_LABEL
        moving_boxes = [
            b for b in boxes
            if b["is_moving"] and b["type"] in FILTER_LABEL
        ]
        if not moving_boxes or frame_idx not in pointcloud:
            continue

        # Load ego pose and points
        ego_pose = load_ego_pose(processed_root, frame_idx)
        pts_ego = pointcloud[frame_idx]  # (N, 3) in ego frame

        # Filter NaN and extreme values
        valid = ~np.isnan(pts_ego).any(axis=1)
        pts_ego = pts_ego[valid]

        # Transform points to world frame
        pts_world = (ego_pose[:3, :3] @ pts_ego.T).T + ego_pose[:3, 3]

        # Create open3d point cloud for efficient cropping
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_world)

        for box in moving_boxes:
            # Build OBB from expanded corners
            corners = get_obb_corners(box["center"], box["size"], box["rot"])
            obb = o3d.geometry.OrientedBoundingBox.create_from_points(
                o3d.utility.Vector3dVector(corners)
            )

            inlier_indices = obb.get_point_indices_within_bounding_box(pcd.points)
            if len(inlier_indices) == 0:
                continue

            inlier_pts_world = pts_world[inlier_indices]

            # Transform to object-local frame
            o2w = np.eye(4)
            o2w[:3, :3] = box["rot"]
            o2w[:3, 3] = box["center"]
            w2o = np.linalg.inv(o2w)

            # Apply w2o transform
            ones = np.ones((inlier_pts_world.shape[0], 1))
            pts_homo = np.hstack([inlier_pts_world, ones])
            pts_local = (w2o @ pts_homo.T).T[:, :3]

            obj_points[box["gid"]].append(pts_local)

    # Concatenate all points per object
    result = {}
    for gid, pts_list in obj_points.items():
        all_pts = np.concatenate(pts_list, axis=0)
        result[gid] = all_pts

    return result


def save_ply_files(obj_points: dict, output_root: Path):
    """Save per-object PLY files. Only saves objects with >= MIN_POINTS_PER_OBJECT points."""
    ply_dir = output_root / "aggregate_lidar" / "dynamic_objects"
    ply_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    skipped = 0
    for gid, pts in obj_points.items():
        if pts.shape[0] < MIN_POINTS_PER_OBJECT:
            print(f"  Skipping {gid}: only {pts.shape[0]} points (need {MIN_POINTS_PER_OBJECT})")
            skipped += 1
            continue

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts.astype(np.float32))
        # Assign random colors (consistent with dynamic_annotation.py fallback)
        colors = np.random.rand(pts.shape[0], 3).astype(np.float32)
        pcd.colors = o3d.utility.Vector3dVector(colors)

        ply_path = ply_dir / f"{gid}.ply"
        o3d.io.write_point_cloud(str(ply_path), pcd)
        saved += 1
        print(f"  Saved {ply_path.name}: {pts.shape[0]} points")

    print(f"PLY summary: {saved} saved, {skipped} skipped (< {MIN_POINTS_PER_OBJECT} points)")


def main():
    parser = argparse.ArgumentParser(
        description="Generate annotation.json and per-object PLY files from processed Waymo data"
    )
    parser.add_argument(
        "--processed_root", type=str, required=True,
        help="Path to processed data directory (e.g., waymo-dataset/processed/002)"
    )
    parser.add_argument(
        "--output_root", type=str, required=True,
        help="Path to output directory (e.g., waymo-dataset/sgn-data/002)"
    )
    parser.add_argument(
        "--skip_ply", action="store_true",
        help="Skip PLY extraction (only generate annotation.json)"
    )
    args = parser.parse_args()

    processed_root = Path(args.processed_root).resolve()
    output_root = Path(args.output_root).resolve()

    # Validate input paths
    assert (processed_root / "track" / "track_info.txt").exists(), \
        f"track_info.txt not found in {processed_root / 'track'}"
    assert (processed_root / "timestamps.json").exists(), \
        f"timestamps.json not found in {processed_root}"
    assert (processed_root / "ego_pose").is_dir(), \
        f"ego_pose directory not found in {processed_root}"
    assert (processed_root / "track" / "track_ids.json").exists(), \
        f"track_ids.json not found in {processed_root / 'track'}"

    output_root.mkdir(parents=True, exist_ok=True)

    # Load track ID mappings
    print("Loading track ID mappings...")
    gid_to_int, int_to_gid = load_track_ids(processed_root)
    print(f"  Found {len(gid_to_int)} tracked objects")

    # Build annotation.json
    print("Building annotation.json...")
    annotation, world_boxes_by_frame = build_annotation_json(processed_root, int_to_gid)
    num_frames = len(annotation["frames"])
    num_objects_total = sum(len(f["objects"]) for f in annotation["frames"])
    num_moving = sum(
        1 for f in annotation["frames"]
        for o in f["objects"]
        if o["is_moving"] and o["type"] in FILTER_LABEL
    )
    print(f"  {num_frames} frames, {num_objects_total} total object entries, {num_moving} moving+filtered entries")

    # Save annotation.json
    anno_path = output_root / "annotation.json"
    with open(anno_path, "w") as f:
        json.dump(annotation, f, indent=2)
    print(f"Saved {anno_path}")

    # Extract and save PLY files
    if not args.skip_ply:
        assert (processed_root / "pointcloud.npz").exists(), \
            f"pointcloud.npz not found in {processed_root}"

        print("Extracting per-object lidar points...")
        obj_points = extract_object_points(processed_root, world_boxes_by_frame)
        print(f"  Extracted points for {len(obj_points)} objects")

        print("Saving PLY files...")
        save_ply_files(obj_points, output_root)
    else:
        print("Skipping PLY extraction (--skip_ply)")

    print("Done!")


if __name__ == "__main__":
    main()
