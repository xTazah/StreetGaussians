"""Inspect transform.json to understand the camera coordinate frame.

Usage: python inspect_transform.py path/to/sgn-data/<scene>
"""
import json
import sys
from pathlib import Path

import numpy as np

scene = Path(sys.argv[1])
tj = json.load(open(scene / "transform.json"))
aj = json.load(open(scene / "annotation.json"))

cam_t = np.array([f["transform_matrix"] for f in tj["frames"]])  # (N, 4, 4)
cam_pos = cam_t[:, :3, 3]
print(f"Cameras: N={cam_pos.shape[0]}")
print(f"  x: [{cam_pos[:,0].min():.2f}, {cam_pos[:,0].max():.2f}]  span={cam_pos[:,0].ptp():.2f}")
print(f"  y: [{cam_pos[:,1].min():.2f}, {cam_pos[:,1].max():.2f}]  span={cam_pos[:,1].ptp():.2f}")
print(f"  z: [{cam_pos[:,2].min():.2f}, {cam_pos[:,2].max():.2f}]  span={cam_pos[:,2].ptp():.2f}")
print(f"  first: {cam_pos[0].tolist()}")
print(f"  mean: {cam_pos.mean(axis=0).tolist()}")

# All bbox translations from annotation.json
bbox_centers = []
for frame in aj["frames"]:
    for obj in frame["objects"]:
        bbox_centers.append(obj["translation"])
bbox_centers = np.array(bbox_centers)
print(f"\nBboxes: N={bbox_centers.shape[0]} (across all frames)")
print(f"  x: [{bbox_centers[:,0].min():.2f}, {bbox_centers[:,0].max():.2f}]  span={bbox_centers[:,0].ptp():.2f}")
print(f"  y: [{bbox_centers[:,1].min():.2f}, {bbox_centers[:,1].max():.2f}]  span={bbox_centers[:,1].ptp():.2f}")
print(f"  z: [{bbox_centers[:,2].min():.2f}, {bbox_centers[:,2].max():.2f}]  span={bbox_centers[:,2].ptp():.2f}")
print(f"  mean: {bbox_centers.mean(axis=0).tolist()}")

print(f"\nCamera mean - Bbox mean: {cam_pos.mean(axis=0) - bbox_centers.mean(axis=0)}")