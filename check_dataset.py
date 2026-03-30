import numpy as np
import json

# Check processed dataset
print("=== PROCESSED DATASET ===")
pts = np.load('waymo-dataset/processed/002/pointcloud.npz', allow_pickle=True)
print(f"Keys: {list(pts.keys())}")
pc = pts['pointcloud']
print(f"pointcloud: type={type(pc)}, shape={getattr(pc, 'shape', 'N/A')}")
if hasattr(pc, 'shape') and len(pc.shape) >= 2:
    print(f"  Range X: [{pc[:,0].min():.2f}, {pc[:,0].max():.2f}]")
    print(f"  Range Y: [{pc[:,1].min():.2f}, {pc[:,1].max():.2f}]")
    print(f"  Range Z: [{pc[:,2].min():.2f}, {pc[:,2].max():.2f}]")
elif hasattr(pc, 'item'):
    pc = pc.item()
    print(f"  Unpacked type: {type(pc)}")
    if isinstance(pc, dict):
        print(f"  Dict keys: {list(pc.keys())[:10]}")

# Check extrinsics
print("\n=== EXTRINSICS ===")
ext = np.load('waymo-dataset/processed/002/extrinsics.npy', allow_pickle=True)
print(f"Shape: {ext.shape}, dtype={ext.dtype}")
if len(ext.shape) == 3:
    print(f"First extrinsic:\n{ext[0]}")
    print(f"Camera positions (first 5):\n{ext[:5, :3, 3]}")

# Check intrinsics
print("\n=== INTRINSICS ===")
intr = np.load('waymo-dataset/processed/002/intrinsics.npy', allow_pickle=True)
print(f"Shape: {intr.shape}, dtype={intr.dtype}")
if len(intr.shape) >= 2:
    print(f"First intrinsic:\n{intr[0]}")

# Check timestamps
print("\n=== TIMESTAMPS ===")
with open('waymo-dataset/processed/002/timestamps.json') as f:
    ts = json.load(f)
print(f"Type: {type(ts)}")
if isinstance(ts, list):
    print(f"Count: {len(ts)}")
    print(f"First 3: {ts[:3]}")
elif isinstance(ts, dict):
    print(f"Keys: {list(ts.keys())}")

# Check track directory
import os
print("\n=== TRACK ===")
track_dir = 'waymo-dataset/processed/002/track'
files = os.listdir(track_dir)
print(f"Files: {sorted(files)[:10]}")

# Check one extrinsic txt file
print("\n=== EXTRINSIC TXT (0.txt) ===")
with open('waymo-dataset/processed/002/extrinsics/0.txt') as f:
    content = f.read()
print(content[:500])

# Check processed images
print("\n=== PROCESSED IMAGES ===")
img_dir = 'waymo-dataset/processed/002/images'
img_files = sorted(os.listdir(img_dir))
print(f"Total files: {len(img_files)}")
print(f"First 10: {img_files[:10]}")

# What script generated the processed data?
print("\n=== SGN-DATA vs PROCESSED comparison ===")
sgn_images = 0
for cam in ["FRONT", "FRONT_LEFT", "FRONT_RIGHT", "SIDE_LEFT", "SIDE_RIGHT"]:
    sgn_images += len(os.listdir(f'waymo-dataset/sgn-data/002/images/{cam}'))
proc_images = len(os.listdir('waymo-dataset/processed/002/images'))
print(f"SGN-DATA images: {sgn_images}")
print(f"Processed images: {proc_images}")
