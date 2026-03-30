
import numpy as np
import cv2
import argparse
from pathlib import Path
import os

def inspect_extras(root):
    root = Path(root)
    print(f"Inspecting extras in {root}")
    
    # Check Pointcloud
    if (root / "pointcloud.npz").exists():
        try:
            pc = np.load(root / "pointcloud.npz")
            print("--- pointcloud.npz ---")
            print(f"Keys: {list(pc.keys())}")
            for k in pc.files:
                print(f"Key '{k}': Shape {pc[k].shape}, Type {pc[k].dtype}")
        except Exception as e:
            print(f"Error reading pointcloud.npz: {e}")

    # Check Sky Mask
    if (root / "sky_mask").exists():
        files = list((root / "sky_mask").glob("*.png"))
        if not files:
            files = list((root / "sky_mask").glob("*.jpg"))
        
        if files:
            print("--- sky_mask ---")
            print(f"Found {len(files)} masks. Example: {files[0].name}")
            img = cv2.imread(str(files[0]), cv2.IMREAD_UNCHANGED)
            if img is not None:
                print(f"Shape: {img.shape}")
                print(f"Unique values: {np.unique(img)}")
        else:
            print("sky_mask folder exists but is empty or no png/jpg found")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", help="Path to the unzipped official data")
    args = parser.parse_args()
    inspect_extras(args.data_root)
