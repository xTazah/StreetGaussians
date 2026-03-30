
import os
import glob
import numpy as np
import json
import argparse
from pathlib import Path

def inspect(root):
    root = Path(root)
    print(f"Inspecting {root}")
    
    # Check folders
    folders = [f.name for f in root.iterdir() if f.is_dir()]
    print(f"Folders found: {folders}")
    
    # Check Extrinsics
    if (root / "extrinsics.npy").exists():
        try:
            ext = np.load(root / "extrinsics.npy", allow_pickle=True)
            print(f"extrinsics.npy shape: {ext.shape}, type: {type(ext)}")
            if len(ext.shape) > 0:
                print(f"First element: {ext[0]}")
        except Exception as e:
            print(f"Error reading extrinsics.npy: {e}")
    elif (root / "extrinsics").exists():
        files = list((root / "extrinsics").glob("*"))
        print(f"Found {len(files)} files in extrinsics folder. Example: {files[0].name}")
        
    # Check Intrinsics
    if (root / "intrinsics.npy").exists():
        try:
            intr = np.load(root / "intrinsics.npy", allow_pickle=True)
            print(f"intrinsics.npy shape: {intr.shape}")
            if len(intr.shape) > 0:
                 print(f"First element: {intr[0]}")
        except Exception as e:
            print(f"Error reading intrinsics.npy: {e}")
            
    # Check Timestamps
    if (root / "timestamps.json").exists():
        try:
            with open(root / "timestamps.json") as f:
                ts = json.load(f)
                print(f"timestamps.json type: {type(ts)}")
                if isinstance(ts, list):
                    print(f"Length: {len(ts)}")
                    print(f"First few: {ts[:3]}")
                elif isinstance(ts, dict):
                    print(f"Keys: {ts.keys()}")
        except Exception as e:
            print(f"Error reading timestamps.json: {e}")

    # Check Track
    if (root / "track").exists():
        files = list((root / "track").glob("*"))
        print(f"Found {len(files)} files in track folder. Example: {files[0].name}")
        # Try reading one
        if files[0].suffix == '.json':
            with open(files[0]) as f:
                print(f"Content of {files[0].name}: {f.read()[:200]}")
        elif files[0].suffix == '.txt':
            with open(files[0]) as f:
                print(f"Content of {files[0].name}: {f.read()[:200]}")
                
    # Check images
    if (root / "images").exists():
        img_folders = list((root / "images").iterdir())
        print(f"Image folders: {[f.name for f in img_folders]}")
        for img_f in img_folders:
            imgs = list(img_f.glob("*"))
            print(f"  {img_f.name}: {len(imgs)} images")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", help="Path to the unzipped official data")
    args = parser.parse_args()
    inspect(args.data_root)
