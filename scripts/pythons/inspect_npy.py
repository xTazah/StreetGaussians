
import numpy as np
import json
import argparse
from pathlib import Path
import os

def inspect_npy(root):
    root = Path(root)
    print(f"Inspecting {root}")
    
    # Extrinsics
    if (root / "extrinsics.npy").exists():
        try:
            ext = np.load(root / "extrinsics.npy", allow_pickle=True)
            print("--- extrinsics.npy ---")
            print(f"Shape: {ext.shape}")
            if ext.shape == ():
                ext_dict = ext.item()
                print(f"Content type: {type(ext_dict)}")
                if isinstance(ext_dict, dict):
                    print(f"Keys: {list(ext_dict.keys())}")
                    first_key = list(ext_dict.keys())[0]
                    print(f"Sample [{first_key}]: {ext_dict[first_key]}")
                    if isinstance(ext_dict[first_key], np.ndarray):
                        print(f"Sample shape: {ext_dict[first_key].shape}")
            else:
                print(f"Data sample (0): {ext[0] if len(ext)>0 else 'empty'}")
        except Exception as e:
            print(f"Error reading extrinsics.npy: {e}")
            
    # Intrinsics
    if (root / "intrinsics.npy").exists():
        try:
            intr = np.load(root / "intrinsics.npy", allow_pickle=True)
            print("--- intrinsics.npy ---")
            print(f"Shape: {intr.shape}")
            if intr.shape == ():
                intr_dict = intr.item()
                print(f"Content type: {type(intr_dict)}")
                if isinstance(intr_dict, dict):
                    print(f"Keys: {list(intr_dict.keys())}")
                    first_key = list(intr_dict.keys())[0]
                    print(f"Sample [{first_key}]: {intr_dict[first_key]}")
            else:
                print(f"Data sample (0): {intr[0] if len(intr)>0 else 'empty'}")
        except Exception as e:
            print(f"Error reading intrinsics.npy: {e}")

    # Timestamps
    if (root / "timestamps.json").exists():
        try:
            with open(root / "timestamps.json") as f:
                ts = json.load(f)
                print("--- timestamps.json ---")
                print(f"Type: {type(ts)}")
                if isinstance(ts, list):
                    print(f"Length: {len(ts)}")
                    print(f"First 5: {ts[:5]}")
                elif isinstance(ts, dict):
                    print(f"Keys: {list(ts.keys())}")
                    first_key = list(ts.keys())[0]
                    print(f"Sample [{first_key}]: {ts[first_key]}")
                    print(f"Sample type: {type(ts[first_key])}")
        except Exception as e:
            print(f"Error reading timestamps.json: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("data_root", help="Path to the unzipped official data")
    args = parser.parse_args()
    inspect_npy(args.data_root)
