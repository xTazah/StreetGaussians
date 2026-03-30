
import os
import shutil
import json
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import cv2

# Waymo Camera Order (Typical 0-based index in lists)
# 0: FRONT
# 1: FRONT_LEFT
# 2: FRONT_RIGHT
# 3: SIDE_LEFT
# 4: SIDE_RIGHT

CAM_MAPPING = {
    0: 'FRONT',
    1: 'FRONT_LEFT',
    2: 'FRONT_RIGHT',
    3: 'SIDE_LEFT',
    4: 'SIDE_RIGHT'
}

def convert_data(source_root, dest_root):
    source_root = Path(source_root)
    dest_path = Path(dest_root)
    dest_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Converting from {source_root} to {dest_path}")
    
    # 1. Load Metadata
    print("Loading metadata...")
    try:
        extrinsics_dict = np.load(source_root / "extrinsics.npy", allow_pickle=True).item()
        intrinsics_dict = np.load(source_root / "intrinsics.npy", allow_pickle=True).item()
        timestamps_dict = None
        if (source_root / "timestamps.json").exists():
            with open(source_root / "timestamps.json", 'r') as f:
                timestamps_dict = json.load(f)
    except Exception as e:
        print(f"Failed to load metadata: {e}")
        return

    # 2. Prepare transform.json structure
    transform_json = {
        "sensor_params": {
            "camera_order": []
        },
        "frames": []
    }
    
    # Helper to clean camera name if needed
    def get_cam_key(cam_name):
        # The keys in npy might differ slightly? 
        # Inspect output says keys are ['FRONT_LEFT', ...], which matches our mapping values.
        return cam_name

    # 3. Process each camera
    # We iterate 0..4 to match file suffixes
    
    for cam_idx, cam_name in CAM_MAPPING.items():
        print(f"Processing Camera: {cam_name}")
        
        # Create image folder
        cam_image_dir = dest_path / "images" / cam_name
        cam_image_dir.mkdir(parents=True, exist_ok=True)
        
        # Get Intrinsic/Extrinsic for this camera
        # Note: Extrinsics in npy are per-frame (N, 4, 4)
        # Intrinsics in npy are per-frame (N, 3, 3)
        
        cam_exts = extrinsics_dict.get(cam_name)
        cam_ints = intrinsics_dict.get(cam_name)
        
        if cam_exts is None or cam_ints is None:
            print(f"Warning: No metadata for {cam_name}, skipping.")
            continue
            
        num_frames = len(cam_exts)
        
        # Update sensor_params (Static Global info)
        # We take the first frame's intrinsic as the static intrinsic for the camera
        # We take the first frame's extrinsic as the 'reference' extrinsic relative to global, 
        # BUT StreetGaussian expects 'extrinsic' field in sensor_params for calculating relative rig pose.
        # Since we have c2w per frame, we'll put the first frame's c2w here, and run_colmap.py will compute relative.
        # This assumes the relative pose between cameras is constant (Rigid Rig).
        
        transform_json["sensor_params"]["camera_order"].append(cam_name)
        transform_json["sensor_params"][cam_name] = {
            "height": 1280, # Standard Waymo
            "width": 1920,  # Standard Waymo
            "camera_intrinsic": cam_ints[0].tolist(),
            "extrinsic": cam_exts[0].tolist() 
        }

        # Process Frames
        # Source files: 000000_0.png, 000001_0.png ...
        
        # We need to find all matching files for this camera index
        # Expecting filenames like: {frame_idx:06d}_{cam_idx}.png
        
        for frame_idx in range(num_frames):
            frame_str = f"{frame_idx:06d}"
            src_fname = f"{frame_str}_{cam_idx}.png"
            src_file = source_root / "images" / src_fname
            
            if not src_file.exists():
                # Try jpg?
                src_file = source_root / "images" / f"{frame_str}_{cam_idx}.jpg"
            
            if not src_file.exists():
                # print(f"Missing frame {src_fname}")
                continue
                
            # Copy Image
            dst_fname = f"{frame_str}.png" # Or keep original extension
            dst_file = cam_image_dir / dst_fname
            shutil.copy2(src_file, dst_file)
            
            # Add to frames list
            # Timestamp
            ts = 0.0
            if timestamps_dict:
                # Try to get timestamp. inspect output showed 'FRAME' key mapping frame_id to float
                # Also keys for cameras.
                if cam_name in timestamps_dict:
                     ts = timestamps_dict[cam_name].get(frame_str, 0.0)
                elif 'FRAME' in timestamps_dict:
                     ts = timestamps_dict['FRAME'].get(frame_str, 0.0)
            
            c2w = cam_exts[frame_idx]
            
            frame_entry = {
                "file_path": f"images/{cam_name}/{dst_fname}",
                "transform_matrix": c2w.tolist(),
                "timestamp": ts,
                "camera_id": cam_idx + 1, # Colmap/SGN might expect 1-based ID matching order? Or maybe distinct?
                # extract_waymo produces unique per camera.
                # run_colmap.py uses 'camera' field? No, it uses 'file_path'.
            }
            transform_json["frames"].append(frame_entry)

    # 4. Save transform.json
    with open(dest_path / "transform.json", "w") as f:
        json.dump(transform_json, f, indent=4)
    print(f"Saved transform.json to {dest_path / 'transform.json'}")
    
    # 5. Handle Masks (Optional but helpful)
    # Convert 'dynamic_mask' to 'masks' folder (Inverted: 0 for dynamic, 255 for static)
    # The source likely has 0 for background, 1 for dynamic? Or 255?
    # We will inspect one if it exists.
    
    mask_source = source_root / "dynamic_mask"
    if mask_source.exists():
        print("Processing Dynamic Masks...")
        dest_mask_dir = dest_path / "masks"
        
        for cam_idx, cam_name in CAM_MAPPING.items():
            cam_mask_dir = dest_mask_dir / cam_name
            cam_mask_dir.mkdir(parents=True, exist_ok=True)
            
            # List files
            # Expecting same naming convention {frame}_{cam}.png
            # We iterate known frames
            
            # Using same range as extrinsics
            cam_exts = extrinsics_dict.get(cam_name)
            if cam_exts is None: continue
            
            for frame_idx in tqdm(range(len(cam_exts)), desc=f"Masks {cam_name}"):
                frame_str = f"{frame_idx:06d}"
                src_fname = f"{frame_str}_{cam_idx}.png"
                src_file = mask_source / src_fname
                
                if src_file.exists():
                    # Read, Invert, Save
                    # If mask is boolean: True (Dynamic) -> Black (0), False (Static) -> White (255)
                    mask = cv2.imread(str(src_file), cv2.IMREAD_GRAYSCALE)
                    if mask is not None:
                        # Assuming source: >0 is dynamic object
                        # Target: 0 is ignore (dynamic), 255 is keep
                        
                        # Create 'keep' mask
                        # If existing mask has values > 0 for objects:
                        new_mask = np.ones_like(mask) * 255
                        new_mask[mask > 0] = 0 
                        
                        cv2.imwrite(str(cam_mask_dir / f"{frame_str}.png"), new_mask)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True, help="Path to unzipped official data (e.g. waymo-dataset/processed/002)")
    parser.add_argument("--dest", required=True, help="Path to output folder suitable for street-gaussians (e.g. waymo-dataset/sgn-format/002)")
    args = parser.parse_args()
    convert_data(args.source, args.dest)
