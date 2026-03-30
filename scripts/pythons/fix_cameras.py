import json
import numpy as np
import cv2
from pathlib import Path

# Paths
data_root = Path("waymo-dataset/sgn-data/002")
transform_path = data_root / "transform.json"
images_root = data_root / "images"

def fix_cameras():
    with open(transform_path, 'r') as f:
        meta = json.load(f)
    
    sensor_params = meta["sensor_params"]
    modified = False
    
    for cam_name in sensor_params["camera_order"]:
        if cam_name not in sensor_params:
            continue
            
        params = sensor_params[cam_name]
        meta_w = params["width"]
        meta_h = params["height"]
        
        # Find an image
        cam_dir = images_root / cam_name
        img_files = list(cam_dir.glob("*.png"))
        if not img_files:
            img_files = list(cam_dir.glob("*.jpg"))
            
        if not img_files:
            print(f"No images for {cam_name}")
            continue
            
        img_path = img_files[0]
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to read {img_path}")
            continue
            
        h, w = img.shape[:2]
        
        if w != meta_w or h != meta_h:
            print(f"Mismatch for {cam_name}: Meta({meta_w}x{meta_h}) vs Image({w}x{h})")
            
            # Assume vertical crop
            if w == meta_w and h < meta_h:
                diff = meta_h - h
                # Assume center crop
                crop_top = diff / 2.0
                
                print(f"  Assuming vertical center crop. Offset: {crop_top}")
                
                # Update params
                params["width"] = w
                params["height"] = h
                
                # Update Cy
                intr = np.array(params["camera_intrinsic"])
                # [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
                # cx stays same
                # cy changes
                old_cy = intr[1, 2]
                new_cy = old_cy - crop_top
                intr[1, 2] = new_cy
                
                print(f"  Cy: {old_cy} -> {new_cy}")
                params["camera_intrinsic"] = intr.tolist()
                modified = True
            else:
                print("  Complex mismatch, not just vertical crop. Updating W/H but not intrinsics (risky).")
                params["width"] = w
                params["height"] = h
                modified = True
        else:
            print(f"Camera {cam_name} OK.")

    if modified:
        print("Saving updated transform.json...")
        with open(data_root / "transform.json", 'w') as f:
            json.dump(meta, f, indent=4)
            
        # Regenerate cameras.txt
        print("Regenerating cameras.txt...")
        colmap_dir = data_root / "colmap" / "sparse" / "0"
        colmap_dir.mkdir(parents=True, exist_ok=True)
        
        with open(colmap_dir / "cameras.txt", "w") as f:
            f.write("# Camera list with one line of data per camera.\n")
            f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
            
            cam_order = meta["sensor_params"]["camera_order"]
            for idx, cam_name in enumerate(cam_order):
                colmap_cam_id = idx + 1 # 1-based
                
                params = meta["sensor_params"][cam_name]
                w = params["width"]
                h = params["height"]
                intr = np.array(params["camera_intrinsic"])
                
                fx = intr[0, 0]
                fy = intr[1, 1]
                cx = intr[0, 2]
                cy = intr[1, 2]
                f_val = (fx + fy) / 2.0
                
                f.write(f"{colmap_cam_id} SIMPLE_PINHOLE {w} {h} {f_val} {cx} {cy}\n")
        print("Done.")
    else:
        print("No changes needed.")

if __name__ == "__main__":
    fix_cameras()
