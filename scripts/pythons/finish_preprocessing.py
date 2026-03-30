
import os
import json
import numpy as np
import argparse
from pathlib import Path
from scipy.spatial.transform import Rotation as R
import cv2
from tqdm import tqdm

def rotmat2qvec(R_mat):
    r = R.from_matrix(R_mat)
    # scipy returns (x, y, z, w)
    x, y, z, w = r.as_quat()
    # colmap expects (w, x, y, z)
    return np.array([w, x, y, z])

def finish_data(sgn_data_root, source_data_root):
    root = Path(sgn_data_root)
    source = Path(source_data_root)
    
    colmap_dir = root / "colmap" / "sparse" / "0"
    colmap_dir.mkdir(parents=True, exist_ok=True)
    
    segs_dir = root / "segs"
    segs_dir.mkdir(parents=True, exist_ok=True)

    print(f"Setting up scene in {root}...")

    # 1. Generate Segments from Sky Mask
    # SemanticType: DEFAULT=0, SKY=2
    # Input Sky Mask: 0 (Non-Sky), 255 (Sky)
    
    sky_mask_source = source / "sky_mask"
    if sky_mask_source.exists():
        print("Integrating Sky Masks into Segments...")
        # We need to match filenames. 
        # Convert script mapped: images/CAM/frame.png
        # Source sky mask: frame_camIdx.png (e.g. 000000_0.png)
        # We need to look up which camera maps to which index.
        CAM_MAPPING = {0: 'FRONT', 1: 'FRONT_LEFT', 2: 'FRONT_RIGHT', 3: 'SIDE_LEFT', 4: 'SIDE_RIGHT'}
        
        # We iterate through all sky masks
        mask_files = list(sky_mask_source.glob("*.png")) + list(sky_mask_source.glob("*.jpg"))
        
        for mf in tqdm(mask_files, desc="Segments"):
            # Parse filename: 000000_0.png -> frame=000000, cam_idx=0
            try:
                stem = mf.stem # 000000_0
                parts = stem.split('_')
                if len(parts) != 2: continue
                
                frame_id = parts[0]
                cam_idx = int(parts[1])
                cam_name = CAM_MAPPING.get(cam_idx)
                
                if not cam_name: continue
                
                # Output dir
                out_folder = segs_dir / cam_name
                out_folder.mkdir(parents=True, exist_ok=True)
                
                # Read mask
                img = cv2.imread(str(mf), cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                
                # Create segment map (0=Default, 2=Sky)
                seg_map = np.zeros_like(img, dtype=np.uint8)
                seg_map[img > 127] = 2 
                
                # Save
                cv2.imwrite(str(out_folder / f"{frame_id}.png"), seg_map)
                
            except Exception as e:
                print(f"Error processing mask {mf.name}: {e}")

    # 2. Generate COLMAP files from transform.json
    print("Generating COLMAP sparse model...")
    transform_path = root / "transform.json"
    if not transform_path.exists():
        print("Error: transform.json not found! Run conversion first.")
        return

    with open(transform_path) as f:
        meta = json.load(f)

    # cameras.txt
    # CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]
    # StreetGaussians run_colmap uses SIMPLE_PINHOLE: f, cx, cy
    
    cam_id_map = {} # name -> id
    
    with open(colmap_dir / "cameras.txt", "w") as f:
        f.write("# Camera list with one line of data per camera.\n")
        f.write("#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        
        cam_order = meta["sensor_params"]["camera_order"]
        for idx, cam_name in enumerate(cam_order):
            colmap_cam_id = idx + 1 # 1-based
            cam_id_map[cam_name] = colmap_cam_id
            
            params = meta["sensor_params"][cam_name]
            w = params["width"]
            h = params["height"]
            intr = np.array(params["camera_intrinsic"])
            
            # Simple Pinhole: f, cx, cy. (Assuming fx=fy)
            fx = intr[0, 0]
            fy = intr[1, 1]
            cx = intr[0, 2]
            cy = intr[1, 2]
            f_val = (fx + fy) / 2.0
            
            f.write(f"{colmap_cam_id} SIMPLE_PINHOLE {w} {h} {f_val} {cx} {cy}\n")

    # images.txt
    # IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    # POINTS2D[] as empty
    
    with open(colmap_dir / "images.txt", "w") as f:
        f.write("# Image list with two lines of data per image.\n")
        f.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
        
        # Sort frames to ensure deterministic ID assignment
        frames = sorted(meta["frames"], key=lambda x: x["file_path"])
        
        for i, frame in enumerate(frames):
            img_id = i + 1
            fpath = frame["file_path"] # images/FRONT/000000.png
            # Colmap expects path relative to image folder? usually relative to dataset root if specified properly.
            # But standard colmap is just filename if images are in root.
            # Nerfstudio often expects relative to "images" folder or data root.
            # Let's use the file_path as is (images/FRONT/000000.png) but remove "images/" if the loader expects it relative to images dir.
            # sgn_dataparser reads `images.txt` and joins with `config.data / config.images_path`.
            # If `images.txt` has `images/FRONT/img.png` and `config.images_path` is `images`, it becomes `images/images/FRONT`.
            # So strip "images/" prefix.
            
            name_for_colmap = fpath
            if name_for_colmap.startswith("images/"):
                name_for_colmap = name_for_colmap.replace("images/", "", 1)
            elif name_for_colmap.startswith("images\\"):
                name_for_colmap = name_for_colmap.replace("images\\", "", 1)
                
            # Transform: We have C2W in transform.json. Colmap needs W2C.
            c2w = np.array(frame["transform_matrix"])
            w2c = np.linalg.inv(c2w)
            
            R_mat = w2c[:3, :3]
            t_vec = w2c[:3, 3]
            
            qvec = rotmat2qvec(R_mat)
            
            # Find camera ID
            # Extract cam name from path
            # images/FRONT/xxx -> FRONT
            cam_name = Path(fpath).parent.name
            cam_id = cam_id_map.get(cam_name, 1)
            
            f.write(f"{img_id} {qvec[0]} {qvec[1]} {qvec[2]} {qvec[3]} {t_vec[0]} {t_vec[1]} {t_vec[2]} {cam_id} {name_for_colmap}\n")
            f.write("\n") # Empty points

    # 3. Points3D from pointcloud.npz
    # The pointcloud.npz contains per-frame point clouds in ego-vehicle (local) coordinates.
    # We must transform them to world coordinates using the ego poses so they match the
    # camera poses in images.txt (which are in world coordinates).
    print("Generating points3D.txt...")
    if (source / "pointcloud.npz").exists():
        try:
            pc_data = np.load(source / "pointcloud.npz", allow_pickle=True)
            if 'pointcloud' in pc_data:
                points_raw = pc_data['pointcloud']
                
                # Handle 0-d array wrapping (dict stored as numpy object)
                if points_raw.ndim == 0:
                    points_raw = points_raw.item()

                # Load ego poses for transforming points from ego-local to world coords
                ego_pose_dir = source / "ego_pose"
                
                all_world_points = []
                
                if isinstance(points_raw, dict):
                    print(f"Pointcloud is per-frame dict with {len(points_raw)} frames")
                    # Sample a subset of frames to avoid excessive points
                    frame_keys = sorted(points_raw.keys())
                    # Use every Nth frame to get good coverage without too many points
                    step = max(1, len(frame_keys) // 10)
                    selected_keys = frame_keys[::step]
                    
                    for frame_key in selected_keys:
                        pts_local = points_raw[frame_key]
                        if not isinstance(pts_local, np.ndarray) or pts_local.ndim != 2:
                            continue
                        if pts_local.shape[1] < 3:
                            continue
                        
                        # Load ego pose for this frame
                        ego_file = ego_pose_dir / f"{int(frame_key):06d}.txt"
                        if not ego_file.exists():
                            print(f"Warning: ego pose {ego_file} not found, skipping frame {frame_key}")
                            continue
                        
                        ego_pose = np.loadtxt(str(ego_file))  # 4x4, ego-local to world
                        
                        # Transform points: world_pts = ego_pose @ [pts_local; 1]
                        N = pts_local.shape[0]
                        pts_h = np.hstack([pts_local[:, :3], np.ones((N, 1))])
                        pts_world = (ego_pose @ pts_h.T).T[:, :3]
                        
                        # Subsample if too many points per frame
                        max_per_frame = 20000
                        if len(pts_world) > max_per_frame:
                            indices = np.random.choice(len(pts_world), max_per_frame, replace=False)
                            pts_world = pts_world[indices]
                        
                        all_world_points.append(pts_world)
                        print(f"  Frame {frame_key}: {N} pts -> {len(pts_world)} sampled, "
                              f"world range X[{pts_world[:,0].min():.1f},{pts_world[:,0].max():.1f}]")
                    
                    if all_world_points:
                        points = np.vstack(all_world_points)
                    else:
                        points = np.zeros((0, 3))
                elif isinstance(points_raw, np.ndarray) and points_raw.ndim == 2:
                    # Single array — try to transform with frame 0 ego pose
                    if points_raw.shape[0] == 3 and points_raw.shape[1] > 3:
                        points_raw = points_raw.T
                    
                    ego_file = ego_pose_dir / "000000.txt"
                    if ego_file.exists():
                        ego_pose = np.loadtxt(str(ego_file))
                        N = points_raw.shape[0]
                        pts_h = np.hstack([points_raw[:, :3], np.ones((N, 1))])
                        points = (ego_pose @ pts_h.T).T[:, :3]
                        print(f"Transformed {N} points using frame 0 ego pose")
                    else:
                        points = points_raw[:, :3]
                        print(f"Warning: no ego pose found, using raw points (may be wrong frame!)")
                else:
                    print(f"Unexpected pointcloud format, using empty.")
                    points = np.zeros((0, 3))
                
                print(f"Total world-frame points: {len(points)}")
                if len(points) > 0:
                    print(f"  Range X: [{points[:,0].min():.2f}, {points[:,0].max():.2f}]")
                    print(f"  Range Y: [{points[:,1].min():.2f}, {points[:,1].max():.2f}]")
                    print(f"  Range Z: [{points[:,2].min():.2f}, {points[:,2].max():.2f}]")
                
                with open(colmap_dir / "points3D.txt", "w") as f:
                    f.write("# 3D point list with one line of data per point.\n")
                    f.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
                    
                    for i in tqdm(range(len(points)), desc="Points3D"):
                        pid = i + 1
                        x, y, z = points[i, 0], points[i, 1], points[i, 2]
                        f.write(f"{pid} {x} {y} {z} 128 128 128 0\n")
        except Exception as e:
            print(f"Error creating points3D.txt: {e}")
            import traceback
            traceback.print_exc()
            with open(colmap_dir / "points3D.txt", "w") as f:
                f.write("# Empty\n")
    else:
        print("pointcloud.npz missing. Creating empty points3D.txt")
        with open(colmap_dir / "points3D.txt", "w") as f:
             f.write("# Empty\n")

    print("Done! Data is ready for training.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sgn_data", required=True, help="Path to your converted sgn data folder (containing images/, transform.json)")
    parser.add_argument("--source_data", required=True, help="Path to unzipped official data (containing sky_mask/, pointcloud.npz)")
    args = parser.parse_args()
    finish_data(args.sgn_data, args.source_data)
