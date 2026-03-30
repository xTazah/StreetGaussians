"""Quick verification that points3D and cameras are now in the same coordinate frame."""
import sys
import numpy as np
import json
import math

def main():
    print("=" * 60)
    print("VERIFICATION: Points and Cameras Alignment")
    print("=" * 60)

    # 1. Load camera positions from transform.json
    with open("waymo-dataset/sgn-data/002/transform.json") as f:
        meta = json.load(f)
    
    cam_positions = []
    for frame in meta["frames"]:
        c2w = np.array(frame["transform_matrix"])
        cam_positions.append(c2w[:3, 3])
    cam_positions = np.array(cam_positions)
    
    print(f"\nCameras: {len(cam_positions)} poses")
    print(f"  X: [{cam_positions[:,0].min():.1f}, {cam_positions[:,0].max():.1f}]")
    print(f"  Y: [{cam_positions[:,1].min():.1f}, {cam_positions[:,1].max():.1f}]")
    print(f"  Z: [{cam_positions[:,2].min():.1f}, {cam_positions[:,2].max():.1f}]")
    cam_center = cam_positions.mean(axis=0)
    print(f"  Center: ({cam_center[0]:.1f}, {cam_center[1]:.1f}, {cam_center[2]:.1f})")

    # 2. Load 3D points
    pts = []
    with open("waymo-dataset/sgn-data/002/colmap/sparse/0/points3D.txt") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            parts = line.split()
            pts.append([float(parts[1]), float(parts[2]), float(parts[3])])
    pts = np.array(pts)
    
    print(f"\n3D Points: {len(pts)}")
    print(f"  X: [{pts[:,0].min():.1f}, {pts[:,0].max():.1f}]")
    print(f"  Y: [{pts[:,1].min():.1f}, {pts[:,1].max():.1f}]")
    print(f"  Z: [{pts[:,2].min():.1f}, {pts[:,2].max():.1f}]")
    pts_center = pts.mean(axis=0)
    print(f"  Center: ({pts_center[0]:.1f}, {pts_center[1]:.1f}, {pts_center[2]:.1f})")

    # 3. Check alignment
    center_dist = np.linalg.norm(cam_center - pts_center)
    print(f"\nDistance between camera center and point cloud center: {center_dist:.1f}m")
    
    # 4. For each camera, count how many points are "in front" after dataparser transform
    # Simulate the dataparser transform
    with open("outputs/output_002/street-gaussians-ns/2026-01-23_223955/dataparser_transforms.json") as f:
        tx = json.load(f)
    tm = np.array(tx["transform"])  # 3x4
    sc = tx["scale"]
    
    # Transform cameras
    cam_h = np.hstack([cam_positions, np.ones((len(cam_positions), 1))])
    cam_t = (cam_h @ tm.T) * sc
    
    # Transform points  
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    pts_t = (pts_h @ tm.T) * sc
    
    print(f"\nAfter dataparser transform (scale={sc:.6f}):")
    print(f"  Cameras: X[{cam_t[:,0].min():.3f},{cam_t[:,0].max():.3f}] "
          f"Y[{cam_t[:,1].min():.3f},{cam_t[:,1].max():.3f}] "
          f"Z[{cam_t[:,2].min():.3f},{cam_t[:,2].max():.3f}]")
    print(f"  Points:  X[{pts_t[:,0].min():.3f},{pts_t[:,0].max():.3f}] "
          f"Y[{pts_t[:,1].min():.3f},{pts_t[:,1].max():.3f}] "
          f"Z[{pts_t[:,2].min():.3f},{pts_t[:,2].max():.3f}]")
    
    # 5. Quick visibility test: for camera 0, how many points have positive depth?
    cam0_c2w = np.array(meta["frames"][0]["transform_matrix"])
    R = cam0_c2w[:3, :3]
    T = cam0_c2w[:3, 3:4]
    
    # Apply R_edit (flip y,z for gsplat)
    R_edit = np.diag([1, -1, -1])
    R_gs = R @ R_edit
    R_inv = R_gs.T
    T_inv = -R_inv @ T
    viewmat = np.eye(4)
    viewmat[:3, :3] = R_inv
    viewmat[:3, 3:4] = T_inv
    
    # Transform all points to camera space
    pts_h_full = np.hstack([pts, np.ones((len(pts), 1))])
    pts_cam = (viewmat @ pts_h_full.T).T
    z_vals = pts_cam[:, 2]
    
    in_front = (z_vals > 0.01).sum()
    print(f"\nVisibility test (camera 0, world coords):")
    print(f"  Points with Z > 0.01 (in front): {in_front} / {len(pts)} ({100*in_front/len(pts):.1f}%)")
    print(f"  Z range: [{z_vals.min():.2f}, {z_vals.max():.2f}]")
    
    if in_front > 0:
        print(f"\n  ** PASS ** Points are visible to cameras!")
    else:
        print(f"\n  ** FAIL ** No points visible — something is still wrong")
    
    # 6. Check after full transform (dataparser normalized space)
    # Apply transform to both points and camera, then redo viewmat in that space
    # Actually the model works in normalized space, so let's check there too
    cam0_t = cam_t[0]  # transformed camera position
    pts_nearby = np.linalg.norm(pts_t - cam0_t, axis=1)
    close_pts = (pts_nearby < 1.0).sum()
    print(f"\n  Points within 1.0 units of camera 0 (normalized space): {close_pts}")
    close_pts5 = (pts_nearby < 5.0).sum()
    print(f"  Points within 5.0 units of camera 0 (normalized space): {close_pts5}")
    
    print(f"\n{'='*60}")
    if in_front > 1000:
        print("RESULT: Data looks correctly aligned. Ready to retrain!")
    else:
        print("RESULT: Something may still be wrong.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
