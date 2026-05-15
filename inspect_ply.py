"""Quick PLY range inspector. Usage: python inspect_ply.py path/to/file.ply"""
import sys
import open3d as o3d
import numpy as np

p = sys.argv[1]
pcd = o3d.io.read_point_cloud(p)
pts = np.asarray(pcd.points)
print(f"file: {p}")
print(f"N points: {pts.shape[0]}")
print(f"x range: [{pts[:,0].min():.3f}, {pts[:,0].max():.3f}]  span={pts[:,0].max()-pts[:,0].min():.3f}")
print(f"y range: [{pts[:,1].min():.3f}, {pts[:,1].max():.3f}]  span={pts[:,1].max()-pts[:,1].min():.3f}")
print(f"z range: [{pts[:,2].min():.3f}, {pts[:,2].max():.3f}]  span={pts[:,2].max()-pts[:,2].min():.3f}")
print(f"mean: ({pts[:,0].mean():.3f}, {pts[:,1].mean():.3f}, {pts[:,2].mean():.3f})")