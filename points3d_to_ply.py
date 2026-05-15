import sys
import numpy as np

infile, outfile = sys.argv[1], sys.argv[2]
xyz, rgb = [], []
with open(infile) as f:
    for line in f:
        if line.startswith("#") or not line.strip():
            continue
        parts = line.split()
        xyz.append([float(parts[1]), float(parts[2]), float(parts[3])])
        rgb.append([int(parts[4]), int(parts[5]), int(parts[6])])
xyz, rgb = np.array(xyz), np.array(rgb, dtype=np.uint8)

with open(outfile, "w") as f:
    f.write(f"ply\nformat ascii 1.0\nelement vertex {len(xyz)}\n")
    f.write("property float x\nproperty float y\nproperty float z\n")
    f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    for p, c in zip(xyz, rgb):
        f.write(f"{p[0]} {p[1]} {p[2]} {c[0]} {c[1]} {c[2]}\n")
print(f"Wrote {len(xyz)} pts to {outfile}")