"""Merge two COLMAP points3D files (any combination of .bin/.txt) → output .txt."""
import argparse, struct
from pathlib import Path

def read_bin(path):
    pts = []
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            pid = struct.unpack('<Q', f.read(8))[0]
            xyz = struct.unpack('<3d', f.read(24))
            rgb = struct.unpack('<3B', f.read(3))
            err = struct.unpack('<d', f.read(8))[0]
            tlen = struct.unpack('<Q', f.read(8))[0]
            tr = list(struct.unpack(f'<{tlen*2}I', f.read(tlen*8))) if tlen else []
            track = list(zip(tr[0::2], tr[1::2]))
            pts.append((pid, xyz, rgb, err, track))
    return pts

def read_txt(path):
    pts = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'): continue
            e = line.split()
            pid = int(e[0])
            xyz = tuple(map(float, e[1:4]))
            rgb = tuple(map(int, e[4:7]))
            err = float(e[7])
            track = [(int(e[i]), int(e[i+1])) for i in range(8, len(e), 2)]
            pts.append((pid, xyz, rgb, err, track))
    return pts

def read_any(path):
    p = Path(path)
    return read_bin(p) if p.suffix == '.bin' else read_txt(p)

def write_txt(pts, path):
    with open(path, 'w') as f:
        f.write('# 3D point list with one line of data per point:\n')
        f.write('#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n')
        f.write(f'# Number of points: {len(pts)}\n')
        for pid, xyz, rgb, err, track in pts:
            ts = ' '.join(f'{a} {b}' for a, b in track)
            f.write(f'{pid} {xyz[0]} {xyz[1]} {xyz[2]} {rgb[0]} {rgb[1]} {rgb[2]} {err} {ts}\n')

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src1', required=True)
    ap.add_argument('--src2', required=True)
    ap.add_argument('--dst', required=True)
    a = ap.parse_args()
    p1, p2 = read_any(a.src1), read_any(a.src2)
    print(f'src1={len(p1)}  src2={len(p2)}')
    offset = max((p[0] for p in p1), default=0) + 1
    p2 = [(p[0] + offset, *p[1:]) for p in p2]
    merged = p1 + p2
    write_txt(merged, a.dst)
    print(f'Wrote {len(merged)} points → {a.dst}')
