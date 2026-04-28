#!/bin/bash
set -e
SEG=$1
if [ -z "$SEG" ]; then echo "Usage: bash preprocess_one.sh <SEGMENT_ID>"; exit 1; fi

cd /mnt/d/Git/StreetGaussians
source ~/miniconda3/bin/activate waymo-prep

OUT_ROOT=waymo-dataset/sgn-data
CLIP=$OUT_ROOT/validation/$SEG

echo "===================="
echo "[1/6] extract_waymo.py"
echo "===================="
python scripts/pythons/extract_waymo.py \
    --waymo_root waymo-dataset/raw \
    --out_root $OUT_ROOT \
    --split validation \
    --specify_segments $SEG \
    --num_workers 1

echo "===================="
echo "[2/6] dummy segs (all 5 cameras)"
echo "===================="
python3 -c "
import os
from PIL import Image
import numpy as np
CLIP='$CLIP'
for cam in ['FRONT','FRONT_LEFT','FRONT_RIGHT','SIDE_LEFT','SIDE_RIGHT']:
    src=f'{CLIP}/images/{cam}'; dst=f'{CLIP}/segs/{cam}'
    os.makedirs(dst, exist_ok=True)
    for fn in sorted(os.listdir(src)):
        if not fn.lower().endswith(('.jpg','.png','.jpeg')): continue
        out=f'{dst}/{os.path.splitext(fn)[0]}.png'
        if os.path.exists(out): continue
        im=Image.open(f'{src}/{fn}')
        Image.fromarray(np.zeros((im.height,im.width),np.uint8)).save(out)
    print(f'  {cam}: {len(os.listdir(dst))} segs')
"

echo "===================="
echo "[3/6] dynamic-object masks"
echo "===================="
bash scripts/shells/masks_generate.sh $CLIP

echo "===================="
echo "[4/6] per-object LiDAR PLYs"
echo "===================="
bash scripts/shells/object_pts_generate.sh $CLIP

echo "===================="
echo "[5/6] COLMAP (sequential matcher + point_triangulator)"
echo "===================="
bash scripts/shells/run_colmap_fast.sh $CLIP

echo "===================="
echo "[6/6] LiDAR densify + merge to points3D_withlidar.txt"
echo "===================="
mkdir -p $CLIP/colmap/sparse/lidar
python scripts/pythons/pcd2colmap_points3D.py \
    --root_path $CLIP \
    --main_lidar_in_transforms lidar_FRONT
python scripts/pythons/colmap_pts_combine_standalone.py \
    --src1 $CLIP/colmap/sparse/lidar/points3D.txt \
    --src2 $CLIP/colmap/sparse/0/points3D.bin \
    --dst $CLIP/colmap/sparse/0/points3D_withlidar.txt

echo "===================="
echo "DONE — clip ready at: $CLIP"
echo "===================="
ls -la $CLIP/colmap/sparse/0/
wc -l $CLIP/colmap/sparse/0/points3D_withlidar.txt
