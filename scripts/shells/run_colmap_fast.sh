#!/bin/bash
set -e
DATASET_PATH=$1

python scripts/pythons/transform2colmap.py --input_path $DATASET_PATH

mkdir -p $DATASET_PATH/colmap

colmap feature_extractor \
    --database_path $DATASET_PATH/colmap/database.db \
    --image_path $DATASET_PATH/images \
    --ImageReader.mask_path $DATASET_PATH/masks

colmap sequential_matcher \
    --database_path $DATASET_PATH/colmap/database.db \
    --SequentialMatching.overlap 30 \
    --SequentialMatching.quadratic_overlap 1

mkdir -p $DATASET_PATH/colmap/sparse/0

colmap point_triangulator \
    --database_path $DATASET_PATH/colmap/database.db \
    --image_path $DATASET_PATH/images \
    --input_path $DATASET_PATH/colmap/sparse/origin \
    --output_path $DATASET_PATH/colmap/sparse/0 \
    --Mapper.ba_refine_focal_length 0 \
    --Mapper.ba_refine_extra_params 0
