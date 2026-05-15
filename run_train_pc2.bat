@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
set "LIB=%LIB%;c:\Users\koehl\anaconda3\libs"

::"D:\Git\StreetGaussians\.venv\Scripts\sgn-train.exe" street-gaussians-ns --data .\waymo-dataset\sgn-data\002 --experiment-name output_002_v8
::"D:\Git\StreetGaussians\.venv\Scripts\sgn-train.exe" street-gaussians-ns --data .\waymo-dataset\sgn-data\031 --experiment-name out031_v1
::"D:\Git\StreetGaussians\.venv\Scripts\sgn-train.exe" street-gaussians-ns --data .\waymo-dataset\sgn-data\002 --experiment-name output_002_v8 --load-checkpoint outputs/output_002_v8/street-gaussians-ns/2026-04-25_211911/nerfstudio_models/step-000009000.ckpt

set CLIP_DIR=.\waymo-dataset\sgn-data\validation\10203656353524179475_7625_000_7645_000
set EXP_NAME=output_desert

set CUDA_LAUNCH_BLOCKING=1

"D:\Git\StreetGaussians\.venv\Scripts\sgn-train.exe" street-gaussians-ns ^
    --experiment-name %EXP_NAME% ^
    colmap-data-parser-config ^
    --data %CLIP_DIR% ^
    --colmap_path colmap/sparse/0 ^
    --load_3D_points True ^
    --max_2D_matches_per_3D_point 0 ^
    --undistort True ^
    --segments-path segs ^
    --masks-path masks ^
    --filter_camera_id 1 ^
    --init_points_filename points3D_withlidar.txt