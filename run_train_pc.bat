@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
set "LIB=%LIB%;c:\Users\koehl\anaconda3\libs"
::"D:\Git\StreetGaussians\.venv\Scripts\sgn-train.exe" street-gaussians-ns --data .\waymo-dataset\sgn-data\002 --experiment-name output_002_v8
::"D:\Git\StreetGaussians\.venv\Scripts\sgn-train.exe" street-gaussians-ns --data .\waymo-dataset\sgn-data\031 --experiment-name out031_v1
"D:\Git\StreetGaussians\.venv\Scripts\sgn-train.exe" street-gaussians-ns --data .\waymo-dataset\sgn-data\002 --experiment-name output_002_v8 --load-checkpoint outputs/output_002_v8/street-gaussians-ns/2026-04-25_211911/nerfstudio_models/step-000009000.ckpt
