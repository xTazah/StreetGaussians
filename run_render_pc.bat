@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
set "LIB=%LIB%;c:\Users\koehl\anaconda3\libs"
"D:\Git\StreetGaussians\.venv\Scripts\sgn-render.exe" --load-config outputs\output_002_v7\street-gaussians-ns\2026-04-25_123331\config.yml --output-path renders\v7 --split train