@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
set "LIB=%LIB%;c:\Users\koehl\anaconda3\libs"
cd /d D:\Git\StreetGaussians
"D:\Git\StreetGaussians\.venv\Scripts\sgn-eval.exe" --load-config outputs\output_desert\street-gaussians-ns\2026-05-06_202140\config.yml --output-path renders\DesertCars.json --render-output-path renders\DesertCars