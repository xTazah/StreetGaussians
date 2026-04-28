@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
set "LIB=%LIB%;c:\Users\koehl\anaconda3\libs"
"D:\Git\StreetGaussians\.venv\Scripts\ns-viewer.exe" --load-config outputs\output_002_v8\street-gaussians-ns\2026-04-27_113142\config.yml