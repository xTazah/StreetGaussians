@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
set "LIB=%LIB%;c:\Users\koehl\anaconda3\libs"
"D:\Git\StreetGaussians\.venv\Scripts\sgn-train.exe" street-gaussians-ns --data .\waymo-dataset\sgn-data\002 --experiment-name output_002_v2
