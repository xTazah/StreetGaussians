@echo off
set "LIB=C:\Users\FinnK\AppData\Local\Programs\Python\Python38\libs"
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
"C:\Git\Uni\street-gaussians-ns\street-gaussians-ns.venv\Scripts\sgn-train.exe" street-gaussians-ns --data .\waymo-dataset\sgn-data\002 --experiment-name output_002_v2
