@echo off
:: Uses the robust environment launcher to ensure all linker paths are correct
call run_compiler_safe.bat "C:\Git\Uni\street-gaussians-ns\street-gaussians-ns.venv\Scripts\sgn-train.exe" street-gaussians-ns --data .\waymo-dataset\sgn-data\002 --experiment-name output_002_fixed
