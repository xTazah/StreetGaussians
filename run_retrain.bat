@echo off
setlocal

echo [Retrain] Starting training with corrected near/far planes...

:: Use the safe launcher to ensure environment is correct
call ".\run_compiler_safe.bat" ns-train street-gaussians-ns ^
    --data data/sgn-data/002 ^
    --experiment-name output_003_fixed_planes ^
    --pipeline.model.collider-params.near-plane 0.01 ^
    --pipeline.model.collider-params.far-plane 1000.0 ^
    --pipeline.model.background-model.collider-params.near-plane 0.01 ^
    --pipeline.model.background-model.collider-params.far-plane 1000.0 ^
    --pipeline.model.object-model-template.collider-params.near-plane 0.01 ^
    --pipeline.model.object-model-template.collider-params.far-plane 1000.0 ^
    --vis viewer+tensorboard ^
    --viewer.quit-on-train-completion True

if errorlevel 1 (
    echo [Error] Training failed.
    pause
    exit /b 1
)

echo [Retrain] Training finished successfully.
pause
