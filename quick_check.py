"""Quick sanity check: load checkpoint and render one image to see if model learned anything."""
import torch
import numpy as np
from pathlib import Path

# Check the checkpoint file to see if parameters have changed from initialization
ckpt_path = Path("outputs/output_002/street-gaussians-ns/2026-03-31_161704/nerfstudio_models/step-000020000.ckpt")
if not ckpt_path.exists():
    print(f"Checkpoint not found at {ckpt_path}")
    exit(1)

print(f"Loading checkpoint: {ckpt_path}")
ckpt = torch.load(str(ckpt_path), map_location="cpu")

# Look at the pipeline state dict for key model parameters
state = ckpt.get("pipeline", {})

# Find background model parameters
bg_keys = [k for k in state.keys() if "background" in k.lower()]
print(f"\nBackground model keys: {len(bg_keys)}")

# Check opacities - if model learned, these should NOT all be at init value
for key in sorted(state.keys()):
    if "opacities" in key and "background" in key:
        vals = state[key]
        print(f"\n{key}:")
        print(f"  Shape: {vals.shape}")
        print(f"  Min: {vals.min().item():.6f}, Max: {vals.max().item():.6f}, Mean: {vals.mean().item():.6f}, Std: {vals.std().item():.6f}")
        # Init value for opacities is logit(0.1) = -2.197
        init_val = -2.197
        pct_at_init = ((vals - init_val).abs() < 0.01).float().mean().item() * 100
        print(f"  % still at init value (~{init_val:.3f}): {pct_at_init:.1f}%")

    if "features_dc" in key and "background" in key:
        vals = state[key]
        print(f"\n{key}:")
        print(f"  Shape: {vals.shape}")
        print(f"  Min: {vals.min().item():.6f}, Max: {vals.max().item():.6f}, Mean: {vals.mean().item():.6f}, Std: {vals.std().item():.6f}")

    if "means" in key and "background" in key and "gauss_params" in key:
        vals = state[key]
        print(f"\n{key}:")
        print(f"  Shape: {vals.shape}")
        print(f"  Num gaussians: {vals.shape[0]}")

# Check number of gaussians
for key in sorted(state.keys()):
    if "num_points" in key.lower() or "gauss_params.means" in key:
        vals = state[key]
        if vals.dim() >= 1:
            print(f"\n{key}: shape={vals.shape}")

print("\n--- Summary ---")
print("If opacities have varied values (not all ~-2.197) and features_dc has spread (std > 0.01),")
print("the model IS learning. If values are still at initialization, something is wrong.")
