"""Diagnostic script to check training results.
Run on the desktop: python check_training.py <path_to_config.yml>
e.g.: python check_training.py outputs/output_002_v2/street-gaussians-ns/2026-04-24_XXXXXX/config.yml
"""
import sys
import os
import torch
import yaml
from pathlib import Path

if len(sys.argv) < 2:
    # try to auto-find the config
    output_dir = Path("outputs/output_seg10243_v1/street-gaussians-ns")
    if output_dir.exists():
        subdirs = sorted([d for d in output_dir.iterdir() if d.is_dir()])
        if subdirs:
            config_path = subdirs[-1] / "config.yml"
            print(f"Auto-found config: {config_path}")
        else:
            print("Usage: python check_training.py <path_to_config.yml>")
            sys.exit(1)
    else:
        print("Usage: python check_training.py <path_to_config.yml>")
        sys.exit(1)
else:
    config_path = Path(sys.argv[1])

# Find the checkpoint
ckpt_dir = config_path.parent / "nerfstudio_models"
if ckpt_dir.exists():
    ckpts = sorted(ckpt_dir.glob("step-*.ckpt"))
    if ckpts:
        latest_ckpt = ckpts[-1]
        print(f"Latest checkpoint: {latest_ckpt}")
        ckpt = torch.load(str(latest_ckpt), map_location="cpu")
        
        # Check pipeline state dict for model info
        state = ckpt.get("pipeline", ckpt)
        
        # Find all model keys
        model_keys = set()
        for key in state.keys():
            # Keys look like: _model.all_models.background.means
            # or: _model.all_models.object_XXXXX.means
            parts = key.split(".")
            for i, p in enumerate(parts):
                if p == "all_models" and i+1 < len(parts):
                    model_keys.add(parts[i+1])
        
        print(f"\nSubmodels found in checkpoint: {sorted(model_keys)}")
        print(f"Total keys in state dict: {len(state)}")
        
        # For each submodel, check the means shape (= number of gaussians)
        for model_name in sorted(model_keys):
            means_key = None
            for key in state.keys():
                if f"all_models.{model_name}" in key and "means" in key:
                    means_key = key
                    break
            if means_key:
                means = state[means_key]
                print(f"\n{model_name}: ({means_key})")
                print(f"  Gaussians: {means.shape[0]}")
                print(f"  Means range: [{means.min(dim=0).values.numpy()} .. {means.max(dim=0).values.numpy()}]")
                print(f"  Means center: {means.mean(dim=0).numpy()}")
                # Also check scales and opacities
                prefix = means_key.rsplit("means", 1)[0]
                scales_key = prefix + "scales"
                opac_key = prefix + "opacities"
                if scales_key in state:
                    scales = state[scales_key]
                    # scales are log-space in splatfacto
                    print(f"  Scales (log): min={scales.min().item():.4f} max={scales.max().item():.4f} mean={scales.mean().item():.4f}")
                    print(f"  Scales (exp): min={scales.exp().min().item():.6f} max={scales.exp().max().item():.6f} mean={scales.exp().mean().item():.6f}")
                if opac_key in state:
                    opac = state[opac_key]
                    # opacities are logit-space (sigmoid to get actual)
                    actual_opac = torch.sigmoid(opac)
                    print(f"  Opacities (sigmoid): min={actual_opac.min().item():.4f} max={actual_opac.max().item():.4f} mean={actual_opac.mean().item():.4f}")
            else:
                print(f"\n{model_name}: NO means tensor found!")
                model_specific = [k for k in state.keys() if f"all_models.{model_name}" in k]
                print(f"  Keys: {model_specific[:5]}")
    else:
        print("No checkpoints found!")
else:
    print(f"Checkpoint dir not found: {ckpt_dir}")

# Also check the annotation loading
print("\n" + "="*60)
print("Checking annotation.json and PLY files...")
data_dir = Path("waymo-dataset/sgn-data/002")
anno_path = data_dir / "annotation.json"
ply_dir = data_dir / "aggregate_lidar" / "dynamic_objects"

if anno_path.exists():
    import json
    with open(anno_path) as f:
        anno_data = json.load(f)
    
    frames = anno_data["frames"]
    # Count unique moving objects
    moving_gids = set()
    for frame in frames:
        for obj in frame["objects"]:
            if obj.get("is_moving") and obj["type"] == "car":
                moving_gids.add(obj["gid"])
    
    print(f"Unique moving car objects in annotation: {len(moving_gids)}")
    for gid in sorted(moving_gids):
        ply_path = ply_dir / f"{gid}.ply"
        exists = ply_path.exists()
        size = ply_path.stat().st_size if exists else 0
        print(f"  {gid[:20]}.. PLY exists={exists} size={size}")
else:
    print("annotation.json not found!")