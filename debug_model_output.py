
import torch
from pathlib import Path
import sys
import numpy as np
from nerfstudio.utils.eval_utils import eval_setup

def debug_render():
    # Hardcoded config path based on previous context
    config_path = Path("outputs/output_002_fixed/street-gaussians-ns/2026-02-08_192902/config.yml")
    
    print(f"Loading config from {config_path}...")
    
    try:
        config, pipeline, _, _ = eval_setup(
            config_path,
            test_mode="inference",
        )
    except Exception as e:
        print(f"Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Model loaded successfully.")
    
    # Get the first camera from the dataset
    print("Getting train dataset...")
    dataset = pipeline.datamanager.train_dataset
    print(f"Dataset has {len(dataset)} images.")
    
    camera_idx = 0
    camera = dataset.cameras[camera_idx : camera_idx + 1].to(pipeline.device)
    
    print(f"Rendering camera {camera_idx}...")
    
    # Check for scene graph model structure
    if hasattr(pipeline.model, "background_model"):
        bg_count = 0
        if hasattr(pipeline.model.background_model, "means"):
            bg_count = pipeline.model.background_model.means.shape[0]
            if hasattr(pipeline.model.background_model, "opacities"):
                 opacities = pipeline.model.background_model.opacities
                 print(f"Background Opacities - Min: {opacities.min().item():.4f}, Max: {opacities.max().item():.4f}, Mean: {opacities.mean().item():.4f}")
        elif hasattr(pipeline.model.background_model, "gauss_params"):
            bg_count = pipeline.model.background_model.gauss_params['means'].shape[0]
            opacities = pipeline.model.background_model.gauss_params['opacities']
            print(f"Background Opacities - Min: {opacities.min().item():.4f}, Max: {opacities.max().item():.4f}, Mean: {opacities.mean().item():.4f}")
        print(f"Background Gaussians: {bg_count}")
        
    # Fallback to standard check
    elif hasattr(pipeline.model, "gauss_params"):
         print(f"Number of Gaussians: {pipeline.model.gauss_params['means'].shape[0]}")
    elif hasattr(pipeline.model, "means"):
         print(f"Number of Gaussians: {pipeline.model.means.shape[0]}")
    else:
         print("Could not determine number of gaussians (attribute lookup failed).")

    with torch.no_grad():
        outputs = pipeline.model.get_outputs_for_camera(camera)
        
    print("\n--- Output Statistics ---")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            val_np = value.cpu().numpy()
            print(f"\n[{key}]")
            print(f"  Shape: {val_np.shape}")
            print(f"  Range: [{val_np.min():.4f}, {val_np.max():.4f}]")
            print(f"  Mean:  {val_np.mean():.4f}")
            print(f"  Has NaNs: {np.isnan(val_np).any()}")
            
            # Check for purple-ish color (approx check)
            if "rgb" in key and val_np.shape[-1] == 3:
                # Purple is roughly R=high, G=low, B=high. e.g. (0.5, 0, 0.5)
                # Let's check the center pixel
                H, W = val_np.shape[:2]
                center_pixel = val_np[H//2, W//2]
                print(f"  Center pixel RGB: {center_pixel}")
                
if __name__ == "__main__":
    debug_render()
