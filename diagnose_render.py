"""Comprehensive diagnostic for blank render output."""
import sys
import os
import math
import torch
import numpy as np

if __name__ == "__main__":
    # Load the checkpoint directly to inspect gaussian parameters
    ckpt_path = r"outputs\output_002\street-gaussians-ns\2026-01-23_223955\nerfstudio_models\step-000029999.ckpt"
    print(f"Loading checkpoint: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    
    print(f"\nCheckpoint keys: {list(ckpt.keys())}")
    
    if "pipeline" in ckpt:
        state_dict = ckpt["pipeline"]
        print(f"\nPipeline state dict has {len(state_dict)} keys")
        
        # Print all keys with shapes
        print("\n=== All state dict keys with shapes ===")
        for k, v in sorted(state_dict.items()):
            if hasattr(v, 'shape'):
                if v.numel() > 0:
                    print(f"  {k}: shape={v.shape}, dtype={v.dtype}, min={v.min().item():.4f}, max={v.max().item():.4f}")
                else:
                    print(f"  {k}: shape={v.shape}, dtype={v.dtype}, EMPTY")
            else:
                print(f"  {k}: type={type(v)}")
        
        # Check background model gaussian params
        print("\n=== Background Model Gaussians ===")
        bg_prefix = "_model.all_models.background."
        for param_name in ["means", "scales", "opacities", "quats", "features_dc", "features_rest"]:
            key = bg_prefix + param_name
            if key in state_dict:
                v = state_dict[key]
                print(f"  {param_name}: shape={v.shape}, dtype={v.dtype}")
                print(f"    min={v.min().item():.6f}, max={v.max().item():.6f}, mean={v.mean().item():.6f}, std={v.std().item():.6f}")
                if param_name == "opacities":
                    # Check what fraction of gaussians have meaningful opacity
                    activated = torch.sigmoid(v)
                    print(f"    sigmoid(opacities): min={activated.min().item():.4f}, max={activated.max().item():.4f}, mean={activated.mean().item():.4f}")
                    print(f"    fraction with opacity > 0.01: {(activated > 0.01).float().mean().item():.4f}")
                    print(f"    fraction with opacity > 0.5: {(activated > 0.5).float().mean().item():.4f}")
                if param_name == "scales":
                    exp_scales = torch.exp(v)
                    print(f"    exp(scales): min={exp_scales.min().item():.6f}, max={exp_scales.max().item():.6f}, mean={exp_scales.mean().item():.6f}")
            else:
                print(f"  {param_name}: NOT FOUND")
        
        # Check object models
        print("\n=== Object Models ===")
        object_keys = [k for k in state_dict.keys() if "object_" in k]
        object_models = set()
        for k in object_keys:
            parts = k.split(".")
            for i, p in enumerate(parts):
                if p.startswith("object_"):
                    object_models.add(p)
        print(f"  Object models found: {sorted(object_models)}")
        for obj_name in sorted(object_models):
            obj_prefix = f"_model.all_models.{obj_name}."
            means_key = obj_prefix + "means"
            if means_key in state_dict:
                print(f"  {obj_name}: {state_dict[means_key].shape[0]} gaussians")
        
        # Check sky sphere / env map
        print("\n=== Sky / Env Map ===")
        sky_keys = [k for k in state_dict.keys() if "env_map" in k or "sky" in k]
        for k in sky_keys:
            v = state_dict[k]
            if hasattr(v, 'shape'):
                print(f"  {k}: shape={v.shape}, dtype={v.dtype}")

    # Now try to actually load the model and run it
    print("\n\n========================================")
    print("=== Loading full pipeline for test ===")
    print("========================================")
    
    config_path = r"outputs\output_002\street-gaussians-ns\2026-01-23_223955\config.yml"
    
    from pathlib import Path
    from nerfstudio.utils.eval_utils import eval_setup
    
    config_path = Path(config_path)
    config, pipeline, checkpoint_path, step = eval_setup(config_path)
    
    pipeline.eval()
    model = pipeline.model
    device = model.device
    
    print(f"\nModel type: {type(model)}")
    print(f"Device: {device}")
    print(f"Training: {model.training}")
    
    # Get a camera from the dataloader
    datamanager = pipeline.datamanager
    # Get first eval camera
    eval_dataset = datamanager.eval_dataset
    print(f"\nEval dataset length: {len(eval_dataset)}")
    
    # Get camera 0
    cam_data = eval_dataset[0]
    camera = eval_dataset.cameras[0:1].to(device)
    print(f"\nCamera 0:")
    print(f"  camera_to_worlds shape: {camera.camera_to_worlds.shape}, dtype: {camera.camera_to_worlds.dtype}")
    print(f"  camera_to_worlds:\n{camera.camera_to_worlds}")
    print(f"  fx={camera.fx.item()}, fy={camera.fy.item()}")
    print(f"  cx={camera.cx.item()}, cy={camera.cy.item()}")
    print(f"  width={camera.width.item()}, height={camera.height.item()}")
    print(f"  times={camera.times}")
    print(f"  camera_type={camera.camera_type}")
    
    # Manually construct viewmat like the model does
    print("\n=== Manual viewmat construction ===")
    c2w = camera.camera_to_worlds[0]
    print(f"  c2w shape: {c2w.shape}, dtype: {c2w.dtype}")
    print(f"  c2w:\n{c2w}")
    
    R = c2w[:3, :3]
    T = c2w[:3, 3:4]
    print(f"  R:\n{R}")
    print(f"  T:\n{T}")
    
    R_edit = torch.diag(torch.tensor([1, -1, -1], device=device, dtype=R.dtype))
    R = R @ R_edit
    print(f"  R after R_edit:\n{R}")
    
    R_inv = R.T
    T_inv = -R_inv @ T
    viewmat = torch.eye(4, device=device, dtype=R.dtype)
    viewmat[:3, :3] = R_inv
    viewmat[:3, 3:4] = T_inv
    print(f"  viewmat (before float cast), dtype={viewmat.dtype}:")
    print(f"{viewmat}")
    
    viewmat_f32 = viewmat.float()
    print(f"  viewmat (after float cast), dtype={viewmat_f32.dtype}:")
    print(f"{viewmat_f32}")
    
    # Check if float64 vs float32 makes a difference
    print(f"\n  Max abs diff after float cast: {(viewmat.double() - viewmat_f32.double()).abs().max().item():.2e}")
    
    # Check projection matrix  
    W, H = int(camera.width.item()), int(camera.height.item())
    fovx = 2 * math.atan(W / (2 * camera.fx.item()))
    fovy = 2 * math.atan(H / (2 * camera.fy.item()))
    
    from street_gaussians_ns.sgn_splatfacto import get_projection_matrix
    projmat = get_projection_matrix(0.01, 100.0, fovx, fovy, device=device)
    print(f"\n=== Projection Matrix ===")
    print(f"  fovx={fovx:.4f}, fovy={fovy:.4f}")
    print(f"  projmat dtype={projmat.dtype}:")
    print(f"{projmat}")
    
    fullmat = projmat @ viewmat_f32
    print(f"\n  projmat @ viewmat dtype={fullmat.dtype}:")
    print(f"{fullmat}")
    
    # Now test project_gaussians directly
    print("\n=== Testing project_gaussians ===")
    
    # Get the actual means from the model
    bg_model = model.all_models["background"]
    print(f"  Background means: shape={bg_model.means.shape}, dtype={bg_model.means.dtype}")
    print(f"  Background means range: [{bg_model.means.min().item():.4f}, {bg_model.means.max().item():.4f}]")
    print(f"  Background means per-dim min: {bg_model.means.min(dim=0).values.tolist()}")
    print(f"  Background means per-dim max: {bg_model.means.max(dim=0).values.tolist()}")
    
    # Transform a few points manually to check
    means_sample = bg_model.means[:5].detach()
    print(f"\n  Sample 5 gaussian means:\n{means_sample}")
    
    # Transform to camera space  
    viewmat_3x4 = viewmat_f32[:3, :]
    ones = torch.ones(means_sample.shape[0], 1, device=device)
    means_h = torch.cat([means_sample, ones], dim=1)  # N x 4
    means_cam = (viewmat_3x4 @ means_h.T).T  # N x 3
    print(f"  Sample means in camera space:\n{means_cam}")
    print(f"  Z values (depth): {means_cam[:, 2].tolist()}")
    
    # Transform ALL means 
    all_means = bg_model.means.detach()
    ones_all = torch.ones(all_means.shape[0], 1, device=device)
    means_h_all = torch.cat([all_means, ones_all], dim=1)
    means_cam_all = (viewmat_3x4 @ means_h_all.T).T
    z_vals = means_cam_all[:, 2]
    print(f"\n  All means in camera space:")
    print(f"    Z (depth) range: [{z_vals.min().item():.4f}, {z_vals.max().item():.4f}]")
    print(f"    Z > 0.01 (in front of camera): {(z_vals > 0.01).sum().item()} / {len(z_vals)}")
    print(f"    Z > 0: {(z_vals > 0).sum().item()} / {len(z_vals)}")
    print(f"    Z <= 0 (behind camera): {(z_vals <= 0).sum().item()} / {len(z_vals)}")
    
    # Now call the actual project_gaussians
    from gsplat.project_gaussians import project_gaussians
    
    scales_crop = torch.exp(bg_model.scales.detach())
    quats_crop = bg_model.quats.detach()
    quats_crop = quats_crop / quats_crop.norm(dim=-1, keepdim=True)
    
    block_width = 16
    tile_bounds = (
        int((W + block_width - 1) // block_width),
        int((H + block_width - 1) // block_width),
        1,
    )
    
    print(f"\n  Calling project_gaussians with:")
    print(f"    means: {all_means.shape}, dtype={all_means.dtype}")
    print(f"    scales: {scales_crop.shape}, dtype={scales_crop.dtype}")
    print(f"    quats: {quats_crop.shape}, dtype={quats_crop.dtype}")
    print(f"    viewmat: {viewmat_f32.squeeze()[:3,:].shape}, dtype={viewmat_f32.dtype}")
    print(f"    fullmat: {fullmat.shape}, dtype={fullmat.dtype}")
    print(f"    fx={camera.fx.item()}, fy={camera.fy.item()}")
    print(f"    cx={camera.cx.item()}, cy={camera.cy.item()}")
    print(f"    H={H}, W={W}")
    print(f"    tile_bounds={tile_bounds}")
    
    xys, depths, radii, conics, num_tiles_hit, cov3d = project_gaussians(
        all_means,
        scales_crop,
        1,
        quats_crop,
        viewmat_f32.squeeze()[:3, :],
        fullmat,
        camera.fx.item(),
        camera.fy.item(),
        camera.cx.item(),
        camera.cy.item(),
        H,
        W,
        tile_bounds,
    )
    
    print(f"\n  Results:")
    print(f"    radii: shape={radii.shape}, sum={radii.sum().item()}, nonzero={(radii > 0).sum().item()}")
    print(f"    depths: range=[{depths.min().item():.4f}, {depths.max().item():.4f}]")
    if (radii > 0).sum() > 0:
        print(f"    xys (visible): range=[{xys[radii>0].min().item():.2f}, {xys[radii>0].max().item():.2f}]")
        print(f"    SUCCESS: {(radii > 0).sum().item()} gaussians are visible!")
    else:
        print(f"    FAILURE: ALL radii are 0 - no visible gaussians")
        
        # Additional diagnostics if failure
        print(f"\n  === Additional diagnostics ===")
        # Check if viewmat was the right dtype when it went to CUDA
        print(f"    viewmat sent to project_gaussians dtype: {viewmat_f32.squeeze()[:3,:].dtype}")
        print(f"    means dtype: {all_means.dtype}")
        
        # Try with explicit float32 on everything
        print(f"\n  Trying with all float32...")
        xys2, depths2, radii2, _, _, _ = project_gaussians(
            all_means.float(),
            scales_crop.float(),
            1,
            quats_crop.float(),
            viewmat_f32.squeeze()[:3, :].float(),
            fullmat.float(),
            camera.fx.item(),
            camera.fy.item(),
            camera.cx.item(),
            camera.cy.item(),
            H,
            W,
            tile_bounds,
        )
        print(f"    radii (all float32): nonzero={(radii2 > 0).sum().item()}")
        
        # Try with a simple identity-like viewmat to see if gaussians are just in wrong space
        print(f"\n  Trying with identity viewmat (looking down -Z)...")
        test_viewmat = torch.eye(4, device=device, dtype=torch.float32)
        # Move camera back
        test_viewmat[2, 3] = -50.0  # move camera back
        test_fullmat = projmat.float() @ test_viewmat
        xys3, depths3, radii3, _, _, _ = project_gaussians(
            all_means.float(),
            scales_crop.float(),
            1,
            quats_crop.float(),
            test_viewmat[:3, :].float(),
            test_fullmat.float(),
            camera.fx.item(),
            camera.fy.item(),
            camera.cx.item(),
            camera.cy.item(),
            H,
            W,
            tile_bounds,
        )
        print(f"    radii (identity viewmat, z=-50): nonzero={(radii3 > 0).sum().item()}")
        if (radii3 > 0).sum() > 0:
            print(f"    -> Gaussians ARE valid, the viewmat construction is wrong!")
        
        # Try looking at where the means actually are
        print(f"\n  Mean of all gaussian positions: {all_means.mean(dim=0).tolist()}")
        print(f"  Camera position (from c2w): {c2w[:3, 3].tolist()}")
        print(f"  Distance camera to mean-of-means: {(c2w[:3, 3] - all_means.mean(dim=0)).norm().item():.4f}")
    
    print("\n=== DONE ===")
