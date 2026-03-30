"""Debug script to check what the model produces for a single frame."""
import multiprocessing
import torch
import sys
import math
sys.path.insert(0, ".")

from nerfstudio.utils.eval_utils import eval_setup
from pathlib import Path
from gsplat.project_gaussians import project_gaussians
from street_gaussians_ns.sgn_splatfacto import get_projection_matrix

if __name__ == "__main__":
    multiprocessing.freeze_support()

    config_path = Path("outputs/output_002/street-gaussians-ns/2026-01-23_223955/config.yml")

config, pipeline, _, _ = eval_setup(
    config_path,
    eval_num_rays_per_chunk=None,
    test_mode="inference",
)

model = pipeline.model
print(f"Model type: {type(model)}")
print(f"Device: {model.device}")

# Check background model
if hasattr(model, 'all_models'):
    for name, sub_model in model.all_models.items():
        print(f"  Sub-model '{name}': {sub_model.num_points} points")
        if sub_model.num_points > 0:
            print(f"    means range: {sub_model.means.min().item():.4f} to {sub_model.means.max().item():.4f}")
            print(f"    scales range: {sub_model.scales.min().item():.4f} to {sub_model.scales.max().item():.4f}")
            print(f"    opacities range: {sub_model.opacities.min().item():.4f} to {sub_model.opacities.max().item():.4f}")
            print(f"    sigmoid(opacities) range: {torch.sigmoid(sub_model.opacities).min().item():.4f} to {torch.sigmoid(sub_model.opacities).max().item():.4f}")

# Get a camera from the dataset
from nerfstudio.data.datamanagers.full_images_datamanager import FullImageDatamanagerConfig
dm_config = config.pipeline.datamanager
dm = dm_config.setup(test_mode="test", device=pipeline.device)
dataset = dm.train_dataset

# Get first camera
camera, batch = dataset[0]
if not isinstance(camera, torch.Tensor):
    from nerfstudio.cameras.cameras import Cameras
    camera = dataset.cameras[0:1].to(model.device)

print(f"\nCamera: {camera}")
print(f"Camera times: {camera.times}")
print(f"Camera shape: {camera.shape}")
print(f"Camera height: {camera.height}, width: {camera.width}")

# Now call get_outputs step by step

with torch.no_grad():
    # Replicate what get_outputs does
    optimized_camera_to_world = camera.camera_to_worlds[0, ...]
    R = optimized_camera_to_world[:3, :3]
    T = optimized_camera_to_world[:3, 3:4]
    R_edit = torch.diag(torch.tensor([1, -1, -1], device=model.device, dtype=R.dtype))
    R = R @ R_edit
    R_inv = R.T
    T_inv = -R_inv @ T
    viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
    viewmat[:3, :3] = R_inv
    viewmat[:3, 3:4] = T_inv
    
    cx = camera.cx.item()
    cy = camera.cy.item()
    W, H = int(camera.width.item()), int(camera.height.item())
    fovx = 2 * math.atan(W / (2 * camera.fx.item()))
    fovy = 2 * math.atan(H / (2 * camera.fy.item()))
    projmat = get_projection_matrix(0.01, 100.0, fovx, fovy, device=model.device)
    
    block_width = 16
    tile_bounds = (
        int((W + block_width - 1) // block_width),
        int((H + block_width - 1) // block_width),
        1,
    )

    print(f"\nviewmat:\n{viewmat}")
    print(f"projmat:\n{projmat}")
    print(f"viewmat[:3,:] shape: {viewmat.squeeze()[:3, :].shape}")
    print(f"(projmat @ viewmat) shape: {(projmat @ viewmat).shape}")
    print(f"H={H}, W={W}, fx={camera.fx.item()}, fy={camera.fy.item()}, cx={cx}, cy={cy}")
    print(f"tile_bounds: {tile_bounds}")
    
    # Use background model means directly
    bg_model = model.all_models["background"]
    means = bg_model.means
    scales = torch.exp(bg_model.scales)
    quats = bg_model.quats / bg_model.quats.norm(dim=-1, keepdim=True)
    
    print(f"\nNum gaussians: {means.shape[0]}")
    print(f"means device: {means.device}, dtype: {means.dtype}")
    print(f"viewmat device: {viewmat.device}, dtype: {viewmat.dtype}")
    
    xys, depths, radii, conics, num_tiles_hit, cov3d = project_gaussians(
        means, scales, 1, quats,
        viewmat.squeeze()[:3, :],
        projmat @ viewmat,
        camera.fx.item(), camera.fy.item(),
        cx, cy, H, W, tile_bounds,
    )
    
    print(f"\n=== project_gaussians results ===")
    print(f"xys shape: {xys.shape}, range: [{xys.min().item():.2f}, {xys.max().item():.2f}]")
    print(f"depths shape: {depths.shape}, range: [{depths.min().item():.4f}, {depths.max().item():.4f}]")
    print(f"radii shape: {radii.shape}, non-zero: {(radii > 0).sum().item()} / {radii.shape[0]}")
    print(f"num_tiles_hit non-zero: {(num_tiles_hit > 0).sum().item()}")
    print(f"conics shape: {conics.shape}")
    
    if (radii > 0).sum() == 0:
        print("\n*** ALL RADII ARE ZERO - NO GAUSSIANS PROJECT ***")
        print("This is why the output is black!")
        
        # Debug: check if means are in view
        means_cam = (viewmat[:3, :3] @ means.T + viewmat[:3, 3:4]).T
        print(f"Means in camera space - z range: [{means_cam[:, 2].min().item():.4f}, {means_cam[:, 2].max().item():.4f}]")
        in_front = (means_cam[:, 2] > 0.01).sum().item()
        print(f"Points in front of camera (z > 0.01): {in_front} / {means.shape[0]}")
    else:
        print(f"\nGaussians DO project. Visible: {(radii > 0).sum().item()}")
