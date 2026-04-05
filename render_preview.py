"""Render a single image from the step-20000 checkpoint and save it as PNG."""
import torch
import sys
import os
import multiprocessing

def main():
    sys.path.insert(0, os.getcwd())

    from pathlib import Path
    from nerfstudio.utils.eval_utils import eval_setup
    import numpy as np

    try:
        from PIL import Image
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
        from PIL import Image

    config_path = Path("outputs/output_002/street-gaussians-ns/2026-03-31_161704/config.yml")
    print(f"Loading config: {config_path}")

    _, pipeline, _, _ = eval_setup(config_path, eval_num_rays_per_chunk=None, test_mode="test")
    pipeline.eval()

    datamanager = pipeline.datamanager
    cameras = datamanager.eval_dataset.cameras
    print(f"Number of eval cameras: {len(cameras)}")

    output_dir = Path("renders/preview")
    output_dir.mkdir(parents=True, exist_ok=True)

    indices = [0, len(cameras) // 2, len(cameras) - 1]

    for idx in indices:
        print(f"\nRendering eval camera {idx}...")
        camera = cameras[idx:idx+1].to(pipeline.device)

        with torch.no_grad():
            outputs = pipeline.model.get_outputs_for_camera(camera)

        if "rgb" in outputs:
            rgb = outputs["rgb"].cpu().numpy()
            rgb = (rgb * 255).clip(0, 255).astype(np.uint8)
            if rgb.ndim == 3:
                img = Image.fromarray(rgb)
                out_path = output_dir / f"render_cam{idx}.png"
                img.save(str(out_path))
                print(f"  Saved RGB: {out_path} (shape: {rgb.shape})")

        if "accumulation" in outputs:
            acc = outputs["accumulation"].cpu().numpy()
            acc = (acc * 255).clip(0, 255).astype(np.uint8)
            if acc.ndim == 3:
                acc = acc.squeeze(-1)
            img = Image.fromarray(acc, mode='L')
            out_path = output_dir / f"accumulation_cam{idx}.png"
            img.save(str(out_path))
            print(f"  Saved accumulation: {out_path}")

    print(f"\nDone! Check renders/preview/ for output images.")

if __name__ == '__main__':
    multiprocessing.freeze_support()
    main()
