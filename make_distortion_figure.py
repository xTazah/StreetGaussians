"""
Generate a publication-quality figure demonstrating Waymo lens distortion
and the effect of OpenCV undistortion.

Output: 2x3 grid
  Top:    full distorted | full undistorted | per-pixel displacement magnitude
  Bottom: corner crop distorted | corner crop undistorted | corner displacement vectors

Run:
    python make_distortion_figure.py --camera FRONT
    python make_distortion_figure.py --camera SIDE_LEFT --frame 50
"""
import argparse
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


CAMERAS = {
    "FRONT": dict(
        fx=2083.091212, fy=2083.091212, cx=957.293829, cy=650.569793,
        k1=0.040672, k2=-0.337427, p1=0.001627, p2=-0.000788,
    ),
    "FRONT_LEFT": dict(
        fx=2083.731821, fy=2083.731821, cx=970.317609, cy=632.525140,
        k1=0.045580, k2=-0.354034, p1=0.000620, p2=0.001482,
    ),
    "FRONT_RIGHT": dict(
        fx=2082.232760, fy=2082.232760, cx=955.404624, cy=653.698903,
        k1=0.044329, k2=-0.339966, p1=0.001604, p2=-0.001470,
    ),
    "SIDE_LEFT": dict(
        fx=2076.181134, fy=2076.181134, cx=990.028170, cy=241.462811,
        k1=0.049287, k2=-0.343929, p1=0.002343, p2=0.000757,
    ),
    "SIDE_RIGHT": dict(
        fx=2074.885757, fy=2074.885757, cx=1003.997133, cy=238.447095,
        k1=0.042897, k2=-0.331386, p1=0.001246, p2=-0.0000834,
    ),
}


def displacement_field(K, dist, w, h, step=1):
    """For every pixel (u,v) in the distorted image, return the (du,dv) it
    would move to in the undistorted image."""
    us, vs = np.meshgrid(np.arange(0, w, step), np.arange(0, h, step))
    pts = np.stack([us, vs], axis=-1).reshape(-1, 1, 2).astype(np.float32)
    # cv2.undistortPoints maps distorted pixel coords -> normalized coords.
    # Pass P=K to get undistorted pixel coords directly.
    und = cv2.undistortPoints(pts, K, dist, P=K).reshape(-1, 2)
    du = und[:, 0].reshape(us.shape) - us
    dv = und[:, 1].reshape(us.shape) - vs
    return us, vs, du, dv


def make_figure(args):
    src = Path(args.source) / "images" / args.camera
    img_paths = sorted(src.glob("*.png")) + sorted(src.glob("*.jpg"))
    if not img_paths:
        raise FileNotFoundError(f"No images in {src}")
    img_path = img_paths[args.frame]
    img = cv2.cvtColor(cv2.imread(str(img_path)), cv2.COLOR_BGR2RGB)
    h, w = img.shape[:2]

    p = CAMERAS[args.camera]
    K = np.array([[p["fx"], 0, p["cx"]],
                  [0, p["fy"], p["cy"]],
                  [0, 0, 1]])
    dist = np.array([p["k1"], p["k2"], p["p1"], p["p2"], 0.0])

    undist = cv2.undistort(img, K, dist, None, K)

    # Per-pixel displacement at low resolution for the heatmap
    us, vs, du, dv = displacement_field(K, dist, w, h, step=8)
    mag = np.sqrt(du ** 2 + dv ** 2)
    mag_max = float(mag.max())
    mag_rms = float(np.sqrt((mag ** 2).mean()))
    print(f"{args.camera} {img_path.name}: max disp={mag_max:.2f}px, "
          f"RMS={mag_rms:.2f}px")

    # Pick the corner for the zoom panel
    s = args.crop
    if args.corner == "auto":
        iy, ix = np.unravel_index(np.argmax(mag), mag.shape)
        cy_c, cx_c = int(vs[iy, ix]), int(us[iy, ix])
        x0, x1 = max(0, cx_c - s // 2), min(w, cx_c + s // 2)
        y0, y1 = max(0, cy_c - s // 2), min(h, cy_c + s // 2)
    elif args.corner == "tl":
        x0, y0, x1, y1 = 0, 0, s, s
    elif args.corner == "tr":
        x0, y0, x1, y1 = w - s, 0, w, s
    elif args.corner == "bl":
        x0, y0, x1, y1 = 0, h - s, s, h
    elif args.corner == "br":
        x0, y0, x1, y1 = w - s, h - s, w, h
    else:
        raise ValueError(args.corner)
    crop_dist = img[y0:y1, x0:x1]
    crop_und = undist[y0:y1, x0:x1]

    # Per-pixel image difference (the "ghost edge" visualisation).
    # Multiplied to make small shifts visible — same idea as the
    # quick-check script, but kept as a true RGB delta.
    diff = np.abs(img.astype(np.float32) - undist.astype(np.float32))
    diff_vis = np.clip(diff * args.diff_gain, 0, 255).astype(np.uint8)
    crop_diff = diff_vis[y0:y1, x0:x1]

    # Vector-field arrows on the corner (subsample further for readability)
    step_arrow = 32
    crop_us, crop_vs, crop_du, crop_dv = displacement_field(
        K, dist, w, h, step=step_arrow
    )
    in_crop = (
        (crop_us >= x0) & (crop_us < x1) & (crop_vs >= y0) & (crop_vs < y1)
    )

    fig, axes = plt.subplots(2, 4, figsize=(18, 7),
                             gridspec_kw={"width_ratios": [1, 1, 1.15, 1]})

    # --- Top row: full image + heatmap + full diff overlay ---
    axes[0, 0].imshow(img)
    axes[0, 0].set_title("(a) Distorted (raw Waymo)")
    axes[0, 0].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                       fill=False, edgecolor="yellow", lw=1.5))
    axes[0, 0].axis("off")

    axes[0, 1].imshow(undist)
    axes[0, 1].set_title("(b) Undistorted (OpenCV, our fix)")
    axes[0, 1].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                       fill=False, edgecolor="yellow", lw=1.5))
    axes[0, 1].axis("off")

    norm = Normalize(vmin=0, vmax=mag_max)
    im = axes[0, 2].imshow(mag, extent=(0, w, h, 0), cmap="magma", norm=norm)
    axes[0, 2].set_title(
        f"(c) Pixel displacement\n"
        f"max = {mag_max:.1f} px,  RMS = {mag_rms:.2f} px"
    )
    axes[0, 2].set_xlabel("x (px)")
    axes[0, 2].set_ylabel("y (px)")
    cb = fig.colorbar(im, ax=axes[0, 2], shrink=0.85, pad=0.02)
    cb.set_label("Displacement (px)")

    axes[0, 3].imshow(diff_vis)
    axes[0, 3].set_title(f"(d) |raw − undistorted| × {args.diff_gain}")
    axes[0, 3].add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0,
                                       fill=False, edgecolor="yellow", lw=1.5))
    axes[0, 3].axis("off")

    # --- Bottom row: corner zoom + vectors + corner diff ---
    axes[1, 0].imshow(crop_dist)
    axes[1, 0].set_title("(e) Corner crop, distorted")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(crop_und)
    axes[1, 1].set_title("(f) Corner crop, undistorted")
    axes[1, 1].axis("off")

    axes[1, 2].imshow(img, alpha=0.55)
    if in_crop.any():
        axes[1, 2].quiver(
            crop_us[in_crop], crop_vs[in_crop],
            crop_du[in_crop], crop_dv[in_crop],
            angles="xy", scale_units="xy", scale=1.0,
            color="cyan", width=0.0035,
        )
    axes[1, 2].set_xlim(x0, x1)
    axes[1, 2].set_ylim(y1, y0)  # invert so origin top-left
    axes[1, 2].set_title("(g) Displacement vectors (corner)")
    axes[1, 2].set_xlabel("x (px)")
    axes[1, 2].set_ylabel("y (px)")

    axes[1, 3].imshow(crop_diff)
    axes[1, 3].set_title(f"(h) |raw − undistorted| × {args.diff_gain} (corner)")
    axes[1, 3].axis("off")

    fig.suptitle(
        f"Waymo {args.camera} camera "
        f"(k1={p['k1']:.4f}, k2={p['k2']:.4f}, p1={p['p1']:.4f}, p2={p['p2']:.4f})",
        y=1.00,
    )
    fig.tight_layout()

    out = Path(args.out_dir) / f"distortion_figure_{args.camera}.png"
    fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
    print(f"Saved {out}")

    out_pdf = out.with_suffix(".pdf")
    fig.savefig(out_pdf, bbox_inches="tight")
    print(f"Saved {out_pdf}")

    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default=r"c:\Git\Uni\street-gaussians-ns\waymo-dataset\sgn-data\002",
        help="Path to sgn-data clip root (must contain images/<CAMERA>/*.png)",
    )
    parser.add_argument("--camera", default="FRONT", choices=list(CAMERAS.keys()))
    parser.add_argument("--frame", type=int, default=0,
                        help="Index into sorted image list")
    parser.add_argument("--crop", type=int, default=600,
                        help="Corner-zoom crop size in pixels")
    parser.add_argument(
        "--corner", default="auto", choices=["auto", "tl", "tr", "bl", "br"],
        help="Which corner to zoom into. 'auto' picks the max-displacement "
             "corner (geometrically strongest, often sky on FRONT). Use 'bl' "
             "or 'br' on FRONT to get road/cars in the crop."
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--diff_gain", type=float, default=5.0,
        help="Visibility multiplier for the |raw − undistorted| panels. "
             "5–10 makes small shifts visible without saturating; lower "
             "values keep colour fidelity for the paper."
    )
    parser.add_argument("--out_dir", default=".")
    args = parser.parse_args()
    make_figure(args)


if __name__ == "__main__":
    main()
