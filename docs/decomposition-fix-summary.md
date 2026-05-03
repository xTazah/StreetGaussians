# Decomposition Fix Summary

## The Problem

After training a model (PSNR ~34, visually good), the rendered output layers showed:

- **`object_rgb`**: Either completely black or solid purple for most of the video.
- **`background_rgb`**: Contained the cars as well — the background model had absorbed everything, including dynamic objects.
- **Composite `rgb`**: Looked nearly identical to ground truth, masking the fact that decomposition had completely failed.

In short: the scene graph was not decomposing. The background model "won the race" and learned all content (cars included), while the object models received no gradient signal and their Gaussians were culled away to nothing.

---

## Root Cause Analysis

The Street Gaussians paper (Yan et al., 2024) uses a multi-loss training objective to enforce foreground/background decomposition:

$$\mathcal{L} = \mathcal{L}_{color} + \lambda_1 \mathcal{L}_{depth} + \lambda_2 \mathcal{L}_{sky} + \lambda_3 \mathcal{L}_{sem} + \lambda_4 \mathcal{L}_{reg}$$

Our codebase was missing most of the supervision signals that make decomposition work. Here is exactly what was wrong:

### 1. Entropy regularisation was 100x too weak and started too late

The paper uses $\lambda_4 = 0.1$ for the accumulated-alpha entropy loss $\mathcal{L}_{reg}$ and activates it after adaptive control stops. Our code had it at **0.001** (100x weaker) and only turned it on after `stop_split_at` (step 15000+). By that point, the background has already learned the cars — the entropy loss on its own cannot reverse that.

### 2. No background suppression signal

There was **nothing** telling the background model to NOT render in car regions. The only training loss was the composite RGB reconstruction, which is satisfied equally well whether the background or the objects render a given pixel. The paper uses a semantic cross-entropy loss ($\mathcal{L}_{sem}$) with per-Gaussian semantic logits to enforce this — our codebase had no equivalent.

### 3. Background seed points included car regions

The paper explicitly masks out dynamic objects during SfM feature extraction so the background point cloud never starts with points ON moving objects. Our code passed **all** SfM points (including those on cars) directly to the background model. This gave the background model a head start on car pixels from step 0.

### 4. SH degree was too high

The paper recommends SH degree 1 for street scenes ("we reduce the SH degree to 1 to prevent overfitting"). Our config used degree 3, giving the background model more capacity to absorb view-dependent car appearances.

### 5. Rendering shape bug caused purple `object_rgb`

When no object Gaussians were visible in a frame, `get_submodel_output` returned a **(H, W, 1)** zero tensor for `rgb`. The render script passes every output through `apply_colormap`, which interprets 1-channel tensors as scalar fields and applies the **turbo** colormap. A value of 0.0 in turbo maps to dark purple — hence the purple `object_rgb` video. When the tensor happened to be exactly zero with no accumulation at all, it showed as black. This is why part of the video was purple and part was black.

---

## Changes Made

All changes are in **`street_gaussians_ns/sgn_splatfacto_scene_graph.py`** and **`street_gaussians_ns/sgn_config.py`**.

### Fix 1: Rendering shape bug
**File:** `sgn_splatfacto_scene_graph.py`, `get_submodel_output()`

Changed the empty-object fallback from returning a **(H,W,1)** tensor to **(H,W,3)**. This ensures `apply_colormap` treats it as RGB (black) rather than applying turbo.

```python
# Before
empty = torch.zeros(H, W, 1, ...)
return {'rgb': empty if sky_capture is None else sky_capture, ...}

# After
empty_3ch = torch.zeros(H, W, 3, ...)
return {'rgb': empty_3ch if sky_capture is None else sky_capture, ...}
```

**Why this fixes the purple:** `apply_colormap` checks `tensor.shape[-1]`. If it's 1, it applies a colormap lookup (turbo). If it's 3, it treats it as already-RGB. Returning 3 channels of zeros gives correct black output.

### Fix 2: Entropy loss weight 0.001 → 0.1
**File:** `sgn_splatfacto_scene_graph.py`, `SplatfactoSceneGraphModelConfig`

```python
object_acc_entropy_loss_mult: float = 0.1  # was 0.001
```

Matches the paper's $\lambda_4 = 0.1$. The entropy loss $-(\alpha \log \alpha + (1-\alpha)\log(1-\alpha))$ encourages binary object accumulation (each pixel is either fully object or fully background), which pushes the model toward clean decomposition.

### Fix 3: Early decomposition activation (step 500 instead of step 15000+)
**File:** `sgn_splatfacto_scene_graph.py`, `SplatfactoSceneGraphModelConfig` + `get_outputs()`

```python
decomp_loss_from_step: int = 500
```

Previously, decomposition losses only activated after `stop_split_at` (which was 15000–25000). By that point, the background has learned everything and object Gaussians have been culled. Now both losses start right after warmup (step 500), so the decomposition signal is present from early training.

### Fix 4: Background suppression loss (new)
**File:** `sgn_splatfacto_scene_graph.py`, `get_outputs()` + `get_loss_dict()`

Added a new loss that directly penalises the background model for rendering inside projected 3D bounding box regions:

$$\mathcal{L}_{bg} = \lambda \cdot \text{mean}(\text{fg\_mask} \cdot \text{bg\_acc})$$

where `fg_mask` is a binary mask obtained by projecting each dynamic object's 3D bounding box corners onto the image plane and filling the bounding rectangle. Weight: 0.05.

This is a simpler substitute for the paper's full semantic loss (which requires per-Gaussian semantic logits). It gives the same core signal: "background, don't render here — these pixels belong to objects."

A new helper function `project_bboxes_to_mask()` handles the 3D→2D projection using the same camera conventions as the main rendering path.

### Fix 5: SfM masking for background seed points (new)
**File:** `sgn_splatfacto_scene_graph.py`, `populate_modules()` + `_filter_seed_points()`

Before creating the background model, we now:
1. Load the object annotations first
2. Filter out any SfM seed points that fall inside any dynamic object's bounding box (across all annotated frames)
3. Pass only the filtered points to the background model

This prevents the background from starting with an unfair advantage on car pixels. The method `_filter_seed_points` transforms each SfM point into each box's local coordinate frame and checks if it's inside the half-extents.

### Fix 6: SH degree 3 → 1
**File:** `sgn_config.py`

```python
sh_degree=1,  # was 3
```

Matches the paper's recommendation. Degree 1 reduces the background model's capacity to absorb complex view-dependent effects (like car paint reflections), making it harder for it to explain car pixels and leaving more room for the object models.

---

## Remaining Gaps (Not Yet Implemented)

These are features from the paper that are still missing but were not the primary cause of the decomposition failure:

| Feature | Paper Detail | Priority |
|---|---|---|
| Semantic loss ($\mathcal{L}_{sem}$) | Per-pixel cross-entropy with per-Gaussian semantic logits ($\lambda_3=0.1$). Needs adding semantic params. | Medium (bg_suppress partially covers this) |
| LiDAR depth loss ($\mathcal{L}_{depth}$) | L1 on rendered vs LiDAR depth ($\lambda_1=0.01$), top-95% filtering | Medium |
| Object bbox pruning | Prune object Gaussians that grow outside their bounding box | Medium |
| LiDAR + SfM combined background init | Paper uses voxel-downsampled LiDAR + SfM; we use SfM only | Low |
| Random init fallback for sparse objects | Objects with <2K LiDAR points get 8K random points inside bbox | Low |

---

## Expected Outcome After Retraining

With these fixes, retraining should produce:
- **`object_rgb`**: Shows cars/pedestrians clearly against a black background
- **`background_rgb`**: Clean street scene without ghostly car silhouettes
- **`rgb` (composite)**: Same or better quality as before (PSNR should be comparable)

The composite quality was never the issue — the problem was purely about decomposition. These changes add the missing supervision to make it work.
