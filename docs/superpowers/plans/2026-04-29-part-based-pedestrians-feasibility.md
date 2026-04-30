# Part-Based Rigid Pedestrians — Feasibility Report

> **Format note:** This is an analysis/scoping document, not an implementation plan. Per the user's request, it does **not** contain implementation code, test code, or step-by-step TDD tasks. A separate plan document (saved alongside) would follow if and only if the user proceeds.

**Goal of analysis:** Assess whether extending [street_gaussians_ns/sgn_splatfacto_scene_graph.py](street_gaussians_ns/sgn_splatfacto_scene_graph.py) to support pedestrians as ~10 rigid body segments per person is realistic for a ~6-week part-time seminar thesis (~60–90 h budget) on a single Waymo scene with a single consumer GPU.

**Existing related plan:** A different extension — pose-conditioned 4D SH (no part decomposition) — was already drafted at [2026-04-28-pose-conditioned-pedestrians.md](docs/superpowers/plans/2026-04-28-pose-conditioned-pedestrians.md). This document evaluates a *different* proposal (part-based rigid decomposition) that overlaps with that plan only at the data-loading layer. Section 6 discusses how they relate.

**TL;DR:**

- The codebase architecture *can* host per-segment rigid objects without changes to the rasterizer. The scene-graph model is genuinely modular at the `all_models: ModuleDict` level.
- **One assumption in the user's framing is wrong for this fork**: there is no bounding-box pruning rule that removes Gaussians outside a per-object box during training. `extent` is set ([sgn_splatfacto_scene_graph.py:81](street_gaussians_ns/sgn_splatfacto_scene_graph.py#L81)) but never read. Pruning is by opacity, scale, and 2D screen size only. This affects two of the user's stated mechanisms (bbox pruning per segment, "the box constrains the geometry"). It is fixable but is *added* work, not "remains structurally unchanged."
- **The biggest blocker is initialization**, not architecture: the LiDAR preprocess drops anything with <10000 points (`MIN_POINTS_PER_OBJECT`) and `dynamic_annotation.py` drops anything with <100 points after subsampling. A 20 m pedestrian has tens of LiDAR returns total. Distributed across 10 segments, this is ~0–5 points per segment. The fallback (random init in `[-0.5, 0.5]^3 × random_scale`) will dominate.
- **Realistic effort estimate: 80–130 h** (median ~100 h) for milestones M1–M6. The budget is 60–90 h. **It is over budget.** The OmniRe-baseline (R) is *not* realistic in addition; substitute with published numbers or upstream black-box.
- **Recommended de-scope**: do M (part-based with 4–6 segments instead of 10) without the OmniRe baseline R. Compare against B0 (rigid pedestrian) only. Frame the thesis around *characterising the regime where part-based rigid helps vs. hurts* — a negative or mixed result is publishable as a "honest baseline study" and is the most likely outcome.

---

## 1. Codebase Mapping for the Extension

### 1.1 Object model abstraction — **MODERATE refactor, mostly clean**

**How a vehicle is currently represented:**

- A vehicle's data lives in [`InterpolatedAnnotation`](street_gaussians_ns/data/utils/dynamic_annotation.py:212) as one entry per `trackId`:
  - `objects_meta[trackId]: Box` — a single canonical box (taken from the first frame the object appears in, [dynamic_annotation.py:337-344](street_gaussians_ns/data/utils/dynamic_annotation.py#L337))
  - `objects_frames[trackId]: List[int]` — frames in which it is visible
  - `seed_pts[trackId]: (points3D, points3D_rgb)` — LiDAR points in object-local frame, loaded from `aggregate_lidar/dynamic_objects/{trackId}.ply`
  - `annos[timestamp]: List[Box]` — per-frame box list (frames hold per-frame center + rot)
- Each vehicle becomes one entry in [`SplatfactoSceneGraphModel.all_models`](street_gaussians_ns/sgn_splatfacto_scene_graph.py:66) keyed `f"object_{trackId}"`. Each entry is a full `SplatfactoModel` instance with its own Gaussian params, optimizer state, and Fourier features.

**Can we instantiate a "parent" object made of segments?**

Not natively — the abstraction is strictly one-cloud-per-trackId, flat namespace. There is no parent/child concept anywhere. Your two clean options are:

- **(A) Segment-as-pseudo-object (recommended).** Each segment gets its own synthetic `trackId` (e.g. `f"{ped_gid}__seg{i}"`) and becomes a full entry in `all_models`. The "parent identity" is implicit: the per-frame `Box.center` and `Box.rot` are derived from forward kinematics on a shared SMPL pose. No code in `SplatfactoSceneGraphModel` changes. **Cost:** moderate work in the dataparser/annotation layer; trivial in the model layer.
- **(B) New parent class with a list-of-segments inside.** Cleaner conceptually but requires modifying the `get_outputs` loop ([sgn_splatfacto_scene_graph.py:320-353](street_gaussians_ns/sgn_splatfacto_scene_graph.py#L320)) and several other places that iterate `all_models` directly. **Cost:** invasive, touches 4–5 files.

**Recommendation: (A).** It avoids any change to the model class.

### 1.2 Pose loading and application — **TRIVIAL substitution point**

- Tracker poses load in [`InterpolatedAnnotation.load_anno_json_one_frame`](street_gaussians_ns/data/utils/dynamic_annotation.py:305) from `annotation.json` (per-frame `translation` + `rotation` quaternion). Each `Box` carries `center: (3,)` and `rot: (3,3)`.
- The rigid transform is applied in [`object2world_gs`](street_gaussians_ns/sgn_splatfacto_scene_graph.py:406):
  ```
  means_w = means @ rot_o2w.T + center
  quat_w = quaternion_multiply(quat_o2w, quats)
  ```
  This is the *exact* substitution point. If you can produce a per-segment-per-frame `(center, rot)` from FK, no rasterizer change is needed.
- The render-time loop ([sgn_splatfacto_scene_graph.py:331-352](street_gaussians_ns/sgn_splatfacto_scene_graph.py#L331)) iterates `annos_t = self.object_annos[camera.times.item()]`. It treats each entry independently. If your data layer produces per-segment `Box` objects in `annos_t`, the render loop is unchanged.

**Verdict:** the substitution path is genuinely clean. **TRIVIAL** at the model layer; the work is in the data layer.

### 1.3 Pose residual optimization — **TRIVIAL once segments are pseudo-objects**

- All pose residuals live in a single tensor in [`BBoxOptimizer`](street_gaussians_ns/data/utils/bbox_optimizers.py:54):
  - `pose_adjustment: Parameter[(num_frames, num_bboxes, 6)]` (mode SO3xR3 or SE3)
- Indexed by `(frame_idx, bbox_idx)` where `bbox_idx = self.bbox_list.index(bbox.trackId)` ([bbox_optimizers.py:146](street_gaussians_ns/data/utils/bbox_optimizers.py#L146)).
- `bbox_list` is built once at scene-graph init from `objects_meta.keys()` ([sgn_splatfacto_scene_graph.py:95](street_gaussians_ns/sgn_splatfacto_scene_graph.py#L95)).
- Applied as `apply_to_bbox(box)` ([bbox_optimizers.py:140](street_gaussians_ns/data/utils/bbox_optimizers.py#L140)) which mutates `bbox.center` and `bbox.rot` *in place* before render.

**If segments are pseudo-objects** under approach (A), each segment is its own row in `bbox_list`. The optimizer scales linearly: 10 vehicles → 100 segments grows `pose_adjustment` from `(F, 10, 6)` to `(F, 100, 6)`. For a 200-frame scene with 8 pedestrians × 10 segments = 80 segment entries, that is `200 × 80 × 6 = 96k` parameters — tiny.

**Verdict:** TRIVIAL. The mechanism reuses without any code change.

### 1.4 Adaptive density control — **MODERATE**

Honest correction to the user's framing: **there is no bounding-box pruning rule in this codebase.** I confirmed this by grep:

- [`extent`](street_gaussians_ns/sgn_splatfacto_scene_graph.py:81) is set on each object's config from `obj_meta.size / 2`, but is never read elsewhere in the package (`grep extent` returns only the assignment).
- The actual culling rules in [`SplatfactoModel.cull_gaussians`](street_gaussians_ns/sgn_splatfacto.py) (around line 678 onward) are: opacity threshold (`cull_alpha_thresh`), scale threshold (`cull_scale_thresh`), and 2D screen-size threshold (`cull_screen_size`). None of these are spatially keyed to the box.
- `crop_box` ([sgn_splatfacto.py:333](street_gaussians_ns/sgn_splatfacto.py#L333)) is a *render-time* mask (`if self.crop_box is not None and not self.training`, line 835), not a training-time prune.

**Implications for your proposal:**

1. The "tight per-segment box constrains geometry" mechanism does not exist out of the box — Gaussians will happily wander outside the segment box during training. For pedestrian segments this is a real problem: an arm-Gaussian that drifts to where the leg is during one frame will be *visible* after the per-frame rigid FK transform places the segment elsewhere.
2. You will likely need to *add* a bbox-prune rule to `cull_gaussians` keyed on each model's local extent. This is small (~20 LOC) but it's net-new work, not free reuse.
3. The change is per-`SplatfactoModel`, applied per-object — so per-segment falls out automatically once added.

**Verdict: MODERATE.** ~20–30 LOC + tuning (the cull threshold matters; pedestrian segments are at the scale where slightly-too-aggressive prune kills the model).

### 1.5 Initialization from LiDAR — **INVASIVE in the preprocess pipeline**

Two filters drop pedestrians today:

- **Preprocess filter:** [scripts/pythons/generate_annotations.py](scripts/pythons/generate_annotations.py) requires `MIN_POINTS_PER_OBJECT = 10000` LiDAR points per object before saving its `.ply` file (per the exploration agent's read; verify by direct read). At Waymo's typical pedestrian distance of 10–30 m, *no pedestrian* will pass this. Lowering it to ~50 is necessary.
- **Filter at parser time:** [dynamic_annotation.py:356](street_gaussians_ns/data/utils/dynamic_annotation.py#L356): `if points3D.shape[0] < 100: return None` — this drops the object entirely. For pedestrians you'd want to keep them even with very few points.
- **Filter on `is_moving`:** [dynamic_annotation.py:314-315](street_gaussians_ns/data/utils/dynamic_annotation.py#L314): `if ignore_static and not obj['is_moving']` — Waymo's pedestrian `is_moving` flag is sometimes flaky for slow walkers. May need to relax for pedestrians.
- **Class filter:** [dynamic_annotation.py:19](street_gaussians_ns/data/utils/dynamic_annotation.py#L19): `FILTER_LABEL = ['car']`. Pedestrian class needs to be added.
- **EXP_RATE inflation:** [dynamic_annotation.py:22](street_gaussians_ns/data/utils/dynamic_annotation.py#L22): `[1.3, 1.3, 1.1]` — applied uniformly. Inflating per-segment boxes by 1.3× will overlap segments significantly. May need per-class EXP_RATE.

**To assign LiDAR points to segments by SMPL skeleton proximity:**

- Need: SMPL joint positions in world-space at each frame (from FK on the loaded SMPL parameters).
- Approach: for each LiDAR point inside a pedestrian's box at frame *t*, transform to pedestrian local frame, find the nearest SMPL bone segment by perpendicular distance, assign to that segment.
- Across the sequence, pool points from all frames a pedestrian is visible (LiDAR is sparse per frame; aggregating across 5–10 frames may give ~50–200 points per pedestrian total).
- Per-segment counts will still be small (~5–20 points). Random fallback (`random_init=True`) will still dominate the head, hands, feet — the high-curvature, low-area body parts that least tolerate random init.

**Verdict: INVASIVE.** This is the hardest milestone. ~15–25 h. Touching: preprocess script (regenerate everything), `dynamic_annotation.py` filters, new utility for skeleton-based point assignment.

### 1.6 Rendering pipeline — **soft scaling concerns, no hard limit**

- All Gaussians (background + all visible objects) are concatenated into one batch ([sgn_splatfacto_scene_graph.py:355-357](street_gaussians_ns/sgn_splatfacto_scene_graph.py#L355)) and rasterized once. The rasterizer doesn't care how many "objects" — it sees one big buffer.
- **But** the per-frame Python loop ([sgn_splatfacto_scene_graph.py:331-352](street_gaussians_ns/sgn_splatfacto_scene_graph.py#L331)) iterates over *every visible object*: per-iteration it does the world-frame rigid transform, builds Fourier features, and appends to lists. With 10 segments per pedestrian × 5 visible pedestrians × 200 frames per epoch × however many gradient steps per frame, this loop is non-trivial. It is not vectorized.
- **Densification is per-`SplatfactoModel`** — each sub-model has its own `xys_grad_norm`, `vis_counts`, `max_2Dsize` buffers and its own optimizer state slot (per the exploration agent's read of [sgn_splatfacto.py](street_gaussians_ns/sgn_splatfacto.py)). With 80 sub-model entries per scene the bookkeeping cost goes up.

**Hard limits:** none I can see. **Soft costs:** Python overhead in the per-frame loop, optimizer state memory.

**Verdict:** no architectural blocker. Expect ~1.5–3× slower training step than the vehicle baseline (see §3.7).

---

## 2. Data Pipeline for SMPL Integration

### 2.1 OmniRe SMPL output format

Confirmed via web research on the upstream [drivestudio repo](https://github.com/ziyc/drivestudio):

- **On-disk:** `<scene>/humanpose/smpl.pkl` — single joblib-pickled dict per scene, keyed by *Waymo GT track-id* (matched in postprocessing). Per-instance fields:
  - `global_orient`: `(num_frames, 1, 3, 3)` — rotation matrices, root pose
  - `body_pose`: `(num_frames, 23, 3, 3)` — rotation matrices, body joints
  - `betas`: `(num_frames, 10)` — shape (per-frame, not per-pedestrian; OmniRe treats it as time-varying)
  - `valid_mask`: `(num_frames,)` — bool
  - `selected_cam_idx`: `(num_frames,)` — which camera was used
- **Translation is recovered separately**: the loader `WaymoPixelSource.load_objects` derives `smpl_trans: (num_frames, 3)` from the camera-to-world projection of the chosen camera's pelvis estimate, not from a flat `transl` in the pkl. **This is a data-conversion gotcha**: if you re-use OmniRe's pkl directly you must port their translation-recovery code, not just load the pkl.
- **Joint count: 24** (1 root + 23 body), stored as 3×3 rotation matrices, **not axis-angle**. Your spec said "axis-angle per joint." Conversion is one `matrix_to_axis_angle` call but the format differs from your stated assumption.

### 2.2 Where SMPL would inject in this codebase

- The current data flow: `sgn_dataparser.py` constructs `InterpolatedAnnotation(...)` and passes it via `metadata["object_annos"]` ([dataparser exploration]). The model reads it in `populate_modules` ([sgn_splatfacto_scene_graph.py:76](street_gaussians_ns/sgn_splatfacto_scene_graph.py#L76)).
- **Cleanest injection point:** extend `InterpolatedAnnotation` (or wrap it) with a parallel SMPL store: per-pedestrian-trackId → `{frame: smpl_pose}`. Then at `load_anno_json_one_frame`, when a pedestrian box is created, expand it into 10 per-segment `Box` objects whose `center`/`rot` are computed by FK from the SMPL params at that frame.
- All segment-`Box`es get unique synthetic `trackId`s of the form `f"{ped_gid}__seg{seg_index:02d}"`. This propagates naturally through `objects_meta`, `seed_pts`, `objects_frames`, and the `bbox_list` for the optimizer. **No code in the model class needs to change.**

### 2.3 Adding a `pedestrian_part` class

There is no truly hard-coded class assumption in the model layer — `obj['type']` is read in [dynamic_annotation.py:312](street_gaussians_ns/data/utils/dynamic_annotation.py#L312) as a string filter and that's it. The class filter (`FILTER_LABEL = ['car']`) is the main lock. Other places to check:

- [generate_annotations.py:39-45 `CLASS_MAP`](scripts/pythons/generate_annotations.py) — needs to map Waymo's `TYPE_PEDESTRIAN` to your label.
- `EXP_RATE` (uniform): may want per-class.
- `MIN_POINTS_PER_OBJECT`: needs to be drastically lowered or class-conditional.
- Color/visualization in [dynamic_annotation.py:24 `COLORMAPS`](street_gaussians_ns/data/utils/dynamic_annotation.py#L24) — cosmetic only.

The label `pedestrian_part` does not need to be a "class" the codebase knows about — segments can carry a synthetic label like `pedestrian_part` purely for your own bookkeeping. The model treats every `Box` the same.

**Verdict:** clean. No surprise breakages.

### 2.4 SMPL utilities in this codebase

- **No existing SMPL utilities** in `street_gaussians_ns/`. There is `pytorch3d` (used in `sgn_splatfacto_scene_graph.py:9` for `quaternion_multiply`) which gives you `matrix_to_axis_angle`, etc. — but not skeletal FK.
- **Reference for FK:** lift from `drivestudio/models/human_body.py::batch_rigid_transform` (~50 LOC, walks the SMPL kinematic tree) plus the SMPL parent-child table (`SMPL_PARENTS`). This is a clean port. You do **not** need `smplx` — you do not need vertex skinning, blend weights, or shape blendshapes for body-only rigid FK on bone frames.
- **You need the SMPL_NEUTRAL.pkl** (or just the rest-pose joint positions array + parents) — the parents table is small enough to hardcode (24 ints).

**Verdict:** lift ~80–120 LOC of FK code from drivestudio + a hardcoded parent table + rest-pose joint positions. ~6–10 h.

---

## 3. Specific Implementation Risks

### 3.1 Per-pedestrian Gaussian count / per-object overhead — **REAL but probably tolerable**

- Per-`SplatfactoModel` state: `xys_grad_norm`, `vis_counts`, `max_2Dsize`, `quats`, `means`, `scales`, `opacities`, `features_dc`, `features_rest` (each as their own tensor). A "small" pedestrian segment with say 200 Gaussians has ~10 KB of state. With 80 segments → ~800 KB. Negligible.
- Optimizer state (Adam: 2× param size) for the same: ~1.6 MB extra. Negligible.
- The Python loop over `annos_t` in [`get_outputs`](street_gaussians_ns/sgn_splatfacto_scene_graph.py:320-352) is the actual cost. Each iteration: 4–5 small tensor ops (`object2world_gs`, Fourier features, list appends). With ~80 iterations vs. ~10 today, a single rendering pass lengthens by maybe 5–15 ms (guess; profile to verify).
- If a scene has 5 pedestrians visible per frame × 10 segments = 50 extra iterations vs. ~10 today → 6× more iterations. 200 ms today → 1.2 s wall-clock per render. This is the dominant cost concern.

**Mitigation:** if benchmarking shows the loop dominates, vectorize the per-frame transform: stack `means_list`, `rot_list`, `t_list` and do one batched matmul. ~20 LOC change in `get_outputs`. Save for last.

### 3.2 Joint discontinuities — **REAL, expect visible seams**

- 3D Gaussian Splatting renders anisotropic ellipsoids. A Gaussian centered near the shoulder joint, with a scale that extends 5–10 cm radially, *can* span the seam — but only if the scale is large enough. With small/well-fit Gaussians (which the codebase's screen-size cull encourages), the seam will be visible as a thin gap when the upper arm and torso rotate apart.
- Anti-aliasing won't save you: the 2D Gaussian splat falls off Gaussian-fast at the boundary; outside its 3σ extent it contributes nothing.
- **Mitigation options (none free):**
  1. Inflate per-segment box such that segments overlap by ~30% near joints, and let two adjacent segments both contribute Gaussians to the joint region. **Cost:** doubles Gaussian count near joints; dual-ownership confuses pose-residual learning.
  2. Add a "joint blob" — one extra small per-joint object whose pose is the joint frame. **Cost:** +12 segments per pedestrian.
  3. Accept it as a thesis-defensible artifact and characterize it.
- **Honest assessment:** seams are likely the single most prominent failure mode and will be the first thing a viewer notices. The thesis can frame this as "the cost of avoiding LBS." This is fine *as a finding*, less fine if you needed pedestrian quality to beat OmniRe.

### 3.3 Pose-residual optimization stability — **MODERATE**

- Vehicle pose residuals are tuned with `bbox_opt` lr=1e-3 → 5e-5 over 70k steps ([sgn_config.py:82-85](street_gaussians_ns/sgn_config.py#L82)).
- Per-segment residuals at the same lr will be *more* aggressive in absolute terms because: (a) segment frames are smaller; a 1 cm correction at the wrist is a meaningful fraction of forearm length, but a noise-level perturbation at a vehicle scale; (b) FK errors compound up the chain — a small root error rotates everything, a small wrist-residual is locally bounded.
- **Likely needs:** lower lr (1e-4 → 1e-5) for distal segments; possibly a per-segment regularizer schedule. The existing `center_l2_penalty` and `rot_l2_penalty` can be raised.
- Alternative: turn pose residuals **off** for pedestrians initially (your M-noref baseline). If it works without them, that's a finding.

**Cost:** ~5–10 h of hyperparameter search; not invasive code-wise.

### 3.4 Initialization quality — **HIGHEST RISK**

- A typical Waymo pedestrian at 15 m has 30–80 LiDAR returns *per frame*. Across 10 visible frames, ~300–800 *if you aggregate* in the canonical SMPL frame.
- Aggregating in the canonical SMPL frame requires inverse-FK to map per-frame world points back to the pelvis-centered rest pose. This is exactly the data your FK code can produce (apply inverse FK to each LiDAR point given the per-frame SMPL pose).
- After aggregation, distributing 300–800 points across 10 segments by skeleton-proximity yields 5–80 points per segment. **Head, hands, feet** will likely have <10 points each. Random fallback init for these segments.
- Random init at 10× the *segment* extent (which is ~30 cm) → `random_scale` should be set very small (e.g. 0.3, not the default 10.0). This is a config issue, not a code issue.
- **Convergence question:** even with bad init, the photometric gradient should pull Gaussians toward correct positions if the segment frame is approximately right. The dangerous failure mode is: Gaussians converge to good positions in *one canonical pose* but the FK transform reveals seams or pose errors that the photometric loss can't fix because pose-residuals are clamped.

**This is the milestone with highest variance in outcome.** Plan for 2× the time you think it will take.

### 3.5 Bounding-box pruning interaction — **NOT A REAL RISK in this fork**

Per §1.4: there is no per-box prune rule. Adding one is on you. Once added, it does interact with adaptive control: if you keep cull thresholds vehicle-tuned and add a tight per-segment box, you'll cull aggressively because (a) the box is 30 cm not 4 m, (b) per-segment Gaussian counts are already low. **Likely fix:** disable the box-prune for the first ~5k steps, then enable.

### 3.6 Pose data quality — **REAL, source-of-truth-dependent**

- OmniRe's postprocessing fills gaps via SLERP within-camera, picks the best camera per frame, and fills short breaks ([drivestudio/datasets/tools/postprocess.py]). For Waymo *training* split scenes this works well in their published examples.
- For *your* scene specifically: depends on how often pedestrians are clearly visible from FRONT (since you're using FRONT only, per the existing 2026-04-28 plan). 4D-Humans + PHALP run on the FRONT camera; OmniRe uses 5 cameras and picks the best. **You will lose poses on frames where the pedestrian faces away from FRONT, is occluded, or is below ~30 px tall.**
- OmniRe also runs PHALP-track ↔ Waymo-gid identity matching, which has its own failure modes (identity swaps mid-sequence). The existing 2026-04-28 plan already devotes Phase 3 to this — so it's a known, scoped problem. Estimate ~15 h to make it robust on one scene.
- **Realistic plan:** drop or freeze any frame where SMPL is `valid_mask=False` and don't render the pedestrian on that frame. This is straightforward; cost is "we don't supervise on those frames" which is fine.

### 3.7 Hardware / training time — **REAL, will likely double or triple training time**

- Existing baseline: 3–4 h on one Waymo scene with vehicles only. Per the existing 2026-04-28 plan, training is on RTX 2070 Super (8 GB), not 3090/4090. **Verify GPU before estimating.**
- Adding ~50 segment sub-models per scene multiplies the per-step Python loop cost by ~6× (§1.6). Steps that were 200 ms become 800 ms–1.2 s. With 70k steps and 1 s/step = ~20 h. **3–4× slower than baseline.**
- 8 GB VRAM is a tight budget if you scale Gaussian count per pedestrian. Each segment with 200 Gaussians is fine; 2000 Gaussians per segment × 80 segments × 100k Gaussians-equivalent → may push memory. Watch for OOM.
- **Estimate:** 8–15 h per training run. Budget 3–5 training runs (B0, M-noref, M, plus debugging) → 40–75 h of GPU wall-clock alone. This consumes the bulk of your 60–90 h budget if you can't run training overnight.

---

## 4. Comparison Baselines

### 4.1 B0: Pedestrian-as-rigid-box

**Verdict:** TRIVIAL in code, NOT TRIVIAL in result quality.

- Code: extend `FILTER_LABEL` to include pedestrian, lower `MIN_POINTS_PER_OBJECT` in preprocess, regenerate annotations and per-object PLYs. ~6–10 h end-to-end including a regenerate-and-train cycle.
- **But the result will be visibly bad** for moving pedestrians: a single rigid box can't represent walking. The Gaussian cloud will smear over the rest-pose silhouette and look like a rigid mannequin sliding. **This is what makes B0 a meaningful baseline** — you need it bad to make M look interesting.
- For *standing* pedestrians (not walking) B0 will look fine, possibly indistinguishable from M. **Make sure your scene has walking pedestrians.** If the chosen Waymo segment has only standing/stationary pedestrians, B0 ≈ M and your thesis has no signal.

### 4.2 R: OmniRe SMPL-node baseline

**Verdict:** UNREALISTIC to reimplement; black-box upstream is plausible if you have the GPU budget.

- Reimplementing the OmniRe SMPL node within `street_gaussians_ns` requires ~2000 LOC (per web research) including LBS skinning, voxel deformer, `SMPLTemplate`, integration with the scene-graph forward pass, and the matching modifications in densification. **Estimate: 80–150 h** alone. Out of scope.
- **Black-box alternative:** install upstream `drivestudio` separately, run their training on the same Waymo scene, evaluate with the same protocol. **Estimate: 10–20 h** including environment setup, scene preprocessing in their format, training (~1× to 2× your other runs depending on their scaling). Drivestudio supports Waymo natively so the preprocessing already exists.
- **Pure-citation alternative:** report OmniRe's published Waymo numbers from their paper alongside a methodological caveat ("not the same scene"). Cheapest, weakest defensibility.

**Recommendation:** drop R from the primary plan. If it gets done, it's a stretch goal. Note this clearly in the thesis.

### 4.3 M-noref: M with pose-residuals disabled

**Verdict:** TRIVIAL once M works.

- Same code path as M; flip `bbox_optimizer.mode = "off"` in config. ~30 min code, plus one training run.
- This is the most informative ablation: it isolates whether the *FK from OmniRe* is doing the work, or whether the per-segment learnable residual is what saves you. Honest hypothesis: residuals will help significantly because the SMPL poses from 4D-Humans + PHALP are noisy (jitter, depth ambiguity).

---

## 5. Concrete Implementation Path

Effort estimates assume 10–15 h/week × 6 weeks ≈ 60–90 h budget. **Honest sum below is 80–130 h** — over budget. Surprise factors flagged with ⚠.

### M0: Verify scene and budget (2–4 h)

- Confirm GPU: 2070 Super 8 GB? 3090/4090? This changes the timeline by 2–3×.
- Confirm scene has walking pedestrians (visualize `annotation.json` pedestrian boxes at start/end frames, look for translation > 1 m). If only standing pedestrians, this thesis won't have signal — pick a different scene.
- Confirm OmniRe `humanpose/smpl.pkl` covers the same scene (or reserve time to run their preprocessing yourself).

### M1: Get B0 running (6–10 h)

- Lower `MIN_POINTS_PER_OBJECT` in `generate_annotations.py` to ~30.
- Add pedestrian to `FILTER_LABEL` and `CLASS_MAP`.
- Regenerate `annotation.json` and `aggregate_lidar/dynamic_objects/*.ply`.
- Train. Compare PSNR against your existing vehicle-only baseline (it should match closely, since pedestrians add little new pixel area).
- ⚠ If pedestrian boxes are smeared visually but PSNR barely changes, your eval metric needs a pedestrian-region mask. See M6.

### M2: SMPL data integration (12–18 h)

- Run OmniRe's `humanpose_process.py` on your scene OR obtain the precomputed `smpl.pkl`. ⚠ This requires a separate Python env (4D-Humans + PHALP) — budget 4 h for env setup alone.
- Write a loader: `smpl.pkl` → per-pedestrian-trackId, per-frame `(global_orient, body_pose, betas, smpl_trans, valid_mask)`.
- Match OmniRe pedestrian IDs to your Waymo `gid`s. ⚠ The existing 2026-04-28 plan dedicates a phase to this; reuse that code if possible.
- Smoke test: project SMPL skeleton joints to image, overlay on pedestrians, visually verify alignment for ~10 frames.

### M3: Forward kinematics on SMPL skeleton (8–12 h)

- Lift `batch_rigid_transform` from `drivestudio/models/human_body.py` (~50 LOC).
- Hardcode `SMPL_PARENTS` table (24 ints).
- Get rest-pose joint positions from `SMPL_NEUTRAL.pkl` (one-time extract and bake into a `.npy` in the repo) so you don't have a runtime dep on the full SMPL model.
- Map SMPL's 24 joints onto your 10 body segments. ⚠ This mapping is non-trivial — pelvis is 1 joint but spans torso, shoulder is 1 joint but the upper-arm segment frame should originate there. Document the mapping table in code.
- Test: compute per-segment world transforms for one frame, render skeleton lines in 3D, eyeball alignment.

### M4: Per-segment Gaussian object class (20–35 h) ⚠⚠ HIGHEST VARIANCE

- In `dynamic_annotation.py`, when a pedestrian box is loaded, expand to 10 per-segment `Box` objects with synthetic `trackId`s.
- Per-segment seed-point assignment: aggregate per-pedestrian LiDAR points across visible frames, inverse-FK to canonical SMPL frame, assign points to segments by perpendicular distance to bones.
- Per-segment local box sizes from rest-pose SMPL geometry (not from Waymo's pedestrian box, which is too coarse).
- Add a per-segment-extent prune to `cull_gaussians` (or a new method called from `refinement_after`). ⚠ Tuning the threshold is the main time-sink.
- Disable the prune for first 5k steps (warmup).
- Run end-to-end training. Expect to iterate.

### M5: Per-segment pose residual optimization (4–8 h)

- No code changes needed: the bbox-optimizer naturally handles 80 entries.
- Tune `bbox_opt` lr — start at 1e-4 (10× lower than vehicle), tune `center_l2_penalty` and `rot_l2_penalty` 5–10× higher.
- Run M-noref (residuals off) and M (residuals on) for ablation.

### M6: Evaluation infrastructure (8–14 h)

- Per-pedestrian-bbox PSNR/LPIPS: project each Waymo pedestrian box to 2D, mask the frame, compute PSNR inside the mask. 4–6 h.
- Per-scene comparison harness: a script that takes 2–3 model checkpoints and produces side-by-side renders + numeric tables. 3–5 h.
- Failure-mode visualization: per-segment color-coded render to see seams. 2–3 h.

### M7: OmniRe baseline R (DROP or 10–20 h black-box)

- Drop from primary plan. If time remains: install drivestudio in a separate worktree, run their Waymo training on your scene, copy out the rendered videos, evaluate with M6 harness.

### Sum

| Milestone | Median (h) | Pessimistic (h) |
|---|---:|---:|
| M0 | 3 | 4 |
| M1 | 8 | 12 |
| M2 | 15 | 22 |
| M3 | 10 | 14 |
| M4 | 28 | 45 |
| M5 | 6 | 10 |
| M6 | 11 | 16 |
| **Subtotal (no R)** | **81** | **123** |
| M7 (black-box only) | 12 | 22 |
| **Total with R** | **93** | **145** |

Budget: 60–90 h. **Median (no R) is right at the upper bound. Pessimistic blows it.**

⚠ **Most likely to surprise you:**

1. **M4 (per-segment object class)** — combination of design decisions (segment grouping, point assignment, local boxes, prune rule) plus the iterative training cycles to validate. 2× factor wouldn't be shocking.
2. **M2 (SMPL data integration)** — if you have to run OmniRe's preprocessing yourself, the env setup + 4D-Humans deps could eat a full day.
3. **GPU wall-clock** — if 2070 Super 8 GB, training time can balloon to 12+ h per run; with 4–5 runs (B0, B0 retry, M-noref, M, M with tuning) that's 50–60 h *of clock time*, much of which can be overnight, but iteration speed is the bottleneck.

---

## 6. Recommendation

### 6.1 Is this realistic for a 6-week seminar thesis?

**Borderline / leaning no for the full proposal.** Median estimate (no R) is 81 h vs. 60–90 h budget; pessimistic is 123 h. The full proposal — 10 segments, FK from SMPL, per-segment residuals, OmniRe R baseline — is over budget.

**There is a much more tractable de-scoped version that delivers the core thesis claim.** See 6.2.

### 6.2 80-of-the-value-at-50-of-the-effort options

Listed strongest to weakest signal-to-cost:

**Option A — Fewer segments (4–6 instead of 10).** Group as: torso+head (1), pelvis+upper-legs (1), lower-legs+feet (2), upper-arms (2), lower-arms+hands (varies). 4 segments captures torso/leg articulation (the main visual win). 6 adds arms. Per-frame loop cost drops, residuals drop, density-control tuning is simpler. Saves 10–20 h with maybe 80% of the visual quality of 10-segment.

**Option B — Drop pose residuals entirely (M ≡ M-noref).** Trust OmniRe's poses. If they're good enough on your scene, this saves an entire ablation study and ~10 h of hyperparameter tuning. Worst case: you discover residuals matter and you fall back to enabling them in week 5. Frame the thesis as "investigating how good off-the-shelf pose data is for explicit-Gaussian rendering."

**Option C — Compare only B0 vs. M, drop R entirely.** B0 is the rigid baseline; M is part-based rigid. R (OmniRe) is the "ceiling" that frames how much of the gap LBS recovers. Without R, your thesis says "part-based rigid beats single rigid by X dB" but cannot say "and it's only Y dB short of the LBS ceiling." Acceptable for a seminar thesis with a clear caveat.

**Option D (most aggressive) — Skip the part-based decomposition and just demonstrate that B0 fails on walking pedestrians.** Pure characterization paper. Set up B0 carefully (M1 + M6 only, ~20 h), pick a scene with walking pedestrians, show side-by-side B0 render vs. ground truth, quantify per-pedestrian PSNR degradation as a function of walking speed. Frame the thesis as "Why explicit Street Gaussians needs articulation: a quantitative study." This is a defensible thesis at 30 h, *much* faster than implementing the part-based extension. **It is also the most honest scientific contribution if you suspect M will not actually beat B0 by enough to matter.**

**Suggested combination:** A + B + C. 4–6 segments, no pose residuals, no R baseline. ~50–70 h. Squarely in budget, leaves room for surprises.

### 6.3 Blockers you may not have anticipated

1. **`extent` is not enforced.** Your spec assumes a bounding-box prune rule that doesn't exist in this fork. You will need to add one (~20–30 LOC + tuning).
2. **`MIN_POINTS_PER_OBJECT = 10000`** in the preprocess kills pedestrians at the data layer before they ever reach training. Lowering this is necessary but may surface other corner cases (objects with bad LiDAR get terrible inits).
3. **Translation is not stored in SMPL pkl** — it's recovered from camera projection. Don't expect a clean `transl: (F, 3)` field.
4. **SMPL parameters are rotation matrices, not axis-angle**, in OmniRe's storage. Your spec assumes axis-angle. Trivial conversion but flag.
5. **Joint discontinuities will be visible** in any walking-pedestrian render. Plan to acknowledge in the thesis rather than try to hide.
6. **Pedestrian `is_moving` flag is unreliable** in Waymo for slow walkers. Verify on your specific scene.
7. **The relationship between this proposal and the existing 2026-04-28 plan** (pose-conditioned 4D SH) needs to be decided. They are two different proposals. If you do both, M2's data work overlaps; M3+M4 are unique to this proposal. Pick one path.

### 6.4 Worst-case outcome

You spend 80 h. M3 works. M4 partially works — segments render plausibly, but joint seams are visible and pedestrian-region PSNR is *worse than B0* because the seams are net-negative compared to the smeared rigid-box baseline. Pose residuals help a little but not enough.

**Is this defensible as a thesis?** Yes — provided you frame the proposal honestly as a hypothesis, set up a clean experimental protocol up front (B0 vs. M with PSNR/LPIPS on pedestrian regions), and the worst-case is a *characterized negative result*. "Part-based rigid does not improve over single-rigid because seam artifacts cost more than articulation buys; LBS-style skinning is necessary. We provide quantitative evidence and ablation on a Waymo scene." That's a legitimate seminar contribution.

**It is not defensible** if the outcome is "I implemented half of M4, ran out of time, and don't have any evaluation numbers." Avoid this by frontloading the evaluation harness (M6 should arguably run *after M1*, not at the end), so that even partial implementations produce numbers.

### 6.5 Would a different formulation be more tractable?

Yes. Three alternatives in order of decreasing departure from your proposal:

1. **2-segment decomposition (torso + legs).** Captures walking gait (the main motion), avoids 90% of the joint seams (one seam at the pelvis, easy to inflate-and-overlap), drops the FK complexity since you only need pelvis frame + legs frame. Probably ~50% the work of 10-segment with maybe ~70% of the value. Strong tractability.

2. **The previous proposal** (pose-conditioned 4D SH, [2026-04-28 plan](docs/superpowers/plans/2026-04-28-pose-conditioned-pedestrians.md)). Keeps Gaussians rigid in pelvis frame, conditions appearance (not geometry) on θ. No segment decomposition, no FK, no joint seams. **Shortcoming:** can model appearance changes (clothing wrinkles, lighting) but not actual articulation — a walking person's silhouette won't change with the pose. Unlikely to beat B0 on geometric quality.

3. **Pre-fitted SMPL mesh as initialization, treat the result as a single rigid object animated by per-frame mesh deformation.** Use OmniRe's SMPL → per-frame mesh; sample Gaussians on the mesh surface; reapply mesh deformation per frame as a "rigid" transform per Gaussian (each Gaussian gets the rigid transform of its closest bone). This is essentially LBS but with hard nearest-bone instead of soft skinning weights. **Closer to OmniRe than to your stated proposal**, but cleaner than full LBS. ~70% of OmniRe's effort. Probably out of scope.

**My recommendation:** start with **2-segment decomposition** as M, run B0 in parallel, compare, and write up. If 2-segment shows clear signal at 4 weeks, extend to 4-segment. If 2-segment doesn't show signal, the thesis becomes the negative-result characterization — which is the worst-case outcome from §6.4 and is still a legitimate seminar contribution.

---

## Appendix: Verified Codebase Anchors

Concrete references the report relies on (verify if any are stale):

- Scene-graph object instantiation loop: [sgn_splatfacto_scene_graph.py:77-88](street_gaussians_ns/sgn_splatfacto_scene_graph.py#L77-L88)
- `extent` set but not enforced: [sgn_splatfacto_scene_graph.py:81](street_gaussians_ns/sgn_splatfacto_scene_graph.py#L81); `grep extent` returns only this line in the `street_gaussians_ns/` package
- Per-frame render loop: [sgn_splatfacto_scene_graph.py:320-376](street_gaussians_ns/sgn_splatfacto_scene_graph.py#L320-L376)
- The rigid-transform substitution point: [sgn_splatfacto_scene_graph.py:406-419](street_gaussians_ns/sgn_splatfacto_scene_graph.py#L406-L419) (`object2world_gs`)
- Pose residual storage and indexing: [bbox_optimizers.py:84-87](street_gaussians_ns/data/utils/bbox_optimizers.py#L84-L87) and [:140-167](street_gaussians_ns/data/utils/bbox_optimizers.py#L140-L167)
- Class filter at parser time: [dynamic_annotation.py:19](street_gaussians_ns/data/utils/dynamic_annotation.py#L19), [dynamic_annotation.py:312](street_gaussians_ns/data/utils/dynamic_annotation.py#L312)
- LiDAR points <100 drop: [dynamic_annotation.py:356-357](street_gaussians_ns/data/utils/dynamic_annotation.py#L356-L357)
- LiDAR points <10000 drop in preprocess: [scripts/pythons/generate_annotations.py:34](scripts/pythons/generate_annotations.py#L34) (`MIN_POINTS_PER_OBJECT`, per exploration agent's read; verify directly)
- Object metadata first-frame canonical: [dynamic_annotation.py:337-344](street_gaussians_ns/data/utils/dynamic_annotation.py#L337-L344)
- Bbox-optimizer config defaults: [bbox_optimizers.py:35-39](street_gaussians_ns/data/utils/bbox_optimizers.py#L35-L39)
- Bbox-optimizer learning rate: [sgn_config.py:82-85](street_gaussians_ns/sgn_config.py#L82-L85)
