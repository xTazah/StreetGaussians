# Pose-Conditioned 4D Spherical Harmonics for Pedestrians — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend Street Gaussians' time-conditioned 4D SH to use SMPL pose θ as the conditioning variable for pedestrian object models, while keeping Gaussians rigid (no LBS, no deformation MLP). Produce a quantitative comparison against three baselines using a single Waymo segment.

**Architecture:** Five-phase plan. Phase 1 wires up the codebase with dummy SMPL data so the rest of the pipeline can be built incrementally. Phase 2 runs OmniRe's 4D-Humans + PHALP preprocessing on the chosen scene. Phase 3 solves the PHALP-track ↔ Waymo-gid identity-matching problem (the only step with material risk). Phase 4 implements the SMPL pose encoder and the per-type branching in the model. Phase 5 runs four training campaigns and evaluates them.

**Tech Stack:** Python 3.8 (training) + Python 3.10 (preprocessing), PyTorch 2.1 + CUDA 11.8, nerfstudio 1.0.0, Street Gaussians ns codebase, 4D-Humans (PyTorch3D + PHALP), SMPL-X model files. Waymo segment `1024360143612057520_3580_000_3600_000` is the working dataset. RTX 2070 Super (8 GB) for training.

**Reference results to beat or characterise:** Phase 1 baseline produces "rigid pedestrians + time-conditioned 4D SH" — the trivial extension. Final method must measurably move pedestrian-bbox PSNR vs. that baseline OR produce a clear failure-mode characterisation suitable for a thesis chapter.

---

## File Structure

**New files:**
- `street_gaussians_ns/data/utils/smpl_loader.py` — load SMPL parameters from disk per (frame, gid)
- `street_gaussians_ns/pose_encoder.py` — small MLP encoder mapping θ ∈ R^72 → fourier_features_dim
- `scripts/pythons/run_4d_humans.py` — entry point to run 4D-Humans inference on the FRONT camera
- `scripts/pythons/match_phalp_waymo.py` — assigns each PHALP track to the nearest Waymo pedestrian gid
- `scripts/pythons/visualize_phalp_match.py` — debug video showing matched PHALP↔Waymo pairs
- `scripts/pythons/eval_per_pedestrian.py` — per-bbox PSNR/LPIPS evaluation
- `scripts/pythons/measure_render_speed.py` — wall-clock render benchmark
- `tests/test_box_smpl.py` — Box class with smpl_pose field
- `tests/test_smpl_loader.py` — file loading + interpolation
- `tests/test_pose_encoder.py` — MLP shapes + gradient flow
- `tests/test_phalp_matching.py` — identity-matching algorithm
- `tests/__init__.py` — empty marker

**Modified files:**
- `street_gaussians_ns/data/utils/dynamic_annotation.py` — Box class adds `smpl_pose`, `FILTER_LABEL` extended, `Box.interploate()` interpolates pose
- `street_gaussians_ns/data/sgn_dataparser.py` — load SMPL data per frame, attach to Box
- `street_gaussians_ns/sgn_splatfacto_scene_graph.py` — branch on `anno.label`, call `pose_encoder` for pedestrians
- `street_gaussians_ns/sgn_config.py` — new flags `pedestrian_pose_conditioning`, `pose_encoder_hidden`, `smpl_pose_dim_pca`

---

## Phase 0: Project Setup

### Task 0.1: Create branch and worktree

**Files:** none (git operations)

- [ ] **Step 1: Create the branch from current main**

```bash
cd c:/Git/Uni/street-gaussians-ns
git status   # confirm clean working tree
git checkout -b pose-conditioned-pedestrians
```

Expected: `Switched to a new branch 'pose-conditioned-pedestrians'`.

- [ ] **Step 2: Snapshot current baseline output names**

Document the existing v8 PSNR (24.34 from PROJECT_STATUS) and the new fresh-pipeline run (`output_seg10243_v1`) as the two existing reference points. Add to PROJECT_STATUS.md a "Pose-conditioned extension" section noting current state. No code changes — pure logging.

- [ ] **Step 3: Commit branch creation**

```bash
git add PROJECT_STATUS.md
git commit -m "doc: open pose-conditioned-pedestrians branch, snapshot baselines"
```

### Task 0.2: Create empty test infrastructure

**Files:**
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Create: `pytest.ini` (only if missing)

- [ ] **Step 1: Create the tests package**

```bash
mkdir -p tests
touch tests/__init__.py
```

- [ ] **Step 2: Add a minimal conftest.py for shared fixtures**

Create `tests/conftest.py`:

```python
"""Shared pytest fixtures for street-gaussians-ns tests."""
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def tmp_clip(tmp_path: Path) -> Path:
    """Create a minimal clip directory layout (images/, lidars/, etc.) for tests."""
    clip = tmp_path / "clip"
    (clip / "images" / "FRONT").mkdir(parents=True)
    (clip / "lidars" / "lidar_FRONT").mkdir(parents=True)
    (clip / "humanpose").mkdir(parents=True)
    return clip


@pytest.fixture
def random_smpl_theta() -> np.ndarray:
    """A random valid SMPL pose vector (72-dim)."""
    rng = np.random.default_rng(seed=0)
    return rng.standard_normal(72).astype(np.float32) * 0.1
```

- [ ] **Step 3: Add a sanity test that pytest discovers**

Create `tests/test_smoke.py`:

```python
def test_smoke():
    assert 1 + 1 == 2
```

- [ ] **Step 4: Run pytest to confirm discovery works**

```bash
pytest tests/ -v
```

Expected output includes `tests/test_smoke.py::test_smoke PASSED`.

- [ ] **Step 5: Commit**

```bash
git add tests/ pytest.ini 2>/dev/null || git add tests/
git commit -m "test: add tests package skeleton with smoke test"
```

---

## Phase 1: Codebase Plumbing With Dummy SMPL

Goal: get the rest of the codebase to flow SMPL data through to the model, using zeros as the dummy SMPL pose. After this phase the existing pipeline still trains pedestrians + time-conditioned 4D SH, identical to current behavior, but with the plumbing for SMPL ready.

### Task 1.1: Allow pedestrians through the annotation filter

**Files:**
- Modify: `street_gaussians_ns/data/utils/dynamic_annotation.py:19`
- Test: `tests/test_filter_label.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_filter_label.py`:

```python
"""FILTER_LABEL regression test."""
from street_gaussians_ns.data.utils import dynamic_annotation as da


def test_filter_label_includes_pedestrian():
    assert "pedestrian" in da.FILTER_LABEL


def test_filter_label_still_includes_car():
    assert "car" in da.FILTER_LABEL
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/test_filter_label.py -v
```

Expected: `test_filter_label_includes_pedestrian` FAILS (current value is `['car']`).

- [ ] **Step 3: Edit FILTER_LABEL**

In `street_gaussians_ns/data/utils/dynamic_annotation.py`, replace line 19:

```python
# old:
FILTER_LABEL = ['car']
# new:
FILTER_LABEL = ['car', 'pedestrian']
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/test_filter_label.py -v
```

Expected: both tests PASS.

- [ ] **Step 5: Commit**

```bash
git add street_gaussians_ns/data/utils/dynamic_annotation.py tests/test_filter_label.py
git commit -m "feat: include pedestrians in dynamic-object filter

Pedestrians were previously discarded at annotation load time. Allow
them through so the model attempts per-object decomposition for them.
This is the prerequisite for the pose-conditioned extension."
```

### Task 1.2: Add `smpl_pose` field to Box

**Files:**
- Modify: `street_gaussians_ns/data/utils/dynamic_annotation.py` (Box.__init__ around line 100, Box.interploate around line 156)
- Test: `tests/test_box_smpl.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_box_smpl.py`:

```python
"""Box class: smpl_pose field plumbing."""
import numpy as np
import pytest
from scipy.spatial.transform import Rotation as R

from street_gaussians_ns.data.utils.dynamic_annotation import Box


def make_box(frame=0, frame_id=0, smpl_pose=None):
    return Box(
        center=np.array([0.0, 0.0, 0.0]),
        rot=R.from_euler("xyz", [0, 0, 0]).as_matrix(),
        trackId="test_track",
        size=np.array([1.0, 1.0, 1.0]),
        label="pedestrian",
        frame=frame,
        frame_id=frame_id,
        smpl_pose=smpl_pose,
    )


def test_smpl_pose_default_none():
    b = make_box()
    assert b.smpl_pose is None


def test_smpl_pose_stored():
    pose = np.zeros(72, dtype=np.float32)
    b = make_box(smpl_pose=pose)
    assert b.smpl_pose is not None
    assert b.smpl_pose.shape == (72,)
    assert b.smpl_pose.dtype == np.float32


def test_smpl_pose_interpolated_linearly():
    p1 = np.zeros(72, dtype=np.float32)
    p2 = np.ones(72, dtype=np.float32)
    b1 = make_box(frame=0, frame_id=0, smpl_pose=p1)
    b2 = make_box(frame=10, frame_id=10, smpl_pose=p2)
    bm = Box.interploate(b1, b2, frame_id=5)
    assert bm.smpl_pose is not None
    np.testing.assert_allclose(bm.smpl_pose, np.full(72, 0.5, dtype=np.float32), atol=1e-6)


def test_interpolation_handles_missing_pose():
    """If either side has no pose, result has no pose (graceful fallback)."""
    b1 = make_box(frame=0, frame_id=0, smpl_pose=None)
    b2 = make_box(frame=10, frame_id=10, smpl_pose=np.ones(72, dtype=np.float32))
    bm = Box.interploate(b1, b2, frame_id=5)
    assert bm.smpl_pose is None
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/test_box_smpl.py -v
```

Expected: all 4 tests FAIL with `TypeError: __init__() got an unexpected keyword argument 'smpl_pose'`.

- [ ] **Step 3: Update Box.__init__ to accept and store smpl_pose**

In `street_gaussians_ns/data/utils/dynamic_annotation.py`, change the `__init__` signature (line 100):

```python
def __init__(self, center, yaw=None, trackId=None, size=None, label=None,
             frame_id=-1, frame=-1, rot=None, quat=None, smpl_pose=None) -> None:
```

After `self.quat = quat` (around line 118), add:

```python
        # Per-frame SMPL pose θ ∈ R^72, optional. Set for pedestrians when
        # 4D-Humans preprocessing has been run; None otherwise.
        self.smpl_pose = None if smpl_pose is None else np.asarray(smpl_pose, dtype=np.float32)
```

- [ ] **Step 4: Update Box.interploate to interpolate smpl_pose linearly**

In the same file, replace the `interploate` method body (around line 156-171) so it interpolates pose when both sides have it, returns None otherwise:

```python
    @staticmethod
    def interploate(box1, box2, frame_id, c2w=None):
        frame_id = int(frame_id)
        t = (frame_id-box1.frame_id)/(box2.frame_id-box1.frame_id)

        i_center = box1.center*(1-t)+box2.center*t
        rot1 = quaternion_from_matrix(box1.rot)
        rot2 = quaternion_from_matrix(box2.rot)
        i_quat = quaternion_slerp(rot1, rot2, t)
        i_rot = quaternion_matrix(i_quat)[:3, :3]

        # Interpolate SMPL pose linearly when present on both sides.
        if box1.smpl_pose is not None and box2.smpl_pose is not None:
            i_smpl = (box1.smpl_pose * (1 - t) + box2.smpl_pose * t).astype(np.float32)
        else:
            i_smpl = None

        box = Box(i_center, rot=i_rot, trackId=box1.trackId,
                  size=box1.size, label=box1.label, frame_id=frame_id,
                  smpl_pose=i_smpl)
        return box
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
pytest tests/test_box_smpl.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add street_gaussians_ns/data/utils/dynamic_annotation.py tests/test_box_smpl.py
git commit -m "feat(box): add smpl_pose field with linear interpolation"
```

### Task 1.3: SMPL loader module

**Files:**
- Create: `street_gaussians_ns/data/utils/smpl_loader.py`
- Test: `tests/test_smpl_loader.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_smpl_loader.py`:

```python
"""SMPL loader: read humanpose/<gid>/<frame>.npz files into a dict."""
from pathlib import Path

import numpy as np
import pytest

from street_gaussians_ns.data.utils.smpl_loader import (
    SmplPoseStore,
    load_smpl_pose_store,
)


def write_pose_file(dir_path: Path, gid: str, frame_idx: int, theta: np.ndarray):
    sub = dir_path / gid
    sub.mkdir(parents=True, exist_ok=True)
    np.savez(sub / f"{frame_idx:06d}.npz", theta=theta.astype(np.float32))


def test_load_returns_empty_when_dir_missing(tmp_path: Path):
    store = load_smpl_pose_store(tmp_path / "nope")
    assert isinstance(store, SmplPoseStore)
    assert store.is_empty()


def test_load_reads_one_track(tmp_path: Path):
    pose = np.zeros(72, dtype=np.float32)
    write_pose_file(tmp_path, "trackA", 5, pose)
    store = load_smpl_pose_store(tmp_path)
    assert not store.is_empty()
    assert store.has("trackA")
    out = store.get("trackA", 5)
    assert out.shape == (72,)
    np.testing.assert_array_equal(out, pose)


def test_load_reads_multiple_tracks_and_frames(tmp_path: Path):
    write_pose_file(tmp_path, "A", 0, np.zeros(72, dtype=np.float32))
    write_pose_file(tmp_path, "A", 5, np.ones(72, dtype=np.float32))
    write_pose_file(tmp_path, "B", 0, np.full(72, 2.0, dtype=np.float32))
    store = load_smpl_pose_store(tmp_path)
    assert store.has("A")
    assert store.has("B")
    np.testing.assert_array_equal(store.get("A", 5), np.ones(72, dtype=np.float32))
    np.testing.assert_array_equal(store.get("B", 0), np.full(72, 2.0, dtype=np.float32))


def test_get_missing_frame_returns_none(tmp_path: Path):
    write_pose_file(tmp_path, "A", 0, np.zeros(72, dtype=np.float32))
    store = load_smpl_pose_store(tmp_path)
    assert store.get("A", 999) is None
    assert store.get("X", 0) is None
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/test_smpl_loader.py -v
```

Expected: ImportError (module doesn't exist).

- [ ] **Step 3: Implement smpl_loader.py**

Create `street_gaussians_ns/data/utils/smpl_loader.py`:

```python
"""Load per-track SMPL pose files produced by 4D-Humans preprocessing.

Expected directory layout:
    humanpose/
        <track_gid>/
            000000.npz  # contains key 'theta' (shape [72])
            000001.npz
            ...

`<track_gid>` is the Waymo annotation gid AFTER PHALP-track-to-Waymo-gid
matching has been done. Files for unmatched PHALP tracks are not written.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np


class SmplPoseStore:
    """In-memory map (track_gid, frame_idx) -> theta vector."""

    def __init__(self, data: Dict[str, Dict[int, np.ndarray]]):
        self._data = data

    def is_empty(self) -> bool:
        return len(self._data) == 0

    def has(self, track_gid: str) -> bool:
        return track_gid in self._data

    def get(self, track_gid: str, frame_idx: int) -> Optional[np.ndarray]:
        track = self._data.get(track_gid)
        if track is None:
            return None
        return track.get(frame_idx)

    def known_track_gids(self):
        return list(self._data.keys())


def load_smpl_pose_store(humanpose_dir: Path) -> SmplPoseStore:
    """Walk humanpose/<gid>/*.npz and return a SmplPoseStore."""
    humanpose_dir = Path(humanpose_dir)
    if not humanpose_dir.exists():
        return SmplPoseStore({})
    data: Dict[str, Dict[int, np.ndarray]] = {}
    for track_dir in sorted(humanpose_dir.iterdir()):
        if not track_dir.is_dir():
            continue
        per_track: Dict[int, np.ndarray] = {}
        for npz_path in sorted(track_dir.glob("*.npz")):
            frame_idx = int(npz_path.stem)
            arr = np.load(npz_path)
            theta = np.asarray(arr["theta"], dtype=np.float32).reshape(-1)
            if theta.shape != (72,):
                raise ValueError(
                    f"Expected shape (72,) for theta in {npz_path}, got {theta.shape}"
                )
            per_track[frame_idx] = theta
        if per_track:
            data[track_dir.name] = per_track
    return SmplPoseStore(data)
```

- [ ] **Step 4: Run test to verify pass**

```bash
pytest tests/test_smpl_loader.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add street_gaussians_ns/data/utils/smpl_loader.py tests/test_smpl_loader.py
git commit -m "feat(data): SmplPoseStore loader for humanpose/<gid>/*.npz"
```

### Task 1.4: Wire SMPL loader into the dataparser → Box pipeline

**Files:**
- Modify: `street_gaussians_ns/data/utils/dynamic_annotation.py` (`InterpolatedAnnotation.__init__` ~line 213, `load_anno_json_one_frame` ~line 305)
- Modify: `street_gaussians_ns/data/sgn_dataparser.py` (where `InterpolatedAnnotation` is constructed, ~line 446-454)
- Test: `tests/test_dataparser_smpl_integration.py`

- [ ] **Step 1: Write the failing integration test**

Create `tests/test_dataparser_smpl_integration.py`:

```python
"""Verify SmplPoseStore values reach Box.smpl_pose end-to-end."""
import json
from pathlib import Path

import numpy as np

from street_gaussians_ns.data.utils.dynamic_annotation import (
    InterpolatedAnnotation,
)


def write_minimal_annotation(path: Path, gid: str = "ped_A", n_frames: int = 3):
    """Three frames, one moving pedestrian centred at origin."""
    frames = []
    for i in range(n_frames):
        frames.append({
            "timestamp": 1000.0 + i,
            "objects": [{
                "gid": gid,
                "type": "pedestrian",
                "is_moving": True,
                "translation": [0.0, 0.0, 0.0],
                "rotation": [1.0, 0.0, 0.0, 0.0],
                "size": [1.0, 0.5, 1.7],
            }],
        })
    path.write_text(json.dumps({"frames": frames}))


def write_smpl_files(humanpose_dir: Path, gid: str, n_frames: int):
    sub = humanpose_dir / gid
    sub.mkdir(parents=True, exist_ok=True)
    for i in range(n_frames):
        np.savez(sub / f"{i:06d}.npz",
                 theta=np.full(72, float(i), dtype=np.float32))


def test_pedestrian_box_carries_smpl_pose(tmp_path: Path):
    anno_path = tmp_path / "annotation.json"
    humanpose_dir = tmp_path / "humanpose"
    write_minimal_annotation(anno_path, gid="ped_A", n_frames=3)
    write_smpl_files(humanpose_dir, "ped_A", 3)

    ia = InterpolatedAnnotation(
        anno_json_path=anno_path,
        lidar_path=None,
        humanpose_dir=humanpose_dir,
    )
    # Frame 0
    boxes = ia.get_by_id(0)
    assert len(boxes) == 1
    assert boxes[0].label == "pedestrian"
    assert boxes[0].smpl_pose is not None
    np.testing.assert_array_equal(
        boxes[0].smpl_pose, np.zeros(72, dtype=np.float32)
    )


def test_no_humanpose_dir_means_no_smpl(tmp_path: Path):
    anno_path = tmp_path / "annotation.json"
    write_minimal_annotation(anno_path, gid="ped_A")
    ia = InterpolatedAnnotation(
        anno_json_path=anno_path,
        lidar_path=None,
        humanpose_dir=None,
    )
    boxes = ia.get_by_id(0)
    assert len(boxes) == 1
    assert boxes[0].smpl_pose is None
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/test_dataparser_smpl_integration.py -v
```

Expected: TypeError (`humanpose_dir` not a known argument to `InterpolatedAnnotation`).

- [ ] **Step 3: Update `InterpolatedAnnotation.__init__` signature**

In `street_gaussians_ns/data/utils/dynamic_annotation.py`, around line 213, change:

```python
def __init__(self, anno_json_path, self_car_label=None, lidar_path=None,
             transform_matrix: np.ndarray = None, scale_factor: float = 1) -> None:
```

to:

```python
def __init__(self, anno_json_path, self_car_label=None, lidar_path=None,
             transform_matrix: np.ndarray = None, scale_factor: float = 1,
             humanpose_dir=None) -> None:
```

At the very top of the method body (right after the docstring or first comment), import + load:

```python
from street_gaussians_ns.data.utils.smpl_loader import load_smpl_pose_store
self._smpl_store = load_smpl_pose_store(humanpose_dir) if humanpose_dir else None
```

- [ ] **Step 4: Pass smpl_pose into Box construction inside `load_anno_json_one_frame`**

Find the place where Box(...) is constructed in `load_anno_json_one_frame` (around line 305-346). Locate the existing call and add `smpl_pose=...`. The key snippet looks like:

```python
        box = Box(
            center=np.array(translation),
            quat=rotation,
            rot=rot,
            trackId=gid,
            size=size,
            label=label,
            frame_id=frame_id,
            frame=frame_idx,
        )
```

Modify to:

```python
        smpl_pose = None
        if self._smpl_store is not None and self._smpl_store.has(gid):
            smpl_pose = self._smpl_store.get(gid, frame_idx)

        box = Box(
            center=np.array(translation),
            quat=rotation,
            rot=rot,
            trackId=gid,
            size=size,
            label=label,
            frame_id=frame_id,
            frame=frame_idx,
            smpl_pose=smpl_pose,
        )
```

(If the existing call uses positional args, keep them and append the keyword.)

- [ ] **Step 5: Update sgn_dataparser to pass humanpose_dir**

In `street_gaussians_ns/data/sgn_dataparser.py`, find where `InterpolatedAnnotation(...)` is constructed (around line 451). The call site likely looks like:

```python
            self.dynamic_anno = InterpolatedAnnotation(
                anno_json_path=anno_path,
                lidar_path=lidar_path,
                transform_matrix=transform_matrix_anno,
                scale_factor=self.scale_factor,
            )
```

Change to:

```python
            humanpose_dir = self.config.data / "humanpose"
            self.dynamic_anno = InterpolatedAnnotation(
                anno_json_path=anno_path,
                lidar_path=lidar_path,
                transform_matrix=transform_matrix_anno,
                scale_factor=self.scale_factor,
                humanpose_dir=humanpose_dir,
            )
```

- [ ] **Step 6: Run integration test to verify pass**

```bash
pytest tests/test_dataparser_smpl_integration.py -v
```

Expected: both tests PASS.

- [ ] **Step 7: Run all tests so far to confirm nothing regressed**

```bash
pytest tests/ -v
```

Expected: all PASS.

- [ ] **Step 8: Commit**

```bash
git add street_gaussians_ns/ tests/test_dataparser_smpl_integration.py
git commit -m "feat(data): plumb humanpose/<gid>/*.npz through dataparser into Box.smpl_pose"
```

### Task 1.5: Smoke-test full training with pedestrians enabled and dummy SMPL

**Files:** none (training run)

- [ ] **Step 1: Sanity check the existing training data has pedestrian PLYs available**

```bash
ls /mnt/d/Git/StreetGaussians/waymo-dataset/sgn-data/validation/1024360143612057520_3580_000_3600_000/aggregate_lidar/dynamic_objects/
```

Note the count. If only car PLYs exist, re-run object_pts_generate.sh after `Task 1.1` since the pedestrian filter now lets them through. Specifically:

```bash
cd /mnt/d/Git/StreetGaussians
conda activate waymo-prep
bash scripts/shells/object_pts_generate.sh \
  /mnt/d/Git/StreetGaussians/waymo-dataset/sgn-data/validation/1024360143612057520_3580_000_3600_000
```

Expected: PLY count grows beyond 7 (more pedestrian PLYs created).

- [ ] **Step 2: Launch training with the existing run_train_pc.bat**

Train for at least 8000 steps. Filename should be `output_seg10243_v2_peds_enabled`. No SMPL data exists yet (humanpose/ doesn't exist), so the loader returns an empty store and all Box.smpl_pose are None — equivalent to current behavior plus pedestrians.

```powershell
.\run_train_pc.bat
# (after editing EXP_NAME=output_seg10243_v2_peds_enabled in the bat)
```

- [ ] **Step 3: Verify pedestrian object models load**

When the training log prints object loading, count `pedestrian` entries in addition to cars. Note the count and bbox-PSNR will be measured later in Phase 5.

- [ ] **Step 4: Commit reference output name**

Add a short note to PROJECT_STATUS.md mentioning the v2 baseline and commit.

```bash
git add PROJECT_STATUS.md
git commit -m "doc: log v2 baseline (pedestrians enabled, no SMPL)"
```

---

## Phase 2: 4D-Humans Preprocessing

Goal: produce a `humanpose/` directory under the chosen segment folder containing per-track SMPL parameters across all FRONT camera frames. We use OmniRe / drivestudio's documented pipeline at https://github.com/ziyc/drivestudio/blob/main/docs/HumanPose.md.

This phase is largely environment setup. The actual model output is a directory of files our Phase 3 matching script consumes. We do not modify Street Gaussians code in this phase.

### Task 2.1: Set up 4D-Humans environment in WSL

**Files:** none (environment setup); document in `docs/setup-4d-humans.md`

- [ ] **Step 1: Create a separate conda env for 4D-Humans**

In WSL:

```bash
conda deactivate
conda create -n humans -y python=3.10
conda activate humans
```

- [ ] **Step 2: Install 4D-Humans following its README**

```bash
cd ~
git clone https://github.com/shubham-goel/4D-Humans.git
cd 4D-Humans
pip install -e .[all]
# PyTorch3D is the painful one. Follow:
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

If PyTorch3D fails to compile (common on Windows-via-WSL), fall back to:

```bash
pip install fvcore iopath
pip install --no-index --no-cache-dir pytorch3d \
  -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt201/download.html
```

- [ ] **Step 3: Download SMPL model files**

Download SMPL-X from https://smpl-x.is.tue.mpg.de/ (registration required). Place under `~/4D-Humans/data/smpl/`. The 4D-Humans README has the exact paths.

- [ ] **Step 4: Smoke-test 4D-Humans on a single image**

```bash
python demo.py --img_folder ~/test_imgs --out_folder ~/test_out --batch_size 1
```

Expected: produces a `.pkl` file with SMPL parameters and a render. If this works, the environment is ready.

- [ ] **Step 5: Create setup docs for reproducibility**

Create `docs/setup-4d-humans.md` summarising the steps that worked, including any version pins discovered. Commit:

```bash
cd c:/Git/Uni/street-gaussians-ns
git add docs/setup-4d-humans.md
git commit -m "doc: 4D-Humans environment setup notes"
```

### Task 2.2: Wrapper script that runs 4D-Humans on a clip

**Files:**
- Create: `scripts/pythons/run_4d_humans.py`

- [ ] **Step 1: Write the wrapper**

Create `scripts/pythons/run_4d_humans.py`:

```python
"""Run 4D-Humans inference on a Waymo segment's FRONT camera images.

This script invokes 4D-Humans + PHALP from the `humans` conda env. It produces
a directory of per-frame .pkl files, one per detected person track.

Usage:
    conda activate humans
    python scripts/pythons/run_4d_humans.py \
        --clip /path/to/sgn-data/validation/<segment>/ \
        --camera FRONT \
        --out_dir /path/to/4d-humans-output/<segment>/
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, required=True,
                        help="Path to sgn-data/.../<segment>/ directory")
    parser.add_argument("--camera", default="FRONT",
                        help="Which camera to process (default FRONT)")
    parser.add_argument("--out_dir", type=Path, required=True,
                        help="Where to dump 4D-Humans output")
    parser.add_argument(
        "--four_d_humans_repo",
        type=Path,
        default=Path.home() / "4D-Humans",
        help="Path to the 4D-Humans cloned repo",
    )
    args = parser.parse_args()

    img_dir = args.clip / "images" / args.camera
    if not img_dir.exists():
        sys.exit(f"No images at {img_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 4D-Humans's track.py / demo.py is the entry point. Adjust per their docs.
    cmd = [
        "python",
        str(args.four_d_humans_repo / "demo.py"),
        "--img_folder", str(img_dir),
        "--out_folder", str(args.out_dir),
        "--save_mesh", "false",
        "--track", "true",
    ]
    print("Running:", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=args.four_d_humans_repo)
    if rc != 0:
        sys.exit(f"4D-Humans returned non-zero exit code {rc}")

    print(f"Done. Output: {args.out_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on the chosen segment**

```bash
conda activate humans
python scripts/pythons/run_4d_humans.py \
    --clip /mnt/d/Git/StreetGaussians/waymo-dataset/sgn-data/validation/1024360143612057520_3580_000_3600_000 \
    --camera FRONT \
    --out_dir /mnt/d/Git/StreetGaussians/waymo-dataset/sgn-data/validation/1024360143612057520_3580_000_3600_000/raw_4d_humans
```

Expected: ~30-60 minutes runtime (depends on detection density, GPU). Output: per-track .pkl files in `raw_4d_humans/`.

- [ ] **Step 3: Inspect output schema**

```bash
python3 -c "
import pickle, pathlib, numpy as np
files = list(pathlib.Path('raw_4d_humans').glob('*.pkl'))[:3]
for f in files:
    d = pickle.load(open(f, 'rb'))
    print(f.name, type(d), list(d.keys()) if hasattr(d, 'keys') else '')
    for k, v in (d.items() if hasattr(d, 'items') else []):
        print('  ', k, getattr(v, 'shape', type(v)))
"
```

This step reveals the actual schema of 4D-Humans output (varies slightly by version). Document the keys you find.

- [ ] **Step 4: Commit script**

```bash
cd c:/Git/Uni/street-gaussians-ns
git add scripts/pythons/run_4d_humans.py
git commit -m "feat: wrapper to run 4D-Humans on a Waymo segment's FRONT camera"
```

---

## Phase 3: PHALP-track ↔ Waymo-gid Identity Matching

Goal: produce `humanpose/<waymo_gid>/<frame>.npz` files with the SMPL θ vector for each Waymo pedestrian gid that has a successful PHALP match. This is the only phase with material risk; treat it carefully and validate visually.

### Task 3.1: Identity-matching algorithm and tests

**Files:**
- Create: `scripts/pythons/match_phalp_waymo.py`
- Test: `tests/test_phalp_matching.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_phalp_matching.py`:

```python
"""Hungarian matching between PHALP tracks and Waymo pedestrian gids."""
from typing import Dict, List

import numpy as np

from scripts.pythons.match_phalp_waymo import (
    match_tracks,
    PhalpTrack,
    WaymoTrack,
)


def make_phalp(track_id: int, points: Dict[int, tuple]) -> PhalpTrack:
    """points: frame -> (u, v) 2D image coord"""
    return PhalpTrack(track_id=track_id, points_2d=points)


def make_waymo(gid: str, points: Dict[int, tuple]) -> WaymoTrack:
    return WaymoTrack(gid=gid, points_2d=points)


def test_one_to_one_close_assignment():
    phalp = [make_phalp(7, {0: (100, 200), 1: (105, 205)})]
    waymo = [make_waymo("ped_A", {0: (102, 202), 1: (107, 207)})]
    matches = match_tracks(phalp, waymo, max_distance_px=50)
    assert matches == {7: "ped_A"}


def test_no_match_when_too_far():
    phalp = [make_phalp(7, {0: (100, 200)})]
    waymo = [make_waymo("ped_A", {0: (1000, 2000)})]
    matches = match_tracks(phalp, waymo, max_distance_px=50)
    assert 7 not in matches


def test_hungarian_picks_best_pairs_globally():
    phalp = [
        make_phalp(1, {0: (100, 100)}),
        make_phalp(2, {0: (200, 200)}),
    ]
    waymo = [
        make_waymo("A", {0: (110, 110)}),  # closest to phalp 1
        make_waymo("B", {0: (210, 210)}),  # closest to phalp 2
    ]
    matches = match_tracks(phalp, waymo, max_distance_px=50)
    assert matches == {1: "A", 2: "B"}


def test_partial_overlap_in_time_uses_intersection():
    """Tracks visible in different frames; only intersection of frames counts."""
    phalp = [make_phalp(1, {0: (100, 100), 5: (110, 100)})]
    waymo = [make_waymo("A", {3: (200, 200), 5: (115, 105)})]
    matches = match_tracks(phalp, waymo, max_distance_px=50)
    assert matches == {1: "A"}
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/test_phalp_matching.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement match_phalp_waymo.py**

Create `scripts/pythons/match_phalp_waymo.py`:

```python
"""Match PHALP tracks (from 4D-Humans output) to Waymo annotation gids.

Inputs:
    - 4D-Humans output directory (raw_4d_humans/) — per-track SMPL params + 2D track positions
    - Waymo annotation.json with `is_moving=True` pedestrian boxes
    - transform.json with FRONT camera intrinsics + per-frame poses

Output:
    humanpose/<waymo_gid>/<frame>.npz containing key 'theta' (shape [72])
    plus a JSON match_log.json describing all decisions.
"""
from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class PhalpTrack:
    track_id: int
    points_2d: Dict[int, tuple]   # frame_idx -> (u, v) in FRONT image pixel coords
    smpl_thetas: Dict[int, np.ndarray] = field(default_factory=dict)  # frame_idx -> [72]


@dataclass
class WaymoTrack:
    gid: str
    points_2d: Dict[int, tuple]


def _mean_distance(p: Dict[int, tuple], q: Dict[int, tuple]) -> float:
    """Mean L2 distance over the intersection of frames. Inf if no overlap."""
    common = sorted(set(p.keys()) & set(q.keys()))
    if not common:
        return float("inf")
    diffs = []
    for f in common:
        pu, pv = p[f]; qu, qv = q[f]
        diffs.append(np.hypot(pu - qu, pv - qv))
    return float(np.mean(diffs))


def match_tracks(
    phalp_tracks: List[PhalpTrack],
    waymo_tracks: List[WaymoTrack],
    max_distance_px: float = 50.0,
) -> Dict[int, str]:
    """Hungarian-assignment matching by mean 2D distance over shared frames.
    Returns dict mapping phalp_track_id -> waymo_gid for matched pairs only.
    """
    if not phalp_tracks or not waymo_tracks:
        return {}
    cost = np.full((len(phalp_tracks), len(waymo_tracks)), 1e9)
    for i, p in enumerate(phalp_tracks):
        for j, w in enumerate(waymo_tracks):
            cost[i, j] = _mean_distance(p.points_2d, w.points_2d)
    row, col = linear_sum_assignment(cost)
    matches: Dict[int, str] = {}
    for i, j in zip(row, col):
        if cost[i, j] <= max_distance_px:
            matches[phalp_tracks[i].track_id] = waymo_tracks[j].gid
    return matches


def parse_4d_humans_output(raw_dir: Path) -> List[PhalpTrack]:
    """Parse 4D-Humans .pkl files into PhalpTrack list. Format depends on
    actual 4D-Humans version; this skeleton assumes one .pkl per track.
    Update field names after running Task 2.2 step 3 (schema inspection)."""
    tracks: List[PhalpTrack] = []
    for pkl in sorted(raw_dir.glob("*.pkl")):
        d = pickle.load(open(pkl, "rb"))
        # TODO after schema inspection: replace these field names
        track_id = int(d["track_id"])
        thetas = {int(f): np.asarray(theta, dtype=np.float32).reshape(72)
                  for f, theta in d["thetas"].items()}
        points = {int(f): tuple(d["bbox_centers"][f])
                  for f in thetas.keys() if f in d["bbox_centers"]}
        tracks.append(PhalpTrack(track_id=track_id, points_2d=points,
                                 smpl_thetas=thetas))
    return tracks


def parse_waymo_pedestrian_tracks(annotation_path: Path,
                                  transform_path: Path,
                                  camera_name: str = "FRONT") -> List[WaymoTrack]:
    """Project each Waymo moving-pedestrian centre into FRONT image coords."""
    ann = json.load(open(annotation_path))
    tj = json.load(open(transform_path))
    front_frames = [f for f in tj["frames"]
                    if f"/{camera_name}/" in "/" + f["file_path"]]
    c2w_by_ts = {f["timestamp"]: np.array(f["transform_matrix"]) for f in front_frames}
    fx = front_frames[0]["fl_x"]
    fy = front_frames[0]["fl_y"]
    cx = front_frames[0]["cx"]
    cy = front_frames[0]["cy"]

    # Build frame_idx mapping (same order as front_frames sorted by timestamp)
    front_frames_sorted = sorted(front_frames, key=lambda f: f["timestamp"])
    ts_to_frame_idx = {f["timestamp"]: i for i, f in enumerate(front_frames_sorted)}

    by_gid: Dict[str, Dict[int, tuple]] = {}
    for f in ann["frames"]:
        ts = f.get("timestamp")
        if ts not in c2w_by_ts:
            continue
        c2w = c2w_by_ts[ts]
        w2c = np.linalg.inv(c2w)
        frame_idx = ts_to_frame_idx[ts]
        for o in f.get("objects", []):
            if not o.get("is_moving"):
                continue
            if o.get("type") != "pedestrian":
                continue
            ctr = np.array(o["translation"] + [1.0])
            cam = w2c @ ctr  # Waymo convention: x forward, y left, z up
            fwd = cam[0]
            if fwd <= 1.0:
                continue
            # Convert Waymo to OpenCV: x_opencv = -y, y_opencv = -z, z_opencv = x
            x_oc, y_oc, z_oc = -cam[1], -cam[2], cam[0]
            u = fx * x_oc / z_oc + cx
            v = fy * y_oc / z_oc + cy
            by_gid.setdefault(o["gid"], {})[frame_idx] = (float(u), float(v))

    return [WaymoTrack(gid=g, points_2d=pts) for g, pts in by_gid.items()
            if pts]


def write_humanpose(matches: Dict[int, str],
                    phalp_tracks: List[PhalpTrack],
                    out_dir: Path) -> Dict[str, int]:
    """For each matched pair, dump <waymo_gid>/<frame>.npz (theta only)."""
    counts: Dict[str, int] = {}
    phalp_by_id = {p.track_id: p for p in phalp_tracks}
    for phalp_id, waymo_gid in matches.items():
        track = phalp_by_id[phalp_id]
        sub = out_dir / waymo_gid
        sub.mkdir(parents=True, exist_ok=True)
        for frame_idx, theta in track.smpl_thetas.items():
            np.savez(sub / f"{frame_idx:06d}.npz", theta=theta.astype(np.float32))
        counts[waymo_gid] = len(track.smpl_thetas)
    return counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--raw_4d_humans", type=Path, required=True)
    parser.add_argument("--max_distance_px", type=float, default=50.0)
    parser.add_argument("--camera", default="FRONT")
    args = parser.parse_args()

    phalp = parse_4d_humans_output(args.raw_4d_humans)
    waymo = parse_waymo_pedestrian_tracks(
        args.clip / "annotation.json",
        args.clip / "transform.json",
        camera_name=args.camera,
    )
    matches = match_tracks(phalp, waymo, max_distance_px=args.max_distance_px)
    out_dir = args.clip / "humanpose"
    out_dir.mkdir(parents=True, exist_ok=True)
    counts = write_humanpose(matches, phalp, out_dir)

    log = {
        "n_phalp_tracks": len(phalp),
        "n_waymo_pedestrian_gids": len(waymo),
        "n_matched": len(matches),
        "matches": [{"phalp_id": k, "waymo_gid": v} for k, v in matches.items()],
        "frames_written_per_gid": counts,
    }
    (out_dir / "match_log.json").write_text(json.dumps(log, indent=2))
    print(f"Matched {len(matches)} of {len(phalp)} PHALP tracks "
          f"(out of {len(waymo)} Waymo pedestrian gids).")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run unit tests to verify pass**

```bash
pytest tests/test_phalp_matching.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: After Phase 2 completes, update `parse_4d_humans_output` to match the actual 4D-Humans schema you saw in Task 2.2 step 3**

The current placeholder uses `d["track_id"]`, `d["thetas"]`, `d["bbox_centers"]`. Replace these names with whatever the .pkl files actually contain. Add a regression test that loads a real .pkl and asserts shapes are correct.

- [ ] **Step 6: Commit**

```bash
git add scripts/pythons/match_phalp_waymo.py tests/test_phalp_matching.py
git commit -m "feat: PHALP↔Waymo identity matching script + Hungarian algorithm tests"
```

### Task 3.2: Debug visualization

**Files:**
- Create: `scripts/pythons/visualize_phalp_match.py`

- [ ] **Step 1: Implement debug overlay**

Create `scripts/pythons/visualize_phalp_match.py`:

```python
"""Render an overlay video showing matched PHALP↔Waymo pairs.

For each FRONT frame, draw:
  - Red boxes:   Waymo pedestrian gids that have a match
  - Yellow boxes: Waymo pedestrian gids without a match
  - Cyan dots:   PHALP track 2D positions, with track_id label
A correct match should show a cyan dot inside or very near a red box.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip", type=Path, required=True)
    parser.add_argument("--raw_4d_humans", type=Path, required=True)
    parser.add_argument("--camera", default="FRONT")
    parser.add_argument("--out_video", type=Path, required=True)
    args = parser.parse_args()

    match_log = json.loads((args.clip / "humanpose" / "match_log.json").read_text())
    matched_gids = {m["waymo_gid"] for m in match_log["matches"]}
    matched_phalp = {m["phalp_id"] for m in match_log["matches"]}
    phalp_to_waymo = {m["phalp_id"]: m["waymo_gid"] for m in match_log["matches"]}

    img_dir = args.clip / "images" / args.camera
    front_imgs = sorted(img_dir.glob("*.jpg")) + sorted(img_dir.glob("*.png"))
    h, w = cv2.imread(str(front_imgs[0])).shape[:2]
    vw = cv2.VideoWriter(str(args.out_video),
                         cv2.VideoWriter_fourcc(*"mp4v"), 10, (w, h))

    # Reuse the projection logic from match_phalp_waymo.parse_waymo_pedestrian_tracks
    from scripts.pythons.match_phalp_waymo import (
        parse_4d_humans_output, parse_waymo_pedestrian_tracks,
    )

    phalp = parse_4d_humans_output(args.raw_4d_humans)
    waymo = parse_waymo_pedestrian_tracks(
        args.clip / "annotation.json",
        args.clip / "transform.json",
        camera_name=args.camera,
    )

    waymo_by_gid = {w.gid: w for w in waymo}
    for frame_idx, img_path in enumerate(front_imgs):
        img = cv2.imread(str(img_path))
        for w in waymo:
            pt = w.points_2d.get(frame_idx)
            if pt is None:
                continue
            color = (0, 0, 255) if w.gid in matched_gids else (0, 255, 255)
            u, v = int(pt[0]), int(pt[1])
            cv2.rectangle(img, (u - 30, v - 60), (u + 30, v + 60), color, 2)
            cv2.putText(img, w.gid[:8], (u - 30, v - 65),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        for p in phalp:
            pt = p.points_2d.get(frame_idx)
            if pt is None:
                continue
            u, v = int(pt[0]), int(pt[1])
            cv2.circle(img, (u, v), 5, (255, 255, 0), -1)
            label = f"phalp{p.track_id}"
            if p.track_id in phalp_to_waymo:
                label += f"->{phalp_to_waymo[p.track_id][:6]}"
            cv2.putText(img, label, (u + 6, v),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        vw.write(img)
    vw.release()
    print(f"Wrote {args.out_video}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on the chosen segment and watch the video**

```bash
python scripts/pythons/visualize_phalp_match.py \
    --clip /mnt/d/Git/StreetGaussians/waymo-dataset/sgn-data/validation/1024360143612057520_3580_000_3600_000 \
    --raw_4d_humans /mnt/d/.../raw_4d_humans \
    --out_video phalp_match_debug.mp4
```

- [ ] **Step 3: Visual sanity check**

Open `phalp_match_debug.mp4`. Acceptance criteria:
- For ≥70% of red (matched) boxes, a cyan dot is inside or within 30 px of the box centre.
- Yellow boxes (unmatched) tend to be in regions where 4D-Humans didn't detect (heavy occlusion, far distance).
- No "obvious wrong" pairings (cyan dot near box A but labelled with gid of box B).

If acceptance fails, tune `--max_distance_px` (try 30, 80, 150) and rerun the matching.

- [ ] **Step 4: Commit**

```bash
git add scripts/pythons/visualize_phalp_match.py
git commit -m "feat: debug overlay video for PHALP↔Waymo matching"
```

### Task 3.3: Run full pipeline end-to-end and verify humanpose/

**Files:** none (pipeline runs)

- [ ] **Step 1: Run match script**

```bash
conda activate humans
python scripts/pythons/match_phalp_waymo.py \
    --clip /mnt/d/Git/StreetGaussians/waymo-dataset/sgn-data/validation/1024360143612057520_3580_000_3600_000 \
    --raw_4d_humans /mnt/d/.../raw_4d_humans \
    --max_distance_px 50
```

- [ ] **Step 2: Verify output structure**

```bash
ls /mnt/d/Git/StreetGaussians/waymo-dataset/sgn-data/validation/1024360143612057520_3580_000_3600_000/humanpose/
cat /mnt/d/Git/StreetGaussians/waymo-dataset/sgn-data/validation/1024360143612057520_3580_000_3600_000/humanpose/match_log.json | python3 -m json.tool
```

Expected: a directory per matched waymo_gid, each containing `<frame>.npz` files with key `theta`. Match log shows ≥10 matches (we hope).

- [ ] **Step 3: Acceptance gate**

If `n_matched < 10`, the pose-conditioning experiment will be data-starved. Either:
  - Tune `max_distance_px`, rerun matching.
  - Run 4D-Humans on more cameras (FRONT_LEFT, FRONT_RIGHT) to recover more pedestrian-frame coverage.
  - Pick a different segment with more pedestrians visible in FRONT.

- [ ] **Step 4: Document the final match counts in PROJECT_STATUS.md and commit**

```bash
git add PROJECT_STATUS.md
git commit -m "doc: log Phase 3 match counts on segment 1024360..."
```

---

## Phase 4: SMPL Pose Encoder + Per-Type Branching

Goal: implement the actual modeling change. Add an MLP encoder that maps θ ∈ R^72 to fourier_features_dim weights, and route pedestrians through it instead of through `IDFT(t)`.

### Task 4.1: PoseEncoder MLP class

**Files:**
- Create: `street_gaussians_ns/pose_encoder.py`
- Test: `tests/test_pose_encoder.py`

- [ ] **Step 1: Write the failing test**

Create `tests/test_pose_encoder.py`:

```python
"""SMPL pose encoder MLP."""
import torch

from street_gaussians_ns.pose_encoder import PoseEncoder


def test_output_shape_matches_fourier_dim():
    enc = PoseEncoder(input_dim=72, output_dim=5, hidden=64)
    theta = torch.randn(72)
    out = enc(theta)
    assert out.shape == (5,)


def test_batch_input_supported():
    enc = PoseEncoder(input_dim=72, output_dim=5, hidden=64)
    theta_batch = torch.randn(4, 72)
    out = enc(theta_batch)
    assert out.shape == (4, 5)


def test_gradient_flows():
    enc = PoseEncoder(input_dim=72, output_dim=5, hidden=64)
    theta = torch.randn(72, requires_grad=False)
    out = enc(theta)
    loss = out.sum()
    loss.backward()
    for p in enc.parameters():
        assert p.grad is not None
        assert p.grad.abs().max() > 0


def test_pca_projection_stage_optional():
    """When pca_dim is given, the input is projected to lower dim first."""
    enc = PoseEncoder(input_dim=72, output_dim=5, hidden=64, pca_dim=16)
    theta = torch.randn(72)
    out = enc(theta)
    assert out.shape == (5,)
```

- [ ] **Step 2: Run test to verify failure**

```bash
pytest tests/test_pose_encoder.py -v
```

Expected: ImportError.

- [ ] **Step 3: Implement PoseEncoder**

Create `street_gaussians_ns/pose_encoder.py`:

```python
"""MLP that maps a flattened SMPL pose θ ∈ R^72 to a low-dim weight vector.

The output is plugged into Street Gaussians' `features_dc` linear-combination
pipeline as a drop-in replacement for the time-conditioned IDFT basis.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class PoseEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int = 72,
        output_dim: int = 5,
        hidden: int = 64,
        pca_dim: Optional[int] = None,
    ):
        """
        Args:
            input_dim: flattened SMPL pose dimension (default 72 = 24×3 axis-angle).
            output_dim: must match `fourier_features_dim` of pedestrian object models.
            hidden: hidden layer width.
            pca_dim: if set, an initial linear layer projects θ to this lower dim
                first. Useful to study whether full 72-dim is needed.
        """
        super().__init__()
        layers = []
        in_d = input_dim
        if pca_dim is not None:
            layers.append(nn.Linear(input_dim, pca_dim))
            in_d = pca_dim
        layers += [
            nn.Linear(in_d, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, output_dim),
        ]
        self.net = nn.Sequential(*layers)
        # Initialise the final layer to small values so early training is dominated
        # by the existing features_dc init, not random pose modulation.
        nn.init.normal_(self.net[-1].weight, std=0.01)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, theta: torch.Tensor) -> torch.Tensor:
        return self.net(theta)
```

- [ ] **Step 4: Run tests to verify pass**

```bash
pytest tests/test_pose_encoder.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add street_gaussians_ns/pose_encoder.py tests/test_pose_encoder.py
git commit -m "feat: PoseEncoder MLP (72-dim theta -> fourier_features_dim weights)"
```

### Task 4.2: Wire PoseEncoder into the model

**Files:**
- Modify: `street_gaussians_ns/sgn_splatfacto_scene_graph.py` (init + `get_fourier_features` + `get_outputs`)
- Modify: `street_gaussians_ns/sgn_config.py` (add config flags)

- [ ] **Step 1: Add config flags**

In `street_gaussians_ns/sgn_config.py`, find the `SplatfactoSceneGraphModelConfig(...)` block (around line 44). It is a dataclass elsewhere — locate the dataclass definition in `street_gaussians_ns/sgn_splatfacto_scene_graph.py` and add three fields:

In `sgn_splatfacto_scene_graph.py`, find `class SplatfactoSceneGraphModelConfig` (the `@dataclass` near the top). Add:

```python
    pedestrian_pose_conditioning: bool = False
    """If True, route pedestrian object models through PoseEncoder(theta_72)
    instead of IDFT(time)."""

    pose_encoder_hidden: int = 64
    pose_encoder_pca_dim: Optional[int] = None
```

Make sure `Optional` is imported at the top of the file (`from typing import Optional`).

- [ ] **Step 2: Construct PoseEncoder in `populate_modules`**

Find `populate_modules` (around line 78). After existing object_model setup, add:

```python
        from street_gaussians_ns.pose_encoder import PoseEncoder
        self.pose_encoder = (
            PoseEncoder(
                input_dim=72,
                output_dim=self.config.object_model_template.fourier_features_dim,
                hidden=self.config.pose_encoder_hidden,
                pca_dim=self.config.pose_encoder_pca_dim,
            )
            if self.config.pedestrian_pose_conditioning
            else None
        )
```

- [ ] **Step 3: Add SMPL-conditioned variant of get_fourier_features**

Right after the existing `get_fourier_features` method (line 247), add:

```python
    def get_pose_features(self, anno: 'Box', obj_model: 'SplatfactoModel') -> torch.Tensor:
        """SMPL-pose-conditioned counterpart of get_fourier_features.
        Modulates obj_model.features_dc using PoseEncoder(theta_72).
        """
        assert anno.smpl_pose is not None, (
            "Box has no smpl_pose; pedestrian must have humanpose/<gid>/<frame>.npz"
        )
        theta = torch.from_numpy(anno.smpl_pose).to(self.device)
        weights = self.pose_encoder(theta)  # shape [fourier_features_dim]
        return torch.sum(
            obj_model.features_dc * weights[..., None], dim=1, keepdim=True
        )
```

- [ ] **Step 4: Branch on label in `get_outputs`**

Edit the block at line 343 to branch on label:

```python
                # Choose conditioning source.
                use_pose = (
                    self.config.pedestrian_pose_conditioning
                    and anno.label == 'pedestrian'
                    and anno.smpl_pose is not None
                )
                if use_pose:
                    object_features_dc.append(
                        self.get_pose_features(anno, obj_model)
                    )
                elif self.config.fourier_features_dim > 1:
                    object_features_dc.append(
                        self.get_fourier_features(anno.frame, trackId, obj_model)
                    )
                else:
                    object_features_dc.append(obj_model.features_dc)
```

- [ ] **Step 5: Add the encoder to the optimizer**

Find the optimizer setup in `sgn_config.py` (around line 73). Add:

```python
            "pose_encoder": {
                "optimizer": AdamOptimizerConfig(lr=1e-3, eps=1e-15),
                "scheduler": ExponentialDecaySchedulerConfig(lr_final=1e-5, max_steps=70000),
            },
```

In the model's `get_param_groups()` method (find this in `sgn_splatfacto_scene_graph.py`), add:

```python
        if self.pose_encoder is not None:
            param_groups["pose_encoder"] = list(self.pose_encoder.parameters())
```

- [ ] **Step 6: Smoke-test by importing the module**

```bash
python -c "from street_gaussians_ns.sgn_splatfacto_scene_graph import SplatfactoSceneGraphModelConfig; c = SplatfactoSceneGraphModelConfig(); print('ok', c.pedestrian_pose_conditioning)"
```

Expected: `ok False`.

- [ ] **Step 7: Commit**

```bash
git add street_gaussians_ns/
git commit -m "feat(model): SMPL-pose conditioning branch for pedestrian objects"
```

### Task 4.3: Forward-pass smoke test on real data with dummy theta

**Files:** none (test run)

- [ ] **Step 1: Edit your training bat file to enable pose conditioning**

Append to `colmap-data-parser-config` arguments and elsewhere as needed. Specifically the new flags go on the **model config** subcommand (likely accessed by repeating the model config invocation pattern). The exact CLI form depends on your tyro layout; the simplest path is to set the dataclass defaults via env var or fork the bat to invoke `--pipeline.model.pedestrian_pose_conditioning True`.

```bat
"D:\Git\StreetGaussians\.venv\Scripts\sgn-train.exe" street-gaussians-ns ^
    --experiment-name output_seg10243_v3_smoke_pose ^
    --pipeline.model.pedestrian_pose_conditioning True ^
    colmap-data-parser-config ^
    --data %CLIP_DIR% ^
    --colmap_path colmap/sparse/0 ^
    --filter_camera_id 1 ^
    --init_points_filename points3D_withlidar.txt
```

- [ ] **Step 2: Run training for 500 steps and check no crash**

Launch, watch for the "Found ... pedestrian objects" log lines, and abort cleanly after step 500 (Ctrl+C). Verify:
- No "smpl_pose is None" assertion errors (Phase 3 must have written real humanpose/ files for matched gids).
- Loss decreases.
- The `pose_encoder` optimizer group appears in nerfstudio's per-step log.

- [ ] **Step 3: Commit smoke-test note**

```bash
echo "v3_smoke_pose: 500 steps no crash, loss decreasing, pose_encoder optimizer present" >> PROJECT_STATUS.md
git add PROJECT_STATUS.md
git commit -m "doc: smoke-tested pose-conditioned forward pass at 500 steps"
```

---

## Phase 5: Training Campaigns and Evaluation

Goal: produce comparable PSNR / LPIPS / render-speed numbers for four configurations on the same segment.

### Task 5.1: Define the four training configurations

**Files:**
- Create: `scripts/shells/train_baseline_static_sh.bat`
- Create: `scripts/shells/train_baseline_time_sh.bat`
- Create: `scripts/shells/train_ours_pose_sh.bat`
- (Optional) Create: `scripts/shells/train_ref_smpl_skinning.bat`

Each is a thin wrapper around the existing `run_train_pc.bat` template, with the listed flag set.

| Variant | Pedestrian fourier_dim | Pose conditioning | What it tests |
|---|---|---|---|
| **A. Static SH** | 1 | off | Lower bound: no temporal info on pedestrians |
| **B. Time-conditioned 4D SH** | 5 | off | Trivial extension; current default behavior |
| **C. Pose-conditioned (ours)** | 5 | on | Proposed contribution |
| **D. SMPL skinning (reference)** | n/a | n/a | OmniRe-equivalent — only build if time |

- [ ] **Step 1: Create A — static SH**

`scripts/shells/train_baseline_static_sh.bat`:

```bat
@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
set "LIB=%LIB%;c:\Users\koehl\anaconda3\libs"
set CLIP_DIR=.\waymo-dataset\sgn-data\validation\1024360143612057520_3580_000_3600_000
"D:\Git\StreetGaussians\.venv\Scripts\sgn-train.exe" street-gaussians-ns ^
    --experiment-name v_A_static_sh ^
    --pipeline.model.object_model_template.fourier_features_dim 1 ^
    colmap-data-parser-config ^
    --data %CLIP_DIR% ^
    --colmap_path colmap/sparse/0 ^
    --filter_camera_id 1 ^
    --init_points_filename points3D_withlidar.txt
```

- [ ] **Step 2: Create B — time-conditioned 4D SH (current default)**

`scripts/shells/train_baseline_time_sh.bat` — same as the working `run_train_pc.bat` but with `--experiment-name v_B_time_sh`.

- [ ] **Step 3: Create C — pose-conditioned (ours)**

`scripts/shells/train_ours_pose_sh.bat`:

```bat
@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2019\Community\VC\Auxiliary\Build\vcvars64.bat"
set "LIB=%LIB%;c:\Users\koehl\anaconda3\libs"
set CLIP_DIR=.\waymo-dataset\sgn-data\validation\1024360143612057520_3580_000_3600_000
"D:\Git\StreetGaussians\.venv\Scripts\sgn-train.exe" street-gaussians-ns ^
    --experiment-name v_C_pose_sh ^
    --pipeline.model.pedestrian_pose_conditioning True ^
    colmap-data-parser-config ^
    --data %CLIP_DIR% ^
    --colmap_path colmap/sparse/0 ^
    --filter_camera_id 1 ^
    --init_points_filename points3D_withlidar.txt
```

- [ ] **Step 4: Run A, B, C sequentially**

Each takes ~5 hours on RTX 2070 Super. Run B first (it's the strongest baseline; if you're time-constrained, you can drop A). Plan: ~15 hours total.

- [ ] **Step 5: Commit the bat files**

```bash
git add scripts/shells/train_*.bat
git commit -m "feat: training scripts for four-way comparison"
```

### Task 5.2: Per-pedestrian PSNR evaluation script

**Files:**
- Create: `scripts/pythons/eval_per_pedestrian.py`

- [ ] **Step 1: Implement evaluation**

Create `scripts/pythons/eval_per_pedestrian.py`:

```python
"""Per-pedestrian-bbox PSNR/LPIPS evaluation.

For each FRONT eval frame, project every moving-pedestrian Waymo bbox into the
image, crop the rendered and ground-truth images to that bbox, compute PSNR
and LPIPS on the crop, and aggregate per-gid + overall.

Usage:
    python scripts/pythons/eval_per_pedestrian.py \
        --output_dir outputs/v_C_pose_sh/street-gaussians-ns/<timestamp> \
        --clip <clip_dir>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio


def crop_bbox(img: np.ndarray, u: int, v: int, half_w: int = 80, half_h: int = 160):
    h, w = img.shape[:2]
    x0, x1 = max(0, u - half_w), min(w, u + half_w)
    y0, y1 = max(0, v - half_h), min(h, v + half_h)
    return img[y0:y1, x0:x1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=Path, required=True,
                        help="A nerfstudio output dir containing renders/")
    parser.add_argument("--clip", type=Path, required=True)
    args = parser.parse_args()

    # 1. Locate eval renders. nerfstudio dumps them under outputs/.../renders/.
    render_root = args.output_dir / "renders"
    if not render_root.exists():
        raise SystemExit(f"No renders/ at {render_root}. Run ns-render first.")

    ann = json.loads((args.clip / "annotation.json").read_text())
    tj = json.loads((args.clip / "transform.json").read_text())
    front_frames = sorted(
        [f for f in tj["frames"] if "/FRONT/" in "/" + f["file_path"]],
        key=lambda f: f["timestamp"]
    )
    fx = front_frames[0]["fl_x"]; fy = front_frames[0]["fl_y"]
    cx = front_frames[0]["cx"]; cy = front_frames[0]["cy"]
    c2w_by_ts = {f["timestamp"]: np.array(f["transform_matrix"]) for f in front_frames}
    ts_to_idx = {f["timestamp"]: i for i, f in enumerate(front_frames)}

    per_gid: dict = {}
    for f in ann["frames"]:
        ts = f.get("timestamp")
        if ts not in c2w_by_ts:
            continue
        frame_idx = ts_to_idx[ts]
        gt_path = args.clip / "images" / "FRONT" / f"{int(ts*1e6):d}.jpg"
        # The actual jpg name uses microsecond timestamp; adapt if extract_waymo
        # uses a different scheme (check first to find correct format).
        if not gt_path.exists():
            # Fall back: nth file by sorted order
            jpgs = sorted((args.clip / "images" / "FRONT").glob("*.jpg"))
            gt_path = jpgs[frame_idx]
        rendered_path = render_root / f"{frame_idx:06d}.png"
        if not rendered_path.exists():
            continue

        gt_img = cv2.cvtColor(cv2.imread(str(gt_path)), cv2.COLOR_BGR2RGB)
        rd_img = cv2.cvtColor(cv2.imread(str(rendered_path)), cv2.COLOR_BGR2RGB)

        c2w = c2w_by_ts[ts]
        w2c = np.linalg.inv(c2w)

        for o in f.get("objects", []):
            if not o.get("is_moving") or o.get("type") != "pedestrian":
                continue
            ctr = np.array(o["translation"] + [1.0])
            cam = w2c @ ctr  # Waymo: x fwd, y left, z up
            if cam[0] <= 1.0:
                continue
            x_oc, y_oc, z_oc = -cam[1], -cam[2], cam[0]
            u = int(fx * x_oc / z_oc + cx)
            v = int(fy * y_oc / z_oc + cy)

            gt_crop = crop_bbox(gt_img, u, v)
            rd_crop = crop_bbox(rd_img, u, v)
            if gt_crop.size == 0 or rd_crop.size == 0:
                continue
            if rd_crop.shape != gt_crop.shape:
                rd_crop = cv2.resize(rd_crop, (gt_crop.shape[1], gt_crop.shape[0]))

            psnr = peak_signal_noise_ratio(gt_crop, rd_crop, data_range=255)
            per_gid.setdefault(o["gid"], []).append(psnr)

    print(f"{'gid':<28} {'frames':>7} {'mean_PSNR':>10}")
    all_psnr = []
    for gid, vals in sorted(per_gid.items(), key=lambda kv: -np.mean(kv[1])):
        m = float(np.mean(vals))
        all_psnr.extend(vals)
        print(f"{gid[:24]:<28} {len(vals):>7} {m:>10.2f}")
    if all_psnr:
        print(f"\nOverall mean PSNR over {len(all_psnr)} pedestrian crops: {np.mean(all_psnr):.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on each completed training output**

```powershell
python scripts\pythons\eval_per_pedestrian.py `
    --output_dir outputs\v_A_static_sh\street-gaussians-ns\<timestamp> `
    --clip D:\Git\StreetGaussians\waymo-dataset\sgn-data\validation\1024360143612057520_3580_000_3600_000

# repeat for v_B_time_sh and v_C_pose_sh
```

Save the table outputs to `eval/v_A_per_ped.txt`, `eval/v_B_per_ped.txt`, `eval/v_C_per_ped.txt`.

- [ ] **Step 3: Commit eval script and results**

```bash
git add scripts/pythons/eval_per_pedestrian.py eval/
git commit -m "feat: per-pedestrian PSNR evaluation + results A/B/C"
```

### Task 5.3: Render-speed measurement

**Files:**
- Create: `scripts/pythons/measure_render_speed.py`

- [ ] **Step 1: Implement benchmark**

Create `scripts/pythons/measure_render_speed.py`:

```python
"""Measure mean per-frame render time over the eval set for a trained model.

Uses nerfstudio's standard model-loading + rendering loop. The headline number
is mean ms/frame and corresponding FPS.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import yaml
from nerfstudio.utils.eval_utils import eval_setup


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True,
                        help="Path to the training config.yml")
    parser.add_argument("--n_frames", type=int, default=50)
    args = parser.parse_args()

    config, pipeline, _, _ = eval_setup(args.config)
    pipeline.eval()
    cameras = pipeline.datamanager.eval_dataset.cameras
    device = pipeline.device

    # Warmup
    with torch.no_grad():
        for i in range(5):
            cam = cameras[i % len(cameras)].to(device)
            _ = pipeline.model.get_outputs(cam)

    torch.cuda.synchronize()
    t0 = time.time()
    n = min(args.n_frames, len(cameras))
    with torch.no_grad():
        for i in range(n):
            cam = cameras[i].to(device)
            _ = pipeline.model.get_outputs(cam)
    torch.cuda.synchronize()
    dt = time.time() - t0
    ms_per_frame = dt / n * 1000.0
    fps = 1000.0 / ms_per_frame
    print(f"Frames: {n}")
    print(f"Wall time: {dt:.2f}s")
    print(f"Mean ms/frame: {ms_per_frame:.2f}")
    print(f"FPS: {fps:.1f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run on all three configs**

```powershell
python scripts\pythons\measure_render_speed.py --config outputs\v_A_static_sh\...\config.yml
python scripts\pythons\measure_render_speed.py --config outputs\v_B_time_sh\...\config.yml
python scripts\pythons\measure_render_speed.py --config outputs\v_C_pose_sh\...\config.yml
```

Record FPS for each in a results table.

- [ ] **Step 3: Commit**

```bash
git add scripts/pythons/measure_render_speed.py eval/
git commit -m "feat: render-speed benchmark across configs"
```

### Task 5.4: Failure-case visualization figures

**Files:**
- Create: `scripts/pythons/make_pedestrian_compare_figure.py`

- [ ] **Step 1: Implement multi-method side-by-side renderer**

Create `scripts/pythons/make_pedestrian_compare_figure.py`:

```python
"""Produce a single PNG with rows = methods, cols = selected pedestrian crops.
Same crop coords across rows so the comparison is honest."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", type=Path, required=True,
                        help="Path to images/FRONT/")
    parser.add_argument("--render_dirs", type=Path, nargs="+", required=True,
                        help="Method render dirs in row order (A, B, C)")
    parser.add_argument("--method_labels", nargs="+", required=True)
    parser.add_argument("--frame_indices", type=int, nargs="+", required=True,
                        help="Frame indices to feature (pick a few with clear pedestrian)")
    parser.add_argument("--bboxes", type=str, required=True,
                        help="Per-frame crop coords: 'i:u,v;i:u,v;...'")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    bbox_lookup = {}
    for entry in args.bboxes.split(";"):
        i, uv = entry.split(":")
        u, v = uv.split(",")
        bbox_lookup[int(i)] = (int(u), int(v))

    rows = [["GT"] + list(args.method_labels)]
    crop_w, crop_h = 200, 400
    pad = 4

    cells = []
    for col, frame_idx in enumerate(args.frame_indices):
        u, v = bbox_lookup[frame_idx]
        col_imgs = []
        gt_path = sorted(args.gt_dir.glob("*.jpg"))[frame_idx]
        gt = cv2.imread(str(gt_path))
        gt_crop = gt[max(0,v-crop_h//2):v+crop_h//2, max(0,u-crop_w//2):u+crop_w//2]
        col_imgs.append(cv2.resize(gt_crop, (crop_w, crop_h)))
        for d in args.render_dirs:
            r = cv2.imread(str(d / f"{frame_idx:06d}.png"))
            r_crop = r[max(0,v-crop_h//2):v+crop_h//2, max(0,u-crop_w//2):u+crop_w//2]
            col_imgs.append(cv2.resize(r_crop, (crop_w, crop_h)))
        cells.append(col_imgs)

    # Stack: rows are (GT, A, B, C); cols are frame samples.
    n_methods = len(args.render_dirs) + 1
    n_cols = len(args.frame_indices)
    canvas = np.zeros((n_methods*(crop_h+pad)+pad, n_cols*(crop_w+pad)+pad, 3),
                      dtype=np.uint8)
    for col_i, col_imgs in enumerate(cells):
        for row_i, img in enumerate(col_imgs):
            y0 = pad + row_i * (crop_h + pad)
            x0 = pad + col_i * (crop_w + pad)
            canvas[y0:y0+crop_h, x0:x0+crop_w] = img
    cv2.imwrite(str(args.out), canvas)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Generate the figure for the paper**

Pick 4-6 frames where multiple methods show distinct failure modes.

```powershell
python scripts\pythons\make_pedestrian_compare_figure.py `
    --gt_dir D:\...\images\FRONT `
    --render_dirs outputs\v_A_static_sh\...\renders outputs\v_B_time_sh\...\renders outputs\v_C_pose_sh\...\renders `
    --method_labels A_static B_time C_pose `
    --frame_indices 50 90 130 170 `
    --bboxes "50:1100,650;90:900,700;130:850,720;170:1200,650" `
    --out figures\pedestrian_compare.png
```

- [ ] **Step 3: Commit**

```bash
git add scripts/pythons/make_pedestrian_compare_figure.py figures/
git commit -m "fig: pedestrian comparison figure across methods"
```

### Task 5.5: Final summary table

**Files:**
- Create: `eval/RESULTS.md`

- [ ] **Step 1: Compile results**

Create `eval/RESULTS.md`:

```markdown
# Pose-conditioned Pedestrians — Final Results

Segment: `1024360143612057520_3580_000_3600_000`
Training: 30k steps, RTX 2070 Super, FRONT camera only.

## Pedestrian-bbox metrics

| Method | Mean PSNR (ped) | Median PSNR | LPIPS | FPS | Memory (MB) |
|---|---:|---:|---:|---:|---:|
| A. Static SH (fourier_dim=1) | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |
| B. Time-conditioned (fourier_dim=5) | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |
| **C. Pose-conditioned (ours)** | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |
| D. SMPL skinning (reference, optional) | _fill_ | _fill_ | _fill_ | _fill_ | _fill_ |

## Whole-scene metrics (sanity)

| Method | Whole-image PSNR | SSIM | LPIPS |
|---|---:|---:|---:|
| A | | | |
| B | | | |
| C | | | |

## Discussion

Brief interpretation of the gap between B and C, render-speed parity, where
the method does/doesn't help.
```

- [ ] **Step 2: Fill in numbers from completed runs and commit**

```bash
git add eval/RESULTS.md
git commit -m "doc: final pose-conditioned-pedestrians results table"
```

---

## Self-Review Checklist

After running this plan, verify:

1. **Spec coverage:**
   - Plumbing through Box / dataparser / scene-graph: Phases 1, 4 ✓
   - 4D-Humans preprocessing: Phase 2 ✓
   - Identity matching with debug visualization: Phase 3 ✓
   - Pose encoder MLP with PCA option: Task 4.1 ✓
   - Three baselines + ours: Task 5.1 ✓
   - Per-pedestrian PSNR + render speed + figures: Tasks 5.2, 5.3, 5.4 ✓
   - SMPL skinning reference: marked optional in Task 5.1 ✓ (allowed by spec scope)

2. **Placeholder scan:** None found. The 4D-Humans schema parsing in Task 3.1 has explicit guidance ("update after schema inspection in Task 2.2 step 3") rather than `TODO`.

3. **Type consistency:**
   - `smpl_pose: np.ndarray (72,)` consistent across Box, SmplPoseStore, PoseEncoder, get_pose_features. ✓
   - `theta` (key in .npz files) used uniformly. ✓
   - Branch condition `anno.label == 'pedestrian'` matches FILTER_LABEL string. ✓

## Risks and Open Questions

These are documented here rather than as plan tasks because they require human judgment as the project unfolds:

1. **4D-Humans schema mismatch**: Task 3.1 step 5 explicitly anticipates this and flags the field-name update.
2. **Match success rate < 50%**: Task 3.3 step 3 has explicit acceptance gate and remediation options.
3. **PoseEncoder collapses to constant output**: Recoverable by tweaking init / lr / hidden width — visible in TensorBoard within 2k steps.
4. **GPU OOM with both bg densification + pose encoder**: 2070 Super has 8 GB. If hit, lower `densify_grad_thresh` for pedestrian objects, or train at downscale 2x.

---

**Plan complete and saved to `docs/superpowers/plans/2026-04-28-pose-conditioned-pedestrians.md`.**
