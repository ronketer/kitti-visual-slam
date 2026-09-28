# Stereo Visual SLAM — KITTI Odometry Sequence 05

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![GTSAM](https://img.shields.io/badge/GTSAM-4.x-orange)
![NumPy](https://img.shields.io/badge/NumPy-1.x-013243?logo=numpy&logoColor=white)

Full stereo visual odometry pipeline built from scratch — feature tracking, RANSAC-PnP motion estimation, windowed bundle adjustment, and pose graph loop closure — evaluated against KITTI ground truth over 2,600 frames.

![Trajectory comparison: Ground Truth vs PnP vs Bundle Adjustment vs Pose Graph](results/trajectory_comparison.png)
*Camera trajectory comparison on KITTI Sequence 05 (X-Z plane). Each stage reduces drift.*

## Results

| Stage | Translation Error / 100 m |
|---|---|
| PnP (RANSAC) | 3.2 m |
| + Windowed Bundle Adjustment | 1.1 m |
| + Pose Graph + Loop Closure | **0.9 m** |

**3.5× error reduction** end-to-end. Metrics computed over all 230 keyframe windows across the full 2,600-frame sequence.

| | |
|---|---|
| ![Relative translation errors across pipeline stages](results/relative_errors_subsequence_400.png) | ![Feature track length distribution](results/track_length_histogram.png) |
| *Relative translation error for 400-frame subsequences — improvement visible at each stage* | *Track length histogram — most features tracked across 2–5 frames; long-lived tracks anchor BA windows* |

## Pipeline

1. **AKAZE stereo matching** — detect ~1,200 keypoints/frame, filter by epipolar constraint (y-deviation < 2 px), triangulate to 8,000–12,000 3D points
2. **RANSAC-PnP motion estimation** — 4-point PnP with adaptive iteration count (99.9% confidence), refined on all inliers
3. **Windowed bundle adjustment** — GTSAM `GenericStereoFactor3D` over 5–20-frame windows selected by feature count, track overlap, and translation criteria
4. **Pose graph + loop closure** — relative pose constraints with covariance extracted from BA marginals, GTSAM ISAM2 optimization with loop detection

## Project Structure

| Path | Contents |
|---|---|
| `code/ex1.py` – `ex5.py` | Sequential exercises building up the pipeline |
| `code/pose_graph_loop_closure.py` | Stage 4: pose graph optimization |
| `code/stereo.py` | Stereo filtering, OpenCV and handwritten SVD triangulation, four-view correspondences |
| `code/motion.py` | Custom adaptive RANSAC, P3P hypothesis generation, iterative PnP refinement |
| `code/supporters.py` | Four-view reprojection validation and supporter classification |
| `code/tracking.py` | Canonical frame loop: stereo processing, motion estimation, track updates, and pose accumulation |
| `code/alg.py` | BA construction and compatibility exports for extracted frontend/tracking functions |
| `code/tracking_database.py` | `TrackingDB` — central data structure mapping frames ↔ tracks |
| `code/window_selector.py` | Pluggable keyframe selection criteria for BA windows |
| `code/final/` | Analysis scripts generating the plots in `results/` |
| `results/` | Key output plots (committed) |
| `dataset/` | KITTI Sequence 05 — not included, see Setup |

## Setup

```bash
git clone https://github.com/ronketer/kitti-visual-slam
cd kitti-visual-slam
pip install -r requirements.txt
```

**Dataset**: Download [KITTI Odometry Sequence 05](https://www.cvlibs.net/datasets/kitti/eval_odometry.php) (grayscale images + ground truth poses) and place it at `dataset/sequences/05/` and `dataset/poses/05.txt`.

```bash
# Run the full pipeline (auto-resumes from checkpoints)
python run_pipeline.py

# Re-run from scratch, ignoring cached results
python run_pipeline.py --force
```

Individual exercises can also be run directly:

```bash
python code/ex1.py   # feature detection & matching
python code/ex5.py   # bundle adjustment
```

All scripts must be run from the **repo root**.

### Native Windows and incremental refactoring

The geometry, dataset readers, stereo frontend, motion estimation, tracking
database, and window selection can be imported and tested without GTSAM.
GTSAM is still required for stereo backprojection through GTSAM, Bundle
Adjustment, pose graph optimization, and the full pipeline.

The checked Windows environment uses CPython 3.13 x64. A wheel-only dependency
check found no compatible stable GTSAM package. The
[official installation guide](https://gtsam.org/get_started/) states that GTSAM
4.3.0 has no Windows wheels. BA and pose graph validation are deferred to a
suitable environment; their algorithms are unchanged.

Run the small, dataset-independent tests with:

```bash
python -B -m unittest discover -s tests -v
```

`code/config.py` defines repository-relative default locations, independent of
the working directory. The pipeline runner uses these defaults. Historical
exercise/report scripts may still contain working-directory-relative paths.
`ProjectPaths` and the tracking function also allow explicit inputs, for example
with `code/` on the Python import path:

```python
from pathlib import Path
from config import ProjectPaths
from detector_config import create_detector_and_matcher
from tracking import build_tracking_database

paths = ProjectPaths(dataset_root=Path("D:/KITTI"), sequence="05")
detector, matcher = create_detector_and_matcher("AKAZE")
db = build_tracking_database(paths=paths, detector=detector, matcher=matcher,
                             last_frame=20)  # inclusive frames 0 through 20
```

An already-loaded `(K, M1, M2)` tuple can be passed as `calibration=`. Existing
zero-argument calls retain sequence 05, frames 0–2599, and AKAZE defaults.
Nonzero starting frames are explicitly rejected because the current tracking
database assumes zero-based frame IDs. The reader functions remain available
through `utility.py` for existing callers.

The extracted `stereo`, `motion`, and `supporters` modules also import without
Matplotlib. Existing imports from `alg.py`, `utility.py`, and the old motion
module filenames remain supported through re-exports. Identical exercise
helpers share these implementations; the historical stereo policy and timed
exercise RANSAC loop remain distinct. This extraction preserves the original
algorithm bodies, including existing thresholds, refinement behavior, and
failure cases. The tests include a seeded synthetic RANSAC/PnP case with
outliers in the right-camera observations.

`tracking.build_tracking_database()` is shared by the pipeline and tracking
analysis. `collect_diagnostics=True` records temporal-match, four-view-match,
and RANSAC-supporter counts in the existing database fields. It defaults to
`False`, preserving the original pipeline's zero counts. The report's
`updated_create_tracking_db()` wrapper enables diagnostics, while
`alg.create_tracking_db` remains a compatibility alias. `TrackingDB`, `Link`,
and their pickle format/import paths are unchanged. Synthetic three-frame
tests cover track creation, continuation, competing matches, pose composition,
and checkpoint compatibility; existing LFS pointer files have not been loaded
as numerical regression baselines.

## Report

Full methodology, quantitative analysis, and comparison plots: [slam_final_submission.pdf](slam_final_submission.pdf)
