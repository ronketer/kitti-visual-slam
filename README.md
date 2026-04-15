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
| `code/alg.py` | Core algorithms (triangulation, tracking DB construction, BA) |
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

## Report

Full methodology, quantitative analysis, and comparison plots: [slam_final_submission.pdf](slam_final_submission.pdf)
