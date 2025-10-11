# SLAM-VN — Stereo Visual SLAM on KITTI (Seq 05)

A hands-on stereo visual SLAM pipeline built on the KITTI Odometry dataset (sequence 05). The project walks through feature detection and matching, stereo triangulation, multi-view geometry, track database and statistics, and windowed bundle adjustment/pose-graph optimization using GTSAM.

Co-developed with Yonatahan Hirchel as part of the SLAM-NAVIGATION course at The Hebrew University.

## What’s inside

The pipeline is organized as a set of exercises/stages under `code/`:

- `ex1.py` — Keypoint detection (ORB/AKAZE/SIFT), stereo feature matching, Lowe’s ratio filtering, and visualizations.
- `ex2.py` — Stereo consistency check (epipolar deviation), inlier/outlier classification, and triangulation using OpenCV and Linear Least Squares.
- `ex3.py` — Stereo-temporal matching, RANSAC supporters, camera location estimation, and point-cloud alignment across frames.
- `ex4.py` — Tracking database creation and analysis: track length stats, connectivity, inlier percentages, and reprojection error vs. track length. Saves the tracking DB as a `.pkl` artifact.
- `ex5.py` — Factor-graph formulation with GTSAM for windowed bundle adjustment and trajectory comparison (initial vs. optimized vs. ground truth). Exports results as pickles and plots.
- `final/` — Analysis and plotting scripts (e.g., detector comparisons, optimization error curves, trajectories) that consume artifacts produced by the stages above.

Supporting modules include `alg.py`, `utility.py`, `tracking_database.py`, `window_selector.py`, `find_ransac_iteration_supporters.py`, `perform_motion_estimation.py`, and more.

Outputs (plots and pickles) are written to `code/output/` and `code/final/plots/`.

## Repository layout

- `code/` — Core pipeline and analysis scripts
	- `final/` — Plotting/analysis entry points
	- `output/` — Generated plots and artifacts (e.g., `.pkl` and `.png` files)
- `dataset/` — Expected KITTI data layout
	- `sequences/05/` — Images and calibration (`image_0/`, `image_1/`, `calib.txt`)
	- `poses/05.txt` — Ground-truth poses
- `requirement.txt` — Python dependencies

Note: `.gitignore` is configured to ignore Python caches, macOS metadata, and `*.pkl` artifacts.

## Requirements

- Python 3.9+ (3.10/3.11 recommended)
- Packages (installed via `requirement.txt`):
	- numpy, opencv-python, matplotlib, tqdm, gtsam

If you switch the feature detector to SIFT, you may need `opencv-contrib-python` instead of `opencv-python`.

## Setup

On Windows PowerShell:

```powershell
# 1) (Optional) Create a virtual environment
python -m venv .venv
./.venv/Scripts/Activate.ps1

# 2) Install dependencies
pip install -r requirement.txt
```

### Prepare KITTI data (sequence 05)

The code assumes the following paths (see `code/consts.py`):

- `./dataset/sequences/05/image_0/` — Left grayscale images (000000.png …)
- `./dataset/sequences/05/image_1/` — Right grayscale images
- `./dataset/sequences/05/calib.txt` — Calibration file
- `./dataset/poses/05.txt` — Ground-truth poses

Steps:

1) Download KITTI Odometry “sequences/05” images and `calib.txt` and place them under `dataset/sequences/05/`.
2) Download `poses/05.txt` (ground truth) and place it under `dataset/poses/`.

You can reduce runtime by editing `FIRSTFRAME`/`LASTFRAME` in `code/consts.py`.

## How to run

Each stage is a separate script. Outputs are saved into `code/output/`.

```powershell
# Feature detection and stereo matching
python code/ex1.py

# Stereo consistency + triangulation
python code/ex2.py

# Stereo-temporal matches, supporters, camera locations, cloud alignment
python code/ex3.py

# Tracking DB build + statistics + reprojection analysis
python code/ex4.py

# Windowed BA / factor-graph optimization with GTSAM
python code/ex5.py
```

Time note: `ex4.py` and especially `ex5.py` can take several minutes depending on your machine and the `LASTFRAME` value.

## Configuration

- `code/consts.py`
	- Dataset and output paths (`DATASET_RELATIVE_PATH`, `OUTPUT_RELATIVE_PATH`)
	- Frame range (`FIRSTFRAME`, `LASTFRAME`)
- `code/detector_config.py`
	- Choose a detector by setting `DETECTOR = "AKAZE" | "ORB" | "SIFT"`
	- Default is `AKAZE`. If you use `SIFT`, make sure OpenCV supports it (you may need `opencv-contrib-python`).

## Artifacts and outputs

Generated files live under `code/output/`, including:

- Plots: `*.png` (e.g., keypoints/matches, histograms, 3D clouds, trajectories)
- Pickles: `*.pkl` (ignored by Git)
	- Tracking DB: `tracking_with_geometric_validation_without_far_tracks.pkl`
	- Optimization results: `ba_results.pkl`, `loop_closure_results.pkl` (when produced)

Analysis/plotting scripts in `code/final/` expect these artifacts to exist.

## Analysis and plotting (optional)

After running the stages that generate `ba_results.pkl` and/or `loop_closure_results.pkl`, you can produce summarizing plots:

```powershell
python code/final/plot_trajectories.py
python code/final/plot_absolute_errors.py
python code/final/plot_relative_errors.py
python code/final/plot_median_projection_errors.py
python code/final/plot_optimization_errors.py
python code/final/plot_projection_vs_distance.py
```

Detector or PnP/RANSAC analyses are available via:

```powershell
python code/final/detector_analysis.py
python code/final/ransac_pnp_analysis.py
```

Outputs are written to `code/final/plots/`.

## Tips & troubleshooting

- GTSAM install: If `pip install gtsam` fails on your platform, consider using a Conda environment with `conda install -c conda-forge gtsam` or consult GTSAM’s wheels for your Python version.
- OpenCV SIFT errors: If you switch `DETECTOR = "SIFT"` and see errors, install `opencv-contrib-python` and ensure versions match.
- Missing files: Ensure the KITTI folder structure exactly matches what `code/consts.py` expects.
- Slow runs: Lower `LASTFRAME` in `code/consts.py` to process fewer frames while testing.

## Acknowledgments

- KITTI Odometry dataset: http://www.cvlibs.net/datasets/kitti/
- GTSAM: https://gtsam.org/
- Course project — SLAM-NAVIGATION @ The Hebrew University
