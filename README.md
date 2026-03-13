# SLAM-VN — Stereo Visual SLAM on KITTI

A robust stereo visual SLAM pipeline built to process high-volume spatial data from the **KITTI Odometry dataset** (Sequence 05). This project demonstrates expertise in computer vision, multi-view geometry, and factor-graph optimization.

## 🚀 System Features

* **Feature Tracking:** Analyzed 2,600+ frames** using AKAZE/ORB/SIFT detectors with Lowe’s ratio filtering and epipolar consistency checks.
* **Robust Motion Estimation:** Implemented a temporal-stereo matching pipeline via **RANSAC** to identify inlier supporters for accurate camera pose estimation.
* **Factor-Graph Optimization:** Designed windowed **Bundle Adjustment** with **GTSAM**, optimizing trajectory parameters to reduce relative translation error by **3x**.
* **Spatial Database:** Designed a track database to analyze track length statistics, connectivity, and reprojection errors across frames.

## 🛠️ Requirements
* **Python 3.9+**
* **Core Libraries:** OpenCV, NumPy, Matplotlib, GTSAM, TQDM.

## ▶️ Execution Flow
1. **Feature Detection:** `python code/ex1.py`
2. **Triangulation:** `python code/ex2.py`
3. **Motion Estimation:** `python code/ex3.py`
4. **Trajectory Optimization:** `python code/ex5.py`

## 📊 Results
The system generates comprehensive trajectory comparisons (Initial vs. Optimized vs. Ground Truth) and absolute/relative error plots stored in `code/final/plots/`.
