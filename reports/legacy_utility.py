import os
import matplotlib.pyplot as plt
import numpy as np
from kitti_slam.geometry import compose_extrinsics, find_camera_location, coordinate_transform
import cv2
from kitti_slam.gtsam_geometry import read_calibration
from kitti_slam.dataset import read_images, read_cameras, parse_gt_line_matrix
from kitti_slam.stereo import find_stereo_temporal_matches
from kitti_slam.motion import rodriguez_to_mat

from mpl_toolkits.mplot3d import Axes3D
import time
from reports.paths import (
    LEFT_IMG_DIR,
    RIGHT_IMG_DIR,
    CALIB_FILE,
    GT_POSES_FILE,
    LASTFRAME,
    DATA_PATH,
)


from kitti_slam.geometry import compute_camera_to_camera_transform, find_transformation
from kitti_slam.evaluation.ground_truth import parse_gt_line, read_ground_truth_poses
from reports.plots import plot_and_save, crop_patch, save_figure, setup_plot, plot_keypoints, plot_matches_and_supporters, plot_feature_counts, plot_filtering_efficiency, plot_connectivity, plot_track_length_histogram, plot_average_inliers, plot_spread_performance, plot_processing_time, plot_detector_repeatability
