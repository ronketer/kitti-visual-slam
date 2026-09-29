from reports.paths import GT_POSES_FILE
import sys
import os
from dataclasses import dataclass

import numpy as np
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm

from kitti_slam.dataset import read_cameras
from kitti_slam.dataset import read_images
from kitti_slam.geometry import find_camera_location
from kitti_slam.geometry import compose_extrinsics
from kitti_slam.stereo import find_stereo_temporal_matches
from kitti_slam.detector_config import *
from .q1 import q1
from .q2 import q2
from .q5 import q5
from .performance_logger import TimeLogger
from reports.paths import OUTPUT_RELATIVE_PATH, LASTFRAME

FIRSTFRAME = 0
CMAP = "tab20"

@dataclass
class FrameState:
    """
    Represents the state of a camera frame, storing both relative and absolute transformations.
    Relative transformations are with respect to the previous frame.
    Absolute transformations are with respect to the world coordinate system.
    """
    # All fields are initialized to None by default
    relative_extrinsics: object = None
    absolute_extrinsics: object = None
    relative_camera_location: object = None
    absolute_camera_location: object = None
    ground_truth_location: object = None

    @classmethod
    def create_initial_frame(cls, initial_extrinsics, initial_location):
        """Create the initial frame state with absolute positions only"""
        frame = cls()
        frame.absolute_extrinsics = initial_extrinsics
        frame.absolute_camera_location = initial_location
        return frame

    def update_from_relative_motion(self, relative_extrinsics, relative_location,
                                  absolute_extrinsics, absolute_location):
        """Update frame state with new motion information"""
        self.relative_extrinsics = relative_extrinsics
        self.relative_camera_location = relative_location
        self.absolute_extrinsics = absolute_extrinsics
        self.absolute_camera_location = absolute_location

def visualize(frame_states, frame_inliers_matches_count, frame_common_matches_count, frame_supporters_count):
    """
    Visualizes the camera trajectory comparing estimated and ground truth paths.

    Args:
        frame_states: List of FrameState objects containing camera positions
        frame_inliers_matches_count: Number of inlier matches per frame (unused)
        frame_common_matches_count: Number of common matches per frame (unused)
        frame_supporters_count: Number of supporting matches per frame (unused)
    """
    all_camera_location = [state.absolute_camera_location for state in frame_states]
    ground_truth_locations = [state.ground_truth_location for state in frame_states]

    all_camera_location = np.array(all_camera_location)
    ground_truth_locations = np.array(ground_truth_locations)

    X = all_camera_location[:, 0]
    Z = all_camera_location[:, 2]
    X_gt = ground_truth_locations[:, 0]
    Z_gt = ground_truth_locations[:, 2]

    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot both trajectories
    ax.plot(X, Z, 'b-', label='Estimated Trajectory')
    ax.plot(X_gt, Z_gt, 'r-', label='Ground Truth')

    ax.set_title("Camera Trajectory: Estimated vs Ground Truth")
    ax.set_xlabel("X (Meters)")
    ax.set_ylabel("Z (Meters)")
    ax.axis("equal")
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.savefig(OUTPUT_RELATIVE_PATH + "camera_analysis.png")

def process_image_pair(detector, camera_left, camera_right, img_left, img_right, logger):
    """Process a stereo image pair to extract features and compute matches."""
    logger.start_operation("Feature Detection")
    kp_left, des_left = detector.detectAndCompute(img_left, None)
    kp_right, des_right = detector.detectAndCompute(img_right, None)
    logger.end_operation("Feature Detection")

    logger.start_operation("Stereo Matching")
    inliers_matches, cloud = q1(camera_left, camera_right, kp_left, des_left, kp_right, des_right)
    logger.end_operation("Stereo Matching")

    return kp_left, des_left, kp_right, des_right, inliers_matches, cloud

def q6():
    logger = TimeLogger()

    # Initialize cameras and feature detector
    K, M1, M2 = read_cameras()
    camera_left, camera_right = K @ M1, K @ M2

    # Initialize first frame
    first_frame = FrameState.create_initial_frame(M1, find_camera_location(M1))
    frame_states = [first_frame]

    # Read ground truth poses
    def parse_gt_line(line):
        nums = np.array([float(x) for x in line.strip().split()]).reshape(3, 4)
        return find_camera_location(nums)

    # Read and parse ground truth poses
    gt_poses = []
    with open(GT_POSES_FILE, 'r') as f:
        gt_poses = [parse_gt_line(line) for line in f.readlines()[:LASTFRAME+1]]

    # Set ground truth for initial frame
    first_frame.ground_truth_location = gt_poses[FIRSTFRAME]

    # Initialize tracking variables
    prev_kp_left, prev_des_left, prev_kp_right, prev_des_right = None, None, None, None
    prev_inliers_matches, prev_cloud = None, None

    # Count stats for each frame
    frame_inliers_matches_count = []
    frame_common_matches_count = []
    frame_supporters_count = []
    for i in tqdm(range(FIRSTFRAME, LASTFRAME)):
        logger.start_operation(f"Iteration {i}")

        # Process first image pair (either new or from previous iteration)
        if prev_kp_left is None:
            logger.start_operation("Image Reading")
            img_left0, img_right0 = read_images(i)
            logger.end_operation("Image Reading")

            kp_left0, des_left0, kp_right0, des_right0, inliers_matches0, cloud0 = process_image_pair(
                detector, camera_left, camera_right, img_left0, img_right0, logger
            )
        else:
            kp_left0, des_left0 = prev_kp_left, prev_des_left
            kp_right0, des_right0 = prev_kp_right, prev_des_right
            inliers_matches0, cloud0 = prev_inliers_matches, prev_cloud

        frame_inliers_matches_count.append(len(inliers_matches0))

        # Process second image pair
        logger.start_operation("Image Reading")
        img_left1, img_right1 = read_images(i + 1)
        logger.end_operation("Image Reading")

        kp_left1, des_left1, kp_right1, des_right1, inliers_matches1, cloud1 = process_image_pair(
            detector, camera_left, camera_right, img_left1, img_right1, logger
        )

        # Store current features for next iteration
        prev_kp_left, prev_des_left = kp_left1, des_left1
        prev_kp_right, prev_des_right = kp_right1, des_right1
        prev_inliers_matches, prev_cloud = inliers_matches1, cloud1

        # Find matches between consecutive frames
        logger.start_operation("Temporal Matching")
        matches_between_img0_img1 = q2(des_left0, des_left1)
        common_matches, common_matches_indices = find_stereo_temporal_matches(
            inliers_matches0, inliers_matches1, matches_between_img0_img1
        )
        logger.end_operation("Temporal Matching")
        frame_common_matches_count.append(len(common_matches_indices))

        # Estimate motion between frames
        logger.start_operation("Motion Estimation")
        tvec0 = M2[:,3].reshape(3,1)
        relative_extrinsics_l1, _, supporters = q5(
            common_matches_indices, common_matches, cloud0, cloud1,
            inliers_matches0, kp_left0, kp_right0, kp_left1, kp_right1,
            K, M1, M2, tvec0, logger
        )
        logger.end_operation("Motion Estimation")
        frame_supporters_count.append(len(supporters))

        # Update frame state
        relative_camera_location_l1 = find_camera_location(relative_extrinsics_l1)
        absolute_extrinsics_l1 = compose_extrinsics(relative_extrinsics_l1, frame_states[-1].absolute_extrinsics)
        absolute_camera_location_l1 = find_camera_location(absolute_extrinsics_l1)

        new_frame = FrameState()
        new_frame.update_from_relative_motion(
            relative_extrinsics_l1, relative_camera_location_l1,
            absolute_extrinsics_l1, absolute_camera_location_l1
        )
        # Set ground truth for current frame
        new_frame.ground_truth_location = gt_poses[i + 1]
        frame_states.append(new_frame)

        logger.end_operation(f"Iteration {i}")

    visualize(frame_states, frame_inliers_matches_count, frame_common_matches_count, frame_supporters_count)
    logger.get_report()
    # print the distance between the last estimated and ground truth camera locations
    estimated_location = frame_states[-1].absolute_camera_location
    gt_location = frame_states[-1].ground_truth_location
    distance = np.linalg.norm(estimated_location - gt_location)
    print(f"Distance between estimated and ground truth camera locations: {distance:.2f} meters")
