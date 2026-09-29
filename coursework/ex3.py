"""Exercise 3: stereo correspondences through timed sequence visual odometry.

The exercise stereo path intentionally lacks the runtime disparity filter.
The custom timed supporter/RANSAC loops retain their original numerical policy.
Run with ``python -m coursework.ex3``; this performs the two-frame demonstration
and then the historical sequence experiment.
"""

from dataclasses import dataclass
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from kitti_slam import detector_config
from kitti_slam.detector_config import detector
from kitti_slam.dataset import read_cameras, read_images
from kitti_slam.geometry import find_camera_location, coordinate_transform, compose_extrinsics
from kitti_slam.stereo import classify_matches_by_deviation, find_stereo_temporal_matches
# Exercise 3.3 uses the canonical four-point P3P hypothesis implementation.
from kitti_slam.motion import solve_pnp_and_locations, refine_pose
from kitti_slam.supporters import validate_projections_batch
from reports.paths import COURSEWORK_OUTPUT_DIR, GT_POSES_FILE, LAST_FRAME

ORANGE = (0, 165, 255)
CYAN = (255, 255, 0)
FIRST_FRAME = 0
START_OUTLIER_RATIO = 0.99
SUCCESS_PROBABILITY = 0.999
SAMPLE_SIZE = 4
iteration_calculator = lambda outlier_ratio: int(
    np.ceil(np.log(1 - SUCCESS_PROBABILITY) / np.log(1 - ((1 - outlier_ratio) ** SAMPLE_SIZE))))

# Exercise 3.1 — stereo preparation

def prepare_stereo_matches(camera1, camera2, kp_left, des_left, kp_right, des_right):
    matches = detector_config.matcher.match(des_left, des_right)
    inliers_matches, _ = classify_matches_by_deviation(matches, kp_left, kp_right)
    pts_left = np.float32([kp_left[m.queryIdx].pt for m in inliers_matches]).T
    pts_right = np.float32([kp_right[m.trainIdx].pt for m in inliers_matches]).T
    cloud4D = cv2.triangulatePoints(camera1, camera2, pts_left, pts_right)
    return inliers_matches, (cloud4D[:3] / cloud4D[3]).T

# Exercise 3.2 — temporal descriptor matching

def match_temporal_descriptors(des0, des1):
    matches = detector_config.matcher.match(des0, des1)
    return matches

# Exercise 3.4 — timed four-view supporter classification

def classify_timed_supporters(common_matches_indices, cloud0, kp_left0, kp_right0, kp_left1, kp_right1, K,
       extrinsic_l1, extrinsic_r1, extrinsic_l0, extrinsic_r0, inliers0, logger=None):
    if logger:
        logger.start_operation("Motion Estimation/Inlier Check/Setup")

    # Convert common_matches_indices to numpy array for faster indexing
    common_matches_indices = np.array(common_matches_indices)

    # Create mapping and get cloud points in one step
    queryidx_to_cloud_idx = {m.queryIdx: i for i, m in enumerate(inliers0)}
    points3D = cloud0[[queryidx_to_cloud_idx[idx_l0] for idx_l0 in common_matches_indices[:, 0]]]

    # Get all keypoints in one step using numpy array indexing
    kp_l0 = [kp_left0[idx] for idx in common_matches_indices[:, 0]]
    kp_r0 = [kp_right0[idx] for idx in common_matches_indices[:, 1]]
    kp_l1 = [kp_left1[idx] for idx in common_matches_indices[:, 2]]
    kp_r1 = [kp_right1[idx] for idx in common_matches_indices[:, 3]]

    if logger:
        logger.end_operation("Motion Estimation/Inlier Check/Setup")

    if logger:
        logger.start_operation("Motion Estimation/Inlier Check/Validation")

    # Validate all projections in batch
    valid_l0 = validate_projections_batch(points3D, extrinsic_l0, K, kp_l0)
    valid_r0 = validate_projections_batch(points3D, extrinsic_r0, K, kp_r0)
    valid_l1 = validate_projections_batch(points3D, extrinsic_l1, K, kp_l1)
    valid_r1 = validate_projections_batch(points3D, extrinsic_r1, K, kp_r1)

    # Find points that are valid in all views using vectorized operations
    all_valid = valid_l0 & valid_r0 & valid_l1 & valid_r1

    # Extract the valid matches
    supporters = common_matches_indices[all_valid].tolist()

    if logger:
        logger.end_operation("Motion Estimation/Inlier Check/Validation")

    return supporters

# Exercise 3.5 — custom adaptive RANSAC and canonical PnP refinement

def run_timed_ransac(
    common_matches_indices, common_matches, queryidx_to_cloud_idx, cloud0, kp_left1, K, tvec0,
    kp_left0, kp_right0, kp_right1, extrinsic_l0, extrinsic_r0, inliers0, logger
):
    best_num_inliers = 0
    best_inliers = []
    best_extrinsic_l1 = None
    best_extrinsic_r1 = None


    max_iters = iteration_calculator(START_OUTLIER_RATIO)

    i = 0
    while i < max_iters:
        if logger:
            logger.start_operation("Motion Estimation/RANSAC Sample")

        extrinsic_matrix_l1, extrinsic_matrix_r1, location_l1, location_r1 = solve_pnp_and_locations(common_matches, cloud0, inliers0, kp_left1, K, tvec0)

        current_inliers = classify_timed_supporters(common_matches_indices, cloud0, kp_left0, kp_right0, kp_left1, kp_right1,
                            K, extrinsic_matrix_l1, extrinsic_matrix_r1, extrinsic_l0, extrinsic_r0, inliers0, logger)
        if len(current_inliers) > best_num_inliers:
            best_num_inliers = len(current_inliers)
            best_inliers = current_inliers
            best_extrinsic_l1 = extrinsic_matrix_l1.copy()
            best_extrinsic_r1 = extrinsic_matrix_r1.copy()
            outlier_ratio = 1 - (len(current_inliers) / len(common_matches_indices))
            max_iters = min(iteration_calculator(outlier_ratio), max_iters)
        if logger:
            logger.end_operation("Motion Estimation/RANSAC Sample")
        i+=1
    return best_inliers, best_extrinsic_l1, best_extrinsic_r1

def estimate_timed_motion(common_matches_indices, common_matches, cloud0, cloud1, inliers0, kp_left0, kp_right0, kp_left1, kp_right1, K, extrinsic_l0, extrinsic_r0, tvec0, logger=None):
    if logger:
        logger.start_operation("Motion Estimation/Setup")

    # ransac_iters = MAXITERATIONS
    queryidx_to_cloud_idx = {m.queryIdx: i for i, m in enumerate(inliers0)}
    if logger:
        logger.end_operation("Motion Estimation/Setup")

    if len(common_matches_indices) < SAMPLE_SIZE:
        print(f"Only {len(common_matches_indices)} matches - too few for RANSAC")
        return None, None, []  # Return empty results



    # --- 2. Perform RANSAC Loop ---
    best_inliers, best_extrinsic_l1, best_extrinsic_r1 = run_timed_ransac(
        common_matches_indices, common_matches, queryidx_to_cloud_idx, cloud0, kp_left1, K, tvec0,
        kp_left0, kp_right0, kp_right1, extrinsic_l0, extrinsic_r0, inliers0, logger
    )
    refined_extrinsic_l1, refined_extrinsic_r1 = refine_pose(best_inliers, queryidx_to_cloud_idx, cloud0, inliers0, kp_left1, K, tvec0)
    return refined_extrinsic_l1, refined_extrinsic_r1, best_inliers

# Timing for the exercise experiment

class OperationTimer:
    def __init__(self):
        self.timings = {}
        self.start_times = {}
        self.operation_order = []

    def start_operation(self, name):
        self.start_times[name] = time.perf_counter()

    def end_operation(self, name):
        end_time = time.perf_counter()
        if name not in self.timings:
            self.timings[name] = []
            if name not in self.operation_order:
                self.operation_order.append(name)
        self.timings[name].append(end_time - self.start_times[name])

    def get_report(self):
        print("\nOperation Times (in chronological order):")
        iteration_times = []

        # First display non-iteration operations
        for operation in self.operation_order:
            times = self.timings[operation]
            if operation.startswith("Iteration "):
                iteration_times.extend(times)
                continue

            # Handle nested operations with indentation
            indent = "  " if "/" in operation else ""
            avg_time = sum(times) / len(times)
            print(f"{indent}- {operation}: {avg_time:.6f} seconds (called {len(times)} times)")

        if iteration_times:
            print("\nIteration Statistics:")
            avg_iter_time = sum(iteration_times) / len(iteration_times)
            print(f"- Average iteration time: {avg_iter_time:.6f} seconds")

            # Use only iteration times for total time calculation
            total_time = sum(iteration_times)
            print(f"Total Time: {total_time:.4f} seconds")

# Exercise 3.6 — sequential visual odometry

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

def plot_sequence_trajectory(frame_states, frame_inliers_matches_count, frame_common_matches_count, frame_supporters_count):
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
    plt.savefig(COURSEWORK_OUTPUT_DIR + "camera_analysis.png")

def process_timed_stereo_pair(detector, camera_left, camera_right, img_left, img_right, logger):
    """Process a stereo image pair to extract features and compute matches."""
    logger.start_operation("Feature Detection")
    kp_left, des_left = detector.detectAndCompute(img_left, None)
    kp_right, des_right = detector.detectAndCompute(img_right, None)
    logger.end_operation("Feature Detection")

    logger.start_operation("Stereo Matching")
    inliers_matches, cloud = prepare_stereo_matches(camera_left, camera_right, kp_left, des_left, kp_right, des_right)
    logger.end_operation("Stereo Matching")

    return kp_left, des_left, kp_right, des_right, inliers_matches, cloud

def run_sequence_odometry():
    logger = OperationTimer()

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
        gt_poses = [parse_gt_line(line) for line in f.readlines()[:LAST_FRAME+1]]

    # Set ground truth for initial frame
    first_frame.ground_truth_location = gt_poses[FIRST_FRAME]

    # Initialize tracking variables
    prev_kp_left, prev_des_left, prev_kp_right, prev_des_right = None, None, None, None
    prev_inliers_matches, prev_cloud = None, None

    # Count stats for each frame
    frame_inliers_matches_count = []
    frame_common_matches_count = []
    frame_supporters_count = []
    for i in tqdm(range(FIRST_FRAME, LAST_FRAME)):
        logger.start_operation(f"Iteration {i}")

        # Process first image pair (either new or from previous iteration)
        if prev_kp_left is None:
            logger.start_operation("Image Reading")
            img_left0, img_right0 = read_images(i)
            logger.end_operation("Image Reading")

            kp_left0, des_left0, kp_right0, des_right0, inliers_matches0, cloud0 = process_timed_stereo_pair(
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

        kp_left1, des_left1, kp_right1, des_right1, inliers_matches1, cloud1 = process_timed_stereo_pair(
            detector, camera_left, camera_right, img_left1, img_right1, logger
        )

        # Store current features for next iteration
        prev_kp_left, prev_des_left = kp_left1, des_left1
        prev_kp_right, prev_des_right = kp_right1, des_right1
        prev_inliers_matches, prev_cloud = inliers_matches1, cloud1

        # Find matches between consecutive frames
        logger.start_operation("Temporal Matching")
        matches_between_img0_img1 = match_temporal_descriptors(des_left0, des_left1)
        common_matches, common_matches_indices = find_stereo_temporal_matches(
            inliers_matches0, inliers_matches1, matches_between_img0_img1
        )
        logger.end_operation("Temporal Matching")
        frame_common_matches_count.append(len(common_matches_indices))

        # Estimate motion between frames
        logger.start_operation("Motion Estimation")
        tvec0 = M2[:,3].reshape(3,1)
        relative_extrinsics_l1, _, supporters = estimate_timed_motion(
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

    plot_sequence_trajectory(frame_states, frame_inliers_matches_count, frame_common_matches_count, frame_supporters_count)
    logger.get_report()
    # print the distance between the last estimated and ground truth camera locations
    estimated_location = frame_states[-1].absolute_camera_location
    gt_location = frame_states[-1].ground_truth_location
    distance = np.linalg.norm(estimated_location - gt_location)
    print(f"Distance between estimated and ground truth camera locations: {distance:.2f} meters")

# Two-frame demonstration plots

def create_visualization_with_matches(img, keypoints, supporters, idx_pos):
    """Create an image with keypoints colored based on supporter status"""
    supporter_indices = [support[idx_pos] for support in supporters]
    supporters_kp = [keypoints[idx] for idx in supporter_indices]
    non_supporter_indices = [i for i in range(len(keypoints)) if i not in supporter_indices]
    non_supporters_kp = [keypoints[idx] for idx in non_supporter_indices]


    img_with_matches = cv2.drawKeypoints(img, non_supporters_kp, None, color=CYAN, flags=0)
    img_with_matches = cv2.drawKeypoints(img_with_matches, supporters_kp, None, color=ORANGE, flags=0)


    # Add legend
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img_with_matches, f'{len(supporters)} Supporters', (10, 30), font, 1, ORANGE, 2)
    cv2.putText(img_with_matches, f'{len(non_supporters_kp)} Non-Supporters', (10, 60), font, 1, CYAN, 2)

    return img_with_matches

def plot_camera_locations(location_l0, location_r0, location_l1, location_r1):
    plt.figure()
    plt.scatter(location_l0[0], location_l0[2], color='r', label='Left Camera 0')
    plt.scatter(location_r0[0], location_r0[2], color='b', label='Right Camera 0')
    plt.scatter(location_l1[0], location_l1[2], color='g', label='Left Camera 1')
    plt.scatter(location_r1[0], location_r1[2], color='y', label='Right Camera 1')
    plt.xlabel('X')
    plt.ylabel('Z')
    plt.legend()
    plt.savefig(COURSEWORK_OUTPUT_DIR + 'camera_locations.png')

def plot_supporter_matches(img_left0, img_left1, kp_left0, kp_left1, supporters):
    # Create visualizations with matches - use index 0 for left0 and index 2 for left1
    left0_vis = create_visualization_with_matches(img_left0, kp_left0, supporters, 0)
    left1_vis = create_visualization_with_matches(img_left1, kp_left1, supporters, 2)

    # Show matches visualization
    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(left0_vis, cv2.COLOR_BGR2RGB))
    plt.title('Left Image 0')
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(left1_vis, cv2.COLOR_BGR2RGB))
    plt.title('Left Image 1')
    plt.savefig(COURSEWORK_OUTPUT_DIR + 'matches_visualization.png')

def plot_motion_alignment(cloud0, cloud1, best_extrinsic_l1, img_left0, img_left1, kp_left0, kp_left1, best_inliers, common_matches_indices):
    """Plot the transformed point clouds and inliers/outliers visualization for q5."""
    # Part 1: Plot transformed point clouds
    transformed_cloud0 = coordinate_transform(cloud0, best_extrinsic_l1)
    transformed_cloud0 = transformed_cloud0.T  # Convert to same format as ex2.py (3xN)
    cloud1 = cloud1.T  # Convert to same format as ex2.py (3xN)

    # Create 3D visualization of point clouds
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')

    # Plot both clouds with different colors
    ax.scatter(transformed_cloud0[0], transformed_cloud0[2], transformed_cloud0[1],
              c='red', marker='o', label='Transformed Cloud 0')
    ax.scatter(cloud1[0], cloud1[2], cloud1[1],
              c='blue', marker='o', label='Cloud 1')

    ax.set_xlabel('X-axis (Meters)')
    ax.set_ylabel('Z-axis (Meters)')
    ax.set_zlabel('Y-axis (Meters)')

    plt.gca().invert_zaxis()
    ax.set_xlim((-50, 50))
    ax.set_ylim((-10, 100))
    ax.set_zlim((-10, 20))
    ax.view_init(elev=20, azim=-70)
    ax.set_aspect('equal')
    ax.set_title('Transformed Point Clouds Comparison')
    ax.legend()

    plt.savefig(COURSEWORK_OUTPUT_DIR + 'transformed_clouds.png')
    plt.close(fig)

    # Part 2: Plot inliers and outliers
    left0_vis = create_visualization_with_matches(img_left0, kp_left0, best_inliers, 0)
    left1_vis = create_visualization_with_matches(img_left1, kp_left1, best_inliers, 2)

    # Show matches visualization
    plt.figure(figsize=(20, 10))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(left0_vis, cv2.COLOR_BGR2RGB))
    plt.title('Left Image 0')
    plt.subplot(122)
    plt.imshow(cv2.cvtColor(left1_vis, cv2.COLOR_BGR2RGB))
    plt.title('Left Image 1')
    plt.savefig(COURSEWORK_OUTPUT_DIR + 'inliers_outliers_q5.png')
    plt.close()

# Exercise 3.1–3.6 — executable progression

def main():
    K, M1, M2 = read_cameras()
    camera_left, camera_right = K @ M1, K @ M2

    img_left0, img_right0 = read_images(0)
    kp_left0, des_left0 = detector.detectAndCompute(img_left0, None)
    kp_right0, des_right0 = detector.detectAndCompute(img_right0, None)
    inliers_matches0, cloud0 = prepare_stereo_matches(camera_left, camera_right, kp_left0, des_left0, kp_right0, des_right0)

    img_left1, img_right1 = read_images(1)
    kp_left1, des_left1 = detector.detectAndCompute(img_left1, None)
    kp_right1, des_right1 = detector.detectAndCompute(img_right1, None)
    inliers_matches1, cloud1 = prepare_stereo_matches(camera_left, camera_right, kp_left1, des_left1, kp_right1, des_right1)

    matches_between_img0_img1 = match_temporal_descriptors(des_left0, des_left1)

    common_matches, common_matches_indices = find_stereo_temporal_matches(
        inliers_matches0, inliers_matches1, matches_between_img0_img1
    )

    tvec0 = M2[:,3]
    extrinsic_matrix_l1, extrinsic_matrix_r1, location_l1, location_r1 = solve_pnp_and_locations(common_matches, cloud0, inliers_matches0, kp_left1, K, tvec0)
    location_l0 = find_camera_location(M1)
    location_r0 = find_camera_location(M2)
    plot_camera_locations(location_l0, location_r0, location_l1, location_r1)

    supporters = classify_timed_supporters(common_matches_indices, cloud0, kp_left0,kp_right0,kp_left1,kp_right1, K, extrinsic_matrix_l1,extrinsic_matrix_r1,M1,M2, inliers_matches0)
    plot_supporter_matches(img_left0, img_left1, kp_left0, kp_left1, supporters)

    best_extrinsic_l1, best_extrinsic_r1, best_inliers =  estimate_timed_motion(common_matches_indices, common_matches, cloud0, cloud1, inliers_matches0, kp_left0, kp_right0, kp_left1, kp_right1, K, M1, M2, tvec0)
    plot_motion_alignment(cloud0, cloud1, best_extrinsic_l1, img_left0, img_left1, kp_left0, kp_left1, best_inliers, common_matches_indices)

    run_sequence_odometry()


if __name__ == "__main__":
    from reports.paths import ensure_output_directories
    ensure_output_directories()
    main()
