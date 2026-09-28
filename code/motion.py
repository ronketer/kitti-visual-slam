"""Custom adaptive RANSAC with OpenCV P3P hypotheses and iterative refinement."""

import random
import cv2
import numpy as np
from geometry import compose_extrinsics, find_camera_location
from supporters import find_ransac_iteration_supporters


def rodriguez_to_mat(rvec, tvec):
    rot, _ = cv2.Rodrigues(rvec)
    return np.hstack((rot, tvec))


def solve_pnp_and_locations(common_matches, cloud0, inliers0, kp_left1, K, tvec0):
    idx_map = {m.queryIdx: i for i, m in enumerate(inliers0)}
    success = False
    attempts = 0
    while not success and attempts < 100:
        # Randomly sample 4 matches from common_matches
        sample = random.sample(common_matches, 4)
        pts3D = np.array([cloud0[idx_map[m.queryIdx]] for m in sample], dtype=np.float32)
        pts2D = np.array([kp_left1[m.trainIdx].pt for m in sample], dtype=np.float32)

        # Solve PnP to get the rotation and translation vectors
        success, rvec, tvec = cv2.solvePnP(pts3D, pts2D, K, None, flags=cv2.SOLVEPNP_P3P)
        attempts += 1

    if not success:
        raise RuntimeError("Failed to solve PnP after 100 attempts")
    extrinsic_matrix_l1 =  rodriguez_to_mat(rvec, tvec)
    location_l1 = find_camera_location(extrinsic_matrix_l1)
    relative_extrinsic_matrix_r1 = np.hstack((np.eye(3), tvec0.reshape(3, 1)))
    extrinsic_matrix_r1 = compose_extrinsics(extrinsic_matrix_l1, relative_extrinsic_matrix_r1)
    location_r1 = find_camera_location(extrinsic_matrix_r1)
    return extrinsic_matrix_l1, extrinsic_matrix_r1, location_l1, location_r1


START_OUTLIER_RATIO = 0.99
ORANGE = (0, 165, 255)
CYAN = (255, 255, 0)
MAXITERATIONS = 10
SUCCESS_PROBABILITY = 0.999
SAMPLE_SIZE = 4

iteration_calculator = lambda outlier_ratio : int(
    np.ceil(np.log(1 - SUCCESS_PROBABILITY) / np.log(1 - ((1 - outlier_ratio) ** SAMPLE_SIZE))))


def perform_ransac_loop(
    common_matches_indices, common_matches, queryidx_to_cloud_idx, cloud0, kp_left1, K, tvec0,
    kp_left0, kp_right0, kp_right1, extrinsic_l0, extrinsic_r0, inliers0
):
    best_num_inliers = 0
    best_inliers = []
    best_extrinsic_l1 = None
    best_extrinsic_r1 = None


    max_iters = iteration_calculator(START_OUTLIER_RATIO)

    i = 0
    while i < max_iters:
        extrinsic_matrix_l1, extrinsic_matrix_r1, location_l1, location_r1 = solve_pnp_and_locations(common_matches, cloud0, inliers0, kp_left1, K, tvec0)

        current_inliers = find_ransac_iteration_supporters(common_matches_indices, cloud0, kp_left0, kp_right0, kp_left1, kp_right1,
                        K, extrinsic_matrix_l1, extrinsic_matrix_r1, extrinsic_l0, extrinsic_r0, inliers0)
        if len(current_inliers) > best_num_inliers:
            best_num_inliers = len(current_inliers)
            best_inliers = current_inliers
            best_extrinsic_l1 = extrinsic_matrix_l1.copy()
            best_extrinsic_r1 = extrinsic_matrix_r1.copy()
            outlier_ratio = 1 - (len(current_inliers) / len(common_matches_indices))
            max_iters = min(iteration_calculator(outlier_ratio), max_iters)
        i+=1
    return best_inliers, best_extrinsic_l1, best_extrinsic_r1

def refine_pose(best_inliers, queryidx_to_cloud_idx, cloud0, inliers0, kp_left1, K, tvec0
):
    """
    Refines the pose using all provided inliers and re-validates inliers.
    Returns the refined extrinsics and the refined inlier matches.
    """
    pts3D, pts2D = extract_correspondences(best_inliers, queryidx_to_cloud_idx, cloud0, kp_left1)
    if len(pts3D) == 0:
        raise ValueError("No valid correspondences found for PnP")

    # Ensure points are properly shaped for solvePnP
    pts3D = pts3D.reshape(-1, 3)
    pts2D = pts2D.reshape(-1, 1, 2)

    # Use initial rotation and translation from RANSAC as starting point
    initial_rvec = np.zeros(3, dtype=np.float32)  # Start with identity rotation
    initial_tvec = np.zeros(3, dtype=np.float32)  # Start at origin
    success, rvec, tvec = cv2.solvePnP(pts3D, pts2D, K, None, initial_rvec, initial_tvec, True, cv2.SOLVEPNP_ITERATIVE)
    if not success:
        raise ValueError("Refinement solvePnP failed")

    R, _ = cv2.Rodrigues(rvec)
    refined_extrinsic_l1 = np.hstack((R, tvec.reshape(3, 1)))
    relative_extrinsic_r1 = np.hstack((np.eye(3), tvec0.reshape(3, 1)))
    refined_extrinsic_r1 = compose_extrinsics(refined_extrinsic_l1, relative_extrinsic_r1)



    return refined_extrinsic_l1, refined_extrinsic_r1

def extract_correspondences(matches, queryidx_to_cloud_idx, cloud0, kp_left1):
    """
    Given a list of match tuples and a mapping from query indices to cloud indices,
    extract the corresponding 3D points from cloud0 and 2D points from kp_left1.

    Args:
        matches: List of (idx_l0, idx_r0, idx_l1, idx_r1) tuples.
        queryidx_to_cloud_idx: Dict mapping idx_l0 to index in cloud0.
        cloud0: Nx3 array of 3D points.
        kp_left1: List of keypoints for left1 image.

    Returns:
        pts3D: (N, 3) array of 3D points.
        pts2D: (N, 2) array of 2D points.
    """
    pts3D, pts2D = [], []
    for (idx_l0, idx_r0, idx_l1, idx_r1) in matches:
        cloud_idx = queryidx_to_cloud_idx.get(idx_l0)
        if cloud_idx is not None:
            pts3D.append(cloud0[cloud_idx])
            pts2D.append(kp_left1[idx_l1].pt)
    return np.array(pts3D, dtype=np.float32), np.array(pts2D, dtype=np.float32)

def perform_motion_estimation(common_matches_indices, common_matches, cloud0, cloud1, inliers0, kp_left0, kp_right0, kp_left1, kp_right1, K, extrinsic_l0, extrinsic_r0, tvec0):
    # ransac_iters = MAXITERATIONS
    queryidx_to_cloud_idx = {m.queryIdx: i for i, m in enumerate(inliers0)}


    if len(common_matches_indices) < SAMPLE_SIZE:
        print(f"Only {len(common_matches_indices)} matches - too few for RANSAC")
        return None, None, []  # Return empty results



    # --- 2. Perform RANSAC Loop ---
    best_inliers, best_extrinsic_l1, best_extrinsic_r1 = perform_ransac_loop(
        common_matches_indices, common_matches, queryidx_to_cloud_idx, cloud0, kp_left1, K, tvec0,
        kp_left0, kp_right0, kp_right1, extrinsic_l0, extrinsic_r0, inliers0
    )
    refined_extrinsic_l1, refined_extrinsic_r1 = refine_pose(best_inliers, queryidx_to_cloud_idx, cloud0, inliers0, kp_left1, K, tvec0)
    return refined_extrinsic_l1, refined_extrinsic_r1, best_inliers
