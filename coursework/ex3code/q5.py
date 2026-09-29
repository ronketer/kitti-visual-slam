from kitti_slam.motion import refine_pose, extract_correspondences
import numpy as np
import cv2
from .q4 import *
from .q3 import *

START_OUTLIER_RATIO = 0.99
ORANGE = (0, 165, 255)
CYAN = (255, 255, 0)
MAXITERATIONS = 10
SUCCESS_PROBABILITY = 0.999
SAMPLE_SIZE = 4
MINITERATIONS = 100

iteration_calculator = lambda outlier_ratio : int(
    np.ceil(np.log(1 - SUCCESS_PROBABILITY) / np.log(1 - ((1 - outlier_ratio) ** SAMPLE_SIZE))))



def perform_ransac_loop(
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

        extrinsic_matrix_l1, extrinsic_matrix_r1, location_l1, location_r1 = q3(common_matches, cloud0, inliers0, kp_left1, K, tvec0)

        current_inliers = q4(common_matches_indices, cloud0, kp_left0, kp_right0, kp_left1, kp_right1,
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


def q5(common_matches_indices, common_matches, cloud0, cloud1, inliers0, kp_left0, kp_right0, kp_left1, kp_right1, K, extrinsic_l0, extrinsic_r0, tvec0, logger=None):
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
    best_inliers, best_extrinsic_l1, best_extrinsic_r1 = perform_ransac_loop(
        common_matches_indices, common_matches, queryidx_to_cloud_idx, cloud0, kp_left1, K, tvec0,
        kp_left0, kp_right0, kp_right1, extrinsic_l0, extrinsic_r0, inliers0, logger
    )
    refined_extrinsic_l1, refined_extrinsic_r1 = refine_pose(best_inliers, queryidx_to_cloud_idx, cloud0, inliers0, kp_left1, K, tvec0)
    return refined_extrinsic_l1, refined_extrinsic_r1, best_inliers



def createImgWithMatches(img, kp, inliers, outliers, color_in=ORANGE, color_out=CYAN):
    """
    Draw inlier and outlier keypoints on an image in different colors.

    Args:
        img: Input image.
        kp: List of keypoints.
        inliers: List of inlier match tuples.
        outliers: List of outlier match tuples.
        color_in: Color for inliers (BGR tuple).
        color_out: Color for outliers (BGR tuple).

    Returns:
        img_with_both: Image with inliers and outliers drawn.
    """
    img_with_inliers = cv2.drawKeypoints(
        img,
        [kp[m[0]] for m in inliers],  # or m[2] for left1
        None,
        color=color_in,
        flags=0,
    )
    img_with_both = cv2.drawKeypoints(
        img_with_inliers,
        [kp[m[0]] for m in outliers],  # or m[2] for left1
        None,
        color=color_out,
        flags=0,
    )
    return img_with_both
