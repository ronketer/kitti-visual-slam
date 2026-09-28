"""Stereo matching, triangulation, and four-view correspondence assembly."""

import cv2
import numpy as np
from detector_config import create_detector_and_matcher


def filter_by_ratio_test(matches, thresholds):
    """
    Apply ratio test with multiple thresholds.

    Args:
        matches: List of knn matches (k=2) from BFMatcher
        thresholds: List of ratio thresholds (e.g., [0.75, 0.8])

    Returns:
        {
            "good": {threshold: list_of_indices_passing_threshold},
            "bad": {threshold: list_of_indices_failing_threshold}
        }
    """
    results = {"good": {t: [] for t in thresholds}, "bad": {t: [] for t in thresholds}}
    for idx, (best_match, second_best_match) in enumerate(matches):
        for t in thresholds:
            if best_match.distance < t * second_best_match.distance:
                results["good"][t].append(idx)
            else:
                results["bad"][t].append(idx)

    return results


def classify_matches_by_deviation(matches, kp0, kp1):
    """
    Classify matches into inliers and outliers based on y-coordinate deviation.

    Args:
        matches: List of matches to classify.
        kp0: Keypoints from the query image.
        kp1: Keypoints from the train image.

    Returns:
        Tuple of inliers and outliers.
    """
    inliers = []
    outliers = []

    for match in matches:
        train_keypoint = kp1[match.trainIdx]
        query_keypoint = kp0[match.queryIdx]
        cur_deviation = abs(query_keypoint.pt[1] - train_keypoint.pt[1])

        if cur_deviation < 2:
            inliers.append(match)
        else:
            outliers.append(match)

    return inliers, outliers


def trangulate_inliers(camera1, camera2, kp_left, des_left, kp_right, des_right, *, matcher=None):
    """
    Triangulate inliers from stereo matches using the provided cameras.
    Uses the matcher to find matches, classifies them by deviation, and performs triangulation.
    Validate kp_left.x > kp_right.x for non-behind-the-car points.
    Args:
        camera1: Camera matrix for the first camera.
        camera2: Camera matrix for the second camera.
        kp_left: Keypoints from the left image.
        des_left: Descriptors from the left image.
        kp_right: Keypoints from the right image.
        des_right: Descriptors from the right image.
    Returns:
        Tuple containing:
            - Inlier matches after validation
            - 3D point cloud from triangulation
    """
    if matcher is None:
        _, matcher = create_detector_and_matcher()
    matches = matcher.match(des_left, des_right)
    inliers_matches, _ = classify_matches_by_deviation(matches, kp_left, kp_right)
    MIN_DISPARITY_THRESHOLD = 2
    filtered_matches = [
        m
        for m in inliers_matches
        if kp_left[m.queryIdx].pt[0] > kp_right[m.trainIdx].pt[0]
        and (kp_left[m.queryIdx].pt[0] - kp_right[m.trainIdx].pt[0])
        >= MIN_DISPARITY_THRESHOLD
    ]
    pts_left = np.float32([kp_left[m.queryIdx].pt for m in filtered_matches]).T
    pts_right = np.float32([kp_right[m.trainIdx].pt for m in filtered_matches]).T
    if pts_left.shape[1] == 0:
        return [], np.empty((0, 3))
    cloud4D = cv2.triangulatePoints(camera1, camera2, pts_left, pts_right)
    points3D = (cloud4D[:3] / cloud4D[3]).T
    return filtered_matches, points3D


def solveLLST(points2D1, points2D2, camera1, camera2):
    """
    Solve the linear least squares triangulation problem.
    Args:
        points2D1: 2D points from the first camera.
        points2D2: 2D points from the second camera.
        camera1: Camera matrix for the first camera.
        camera2: Camera matrix for the second camera.
    Returns:
        3D point in homogeneous coordinates.
    """
    row1_camera1 = np.array([camera1[0, :]])
    row2_camera1 = np.array([camera1[1, :]])
    row3_camera1 = np.array([camera1[2, :]])

    row1_camera2 = np.array([camera2[0, :]])
    row2_camera2 = np.array([camera2[1, :]])
    row3_camera2 = np.array([camera2[2, :]])

    px1 = points2D1[0]
    py1 = points2D1[1]
    px2 = points2D2[0]
    py2 = points2D2[1]
    A = np.array(
        [
            px1 * row3_camera1 - row1_camera1,
            py1 * row3_camera1 - row2_camera1,
            px2 * row3_camera2 - row1_camera2,
            py2 * row3_camera2 - row2_camera2,
        ]
    )
    A = np.vstack(A)
    U, S, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X / X[3]
    return X[:3]


def process_stereo_pair(img_left, img_right, camera_left, camera_right, *, detector=None, matcher=None):
    """
    Process stereo image pair using geometric validation from q6.

    Uses triangulation to validate stereo matches and extract 3D points.
    Args:
        img_left: Left stereo image.
        img_right: Right stereo image.
        camera_left: Camera matrix for the left camera.
        camera_right: Camera matrix for the right camera.
    Returns:
        Tuple containing:
            - Keypoints and descriptors for left image
            - Keypoints and descriptors for right image
            - Inlier matches after validation
            - 3D point cloud from triangulation
    """
    if detector is None or matcher is None:
        default_detector, default_matcher = create_detector_and_matcher()
        detector = default_detector if detector is None else detector
        matcher = default_matcher if matcher is None else matcher

    # Detect features
    kp_left, desc_left = detector.detectAndCompute(img_left, None)
    kp_right, desc_right = detector.detectAndCompute(img_right, None)

    # Match and validate stereo features using q1
    inliers_matches, cloud = trangulate_inliers(
        camera_left, camera_right, kp_left, desc_left, kp_right, desc_right, matcher=matcher
    )

    return kp_left, desc_left, kp_right, desc_right, inliers_matches, cloud


def find_stereo_temporal_matches(
    inliers_matches0, inliers_matches1, matches_between_frames
):
    """
    Find common matches between two consecutive stereo frame pairs.

    Args:
        inliers_matches0: Matches from first stereo pair
        inliers_matches1: Matches from second stereo pair
        matches_between_frames: Matches between consecutive frames

    Returns:
        Tuple of (common_matches, common_matches_indices), where common_matches_indices contains
        4-tuples of (left0_idx, right0_idx, left1_idx, right1_idx) for corresponding points
    """

    inlier0_idx = {m.queryIdx for m in inliers_matches0}
    inlier1_idx = {m.queryIdx for m in inliers_matches1}
    common_matches = [
        m
        for m in matches_between_frames
        if m.queryIdx in inlier0_idx and m.trainIdx in inlier1_idx
    ]

    inlier0_right = {m.queryIdx: m.trainIdx for m in inliers_matches0}
    inlier1_right = {m.queryIdx: m.trainIdx for m in inliers_matches1}

    common_matches_indices = [
        (
            m.queryIdx,
            inlier0_right[m.queryIdx],
            m.trainIdx,
            inlier1_right[m.trainIdx],
        )
        for m in common_matches
    ]

    return common_matches, common_matches_indices
