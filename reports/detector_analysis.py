import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import sys
import os
import random



from reports.paths import LASTFRAME, FIRSTFRAME, FINAL_PLOTS_RELATIVE_PATH
from kitti_slam.dataset import read_images
from kitti_slam.dataset import read_cameras
from reports.plots import plot_average_inliers
from reports.plots import plot_spread_performance
from reports.plots import plot_processing_time
from reports.plots import plot_detector_repeatability

from kitti_slam.stereo import classify_matches_by_deviation


def process_stereo_pair_by_detector(
    img_left,
    img_right,
    camera1_proj_mat,
    camera2_proj_mat,
    detector_obj,
    current_matcher,
):
    """
    Processes a stereo image pair using a specified detector and matcher,
    performing feature detection, matching, epipolar filtering, disparity filtering,
    and triangulation. The filtering logic exactly matches that found in
    classify_matches_by_deviation and trangulate_inliers from alg.py.

    Args:
        img_left (np.array): Left stereo image.
        img_right (np.array): Right stereo image.
        camera1_proj_mat (np.array): Projection matrix for the left camera.
        camera2_proj_mat (np.array): Projection matrix for the right camera.
        detector_obj: An OpenCV feature detector object (e.g., cv2.SIFT_create()).
        current_matcher: An OpenCV matcher object (e.g., cv2.BFMatcher()).

    Returns:
        Tuple: (kp_left, des_left, kp_right, des_right, inliers, total_raw_matches)
            - kp_left, des_left: Keypoints and descriptors for the left image.
            - kp_right, des_right: Keypoints and descriptors for the right image.
            - inliers (list): Validated inlier matches after epipolar and disparity filtering.
            - total_raw_matches (int): Total number of raw matches found.
    """

    min_disparity_threshold = 2

    kp_left, des_left = detector_obj.detectAndCompute(img_left, None)
    kp_right, des_right = detector_obj.detectAndCompute(img_right, None)

    raw_matches = current_matcher.match(des_left, des_right)
    total_raw_matches = len(raw_matches) if raw_matches else 0

    epipolar_inliers, _ = classify_matches_by_deviation(raw_matches, kp_left, kp_right)

    disparity_inliers = [
        m
        for m in epipolar_inliers
        if kp_left[m.queryIdx].pt[0] > kp_right[m.trainIdx].pt[0]
        and (kp_left[m.queryIdx].pt[0] - kp_right[m.trainIdx].pt[0])
        >= min_disparity_threshold
    ]

    pts_left = np.float32([kp_left[m.queryIdx].pt for m in disparity_inliers]).T
    pts_right = np.float32([kp_right[m.trainIdx].pt for m in disparity_inliers]).T

    if pts_left.shape[1] == 0:
        return (
            kp_left,
            des_left,
            kp_right,
            des_right,
            [],
            total_raw_matches,
            np.array([]),
        )

    cloud4D = cv2.triangulatePoints(
        camera1_proj_mat, camera2_proj_mat, pts_left, pts_right
    )

    points3D = (cloud4D[:3] / cloud4D[3]).T

    return (
        kp_left,
        des_left,
        kp_right,
        des_right,
        disparity_inliers,
        total_raw_matches,
        points3D,
    )


def calculate_keypoint_spread(keypoints, image_shape, grid_size=(20, 20)):
    """
    Calculates the spread of keypoints across an image grid.

    Args:
        keypoints: List of cv2.KeyPoint objects.
        image_shape: Tuple representing the image shape (height, width, channels).
        grid_size: Tuple (rows, cols) defining the grid dimensions.

    Returns:
        float: Percentage of occupied grid cells.
    """
    if not keypoints:
        return 0.0

    h, w = image_shape[:2]
    cell_width = w / grid_size[1]
    cell_height = h / grid_size[0]

    occupied_cells = np.zeros(grid_size, dtype=bool)

    for kp in keypoints:
        x, y = kp.pt
        if 0 <= x < w and 0 <= y < h:
            col = int(x / cell_width)
            row = int(y / cell_height)
            col = min(col, grid_size[1] - 1)
            row = min(row, grid_size[0] - 1)
            occupied_cells[row, col] = True

    total_cells = grid_size[0] * grid_size[1]
    covered_cells = np.sum(occupied_cells)
    return (covered_cells / total_cells) * 100


def generate_detector_performance_data(first_frame, last_frame):
    """
    Generates performance data for different feature detectors (SIFT, ORB, AKAZE).
    This function processes image sequences and collects metrics for each detector.
    It now uses process_stereo_pair_by_detector for the main processing logic.

    Args:
        first_frame (int): The starting frame index for analysis.
        last_frame (int): The ending frame index for analysis.

    Returns:
        dict: A dictionary containing performance metrics for each detector.
    """
    detectors = {
        "SIFT": cv2.SIFT_create(),
        "ORB": cv2.ORB_create(),
        "AKAZE": cv2.AKAZE_create(),
    }
    matchers = {
        "SIFT": cv2.BFMatcher(cv2.NORM_L2, crossCheck=False),
        "ORB": cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False),
        "AKAZE": cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False),
    }

    K_mat, M1, M2 = read_cameras()
    camera1_proj_mat = K_mat @ M1
    camera2_proj_mat = K_mat @ M2

    results = {
        name: {
            "inlier_counts": [],
            "inlier_percentages": [],
            "times": 0,
            "spread_metrics": [],
        }
        for name in detectors
    }

    for name, detector_obj in detectors.items():
        start_time = time.time()
        current_matcher = matchers[name]

        for frame_idx in tqdm(
            range(first_frame, last_frame + 1), desc=f"Processing {name}"
        ):
            img_left, img_right = read_images(frame_idx)

            kp_left, des_left, kp_right, des_right, inliers, total_matches, _ = (
                process_stereo_pair_by_detector(
                    img_left,
                    img_right,
                    camera1_proj_mat,
                    camera2_proj_mat,
                    detector_obj,
                    current_matcher,
                )
            )

            spread = calculate_keypoint_spread(kp_left, img_left.shape)
            results[name]["spread_metrics"].append(spread)

            num_inliers = len(inliers)
            percentage = (num_inliers / total_matches) * 100 if total_matches > 0 else 0

            results[name]["inlier_counts"].append(num_inliers)
            results[name]["inlier_percentages"].append(percentage)

        results[name]["times"] = time.time() - start_time
    return results


def evaluate_detector_repeatibility(detector=cv2.ORB_create(), img_index=1, plot=False):
    """
    Evaluates the repeatability of a feature detector by comparing keypoints
    detected in an original image and a randomly transformed version of that image.

    The repeatability score is a metric of how consistently a detector finds the
    same features under affine transformations (rotation, scaling, and translation).

    Args:
        detector: An OpenCV feature detector object (e.g., cv2.ORB_create(),
                  cv2.SIFT_create()).
        img_index (int): The index of the image to be loaded and tested.
        plot (bool, optional): If True, displays the original and warped images
                               with detected keypoints. Defaults to False.

    Returns:
        float: The repeatability score, calculated as the ratio of repeatable
               keypoints to the average number of keypoints detected in the
               original and warped images. The value is between 0.0 and 1.0.
    """

    ROTATION_DEG = random.uniform(-30, 30)
    SCALE = random.uniform(0.8, 1.2)
    TX = random.randint(-50, 50)
    TY = random.randint(-50, 50)

    img, _ = read_images(img_index)
    if img is None:
        raise FileNotFoundError(
            "Could not load image. Replace with your own if needed."
        )

    h, w = img.shape
    center = (w // 2, h // 2)

    M_affine = cv2.getRotationMatrix2D(center, ROTATION_DEG, SCALE)

    M_affine[0, 2] += TX
    M_affine[1, 2] += TY

    img_warped = cv2.warpAffine(img, M_affine, (w, h))

    H_affine = np.vstack([M_affine, [0, 0, 1]])

    kp1, des1 = detector.detectAndCompute(img, None)
    kp2, des2 = detector.detectAndCompute(img_warped, None)

    pts1_kp = np.array([k.pt for k in kp1], dtype=np.float32).reshape(-1, 1, 2)
    pts1_kp_warped = cv2.perspectiveTransform(pts1_kp, H_affine)

    threshold = 3.0
    pts2_kp = np.array([k.pt for k in kp2], dtype=np.float32)

    repeatable_count = 0
    for pt in pts1_kp_warped[:, 0, :]:
        dists = np.linalg.norm(pts2_kp - pt, axis=1)
        if np.min(dists) < threshold:
            repeatable_count += 1

    repeatability = repeatable_count / ((len(kp1) + len(kp2)) / 2)

    if plot:
        img_matches = cv2.drawKeypoints(img, kp1, None, color=(0, 255, 0))
        img_warped_matches = cv2.drawKeypoints(img_warped, kp2, None, color=(0, 255, 0))

        plt.subplot(1, 2, 1)
        plt.title("Original Image")
        plt.imshow(img_matches, cmap="gray")
        plt.subplot(1, 2, 2)
        plt.title("Warped Image")
        plt.imshow(img_warped_matches, cmap="gray")
        plt.tight_layout()
        plt.show()

    return repeatability


if __name__ == "__main__":
    output_dir = os.path.join(FINAL_PLOTS_RELATIVE_PATH, "detector-comparison")

    import pickle

    with open(os.path.join(output_dir, "detector_performance_data.pkl"), "rb") as f:
        all_results = pickle.load(f)
        print(len(all_results["AKAZE"]["spread_metrics"]))

    plot_average_inliers(all_results, os.path.join(output_dir, "average_inliers.png"))

    plot_spread_performance(
        all_results,
        FIRSTFRAME,
        LASTFRAME,
        os.path.join(output_dir, "spread_performance.png"),
    )

    plot_processing_time(
        all_results,
        FIRSTFRAME,
        LASTFRAME,
        os.path.join(output_dir, "processing_time.png"),
    )

    detectors = {
        "ORB": cv2.ORB_create(),
        "AKAZE": cv2.AKAZE_create(),
        "SIFT": cv2.SIFT_create(),
    }

    detector_results = {name: [] for name in detectors.keys()}

    num_test_images = 100

    for detector_name, detector_obj in detectors.items():
        print(f"Testing {detector_name} detector...")
        for _ in tqdm(range(num_test_images)):
            img_index = random.randint(0, LASTFRAME + 1)
            repeatability = evaluate_detector_repeatibility(
                detector=detector_obj, img_index=img_index
            )
            detector_results[detector_name].append(repeatability)

        mean_repeatability = np.mean(detector_results[detector_name])
        std_repeatability = np.std(detector_results[detector_name])
        print(
            f"{detector_name} - Mean Repeatability: {mean_repeatability:.3f}, Std: {std_repeatability:.4f}"
        )
    detector_names = list(detector_results.keys())

    mean_repeatabilities = [np.mean(detector_results[name]) for name in detector_names]
    std_repeatabilities = [np.std(detector_results[name]) for name in detector_names]

    plot_detector_repeatability(
        detector_names,
        mean_repeatabilities,
        std_repeatabilities,
        "detector_repeatability_comparison.png",
    )
