import os
import matplotlib.pyplot as plt
import numpy as np
from geometry import compose_extrinsics, find_camera_location, coordinate_transform
import cv2
from gtsam_geometry import read_calibration
from dataset import read_images, read_cameras, parse_gt_line_matrix
from stereo import find_stereo_temporal_matches
from motion import rodriguez_to_mat

from mpl_toolkits.mplot3d import Axes3D
import time
from consts import (
    LEFT_IMG_DIR,
    RIGHT_IMG_DIR,
    CALIB_FILE,
    GT_POSES_FILE,
    LASTFRAME,
    DATA_PATH,
)


def parse_gt_line(line):
    """Parse one line of ground truth pose and return camera location."""
    mat = parse_gt_line_matrix(line)
    return find_camera_location(mat)


def read_ground_truth_poses(lastframe=LASTFRAME):
    """Read and parse all ground truth poses up to LASTFRAME."""
    with open(GT_POSES_FILE, "r") as f:
        return [parse_gt_line(line) for line in f.readlines()[: lastframe + 1]]



def plot_and_save(img, title, output_path, figsize, dpi):
    """Helper function to plot and save an image."""
    plt.figure(figsize=figsize, dpi=dpi)
    plt.imshow(img)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", dpi=dpi)
    plt.close()


def compute_camera_to_camera_transform(source_cam_extrinsic, target_cam_extrinsic):
    """Compute transformation matrix from one camera's coordinate system to another.

    Args:
        source_cam_extrinsic: Extrinsic matrix of the source camera (camera A)
        target_cam_extrinsic: Extrinsic matrix of the target camera (camera B)

    Returns:
        Transformation matrix that converts points from source camera's coordinate system
        to target camera's coordinate system.

    The transformation is derived from:
        x_source = R_source x_world + t_source
        x_target = R_target x_world + t_target
        Solving for x_world in first equation and substituting into second:
        x_target = R_target R_source^T (x_source - t_source) + t_target
        Which gives the transformation matrix: [R_target R_source^T | t_target - R_target R_source^T t_source]
    """
    R_source = source_cam_extrinsic[:, :3]
    t_source = source_cam_extrinsic[:, 3].reshape(3, 1)
    R_target = target_cam_extrinsic[:, :3]
    t_target = target_cam_extrinsic[:, 3].reshape(3, 1)

    cam_to_cam_rotation = R_target @ R_source.T
    cam_to_cam_translation = t_target - cam_to_cam_rotation @ t_source

    return np.hstack((cam_to_cam_rotation, cam_to_cam_translation))


def find_transformation(extrinsic_matrix):
    """Find the transformation matrix from the extrinsic matrix."""
    R = extrinsic_matrix[:, :3]
    t = extrinsic_matrix[:, 3]
    return lambda x: R @ x + t


def crop_patch(img, center, patch_size):
    """Crop a square patch from the image centered at the given point.

    Args:
        img: Input image
        center: (x, y) coordinates of the center of the patch
        patch_size: Size of the patch (width and height)

    Returns:
        Cropped patch as a numpy array
    """
    half_size = patch_size // 2
    x1 = max(center[0] - half_size, 0)
    x2 = min(center[0] + half_size, img.shape[1])
    y1 = max(center[1] - half_size, 0)
    y2 = min(center[1] + half_size, img.shape[0])

    return img[y1:y2, x1:x2]


def save_figure(path):
    """Saves the current matplotlib figure to the specified path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", dpi=300)
    plt.close()


def setup_plot(title, xlabel, ylabel, log_y=False):
    """Sets up common plot elements like title, labels, grid, and optional log scale."""
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True, alpha=0.3)
    if log_y:
        plt.yscale("log")


def plot_keypoints(img, keypoints, color=(0, 255, 0)):
    """Draw keypoints on an image. img can be grayscale or BGR."""

    if img.ndim == 2:
        img_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    else:
        img_bgr = img.copy()
    return cv2.drawKeypoints(img_bgr, keypoints, None, color=color, flags=0)


def plot_matches_and_supporters(
    frame_ids,
    total_matches_counts,
    common_matches_counts,
    supporters_counts,
    save_path_prefix,
):
    """
    Main coordinator function to generate all match and supporter plots.
    Handles data validation and downsampling before calling specific plotters.
    """

    plot_feature_counts(
        frame_ids,
        total_matches_counts,
        common_matches_counts,
        supporters_counts,
        save_path_prefix,
    )

    plot_filtering_efficiency(
        frame_ids,
        total_matches_counts,
        common_matches_counts,
        supporters_counts,
        save_path_prefix,
    )


def plot_feature_counts(
    frame_indices,
    total_matches_counts,
    common_matches_counts,
    supporters_counts,
    save_path_prefix,
):
    """
    Plots the absolute number of total matches, common matches, and RANSAC supporters.
    """
    plt.figure(figsize=(12, 7), layout="constrained")
    plt.plot(frame_indices, total_matches_counts, label="Total Left-Left Matches")
    plt.plot(frame_indices, common_matches_counts, label="Common Matches")
    plt.plot(
        frame_indices,
        supporters_counts,
        label="RANSAC Supporters (Inliers)",
        color="green",
    )

    max_y = max(max(total_matches_counts) if total_matches_counts else 0, 1) * 1.1
    plt.ylim(0, max_y)

    setup_plot(
        "Feature Matching & Filtering Counts per Frame",
        "Frame Index",
        "Number of Features",
    )
    plt.legend()
    plt.savefig(f"{save_path_prefix}_all_counts.png", dpi=300)


def plot_filtering_efficiency(
    frame_indices,
    total_matches_counts,
    common_matches_counts,
    supporters_counts,
    save_path_prefix,
):
    """
    Calculates and plots the percentage of common matches and RANSAC supporters.
    """

    percent_common_matches_of_total = [
        (common / total) * 100 if total > 0 else 0
        for common, total in zip(common_matches_counts, total_matches_counts)
    ]
    percent_supporters_of_common = [
        (supporters / common) * 100 if common > 0 else 0
        for supporters, common in zip(supporters_counts, common_matches_counts)
    ]

    fig, ax = plt.subplots(figsize=(12, 7), layout="constrained")

    ax.plot(
        frame_indices,
        percent_common_matches_of_total,
        label="% Common Matches (of Total)",
        color="purple",
    )
    ax.plot(
        frame_indices,
        percent_supporters_of_common,
        label="% Supporters (of Common Matches)",
        color="orange",
    )

    setup_plot(
        "Efficiency of Matching and Filtering Stages",
        "Frame Index",
        "Percentage (%)",
    )
    plt.legend()
    plt.savefig(f"{save_path_prefix}_efficiency.png", dpi=300)


def plot_connectivity(db, save_path_prefix):
    """
    Plots the number of outgoing tracks per frame.
    Requires a TrackingDB object with all_frames() and tracks(frame_id) methods.
    Applies downsampling for large datasets.
    """
    all_frames = list(db.all_frames())

    frame_id_to_num_outgoing_tracks = {fid: 0 for fid in all_frames}

    for frame_id in all_frames:
        if frame_id + 1 < db.frame_num():
            for track_id in db.tracks(frame_id):
                if (frame_id + 1) in db.frames(track_id):
                    frame_id_to_num_outgoing_tracks[frame_id] += 1

    plot_frames = sorted(
        [
            f
            for f in frame_id_to_num_outgoing_tracks.keys()
            if f < max(all_frames)
            if f < db.last_frameId
        ]
    )
    plot_outgoing_tracks = [frame_id_to_num_outgoing_tracks[f] for f in plot_frames]

    mean_num_outgoing_tracks = np.mean(plot_outgoing_tracks)
    fig, ax = plt.subplots(figsize=(12, 6), layout="constrained")
    ax.plot(
        plot_frames,
        plot_outgoing_tracks,
        label="Outgoing Tracks",
        marker=".",
        linestyle="-",
        markersize=3,
        alpha=0.7,
    )
    plt.axhline(
        mean_num_outgoing_tracks,
        color="red",
        linestyle="--",
        label=f"Mean: {mean_num_outgoing_tracks:.2f}",
    )
    setup_plot("Connectivity: Outgoing Tracks Per Frame", "Frame", "Number of Tracks")
    plt.legend()
    plt.savefig(f"{save_path_prefix}_connectivity.png", dpi=300)


def plot_track_length_histogram(db, save_path_prefix):
    """
    Plots a histogram of feature track lengths.
    Uses log scale on y-axis as requested.
    """
    all_tracks = db.all_tracks()
    track_lengths = [len(db.frames(track)) for track in all_tracks]

    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    min_len = min(track_lengths)
    max_len = max(track_lengths)
    bins = np.arange(min_len, max_len + 2) - 0.5

    ax.hist(
        track_lengths,
        bins=bins,
        edgecolor="black",
        alpha=0.7,
        align="mid",
    )
    setup_plot(
        "Track Length Histogram",
        "Track Length (number of Frames)",
        "Number of Tracks",
        log_y=True,
    )
    plt.savefig(f"{save_path_prefix}_track_length_histogram.png", dpi=300)


def plot_average_inliers(results, output_path):
    """
    Generates and saves a bar plot showing average inlier count and percentage.
    """
    detector_names = list(results.keys())

    avg_inlier_counts = {
        name: np.mean(data["inlier_counts"]) for name, data in results.items()
    }

    avg_inlier_percentages = {
        name: np.mean(data["inlier_percentages"]) for name, data in results.items()
    }

    print("Average Inlier Count and Percentage per Detector:")
    for name in detector_names:
        print(
            f"  {name}: {avg_inlier_counts[name]:.2f} inliers, {avg_inlier_percentages[name]:.2f}% inlier percentage"
        )

    fig1, ax1 = plt.subplots(1, 2, figsize=(16, 8), layout="constrained")

    avg_counts = [avg_inlier_counts[name] for name in detector_names]
    bars_counts = ax1[0].bar(
        detector_names, avg_counts, color=["skyblue", "lightcoral", "lightgreen"]
    )
    ax1[0].set_ylabel("Average Number of Inliers")
    ax1[0].set_xlabel("Detector")
    ax1[0].set_title("Average Inlier Count Across All Frames")
    ax1[0].grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars_counts:
        yval = bar.get_height()
        ax1[0].text(
            bar.get_x() + bar.get_width() / 2.0,
            yval,
            f"{yval:.0f}",
            va="bottom",
            ha="center",
            fontsize=10,
        )

    avg_percentages = [avg_inlier_percentages[name] for name in detector_names]
    bars_percentages = ax1[1].bar(
        detector_names, avg_percentages, color=["skyblue", "lightcoral", "lightgreen"]
    )
    ax1[1].set_ylabel("Average Inlier Percentage (%)")
    ax1[1].set_xlabel("Detector")
    ax1[1].set_title("Average Inlier Percentage Across All Frames")
    ax1[1].grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars_percentages:
        yval = bar.get_height()
        ax1[1].text(
            bar.get_x() + bar.get_width() / 2.0,
            yval,
            f"{yval:.2f}%",
            va="bottom",
            ha="center",
            fontsize=10,
        )

    plt.suptitle("Overall Detector Performance: Average Inliers", fontsize=16)
    plt.savefig(output_path, dpi=300)


def plot_spread_performance(results, first_frame, last_frame, output_path):
    """
    Generates and saves two separate plots for spread performance.
    """
    detector_names = list(results.keys())
    frame_indices = list(range(first_frame, last_frame + 1))

    fig1, ax1 = plt.subplots(figsize=(10, 7), layout="constrained")

    avg_spreads = {
        name: np.mean(data["spread_metrics"]) for name, data in results.items()
    }
    avg_spread_values = [avg_spreads[name] for name in detector_names]

    for detector_name, avg_spreading in avg_spreads.items():
        print(f"Average Keypoint Spread for {detector_name}: {avg_spreading:.2f}%")

    bars_spread = ax1.bar(
        detector_names, avg_spread_values, color=["skyblue", "lightcoral", "lightgreen"]
    )
    ax1.set_ylabel("Average Keypoint Spread (% of 20x20px Grid Cells Covered)")
    ax1.set_xlabel("Detector")
    ax1.set_title("Average Keypoint Spread Across All Frames")
    ax1.grid(axis="y", linestyle="--", alpha=0.7)
    ax1.set_ylim(0, 100)

    for bar in bars_spread:
        yval = bar.get_height()
        ax1.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval,
            f"{yval:.2f}%",
            va="bottom",
            ha="center",
            fontsize=10,
        )

    path_without_ext, ext = os.path.splitext(output_path)
    bar_chart_path = f"{path_without_ext}_avg_spread{ext}"
    plt.savefig(bar_chart_path, dpi=300)
    plt.close(fig1)

    fig2, ax2 = plt.subplots(figsize=(12, 7), layout="constrained")

    for name in detector_names:
        ax2.plot(
            frame_indices,
            results[name]["spread_metrics"],
            label=name,
            marker=".",
            markersize=4,
            alpha=0.8,
        )
    ax2.set_ylabel("Keypoint Spread (% of 20x20px Grid Cells Covered)")
    ax2.set_xlabel("Frame Index")
    ax2.set_title("Keypoint Spread Over Sequence")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.set_ylim(0, 100)

    line_chart_path = f"{path_without_ext}_spread_sequence{ext}"
    plt.savefig(line_chart_path, dpi=300)
    plt.close(fig2)


def plot_processing_time(results, first_frame, last_frame, output_path):
    """
    Generates and saves a bar plot showing total processing time for each detector.
    """
    detector_names = list(results.keys())
    total_times = [results[name]["times"] for name in detector_names]

    for name, t in zip(detector_names, total_times):
        print(f"Total Processing Time for {name}: {t:.2f} seconds")

    plt.figure(figsize=(10, 6), layout="constrained")
    bars_time = plt.bar(
        detector_names, total_times, color=["skyblue", "lightcoral", "lightgreen"]
    )
    plt.ylabel("Total Processing Time (seconds)")
    plt.xlabel("Detector")
    plt.title(f"Processing Time for {last_frame - first_frame + 1} Frames")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    for bar in bars_time:
        yval = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval,
            f"{yval:.2f}s",
            va="bottom",
            ha="center",
            fontsize=10,
        )
    plt.savefig(output_path, dpi=300)


def plot_detector_repeatability(
    detector_names, mean_repeatabilities, std_repeatabilities, output_path
):
    """
    Generates and saves a bar plot comparing the mean repeatability of different detectors,
    including standard deviation as error bars.

    Args:
        detector_names (list): A list of names for each detector (e.g., ["ORB", "AKAZE", "SIFT"]).
        mean_repeatabilities (list): A list of the mean repeatability scores for each detector.
        std_repeatabilities (list): A list of the standard deviations of repeatability for each detector.
        output_path (str): The path to save the generated plot.
    """
    plt.figure(figsize=(10, 6), layout="constrained")
    bars = plt.bar(
        detector_names,
        mean_repeatabilities,
        yerr=std_repeatabilities,
        capsize=5,
        color=["skyblue", "lightcoral", "lightgreen"],
    )

    plt.ylabel("Mean Repeatability Ratio (%)")
    plt.xlabel("Detector")
    plt.title("Detector Repeatability Comparison")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.ylim(0, 1)

    for i, bar in enumerate(bars):
        yval = bar.get_height()

        plt.text(
            bar.get_x() + bar.get_width() * 0.30,
            yval + 0.02,
            f"{yval:.3f}",
            va="center",
            ha="center",
            fontsize=10,
            color="black",
        )

    current_ylim = plt.gca().get_ylim()
    max_y_value_on_plot = max(mean_repeatabilities) + max(std_repeatabilities) + 0.03
    if max_y_value_on_plot > current_ylim[1]:
        plt.ylim(current_ylim[0], max_y_value_on_plot * 1.1)

    plt.savefig(output_path, dpi=300)
    plt.close()
