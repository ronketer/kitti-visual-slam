from reports.paths import GT_POSES_FILE
import numpy as np
import matplotlib.pyplot as plt
import pickle
import cv2
import sys
import os


from kitti_slam.evaluation.trajectory_errors import (
    calculate_rotation_error_degrees, find_closest_keyframe, calculate_total_distance, calculate_relative_pose_error, calculate_consecutive_relative_errors, calculate_subsequence_relative_errors
)
from kitti_slam.trajectory import extract_pnp_poses, extract_bundle_poses

from tqdm import tqdm
from reports.paths import FINAL_PLOTS_RELATIVE_PATH, OUTPUT_RELATIVE_PATH, LASTFRAME
from kitti_slam.geometry import find_camera_location
from kitti_slam.gtsam_geometry import read_calibration
from kitti_slam.dataset import parse_gt_line_matrix
from kitti_slam.tracking_database import TrackingDB
from kitti_slam.gtsam_geometry import create_pose_from_extrinsics



def load_loop_closure_results():
    """
    Load loop closure results from pickle file.

    Returns:
        tuple: (poses_without_loop_closure, poses_with_loop_closure, kf_pose_keys)
    """
    print("Loading loop closure results...")

    loop_closure_save_path = OUTPUT_RELATIVE_PATH + "loop_closure_results.pkl"

    with open(loop_closure_save_path, "rb") as f:
        loop_closure_results = pickle.load(f)

    poses_without_loop_closure = loop_closure_results['poses_without_loop_closure']
    poses_with_loop_closure = loop_closure_results['poses_with_loop_closure']
    kf_pose_keys = loop_closure_results['kf_pose_keys']

    print(f"Loaded {len(poses_without_loop_closure)} poses without loop closure")
    print(f"Loaded {len(poses_with_loop_closure)} poses with loop closure")

    return poses_without_loop_closure, poses_with_loop_closure, kf_pose_keys

def plot_consecutive_relative_errors(pnp_data, bundle_data, pg_data, save_path):
    """
    Plot consecutive relative errors for all methods, with average errors annotated.

    Args:
        pnp_data: tuple of (translation_errors, rotation_errors, frame_pairs, distances)
        bundle_data: tuple of (translation_errors, rotation_errors, frame_pairs, distances)
        pg_data: tuple of (translation_errors, rotation_errors, frame_pairs, distances)
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Translation errors
    if pnp_data[0]:
        start_frames = [pair[0] for pair in pnp_data[2]]
        ax1.plot(start_frames, pnp_data[0], 'g-', label='PnP', linewidth=2, alpha=0.8)
    if bundle_data[0]:
        start_frames = [pair[0] for pair in bundle_data[2]]
        ax1.plot(start_frames, bundle_data[0], 'r-', label='Bundle Adjustment', linewidth=2, alpha=0.8)
    if pg_data[0]:
        start_frames = [pair[0] for pair in pg_data[2]]
        ax1.plot(start_frames, pg_data[0], 'm-', label='Pose Graph (with loop closure)', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Start Frame Index')
    ax1.set_xticks(np.arange(0, max(start_frames) + 1, 100))
    ax1.set_ylabel('Translation Error (%)')
    ax1.set_title('Consecutive Keyframe Relative Translation Errors')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    # Annotate average translation errors
    avg_text = ""
    if pnp_data[0]:
        avg_text += f"PnP avg: {np.mean(pnp_data[0]):.2f}%\n"
    if bundle_data[0]:
        avg_text += f"Bundle avg: {np.mean(bundle_data[0]):.2f}%\n"
    if pg_data[0]:
        avg_text += f"PoseGraph avg: {np.mean(pg_data[0]):.2f}%"
    ax1.text(0.01, 0.98, avg_text, transform=ax1.transAxes, fontsize=11, verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Rotation errors
    if pnp_data[1]:
        start_frames = [pair[0] for pair in pnp_data[2]]
        ax2.plot(start_frames, pnp_data[1], 'g-', label='PnP', linewidth=2, alpha=0.8)
    if bundle_data[1]:
        start_frames = [pair[0] for pair in bundle_data[2]]
        ax2.plot(start_frames, bundle_data[1], 'r-', label='Bundle Adjustment', linewidth=2, alpha=0.8)
    if pg_data[1]:
        start_frames = [pair[0] for pair in pg_data[2]]
        ax2.plot(start_frames, pg_data[1], 'm-', label='Pose Graph (with loop closure)', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Start Frame Index')
    ax2.set_ylabel('Rotation Error (deg/m)')
    ax2.set_title('Consecutive Keyframe Relative Rotation Errors')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    # Annotate average rotation errors
    avg_text_rot = ""
    if pnp_data[1]:
        avg_text_rot += f"PnP avg: {np.mean(pnp_data[1]):.3f} deg/m\n"
    if bundle_data[1]:
        avg_text_rot += f"Bundle avg: {np.mean(bundle_data[1]):.3f} deg/m\n"
    if pg_data[1]:
        avg_text_rot += f"PoseGraph avg: {np.mean(pg_data[1]):.3f} deg/m"
    ax2.text(0.01, 0.98, avg_text_rot, transform=ax2.transAxes, fontsize=11, verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Consecutive relative errors plot saved to: {save_path}")
    plt.close()

def plot_subsequence_relative_errors(pnp_data, bundle_data, pg_data, sequence_length, save_path):
    """
    Plot subsequence relative errors for all methods, with average errors annotated.

    Args:
        pnp_data: tuple of (translation_errors, rotation_errors, start_frames, distances)
        bundle_data: tuple of (translation_errors, rotation_errors, start_frames, distances)
        pg_data: tuple of (translation_errors, rotation_errors, start_frames, distances)
        sequence_length: Length of subsequences
        save_path: Path to save the plot
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

    # Translation errors
    if pnp_data[0]:
        ax1.plot(pnp_data[2], pnp_data[0], 'g-', label='PnP', linewidth=2, alpha=0.8)
    if bundle_data[0]:
        ax1.plot(bundle_data[2], bundle_data[0], 'r-', label='Bundle Adjustment', linewidth=2, alpha=0.8)
    if pg_data[0]:
        ax1.plot(pg_data[2], pg_data[0], 'm-', label='Pose Graph (with loop closure)', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Start Frame Index')
    ax1.set_ylabel('Translation Error (%)')
    ax1.set_title(f'Subsequence Relative Translation Errors (Length: {sequence_length})')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    # Annotate average translation errors
    avg_text = ""
    if pnp_data[0]:
        avg_text += f"PnP avg: {np.mean(pnp_data[0]):.2f}%\n"
    if bundle_data[0]:
        avg_text += f"Bundle avg: {np.mean(bundle_data[0]):.2f}%\n"
    if pg_data[0]:
        avg_text += f"PoseGraph avg: {np.mean(pg_data[0]):.2f}%"
    ax1.text(0.01, 0.98, avg_text, transform=ax1.transAxes, fontsize=11, verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    # Rotation errors
    if pnp_data[1]:
        ax2.plot(pnp_data[2], pnp_data[1], 'g-', label='PnP', linewidth=2, alpha=0.8)
    if bundle_data[1]:
        ax2.plot(bundle_data[2], bundle_data[1], 'r-', label='Bundle Adjustment', linewidth=2, alpha=0.8)
    if pg_data[1]:
        ax2.plot(pg_data[2], pg_data[1], 'm-', label='Pose Graph (with loop closure)', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Start Frame Index')
    ax2.set_ylabel('Rotation Error (deg/m)')
    ax2.set_title(f'Subsequence Relative Rotation Errors (Length: {sequence_length})')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    # Annotate average rotation errors
    avg_text_rot = ""
    if pnp_data[1]:
        avg_text_rot += f"PnP avg: {np.mean(pnp_data[1]):.3f} deg/m\n"
    if bundle_data[1]:
        avg_text_rot += f"Bundle avg: {np.mean(bundle_data[1]):.3f} deg/m\n"
    if pg_data[1]:
        avg_text_rot += f"PoseGraph avg: {np.mean(pg_data[1]):.3f} deg/m"
    ax2.text(0.01, 0.98, avg_text_rot, transform=ax2.transAxes, fontsize=11, verticalalignment='top', horizontalalignment='left', bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Subsequence relative errors plot (length {sequence_length}) saved to: {save_path}")
    plt.close()

def print_relative_error_statistics(method_name, trans_errors, rot_errors):
    """
    Print summary statistics for relative errors.

    Args:
        method_name: Name of the method
        trans_errors: List of translation errors
        rot_errors: List of rotation errors
    """
    if not trans_errors or not rot_errors:
        print(f"{method_name}: No data available")
        return

    print(f"\n{method_name}:")
    print(f"  Translation errors (%):")
    print(f"    mean={np.mean(trans_errors):.4f}, std={np.std(trans_errors):.4f}, max={np.max(trans_errors):.4f}")
    print(f"  Rotation errors (deg/m):")
    print(f"    mean={np.mean(rot_errors):.4f}, std={np.std(rot_errors):.4f}, max={np.max(rot_errors):.4f}")

def plot_relative_errors():
    """
    Main function to plot relative errors for all methods.
    """
    print("=== Relative Error Analysis ===")

    # Load data
    print("Loading data...")

    # Load tracking database for PnP poses
    db = TrackingDB()
    OUTPUT_PATH = OUTPUT_RELATIVE_PATH + "tracking_with_geometric_validation_without_far_tracks"
    try:
        db.load(OUTPUT_PATH)
        print("Loaded tracking database")
    except Exception as e:
        print(f"Error loading tracking database: {e}")
        return

    # Load bundle adjustment results
    try:
        with open(OUTPUT_RELATIVE_PATH + "ba_results.pkl", "rb") as f:
            windows_graph_list = pickle.load(f)
        print(f"Loaded {len(windows_graph_list)} bundle windows")
    except Exception as e:
        print(f"Error loading bundle results: {e}")
        return

    # Load ground truth
    try:
        with open(GT_POSES_FILE, "r") as f:
            gt_matrices = [parse_gt_line_matrix(line) for line in f.readlines()[:LASTFRAME + 1]]
        print(f"Loaded {len(gt_matrices)} ground truth poses")
    except Exception as e:
        print(f"Error loading ground truth: {e}")
        return

    # Extract poses from different methods
    pnp_poses = extract_pnp_poses(db)
    bundle_poses = extract_bundle_poses(windows_graph_list)

    # Load loop closure results
    _, poses_with_loop_closure, _ = load_loop_closure_results()

    # Calculate consecutive relative errors
    print("\nCalculating consecutive relative errors...")
    pnp_consecutive = calculate_consecutive_relative_errors(pnp_poses, gt_matrices)
    bundle_consecutive = calculate_consecutive_relative_errors(bundle_poses, gt_matrices)
    pg_consecutive = calculate_consecutive_relative_errors(poses_with_loop_closure, gt_matrices)

    # Plot consecutive relative errors
    plot_consecutive_relative_errors(
        pnp_consecutive, bundle_consecutive, pg_consecutive,
        FINAL_PLOTS_RELATIVE_PATH + "relative_errors_consecutive.png"
    )
    # pnp is tupple of (translation_errors, rotation_errors, frame_pairs, distances), take half of each list
    half_pnp_consecutive = (
        pnp_consecutive[0][:len(pnp_consecutive[0]) // 2],
        pnp_consecutive[1][:len(pnp_consecutive[1]) // 2],
        pnp_consecutive[2][:len(pnp_consecutive[2]) // 2],
        pnp_consecutive[3][:len(pnp_consecutive[3]) // 2]
    )
    half_bundle_consecutive = (
        bundle_consecutive[0][:len(bundle_consecutive[0]) // 2],
        bundle_consecutive[1][:len(bundle_consecutive[1]) // 2],
        bundle_consecutive[2][:len(bundle_consecutive[2]) // 2],
        bundle_consecutive[3][:len(bundle_consecutive[3]) // 2]
    )
    half_pg_consecutive = (
        pg_consecutive[0][:len(pg_consecutive[0]) // 2],
        pg_consecutive[1][:len(pg_consecutive[1]) // 2],
        pg_consecutive[2][:len(pg_consecutive[2]) // 2],
        pg_consecutive[3][:len(pg_consecutive[3]) // 2]
    )
    # Plot consecutive relative errors with half of the data
    plot_consecutive_relative_errors(
        half_pnp_consecutive, half_bundle_consecutive, half_pg_consecutive,
        FINAL_PLOTS_RELATIVE_PATH + "relative_errors_consecutive_half.png"
    )

    # Calculate and plot subsequence relative errors
    sequence_lengths = [100, 400, 800]

    for seq_length in sequence_lengths:
        print(f"\nCalculating subsequence relative errors (length {seq_length})...")

        pnp_subseq = calculate_subsequence_relative_errors(pnp_poses, gt_matrices, seq_length)
        bundle_subseq = calculate_subsequence_relative_errors(bundle_poses, gt_matrices, seq_length)
        pg_subseq = calculate_subsequence_relative_errors(poses_with_loop_closure, gt_matrices, seq_length)

        # Plot subsequence relative errors
        plot_subsequence_relative_errors(
            pnp_subseq, bundle_subseq, pg_subseq, seq_length,
            FINAL_PLOTS_RELATIVE_PATH + f"relative_errors_subsequence_{seq_length}.png"
        )

        # Print statistics
        print(f"\n=== Summary Statistics for Subsequence Length {seq_length} ===")
        print_relative_error_statistics("PnP", pnp_subseq[0], pnp_subseq[1])
        print_relative_error_statistics("Bundle Adjustment", bundle_subseq[0], bundle_subseq[1])
        print_relative_error_statistics("Pose Graph (with loop closure)", pg_subseq[0], pg_subseq[1])

    # Print consecutive errors statistics
    print("\n=== Summary Statistics for Consecutive Keyframes ===")
    print_relative_error_statistics("PnP", pnp_consecutive[0], pnp_consecutive[1])
    print_relative_error_statistics("Bundle Adjustment", bundle_consecutive[0], bundle_consecutive[1])
    print_relative_error_statistics("Pose Graph (with loop closure)", pg_consecutive[0], pg_consecutive[1])

def main():
    """
    Main function to generate relative error plots.
    """
    print("=== Relative Error Analysis for SLAM-VN Project ===")
    plot_relative_errors()
    print("=== Analysis Complete ===")

if __name__ == "__main__":
    main()
