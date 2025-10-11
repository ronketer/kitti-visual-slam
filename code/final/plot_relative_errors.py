import numpy as np
import matplotlib.pyplot as plt
import pickle
import gtsam
import cv2
import sys
import os

# Add the code directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tqdm import tqdm
from consts import FINAL_PLOTS_RELATIVE_PATH, OUTPUT_RELATIVE_PATH, LASTFRAME
from utility import find_camera_location, read_calibration, parse_gt_line_matrix
from tracking_database import TrackingDB
from alg import create_pose_from_extrinsics



def calculate_rotation_error_degrees(pose1, pose2):
    """
    Calculate rotation error between two poses in degrees using Rodrigues representation.
    
    Args:
        pose1: gtsam.Pose3 - First pose
        pose2: gtsam.Pose3 - Second pose
    
    Returns:
        float: Rotation error in degrees
    """
    # Get relative rotation
    rel_pose = pose1.between(pose2)
    R = rel_pose.rotation().matrix()
    
    # Convert to Rodrigues representation
    rvec, _ = cv2.Rodrigues(R)
    angle_rad = np.linalg.norm(rvec)
    angle_deg = angle_rad * 180.0 / np.pi
    
    return angle_deg

def find_closest_keyframe(target_frame, available_keyframes):
    """Find the closest available keyframe to the target frame"""
    return min(available_keyframes, key=lambda x: abs(x - target_frame))

def calculate_total_distance(start_frame, end_frame, gt_matrices):
    """
    Calculate total distance traveled between two frames based on ground truth.
    
    Args:
        start_frame: Starting frame index
        end_frame: Ending frame index
        gt_matrices: List of ground truth matrices
    
    Returns:
        float: Total distance in meters
    """
    total_dist = 0.0
    for i in range(start_frame, end_frame):
        if i + 1 < len(gt_matrices):
            pose_i = create_pose_from_extrinsics(gt_matrices[i])
            pose_i_plus_1 = create_pose_from_extrinsics(gt_matrices[i + 1])
            rel_pose = pose_i.between(pose_i_plus_1)
            total_dist += np.linalg.norm(rel_pose.translation())
    return total_dist

def calculate_relative_pose_error(est_pose_start, est_pose_end, gt_pose_start, gt_pose_end):
    """
    Calculate relative pose error between estimated and ground truth poses.
    
    Args:
        est_pose_start: Estimated start pose (gtsam.Pose3)
        est_pose_end: Estimated end pose (gtsam.Pose3)
        gt_pose_start: Ground truth start pose (gtsam.Pose3)
        gt_pose_end: Ground truth end pose (gtsam.Pose3)
    
    Returns:
        tuple: (translation_error_norm, rotation_error_degrees)
    """
    # Calculate relative poses
    est_relative = est_pose_start.between(est_pose_end)
    gt_relative = gt_pose_start.between(gt_pose_end)
    
    # Calculate error between relative poses
    error_pose = est_relative.between(gt_relative)
    
    # Translation error (norm)
    translation_error = np.linalg.norm(error_pose.translation())
    
    # Rotation error (degrees)
    rotation_error = calculate_rotation_error_degrees(gtsam.Pose3(), error_pose)
    
    return translation_error, rotation_error

def extract_pnp_poses(db):
    """
    Extract PnP poses from TrackingDB.
    
    Args:
        db: TrackingDB instance
    
    Returns:
        dict: frame_id -> gtsam.Pose3 mapping
    """
    print("Extracting PnP poses from TrackingDB...")
    pnp_poses = {}
    
    for frame_id in tqdm(db.all_frames(), desc="Processing PnP poses"):
        abs_extrinsics = db.get_absolute_extrinsics(frame_id)
        if abs_extrinsics is not None:
            # Convert 4x4 extrinsics matrix to gtsam.Pose3
            pose = create_pose_from_extrinsics(abs_extrinsics)
            pnp_poses[frame_id] = pose
    
    print(f"Extracted {len(pnp_poses)} PnP poses")
    return pnp_poses

def extract_bundle_poses(windows_graph_list):
    """
    Extract Bundle Adjustment poses from bundle results using existing absolute poses.
    
    Args:
        windows_graph_list: List of bundle adjustment results
    
    Returns:
        dict: frame_id -> gtsam.Pose3 mapping
    """
    print("Extracting Bundle Adjustment poses...")
    bundle_poses = {}
    
    for window_dict in tqdm(windows_graph_list, desc="Processing bundle windows"):
        start_kf = window_dict["start_kf"]
        end_kf = window_dict["end_kf"]
        
        # Use the absolute poses directly from the windows graph list
        abs_start_pose = window_dict["abs_start_pose"]
        abs_end_pose = window_dict["abs_end_pose"]
        
        bundle_poses[start_kf] = abs_start_pose
        bundle_poses[end_kf] = abs_end_pose
    
    print(f"Extracted {len(bundle_poses)} Bundle Adjustment poses")
    return bundle_poses

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
        
def calculate_consecutive_relative_errors(estimated_poses, gt_matrices):
    """
    Calculate relative errors for consecutive keyframe pairs.
    
    Args:
        estimated_poses: dict of frame_id -> gtsam.Pose3
        gt_matrices: List of ground truth matrices
    
    Returns:
        tuple: (translation_errors, rotation_errors, frame_pairs, distances)
    """
    frame_ids = sorted(estimated_poses.keys())
    translation_errors = []
    rotation_errors = []
    frame_pairs = []
    distances = []
    
    for i in range(len(frame_ids) - 1):
        start_frame = frame_ids[i]
        end_frame = frame_ids[i + 1]
        
        # Get estimated poses
        est_start = estimated_poses[start_frame]
        est_end = estimated_poses[end_frame]
        
        # Get ground truth poses
        gt_start = create_pose_from_extrinsics(gt_matrices[start_frame])
        gt_end = create_pose_from_extrinsics(gt_matrices[end_frame])
        
        # Calculate relative pose error
        trans_error, rot_error = calculate_relative_pose_error(est_start, est_end, gt_start, gt_end)
        
        # Calculate total distance for normalization
        total_dist = calculate_total_distance(start_frame, end_frame, gt_matrices)
        
        if total_dist > 0:  # Avoid division by zero
            translation_errors.append((trans_error / total_dist) * 100)  # Convert to percentage
            rotation_errors.append(rot_error / total_dist)  # degrees per meter
            frame_pairs.append((start_frame, end_frame))
            distances.append(total_dist)
    
    return translation_errors, rotation_errors, frame_pairs, distances

def calculate_subsequence_relative_errors(estimated_poses, gt_matrices, sequence_length):
    """
    Calculate relative errors for subsequences of given length.
    
    Args:
        estimated_poses: dict of frame_id -> gtsam.Pose3
        gt_matrices: List of ground truth matrices
        sequence_length: Length of subsequences to analyze
    
    Returns:
        tuple: (translation_errors, rotation_errors, start_frames, distances)
    """
    available_keyframes = sorted(estimated_poses.keys())
    available_keyframes = available_keyframes[:len(available_keyframes)//2]
    translation_errors = []
    rotation_errors = []
    start_frames = []
    distances = []

    max_start_frame = available_keyframes[-1] - sequence_length

    for start_frame in tqdm(available_keyframes):
        if start_frame > max_start_frame:
            break

        end_frame = start_frame + sequence_length
        closest_end_kf = find_closest_keyframe(end_frame , available_keyframes)

        # Skip if keyframes are too close or the same
        if start_frame >= closest_end_kf:
            continue
        
        # Get estimated poses
        est_start = estimated_poses[start_frame]
        est_end = estimated_poses[closest_end_kf]
        
        # Get ground truth poses (use original frame indices)
        if start_frame < len(gt_matrices) and closest_end_kf < len(gt_matrices):
            gt_start = create_pose_from_extrinsics(gt_matrices[start_frame])
            gt_closest_end_kf = create_pose_from_extrinsics(gt_matrices[closest_end_kf])

            # Calculate relative pose error
            trans_error, rot_error = calculate_relative_pose_error(est_start, est_end, gt_start, gt_closest_end_kf)

            # Calculate total distance for normalization (use original frames)
            total_dist = calculate_total_distance(start_frame, closest_end_kf, gt_matrices)
            
            if total_dist > 0:  # Avoid division by zero
                translation_errors.append((trans_error / total_dist) * 100)  # Convert to percentage
                rotation_errors.append(rot_error / total_dist)  # degrees per meter
                start_frames.append(start_frame)
                distances.append(total_dist)
    
    return translation_errors, rotation_errors, start_frames, distances

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
        with open("dataset/poses/05.txt", "r") as f:
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
