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
    Extract Bundle Adjustment poses from bundle results.
    
    Args:
        windows_graph_list: List of bundle adjustment results
    
    Returns:
        dict: frame_id -> gtsam.Pose3 mapping
    """
    print("Extracting Bundle Adjustment poses...")
    bundle_poses = {}
    
    # Start with identity pose at frame 0
    bundle_poses[0] = gtsam.Pose3()
    current_pose = gtsam.Pose3()
    
    for window_dict in tqdm(windows_graph_list, desc="Processing bundle windows"):
        start_kf = window_dict["start_kf"]
        end_kf = window_dict["end_kf"]
        result = window_dict["result"]
        pose_keys = window_dict["pose_keys"]
        
        # Get relative pose from bundle result
        if start_kf in pose_keys and end_kf in pose_keys:
            relative_pose = result.atPose3(pose_keys[end_kf])
            
            # Update current pose (compose with relative pose)
            if start_kf in bundle_poses:
                current_pose = bundle_poses[start_kf]
            
            absolute_pose = current_pose.compose(relative_pose)
            bundle_poses[end_kf] = absolute_pose
    
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
    try:
        with open(loop_closure_save_path, "rb") as f:
            loop_closure_results = pickle.load(f)
        
        poses_without_loop_closure = loop_closure_results['poses_without_loop_closure']
        poses_with_loop_closure = loop_closure_results['poses_with_loop_closure']
        kf_pose_keys = loop_closure_results['kf_pose_keys']
        
        print(f"Loaded {len(poses_without_loop_closure)} poses without loop closure")
        print(f"Loaded {len(poses_with_loop_closure)} poses with loop closure")
        
        return poses_without_loop_closure, poses_with_loop_closure, kf_pose_keys
        
    except FileNotFoundError:
        print(f"Loop closure results file not found: {loop_closure_save_path}")
        print("Please run the loop closure optimization first (pose_graph_loop_closure.py)")
        return {}, {}, {}
    except Exception as e:
        print(f"Error loading loop closure results: {e}")
        return {}, {}, {}

def calculate_absolute_errors(estimated_poses, gt_matrices):
    """
    Calculate absolute translation and rotation errors for estimated poses.
    
    Args:
        estimated_poses: dict of frame_id -> gtsam.Pose3
        gt_matrices: List of ground truth matrices
    
    Returns:
        tuple: (translation_errors, rotation_errors, frame_ids)
    """
    frame_ids = sorted(estimated_poses.keys())
    translation_errors = {'x': [], 'y': [], 'z': [], 'norm': []}
    rotation_errors = []
    
    for frame_id in frame_ids:
        # Get estimated pose
        estimated_pose = estimated_poses[frame_id]
        estimated_translation = estimated_pose.translation()
        
        # Get ground truth pose
        gt_matrix = gt_matrices[frame_id]
        gt_pose = create_pose_from_extrinsics(gt_matrix)
        gt_translation = gt_pose.translation()
        
        # Calculate translation errors
        translation_error = estimated_translation - gt_translation
        translation_errors['x'].append(abs(translation_error[0]))
        translation_errors['y'].append(abs(translation_error[1]))
        translation_errors['z'].append(abs(translation_error[2]))
        translation_errors['norm'].append(np.linalg.norm(translation_error))
        
        # Calculate rotation error
        rotation_error = calculate_rotation_error_degrees(estimated_pose, gt_pose)
        rotation_errors.append(rotation_error)
    
    return translation_errors, rotation_errors, frame_ids

def plot_absolute_errors():
    """
    Plot absolute errors for all three methods: PnP, Bundle, and Pose Graph (with/without loop closure).
    """
    print("=== Absolute Error Analysis ===")
    
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
    poses_without_loop_closure, poses_with_loop_closure, kf_pose_keys = load_loop_closure_results()
    
    # Calculate absolute errors for each method
    print("Calculating absolute errors...")
    
    # Find common frame IDs across all methods for fair comparison
    common_frames = set(pnp_poses.keys()) & set(bundle_poses.keys()) & set(poses_with_loop_closure.keys())
    common_frames = sorted(list(common_frames))
    print(f"Common frames across all methods: {len(common_frames)}")
    
    # Filter poses to common frames
    pnp_poses_filtered = {fid: pnp_poses[fid] for fid in common_frames if fid in pnp_poses}
    bundle_poses_filtered = {fid: bundle_poses[fid] for fid in common_frames if fid in bundle_poses}
    poses_without_lc_filtered = {fid: poses_without_loop_closure[fid] for fid in common_frames if fid in poses_without_loop_closure}
    poses_with_lc_filtered = {fid: poses_with_loop_closure[fid] for fid in common_frames if fid in poses_with_loop_closure}
    
    # Calculate errors
    pnp_trans_errors, pnp_rot_errors, pnp_frame_ids = calculate_absolute_errors(pnp_poses_filtered, gt_matrices)
    bundle_trans_errors, bundle_rot_errors, bundle_frame_ids = calculate_absolute_errors(bundle_poses_filtered, gt_matrices)
    pg_without_lc_trans_errors, pg_without_lc_rot_errors, pg_without_lc_frame_ids = calculate_absolute_errors(poses_without_lc_filtered, gt_matrices)
    pg_with_lc_trans_errors, pg_with_lc_rot_errors, pg_with_lc_frame_ids = calculate_absolute_errors(poses_with_lc_filtered, gt_matrices)
    
    # Create comprehensive plots
    create_translation_error_plots(
        pnp_trans_errors, bundle_trans_errors, pg_without_lc_trans_errors, pg_with_lc_trans_errors,
        pnp_frame_ids, bundle_frame_ids, pg_without_lc_frame_ids, pg_with_lc_frame_ids
    )
    
    create_rotation_error_plots(
        pnp_rot_errors, bundle_rot_errors, pg_without_lc_rot_errors, pg_with_lc_rot_errors,
        pnp_frame_ids, bundle_frame_ids, pg_without_lc_frame_ids, pg_with_lc_frame_ids
    )
    
    # Print summary statistics
    print_summary_statistics(
        pnp_trans_errors, pnp_rot_errors,
        bundle_trans_errors, bundle_rot_errors,
        pg_without_lc_trans_errors, pg_without_lc_rot_errors,
        pg_with_lc_trans_errors, pg_with_lc_rot_errors
    )

def create_translation_error_plots(pnp_trans, bundle_trans, pg_without_lc_trans, pg_with_lc_trans,
                                   pnp_frames, bundle_frames, pg_without_lc_frames, pg_with_lc_frames):
    """
    Create translation error plots comparing all methods.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Absolute Translation Errors Comparison', fontsize=16, fontweight='bold')
    
    # X-axis errors
    axes[0, 0].plot(pnp_frames, pnp_trans['x'], 'g-', label='PnP', linewidth=2, alpha=0.8)
    axes[0, 0].plot(bundle_frames, bundle_trans['x'], 'r-', label='Bundle Adjustment', linewidth=2, alpha=0.8)
    # axes[0, 0].plot(pg_without_lc_frames, pg_without_lc_trans['x'], 'b-', label='Pose Graph (no loop closure)', linewidth=2, alpha=0.8)
    axes[0, 0].plot(pg_with_lc_frames, pg_with_lc_trans['x'], 'm-', label='Pose Graph (with loop closure)', linewidth=2, alpha=0.8)
    axes[0, 0].set_xlabel('Frame Index')
    axes[0, 0].set_ylabel('X Translation Error (m)')
    axes[0, 0].set_title('X-Axis Translation Error')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Y-axis errors
    axes[0, 1].plot(pnp_frames, pnp_trans['y'], 'g-', label='PnP', linewidth=2, alpha=0.8)
    axes[0, 1].plot(bundle_frames, bundle_trans['y'], 'r-', label='Bundle Adjustment', linewidth=2, alpha=0.8)
    # axes[0, 1].plot(pg_without_lc_frames, pg_without_lc_trans['y'], 'b-', label='Pose Graph (no loop closure)', linewidth=2, alpha=0.8)
    axes[0, 1].plot(pg_with_lc_frames, pg_with_lc_trans['y'], 'm-', label='Pose Graph (with loop closure)', linewidth=2, alpha=0.8)
    axes[0, 1].set_xlabel('Frame Index')
    axes[0, 1].set_ylabel('Y Translation Error (m)')
    axes[0, 1].set_title('Y-Axis Translation Error')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Z-axis errors
    axes[1, 0].plot(pnp_frames, pnp_trans['z'], 'g-', label='PnP', linewidth=2, alpha=0.8)
    axes[1, 0].plot(bundle_frames, bundle_trans['z'], 'r-', label='Bundle Adjustment', linewidth=2, alpha=0.8)
    # axes[1, 0].plot(pg_without_lc_frames, pg_without_lc_trans['z'], 'b-', label='Pose Graph (no loop closure)', linewidth=2, alpha=0.8)
    axes[1, 0].plot(pg_with_lc_frames, pg_with_lc_trans['z'], 'm-', label='Pose Graph (with loop closure)', linewidth=2, alpha=0.8)
    axes[1, 0].set_xlabel('Frame Index')
    axes[1, 0].set_ylabel('Z Translation Error (m)')
    axes[1, 0].set_title('Z-Axis Translation Error')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Total norm errors
    axes[1, 1].plot(pnp_frames, pnp_trans['norm'], 'g-', label='PnP', linewidth=2, alpha=0.8)
    axes[1, 1].plot(bundle_frames, bundle_trans['norm'], 'r-', label='Bundle Adjustment', linewidth=2, alpha=0.8)
    # axes[1, 1].plot(pg_without_lc_frames, pg_without_lc_trans['norm'], 'b-', label='Pose Graph (no loop closure)', linewidth=2, alpha=0.8)
    axes[1, 1].plot(pg_with_lc_frames, pg_with_lc_trans['norm'], 'm-', label='Pose Graph (with loop closure)', linewidth=2, alpha=0.8)
    axes[1, 1].set_xlabel('Frame Index')
    axes[1, 1].set_ylabel('Total Translation Error (m)')
    axes[1, 1].set_title('Total Translation Error Norm')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = FINAL_PLOTS_RELATIVE_PATH + "absolute_translation_errors.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Translation error plots saved to: {save_path}")
    plt.close()

def create_rotation_error_plots(pnp_rot, bundle_rot, pg_without_lc_rot, pg_with_lc_rot,
                                pnp_frames, bundle_frames, pg_without_lc_frames, pg_with_lc_frames):
    """
    Create rotation error plots comparing all methods.
    """
    fig, ax = plt.subplots(figsize=(12, 8))
    
    ax.plot(pnp_frames, pnp_rot, 'g-', label='PnP', linewidth=2, alpha=0.8)
    ax.plot(bundle_frames, bundle_rot, 'r-', label='Bundle Adjustment', linewidth=2, alpha=0.8)
    # ax.plot(pg_without_lc_frames, pg_without_lc_rot, 'b-', label='Pose Graph (no loop closure)', linewidth=2, alpha=0.8)
    ax.plot(pg_with_lc_frames, pg_with_lc_rot, 'm-', label='Pose Graph (with loop closure)', linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Frame Index', fontsize=12)
    ax.set_ylabel('Rotation Error (degrees)', fontsize=12)
    ax.set_title('Absolute Rotation Errors Comparison', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = FINAL_PLOTS_RELATIVE_PATH + "absolute_rotation_errors.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Rotation error plots saved to: {save_path}")
    plt.close()

def print_summary_statistics(pnp_trans, pnp_rot, bundle_trans, bundle_rot,
                             pg_without_lc_trans, pg_without_lc_rot, pg_with_lc_trans, pg_with_lc_rot):
    """
    Print summary statistics for all methods.
    """
    print("\n=== Summary Statistics ===")
    
    methods = {
        'PnP': (pnp_trans, pnp_rot),
        'Bundle Adjustment': (bundle_trans, bundle_rot),
        'Pose Graph (no loop closure)': (pg_without_lc_trans, pg_without_lc_rot),
        'Pose Graph (with loop closure)': (pg_with_lc_trans, pg_with_lc_rot)
    }
    
    for method_name, (trans_errors, rot_errors) in methods.items():
        print(f"\n{method_name}:")
        print(f"  Translation errors (m):")
        print(f"    X: mean={np.mean(trans_errors['x']):.4f}, std={np.std(trans_errors['x']):.4f}, max={np.max(trans_errors['x']):.4f}")
        print(f"    Y: mean={np.mean(trans_errors['y']):.4f}, std={np.std(trans_errors['y']):.4f}, max={np.max(trans_errors['y']):.4f}")
        print(f"    Z: mean={np.mean(trans_errors['z']):.4f}, std={np.std(trans_errors['z']):.4f}, max={np.max(trans_errors['z']):.4f}")
        print(f"    Norm: mean={np.mean(trans_errors['norm']):.4f}, std={np.std(trans_errors['norm']):.4f}, max={np.max(trans_errors['norm']):.4f}")
        print(f"  Rotation errors (deg):")
        print(f"    mean={np.mean(rot_errors):.4f}, std={np.std(rot_errors):.4f}, max={np.max(rot_errors):.4f}")

def main():
    """
    Main function to generate absolute error plots.
    """
    print("=== Absolute Error Analysis for SLAM-VN Project ===")
    plot_absolute_errors()
    print("=== Analysis Complete ===")

if __name__ == "__main__":
    main()
