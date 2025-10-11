import numpy as np
import matplotlib.pyplot as plt
import pickle
import gtsam
from collections import defaultdict
import sys
import os
# Add the code directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tqdm import tqdm
from consts import FINAL_PLOTS_RELATIVE_PATH, OUTPUT_RELATIVE_PATH
from utility import read_calibration
from tracking_database import TrackingDB
from alg import create_stereo_camera, create_pose_from_extrinsics

def calculate_projection_error(measurement, projection):
    """
    Calculate projection error between measurement and projection for stereo cameras.
    
    Args:
        measurement: gtsam.StereoPoint2 - actual observed stereo point
        projection: gtsam.StereoPoint2 - projected stereo point
    
    Returns:
        float: Combined projection error in pixels
    """
    # Calculate left camera error
    left_error = np.linalg.norm(
        np.array([measurement.uL(), measurement.v()]) - 
        np.array([projection.uL(), projection.v()])
    )
    
    # Calculate right camera error
    right_error = np.linalg.norm(
        np.array([measurement.uR(), measurement.v()]) - 
        np.array([projection.uR(), projection.v()])
    )
    
    # Return average of left and right errors
    return (left_error + right_error) / 2.0

def analyze_pnp_projection_errors(db, K):
    """
    Analyze projection errors for PnP estimation as a function of distance from triangulation frame.
    
    Args:
        db: TrackingDB instance
        K: Camera calibration matrix
    
    Returns:
        dict: Distance -> list of projection errors
    """
    print("Analyzing PnP projection errors...")
    
    distance_to_errors = defaultdict(list)
    processed_tracks = 0
    
    # Get all tracks with sufficient length
    all_tracks = db.all_tracks()
    valid_tracks = [tid for tid in all_tracks if len(db.frames(tid)) >= 3]
    
    for track_id in tqdm(valid_tracks, desc="Processing PnP tracks"):
        frame_ids = db.frames(track_id)
        triangulation_frame = frame_ids[-1]  # Last frame is reference for PnP
        
        # Get triangulation frame pose and triangulate 3D point
        triangulation_extrinsics = db.get_absolute_extrinsics(triangulation_frame)
        if triangulation_extrinsics is None:
            raise ValueError(f"Missing extrinsics for triangulation frame {triangulation_frame}")
            
        triangulation_link = db.link(triangulation_frame, track_id)
        if triangulation_link is None:
            raise ValueError(f"Missing link for triangulation frame {triangulation_frame} and track {track_id}")

        # Create stereo point and triangulate
        stereo_point = gtsam.StereoPoint2(
            triangulation_link.left_keypoint()[0],
            triangulation_link.right_keypoint()[0], 
            triangulation_link.left_keypoint()[1]
        )
        triangulation_camera = create_stereo_camera(triangulation_extrinsics, K)
        point3d = triangulation_camera.backproject(stereo_point)
        
        # Calculate projection errors for all frames in track
        for frame_id in frame_ids:
            distance_from_reference = abs(triangulation_frame - frame_id)
            
            # Get frame pose and link
            frame_extrinsics = db.get_absolute_extrinsics(frame_id)
            if frame_extrinsics is None:
                raise ValueError(f"Missing extrinsics for frame {frame_id}")

                
            frame_link = db.link(frame_id, track_id)
            if frame_link is None:
                raise ValueError(f"Missing link for frame {frame_id} and track {track_id}")

            # Project 3D point using PnP pose
            frame_camera = create_stereo_camera(frame_extrinsics, K)
            projected_point = frame_camera.project(point3d)
            
            # Calculate projection error
            measurement = gtsam.StereoPoint2(
                frame_link.left_keypoint()[0],
                frame_link.right_keypoint()[0],
                frame_link.left_keypoint()[1]
            )
            
            projection_error = calculate_projection_error(measurement, projected_point)
            distance_to_errors[distance_from_reference].append(projection_error)
        
        processed_tracks += 1
            
    
    print(f"Processed {processed_tracks} tracks for PnP analysis")
    return distance_to_errors

def analyze_bundle_projection_errors(windows_graph_list, K):
    """
    Analyze projection errors for Bundle Adjustment as a function of distance from first frame.
    
    Args:
        windows_graph_list: List of bundle adjustment results
        K: Camera calibration matrix
    
    Returns:
        tuple: (initial_distance_to_errors, optimized_distance_to_errors)
    """
    print("Analyzing Bundle Adjustment projection errors...")
    
    initial_distance_to_errors = defaultdict(list)
    optimized_distance_to_errors = defaultdict(list)
    
    for window_dict in tqdm(windows_graph_list, desc="Processing bundle windows"):
        start_kf = window_dict["start_kf"]
        graph = window_dict["graph"]
        initial_estimate = window_dict["initialEstimate"]
        result = window_dict["result"]
        pose_keys = window_dict["pose_keys"]
        point_keys = window_dict["point_keys"]
        
        # Process each stereo factor in the graph
        for i in range(graph.size()):
            factor = graph.at(i)
            
            # Check if this is a stereo factor
            if factor.__class__.__name__ != 'GenericStereoFactor3D':
                continue
            
            # Get factor keys
            keys = factor.keys()
            if len(keys) != 2:
                continue
                
            pose_key = keys[0]
            point_key = keys[1]
            
            # Find frame ID from pose key
            frame_id = None
            for fid, key in pose_keys.items():
                if key == pose_key:
                    frame_id = fid
                    break
            
            if frame_id is None:
                raise ValueError(f"Could not find frame ID for pose key {pose_key}")
            
            # Calculate distance from reference frame (first frame of bundle)
            distance_from_reference = abs(frame_id - start_kf)
            
            # Get measurement
            measurement = factor.measured()
            
            # Calculate initial projection error
            if initial_estimate.exists(pose_key) and initial_estimate.exists(point_key):
                pose_init = initial_estimate.atPose3(pose_key)
                point_init = initial_estimate.atPoint3(point_key)
                stereo_cam_init = gtsam.StereoCamera(pose_init, K)
                projection_init = stereo_cam_init.project(point_init)
                
                initial_error = calculate_projection_error(measurement, projection_init)
                initial_distance_to_errors[distance_from_reference].append(initial_error)
            
            # Calculate optimized projection error
            if result.exists(pose_key) and result.exists(point_key):
                pose_opt = result.atPose3(pose_key)
                point_opt = result.atPoint3(point_key)
                stereo_cam_opt = gtsam.StereoCamera(pose_opt, K)
                projection_opt = stereo_cam_opt.project(point_opt)
                
                optimized_error = calculate_projection_error(measurement, projection_opt)
                optimized_distance_to_errors[distance_from_reference].append(optimized_error)
                    
    
    return initial_distance_to_errors, optimized_distance_to_errors

def plot_projection_vs_distance():
    """
    Plot median projection errors as a function of distance from reference frame.
    """
    print("Loading data...")
    
    # Load tracking database
    db = TrackingDB()
    OUTPUT_PATH = OUTPUT_RELATIVE_PATH + "tracking_with_geometric_validation_without_far_tracks"
    try:
        db.load(OUTPUT_PATH)
        print("Loaded tracking database")
    except Exception as e:
        print(f"Error loading tracking database: {e}")
        return
    
    # Load calibration
    try:
        K = read_calibration()
        print("Loaded camera calibration")
    except Exception as e:
        print(f"Error loading calibration: {e}")
        return
    
    # Load bundle adjustment results
    try:
        with open(OUTPUT_RELATIVE_PATH + "ba_results.pkl", "rb") as f:
            windows_graph_list = pickle.load(f)
        print(f"Loaded {len(windows_graph_list)} bundle windows")
    except Exception as e:
        print(f"Error loading bundle results: {e}")
        return
    
    # Analyze PnP projection errors
    pnp_distance_to_errors = analyze_pnp_projection_errors(db, K)
    
    # Analyze Bundle Adjustment projection errors
    bundle_init_distance_to_errors, bundle_opt_distance_to_errors = analyze_bundle_projection_errors(windows_graph_list, K)
    
    # Calculate mean and std for each distance
    def calculate_stats(distance_to_errors):
        """
        Calculate mean and std for each distance in the error dictionary.
        
        Args:
            distance_to_errors: dict - Distance -> list of errors
        Returns:
            tuple: (distances, means, stds)
        """
        distances = []
        means = []
        stds = []
        for distance in sorted(distance_to_errors.keys()):
            errors = distance_to_errors[distance]
            if len(errors) > 0:
                distances.append(distance)
                means.append(np.mean(errors))
                stds.append(np.std(errors)/2) # Use half std for error bars
        return distances, means, stds
    
    pnp_distances, pnp_means, pnp_stds = calculate_stats(pnp_distance_to_errors)
    bundle_init_distances, bundle_init_means, bundle_init_stds = calculate_stats(bundle_init_distance_to_errors)
    bundle_opt_distances, bundle_opt_means, bundle_opt_stds = calculate_stats(bundle_opt_distance_to_errors)
    
    # Create the figure with two subplots (vertical stack)
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 10))

    # Upper subplot: PnP
    ax1.errorbar(pnp_distances, pnp_means, yerr=pnp_stds, fmt='go-', 
                 label='PnP Estimation', 
                 linewidth=2, markersize=4, alpha=0.8, capsize=3)
    ax1.set_xlabel('Distance from Reference Frame (frames)', fontsize=12)
    ax1.set_ylabel('Mean Projection Error (pixels)', fontsize=12)
    ax1.set_title('PnP Mean Projection Error vs Distance', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(bottom=1e-2)

    # Lower subplot: Bundle (initial and optimized)
    ax2.errorbar(bundle_init_distances, bundle_init_means, yerr=bundle_init_stds, fmt='rs-', 
                 label='Bundle Initial (Before Optimization)', 
                 linewidth=2, markersize=4, alpha=0.8, capsize=3)
    ax2.set_xlabel('Distance from Reference Frame (frames)', fontsize=12)
    ax2.set_ylabel('Mean Projection Error (pixels)', fontsize=12)
    ax2.set_title('Bundle Adjustment Initial Mean Projection Error vs Distance', fontsize=13, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=1e-2)

    ax3.errorbar(bundle_opt_distances, bundle_opt_means, yerr=bundle_opt_stds, fmt='b^-', 
                label='Bundle Optimized (After Optimization)', 
                linewidth=2, markersize=4, alpha=0.8, capsize=3)
    ax3.set_xlabel('Distance from Reference Frame (frames)', fontsize=12)
    ax3.set_ylabel('Mean Projection Error (pixels)', fontsize=12)
    ax3.set_title('Bundle Adjustment Optimized Mean Projection Error vs Distance', fontsize=13, fontweight='bold')
    ax3.legend(fontsize=11)
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(bottom=1e-2)

    # Adjust layout
    plt.tight_layout()

    # Save the plot
    save_path = FINAL_PLOTS_RELATIVE_PATH + "projection_vs_distance.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Plot saved to: {save_path}")

    plt.close()
    
    # Print summary statistics
    print("\n--- Summary Statistics ---")
    
    if pnp_distances:
        print(f"PnP Analysis:")
        print(f"  Distance range: 0 to {max(pnp_distances)} frames")
        print(f"  Min mean error: {min(pnp_means):.4f} pixels")
        print(f"  Max mean error: {max(pnp_means):.4f} pixels")
        print(f"  Data points: {len(pnp_distances)}")
    
    if bundle_init_distances:
        print(f"Bundle Initial Analysis:")
        print(f"  Distance range: 0 to {max(bundle_init_distances)} frames")
        print(f"  Min mean error: {min(bundle_init_means):.4f} pixels")
        print(f"  Max mean error: {max(bundle_init_means):.4f} pixels")
        print(f"  Data points: {len(bundle_init_distances)}")
    
    if bundle_opt_distances:
        print(f"Bundle Optimized Analysis:")
        print(f"  Distance range: 0 to {max(bundle_opt_distances)} frames")
        print(f"  Min mean error: {min(bundle_opt_means):.4f} pixels")
        print(f"  Max mean error: {max(bundle_opt_means):.4f} pixels")
        print(f"  Data points: {len(bundle_opt_distances)}")

def main():
    """
    Main function to generate the projection vs distance analysis.
    """
    print("=== Projection Error vs Distance Analysis ===")
    plot_projection_vs_distance()
    print("=== Analysis Complete ===")

if __name__ == "__main__":
    main()
