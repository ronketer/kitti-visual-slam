import numpy as np
import matplotlib.pyplot as plt
import pickle
import gtsam
import sys
import os

# Add the code directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from tqdm import tqdm
from consts import FINAL_PLOTS_RELATIVE_PATH, OUTPUT_RELATIVE_PATH
from utility import read_calibration

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

def extract_projection_errors_from_window(graph, initial_estimate, result, pose_keys, point_keys, K):
    """
    Extract all projection errors from a bundle window.
    
    Args:
        graph: gtsam.NonlinearFactorGraph
        initial_estimate: gtsam.Values
        result: gtsam.Values  
        pose_keys: dict mapping frame_id to pose keys
        point_keys: dict mapping track_id to point keys
        K: gtsam.Cal3_S2Stereo calibration matrix
    
    Returns:
        tuple: (initial_errors, optimized_errors) - lists of projection errors
    """
    initial_errors = []
    optimized_errors = []
    
    # Iterate through all factors in the graph
    for i in range(graph.size()):
        factor = graph.at(i)
        
        # Check if this is a stereo factor (GenericStereoFactor3D)
        if factor.__class__.__name__ != 'GenericStereoFactor3D':
            continue
            
        # Get the keys this factor connects
        keys = factor.keys()
        if len(keys) != 2:
            continue
            
        pose_key = keys[0]
        point_key = keys[1]
        
        # Get measurement
        measurement = factor.measured()
        
        # Calculate initial projection error
        if initial_estimate.exists(pose_key) and initial_estimate.exists(point_key):
            pose_init = initial_estimate.atPose3(pose_key)
            point_init = initial_estimate.atPoint3(point_key)
            stereo_cam_init = gtsam.StereoCamera(pose_init, K)
            projection_init = stereo_cam_init.project(point_init)
            
            initial_error = calculate_projection_error(measurement, projection_init)
            initial_errors.append(initial_error)
        
        # Calculate optimized projection error
        if result.exists(pose_key) and result.exists(point_key):
            pose_opt = result.atPose3(pose_key)
            point_opt = result.atPoint3(point_key)
            stereo_cam_opt = gtsam.StereoCamera(pose_opt, K)
            projection_opt = stereo_cam_opt.project(point_opt)
            
            optimized_error = calculate_projection_error(measurement, projection_opt)
            optimized_errors.append(optimized_error)
                
    
    return initial_errors, optimized_errors

def plot_median_projection_errors():
    """
    Plot median projection errors for each bundle window.
    
    Creates a plot showing the median projection error (in pixels) for each
    bundle window, with one line for initial error (before optimization) and one
    for optimized error (after optimization).
    
    X-axis: Keyframes (starting keyframe of each bundle window)
    Y-axis: Median projection error (pixels)
    """
    print("Loading bundle adjustment results...")
    
    # Load BA results
    with open(OUTPUT_RELATIVE_PATH + "ba_results.pkl", "rb") as f:
        windows_graph_list = pickle.load(f)
    print(f"Loaded {len(windows_graph_list)} bundle windows")

    
    # Load calibration matrix
    K = read_calibration()
    print("Loaded camera calibration")
    if K is None:
        print("Error: Calibration matrix not found. Ensure calibration file exists.")
        return
    # Extract data for each window
    keyframes = []
    initial_median_errors = []
    optimized_median_errors = []
    
    print("Calculating median projection errors for each window...")

    for window_dict in tqdm(windows_graph_list):
        start_kf = window_dict["start_kf"]
        end_kf = window_dict["end_kf"]
        graph = window_dict["graph"]
        initial_estimate = window_dict["initialEstimate"]
        result = window_dict["result"]
        pose_keys = window_dict["pose_keys"]
        point_keys = window_dict["point_keys"]
        
        # Extract projection errors for this window
        initial_errors, optimized_errors = extract_projection_errors_from_window(
            graph, initial_estimate, result, pose_keys, point_keys, K
        )
        
        # Calculate median errors (skip if no valid errors found)
        if len(initial_errors) > 0 and len(optimized_errors) > 0:
            initial_median = np.median(initial_errors)
            optimized_median = np.median(optimized_errors)
            
            keyframes.append(start_kf)
            initial_median_errors.append(initial_median)
            optimized_median_errors.append(optimized_median)

    if not keyframes:
        print("Error: No valid projection data found in bundle adjustment results")
        return
    
    print(f"Successfully processed {len(keyframes)} windows")
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    # Plot both lines
    plt.plot(keyframes, initial_median_errors, 'r-', 
             label='Initial Error (Before Optimization)', 
             linewidth=2, marker='o', markersize=3, alpha=0.7)
    plt.plot(keyframes, optimized_median_errors, 'b-', 
             label='Optimized Error (After Optimization)', 
             linewidth=2, marker='s', markersize=3, alpha=0.7)
    
    # Formatting
    plt.xlabel('Keyframes', fontsize=12)
    plt.ylabel('Median Projection Error (pixels)', fontsize=12)
    plt.title('Median Projection Error vs Keyframes', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    # Set y-axis to start from 0 for better visualization
    plt.ylim(bottom=0)
    
    # Save the plot
    save_path = FINAL_PLOTS_RELATIVE_PATH + "median_projection_errors.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Plot saved to: {save_path}")
    
    plt.close()
    
    # Print summary statistics
    print("\n--- Summary Statistics ---")
    print(f"Number of bundle windows: {len(keyframes)}")
    print(f"Keyframe range: {min(keyframes)} to {max(keyframes)}")
    print(f"Max initial error: {max(initial_median_errors):.6f} pixels")
    print(f"Max optimized error: {max(optimized_median_errors):.6f} pixels")
    print(f"Min initial error: {min(initial_median_errors):.6f} pixels")
    print(f"Min optimized error: {min(optimized_median_errors):.6f} pixels")

def main():
    """
    Main function to generate the median projection error plot.
    """
    print("=== Bundle Adjustment Median Projection Error Analysis ===")
    plot_median_projection_errors()
    print("=== Analysis Complete ===")

if __name__ == "__main__":
    main()
