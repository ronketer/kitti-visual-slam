from kitti_slam.evaluation.projection_errors import calculate_projection_error, extract_projection_errors_from_window
import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import os


from tqdm import tqdm
from reports.paths import FINAL_PLOTS_RELATIVE_PATH, OUTPUT_RELATIVE_PATH
from kitti_slam.gtsam_geometry import read_calibration



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
