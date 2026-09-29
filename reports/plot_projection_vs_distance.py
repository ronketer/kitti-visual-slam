from kitti_slam.evaluation.projection_errors import calculate_projection_error, analyze_pnp_projection_errors, analyze_bundle_projection_errors
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict
import sys
import os

from tqdm import tqdm
from reports.paths import FINAL_PLOTS_RELATIVE_PATH, OUTPUT_RELATIVE_PATH
from kitti_slam.gtsam_geometry import read_calibration
from kitti_slam.tracking_database import TrackingDB
from kitti_slam.gtsam_geometry import create_stereo_camera, create_pose_from_extrinsics




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
