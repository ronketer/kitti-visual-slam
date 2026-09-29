import numpy as np
import matplotlib.pyplot as plt
import pickle
import sys
import os
from tqdm import tqdm
from reports.paths import FINAL_PLOTS_RELATIVE_PATH, OUTPUT_RELATIVE_PATH

def plot_optimization_errors():
    """
    Plot optimization errors for each bundle window.

    Creates a plot showing the mean factor error (total error / #factors) for each
    bundle window, with one line for initial error (before optimization) and one
    for optimized error (after optimization).

    X-axis: Keyframes (starting keyframe of each bundle window)
    Y-axis: Mean factor error
    """
    print("Loading bundle adjustment results...")

    # Load BA results
    try:
        with open(OUTPUT_RELATIVE_PATH + "ba_results.pkl", "rb") as f:
            windows_graph_list = pickle.load(f)
        print(f"Loaded {len(windows_graph_list)} bundle windows")
    except FileNotFoundError:
        print(f"Error: Could not find ba_results.pkl in {OUTPUT_RELATIVE_PATH}")
        print("Please ensure you have run the bundle adjustment (ex5.py) first.")
        return
    except Exception as e:
        print(f"Error loading ba_results.pkl: {e}")
        return

    # Extract data for each window
    keyframes = []
    initial_errors = []
    optimized_errors = []

    print("Calculating mean factor errors for each window...")

    for window_dict in tqdm(windows_graph_list):
        start_kf = window_dict["start_kf"]
        end_kf = window_dict["end_kf"]
        graph = window_dict["graph"]
        initial_estimate = window_dict["initialEstimate"]
        result = window_dict["result"]

        # Calculate total errors
        initial_total_error = graph.error(initial_estimate)
        optimized_total_error = graph.error(result)
        num_factors = graph.size()

        # Calculate mean factor errors
        initial_mean_error = initial_total_error / num_factors
        optimized_mean_error = optimized_total_error / num_factors

        keyframes.append(start_kf)
        initial_errors.append(initial_mean_error)
        optimized_errors.append(optimized_mean_error)


    if not keyframes:
        print("Error: No valid data found in bundle adjustment results")
        return

    print(f"Successfully processed {len(keyframes)} windows")

    # Create the plot
    plt.figure(figsize=(12, 8))

    # Plot both lines
    plt.plot(keyframes, initial_errors, 'r-',
             label='Initial Error (Before Optimization)',
             linewidth=2, marker='o', markersize=3, alpha=0.7)
    plt.plot(keyframes, optimized_errors, 'b-',
             label='Optimized Error (After Optimization)',
             linewidth=2, marker='s', markersize=3, alpha=0.7)

    # Formatting
    plt.xlabel('Keyframes', fontsize=12)
    plt.ylabel('Mean Factor Error', fontsize=12)
    plt.title('Optimization Error – Mean Factor Error vs Keyframes', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)

    # Set y-axis to start from 0 for better visualization
    plt.ylim(bottom=0)

    # Save the plot
    save_path = FINAL_PLOTS_RELATIVE_PATH + "optimization_errors.png"
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    print(f"Plot saved to: {save_path}")

    plt.close()

    # Print summary statistics
    print("\n--- Summary Statistics ---")
    print(f"Number of bundle windows: {len(keyframes)}")
    print(f"Keyframe range: {min(keyframes)} to {max(keyframes)}")
    print(f"Max initial error: {max(initial_errors):.6f}")
    print(f"Max optimized error: {max(optimized_errors):.6f}")
    print(f"Min initial error: {min(initial_errors):.6f}")
    print(f"Min optimized error: {min(optimized_errors):.6f}")

def main():
    """
    Main function to generate the optimization error plot.
    """
    print("=== Bundle Adjustment Optimization Error Analysis ===")
    plot_optimization_errors()
    print("=== Analysis Complete ===")

if __name__ == "__main__":
    main()
