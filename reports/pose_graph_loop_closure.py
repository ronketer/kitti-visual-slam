"""Historical endpoint-prior experiment: plotting, I/O, and compatibility wrappers."""
import numpy as np
import matplotlib.pyplot as plt
import gtsam
import gtsam.utils.plot as gtsam_plot
from tqdm import tqdm
import pickle
from reports.paths import OUTPUT_RELATIVE_PATH, LASTFRAME, GT_POSES_FILE
from kitti_slam.gtsam_geometry import create_pose_from_extrinsics
from kitti_slam.geometry import find_camera_location
from kitti_slam.evaluation.plots import plot_trajectories, plot_trajectory_errors, plot_relative_translation_errors

from kitti_slam.pose_graph import extract_relative_pose_covariance, build_pose_graph, optimize_pose_graph as optimize_pose_graph_result

def plot_3d_poses(poses: gtsam.Values, save_path: str, title: str):
    """
    Plots 3D poses without covariance ellipses.

    Args:
        poses: GTSAM Values containing Pose3 objects
        save_path: Path to save the plot (if None, plot is not saved)
        title: Title for the plot
    """
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")

    xs, ys, zs = [], [], []
    for key in poses.keys():
        pose = poses.atPose3(key)
        t = pose.translation()
        xs.append(t[0])
        ys.append(t[1])
        zs.append(t[2])
    ax.view_init(elev=0, azim=-90, roll=0)
    ax.plot(xs, ys, zs, "bo-", markersize=3, label="Poses")
    ax.legend()
    ax.grid(True)

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=1200)
    plt.close(fig)

def plot_3d_poses_with_covariance(poses: gtsam.Values, marginals: gtsam.Marginals, save_path: str, title: str):
    """
    Plots 3D poses with covariance ellipses using GTSAM plotting utilities.

    Args:
        poses: GTSAM Values containing Pose3 objects
        marginals: GTSAM Marginals for covariance information
        save_path: Path to save the plot (if None, plot is not saved)
        title: Title for the plot
    """
    # Close any existing figures to prevent conflicts
    plt.close('all')

    # Create a new figure with a unique number
    fig_num = np.random.randint(1000, 9999)  # Random figure number to avoid conflicts
    fig = plt.figure(fig_num)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=0, azim=-90, roll=0)
    gtsam_plot.plot_trajectory(
        fignum=fig_num,
        values=poses,
        marginals=marginals,
        title=title,
        axis_labels=("X axis (m)", "Y axis (m)", "Z axis (m)"),
    )
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1, dpi=1200)
    plt.close(fig_num)

from kitti_slam.dataset import parse_gt_line_matrix as parse_gt_line

def build_pose_graph_with_loop_closure(relative_poses_and_covariances, last_frame_gt_matrix):
    """Compatibility wrapper for the report's ground-truth endpoint experiment."""
    return build_pose_graph(
        relative_poses_and_covariances,
        endpoint_frame=LASTFRAME,
        endpoint_pose=create_pose_from_extrinsics(last_frame_gt_matrix),
    )


def plot_pose_graph_results(results, gt_matrices):
    """Reproduce the historical pose graph plots from returned results."""
    pose_graph = results["pose_graph"]
    initial_estimate = results["initial_estimate"]
    optimized_pose_graph_result = results["optimized_result"]
    kf_pose_keys = results["kf_pose_keys"]
    relative_poses_and_covariances = results["relative_poses_and_covariances"]
    gt_centers = np.array([find_camera_location(m) for m in gt_matrices])

    plot_3d_poses(
        optimized_pose_graph_result,
        f"{OUTPUT_RELATIVE_PATH}/pose_graph_optimized_poses_with_loop_closure.png",
        "Pose Graph Optimized Poses with Loop Closure",
    )

    # Plot optimized poses with covariance after loop closure
    pose_graph_marginals = gtsam.Marginals(pose_graph, optimized_pose_graph_result)
    plot_3d_poses_with_covariance(
        optimized_pose_graph_result,
        pose_graph_marginals,
        f"{OUTPUT_RELATIVE_PATH}/pose_graph_optimized_poses_with_loop_closure_with_cov.png",
        "Pose Graph Optimized Poses with Loop Closure"
    )

    # Plot comparison trajectories again with loop closure. create new init and opt poses dicts
    init_poses_dict = results["poses_without_loop_closure"]
    opt_poses_dict = results["poses_with_loop_closure"]

    # Plot trajectory comparison with loop closure using ex5 function
    plot_trajectories(
        gt_centers,
        init_poses_dict,
        opt_poses_dict,
        label_every=5,
        save_path=f"{OUTPUT_RELATIVE_PATH}/pose_graph_trajectory_comparison_with_loop_closure.png"
    )
    print("Trajectory comparison plot with loop closure saved.")

    # Plot trajectory errors with loop closure using ex5 function
    plot_trajectory_errors(
        gt_centers,
        init_poses_dict,
        opt_poses_dict,
        save_path=f"{OUTPUT_RELATIVE_PATH}/pose_graph_trajectory_errors_with_loop_closure.png"
    )
    print("Trajectory errors plot with loop closure saved.")

    # Update optimized relative poses after loop closure

    t_rel_init = {}
    t_rel_opt_loop = {}

    for start_kf, end_kf in [(window_dict["start_kf"], window_dict["end_kf"]) for window_dict in relative_poses_and_covariances]:
        if end_kf == LASTFRAME:
            continue
        if initial_estimate.exists(kf_pose_keys[start_kf]) and initial_estimate.exists(kf_pose_keys[end_kf]):
            pose_start_init = initial_estimate.atPose3(kf_pose_keys[start_kf])
            pose_end_init = initial_estimate.atPose3(kf_pose_keys[end_kf])
            t_rel_init[(start_kf, end_kf)] = pose_start_init.between(pose_end_init)
        if optimized_pose_graph_result.exists(kf_pose_keys[start_kf]) and optimized_pose_graph_result.exists(kf_pose_keys[end_kf]):
            pose_start_opt = optimized_pose_graph_result.atPose3(kf_pose_keys[start_kf])
            pose_end_opt = optimized_pose_graph_result.atPose3(kf_pose_keys[end_kf])
            t_rel_opt_loop[(start_kf, end_kf)] = pose_start_opt.between(pose_end_opt)


    # Plot relative translation errors with loop closure using ex5 function
    plot_relative_translation_errors(
        gt_matrices,
        t_rel_init,
        t_rel_opt_loop,
        create_pose_from_extrinsics,
        save_path=f"{OUTPUT_RELATIVE_PATH}/pose_graph_relative_errors_with_loop_closure.png"
    )
    print("Relative translation errors plot with loop closure saved.")



def optimize_pose_graph(pose_graph, initial_estimate, gt_matrices, kf_pose_keys, relative_poses_and_covariances):
    """Legacy optimize/plot/save entry point; computational callers use pose_graph."""
    results = optimize_pose_graph_result(
        pose_graph, initial_estimate, kf_pose_keys, relative_poses_and_covariances
    )
    plot_pose_graph_results(results, gt_matrices)
    path = OUTPUT_RELATIVE_PATH + "loop_closure_results.pkl"
    with open(path, "wb") as f:
        pickle.dump(results, f)
    print(f"Loop closure results saved to: {path}")


def main():
    bundle_save_path = OUTPUT_RELATIVE_PATH + "ba_results.pkl"
    print("\n--- Loading Bundle Adjustment Results ---")
    with open(bundle_save_path, "rb") as f:
        windows_graph_list = pickle.load(f)
    print(f"Loaded {len(windows_graph_list)} windows from {bundle_save_path}")
    relative_poses_and_covariances = extract_relative_pose_covariance(
        windows_graph_list
    )
    gt_matrices = []
    with open(GT_POSES_FILE, "r") as f:
        gt_matrices = [parse_gt_line(line) for line in f.readlines()[:LASTFRAME + 1]]
    last_frame_gt_matrix = gt_matrices[LASTFRAME]
    pose_graph, pose_graph_initial_estimate, kf_pose_keys = build_pose_graph_with_loop_closure(relative_poses_and_covariances , last_frame_gt_matrix)

    optimize_pose_graph(
        pose_graph,
        pose_graph_initial_estimate,
        gt_matrices,
        kf_pose_keys,
        relative_poses_and_covariances
    )


if __name__ == "__main__":
    main()
