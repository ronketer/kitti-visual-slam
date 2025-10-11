import numpy as np
import matplotlib.pyplot as plt
import gtsam
import gtsam.utils.plot as gtsam_plot
from tqdm import tqdm
import pickle
from consts import OUTPUT_RELATIVE_PATH, LASTFRAME
from alg import create_pose_from_extrinsics
from utility import find_camera_location
from ex5 import plot_trajectories, plot_trajectory_errors, plot_relative_translation_errors

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

def extract_relative_pose_covariance(all_ba_results):
    print("\n--- Extract relative pose constraint from Bundle optimization ---")
    relative_poses_and_covariances = []
    for window_dict in tqdm(all_ba_results):
        start_kf, end_kf = window_dict["start_kf"], window_dict["end_kf"]
        graph, result = (
            window_dict["graph"],
            window_dict["result"],
        )
        pose_keys = window_dict["pose_keys"]
        marginals = gtsam.Marginals(graph, result)
        keys = gtsam.KeyVector()
        keys.append(pose_keys[start_kf])
        keys.append(pose_keys[end_kf])
        marginal_cov = marginals.jointMarginalCovariance(keys).fullMatrix()
        pose_start = result.atPose3(pose_keys[start_kf])
        pose_end = result.atPose3(pose_keys[end_kf])
        relative_pose = pose_start.between(pose_end)
        eps = 1e-4  # Small value to avoid numerical issues
        # information_matrix = marginals.jointMarginalInformation(keys).fullMatrix()
        information_matrix = np.linalg.inv(marginal_cov + eps * np.eye(12))
        conditional_cov = np.linalg.inv(
            np.eye(6)*eps + information_matrix[6:, 6:]
        )  # 6x6 covariance for relative pose

        relative_poses_and_covariances.append(dict())
        relative_poses_and_covariances[-1]["start_kf"] = start_kf
        relative_poses_and_covariances[-1]["end_kf"] = end_kf
        relative_poses_and_covariances[-1]["relative_pose"] = relative_pose
        relative_poses_and_covariances[-1]["conditional_cov"] = conditional_cov  

    return relative_poses_and_covariances  # Return computed results

def parse_gt_line(line):
    mat = np.array([float(x) for x in line.strip().split()]).reshape(3, 4)
    return mat  # Return full extrinsic matrix [R|t]

def build_pose_graph_with_loop_closure(relative_poses_and_covariances, last_frame_gt_matrix):
    
    print("\n--- 6.2 Pose Graph Optimization ---")

    pose_graph = gtsam.NonlinearFactorGraph()
    pose_graph_initial_estimate = gtsam.Values()
    kf_pose_keys = {}
    kf_pose_keys[0] = gtsam.symbol('c', 0)  # Initialize first keyframe pose key
    # Add a prior factor for the first keyframe (absolute constraint)
    first_kf_id = 0
    identity_pose = gtsam.Pose3()
    prior_noise = gtsam.noiseModel.Diagonal.Sigmas(
        np.array([1e-6, 1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
    )
    pose_graph.add(
        gtsam.PriorFactorPose3(kf_pose_keys[first_kf_id], identity_pose, prior_noise)
    )
    pose_graph_initial_estimate.insert(kf_pose_keys[first_kf_id], identity_pose)

    cur_start_kf_pose = identity_pose

    for window_dict in tqdm(relative_poses_and_covariances):
        start_kf, end_kf = window_dict["start_kf"], window_dict["end_kf"]
        # Store keyframe pose keys
        kf_pose_keys[end_kf] = gtsam.symbol('c', end_kf)
        relative_pose = window_dict["relative_pose"]
        conditional_cov = window_dict["conditional_cov"]  # Use 6x6 relative covariance
        
        # Create noise model from covariance matrix
        noise_model_covariance = gtsam.noiseModel.Gaussian.Covariance(conditional_cov)
        # Add BetweenFactor for the relative pose
        pose_graph.add(
            gtsam.BetweenFactorPose3(
                kf_pose_keys[start_kf],
                kf_pose_keys[end_kf],
                relative_pose,
                noise_model_covariance,
            )
        )
        # Add initial estimate for the end keyframe pose
        pose_graph_initial_estimate.insert(
            kf_pose_keys[end_kf], cur_start_kf_pose.compose(relative_pose)
        )
        cur_start_kf_pose = pose_graph_initial_estimate.atPose3(kf_pose_keys[end_kf])
        
    # Adding loop closure, read the last frame gt and add it to the pose graph as prior factor
    print("\n--- Adding Loop Closure ---")
    last_pose = create_pose_from_extrinsics(last_frame_gt_matrix)
    last_kf_key = kf_pose_keys[LASTFRAME]
    pose_graph.add(
        gtsam.PriorFactorPose3(
            last_kf_key, last_pose, gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6] * 6))
        )
    )
    # Instead of insert, use update to overwrite the initial estimate for last_kf_key
    pose_graph_initial_estimate.update(last_kf_key, last_pose)
    return pose_graph, pose_graph_initial_estimate, kf_pose_keys


    
def optimize_pose_graph(pose_graph, initial_estimate, gt_matrices, kf_pose_keys, relative_poses_and_covariances):
    """
    Optimize the pose graph using the Levenberg-Marquardt optimizer and visualize results.
    This function performs pose graph optimization with loop closure using GTSAM's Levenberg-Marquardt optimizer.
    It computes and prints the initial and optimized errors, visualizes the optimized poses and their covariances,
    compares trajectories before and after optimization, and plots trajectory and relative translation errors.
    Args:
        pose_graph: GTSAM NonlinearFactorGraph containing the pose graph
        initial_estimate: GTSAM Values containing the initial estimate for optimization
        gt_matrices: List of ground truth matrices for camera poses
        kf_pose_keys: Dictionary mapping keyframe IDs to GTSAM keys
        relative_poses_and_covariances: List of dictionaries containing relative poses and covariances
    Returns:
        None
        
    """

    print("\n--- Optimizing Pose Graph ---")
    gt_centers = np.array([find_camera_location(m) for m in gt_matrices])

    # Optimize the pose graph again with loop closure
    optimizer = gtsam.LevenbergMarquardtOptimizer(
        pose_graph, initial_estimate
    )

    optimized_pose_graph_result = optimizer.optimize()
    optimized_error_pose_graph = pose_graph.error(optimized_pose_graph_result)
    init_error_pose_graph = pose_graph.error(initial_estimate)
    print(f"Pose Graph Initial Error before Loop Closure: {init_error_pose_graph:.4f}")
    print(f"Pose Graph Optimized Error after Loop Closure: {optimized_error_pose_graph:.4f}")
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
    init_poses_dict = {}
    opt_poses_dict = {}
    for frame_id in kf_pose_keys:
        if initial_estimate.exists(kf_pose_keys[frame_id]):
            init_poses_dict[frame_id] = initial_estimate.atPose3(kf_pose_keys[frame_id])
        if optimized_pose_graph_result.exists(kf_pose_keys[frame_id]):
            opt_poses_dict[frame_id] = optimized_pose_graph_result.atPose3(kf_pose_keys[frame_id])
    
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
    
    # Save loop closure results to pickle file for future use
    poses_without_loop_closure = {}
    poses_with_loop_closure = {}
    for frame_id in kf_pose_keys:
        if initial_estimate.exists(kf_pose_keys[frame_id]):
            poses_without_loop_closure[frame_id] = initial_estimate.atPose3(kf_pose_keys[frame_id])
        if optimized_pose_graph_result.exists(kf_pose_keys[frame_id]):
            poses_with_loop_closure[frame_id] = optimized_pose_graph_result.atPose3(kf_pose_keys[frame_id])
    
    loop_closure_results = {
        'poses_without_loop_closure': poses_without_loop_closure,
        'poses_with_loop_closure': poses_with_loop_closure,
        'kf_pose_keys': kf_pose_keys,
        'pose_graph': pose_graph,
        'initial_estimate': initial_estimate,
        'optimized_result': optimized_pose_graph_result,
        'relative_poses_and_covariances': relative_poses_and_covariances
    }
    
    loop_closure_save_path = OUTPUT_RELATIVE_PATH + "loop_closure_results.pkl"
    with open(loop_closure_save_path, "wb") as f:
        pickle.dump(loop_closure_results, f)
    print(f"Loop closure results saved to: {loop_closure_save_path}")


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
    with open("dataset/poses/05.txt", "r") as f:
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
