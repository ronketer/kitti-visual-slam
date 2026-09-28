"""BA constraint extraction and pose graph optimization without plotting or I/O."""

import numpy as np
from tqdm import tqdm
from .trajectory import poses_from_values


def extract_relative_pose_covariance(all_ba_results):
    import gtsam

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


def build_pose_graph(relative_poses_and_covariances, *, endpoint_frame, endpoint_pose):
    """Build the original pose chain with an explicitly supplied endpoint prior.

    endpoint_pose is a camera-to-world GTSAM Pose3. The historical experiment
    supplies it from ground truth. This function does not detect loop closures.
    The endpoint initial estimate is overwritten, preserving the original
    behavior and the legacy before/after result semantics.
    """
    import gtsam

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

    # Apply the explicit endpoint prior with the original tight noise model.
    print("\n--- Adding Endpoint Prior ---")
    last_pose = endpoint_pose
    last_kf_key = kf_pose_keys[endpoint_frame]
    pose_graph.add(
        gtsam.PriorFactorPose3(
            last_kf_key, last_pose, gtsam.noiseModel.Diagonal.Sigmas(np.array([1e-6] * 6))
        )
    )
    # Instead of insert, use update to overwrite the initial estimate for last_kf_key
    pose_graph_initial_estimate.update(last_kf_key, last_pose)
    return pose_graph, pose_graph_initial_estimate, kf_pose_keys


def optimize_pose_graph(pose_graph, initial_estimate, kf_pose_keys, relative_poses_and_covariances):
    """Run the existing batch LM optimizer and return checkpoint-compatible data.

    The legacy poses_without_loop_closure key denotes the initial estimate,
    including its endpoint overwrite; it is not an unconstrained optimization.
    """
    import gtsam

    # Optimize the pose graph again with loop closure
    optimizer = gtsam.LevenbergMarquardtOptimizer(
        pose_graph, initial_estimate
    )

    optimized_pose_graph_result = optimizer.optimize()
    optimized_error_pose_graph = pose_graph.error(optimized_pose_graph_result)
    init_error_pose_graph = pose_graph.error(initial_estimate)
    print(f"Pose Graph Initial Error before optimization: {init_error_pose_graph:.4f}")
    print(f"Pose Graph Optimized Error after optimization: {optimized_error_pose_graph:.4f}")

    poses_without_loop_closure = poses_from_values(initial_estimate, kf_pose_keys)
    poses_with_loop_closure = poses_from_values(optimized_pose_graph_result, kf_pose_keys)
    loop_closure_results = {
        'poses_without_loop_closure': poses_without_loop_closure,
        'poses_with_loop_closure': poses_with_loop_closure,
        'kf_pose_keys': kf_pose_keys,
        'pose_graph': pose_graph,
        'initial_estimate': initial_estimate,
        'optimized_result': optimized_pose_graph_result,
        'relative_poses_and_covariances': relative_poses_and_covariances
    }

    return loop_closure_results



def run_pose_graph(all_ba_results, *, endpoint_frame, endpoint_pose):
    """Extract BA constraints, construct the pose graph, and optimize it."""
    constraints = extract_relative_pose_covariance(all_ba_results)
    graph, initial, keys = build_pose_graph(
        constraints, endpoint_frame=endpoint_frame, endpoint_pose=endpoint_pose
    )
    return optimize_pose_graph(graph, initial, keys, constraints)
