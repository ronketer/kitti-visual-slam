"""Four-view reprojection validation for RANSAC hypotheses."""

from geometry import coordinate_transform
import numpy as np

def validate_projections_batch(points, extrinsic_matrix, K, keypoints, threshold=2):
    """
    Validate projections for multiple points at once using vectorized operations.

    Args:
        points: (N, 3) array of 3D points
        extrinsic_matrix: (3, 4) camera extrinsic matrix
        K: (3, 3) camera intrinsic matrix
        keypoints: list of N keypoints
        threshold: distance threshold for validation

    Returns:
        (N,) boolean array where True indicates valid projections
    """
    # Convert keypoints to numpy array
    keypoints_array = np.array([kp.pt for kp in keypoints], dtype=np.float32)

    # Transform and project points
    points_transformed = coordinate_transform(points, extrinsic_matrix)
    projected_points = (K @ points_transformed.T).T

    # Normalize homogeneous coordinates
    projected_points_2D = projected_points[:, :2] / projected_points[:, 2:3]

    # Calculate distances
    distances = np.linalg.norm(projected_points_2D - keypoints_array, axis=1)
    return distances < threshold

def find_ransac_iteration_supporters(common_matches_indices, cloud0, kp_left0, kp_right0, kp_left1, kp_right1, K,
       extrinsic_l1, extrinsic_r1, extrinsic_l0, extrinsic_r0, inliers0):
    """
    Find supporters for RANSAC iteration by validating projections of 3D points
    against keypoints in all views.
    Args:
        common_matches_indices: Array of shape (N, 4) with indices of matches
        cloud0: Nx3 array of 3D points
        kp_left0, kp_right0, kp_left1, kp_right1: Lists of keypoints for each view
        K: Camera intrinsic matrix
        extrinsic_l1, extrinsic_r1, extrinsic_l0, extrinsic_r0: Camera extrinsics for each view
        inliers0: List of inlier matches from the first image pair
    Returns:
        supporters: List of indices of matches that are valid in all views
    """
    # Convert common_matches_indices to numpy array for faster indexing
    common_matches_indices = np.array(common_matches_indices)

    # Create mapping and get cloud points in one step
    queryidx_to_cloud_idx = {m.queryIdx: i for i, m in enumerate(inliers0)}
    points3D = cloud0[[queryidx_to_cloud_idx[idx_l0] for idx_l0 in common_matches_indices[:, 0]]]

    # Get all keypoints in one step using numpy array indexing
    kp_l0 = [kp_left0[idx] for idx in common_matches_indices[:, 0]]
    kp_r0 = [kp_right0[idx] for idx in common_matches_indices[:, 1]]
    kp_l1 = [kp_left1[idx] for idx in common_matches_indices[:, 2]]
    kp_r1 = [kp_right1[idx] for idx in common_matches_indices[:, 3]]

    # Validate all projections in batch
    valid_l0 = validate_projections_batch(points3D, extrinsic_l0, K, kp_l0)
    valid_r0 = validate_projections_batch(points3D, extrinsic_r0, K, kp_r0)
    valid_l1 = validate_projections_batch(points3D, extrinsic_l1, K, kp_l1)
    valid_r1 = validate_projections_batch(points3D, extrinsic_r1, K, kp_r1)

    # Find points that are valid in all views using vectorized operations
    all_valid = valid_l0 & valid_r0 & valid_l1 & valid_r1

    # Extract the valid matches
    supporters = common_matches_indices[all_valid].tolist()

    return supporters
