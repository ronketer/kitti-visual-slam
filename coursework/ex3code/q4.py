from kitti_slam.supporters import validate_projections_batch
from kitti_slam.geometry import coordinate_transform
import numpy as np

def q4(common_matches_indices, cloud0, kp_left0, kp_right0, kp_left1, kp_right1, K,
       extrinsic_l1, extrinsic_r1, extrinsic_l0, extrinsic_r0, inliers0, logger=None):
    if logger:
        logger.start_operation("Motion Estimation/Inlier Check/Setup")

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

    if logger:
        logger.end_operation("Motion Estimation/Inlier Check/Setup")

    if logger:
        logger.start_operation("Motion Estimation/Inlier Check/Validation")

    # Validate all projections in batch
    valid_l0 = validate_projections_batch(points3D, extrinsic_l0, K, kp_l0)
    valid_r0 = validate_projections_batch(points3D, extrinsic_r0, K, kp_r0)
    valid_l1 = validate_projections_batch(points3D, extrinsic_l1, K, kp_l1)
    valid_r1 = validate_projections_batch(points3D, extrinsic_r1, K, kp_r1)

    # Find points that are valid in all views using vectorized operations
    all_valid = valid_l0 & valid_r0 & valid_l1 & valid_r1

    # Extract the valid matches
    supporters = common_matches_indices[all_valid].tolist()

    if logger:
        logger.end_operation("Motion Estimation/Inlier Check/Validation")

    return supporters
