"""NumPy helpers for 3x4 extrinsics and point transformations."""

import numpy as np


def compose_extrinsics(second_transform, first_transform):
    """Compose two extrinsic transformations.

    Args:
        second_transform: The transformation that will be applied second
        first_transform: The transformation that will be applied first

    Returns:
        Combined transformation that applies first_transform followed by second_transform

    Example:
        If you want to transform from coordinate system A to B to C:
        - first_transform: A to B transformation
        - second_transform: B to C transformation
        Result will be: A to C transformation
    """
    R_second = second_transform[:, :3]
    t_second = second_transform[:, 3].reshape(3, 1)
    R_first = first_transform[:, :3]
    t_first = first_transform[:, 3].reshape(3, 1)

    return np.hstack((R_second @ R_first, R_second @ t_first + t_second))


def find_camera_location(extrinsic_matrix):
    """Find the camera location from the extrinsic matrix."""
    R = extrinsic_matrix[:, :3]
    t = extrinsic_matrix[:, 3]
    C = -R.T @ t
    return C


def coordinate_transform(points, extrinsic_matrix):
    """Transform multiple points using the given extrinsic matrix.

    Args:
        points: (N, 3) array of points
        extrinsic_matrix: (3, 4) transformation matrix

    Returns:
        (N, 3) array of transformed points
    """

    points_homogeneous = np.hstack((points, np.ones((points.shape[0], 1))))

    transformed_points = points_homogeneous @ extrinsic_matrix.T
    return transformed_points


def compute_camera_to_camera_transform(source_cam_extrinsic, target_cam_extrinsic):
    """Compute transformation matrix from one camera's coordinate system to another.

    Args:
        source_cam_extrinsic: Extrinsic matrix of the source camera (camera A)
        target_cam_extrinsic: Extrinsic matrix of the target camera (camera B)

    Returns:
        Transformation matrix that converts points from source camera's coordinate system
        to target camera's coordinate system.

    The transformation is derived from:
        x_source = R_source x_world + t_source
        x_target = R_target x_world + t_target
        Solving for x_world in first equation and substituting into second:
        x_target = R_target R_source^T (x_source - t_source) + t_target
        Which gives the transformation matrix: [R_target R_source^T | t_target - R_target R_source^T t_source]
    """
    R_source = source_cam_extrinsic[:, :3]
    t_source = source_cam_extrinsic[:, 3].reshape(3, 1)
    R_target = target_cam_extrinsic[:, :3]
    t_target = target_cam_extrinsic[:, 3].reshape(3, 1)

    cam_to_cam_rotation = R_target @ R_source.T
    cam_to_cam_translation = t_target - cam_to_cam_rotation @ t_source

    return np.hstack((cam_to_cam_rotation, cam_to_cam_translation))


def find_transformation(extrinsic_matrix):
    """Find the transformation matrix from the extrinsic matrix."""
    R = extrinsic_matrix[:, :3]
    t = extrinsic_matrix[:, 3]
    return lambda x: R @ x + t
