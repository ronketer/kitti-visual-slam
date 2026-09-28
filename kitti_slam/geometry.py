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
