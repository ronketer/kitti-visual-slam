"""GTSAM pose conversion and stereo backprojection; imports are deferred."""

import numpy as np
from .config import DEFAULT_PATHS
from .dataset import read_cameras


def create_pose_from_extrinsics(extrinsics):
    """Convert extrinsic matrix (world-to-camera) to gtsam.Pose3.
    Args:
        extrinsics: 3x4 extrinsic matrix [R|t] in world-to
    camera format
    Returns:
        gtsam.Pose3 object with camera-to-world transformation
    """
    import gtsam
    # Convert from world-to-camera to camera-to-world transformation
    # Note: gtsam.Pose3 expects rotation as gtsam.Rot3 and translation as gtsam.Point3
    R = extrinsics[:3, :3]
    t = extrinsics[:3, 3]
    R_inv = R.T  # transpose of rotation matrix is its inverse
    t_inv = -R_inv @ t  # inverse translation
    return gtsam.Pose3(gtsam.Rot3(R_inv), gtsam.Point3(t_inv))


def create_stereo_camera(extrinsics, K):
    """
    Convert extrinsic matrix (world-to-camera) to gtsam.StereoCamera.

    Args:
        extrinsics: 3x4 extrinsic matrix [R|t] in world-to-camera format
        K: gtsam.Cal3_S2Stereo calibration object

    Returns:
        gtsam.StereoCamera object with camera-to-world transformation
    """
    import gtsam
    pose3 = create_pose_from_extrinsics(extrinsics)
    return gtsam.StereoCamera(pose3, K)


def triangulate_with_gtsam(link, extr, K):
    """
    Triangulates a 3D point from a stereo keypoint observation and camera extrinsics.

    Args:
        link: The tracking database link object containing keypoint coordinates.
        extr: The camera extrinsics for the frame.
        K (gtsam.Cal3_S2Stereo): The stereo camera calibration.

    Returns:
        gtsam.Point3: The triangulated 3D point.
    """
    import gtsam
    xleft, y, xright = (
        link.left_keypoint()[0],
        link.left_keypoint()[1],
        link.right_keypoint()[0],
    )
    stereo_point = gtsam.StereoPoint2(xleft, xright, y)
    pose = create_pose_from_extrinsics(extr)
    camera = gtsam.StereoCamera(pose, K)
    return camera.backproject(stereo_point)


def read_calibration(calibration=None, paths=DEFAULT_PATHS):
    """Read camera calibration parameters from the dataset."""
    from gtsam import Cal3_S2Stereo

    K_mat, M1, M2 = read_cameras(paths) if calibration is None else calibration
    fx, fy, skew, cx, cy, basline = (
        K_mat[0, 0],
        K_mat[1, 1],
        K_mat[0, 1],
        K_mat[0, 2],
        K_mat[1, 2],
        M2[0, 3],
    )
    K = Cal3_S2Stereo(fx, fy, skew, cx, cy, -basline)
    return K
