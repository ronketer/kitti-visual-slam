"""Historical trajectory metrics with their original frame-selection policies.

Consecutive PnP/BA samples and half-sequence truncation remain unchanged.
These definitions are preserved for regression, not standardized KITTI scoring.
"""

import numpy as np
import cv2
from tqdm import tqdm
from ..gtsam_geometry import create_pose_from_extrinsics


def calculate_rotation_error_degrees(pose1, pose2):
    """
    Calculate rotation error between two poses in degrees using Rodrigues representation.

    Args:
        pose1: gtsam.Pose3 - First pose
        pose2: gtsam.Pose3 - Second pose

    Returns:
        float: Rotation error in degrees
    """
    # Get relative rotation
    rel_pose = pose1.between(pose2)
    R = rel_pose.rotation().matrix()

    # Convert to Rodrigues representation
    rvec, _ = cv2.Rodrigues(R)
    angle_rad = np.linalg.norm(rvec)
    angle_deg = angle_rad * 180.0 / np.pi

    return angle_deg


def find_closest_keyframe(target_frame, available_keyframes):
    """Find the closest available keyframe to the target frame"""
    return min(available_keyframes, key=lambda x: abs(x - target_frame))


def calculate_total_distance(start_frame, end_frame, gt_matrices):
    """
    Calculate total distance traveled between two frames based on ground truth.

    Args:
        start_frame: Starting frame index
        end_frame: Ending frame index
        gt_matrices: List of ground truth matrices

    Returns:
        float: Total distance in meters
    """
    total_dist = 0.0
    for i in range(start_frame, end_frame):
        if i + 1 < len(gt_matrices):
            pose_i = create_pose_from_extrinsics(gt_matrices[i])
            pose_i_plus_1 = create_pose_from_extrinsics(gt_matrices[i + 1])
            rel_pose = pose_i.between(pose_i_plus_1)
            total_dist += np.linalg.norm(rel_pose.translation())
    return total_dist


def calculate_relative_pose_error(est_pose_start, est_pose_end, gt_pose_start, gt_pose_end):
    """
    Calculate relative pose error between estimated and ground truth poses.

    Args:
        est_pose_start: Estimated start pose (gtsam.Pose3)
        est_pose_end: Estimated end pose (gtsam.Pose3)
        gt_pose_start: Ground truth start pose (gtsam.Pose3)
        gt_pose_end: Ground truth end pose (gtsam.Pose3)

    Returns:
        tuple: (translation_error_norm, rotation_error_degrees)
    """
    import gtsam

    # Calculate relative poses
    est_relative = est_pose_start.between(est_pose_end)
    gt_relative = gt_pose_start.between(gt_pose_end)

    # Calculate error between relative poses
    error_pose = est_relative.between(gt_relative)

    # Translation error (norm)
    translation_error = np.linalg.norm(error_pose.translation())

    # Rotation error (degrees)
    rotation_error = calculate_rotation_error_degrees(gtsam.Pose3(), error_pose)

    return translation_error, rotation_error


def calculate_consecutive_relative_errors(estimated_poses, gt_matrices):
    """
    Calculate relative errors for consecutive keyframe pairs.

    Args:
        estimated_poses: dict of frame_id -> gtsam.Pose3
        gt_matrices: List of ground truth matrices

    Returns:
        tuple: (translation_errors, rotation_errors, frame_pairs, distances)
    """
    frame_ids = sorted(estimated_poses.keys())
    translation_errors = []
    rotation_errors = []
    frame_pairs = []
    distances = []

    for i in range(len(frame_ids) - 1):
        start_frame = frame_ids[i]
        end_frame = frame_ids[i + 1]

        # Get estimated poses
        est_start = estimated_poses[start_frame]
        est_end = estimated_poses[end_frame]

        # Get ground truth poses
        gt_start = create_pose_from_extrinsics(gt_matrices[start_frame])
        gt_end = create_pose_from_extrinsics(gt_matrices[end_frame])

        # Calculate relative pose error
        trans_error, rot_error = calculate_relative_pose_error(est_start, est_end, gt_start, gt_end)

        # Calculate total distance for normalization
        total_dist = calculate_total_distance(start_frame, end_frame, gt_matrices)

        if total_dist > 0:  # Avoid division by zero
            translation_errors.append((trans_error / total_dist) * 100)  # Convert to percentage
            rotation_errors.append(rot_error / total_dist)  # degrees per meter
            frame_pairs.append((start_frame, end_frame))
            distances.append(total_dist)

    return translation_errors, rotation_errors, frame_pairs, distances


def calculate_subsequence_relative_errors(estimated_poses, gt_matrices, sequence_length):
    """
    Calculate relative errors for subsequences of given length.

    Args:
        estimated_poses: dict of frame_id -> gtsam.Pose3
        gt_matrices: List of ground truth matrices
        sequence_length: Length of subsequences to analyze

    Returns:
        tuple: (translation_errors, rotation_errors, start_frames, distances)
    """
    available_keyframes = sorted(estimated_poses.keys())
    available_keyframes = available_keyframes[:len(available_keyframes)//2]
    translation_errors = []
    rotation_errors = []
    start_frames = []
    distances = []

    max_start_frame = available_keyframes[-1] - sequence_length

    for start_frame in tqdm(available_keyframes):
        if start_frame > max_start_frame:
            break

        end_frame = start_frame + sequence_length
        closest_end_kf = find_closest_keyframe(end_frame , available_keyframes)

        # Skip if keyframes are too close or the same
        if start_frame >= closest_end_kf:
            continue

        # Get estimated poses
        est_start = estimated_poses[start_frame]
        est_end = estimated_poses[closest_end_kf]

        # Get ground truth poses (use original frame indices)
        if start_frame < len(gt_matrices) and closest_end_kf < len(gt_matrices):
            gt_start = create_pose_from_extrinsics(gt_matrices[start_frame])
            gt_closest_end_kf = create_pose_from_extrinsics(gt_matrices[closest_end_kf])

            # Calculate relative pose error
            trans_error, rot_error = calculate_relative_pose_error(est_start, est_end, gt_start, gt_closest_end_kf)

            # Calculate total distance for normalization (use original frames)
            total_dist = calculate_total_distance(start_frame, closest_end_kf, gt_matrices)

            if total_dist > 0:  # Avoid division by zero
                translation_errors.append((trans_error / total_dist) * 100)  # Convert to percentage
                rotation_errors.append(rot_error / total_dist)  # degrees per meter
                start_frames.append(start_frame)
                distances.append(total_dist)

    return translation_errors, rotation_errors, start_frames, distances


def calculate_absolute_errors(estimated_poses, gt_matrices):
    """
    Calculate absolute translation and rotation errors for estimated poses.

    Args:
        estimated_poses: dict of frame_id -> gtsam.Pose3
        gt_matrices: List of ground truth matrices

    Returns:
        tuple: (translation_errors, rotation_errors, frame_ids)
    """
    frame_ids = sorted(estimated_poses.keys())
    translation_errors = {'x': [], 'y': [], 'z': [], 'norm': []}
    rotation_errors = []

    for frame_id in frame_ids:
        # Get estimated pose
        estimated_pose = estimated_poses[frame_id]
        estimated_translation = estimated_pose.translation()

        # Get ground truth pose
        gt_matrix = gt_matrices[frame_id]
        gt_pose = create_pose_from_extrinsics(gt_matrix)
        gt_translation = gt_pose.translation()

        # Calculate translation errors
        translation_error = estimated_translation - gt_translation
        translation_errors['x'].append(abs(translation_error[0]))
        translation_errors['y'].append(abs(translation_error[1]))
        translation_errors['z'].append(abs(translation_error[2]))
        translation_errors['norm'].append(np.linalg.norm(translation_error))

        # Calculate rotation error
        rotation_error = calculate_rotation_error_degrees(estimated_pose, gt_pose)
        rotation_errors.append(rotation_error)

    return translation_errors, rotation_errors, frame_ids
