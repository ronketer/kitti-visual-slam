import numpy as np
from tracking_database import TrackingDB
from geometry import compose_extrinsics
from stereo import (
    filter_by_ratio_test, classify_matches_by_deviation, trangulate_inliers,
    solveLLST, process_stereo_pair, find_stereo_temporal_matches,
)
from dataset import read_images, read_cameras
from tracking import initialize_tracking, build_tracking_database as create_tracking_db


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


def add_stereo_factors_to_graph(
    graph, db, K, sigma, frames, pose_keys, point_keys, tid, initialEstimate
):
    """
    Adds stereo projection factors for a given track to the factor graph.

    Args:
        graph (gtsam.NonlinearFactorGraph): The factor graph to add to.
        db (TrackingDB): The tracking database.
        K (gtsam.Cal3_S2Stereo): The stereo camera calibration.
        sigma: The noise model for the measurement.
        frames (list): List of frame indices in the current window.
        pose_keys (dict): Mapping from frame index to pose key.
        point_keys (dict): Mapping from track id to point key.
        tid: The track id for which to add factors.
    """
    import gtsam
    factors = []
    track_frames = [f for f in db.frames(tid) if f in frames]
    for fid in track_frames:
        link = db.link(fid, tid)
        xleft = link.left_keypoint()[0]
        y = link.left_keypoint()[1]
        xright = link.right_keypoint()[0]
        stereo_point = gtsam.StereoPoint2(xleft, xright, y)
        factor = gtsam.GenericStereoFactor3D(
            stereo_point, sigma, pose_keys[fid], point_keys[tid], K
        )
        error = factor.error(initialEstimate)
        if error > 1000:
            raise ValueError(
                f"Warning: Factor error {error} for frame {fid} and track {tid}"
            )
        graph.add(factor)
        factors.append((factor, fid, tid))

    return factors


def initialize_landmarks(
    tracks, frames, point_keys, db, K, initialEstimate, frame_id_to_relative_extrinsics
):
    """
    Initializes 3D landmark positions for all tracks in the current window.

    Args:
        tracks (iterable): Track ids to initialize.
        frames (list): List of frame indices in the current window.
        point_keys (dict): Mapping from track id to point key.
        db (TrackingDB): The tracking database.
        K (gtsam.Cal3_S2Stereo): The stereo camera calibration.
        initialEstimate (gtsam.Values): The initial estimate to insert points into.
        frame_id_to_relative_extrinsics: Mapping from frame id to relative extrinsics.
    """
    for tid in tracks:
        track_frames = [f for f in db.frames(tid) if f in frames]
        if not track_frames:
            raise Exception("error - should not be thrown")

        last_frame = track_frames[-1]
        link = db.link(last_frame, tid)
        extr = frame_id_to_relative_extrinsics.get(last_frame)
        if extr is None:
            raise Exception(
                f"Error: No extrinsics found for frame {last_frame} in track {tid}"
            )
        point3 = triangulate_with_gtsam(link, extr, K)
        initialEstimate.insert(point_keys[tid], point3)


def initialize_pose_keys(initialEstimate, pose_keys, frames, identity_pose, db):
    """
    Initializes pose keys in the initial estimate for all frames in the current window.
    Args:
        initialEstimate (gtsam.Values): The initial estimate to insert poses into.
        pose_keys (dict): Mapping from frame index to pose key.
        frames (list): List of frame indices in the current window.
        identity_pose (gtsam.Pose3): The identity pose for the first frame.
        db (TrackingDB): The tracking database.
    Returns:
        dict: Mapping from frame id to relative extrinsics.
    """
    # Set initial pose for first frame
    initialEstimate.insert(pose_keys[frames[0]], identity_pose)

    _, M1, M2 = read_cameras()
    frame_id_to_relative_extrinsics = {}
    frame_id_to_relative_extrinsics[frames[0]] = M1
    for fid in frames[1:]:
        # Get relative transform from previous to current frame
        relative_transform = db.get_relative_extrinsics(fid)
        if relative_transform is None:
            raise Exception(f"No relative extrinsics found for frame {fid}")

        cur_transform = compose_extrinsics(
            relative_transform, frame_id_to_relative_extrinsics[fid - 1]
        )
        # Store the relative extrinsics for this frame
        frame_id_to_relative_extrinsics[fid] = cur_transform

        # Convert to GTSAM pose and add to initial estimate
        cur_pose = create_pose_from_extrinsics(cur_transform)
        initialEstimate.insert(pose_keys[fid], cur_pose)
    return frame_id_to_relative_extrinsics


def initialize_factor_graph_in_window(db: TrackingDB, start, end, K):
    """
    Initializes the factor graph for a window of frames.

    Args:
        db (TrackingDB): The tracking database containing frame and track data.
        start (int): The starting frame index for the window.
        end (int): The ending frame index for the window.
        K (gtsam.Cal3_S2Stereo): The stereo camera calibration.
    Returns:
        tuple: A tuple containing:
            - gtsam.NonlinearFactorGraph: The initialized factor graph.
            - gtsam.Values: The initial estimate for the factor graph.
            - dict: Mapping from frame index to pose key.
            - dict: Mapping from point index to point key.
            - set: Set of track ids in the window.
            - list: List of frame indices in the window.
            - gtsam.PriorFactorPose3: The prior factor for the first frame.
            - list: List of stereo factors added to the graph.
    """
    import gtsam
    # --- 1. Select window of frames using keyframe selection ---

    sigma = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0, 1.0, 1.0]))
    sigma6d = gtsam.noiseModel.Unit.Create(6)
    frames = list(range(start, end + 1))
    # print(f"Selected window: {start} to {end}")

    # --- 2. Prepare keys for poses and points ---
    pose_keys = {f: gtsam.symbol("c", f) for f in frames}
    tracks = {tid for fid in frames for tid in db.tracks(fid)}
    point_keys = {tid: gtsam.symbol("q", i) for i, tid in enumerate(tracks)}

    graph = gtsam.NonlinearFactorGraph()
    # Set identity pose for first frame as reference
    identity_pose = gtsam.Pose3()
    prior_factor = gtsam.PriorFactorPose3(pose_keys[frames[0]], identity_pose, sigma6d)
    graph.add(prior_factor)
    # Add stereo projection factors for all tracks in window

    # --- 3. Add initalestimate  ---
    initialEstimate = gtsam.Values()

    frame_id_to_relative_extrinsics = initialize_pose_keys(
        initialEstimate, pose_keys, frames, identity_pose, db
    )
    initialize_landmarks(
        tracks,
        frames,
        point_keys,
        db,
        K,
        initialEstimate,
        frame_id_to_relative_extrinsics,
    )

    # --- 4. Add stereo factors to the graph ---
    factors = []
    for tid in tracks:
        cur_factors = add_stereo_factors_to_graph(
            graph, db, K, sigma, frames, pose_keys, point_keys, tid, initialEstimate
        )
        factors.extend(cur_factors)

    return (
        graph,
        initialEstimate,
        pose_keys,
        point_keys,
        tracks,
        frames,
        prior_factor,
        factors,
    )
