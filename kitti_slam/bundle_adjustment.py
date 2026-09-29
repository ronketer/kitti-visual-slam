"""Windowed Bundle Adjustment graph construction and optimization."""

import numpy as np
from tqdm import tqdm
from .tracking_database import TrackingDB
from .geometry import compose_extrinsics
from .dataset import read_cameras
from .gtsam_geometry import create_pose_from_extrinsics, triangulate_with_gtsam
from .window_selector import WindowSelector


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


def initialize_pose_keys(initialEstimate, pose_keys, frames, identity_pose, db, *, reference_extrinsics=None):
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

    M1 = read_cameras()[1] if reference_extrinsics is None else reference_extrinsics
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


def initialize_factor_graph_in_window(db: TrackingDB, start, end, K, *, reference_extrinsics=None):
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
        initialEstimate, pose_keys, frames, identity_pose, db,
        reference_extrinsics=reference_extrinsics,
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


def run_bundle_adjustment(db, K, *, reference_extrinsics=None, window_params=None):
    """Optimize selected windows and return the legacy list of window dictionaries.

    K is a GTSAM stereo calibration. Supply reference_extrinsics (the left
    calibration extrinsics) to avoid dataset I/O; omission preserves the old
    default calibration lookup. No ground truth, plotting, or checkpoint I/O
    is performed. The original 50% error-reduction gate and endpoint pose
    composition are retained.
    """
    import gtsam

    if reference_extrinsics is None:
        reference_extrinsics = read_cameras()[1]

    window_selector = WindowSelector(db, window_params)
    windows = []
    while True:
        window = window_selector.next_window()
        if window is None:
            break
        windows.append(window)

    max_iterations = len(windows)
    windows_graph_list = []
    T_global = gtsam.Pose3()
    optimizied_global_poses = {0: T_global}

    for i in tqdm(range(max_iterations)):
        start_kf, end_kf = windows[i]
        (
            graph,
            initialEstimate,
            pose_keys,
            point_keys,
            tracks,
            frames,
            prior_factor,
            _,
        ) = initialize_factor_graph_in_window(
            db, start_kf, end_kf, K, reference_extrinsics=reference_extrinsics
        )
        init_error = graph.error(initialEstimate)
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate)
        result = optimizer.optimize()
        optimized_error = graph.error(result)
        # validate that the optimized error is lower than the initial error 50%
        if optimized_error * 2 > init_error:
            raise ValueError(
                f"Optimized error {optimized_error} is not significantly lower than initial error {init_error}."
            )

        T_rel_opt = result.atPose3(pose_keys[end_kf])
        T_start_opt_abs = optimizied_global_poses[start_kf]
        T_end_opt_asb = T_start_opt_abs.compose(T_rel_opt)
        optimizied_global_poses[end_kf] = T_end_opt_asb

        windows_graph_list.append(dict())
        windows_graph_list[-1]["start_kf"] = start_kf
        windows_graph_list[-1]["end_kf"] = end_kf
        windows_graph_list[-1]["graph"] = graph
        windows_graph_list[-1]["initialEstimate"] = initialEstimate
        windows_graph_list[-1]["pose_keys"] = pose_keys
        windows_graph_list[-1]["point_keys"] = point_keys
        windows_graph_list[-1]["tracks"] = tracks
        windows_graph_list[-1]["frames"] = frames
        windows_graph_list[-1]["result"] = result
        windows_graph_list[-1]["abs_start_pose"] = T_start_opt_abs
        windows_graph_list[-1]["abs_end_pose"] = T_end_opt_asb

    return windows_graph_list
