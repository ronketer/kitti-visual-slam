import numpy as np
from tqdm import tqdm
import cv2
import gtsam
from tracking_database import TrackingDB
from utility import (
    read_images,
    read_cameras,
    find_stereo_temporal_matches,
    compose_extrinsics,
    compute_camera_to_camera_transform,
)
from detector_config import detector, matcher
from consts import FIRSTFRAME, LASTFRAME
from perform_motion_estimation import perform_motion_estimation


def filter_by_ratio_test(matches, thresholds):
    """
    Apply ratio test with multiple thresholds.

    Args:
        matches: List of knn matches (k=2) from BFMatcher
        thresholds: List of ratio thresholds (e.g., [0.75, 0.8])

    Returns:
        {
            "good": {threshold: list_of_indices_passing_threshold},
            "bad": {threshold: list_of_indices_failing_threshold}
        }
    """
    results = {"good": {t: [] for t in thresholds}, "bad": {t: [] for t in thresholds}}
    for idx, (best_match, second_best_match) in enumerate(matches):
        for t in thresholds:
            if best_match.distance < t * second_best_match.distance:
                results["good"][t].append(idx)
            else:
                results["bad"][t].append(idx)

    return results


def classify_matches_by_deviation(matches, kp0, kp1):
    """
    Classify matches into inliers and outliers based on y-coordinate deviation.

    Args:
        matches: List of matches to classify.
        kp0: Keypoints from the query image.
        kp1: Keypoints from the train image.

    Returns:
        Tuple of inliers and outliers.
    """
    inliers = []
    outliers = []

    for match in matches:
        train_keypoint = kp1[match.trainIdx]
        query_keypoint = kp0[match.queryIdx]
        cur_deviation = abs(query_keypoint.pt[1] - train_keypoint.pt[1])

        if cur_deviation < 2:
            inliers.append(match)
        else:
            outliers.append(match)

    return inliers, outliers


def trangulate_inliers(camera1, camera2, kp_left, des_left, kp_right, des_right):
    """
    Triangulate inliers from stereo matches using the provided cameras.
    Uses the matcher to find matches, classifies them by deviation, and performs triangulation.
    Validate kp_left.x > kp_right.x for non-behind-the-car points.
    Args:
        camera1: Camera matrix for the first camera.
        camera2: Camera matrix for the second camera.
        kp_left: Keypoints from the left image.
        des_left: Descriptors from the left image.
        kp_right: Keypoints from the right image.
        des_right: Descriptors from the right image.
    Returns:
        Tuple containing:
            - Inlier matches after validation
            - 3D point cloud from triangulation
    """
    matches = matcher.match(des_left, des_right)
    inliers_matches, _ = classify_matches_by_deviation(matches, kp_left, kp_right)
    MIN_DISPARITY_THRESHOLD = 2
    filtered_matches = [
        m
        for m in inliers_matches
        if kp_left[m.queryIdx].pt[0] > kp_right[m.trainIdx].pt[0]
        and (kp_left[m.queryIdx].pt[0] - kp_right[m.trainIdx].pt[0])
        >= MIN_DISPARITY_THRESHOLD
    ]
    pts_left = np.float32([kp_left[m.queryIdx].pt for m in filtered_matches]).T
    pts_right = np.float32([kp_right[m.trainIdx].pt for m in filtered_matches]).T
    if pts_left.shape[1] == 0:
        return [], np.empty((0, 3))
    cloud4D = cv2.triangulatePoints(camera1, camera2, pts_left, pts_right)
    points3D = (cloud4D[:3] / cloud4D[3]).T
    return filtered_matches, points3D


def solveLLST(points2D1, points2D2, camera1, camera2):
    """
    Solve the linear least squares triangulation problem.
    Args:
        points2D1: 2D points from the first camera.
        points2D2: 2D points from the second camera.
        camera1: Camera matrix for the first camera.
        camera2: Camera matrix for the second camera.
    Returns:
        3D point in homogeneous coordinates.
    """
    row1_camera1 = np.array([camera1[0, :]])
    row2_camera1 = np.array([camera1[1, :]])
    row3_camera1 = np.array([camera1[2, :]])

    row1_camera2 = np.array([camera2[0, :]])
    row2_camera2 = np.array([camera2[1, :]])
    row3_camera2 = np.array([camera2[2, :]])

    px1 = points2D1[0]
    py1 = points2D1[1]
    px2 = points2D2[0]
    py2 = points2D2[1]
    A = np.array(
        [
            px1 * row3_camera1 - row1_camera1,
            py1 * row3_camera1 - row2_camera1,
            px2 * row3_camera2 - row1_camera2,
            py2 * row3_camera2 - row2_camera2,
        ]
    )
    A = np.vstack(A)
    U, S, Vt = np.linalg.svd(A)
    X = Vt[-1]
    X = X / X[3]
    return X[:3]


def initialize_tracking():
    """Initialize tracking components and camera calibration"""
    # Read camera calibration
    K, M1, M2 = read_cameras()
    camera_left = K @ M1
    camera_right = K @ M2

    # Initialize tracking database
    db = TrackingDB()

    return db, camera_left, camera_right, K, M1, M2


def process_stereo_pair(img_left, img_right, camera_left, camera_right):
    """
    Process stereo image pair using geometric validation from q6.

    Uses triangulation to validate stereo matches and extract 3D points.
    Args:
        img_left: Left stereo image.
        img_right: Right stereo image.
        camera_left: Camera matrix for the left camera.
        camera_right: Camera matrix for the right camera.
    Returns:
        Tuple containing:
            - Keypoints and descriptors for left image
            - Keypoints and descriptors for right image
            - Inlier matches after validation
            - 3D point cloud from triangulation
    """
    # Detect features
    kp_left, desc_left = detector.detectAndCompute(img_left, None)
    kp_right, desc_right = detector.detectAndCompute(img_right, None)

    # Match and validate stereo features using q1
    inliers_matches, cloud = trangulate_inliers(
        camera_left, camera_right, kp_left, desc_left, kp_right, desc_right
    )

    return kp_left, desc_left, kp_right, desc_right, inliers_matches, cloud


def create_pose_from_extrinsics(extrinsics):
    """Convert extrinsic matrix (world-to-camera) to gtsam.Pose3.
    Args:
        extrinsics: 3x4 extrinsic matrix [R|t] in world-to
    camera format
    Returns:
        gtsam.Pose3 object with camera-to-world transformation
    """
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
    pose3 = create_pose_from_extrinsics(extrinsics)
    return gtsam.StereoCamera(pose3, K)


def create_tracking_db():
    """
    Main processing loop with improved feature tracking.

    Uses the q6.py approach for better matching:
    1. Stereo matches validated through 3D reconstruction
    2. Temporal matches validated for 4-way correspondence
    """
    # Initialize tracking database and cameras
    db, camera_left, camera_right, K, M1, M2 = initialize_tracking()

    prev_kp_left, prev_des_left, prev_kp_right, prev_des_right = None, None, None, None
    prev_features, prev_links = None, None
    prev_query_idx_to_features_idx_map = None
    prev_inliers_matches, prev_cloud = None, None
    tvec0 = M2[:, 3].reshape(3, 1)
    # Process all frames
    for frame_idx in tqdm(range(FIRSTFRAME, LASTFRAME + 1)):
        # Load stereo image pair
        img_left, img_right = read_images(frame_idx)

        # Process current stereo pair with geometric validation
        kp_left, desc_left, kp_right, desc_right, inliers_matches, cloud = (
            process_stereo_pair(img_left, img_right, camera_left, camera_right)
        )
        features, links = TrackingDB.create_links(
            features=desc_left,
            kp_left=kp_left,
            kp_right=kp_right,
            matches=[
                (m,) for m in inliers_matches
            ],  # Convert to format expected by create_links
        )
        query_idx_to_features_idx_map = {
            m.queryIdx: i for i, m in enumerate(inliers_matches)
        }

        if frame_idx == FIRSTFRAME:
            # Add first frame without temporal matches
            _ = db.add_frame(links, features)
            db.add_absolute_extrinsics(frame_idx, M1)
        else:
            matches_between_img0_img1 = matcher.match(prev_des_left, desc_left)
            common_matches, common_matches_indices = find_stereo_temporal_matches(
                prev_inliers_matches, inliers_matches, matches_between_img0_img1
            )
            relative_extrinsics_l1, _, supporters = perform_motion_estimation(
                common_matches_indices,
                common_matches,
                prev_cloud,
                cloud,
                prev_inliers_matches,
                prev_kp_left,
                prev_kp_right,
                kp_left,
                kp_right,
                K,
                M1,
                M2,
                tvec0,
            )
            # Get sizes of feature arrays
            prev_frame_size = db.frameId_to_lfeature[db.last_frameId].shape[0]

            # Create matches array with None for invalid matches
            matches_to_previous_left = [None] * prev_frame_size
            inliers = [False] * prev_frame_size
            # Create mapping from previous query index to current query index
            prev_to_curr_idx = {cm[0]: cm[2] for cm in supporters}
            for i in range(prev_frame_size):
                prev_queryidx = prev_inliers_matches[i].queryIdx
                # Get current query index if it exists
                cur_queryidx = prev_to_curr_idx.get(prev_queryidx)
                if cur_queryidx is not None:
                    matches_to_previous_left[i] = cv2.DMatch(
                        _queryIdx=prev_query_idx_to_features_idx_map[prev_queryidx],
                        _trainIdx=query_idx_to_features_idx_map[cur_queryidx],
                        _distance=0,
                    )
                    inliers[i] = True

            _ = db.add_frame(links, features, matches_to_previous_left, inliers)
            # Store the relative extrinsics matrix for this frame
            db.add_relative_extrinsics(frame_idx, relative_extrinsics_l1)
            absolute_extrinsics_l1 = compose_extrinsics(
                relative_extrinsics_l1, db.get_absolute_extrinsics(frame_idx - 1)
            )
            db.add_absolute_extrinsics(frame_idx, absolute_extrinsics_l1)

        # Update previous frame data
        prev_kp_left, prev_des_left = kp_left, desc_left
        prev_kp_right, prev_des_right = kp_right, desc_right
        prev_features, prev_links = features, links
        prev_query_idx_to_features_idx_map = query_idx_to_features_idx_map
        prev_inliers_matches, prev_cloud = inliers_matches, cloud
    return db


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
