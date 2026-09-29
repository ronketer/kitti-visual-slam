"""Historical projection residuals and aggregation, separated from report I/O.

Sampling, reference frames and exception behavior are intentionally preserved.
"""
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from ..gtsam_geometry import create_stereo_camera, create_pose_from_extrinsics

def calculate_projection_error(measurement, projection):
    """
    Calculate projection error between measurement and projection for stereo cameras.

    Args:
        measurement: gtsam.StereoPoint2 - actual observed stereo point
        projection: gtsam.StereoPoint2 - projected stereo point

    Returns:
        float: Combined projection error in pixels
    """
    # Calculate left camera error
    left_error = np.linalg.norm(
        np.array([measurement.uL(), measurement.v()]) -
        np.array([projection.uL(), projection.v()])
    )

    # Calculate right camera error
    right_error = np.linalg.norm(
        np.array([measurement.uR(), measurement.v()]) -
        np.array([projection.uR(), projection.v()])
    )

    # Return average of left and right errors
    return (left_error + right_error) / 2.0

def analyze_pnp_projection_errors(db, K):
    """
    Analyze projection errors for PnP estimation as a function of distance from triangulation frame.

    Args:
        db: TrackingDB instance
        K: Camera calibration matrix

    Returns:
        dict: Distance -> list of projection errors
    """
    import gtsam
    print("Analyzing PnP projection errors...")

    distance_to_errors = defaultdict(list)
    processed_tracks = 0

    # Get all tracks with sufficient length
    all_tracks = db.all_tracks()
    valid_tracks = [tid for tid in all_tracks if len(db.frames(tid)) >= 3]

    for track_id in tqdm(valid_tracks, desc="Processing PnP tracks"):
        frame_ids = db.frames(track_id)
        triangulation_frame = frame_ids[-1]  # Last frame is reference for PnP

        # Get triangulation frame pose and triangulate 3D point
        triangulation_extrinsics = db.get_absolute_extrinsics(triangulation_frame)
        if triangulation_extrinsics is None:
            raise ValueError(f"Missing extrinsics for triangulation frame {triangulation_frame}")

        triangulation_link = db.link(triangulation_frame, track_id)
        if triangulation_link is None:
            raise ValueError(f"Missing link for triangulation frame {triangulation_frame} and track {track_id}")

        # Create stereo point and triangulate
        stereo_point = gtsam.StereoPoint2(
            triangulation_link.left_keypoint()[0],
            triangulation_link.right_keypoint()[0],
            triangulation_link.left_keypoint()[1]
        )
        triangulation_camera = create_stereo_camera(triangulation_extrinsics, K)
        point3d = triangulation_camera.backproject(stereo_point)

        # Calculate projection errors for all frames in track
        for frame_id in frame_ids:
            distance_from_reference = abs(triangulation_frame - frame_id)

            # Get frame pose and link
            frame_extrinsics = db.get_absolute_extrinsics(frame_id)
            if frame_extrinsics is None:
                raise ValueError(f"Missing extrinsics for frame {frame_id}")


            frame_link = db.link(frame_id, track_id)
            if frame_link is None:
                raise ValueError(f"Missing link for frame {frame_id} and track {track_id}")

            # Project 3D point using PnP pose
            frame_camera = create_stereo_camera(frame_extrinsics, K)
            projected_point = frame_camera.project(point3d)

            # Calculate projection error
            measurement = gtsam.StereoPoint2(
                frame_link.left_keypoint()[0],
                frame_link.right_keypoint()[0],
                frame_link.left_keypoint()[1]
            )

            projection_error = calculate_projection_error(measurement, projected_point)
            distance_to_errors[distance_from_reference].append(projection_error)

        processed_tracks += 1


    print(f"Processed {processed_tracks} tracks for PnP analysis")
    return distance_to_errors

def analyze_bundle_projection_errors(windows_graph_list, K):
    """
    Analyze projection errors for Bundle Adjustment as a function of distance from first frame.

    Args:
        windows_graph_list: List of bundle adjustment results
        K: Camera calibration matrix

    Returns:
        tuple: (initial_distance_to_errors, optimized_distance_to_errors)
    """
    import gtsam
    print("Analyzing Bundle Adjustment projection errors...")

    initial_distance_to_errors = defaultdict(list)
    optimized_distance_to_errors = defaultdict(list)

    for window_dict in tqdm(windows_graph_list, desc="Processing bundle windows"):
        start_kf = window_dict["start_kf"]
        graph = window_dict["graph"]
        initial_estimate = window_dict["initialEstimate"]
        result = window_dict["result"]
        pose_keys = window_dict["pose_keys"]
        point_keys = window_dict["point_keys"]

        # Process each stereo factor in the graph
        for i in range(graph.size()):
            factor = graph.at(i)

            # Check if this is a stereo factor
            if factor.__class__.__name__ != 'GenericStereoFactor3D':
                continue

            # Get factor keys
            keys = factor.keys()
            if len(keys) != 2:
                continue

            pose_key = keys[0]
            point_key = keys[1]

            # Find frame ID from pose key
            frame_id = None
            for fid, key in pose_keys.items():
                if key == pose_key:
                    frame_id = fid
                    break

            if frame_id is None:
                raise ValueError(f"Could not find frame ID for pose key {pose_key}")

            # Calculate distance from reference frame (first frame of bundle)
            distance_from_reference = abs(frame_id - start_kf)

            # Get measurement
            measurement = factor.measured()

            # Calculate initial projection error
            if initial_estimate.exists(pose_key) and initial_estimate.exists(point_key):
                pose_init = initial_estimate.atPose3(pose_key)
                point_init = initial_estimate.atPoint3(point_key)
                stereo_cam_init = gtsam.StereoCamera(pose_init, K)
                projection_init = stereo_cam_init.project(point_init)

                initial_error = calculate_projection_error(measurement, projection_init)
                initial_distance_to_errors[distance_from_reference].append(initial_error)

            # Calculate optimized projection error
            if result.exists(pose_key) and result.exists(point_key):
                pose_opt = result.atPose3(pose_key)
                point_opt = result.atPoint3(point_key)
                stereo_cam_opt = gtsam.StereoCamera(pose_opt, K)
                projection_opt = stereo_cam_opt.project(point_opt)

                optimized_error = calculate_projection_error(measurement, projection_opt)
                optimized_distance_to_errors[distance_from_reference].append(optimized_error)


    return initial_distance_to_errors, optimized_distance_to_errors

def extract_projection_errors_from_window(graph, initial_estimate, result, pose_keys, point_keys, K):
    """
    Extract all projection errors from a bundle window.

    Args:
        graph: gtsam.NonlinearFactorGraph
        initial_estimate: gtsam.Values
        result: gtsam.Values
        pose_keys: dict mapping frame_id to pose keys
        point_keys: dict mapping track_id to point keys
        K: gtsam.Cal3_S2Stereo calibration matrix

    Returns:
        tuple: (initial_errors, optimized_errors) - lists of projection errors
    """
    import gtsam
    initial_errors = []
    optimized_errors = []

    # Iterate through all factors in the graph
    for i in range(graph.size()):
        factor = graph.at(i)

        # Check if this is a stereo factor (GenericStereoFactor3D)
        if factor.__class__.__name__ != 'GenericStereoFactor3D':
            continue

        # Get the keys this factor connects
        keys = factor.keys()
        if len(keys) != 2:
            continue

        pose_key = keys[0]
        point_key = keys[1]

        # Get measurement
        measurement = factor.measured()

        # Calculate initial projection error
        if initial_estimate.exists(pose_key) and initial_estimate.exists(point_key):
            pose_init = initial_estimate.atPose3(pose_key)
            point_init = initial_estimate.atPoint3(point_key)
            stereo_cam_init = gtsam.StereoCamera(pose_init, K)
            projection_init = stereo_cam_init.project(point_init)

            initial_error = calculate_projection_error(measurement, projection_init)
            initial_errors.append(initial_error)

        # Calculate optimized projection error
        if result.exists(pose_key) and result.exists(point_key):
            pose_opt = result.atPose3(pose_key)
            point_opt = result.atPoint3(point_key)
            stereo_cam_opt = gtsam.StereoCamera(pose_opt, K)
            projection_opt = stereo_cam_opt.project(point_opt)

            optimized_error = calculate_projection_error(measurement, projection_opt)
            optimized_errors.append(optimized_error)


    return initial_errors, optimized_errors
