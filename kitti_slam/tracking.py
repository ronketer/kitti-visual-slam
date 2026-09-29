"""Sequential stereo odometry and tracking database construction.

The database owns track IDs and observation updates. This module owns the
frame loop, correspondence remapping, motion calls, and pose accumulation.
"""

import cv2
from tqdm import tqdm
from .config import DEFAULT_PATHS, FIRST_FRAME, LAST_FRAME
from .dataset import read_images, read_cameras
from .detector_config import create_detector_and_matcher
from .geometry import compose_extrinsics
from .motion import perform_motion_estimation
from .stereo import process_stereo_pair, find_stereo_temporal_matches
from .tracking_database import TrackingDB


def initialize_tracking(calibration=None, paths=DEFAULT_PATHS):
    """Initialize tracking components and camera calibration"""
    # Read camera calibration
    K, M1, M2 = read_cameras(paths) if calibration is None else calibration
    camera_left = K @ M1
    camera_right = K @ M2

    # Initialize tracking database
    db = TrackingDB()

    return db, camera_left, camera_right, K, M1, M2


def build_tracking_database(*, paths=DEFAULT_PATHS, calibration=None, detector=None, matcher=None,
                       first_frame=FIRST_FRAME, last_frame=LAST_FRAME, collect_diagnostics=False):
    """
    Main processing loop with improved feature tracking.

    Uses the q6.py approach for better matching:
    1. Stereo matches validated through 3D reconstruction
    2. Temporal matches validated for 4-way correspondence

    Optional inputs supply dataset paths, a (K, M1, M2) calibration tuple,
    and OpenCV detector/matcher objects. Frame bounds are inclusive; the
    current TrackingDB implementation requires first_frame=0.
    collect_diagnostics records temporal match, common-match, and supporter
    counts. It defaults to False to preserve the original runtime checkpoints;
    the report entry point enables it.
    """
    # TrackingDB uses contiguous, zero-based frame IDs.
    if first_frame != 0:
        raise ValueError("TrackingDB currently requires frame numbering to start at zero")
    if last_frame < first_frame:
        raise ValueError("last_frame must be at least first_frame")
    if detector is None or matcher is None:
        default_detector, default_matcher = create_detector_and_matcher()
        detector = default_detector if detector is None else detector
        matcher = default_matcher if matcher is None else matcher
    db, camera_left, camera_right, K, M1, M2 = initialize_tracking(calibration, paths)

    prev_kp_left, prev_des_left, prev_kp_right, prev_des_right = None, None, None, None
    prev_query_idx_to_features_idx_map = None
    prev_inliers_matches, prev_cloud = None, None
    tvec0 = M2[:, 3].reshape(3, 1)
    # Process all frames
    for frame_idx in tqdm(range(first_frame, last_frame + 1)):
        # Load stereo image pair
        img_left, img_right = read_images(frame_idx, paths)

        # Process current stereo pair with geometric validation
        kp_left, desc_left, kp_right, desc_right, inliers_matches, cloud = (
            process_stereo_pair(img_left, img_right, camera_left, camera_right,
                                detector=detector, matcher=matcher)
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

        if frame_idx == first_frame:
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
            prev_frame_size = db.last_features().shape[0]

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

            counts = {}
            if collect_diagnostics:
                counts = {
                    "total_matches_count": len(matches_between_img0_img1),
                    "common_matches_count": len(common_matches),
                    "supporters_count": len(supporters),
                }
            _ = db.add_frame(links, features, matches_to_previous_left, inliers, **counts)
            # Store the relative extrinsics matrix for this frame
            db.add_relative_extrinsics(frame_idx, relative_extrinsics_l1)
            absolute_extrinsics_l1 = compose_extrinsics(
                relative_extrinsics_l1, db.get_absolute_extrinsics(frame_idx - 1)
            )
            db.add_absolute_extrinsics(frame_idx, absolute_extrinsics_l1)

        # Update previous frame data
        prev_kp_left, prev_des_left = kp_left, desc_left
        prev_kp_right, prev_des_right = kp_right, desc_right
        prev_query_idx_to_features_idx_map = query_idx_to_features_idx_map
        prev_inliers_matches, prev_cloud = inliers_matches, cloud
    return db
