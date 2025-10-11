import sys
import os


sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import random
import numpy as np
from tqdm import tqdm
from tracking_database import TrackingDB
from utility import (
    plot_connectivity,
    plot_track_length_histogram,
    plot_matches_and_supporters,
    read_cameras,
    read_images,
    find_stereo_temporal_matches,
)
from perform_motion_estimation import perform_motion_estimation
from consts import FIRSTFRAME, LASTFRAME, FINAL_PLOTS_RELATIVE_PATH
from detector_config import matcher
from alg import (
    initialize_tracking,
    compose_extrinsics,
    process_stereo_pair,
)
import cv2


def updated_create_tracking_db():
    """
    Main processing loop with improved feature tracking.

    Uses the q6.py approach for better matching:
    1. Stereo matches validated through 3D reconstruction
    2. Temporal matches validated for 4-way correspondence
    """

    db, camera_left, camera_right, K, M1, M2 = initialize_tracking()

    prev_kp_left, prev_des_left, prev_kp_right, prev_des_right = None, None, None, None
    prev_features, prev_links = None, None
    prev_query_idx_to_features_idx_map = None
    prev_inliers_matches, prev_cloud = None, None
    tvec0 = M2[:, 3].reshape(3, 1)

    for frame_idx in tqdm(range(FIRSTFRAME, LASTFRAME + 1)):
        img_left, img_right = read_images(frame_idx)

        kp_left, desc_left, kp_right, desc_right, inliers_matches, cloud = (
            process_stereo_pair(img_left, img_right, camera_left, camera_right)
        )
        features, links = TrackingDB.create_links(
            features=desc_left,
            kp_left=kp_left,
            kp_right=kp_right,
            matches=[(m,) for m in inliers_matches],
        )
        query_idx_to_features_idx_map = {
            m.queryIdx: i for i, m in enumerate(inliers_matches)
        }
        if frame_idx == FIRSTFRAME:
            _ = db.add_frame(links, features)
            db.add_absolute_extrinsics(frame_idx, M1)
        else:
            total_matches_count = 0
            common_matches_count = 0
            supporters_count = 0

            matches_between_img0_img1 = matcher.match(prev_des_left, desc_left)
            total_matches_count = len(matches_between_img0_img1)

            common_matches, common_matches_indices = find_stereo_temporal_matches(
                prev_inliers_matches, inliers_matches, matches_between_img0_img1
            )
            common_matches_count = len(common_matches)

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
            supporters_count = len(supporters)

            prev_frame_size = db.frameId_to_lfeature[db.last_frameId].shape[0]

            matches_to_previous_left = [None] * prev_frame_size
            inliers = [False] * prev_frame_size

            prev_to_curr_idx = {cm[0]: cm[2] for cm in supporters}
            for i in range(prev_frame_size):
                prev_queryidx = prev_inliers_matches[i].queryIdx

                cur_queryidx = prev_to_curr_idx.get(prev_queryidx)
                if cur_queryidx is not None:
                    matches_to_previous_left[i] = cv2.DMatch(
                        _queryIdx=prev_query_idx_to_features_idx_map[prev_queryidx],
                        _trainIdx=query_idx_to_features_idx_map[cur_queryidx],
                        _distance=0,
                    )
                    inliers[i] = True

            _ = db.add_frame(
                links,
                features,
                matches_to_previous_left,
                inliers,
                total_matches_count,
                common_matches_count,
                supporters_count,
            )

            db.add_relative_extrinsics(frame_idx, relative_extrinsics_l1)
            absolute_extrinsics_l1 = compose_extrinsics(
                relative_extrinsics_l1, db.get_absolute_extrinsics(frame_idx - 1)
            )
            db.add_absolute_extrinsics(frame_idx, absolute_extrinsics_l1)

        prev_kp_left, prev_des_left = kp_left, desc_left
        prev_kp_right, prev_des_right = kp_right, desc_right
        prev_features, prev_links = features, links
        prev_query_idx_to_features_idx_map = query_idx_to_features_idx_map
        prev_inliers_matches, prev_cloud = inliers_matches, cloud
    return db


if __name__ == "__main__":
    output_directory = os.path.join(FINAL_PLOTS_RELATIVE_PATH, "ransac-pnp-final-plots")
    os.makedirs(output_directory, exist_ok=True)

    db = TrackingDB()
    db.load(os.path.join(output_directory, "tracking_enhanced"))

    K_intrinsic, M1_extrinsic, M2_extrinsic = read_cameras()

    camera_left_proj = K_intrinsic @ M1_extrinsic
    camera_right_proj = K_intrinsic @ M2_extrinsic

    all_frame_indices = list(db.all_frames())
    if not all_frame_indices:
        print("Loaded DB contains no frames. Cannot generate plots.")
        exit()

    track_lengths = [len(db.frames(track)) for track in db.all_tracks()]

    data = db.get_tracking_statistics()

    (
        total_tracks,
        total_frames,
        mean_track_length,
        min_track_length,
        max_track_length,
        mean_links_per_frame,
    ) = (
        data["total_tracks"],
        data["total_frames"],
        data["mean_track_length"],
        data["min_track_length"],
        data["max_track_length"],
        data["mean_links_per_frame"],
    )


    # plot_connectivity(db, os.path.join(output_directory, "db_connectivity"))

    # plot_track_length_histogram(db, os.path.join(output_directory, "db_track_length"))

    frame_indices_for_plots = sorted(db.frameId_to_total_matches_count.keys())
    total_left_left_matches_counts = [
        db.frameId_to_total_matches_count[idx] for idx in frame_indices_for_plots
    ]
    common_matches_4way_counts = [
        db.frameId_to_common_matches_count[idx] for idx in frame_indices_for_plots
    ]
    ransac_supporters_counts = [
        db.frameId_to_supporters_count[idx] for idx in frame_indices_for_plots
    ]

    plot_matches_and_supporters(
        frame_indices_for_plots,
        total_left_left_matches_counts,
        common_matches_4way_counts,
        ransac_supporters_counts,
        os.path.join(output_directory, "feature_analysis"),
    )
