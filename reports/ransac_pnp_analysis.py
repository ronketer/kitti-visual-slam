"""Report plots from a tracking checkpoint with diagnostic counts."""
import sys
import os

from kitti_slam.tracking_database import TrackingDB
from kitti_slam.dataset import read_cameras
from reports.plots import plot_matches_and_supporters
from reports.paths import REPORT_OUTPUT_DIR, HISTORICAL_REPORT_DIR


if __name__ == "__main__":
    from reports.paths import ensure_output_directories
    ensure_output_directories()
    output_directory = os.path.join(REPORT_OUTPUT_DIR, "ransac-pnp-final-plots")
    os.makedirs(output_directory, exist_ok=True)

    db = TrackingDB()
    db.load(os.path.join(HISTORICAL_REPORT_DIR, "ransac-pnp-final-plots", "tracking_enhanced"))

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
