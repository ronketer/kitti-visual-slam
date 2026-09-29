from reports.paths import CHECKPOINT_DIR
from reports.paths import GT_POSES_FILE
import numpy as np
import matplotlib.pyplot as plt
import pickle

import sys
import os


from kitti_slam.dataset import parse_gt_line_matrix
from kitti_slam.geometry import find_camera_location
from reports.paths import REPORT_OUTPUT_DIR

def plot_trajectories(gt_poses, init_poses, opt_poses, pnp_poses, label_every=5, save_path=None):
    """
    Generates and plots a comparison of different trajectory estimation methods
    (Ground Truth, PnP, Bundle Adjustment, and Pose Graph) in the X-Z plane.

    Args:
        gt_poses (np.array): A NumPy array of shape (N, 3) containing the X, Y, Z
                             coordinates of the ground truth camera positions.
        init_poses (dict): A dictionary mapping frame IDs to initial Pose3 objects
                           from the Bundle Adjustment (before optimization).
        opt_poses (dict): A dictionary mapping frame IDs to optimized Pose3 objects
                          from the Pose Graph (with loop closure).
        pnp_poses (dict, optional): A dictionary mapping frame IDs to PnP-estimated
                                    Pose3 objects. If None, the PnP trajectory is not plotted.
        label_every (int, optional): The frequency (in frames) at which to add
                                     frame number labels to the optimized trajectory.
                                     Defaults to 5.
        save_path (str, optional): The file path to save the generated plot. If None,
                                   the plot is displayed. Defaults to None.
    """

    fig, ax = plt.subplots(figsize=(12, 8))
    last_frame = max(max(init_poses.keys()), max(opt_poses.keys()))
    ax.plot(
        gt_poses[:last_frame, 0],
        gt_poses[:last_frame, 2],
        "g--",
        label="Ground Truth",
        linewidth=1.5,
        alpha=0.5,
    )

    init_xz = np.array(
        [init_poses[i].translation()[[0, 2]] for i in sorted(init_poses)]
    )
    opt_xz = np.array([opt_poses[i].translation()[[0, 2]] for i in sorted(opt_poses)])
    gt_xz = np.array([[gt_poses[i][0], gt_poses[i][2]] for i in sorted(opt_poses)])

    ax.plot(
        init_xz[:, 0], init_xz[:, 1], "ro", label="Bundle Adjustment Trajectory", markersize=3
    )
    ax.plot(
        opt_xz[:, 0], opt_xz[:, 1], "bo", label="Pose Graph (with loop closure) Trajectory", markersize=3
    )
    ax.plot(
        gt_xz[:last_frame, 0],
        gt_xz[:last_frame, 1],
        "go",
        label="Ground Truth Trajectory",
        markersize=3,
    )

    # Plot PnP trajectory if provided
    if pnp_poses is not None:
        pnp_xz = np.array([pnp_poses[i].translation()[[0, 2]] for i in sorted(pnp_poses)])
        ax.plot(
            pnp_xz[:, 0], pnp_xz[:, 1], "c-", label="PnP Trajectory", linewidth=2, alpha=0.7
        )

    for idx, key in enumerate(sorted(opt_poses)):
        if idx % label_every == 0:
            t = opt_poses[key].translation()
            ax.text(t[0], t[2], str(key), fontsize=8, color="black")

    ax.set_title("Camera Trajectories Comparison (X-Z Plane)")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Z Position (m)")
    ax.legend()
    ax.grid(True)
    ax.axis("equal")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def main():


    # Load loop closure results
    with open(CHECKPOINT_DIR + "loop_closure_results.pkl", "rb") as f:
        results = pickle.load(f)
    init_poses = results["poses_without_loop_closure"]
    opt_poses = results["poses_with_loop_closure"]

    # Load ground truth poses and convert to camera centers
    gt_poses = []
    with open(GT_POSES_FILE, "r") as f:
        for line in f:
            extrinsic = parse_gt_line_matrix(line)
            center = find_camera_location(extrinsic)
            gt_poses.append(center)
    gt_poses = np.array(gt_poses)

    from kitti_slam.tracking_database import TrackingDB
    from kitti_slam.gtsam_geometry import create_pose_from_extrinsics

    # Load tracking database and extract PnP poses
    db = TrackingDB()
    db.load(CHECKPOINT_DIR + "tracking_with_geometric_validation_without_far_tracks")
    pnp_poses = {}
    for frame_id in db.all_frames():
        abs_extrinsics = db.get_absolute_extrinsics(frame_id)
        if abs_extrinsics is not None:
            pose = create_pose_from_extrinsics(abs_extrinsics)
            pnp_poses[frame_id] = pose

    save_path = REPORT_OUTPUT_DIR + "trajectory_comparison.png"
    plot_trajectories(gt_poses, init_poses, opt_poses, pnp_poses=pnp_poses, label_every=5, save_path=save_path)
    print(f"Plot saved to: {save_path}")


if __name__ == "__main__":
    from reports.paths import ensure_output_directories
    ensure_output_directories()
    main()
