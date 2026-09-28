"""Shared coursework trajectory plots, preserving their original definitions."""

import numpy as np
import matplotlib.pyplot as plt


def plot_trajectories(gt_poses, init_poses, opt_poses, label_every=5, save_path=None):
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
        init_xz[:, 0], init_xz[:, 1], "ro", label="Initial Trajectory", markersize=3
    )
    ax.plot(
        opt_xz[:, 0], opt_xz[:, 1], "bo", label="Optimized Trajectory", markersize=3
    )
    ax.plot(
        gt_xz[:last_frame, 0],
        gt_xz[:last_frame, 1],
        "go",
        label="Ground Truth Trajectory",
        markersize=3,
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


def plot_trajectory_errors(gt_centers, init_poses, opt_poses, save_path=None):
    frame_indices = sorted(init_poses.keys())
    init_errors = []
    opt_errors = []

    for i in frame_indices:
        gt = gt_centers[i]
        t_init = init_poses[i].translation()
        t_opt = opt_poses[i].translation()
        init_errors.append(np.linalg.norm(gt - t_init))
        opt_errors.append(np.linalg.norm(gt - t_opt))

    plt.figure(figsize=(10, 6))
    plt.plot(frame_indices, init_errors, "r-", label="Initial Estimate Error")
    plt.plot(frame_indices, opt_errors, "b-", label="Optimized Error")
    plt.title("3D Euclidean Distance to Ground Truth Over Frames")
    plt.xlabel("Frame Index")
    plt.ylabel("Distance (m)")
    plt.legend()
    plt.tight_layout()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()


def plot_relative_translation_errors(
    gt_matrices, t_rel_init, t_rel_opt, create_pose_from_extrinsics, save_path=None
):
    rel_errors_init = []
    rel_errors_opt = []
    frame_pairs = sorted(t_rel_init.keys())

    for i, j in frame_pairs:
        GT_i = create_pose_from_extrinsics(gt_matrices[i])
        GT_j = create_pose_from_extrinsics(gt_matrices[j])
        GT_rel = GT_i.between(GT_j)

        e_init = np.linalg.norm(t_rel_init[(i, j)].translation() - GT_rel.translation())
        e_opt = np.linalg.norm(t_rel_opt[(i, j)].translation() - GT_rel.translation())

        rel_errors_init.append(e_init)
        rel_errors_opt.append(e_opt)

    frame_indices = [i for (i, _) in frame_pairs]
    cum_init = np.cumsum(rel_errors_init)
    cum_opt = np.cumsum(rel_errors_opt)
    diff = np.array(rel_errors_init) - np.array(rel_errors_opt)

    fig, axs = plt.subplots(3, 1, figsize=(10, 16), layout="constrained")

    # Subplot 1: Relative errors
    axs[0].plot(frame_indices, rel_errors_init, "r-", label="Initial Relative Error")
    axs[0].plot(frame_indices, rel_errors_opt, "b-", label="Optimized Relative Error")
    axs[0].set_title("3D Translation Error of Relative Poses")
    axs[0].set_xlabel("Start Frame of Window")
    axs[0].set_ylabel("Distance (m)")
    axs[0].legend()
    axs[0].grid(True)

    # Subplot 2: Cumulative errors
    axs[1].plot(frame_indices, cum_init, "r-", label="Cumulative Initial Error")
    axs[1].plot(frame_indices, cum_opt, "b-", label="Cumulative Optimized Error")
    axs[1].set_title("Cumulative Relative Translation Error")
    axs[1].set_xlabel("Start Frame of Window")
    axs[1].set_ylabel("Cumulative Distance (m)")
    axs[1].legend()
    axs[1].grid(True)

    # Subplot 3: Difference (improvement)
    axs[2].plot(frame_indices, diff, "g-", label="Initial - Optimized Error")
    axs[2].set_title("Improvement in Relative Translation Error")
    axs[2].set_xlabel("Start Frame of Window")
    axs[2].set_ylabel("Error Difference (m)")
    axs[2].legend()
    axs[2].grid(True)

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close(fig)
