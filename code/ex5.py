from consts import OUTPUT_RELATIVE_PATH, LASTFRAME
from alg import (
    create_tracking_db,
    create_stereo_camera,
    create_pose_from_extrinsics,
    initialize_factor_graph_in_window,
)
from detector_config import DETECTOR
from tracking_database import TrackingDB
from utility import find_camera_location, read_calibration, read_images, crop_patch
import gtsam
import gtsam.utils.plot as gtsam_plot
import numpy as np
import matplotlib.pyplot as plt
from window_selector import WindowSelector
from tqdm import tqdm
import cv2
import pickle

OUTPUT_PATH = OUTPUT_RELATIVE_PATH + f"tracking_with_geometric_validation_without_far_tracks"

def q1(db, K):
    np.random.seed(0)
    all_tracks = db.all_tracks()
    track_id_of_over_ten = [tid for tid in all_tracks if len(db.frames(tid)) >= 10]

    track_id = np.random.choice(track_id_of_over_ten)
    frame_ids = db.frames(track_id)
    last_frame = frame_ids[-1]
    link = db.link(last_frame, track_id)
    xleft = link.left_keypoint()[0]
    y = link.left_keypoint()[1]
    xright = link.right_keypoint()[0]
    steroPoint = gtsam.StereoPoint2(xleft, xright, y)
    last_frame_extrinsics = db.get_absolute_extrinsics(last_frame)
    if last_frame_extrinsics is None:
        raise ValueError(f"Frame {last_frame} has no absolute extrinsics.")
    frame = create_stereo_camera(last_frame_extrinsics, K)
    point3 = frame.backproject(steroPoint)

    # Create factor graph
    graph = gtsam.NonlinearFactorGraph()
    point_key = gtsam.symbol("p", 0)  # Single 3D point key

    # Initialize values with initial point
    initial = gtsam.Values()
    initial.insert(point_key, point3)

    errors_left, errors_right = [], []
    factor_errors = []
    sigma = gtsam.noiseModel.Diagonal.Sigmas(np.array([1.0, 1.0, 1.0]))

    # Process frames in reverse order to maintain distance from reference
    for frame_idx in reversed(frame_ids):
        extrinsics = db.get_absolute_extrinsics(frame_idx)
        if extrinsics is None:
            raise ValueError(f"Frame {frame_idx} has no absolute extrinsics.")

        # Create camera and calculate original projection errors
        frame = create_stereo_camera(extrinsics, K)
        steroPoint = frame.project(point3)
        link = db.link(frame_idx, track_id)
        errors_left.append(
            np.linalg.norm(
                link.left_keypoint() - np.array([steroPoint.uL(), steroPoint.v()])
            )
        )
        errors_right.append(
            np.linalg.norm(
                link.right_keypoint() - np.array([steroPoint.uR(), steroPoint.v()])
            )
        )

        pose_key = gtsam.symbol("p", frame_idx)  # Arbitrary offset for pose keys
        pose3 = create_pose_from_extrinsics(extrinsics)
        # Add stereo factor for this frame
        stereo_measurement = gtsam.StereoPoint2(
            link.left_keypoint()[0], link.right_keypoint()[0], link.left_keypoint()[1]
        )
        factor = gtsam.GenericStereoFactor3D(
            stereo_measurement, sigma, pose_key, point_key, K
        )
        graph.add(factor)

        # Add pose to initial estimate
        initial.insert(pose_key, pose3)

    # Optimize factor graph
    # params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial)
    result = optimizer.optimize()

    # Calculate factor errors after optimization
    initial_factor_errors = [graph.at(i).error(initial) for i in range(graph.size())]
    factor_errors = [graph.at(i).error(result) for i in range(graph.size())]

    # Create figure with three subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, layout="constrained", figsize=(10, 12))

    # Plot original projection errors
    ax1.plot(errors_left, label="Left")
    ax1.plot(errors_right, label="Right")
    ax1.set_title("PnP - projection error vs track length")
    ax1.set_xlabel("distance from reference (frames)")
    ax1.set_ylabel("projection error (pixels)")
    ax1.legend()

    # Plot initial factor graph errors
    ax2.plot(initial_factor_errors, label="Initial Factor Error", color="blue")
    ax2.set_title("Factor Graph - Initial Error vs Track Length")
    ax2.set_xlabel("distance from reference (frames)")
    ax2.set_ylabel("factor error")
    ax2.legend()

    # Plot optimized factor errors
    ax3.plot(factor_errors, label="Optimized Factor Error", color="orange")
    ax3.set_title("Factor Graph - Optimized Error vs Track Length")
    ax3.set_xlabel("distance from reference (frames)")
    ax3.set_ylabel("factor error")
    ax3.legend()

    plt.savefig(OUTPUT_RELATIVE_PATH + "ex5_q1.png")
    # plt.show()


def parse_gt_line(line):
    mat = np.array([float(x) for x in line.strip().split()]).reshape(3, 4)
    return mat  # Return full extrinsic matrix [R|t]


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


def plot_factor_projection(frame_id, measurement, projection, save_path, title):
    left_gray, right_gray = read_images(frame_id)
    left_img = cv2.cvtColor(left_gray, cv2.COLOR_GRAY2RGB)
    right_img = cv2.cvtColor(right_gray, cv2.COLOR_GRAY2RGB)

    cv2.drawMarker(
        left_img,
        (int(projection.uL()), int(projection.v())),
        (0, 0, 255),
        cv2.MARKER_TILTED_CROSS,
        12,
        1,
    )
    cv2.drawMarker(
        left_img,
        (int(measurement.uL()), int(measurement.v())),
        (0, 255, 0),
        cv2.MARKER_CROSS,
        12,
        1,
    )

    cv2.drawMarker(
        right_img,
        (int(projection.uR()), int(projection.v())),
        (0, 0, 255),
        cv2.MARKER_TILTED_CROSS,
        12,
        1,
    )
    cv2.drawMarker(
        right_img,
        (int(measurement.uR()), int(measurement.v())),
        (0, 255, 0),
        cv2.MARKER_CROSS,
        12,
        1,
    )

    patch_left = crop_patch(left_img, (int(projection.uL()), int(projection.v())), 50)
    patch_right = crop_patch(right_img, (int(projection.uR()), int(projection.v())), 50)

    fig, axs = plt.subplots(2, 2, figsize=(40, 10), layout="constrained")
    fig.suptitle(title, fontsize=16)

    axs[0, 0].imshow(left_img)
    axs[0, 0].set_title("Left Image")
    axs[0, 0].axis("off")

    axs[0, 1].imshow(right_img)
    axs[0, 1].set_title("Right Image")
    axs[0, 1].axis("off")

    axs[1, 0].imshow(patch_left)
    axs[1, 0].axis("off")

    axs[1, 1].imshow(patch_right)
    axs[1, 1].axis("off")

    plt.savefig(save_path, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


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


def plot_3d_pose_from_above(result, pose_keys, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title("2D View from Above: Camera Trajectory")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Z Position (m)")
    ax.axis("equal")
    ax.grid(True)
    cam_x = [result.atPose3(pose_keys[i]).translation()[0] for i in pose_keys]
    cam_z = [result.atPose3(pose_keys[i]).translation()[2] for i in pose_keys]
    ax.plot(cam_x, cam_z, "bo-", label="Camera Trajectory", markersize=3, alpha=0.8)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    else:
        plt.show()
    plt.close(fig)


def plot_landmarks_from_above(result, point_keys, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_title("2D View from Above: Landmarks")
    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Z Position (m)")

    ax.axis("equal")
    ax.set_ylim(0, 750)
    ax.set_xlim(-400, 400)

    ax.grid(True)

    point_x = [result.atPoint3(point_keys[i])[0] for i in point_keys]
    point_z = [result.atPoint3(point_keys[i])[2] for i in point_keys]
    ax.scatter(point_x, point_z, c="r", marker="o", label="Landmarks", s=8, alpha=0.5)
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", pad_inches=0)
    else:
        plt.show()
    plt.close(fig)


def analyze_factor(factor, initialEstimate, result, pose_keys, point_keys, K):
    """
    Analyze a given projection factor: print errors, project points, and print distances
    for both initial and optimized values.
    """

    _, fid, tid = factor
    initial_error = factor[0].error(initialEstimate)
    optimized_error = factor[0].error(result)
    print("factor analysis:")
    print(f"Frame (c): {fid}, Landmark (q): {tid}")
    print(f"Initial error: {initial_error:.4f}")
    print(f"Optimized error: {optimized_error:.4f}")

    pose_c_init = initialEstimate.atPose3(pose_keys[fid])
    point_q_init = initialEstimate.atPoint3(point_keys[tid])
    stereo_cam_init = gtsam.StereoCamera(pose_c_init, K)
    proj_init = stereo_cam_init.project(point_q_init)
    measurement = factor[0].measured()

    pose_c_opt = result.atPose3(pose_keys[fid])
    point_q_opt = result.atPoint3(point_keys[tid])
    stereo_cam_opt = gtsam.StereoCamera(pose_c_opt, K)
    proj_opt = stereo_cam_opt.project(point_q_opt)

    initial_dist_left = np.linalg.norm(
        np.array([proj_init.uL(), proj_init.v()])
        - np.array([measurement.uL(), measurement.v()])
    )
    initial_dist_right = np.linalg.norm(
        np.array([proj_init.uR(), proj_init.v()])
        - np.array([measurement.uR(), measurement.v()])
    )
    print(f"Initial distance left: {initial_dist_left:.2f} px")
    print(f"Initial distance right: {initial_dist_right:.2f} px")

    optimized_dist_left = np.linalg.norm(
        np.array([proj_opt.uL(), proj_opt.v()])
        - np.array([measurement.uL(), measurement.v()])
    )
    optimized_dist_right = np.linalg.norm(
        np.array([proj_opt.uR(), proj_opt.v()])
        - np.array([measurement.uR(), measurement.v()])
    )
    print(f"Optimized distance left: {optimized_dist_left:.2f} px")
    print(f"Optimized distance right: {optimized_dist_right:.2f} px")

    return fid, measurement, proj_init, proj_opt


def q3(db, K):
    """
    Perform Bundle Adjustment on the first window of keyframes.
    """

    window_selector = WindowSelector(db)
    window = window_selector.next_window()
    if window is None:
        print("Could not find a suitable window for Bundle Adjustment.")
        return
    start_kf, end_kf = window
    print(f"Selected first window from keyframe {start_kf} to {end_kf}")

    graph, initialEstimate, pose_keys, point_keys, tracks, frames, _, factors = (
        initialize_factor_graph_in_window(db, start_kf, end_kf, K)
    )

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initialEstimate)
    result = optimizer.optimize()

    total_initial_error, total_optimized_error = (
        graph.error(initialEstimate),
        graph.error(result),
    )
    initial_avg_error = total_initial_error / graph.size()
    optimized_avg_error = total_optimized_error / graph.size()

    print(f"Number of factors in the graph: {graph.size()}")
    print(f"Total initial error: {total_initial_error:.4f}")
    print(f"Total optimized error: {total_optimized_error:.4f}")
    print(
        f"average factor error: before optimization: {initial_avg_error:.4f}, after optimization: {optimized_avg_error:.4f}"
    )

    worst_factor = max(factors, key=lambda f: f[0].error(initialEstimate))
    fid, measurement, proj_init, proj_opt = analyze_factor(
        worst_factor, initialEstimate, result, pose_keys, point_keys, K
    )
    plot_factor_projection(
        frame_id=fid,
        measurement=measurement,
        projection=proj_init,
        save_path=OUTPUT_RELATIVE_PATH + "ex5_q3_initial_projection.png",
        title=f"Initial Projection vs Measurement - Frame {fid}",
    )

    plot_factor_projection(
        frame_id=fid,
        measurement=measurement,
        projection=proj_opt,
        save_path=OUTPUT_RELATIVE_PATH + "ex5_q3_optimized_projection.png",
        title=f"Optimized Projection vs Measurement - Frame {fid}",
    )

    gtsam_plot.plot_trajectory(
        fignum=0,
        scale=4,
        values=result,
        title="3D trajectory after Bundle Adjustment",
        axis_labels=("X axis (m)", "Y axis (m)", "Z axis (m)"),
    )
    gtsam_plot.set_axes_equal(0)
    plt.savefig(OUTPUT_RELATIVE_PATH + "ex5_q3_3d.png")

    plot_3d_pose_from_above(
        result, pose_keys, save_path=OUTPUT_RELATIVE_PATH + "ex5_q3_2d_camera.png"
    )
    plot_landmarks_from_above(
        result, point_keys, save_path=OUTPUT_RELATIVE_PATH + "ex5_q3_2d_landmarks.png"
    )


def q4(db, K):
    gt_matrices = []
    with open("dataset/poses/05.txt", "r") as f:
        gt_matrices = [parse_gt_line(line) for line in f.readlines()[: LASTFRAME + 1]]
    gt_centers = np.array([find_camera_location(m) for m in gt_matrices])

    window_selector = WindowSelector(db)
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
    init_estimate_global_poses = {0: T_global}
    t_rel_init = {}
    t_rel_opt = {}

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
        ) = initialize_factor_graph_in_window(db, start_kf, end_kf, K)
        T_rel_init = initialEstimate.atPose3(pose_keys[end_kf])
        T_start_init = init_estimate_global_poses[start_kf]
        T_abs_init = T_start_init.compose(T_rel_init)
        init_estimate_global_poses[end_kf] = T_abs_init
        t_rel_init[(start_kf, end_kf)] = T_rel_init
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
        t_rel_opt[(start_kf, end_kf)] = T_rel_opt

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

        if i == max_iterations - 1:
            error = prior_factor.error(result)
            print(
                f"Anchoring factor error for last window ({start_kf}, {end_kf}): {error}"
            )
            bundle_first_frame = optimizied_global_poses[start_kf]
            print(
                f"Optimized position of first frame in last bundle ({start_kf}): {bundle_first_frame.translation()}"
            )

    plot_trajectories(
        gt_centers,
        init_estimate_global_poses,
        optimizied_global_poses,
        label_every=5,
        save_path=OUTPUT_RELATIVE_PATH + f"ex5_trajectory.png",
    )

    plot_trajectory_errors(
        gt_centers,
        init_estimate_global_poses,
        optimizied_global_poses,
        save_path=OUTPUT_RELATIVE_PATH + f"ex5_trajectory_error.png",
    )

    plot_relative_translation_errors(
        gt_matrices,
        t_rel_init,
        t_rel_opt,
        create_pose_from_extrinsics,
        save_path=OUTPUT_RELATIVE_PATH + f"ex5_relative_error.png",
    )
    # At the end of q4, ensure it returns all_ba_results

    return windows_graph_list


def main():
    # db = create_tracking_db()
    # # Save final database
    # db.serialize(OUTPUT_PATH)
    # print(f"Tracking database saved to {OUTPUT_PATH}.pkl")
    db = TrackingDB()
    db.load(OUTPUT_PATH)
    K = read_calibration()

    q1(db, K)
    q3(db, K)

    bundle_results = q4(db, K)

    result_save_path = OUTPUT_RELATIVE_PATH + "ba_results.pkl"
    print(f"Saving BA results to {result_save_path}...")
    with open(result_save_path, "wb") as f:
        # Save all the necessary data in a single tuple or dictionary
        pickle.dump(
            bundle_results,
            f,
        )
    print("BA results saved.")


if __name__ == "__main__":
    main()

"""
Selected first window from keyframe 0 to 15
Number of factors in the graph: 10543
Total initial error: 12382.5391
Total optimized error: 2045.6831
average factor error: before optimization: 1.1745, after optimization: 0.1940
factor analysis:
Frame (c): 11, Landmark (q): 1910
Initial error: 59.2802
Optimized error: 0.9457
Initial distance left: 8.69 px
Initial distance right: 6.94 px
Optimized distance left: 1.15 px
Optimized distance right: 1.27 px
Ignoring fixed x limits to fulfill fixed data aspect with adjustable data limits.
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████▎| 229/230 [01:18<00:00,  3.20it/s]Anchoring factor error for last window (2590, 2599): 8.088593044821155e-09
Optimized position of first frame in last bundle (2590): [-15.65374653  -9.20064505 202.06543106]
"""
