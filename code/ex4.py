from consts import OUTPUT_RELATIVE_PATH
from tracking import build_tracking_database as create_tracking_db
import numpy as np
import cv2
import matplotlib.pyplot as plt
from utility import read_images, read_cameras, compose_extrinsics, coordinate_transform
from tracking_database import TrackingDB

OUTPUT_PATH = OUTPUT_RELATIVE_PATH + "tracking_with_geometric_validation_without_far_tracks"

def q42(db):
    all_tracks = db.all_tracks()
    non_trivial_tracks = [track_id for track_id in all_tracks if len(db.frames(track_id)) > 1]

    trackid_to_length = {track_id: len(db.frames(track_id)) for track_id in non_trivial_tracks}
    max_length = max(trackid_to_length.values())
    min_length = min(trackid_to_length.values())

    track_min = [track for track, l in trackid_to_length.items() if l == min_length]
    track_max = [track for track, l in trackid_to_length.items() if l == max_length]

    track_mean_length = sum(trackid_to_length.values()) / len(trackid_to_length)

    all_frames = db.all_frames()
    frame_tracks_counts = [len(db.tracks(frame_id)) for frame_id in all_frames]
    mean_frames_length = sum(frame_tracks_counts) / len(all_frames)

    print(f"Total number of tracks: {len(non_trivial_tracks)}")
    print(f"Number of frames: {len(all_frames)}")
    print(f"Mean track length: {track_mean_length}")
    print(f"Minimum and maximum track lengths: {min_length}, {max_length}")
    print(f"Mean number of frame links: {mean_frames_length}")

def q43(db):
    all_tracks = db.all_tracks()
    track_to_length_over_six = {
        track_id: len(db.frames(track_id))
        for track_id in all_tracks
        if len(db.frames(track_id)) >= 6
    }
    if not track_to_length_over_six:
        print("No tracks with length >= 6 found.")
        return

    track_id = np.random.choice(list(track_to_length_over_six.keys()))
    frames = db.frames(track_id)

    drawn_images = []
    zoomed_patches = []

    for frame_id, link in db.track(track_id).items():
        img_left, _ = read_images(frame_id)
        height, width = img_left.shape

        x, y = int(link.left_keypoint()[0]), int(link.left_keypoint()[1])

        x1 = max(0, x - 10)
        x2 = min(width, x + 10)
        y1 = max(0, y - 10)
        y2 = min(height, y + 10)

        img_drawn = cv2.cvtColor(img_left.copy(), cv2.COLOR_GRAY2BGR)
        cv2.rectangle(img_drawn, (x1, y1), (x2 - 1, y2 - 1), (0, 0, 255), 3)
        drawn_images.append(img_drawn)

        patch = img_left[y1:y2, x1:x2].copy()
        patch_color = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)

        cx, cy = x - x1, y - y1
        if 0 <= cx < patch_color.shape[1] and 0 <= cy < patch_color.shape[0]:
            cv2.drawMarker(patch_color, (cx, cy), (0, 0, 255), cv2.MARKER_TILTED_CROSS, 5, 1)

        zoomed_patches.append(patch_color)

    num = len(drawn_images)
    fig, axs = plt.subplots(num, 2, figsize=(8, 2 * num))

    for i in range(num):
        axs[i, 0].imshow(cv2.cvtColor(drawn_images[i], cv2.COLOR_BGR2RGB))
        axs[i, 0].axis("off")

        axs[i, 1].imshow(cv2.cvtColor(zoomed_patches[i], cv2.COLOR_BGR2RGB))
        axs[i, 1].axis("off")

    plt.tight_layout()
    plt.suptitle(f"Track #{track_id}, Length: {len(frames)}", y=1.02)
    plt.savefig(OUTPUT_RELATIVE_PATH + "ex4_q43.png")
    # plt.show()


def q44(db):
    all_frames = db.all_frames()
    frame_id_to_num_outgoing_tracks = {fid: 0 for fid in all_frames}

    for frame_id in all_frames[:len(all_frames)-1]:
        for track_id in db.tracks(frame_id):
            if (frame_id + 1) in db.frames(track_id):
                frame_id_to_num_outgoing_tracks[frame_id] += 1

    mean_num_outgoing_tracks = np.mean(list(frame_id_to_num_outgoing_tracks.values()))
    fig, ax = plt.subplots(layout='constrained')
    ax.set_xlabel("Frame")
    ax.set_ylabel("Outgoing Tracks")
    ax.set_title("Connectivity")
    ax.plot(list(frame_id_to_num_outgoing_tracks.keys()),
            list(frame_id_to_num_outgoing_tracks.values()))
    ax.axhline(mean_num_outgoing_tracks, color='red', linestyle='--', label='Mean')
    plt.savefig(OUTPUT_RELATIVE_PATH+"ex4_q44.png")
    # plt.show()

def q45(db):
    all_frames = db.all_frames()
    frame_id_to_num_outgoing_tracks = {fid: 0 for fid in all_frames}
    frame_id_to_percentage = {}

    for frame_id in all_frames[:len(all_frames)-1]:
        for track_id in db.tracks(frame_id):
            if (frame_id + 1) in db.frames(track_id):
                frame_id_to_num_outgoing_tracks[frame_id] += 1
        frame_id_to_percentage[frame_id] = frame_id_to_num_outgoing_tracks[frame_id] / len(db.tracks(frame_id))

    fig, ax = plt.subplots(layout='constrained')
    ax.set_xlabel("Frame")
    ax.set_ylabel("Percentage")
    ax.set_title("Inliers percentage")
    ax.plot(list(frame_id_to_percentage.keys()), list(frame_id_to_percentage.values()))
    plt.savefig(OUTPUT_RELATIVE_PATH + "ex4_q45.png")
    # plt.show()


def q46(db):
    all_tracks = db.all_tracks()
    track_lengths = [len(db.frames(track)) for track in all_tracks if len(db.frames(track)) > 2]

    fig, ax = plt.subplots(layout='constrained')
    ax.hist(track_lengths, bins=np.arange(0, max(track_lengths) + 10, 2), edgecolor='black', alpha=0.7)
    ax.set_xlabel("Track length")
    ax.set_ylabel("Track count")
    ax.set_title("Track length histogram")
    ax.set_yscale("log")
    ax.set_xticks(np.arange(0, max(track_lengths) + 10, 10))
    plt.savefig(OUTPUT_RELATIVE_PATH + "ex4_q46.png")
    # plt.show()

def q47(db):
    # 1. Read ground truth poses
    with open("dataset/poses/05.txt", 'r') as f:
        gt_poses = [np.array([float(x) for x in line.strip().split()]).reshape(3, 4) for line in f.readlines()]
    K, M1, M2 = read_cameras()

    all_tracks = db.all_tracks()
    track_id_of_over_ten = [tid for tid in all_tracks if len(db.frames(tid)) >= 10]

    np.random.seed(2)
    track_id = np.random.choice(track_id_of_over_ten)
    frame_ids = db.frames(track_id)
    last_frame = frame_ids[-1]
    link = db.link(last_frame, track_id)
    pt_left = link.left_keypoint()
    pt_right = link.right_keypoint()
    extrinsic_left = gt_poses[last_frame][:3, :4]
    extrinsic_right = compose_extrinsics(M2, extrinsic_left)

    # Triangulate the point in 3D space
    real_world_track_location = cv2.triangulatePoints(K @ extrinsic_left, K @ extrinsic_right, pt_left.reshape(2, 1), pt_right.reshape(2, 1))
    real_world_track_location_3D = (real_world_track_location / real_world_track_location[3])[:3].T

    # Project to all frames and compute reprojection errors
    errors_left, errors_right = [], []
    for frame_id in reversed(frame_ids):
        cur_extrinsic_left = gt_poses[frame_id]
        cur_extrinsic_right = compose_extrinsics(M2, cur_extrinsic_left)
        # Transform and project points
        points_transformed_left = coordinate_transform(real_world_track_location_3D, cur_extrinsic_left)
        projected_points_left = (K @ points_transformed_left.T).T
        # Normalize homogeneous coordinates
        projected_points_2D_left = projected_points_left[:, :2] / projected_points_left[:, 2:3]

        points_transformed_right = coordinate_transform(real_world_track_location_3D, cur_extrinsic_right)
        projected_points_right = (K @ points_transformed_right.T).T
        # Normalize homogeneous coordinates
        projected_points_2D_right = projected_points_right[:, :2] / projected_points_right[:, 2:3]
        # Get actual feature locations
        link = db.link(frame_id, track_id)
        # Compute L2 reprojection errors
        errors_left.append(np.linalg.norm(projected_points_2D_left - link.left_keypoint()))
        errors_right.append(np.linalg.norm(projected_points_2D_right - link.right_keypoint()))

    fig, ax = plt.subplots(layout="constrained")
    ax.plot(errors_left, label='Left')
    ax.plot(errors_right, label='Right')
    ax.set_title("PnP - projection error vs track length")
    ax.set_xlabel("distance from reference (frames)")
    ax.set_ylabel("projection error (pixels)")
    ax.legend()
    plt.savefig(OUTPUT_RELATIVE_PATH + "ex4_q47.png")
    # plt.show()


def main():
    
    db = create_tracking_db()
    # Save final database
    db.serialize(OUTPUT_PATH)
    print(f"Tracking database saved to {OUTPUT_PATH}.pkl")
    q42(db)
    q43(db)
    q44(db)
    q45(db)
    q46(db)
    q47(db)

if __name__ == "__main__":
    main()
"""
TrackingDB serialized to ./code/output/tracking_with_geometric_validation.pkl
Tracking database saved to ./code/output/tracking_with_geometric_validation.pkl
Total number of tracks: 290258
Number of frames: 2600
Mean track length: 5.270280233447485
Minimum and maximum track lengths: 2, 200
Mean number of frame links: 588.3619230769231
"""
"""
TrackingDB serialized to ./code/output/tracking_with_geometric_validation_without_far_tracks.pkl
Tracking database saved to ./code/output/tracking_with_geometric_validation_without_far_tracks.pkl
Total number of tracks: 287158
Number of frames: 2600
Mean track length: 5.191180465109801
Minimum and maximum track lengths: 2, 163
Mean number of frame links: 573.3419230769231
"""