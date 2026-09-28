"""Extract trajectories from estimation results without plotting or file I/O."""

from tqdm import tqdm
from .gtsam_geometry import create_pose_from_extrinsics


def poses_from_values(values, pose_keys):
    """Extract available Pose3 values using the caller's frame-to-key mapping."""
    return {
        frame_id: values.atPose3(key)
        for frame_id, key in pose_keys.items()
        if values.exists(key)
    }


def bundle_trajectories(windows):
    """Return initial/optimized endpoints and relative poses used by ex5 plots.

    Preserve the original convention: a window's endpoint pose is treated as
    its relative pose, even though its first-pose prior is soft. Correcting
    that convention is a separate mathematical change.
    """
    import gtsam

    identity = gtsam.Pose3()
    initial_poses = {0: identity}
    optimized_poses = {0: identity}
    relative_initial = {}
    relative_optimized = {}
    for window in windows:
        start, end = window["start_kf"], window["end_kf"]
        end_key = window["pose_keys"][end]
        initial_endpoint = window["initialEstimate"].atPose3(end_key)
        optimized_endpoint = window["result"].atPose3(end_key)
        initial_poses[end] = initial_poses[start].compose(initial_endpoint)
        optimized_poses[end] = window["abs_end_pose"]
        relative_initial[(start, end)] = initial_endpoint
        relative_optimized[(start, end)] = optimized_endpoint
    return initial_poses, optimized_poses, relative_initial, relative_optimized


def extract_pnp_poses(db):
    """
    Extract PnP poses from TrackingDB.

    Args:
        db: TrackingDB instance

    Returns:
        dict: frame_id -> gtsam.Pose3 mapping
    """
    print("Extracting PnP poses from TrackingDB...")
    pnp_poses = {}

    for frame_id in tqdm(db.all_frames(), desc="Processing PnP poses"):
        abs_extrinsics = db.get_absolute_extrinsics(frame_id)
        if abs_extrinsics is not None:
            # Convert 4x4 extrinsics matrix to gtsam.Pose3
            pose = create_pose_from_extrinsics(abs_extrinsics)
            pnp_poses[frame_id] = pose

    print(f"Extracted {len(pnp_poses)} PnP poses")
    return pnp_poses


def extract_bundle_poses(windows_graph_list):
    """
    Extract Bundle Adjustment poses from bundle results using existing absolute poses.

    Args:
        windows_graph_list: List of bundle adjustment results

    Returns:
        dict: frame_id -> gtsam.Pose3 mapping
    """
    print("Extracting Bundle Adjustment poses...")
    bundle_poses = {}

    for window_dict in tqdm(windows_graph_list, desc="Processing bundle windows"):
        start_kf = window_dict["start_kf"]
        end_kf = window_dict["end_kf"]

        # Use the absolute poses directly from the windows graph list
        abs_start_pose = window_dict["abs_start_pose"]
        abs_end_pose = window_dict["abs_end_pose"]

        bundle_poses[start_kf] = abs_start_pose
        bundle_poses[end_kf] = abs_end_pose

    print(f"Extracted {len(bundle_poses)} Bundle Adjustment poses")
    return bundle_poses


def recompose_bundle_poses(windows_graph_list):
    """
    Extract Bundle Adjustment poses from bundle results.

    Args:
        windows_graph_list: List of bundle adjustment results

    Returns:
        dict: frame_id -> gtsam.Pose3 mapping
    """
    print("Extracting Bundle Adjustment poses...")
    bundle_poses = {}

    import gtsam

    # Start with identity pose at frame 0
    bundle_poses[0] = gtsam.Pose3()
    current_pose = gtsam.Pose3()

    for window_dict in tqdm(windows_graph_list, desc="Processing bundle windows"):
        start_kf = window_dict["start_kf"]
        end_kf = window_dict["end_kf"]
        result = window_dict["result"]
        pose_keys = window_dict["pose_keys"]

        # Get relative pose from bundle result
        if start_kf in pose_keys and end_kf in pose_keys:
            relative_pose = result.atPose3(pose_keys[end_kf])

            # Update current pose (compose with relative pose)
            if start_kf in bundle_poses:
                current_pose = bundle_poses[start_kf]

            absolute_pose = current_pose.compose(relative_pose)
            bundle_poses[end_kf] = absolute_pose

    print(f"Extracted {len(bundle_poses)} Bundle Adjustment poses")
    return bundle_poses
