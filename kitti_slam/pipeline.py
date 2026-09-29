"""Run the existing estimation stages with validated, separate runtime artifacts."""

import argparse
import importlib
import sys
from pathlib import Path

from .config import DEFAULT_PATHS, LAST_FRAME, REPOSITORY_ROOT
from .checkpoints import (
    CheckpointError, checkpoint_exists, load_checkpoint,
    save_checkpoint, save_tracking_checkpoint,
)

DEFAULT_OUTPUT_DIR = REPOSITORY_ROOT / "artifacts" / "sequence-05" / "checkpoints"
FILENAMES = {
    "tracking": "tracking_with_geometric_validation_without_far_tracks.pkl",
    "ba": "ba_results.pkl",
    "pose-graph": "loop_closure_results.pkl",
}


def checkpoint_path(kind, output_dir=DEFAULT_OUTPUT_DIR):
    return Path(output_dir) / FILENAMES[kind]


def run_stage_1(force: bool, *, output_dir=DEFAULT_OUTPUT_DIR) -> bool:
    output = checkpoint_path("tracking", output_dir)
    if not force and checkpoint_exists(output, "tracking"):
        print("Stage 1: Feature Tracking — validated cached result, skipping.")
        return False
    from .tracking import build_tracking_database

    print("Stage 1: Feature Tracking — building tracking database...")
    db = build_tracking_database()
    save_tracking_checkpoint(output, db)
    print(f"Stage 1 complete. Saved to {output}")
    return True


def run_stage_2(force: bool, *, output_dir=DEFAULT_OUTPUT_DIR) -> bool:
    source = checkpoint_path("tracking", output_dir)
    output = checkpoint_path("ba", output_dir)
    if not force and checkpoint_exists(output, "ba", upstream=source):
        print("Stage 2: Bundle Adjustment — validated cached result, skipping.")
        return False
    load_checkpoint(source, "tracking")
    from .tracking_database import TrackingDB
    from .dataset import read_cameras
    from .gtsam_geometry import read_calibration
    from .bundle_adjustment import run_bundle_adjustment

    print("Stage 2: Bundle Adjustment — running windowed GTSAM optimization...")
    db = TrackingDB()
    db.load(str(source.with_suffix("")))
    calibration = read_cameras()
    K = read_calibration(calibration=calibration)
    results = run_bundle_adjustment(db, K, reference_extrinsics=calibration[1])
    save_checkpoint(output, results, "ba", upstream=source)
    print(f"Stage 2 complete. Saved to {output}")
    return True


def run_stage_3(force: bool, *, output_dir=DEFAULT_OUTPUT_DIR) -> bool:
    source = checkpoint_path("ba", output_dir)
    output = checkpoint_path("pose-graph", output_dir)
    if not force and checkpoint_exists(output, "pose-graph", upstream=source):
        print("Stage 3: Pose Graph — validated cached result, skipping.")
        return False
    windows = load_checkpoint(source, "ba")
    from .dataset import parse_gt_line_matrix
    from .gtsam_geometry import create_pose_from_extrinsics
    from .pose_graph import run_pose_graph

    print("Stage 3: Pose Graph — batch LM with the historical ground-truth endpoint prior...")
    with open(DEFAULT_PATHS.poses_file, "r") as stream:
        endpoint_matrix = parse_gt_line_matrix(stream.readlines()[LAST_FRAME])
    results = run_pose_graph(
        windows, endpoint_frame=LAST_FRAME,
        endpoint_pose=create_pose_from_extrinsics(endpoint_matrix),
    )
    save_checkpoint(output, results, "pose-graph", upstream=source)
    print(f"Stage 3 complete. Saved to {output}")
    return True


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Rebuild selected stages and overwrite their runtime outputs.")
    parser.add_argument("--through", choices=list(FILENAMES), default="pose-graph",
                        help="Last stage to run; tracking works without GTSAM.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                        help="Checkpoint directory (default: artifacts/sequence-05/checkpoints).")
    parser.add_argument("--check", action="store_true",
                        help="Validate selected trusted checkpoints without running estimation or writing files.")
    args = parser.parse_args(argv)
    kinds = list(FILENAMES)[:list(FILENAMES).index(args.through) + 1]
    if args.check:
        errors = []
        for index, kind in enumerate(kinds):
            path = checkpoint_path(kind, args.output_dir)
            upstream = checkpoint_path(kinds[index - 1], args.output_dir) if index else None
            try:
                if not checkpoint_exists(path, kind, upstream=upstream):
                    raise CheckpointError(f"Missing checkpoint: {path}")
                print(f"{kind}: readable, required fields present")
            except CheckpointError as exc:
                errors.append(str(exc))
        for error in errors:
            print(error, file=sys.stderr)
        return int(bool(errors))
    try:
        # Fail before a long tracking run when later stages cannot execute.
        if args.through != "tracking":
            try:
                importlib.import_module("gtsam")
            except ImportError as exc:
                raise CheckpointError(
                    "GTSAM is unavailable. Use --through tracking on this environment; "
                    "run BA/pose graph later in a compatible GTSAM environment."
                ) from exc
        rebuild = args.force
        for stage in [run_stage_1, run_stage_2, run_stage_3][:len(kinds)]:
            rebuild = stage(rebuild, output_dir=args.output_dir) or rebuild
    except CheckpointError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print("Selected pipeline stages complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
