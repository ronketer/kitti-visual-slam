"""
Full SLAM pipeline runner with automatic checkpoint resume.

Stages:
  1. Feature tracking  → code/output/tracking_with_geometric_validation_without_far_tracks.pkl
  2. Bundle adjustment → code/output/ba_results.pkl
  3. Pose graph + loop closure → code/output/loop_closure_results.pkl

Run from the repo root:
  python run_pipeline.py           # skip stages whose .pkl output already exists
  python run_pipeline.py --force   # rerun all stages regardless of cache
"""

import argparse
import os
import pickle
import sys

# Allow importing pipeline modules from code/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "code"))

from config import DEFAULT_PATHS

TRACKING_PKL = str(DEFAULT_PATHS.output_dir / "tracking_with_geometric_validation_without_far_tracks.pkl")
BA_PKL = str(DEFAULT_PATHS.output_dir / "ba_results.pkl")
LOOP_PKL = str(DEFAULT_PATHS.output_dir / "loop_closure_results.pkl")


def run_stage_1(force: bool) -> None:
    if not force and os.path.exists(TRACKING_PKL):
        print("Stage 1: Feature Tracking — found cached result, skipping.")
        return

    print("Stage 1: Feature Tracking — building tracking database (this takes ~10 min)...")
    os.makedirs(DEFAULT_PATHS.output_dir, exist_ok=True)

    from tracking import build_tracking_database

    db = build_tracking_database()
    output_base = TRACKING_PKL.removesuffix(".pkl")
    db.serialize(output_base)
    print(f"Stage 1 complete. Saved to {TRACKING_PKL}")


def run_stage_2(force: bool) -> None:
    if not force and os.path.exists(BA_PKL):
        print("Stage 2: Bundle Adjustment — found cached result, skipping.")
        return

    print("Stage 2: Bundle Adjustment — running windowed GTSAM optimization...")
    import ex5
    ex5.main()
    print(f"Stage 2 complete. Saved to {BA_PKL}")


def run_stage_3(force: bool) -> None:
    if not force and os.path.exists(LOOP_PKL):
        print("Stage 3: Pose Graph + Loop Closure — found cached result, skipping.")
        return

    print("Stage 3: Pose Graph + Loop Closure — running ISAM2 optimization...")
    import pose_graph_loop_closure
    pose_graph_loop_closure.main()
    print(f"Stage 3 complete. Saved to {LOOP_PKL}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the full stereo visual SLAM pipeline."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ignore cached .pkl results and rerun all stages from scratch.",
    )
    args = parser.parse_args()

    print("=== Stereo Visual SLAM Pipeline ===\n")
    run_stage_1(args.force)
    run_stage_2(args.force)
    run_stage_3(args.force)
    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
