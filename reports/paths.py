"""Input archives and generated destinations for coursework/report programs."""
from pathlib import Path
from kitti_slam.config import DEFAULT_PATHS, REPOSITORY_ROOT, FIRST_FRAME, LAST_FRAME

# Strings include a separator because existing report functions concatenate filenames.
CHECKPOINT_DIR = str(DEFAULT_PATHS.output_dir) + "/"
COURSEWORK_OUTPUT_DIR = str(REPOSITORY_ROOT / "artifacts" / "coursework") + "/"
REPORT_OUTPUT_DIR = str(DEFAULT_PATHS.plots_dir) + "/"
HISTORICAL_REPORT_DIR = str(REPOSITORY_ROOT / "results" / "historical" / "final") + "/"
GT_POSES_FILE = str(DEFAULT_PATHS.poses_file)


def ensure_output_directories():
    """Called by script entry points, never as an import side effect."""
    for directory in (CHECKPOINT_DIR, COURSEWORK_OUTPUT_DIR, REPORT_OUTPUT_DIR):
        Path(directory).mkdir(parents=True, exist_ok=True)
