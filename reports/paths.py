"""Compatibility names for scripts; new code should accept ProjectPaths explicitly."""
from kitti_slam.config import DEFAULT_PATHS, FIRST_FRAME, LAST_FRAME

FIRSTFRAME = FIRST_FRAME
LASTFRAME = LAST_FRAME
# Trailing separators preserve existing script string concatenation.
DATASET_RELATIVE_PATH = str(DEFAULT_PATHS.sequence_dir) + "/"
OUTPUT_RELATIVE_PATH = str(DEFAULT_PATHS.output_dir) + "/"
FINAL_PLOTS_RELATIVE_PATH = str(DEFAULT_PATHS.plots_dir) + "/"
DATA_PATH = DATASET_RELATIVE_PATH
OUTPUT_PATH = OUTPUT_RELATIVE_PATH
CALIB_FILE = str(DEFAULT_PATHS.calibration_file)
GT_POSES_FILE = str(DEFAULT_PATHS.poses_file)
LEFT_IMG_DIR = str(DEFAULT_PATHS.sequence_dir / "image_0")
RIGHT_IMG_DIR = str(DEFAULT_PATHS.sequence_dir / "image_1")
