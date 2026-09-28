"""Explicit default inputs for the existing sequence-05 pipeline."""

from dataclasses import dataclass
from pathlib import Path
import os

_SOURCE_ROOT = Path(__file__).resolve().parents[1]
# Editable/source checkouts retain repository-relative defaults. A wheel uses an
# explicit workspace root, or the working directory when none is supplied.
REPOSITORY_ROOT = Path(os.environ.get(
    "KITTI_SLAM_ROOT",
    _SOURCE_ROOT if (_SOURCE_ROOT / "code").is_dir() and (_SOURCE_ROOT / "pyproject.toml").is_file()
    else Path.cwd(),
)).expanduser().resolve()
FIRST_FRAME = 0
LAST_FRAME = 2599
DEFAULT_DETECTOR = "AKAZE"


@dataclass(frozen=True)
class ProjectPaths:
    dataset_root: Path = REPOSITORY_ROOT / "dataset"
    sequence: str = "05"
    output_dir: Path = REPOSITORY_ROOT / "code" / "output"
    plots_dir: Path = REPOSITORY_ROOT / "code" / "final" / "plots"

    @property
    def sequence_dir(self):
        return Path(self.dataset_root) / "sequences" / self.sequence

    @property
    def calibration_file(self):
        return self.sequence_dir / "calib.txt"

    @property
    def poses_file(self):
        return Path(self.dataset_root) / "poses" / f"{self.sequence}.txt"


DEFAULT_PATHS = ProjectPaths()
