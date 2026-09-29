"""Historical ground-truth camera-center interpretation, without plotting."""
from ..config import DEFAULT_PATHS, LAST_FRAME
from ..dataset import parse_gt_line_matrix
from ..geometry import find_camera_location

def parse_gt_line(line):
    """Parse one line of ground truth pose and return camera location."""
    mat = parse_gt_line_matrix(line)
    return find_camera_location(mat)

def read_ground_truth_poses(lastframe=LAST_FRAME, *, paths=DEFAULT_PATHS):
    """Read and parse all ground truth poses up to LASTFRAME."""
    with open(paths.poses_file, "r") as f:
        return [parse_gt_line(line) for line in f.readlines()[: lastframe + 1]]
