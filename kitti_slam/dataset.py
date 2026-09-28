"""Dataset readers; importing this module does not require GTSAM."""

import os
import cv2
import numpy as np
from .config import DEFAULT_PATHS


def parse_gt_line_matrix(line):
    """Parse one line of ground truth pose and return full extrinsic matrix."""
    mat = np.array([float(x) for x in line.strip().split()]).reshape(3, 4)
    return mat


def read_images(idx, paths=DEFAULT_PATHS):
    """Read left and right images from the dataset."""
    img_name = "{:06d}.png".format(idx)
    img0 = cv2.imread(os.path.join(paths.sequence_dir / "image_0", img_name), 0)
    img1 = cv2.imread(os.path.join(paths.sequence_dir / "image_1", img_name), 0)
    return img0, img1


def read_cameras(paths=DEFAULT_PATHS):
    with open(paths.calibration_file) as f:
        l1 = f.readline().split()[1:]
        l2 = f.readline().split()[1:]
        l1 = [float(i) for i in l1]
        m1 = np.array(l1).reshape(3, 4)
        l2 = [float(i) for i in l2]
        m2 = np.array(l2).reshape(3, 4)
        k = m1[:, :3]
        m1 = np.linalg.inv(k) @ m1
        m2 = np.linalg.inv(k) @ m2
        return k, m1, m2
