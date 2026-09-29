"""Legacy pickle module path; new code should use kitti_slam.tracking_database."""

import sys
from kitti_slam import tracking_database

sys.modules[__name__] = tracking_database
