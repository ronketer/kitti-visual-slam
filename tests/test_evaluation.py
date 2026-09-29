"""Characterize preserved evaluation policies independently of real GTSAM."""

import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from kitti_slam.evaluation import trajectory_errors as metrics
from kitti_slam import trajectory
class EvaluationTests(unittest.TestCase):
    def test_rotation_metric_uses_angle_in_degrees(self):
        rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        pose = Mock()
        pose.between.return_value.rotation.return_value.matrix.return_value = rotation
        self.assertAlmostEqual(metrics.calculate_rotation_error_degrees(pose, object()), 90.)

    def test_subsequence_keeps_half_selection_snapping_and_normalization(self):
        estimates = {fid: object() for fid in range(0, 80, 10)}
        with patch.object(metrics, "create_pose_from_extrinsics", side_effect=lambda x: x), \
             patch.object(metrics, "calculate_relative_pose_error", return_value=(4., 2.)) as error, \
             patch.object(metrics, "calculate_total_distance", return_value=20.) as distance, \
             patch.object(metrics, "tqdm", side_effect=lambda x: x):
            translation, rotation, starts, distances = metrics.calculate_subsequence_relative_errors(
                estimates, list(range(80)), 15)
        self.assertEqual(starts, [0, 10])
        self.assertEqual([call.args[:2] for call in distance.call_args_list], [(0, 10), (10, 20)])
        self.assertEqual(translation, [20., 20.])
        self.assertEqual(rotation, [0.1, 0.1])
        self.assertEqual(distances, [20., 20.])
        self.assertEqual(error.call_count, 2)

    def test_stored_bundle_trajectory_uses_checkpoint_endpoints(self):
        a, b, c = object(), object(), object()
        windows = [dict(start_kf=0, end_kf=5, abs_start_pose=a, abs_end_pose=b),
                   dict(start_kf=5, end_kf=10, abs_start_pose=b, abs_end_pose=c)]
        with redirect_stdout(io.StringIO()), patch.object(trajectory, "tqdm", side_effect=lambda x, **kw: x):
            self.assertEqual(trajectory.extract_bundle_poses(windows), {0: a, 5: b, 10: c})

    def test_pnp_extraction_skips_missing_extrinsics(self):
        db = Mock()
        db.all_frames.return_value = [0, 1, 2]
        first, last = object(), object()
        db.get_absolute_extrinsics.side_effect = [first, None, last]
        with patch.object(trajectory, "create_pose_from_extrinsics", side_effect=lambda x: x), \
             patch.object(trajectory, "tqdm", side_effect=lambda x, **kw: x), redirect_stdout(io.StringIO()):
            self.assertEqual(trajectory.extract_pnp_poses(db), {0: first, 2: last})


if __name__ == "__main__":
    unittest.main()
