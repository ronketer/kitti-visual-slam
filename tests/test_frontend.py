"""Synthetic frontend contracts without KITTI, plotting, or GTSAM."""

import random
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kitti_slam import motion
from kitti_slam import stereo
from kitti_slam import supporters
def keypoints(pixels):
    return [cv2.KeyPoint(float(x), float(y), 1.) for x, y in pixels]


class FrontendTests(unittest.TestCase):
    def test_stereo_thresholds_and_opencv_triangulation(self):
        K = np.array([[100., 0., 50.], [0., 100., 20.], [0., 0., 1.]])
        left = np.eye(3, 4)
        right = left.copy()
        right[0, 3] = -0.5
        kp_left = keypoints([(50, 20)] * 4)
        # Disparity 2 accepted, disparity <2 and vertical deviation 2 rejected.
        kp_right = keypoints([(48, 20), (48.1, 20), (45, 22), (51, 20)])
        matches = [cv2.DMatch(i, i, 0.) for i in range(4)]
        matcher = Mock()
        matcher.match.return_value = matches
        retained, cloud = stereo.trangulate_inliers(
            K @ left, K @ right, kp_left, None, kp_right, None, matcher=matcher)
        self.assertEqual([m.queryIdx for m in retained], [0])
        np.testing.assert_allclose(cloud, [[0., 0., 25.]], atol=1e-5)

    def test_handwritten_svd_triangulation(self):
        K = np.array([[100., 0., 50.], [0., 100., 20.], [0., 0., 1.]])
        left = np.eye(3, 4)
        right = left.copy()
        right[0, 3] = -0.5
        point = stereo.solveLLST([52., 24.], [47., 24.], K @ left, K @ right)
        np.testing.assert_allclose(point, [0.2, 0.4, 10.], atol=1e-10)

    def test_four_view_mapping_preserves_temporal_order(self):
        old = [cv2.DMatch(4, 8, 0.), cv2.DMatch(1, 3, 0.)]
        new = [cv2.DMatch(2, 7, 0.), cv2.DMatch(5, 9, 0.)]
        temporal = [cv2.DMatch(1, 5, 0.), cv2.DMatch(0, 2, 0.), cv2.DMatch(4, 2, 0.)]
        common, indices = stereo.find_stereo_temporal_matches(old, new, temporal)
        self.assertEqual(common, [temporal[0], temporal[2]])
        self.assertEqual(indices, [(1, 3, 5, 9), (4, 8, 2, 7)])

    def test_projection_threshold_is_strict(self):
        points = np.array([[0., 0., 1.]] * 3)
        result = supporters.validate_projections_batch(
            points, np.eye(3, 4), np.eye(3), keypoints([(1.9, 0), (2, 0), (0, 0)]))
        np.testing.assert_array_equal(result, [True, False, True])

    def test_seeded_ransac_refinement_with_right_view_outliers(self):
        rng = np.random.default_rng(14)
        cloud = rng.uniform([-2., -1., 5.], [2., 1., 15.], (30, 3))
        K = np.array([[300., 0., 320.], [0., 300., 180.], [0., 0., 1.]])
        left0 = np.eye(3, 4)
        right0 = left0.copy()
        right0[0, 3] = -0.5
        left1 = left0.copy()
        left1[:, 3] = [0.12, -0.02, 0.1]
        right1 = left1.copy()
        right1[0, 3] -= 0.5

        def project(extrinsics):
            projected = (K @ (cloud @ extrinsics[:, :3].T + extrinsics[:, 3]).T).T
            return projected[:, :2] / projected[:, 2:3]

        right1_pixels = project(right1)
        right1_pixels[-6:, 0] += 40
        views = [keypoints(project(left0)), keypoints(project(right0)),
                 keypoints(project(left1)), keypoints(right1_pixels)]
        matches = [cv2.DMatch(i, i, 0.) for i in range(len(cloud))]
        indices = [(i, i, i, i) for i in range(len(cloud))]
        with patch.object(motion.random, "sample", side_effect=random.Random(7).sample):
            estimated_left, estimated_right, inliers = motion.perform_motion_estimation(
                indices, matches, cloud, None, matches, *views, K,
                left0, right0, right0[:, 3])
        self.assertEqual(inliers, [list(row) for row in indices[:24]])
        np.testing.assert_allclose(estimated_left, left1, atol=1e-5)
        np.testing.assert_allclose(estimated_right, right1, atol=1e-5)


    def test_frontend_imports_without_plotting_or_optimizer(self):
        script = f"""
import importlib.abc
import sys
sys.path.insert(0, {str(ROOT)!r})
class BlockOptionalDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'gtsam', 'matplotlib', 'utility'}}:
            raise AssertionError('Unexpected frontend dependency: ' + fullname)
sys.meta_path.insert(0, BlockOptionalDependencies())
from kitti_slam import stereo, motion, supporters
"""
        result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
