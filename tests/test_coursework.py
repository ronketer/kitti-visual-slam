"""Exercise 3 stereo, timed supporter and frame-state contracts."""
import unittest
from dataclasses import is_dataclass
from unittest.mock import Mock, patch

import cv2
import numpy as np

from coursework import ex3
from kitti_slam import motion, supporters


class ExerciseThreeTests(unittest.TestCase):
    def test_stereo_keeps_small_disparity_but_rejects_vertical_boundary(self):
        left = np.eye(3, 4)
        right = left.copy()
        right[0, 3] = -1.
        keypoints = [cv2.KeyPoint(0., 0., 1.) for _ in range(2)]
        right_keypoints = [cv2.KeyPoint(-1., 0., 1.), cv2.KeyPoint(-1., 2., 1.)]
        matches = [cv2.DMatch(i, i, 0.) for i in range(2)]
        with patch.object(ex3.detector_config, "matcher", Mock(match=lambda *args: matches)):
            retained, cloud = ex3.prepare_stereo_matches(
                left, right, keypoints, None, right_keypoints, None)
        self.assertEqual([match.queryIdx for match in retained], [0])
        np.testing.assert_allclose(cloud, [[0., 0., 1.]], atol=1e-6)

    def test_timed_supporters_preserve_four_view_validation(self):
        points = np.array([[0., 0., 1.], [1., 0., 1.]])
        pixels = [cv2.KeyPoint(0., 0., 1.), cv2.KeyPoint(1., 0., 1.)]
        outliers = [pixels[0], cv2.KeyPoint(3., 0., 1.)]
        matches = [cv2.DMatch(i, i, 0.) for i in range(2)]
        indices = [(i, i, i, i) for i in range(2)]
        pose = np.eye(3, 4)
        args = (indices, points, pixels, pixels, pixels, outliers,
                np.eye(3), pose, pose, pose, pose, matches)
        logger = Mock()
        actual = ex3.classify_timed_supporters(*args, logger=logger)
        self.assertEqual(actual, [[0, 0, 0, 0]])
        self.assertEqual(actual, supporters.find_ransac_iteration_supporters(*args))
        self.assertEqual(logger.start_operation.call_args_list,
                         logger.end_operation.call_args_list)
        self.assertEqual(logger.start_operation.call_count, 2)

    def test_shared_solvers_and_frame_state_contract(self):
        self.assertIs(ex3.solve_pnp_and_locations, motion.solve_pnp_and_locations)
        self.assertIs(ex3.refine_pose, motion.refine_pose)
        self.assertTrue(is_dataclass(ex3.FrameState))
        state = ex3.FrameState(ground_truth_location="known")
        self.assertEqual(state.ground_truth_location, "known")
        self.assertIsNone(state.absolute_extrinsics)


if __name__ == "__main__":
    unittest.main()
