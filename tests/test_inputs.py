"""Input boundaries exercised without KITTI or an installed GTSAM."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kitti_slam import tracking
from kitti_slam.config import DEFAULT_PATHS, ProjectPaths
from kitti_slam.dataset import read_cameras, read_images
from kitti_slam.detector_config import create_detector_and_matcher


class InputTests(unittest.TestCase):
    def test_readers_use_supplied_dataset_and_preserve_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = ProjectPaths(dataset_root=Path(directory), sequence="07")
            for side, value in (("image_0", 40), ("image_1", 90)):
                folder = paths.sequence_dir / side
                folder.mkdir(parents=True)
                cv2.imwrite(str(folder / "000003.png"), np.full((4, 5), value, np.uint8))
            paths.calibration_file.write_text(
                "P0: 100 0 50 0 0 100 20 0 0 0 1 0\n"
                "P1: 100 0 50 -50 0 100 20 0 0 0 1 0\n"
            )
            left, right = read_images(3, paths)
            np.testing.assert_array_equal(left, np.full((4, 5), 40))
            np.testing.assert_array_equal(right, np.full((4, 5), 90))
            K, M1, M2 = read_cameras(paths)
            np.testing.assert_array_equal(K, [[100, 0, 50], [0, 100, 20], [0, 0, 1]])
            np.testing.assert_array_equal(M1, np.eye(3, 4))
            np.testing.assert_array_equal(M2[:, 3], [-0.5, 0, 0])

    def test_tracking_accepts_calibration_components_and_short_range(self):
        K = np.array([[100., 0., 50.], [0., 100., 20.], [0., 0., 1.]])
        M1 = np.eye(3, 4)
        M2 = M1.copy()
        M2[0, 3] = -0.5
        detector, matcher = object(), object()
        kp_left = [cv2.KeyPoint(50., 20., 1.)]
        kp_right = [cv2.KeyPoint(45., 20., 1.)]
        descriptors = np.zeros((1, 8), np.uint8)
        stereo_result = (kp_left, descriptors, kp_right, descriptors,
                         [cv2.DMatch(0, 0, 0.)], np.array([[0., 0., 10.]]))
        with patch.object(tracking, "read_cameras", side_effect=AssertionError("unexpected disk calibration")), \
             patch.object(tracking, "create_detector_and_matcher", side_effect=AssertionError("unexpected defaults")), \
             patch.object(tracking, "read_images", return_value=("left", "right")) as images, \
             patch.object(tracking, "process_stereo_pair", return_value=stereo_result) as stereo:
            db = tracking.build_tracking_database(calibration=(K, M1, M2), detector=detector,
                                        matcher=matcher, last_frame=0)
        self.assertEqual(db.frame_num(), 1)
        np.testing.assert_array_equal(db.get_absolute_extrinsics(0), M1)
        images.assert_called_once_with(0, DEFAULT_PATHS)
        self.assertIs(stereo.call_args.kwargs["detector"], detector)
        self.assertIs(stereo.call_args.kwargs["matcher"], matcher)

    def test_detector_defaults(self):
        detector, matcher = create_detector_and_matcher()
        self.assertIsInstance(detector, cv2.AKAZE)
        descriptors = np.array([[0], [255]], dtype=np.uint8)
        matches = matcher.match(descriptors, np.array([[1]], dtype=np.uint8))
        self.assertEqual([m.distance for m in matches], [1., 7.])

    def test_frame_range_rejects_unsupported_numbering(self):
        with self.assertRaises(ValueError):
            tracking.build_tracking_database(first_frame=10, last_frame=20)
        with self.assertRaises(ValueError):
            tracking.build_tracking_database(last_frame=-1)

    def test_imports_without_gtsam_from_another_working_directory(self):
        script = f"""
import importlib.abc
import sys
from pathlib import Path
sys.path.insert(0, {str(ROOT)!r})
class BlockGtsam(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'gtsam' or fullname.startswith('gtsam.'):
            raise ModuleNotFoundError('GTSAM deliberately unavailable', name='gtsam')
sys.meta_path.insert(0, BlockGtsam())
from kitti_slam import detector_config
assert 'detector' not in vars(detector_config)
from kitti_slam import geometry, dataset, motion, window_selector, gtsam_geometry
assert 'detector' not in vars(detector_config)
from kitti_slam.config import DEFAULT_PATHS
assert DEFAULT_PATHS.calibration_file == Path({str(DEFAULT_PATHS.calibration_file)!r})
try:
    gtsam_geometry.create_pose_from_extrinsics(None)
except ModuleNotFoundError as error:
    assert error.name == 'gtsam'
else:
    raise AssertionError('GTSAM operation must still require GTSAM')
"""
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-B", "-c", script], cwd=directory,
                                    capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
