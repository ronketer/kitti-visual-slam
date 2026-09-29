"""Small geometry contracts independent of KITTI, plotting, and GTSAM."""

import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "code"))
from geometry import compose_extrinsics, coordinate_transform, find_camera_location
from geometry import compute_camera_to_camera_transform, find_transformation


class GeometryTests(unittest.TestCase):
    def test_camera_to_camera_helper_preserves_source_target_direction(self):
        source = np.array([[0., -1., 0., 2.], [1., 0., 0., 3.], [0., 0., 1., 4.]])
        target = np.array([[1., 0., 0., -1.], [0., 0., -1., 2.], [0., 1., 0., 5.]])
        world_point = np.array([1., 2., 3.])
        in_source = find_transformation(source)(world_point)
        in_target = find_transformation(target)(world_point)
        relative = compute_camera_to_camera_transform(source, target)
        np.testing.assert_array_equal(find_transformation(relative)(in_source), in_target)
        import utility
        self.assertIs(utility.compute_camera_to_camera_transform, compute_camera_to_camera_transform)

    def test_identity_and_empty_point_cloud(self):
        identity = np.eye(3, 4)
        points = np.array([[1., 2., 3.], [-4., 5., 6.]])
        np.testing.assert_array_equal(coordinate_transform(points, identity), points)
        self.assertEqual(coordinate_transform(np.empty((0, 3)), identity).shape, (0, 3))
        np.testing.assert_array_equal(find_camera_location(identity), np.zeros(3))

    def test_composition_applies_first_then_second(self):
        first = np.array([[0., -1., 0., 2.], [1., 0., 0., 3.], [0., 0., 1., 4.]])
        second = np.array([[1., 0., 0., -1.], [0., 0., -1., 2.], [0., 1., 0., 5.]])
        point = np.array([[1., 2., 3.]])
        composed = compose_extrinsics(second, first)
        self.assertEqual(composed.shape, (3, 4))
        np.testing.assert_array_equal(coordinate_transform(point, composed), [[-1., -5., 9.]])
        np.testing.assert_array_equal(
            coordinate_transform(point, composed),
            coordinate_transform(coordinate_transform(point, first), second),
        )
        np.testing.assert_array_equal(compose_extrinsics(np.eye(3, 4), first), first)
        np.testing.assert_array_equal(compose_extrinsics(first, np.eye(3, 4)), first)

    def test_camera_center_maps_to_camera_origin(self):
        extrinsics = np.array([[0., -1., 0., 2.], [1., 0., 0., 3.], [0., 0., 1., 4.]])
        center = find_camera_location(extrinsics)
        np.testing.assert_array_equal(center, [-3., 2., -4.])
        np.testing.assert_array_equal(coordinate_transform(center[None, :], extrinsics), [[0., 0., 0.]])

    def test_float32_behavior_is_preserved(self):
        extrinsics = np.eye(3, 4, dtype=np.float32)
        points = np.ones((2, 3), dtype=np.float32)
        self.assertEqual(compose_extrinsics(extrinsics, extrinsics).dtype, np.dtype('float32'))
        self.assertEqual(find_camera_location(extrinsics).dtype, np.dtype('float32'))
        # The existing homogeneous-coordinate helper promotes through np.ones.
        self.assertEqual(coordinate_transform(points, extrinsics).dtype, np.dtype('float64'))


if __name__ == '__main__':
    unittest.main()
