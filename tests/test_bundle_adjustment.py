"""BA orchestration contracts; solver tests explicitly require real GTSAM."""

import importlib.util
import io
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, mock_open, patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kitti_slam import bundle_adjustment as ba
from kitti_slam import gtsam_geometry
from kitti_slam import trajectory
from kitti_slam.tracking_database import TrackingDB, Link
from kitti_slam.window_selector import WindowParameters

HAS_GTSAM = importlib.util.find_spec("gtsam") is not None


class PoseLabel:
    """Call-order marker, not a mathematical pose or optimizer substitute."""
    def __init__(self, label="origin"):
        self.label = label

    def compose(self, other):
        return PoseLabel(self.label + "/" + other.label)


class BundleOrchestrationTests(unittest.TestCase):
    def test_window_results_and_trajectory_contract(self):
        graphs = [Mock(), Mock()]
        initials = [Mock(), Mock()]
        results = [Mock(), Mock()]
        windows = [(0, 5), (5, 10)]
        builds = []
        for i, (start, end) in enumerate(windows):
            graphs[i].error.side_effect = [10., 4.]
            initials[i].atPose3.return_value = PoseLabel(f"initial{i}")
            results[i].atPose3.return_value = PoseLabel(f"optimized{i}")
            builds.append((graphs[i], initials[i], {start: start, end: end},
                           {8: 80}, {8}, list(range(start, end + 1)), Mock(), []))
        optimizer = Mock(side_effect=[Mock(optimize=Mock(return_value=r)) for r in results])
        # Doubles verify data flow only. Real solver tests below are skipped if unavailable.
        api = SimpleNamespace(Pose3=PoseLabel, LevenbergMarquardtOptimizer=optimizer)
        reference = np.eye(3, 4)
        db, calibration, params = object(), object(), object()
        with patch.dict(sys.modules, {"gtsam": api}), \
             patch.object(ba, "WindowSelector") as selector, \
             patch.object(ba, "initialize_factor_graph_in_window", side_effect=builds) as builder, \
             patch.object(ba, "read_cameras", side_effect=AssertionError("unexpected calibration I/O")), \
             patch("builtins.open", side_effect=AssertionError("unexpected file I/O")), \
             patch.object(ba, "tqdm", side_effect=lambda x: x):
            selector.return_value.next_window.side_effect = [*windows, None]
            output = ba.run_bundle_adjustment(db, calibration, reference_extrinsics=reference,
                                             window_params=params)
            initial, optimized, relative_initial, relative_optimized = trajectory.bundle_trajectories(output)
        selector.assert_called_once_with(db, params)
        self.assertEqual(builder.call_count, 2)
        for call in builder.call_args_list:
            self.assertIs(call.kwargs["reference_extrinsics"], reference)
        self.assertEqual(optimizer.call_args_list[0].args, (graphs[0], initials[0]))
        self.assertEqual(optimizer.call_args_list[1].args, (graphs[1], initials[1]))
        self.assertEqual(set(output[0]), {"start_kf", "end_kf", "graph", "initialEstimate",
                         "pose_keys", "point_keys", "tracks", "frames", "result",
                         "abs_start_pose", "abs_end_pose"})
        self.assertIs(output[0]["result"], results[0])
        self.assertIs(output[1]["abs_start_pose"], output[0]["abs_end_pose"])
        self.assertEqual(output[1]["abs_end_pose"].label, "origin/optimized0/optimized1")
        self.assertEqual(initial[10].label, "origin/initial0/initial1")
        self.assertIs(optimized[10], output[1]["abs_end_pose"])
        self.assertEqual(relative_initial[(5, 10)].label, "initial1")
        self.assertEqual(relative_optimized[(5, 10)].label, "optimized1")

    def test_original_error_reduction_gate_is_preserved(self):
        graph = Mock()
        graph.error.side_effect = [10., 6.]
        api = SimpleNamespace(Pose3=PoseLabel, LevenbergMarquardtOptimizer=Mock())
        with patch.dict(sys.modules, {"gtsam": api}), \
             patch.object(ba, "WindowSelector") as selector, \
             patch.object(ba, "initialize_factor_graph_in_window", return_value=(
                 graph, Mock(), {0: 0, 1: 1}, {}, set(), [0, 1], Mock(), [])), \
             patch.object(ba, "tqdm", side_effect=lambda x: x):
            selector.return_value.next_window.side_effect = [(0, 1), None]
            with self.assertRaisesRegex(ValueError, "not significantly lower"):
                ba.run_bundle_adjustment(object(), object(), reference_extrinsics=np.eye(3, 4))

    def test_stage_two_writes_results_without_exercise_import(self):
        from kitti_slam import pipeline
        calibration = (object(), object(), object())
        results = [object()]
        with patch.dict(sys.modules, {"ex5": None}), \
             patch("kitti_slam.tracking_database.TrackingDB") as database, \
             patch("kitti_slam.dataset.read_cameras", return_value=calibration), \
             patch.object(gtsam_geometry, "read_calibration", return_value="stereo calibration"), \
             patch.object(ba, "run_bundle_adjustment", return_value=results) as run, \
             patch.object(pipeline, "load_checkpoint") as load, \
             patch.object(pipeline, "save_checkpoint") as save, \
             redirect_stdout(io.StringIO()):
            pipeline.run_stage_2(force=True)
        source = pipeline.checkpoint_path("tracking")
        load.assert_called_once_with(source, "tracking")
        database.return_value.load.assert_called_once_with(str(source.with_suffix("")))
        run.assert_called_once_with(database.return_value, "stereo calibration", reference_extrinsics=calibration[1])
        save.assert_called_once_with(pipeline.checkpoint_path("ba"), results, "ba", upstream=source)



@unittest.skipUnless(HAS_GTSAM, "Real GTSAM unavailable: BA numerical validation deferred")
class BundleNumericalTests(unittest.TestCase):
    def setUp(self):
        import gtsam
        self.K = gtsam.Cal3_S2Stereo(300., 300., 0., 320., 180., 0.5)
        self.db = TrackingDB()
        points = np.random.default_rng(4).uniform([-2., -1., 5.], [2., 1., 12.], (12, 3))
        for fid in range(3):
            local = points - [0.1 * fid, 0., 0.]
            links = [Link(300*x/z+320, 300*(x-0.5)/z+320, 300*y/z+180) for x, y, z in local]
            features = np.zeros((len(points), 8), dtype=np.uint8)
            matches = None if fid == 0 else [cv2.DMatch(i, i, 0.) for i in range(len(points))]
            self.db.add_frame(links, features, matches)
            absolute = np.eye(3, 4)
            absolute[0, 3] = -0.12 * fid
            self.db.add_absolute_extrinsics(fid, absolute)
            if fid:
                relative = np.eye(3, 4)
                relative[0, 3] = -0.12
                self.db.add_relative_extrinsics(fid, relative)

    def test_stereo_graph_optimization(self):
        import gtsam
        graph, initial, poses, points, tracks, frames, prior, factors = ba.initialize_factor_graph_in_window(
            self.db, 0, 1, self.K, reference_extrinsics=np.eye(3, 4))
        self.assertEqual(frames, [0, 1])
        self.assertEqual(len(tracks), 12)
        self.assertEqual(graph.size(), 25)
        self.assertEqual(len(factors), 24)
        result = gtsam.LevenbergMarquardtOptimizer(graph, initial).optimize()
        self.assertLess(graph.error(result), graph.error(initial))
        np.testing.assert_allclose(result.atPose3(poses[1]).translation(), [0.1, 0., 0.], atol=1e-4)

    def test_two_window_optimization(self):
        windows = ba.run_bundle_adjustment(self.db, self.K, reference_extrinsics=np.eye(3, 4),
                                          window_params=WindowParameters(min_bundle_size=1, max_bundle_size=1))
        self.assertEqual([(w["start_kf"], w["end_kf"]) for w in windows], [(0, 1), (1, 2)])
        np.testing.assert_allclose(windows[-1]["abs_end_pose"].translation(), [0.2, 0., 0.], atol=1e-4)


if __name__ == "__main__":
    unittest.main()
