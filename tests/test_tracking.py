"""Multi-frame orchestration and legacy checkpoint contracts."""

import io
import pickle
import sys
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))
import tracking
from tracking_database import Link, TrackingDB


def fixture():
    """Sparse feature indices, a competing match, a lost track, and a new track."""
    K = np.diag([100., 100., 1.])
    M1 = np.eye(3, 4)
    M2 = M1.copy()
    M2[0, 3] = -0.5
    stereo_frames = []
    for fid, selected in enumerate(([0, 2, 4], [1, 2, 4], [0, 3, 4])):
        left = [cv2.KeyPoint(30. + i + fid, 20., 1.) for i in range(5)]
        right = [cv2.KeyPoint(25. + i + fid, 20., 1.) for i in range(5)]
        descriptors = np.arange(20, dtype=np.uint8).reshape(5, 4) + fid
        matches = [cv2.DMatch(i, i, 0.) for i in selected]
        stereo_frames.append((left, descriptors, right, descriptors.copy(), matches,
                              np.array([[float(i), 0., 10.] for i in selected])))
    temporal = [
        [cv2.DMatch(a, b, 0.) for a, b in [(0, 1), (2, 2), (4, 2), (1, 3)]],
        [cv2.DMatch(a, b, 0.) for a, b in [(1, 0), (2, 3), (4, 4)]],
    ]
    relative = [
        np.array([[0., -1., 0., 1.], [1., 0., 0., 2.], [0., 0., 1., 3.]]),
        np.array([[1., 0., 0., -2.], [0., 0., -1., 1.], [0., 1., 0., 4.]]),
    ]
    motion_results = [
        (relative[0], None, [[0, 0, 1, 1], [2, 2, 2, 2], [4, 4, 2, 2]]),
        (relative[1], None, [[1, 1, 0, 0], [4, 4, 4, 4]]),
    ]
    return (K, M1, M2), stereo_frames, temporal, motion_results


def run_fixture(builder=tracking.build_tracking_database, **kwargs):
    calibration, frames, temporal, motion_results = fixture()
    matcher = Mock()
    matcher.match.side_effect = temporal
    with ExitStack() as stack:
        stack.enter_context(patch.object(tracking, "read_images", side_effect=[(i, i) for i in range(3)]))
        stack.enter_context(patch.object(tracking, "process_stereo_pair", side_effect=frames))
        estimator = stack.enter_context(patch.object(tracking, "perform_motion_estimation", side_effect=motion_results))
        stack.enter_context(patch.object(tracking, "tqdm", side_effect=lambda values: values))
        db = builder(calibration=calibration, detector=object(), matcher=matcher,
                     last_frame=2, **kwargs)
    return db, estimator


def snapshot(value):
    if isinstance(value, np.ndarray):
        return (str(value.dtype), value.tolist())
    if isinstance(value, Link):
        return vars(value)
    if isinstance(value, dict):
        return {key: snapshot(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [snapshot(item) for item in value]
    return value


class TrackingTests(unittest.TestCase):
    def test_multiframe_tracks_remapping_and_pose_composition(self):
        db, estimator = run_fixture()
        self.assertEqual(db.trackId_to_frames, {0: [0, 1, 2], 1: [0, 1], 2: [1, 2]})
        self.assertEqual(db.frameId_to_trackIds_list, {0: [0, 1, -1], 1: [0, 1, 2], 2: [0, -1, 2]})
        self.assertEqual([link.x_left for link in db.leftover_links[0]], [34.])
        self.assertEqual(db.leftover_links[1], [])
        self.assertEqual([link.x_left for link in db.all_frame_links(2)], [32., 35., 36.])
        np.testing.assert_array_equal(db.features(0), np.arange(20, dtype=np.uint8).reshape(5, 4)[[0, 2, 4]])
        np.testing.assert_array_equal(db.get_absolute_extrinsics(2),
                                      [[0., -1., 0., -1.], [0., 0., -1., -2.], [1., 0., 0., 6.]])
        self.assertEqual(estimator.call_count, 2)
        self.assertEqual(estimator.call_args_list[0].args[0], [(0, 0, 1, 1), (2, 2, 2, 2), (4, 4, 2, 2)])
        with redirect_stdout(io.StringIO()):
            db._check_consistency()

    def test_diagnostics_do_not_change_tracking(self):
        ordinary, _ = run_fixture()
        diagnostic, _ = run_fixture(collect_diagnostics=True)
        expected_counts = {
            "frameId_to_total_matches_count": {0: 0, 1: 4, 2: 3},
            "frameId_to_common_matches_count": {0: 0, 1: 3, 2: 3},
            "frameId_to_supporters_count": {0: 0, 1: 3, 2: 2},
        }
        for name, counts in expected_counts.items():
            self.assertEqual(getattr(diagnostic, name), counts)
            self.assertEqual(getattr(ordinary, name), {0: 0, 1: 0, 2: 0})
        self.assertEqual(
            snapshot({k: v for k, v in vars(ordinary).items() if k not in expected_counts}),
            snapshot({k: v for k, v in vars(diagnostic).items() if k not in expected_counts}),
        )

    def test_legacy_entry_points(self):
        from alg import create_tracking_db
        from final.ransac_pnp_analysis import updated_create_tracking_db
        self.assertIs(create_tracking_db, tracking.build_tracking_database)
        report, _ = run_fixture(updated_create_tracking_db)
        canonical, _ = run_fixture(collect_diagnostics=True)
        self.assertEqual(snapshot(vars(report)), snapshot(vars(canonical)))

    def test_checkpoint_roundtrip_and_older_optional_statistics(self):
        db, _ = run_fixture(collect_diagnostics=True)

        class Buffer(io.BytesIO):
            def close(self):
                pass  # Keep the in-memory checkpoint available after the file context.

        buffer = Buffer()
        with patch("builtins.open", return_value=buffer), redirect_stdout(io.StringIO()):
            db.serialize("synthetic")
        payload = buffer.getvalue()
        self.assertEqual(Link.__module__, "tracking_database")
        self.assertIn(b"tracking_database", payload)
        restored = TrackingDB()
        with patch("builtins.open", return_value=Buffer(payload)), redirect_stdout(io.StringIO()):
            restored.load("synthetic")
        self.assertEqual(snapshot(vars(db)), snapshot(vars(restored)))

        older = pickle.loads(payload)
        for name in ("frameId_to_total_matches_count", "frameId_to_common_matches_count",
                     "frameId_to_supporters_count"):
            del older[name]
        with patch("builtins.open", return_value=Buffer(pickle.dumps(older))), redirect_stdout(io.StringIO()):
            restored.load("older_synthetic")
        self.assertEqual(restored.frameId_to_total_matches_count, {})
        self.assertEqual(restored.frameId_to_common_matches_count, {})
        self.assertEqual(restored.frameId_to_supporters_count, {})
        for name in older:
            self.assertEqual(snapshot(getattr(restored, name)), snapshot(getattr(db, name)))


if __name__ == "__main__":
    unittest.main()
