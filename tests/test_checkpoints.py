"""Checkpoint failures and resume behavior without KITTI or GTSAM."""

import io
import pickle
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code"), str(ROOT)]
import checkpoints as cp
import run_pipeline as pipeline
from tracking_database import TrackingDB, Link


class CheckpointTests(unittest.TestCase):
    def test_lfs_empty_corrupt_and_wrong_stage_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "cache.pkl"
            for payload, message in [
                (b"version https://git-lfs.github.com/spec/v1\noid sha256:abc\n", "Git LFS pointer"),
                (b"", "Cannot load checkpoint"),
                (b"not a pickle", "Cannot load checkpoint"),
                (pickle.dumps([]), "missing required"),
            ]:
                with self.subTest(message=message):
                    path.write_bytes(payload)
                    with self.assertRaisesRegex(cp.CheckpointError, message):
                        cp.checkpoint_exists(path, "tracking")
            self.assertFalse(cp.checkpoint_exists(Path(directory) / "missing.pkl", "tracking"))

    def test_missing_gtsam_is_a_dependency_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gtsam.pkl"
            path.write_bytes(b"cgtsam\nPose3\n.")
            with patch.dict(sys.modules, {"gtsam": None}):
                with self.assertRaisesRegex(cp.CheckpointError, "required module 'gtsam'"):
                    cp.load_checkpoint(path, "ba")

    def test_tracking_roundtrip_preserves_legacy_serializer(self):
        import numpy as np
        db = TrackingDB()
        db.add_frame([Link(5., 4., 3.)], np.zeros((1, 8), dtype=np.uint8))
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(io.StringIO()):
            path = Path(directory) / "tracking.pkl"
            cp.save_tracking_checkpoint(path, db)
            payload = cp.load_checkpoint(path, "tracking")
            self.assertEqual(payload["last_frameId"], 0)
            self.assertIsInstance(payload["prev_frame_links"][0], Link)
            restored = TrackingDB()
            restored.load(str(path.with_suffix("")))
            np.testing.assert_array_equal(restored.features(0), db.features(0))
            self.assertEqual(restored.prev_frame_links[0].x_left, 5.)
            self.assertIn(b"tracking_database", path.read_bytes())
            for field in ("frameId_to_total_matches_count", "frameId_to_common_matches_count", "frameId_to_supporters_count"):
                del payload[field]
            cp.validate_payload(payload, "tracking")

    def test_interrupted_write_preserves_existing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ba.pkl"
            path.write_bytes(b"previous checkpoint")
            payload = [{field: None for field in cp.WINDOW_FIELDS}]
            with patch.object(cp.pickle, "dump", side_effect=RuntimeError("write failed")):
                with self.assertRaisesRegex(RuntimeError, "write failed"):
                    cp.save_checkpoint(path, payload, "ba")
            self.assertEqual(path.read_bytes(), b"previous checkpoint")
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_dependency_hash_rejects_stale_cache_and_missing_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            source, path = Path(directory) / "tracking.pkl", Path(directory) / "ba.pkl"
            source.write_bytes(b"upstream v1")
            payload = [{field: None for field in cp.WINDOW_FIELDS}]
            cp.save_checkpoint(path, payload, "ba", upstream=source)
            self.assertTrue(cp.checkpoint_exists(path, "ba", upstream=source))
            source.write_bytes(b"upstream v2")
            with self.assertRaisesRegex(cp.CheckpointError, "dependency metadata"):
                cp.checkpoint_exists(path, "ba", upstream=source)
            Path(str(path) + ".json").unlink()
            with self.assertRaisesRegex(cp.CheckpointError, "dependency metadata"):
                cp.checkpoint_exists(path, "ba", upstream=source)

    def test_check_mode_never_runs_estimation_or_imports_gtsam(self):
        with tempfile.TemporaryDirectory() as directory:
            path = pipeline.checkpoint_path("tracking", directory)
            path.write_bytes(b"version https://git-lfs.github.com/spec/v1\n")
            with patch.object(pipeline, "run_stage_1", side_effect=AssertionError("estimation")), \
                 patch.object(pipeline.importlib, "import_module", side_effect=AssertionError("import")), \
                 redirect_stderr(io.StringIO()) as errors:
                status = pipeline.main(["--check", "--through", "tracking", "--output-dir", directory])
            self.assertEqual(status, 1)
            self.assertIn("Git LFS pointer", errors.getvalue())
            self.assertEqual(list(Path(directory).iterdir()), [path])

    def test_rebuilding_upstream_forces_downstream(self):
        with patch.object(pipeline.importlib, "import_module"), \
             patch.object(pipeline, "run_stage_1", return_value=True) as first, \
             patch.object(pipeline, "run_stage_2", return_value=True) as second, \
             patch.object(pipeline, "run_stage_3", return_value=True) as third, \
             redirect_stdout(io.StringIO()):
            self.assertEqual(pipeline.main([]), 0)
        self.assertFalse(first.call_args.args[0])
        self.assertTrue(second.call_args.args[0])
        self.assertTrue(third.call_args.args[0])

    def test_missing_gtsam_stops_before_tracking_but_tracking_only_works(self):
        with patch.object(pipeline.importlib, "import_module", side_effect=ModuleNotFoundError("gtsam")), \
             patch.object(pipeline, "run_stage_1", return_value=False) as tracking, \
             redirect_stderr(io.StringIO()), redirect_stdout(io.StringIO()):
            self.assertEqual(pipeline.main([]), 1)
            tracking.assert_not_called()
            self.assertEqual(pipeline.main(["--through", "tracking"]), 0)
            tracking.assert_called_once()


if __name__ == "__main__":
    unittest.main()
