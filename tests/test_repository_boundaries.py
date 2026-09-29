"""Repository contracts: import direction, evaluation policies and window boundaries."""

import ast
import importlib
import io
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, mock_open, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kitti_slam.evaluation import projection_errors as projection
from kitti_slam.window_selector import WindowSelector, WindowParameters, TrackOverlapCriterion


def observation(left, right, vertical):
    return SimpleNamespace(uL=lambda: left, uR=lambda: right, v=lambda: vertical)


class RepositoryBoundaryTests(unittest.TestCase):
    def test_ground_truth_reader_keeps_inclusive_bound_and_center_convention(self):
        from kitti_slam.config import ProjectPaths
        from kitti_slam.evaluation.ground_truth import read_ground_truth_poses
        paths = ProjectPaths(dataset_root=Path("custom-data"), sequence="07")
        lines = "1 0 0 2 0 1 0 3 0 0 1 4\n1 0 0 9 0 1 0 0 0 0 1 0\n"
        with patch("builtins.open", mock_open(read_data=lines)) as reader:
            result = read_ground_truth_poses(0, paths=paths)
        reader.assert_called_once_with(paths.poses_file, "r")
        np.testing.assert_array_equal(result, [[-2., -3., -4.]])


    def test_projection_metric_is_mean_of_two_image_distances(self):
        # Left error 5, right error 4: not a 3D residual norm or RMS.
        actual = projection.calculate_projection_error(observation(3., 0., 4.), observation(0., 0., 0.))
        self.assertEqual(actual, 4.5)
        for name in ("plot_projection_vs_distance", "plot_median_projection_errors"):
            canonical = importlib.import_module("reports." + name)
            self.assertIs(canonical.calculate_projection_error, projection.calculate_projection_error)

    def test_projection_aggregation_keeps_frame_distance_and_missing_values_policy(self):
        # These doubles check aggregation only, not GTSAM projection mathematics.
        class GenericStereoFactor3D:
            def keys(self): return [12, 99]
            def measured(self): return observation(3., 0., 4.)
        factor = GenericStereoFactor3D()
        graph = Mock(size=lambda: 2, at=lambda index: object() if index == 0 else factor)
        initial = Mock(exists=lambda key: True)
        result = Mock(exists=lambda key: key != 99)
        camera = Mock(project=lambda point: observation(0., 0., 0.))
        api = SimpleNamespace(StereoCamera=Mock(return_value=camera))
        window = dict(start_kf=3, graph=graph, initialEstimate=initial, result=result,
                      pose_keys={7: 12}, point_keys={5: 99})
        with patch.dict(sys.modules, {"gtsam": api}), \
             patch.object(projection, "tqdm", side_effect=lambda values, **kwargs: values), \
             redirect_stdout(io.StringIO()):
            before, after = projection.analyze_bundle_projection_errors([window], object())
            flat_before, flat_after = projection.extract_projection_errors_from_window(
                graph, initial, result, window["pose_keys"], window["point_keys"], object())
        self.assertEqual(dict(before), {4: [4.5]})
        self.assertEqual(dict(after), {})
        self.assertEqual(flat_before, [4.5])
        self.assertEqual(flat_after, [])

    def test_window_endpoint_spacing_tail_and_fallback_are_preserved(self):
        db = Mock(frame_num=lambda: 12)
        selector = WindowSelector(db, WindowParameters(min_bundle_size=2, max_bundle_size=5))
        self.assertEqual([selector.next_window() for _ in range(4)], [(0, 5), (5, 10), (10, 11), None])
        fail = Mock(evaluate=lambda *args: False)
        selector = WindowSelector(db, WindowParameters(min_bundle_size=2, max_bundle_size=5, criteria=[fail]))
        self.assertEqual(selector.next_window(), (0, 2))
        # Historical overlap compares start with candidate minus one.
        tracks = {0: [1, 2], 4: [1, 2], 5: [9]}
        self.assertTrue(TrackOverlapCriterion(1.).evaluate(0, 5, Mock(tracks=lambda fid: tracks[fid])))

    def test_historical_modules_import_outside_repo_without_code_path(self):
        script = f'''
import sys, importlib, importlib.abc
sys.path.insert(0, {str(ROOT)!r})
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'gtsam', 'alg', 'utility', 'consts', 'ex3code'}}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
for name in {['coursework.ex'+str(i) for i in range(1,5)] + ['reports.'+p.stem for p in (ROOT/'reports').glob('*.py') if p.stem not in {'__init__','pose_graph_analysis'}]!r}:
    importlib.import_module(name)
from reports.paths import GT_POSES_FILE
assert GT_POSES_FILE == {str(ROOT/'dataset/poses/05.txt')!r}
assert {str(ROOT/'code')!r} not in sys.path
'''
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-I", "-B", "-c", script], cwd=directory,
                                    capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_no_package_dependency_on_historical_directories(self):
        forbidden = {"coursework", "reports", "code", "utility", "alg", "consts", "ex3code"}
        for path in (ROOT / "kitti_slam").rglob("*.py"):
            for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom) and node.level == 0:
                    self.assertNotIn(node.module.split('.')[0], forbidden, str(path))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertNotIn(alias.name.split('.')[0], forbidden, str(path))


if __name__ == "__main__":
    unittest.main()
