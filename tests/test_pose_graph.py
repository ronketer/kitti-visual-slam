"""Pose graph boundaries; doubles test orchestration, real solver tests are gated."""

import importlib.util
import io
import subprocess
import sys
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, mock_open, patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "code"), str(ROOT)]
import pose_graph as pg
import trajectory

HAS_GTSAM = importlib.util.find_spec("gtsam") is not None


class PoseLabel:
    """Symbolic call marker; deliberately provides no numerical pose operations."""
    def __init__(self, label="origin"):
        self.label = label

    def compose(self, other):
        return PoseLabel(self.label + "/" + other.label)

    def between(self, other):
        return PoseLabel(self.label + "->" + other.label)


class ValuesDouble(dict):
    def insert(self, key, value):
        self[key] = value

    def update(self, key, value):
        self[key] = value

    def atPose3(self, key):
        return self[key]

    def exists(self, key):
        return key in self


class PoseGraphBoundaryTests(unittest.TestCase):
    def test_explicit_endpoint_and_factor_order(self):
        graph = Mock()
        api = SimpleNamespace(
            NonlinearFactorGraph=Mock(return_value=graph), Values=ValuesDouble,
            Pose3=PoseLabel, symbol=lambda prefix, fid: (prefix, fid),
            PriorFactorPose3=Mock(side_effect=lambda *args: ("prior", args)),
            BetweenFactorPose3=Mock(side_effect=lambda *args: ("between", args)),
            noiseModel=SimpleNamespace(
                Diagonal=SimpleNamespace(Sigmas=lambda array: array),
                Gaussian=SimpleNamespace(Covariance=lambda array: array)),
        )
        covariance = np.eye(6)
        constraints = [dict(start_kf=a, end_kf=b, relative_pose=PoseLabel(str(b)), conditional_cov=covariance)
                       for a, b in [(0, 5), (5, 10)]]
        endpoint = PoseLabel("provided endpoint")
        with patch.dict(sys.modules, {"gtsam": api}), patch("builtins.open", side_effect=AssertionError("I/O")), \
             patch.object(pg, "tqdm", side_effect=lambda x: x), redirect_stdout(io.StringIO()):
            result_graph, initial, keys = pg.build_pose_graph(constraints, endpoint_frame=10, endpoint_pose=endpoint)
        self.assertIs(result_graph, graph)
        self.assertEqual([call.args[0][0] for call in graph.add.call_args_list], ["prior", "between", "between", "prior"])
        self.assertEqual(initial.atPose3(keys[5]).label, "origin/5")
        self.assertIs(initial.atPose3(keys[10]), endpoint)
        self.assertIs(api.BetweenFactorPose3.call_args_list[0].args[3], covariance)
        self.assertIs(api.PriorFactorPose3.call_args.args[1], endpoint)
        np.testing.assert_array_equal(api.PriorFactorPose3.call_args.args[2], np.full(6, 1e-6))

    def test_covariance_formula_keeps_regularization_and_block_order(self):
        covariance = np.block([[np.eye(6), 0.3*np.eye(6)], [0.3*np.eye(6), 2*np.eye(6)]])
        marginals = Mock()
        marginals.jointMarginalCovariance.return_value.fullMatrix.return_value = covariance
        api = SimpleNamespace(Marginals=Mock(return_value=marginals), KeyVector=list)
        values = ValuesDouble({11: PoseLabel("start"), 22: PoseLabel("end")})
        window = dict(start_kf=3, end_kf=9, pose_keys={3: 11, 9: 22}, graph=object(), result=values)
        with patch.dict(sys.modules, {"gtsam": api}), redirect_stdout(io.StringIO()), \
             patch.object(pg, "tqdm", side_effect=lambda x: x):
            result = pg.extract_relative_pose_covariance([window])[0]
        marginals.jointMarginalCovariance.assert_called_once_with([11, 22])
        eps = 1e-4
        expected = 1 / (1 / (2 + eps - 0.09 / (1 + eps)) + eps)
        np.testing.assert_allclose(result["conditional_cov"], expected*np.eye(6), atol=1e-12)
        self.assertEqual(result["relative_pose"].label, "start->end")
        self.assertEqual((result["start_kf"], result["end_kf"]), (3, 9))

    def test_optimizer_returns_legacy_schema_without_io(self):
        initial = ValuesDouble({1: PoseLabel("initial"), 2: PoseLabel("endpoint overwrite")})
        optimized = ValuesDouble({1: PoseLabel("optimized"), 2: PoseLabel("endpoint")})
        api = SimpleNamespace(LevenbergMarquardtOptimizer=Mock(return_value=Mock(optimize=Mock(return_value=optimized))))
        graph = Mock(error=Mock(side_effect=[1., 5.]))
        keys, constraints = {0: 1, 10: 2, 20: 3}, [object()]
        with patch.dict(sys.modules, {"gtsam": api}), patch("builtins.open", side_effect=AssertionError("I/O")), \
             redirect_stdout(io.StringIO()):
            result = pg.optimize_pose_graph(graph, initial, keys, constraints)
        api.LevenbergMarquardtOptimizer.assert_called_once_with(graph, initial)
        self.assertEqual(set(result), {"poses_without_loop_closure", "poses_with_loop_closure", "kf_pose_keys",
                         "pose_graph", "initial_estimate", "optimized_result", "relative_poses_and_covariances"})
        self.assertIs(result["poses_without_loop_closure"][10], initial[2])
        self.assertNotIn(20, result["poses_with_loop_closure"])
        self.assertIs(result["optimized_result"], optimized)
        self.assertIs(result["relative_poses_and_covariances"], constraints)

    def test_stage_three_supplies_ground_truth_endpoint_explicitly(self):
        import run_pipeline
        lines = ["1 0 0 0 0 1 0 0 0 0 1 0\n", "1 0 0 1 0 1 0 0 0 0 1 0\n",
                 "1 0 0 2 0 1 0 0 0 0 1 0\n"]
        windows, results, endpoint = [object()], {"result": object()}, object()
        files = mock_open(read_data="".join(lines))
        with patch.dict(sys.modules, {"pose_graph_loop_closure": None, "ex5": None}), \
             patch.object(run_pipeline, "LAST_FRAME", 2), patch("builtins.open", files), \
             patch.object(run_pipeline, "load_checkpoint", return_value=windows), \
             patch.object(run_pipeline, "save_checkpoint") as save, \
             patch("gtsam_geometry.create_pose_from_extrinsics", return_value=endpoint) as convert, \
             patch.object(pg, "run_pose_graph", return_value=results) as run, redirect_stdout(io.StringIO()):
            run_pipeline.run_stage_3(force=True)
        run.assert_called_once_with(windows, endpoint_frame=2, endpoint_pose=endpoint)
        np.testing.assert_array_equal(convert.call_args.args[0][:, 3], [2., 0., 0.])
        save.assert_called_once_with(run_pipeline.checkpoint_path("pose-graph"), results, "pose-graph",
                                     upstream=run_pipeline.checkpoint_path("ba"))

    def test_computational_modules_import_without_gtsam_or_plots(self):
        script = f"""
import sys, importlib.abc
sys.path.insert(0, {str(ROOT / 'code')!r})
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'gtsam','matplotlib','ex5','utility'}}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
import pose_graph, trajectory, bundle_adjustment, evaluation.trajectory_errors
"""
        result = subprocess.run([sys.executable, "-B", "-c", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


@unittest.skipUnless(HAS_GTSAM, "Real GTSAM unavailable: pose graph numerical validation deferred")
class PoseGraphNumericalTests(unittest.TestCase):
    def test_endpoint_constrained_chain_optimization(self):
        import gtsam
        delta = gtsam.Pose3(gtsam.Rot3(), np.array([1., 0., 0.]))
        endpoint = gtsam.Pose3(gtsam.Rot3(), np.array([1.8, 0., 0.]))
        constraints = [dict(start_kf=a, end_kf=b, relative_pose=delta, conditional_cov=np.eye(6))
                       for a, b in [(0, 1), (1, 2)]]
        graph, initial, keys = pg.build_pose_graph(constraints, endpoint_frame=2, endpoint_pose=endpoint)
        self.assertEqual(graph.size(), 4)
        result = pg.optimize_pose_graph(graph, initial, keys, constraints)
        self.assertLess(graph.error(result["optimized_result"]), graph.error(initial))
        np.testing.assert_allclose(result["poses_with_loop_closure"][1].translation(), [0.9, 0., 0.], atol=1e-5)
        np.testing.assert_allclose(result["poses_with_loop_closure"][2].translation(), [1.8, 0., 0.], atol=1e-5)

    def test_real_joint_marginal_extraction(self):
        import gtsam
        graph = gtsam.NonlinearFactorGraph()
        keys = {0: gtsam.symbol('c', 0), 1: gtsam.symbol('c', 1)}
        identity = gtsam.Pose3()
        delta = gtsam.Pose3(gtsam.Rot3(), np.array([1., 0., 0.]))
        noise = gtsam.noiseModel.Unit.Create(6)
        graph.add(gtsam.PriorFactorPose3(keys[0], identity, noise))
        graph.add(gtsam.BetweenFactorPose3(keys[0], keys[1], delta, noise))
        values = gtsam.Values()
        values.insert(keys[0], identity)
        values.insert(keys[1], delta)
        constraint = pg.extract_relative_pose_covariance([
            dict(start_kf=0, end_kf=1, graph=graph, result=values, pose_keys=keys)])[0]
        np.testing.assert_allclose(constraint["relative_pose"].translation(), [1., 0., 0.])
        cov = constraint["conditional_cov"]
        self.assertEqual(cov.shape, (6, 6))
        self.assertTrue(np.isfinite(cov).all())
        np.testing.assert_allclose(cov, cov.T, atol=1e-10)
        self.assertTrue((np.linalg.eigvalsh(cov) > 0).all())


if __name__ == "__main__":
    unittest.main()
