"""Package imports and historical tracking-checkpoint compatibility."""

import ast
import importlib
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class PackageTests(unittest.TestCase):

    def test_canonical_imports_and_old_pickle_without_code_on_path(self):
        script = f'''
import sys, importlib.abc, pickle
sys.path.insert(0, {str(ROOT)!r})
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'gtsam', 'matplotlib', 'alg', 'utility', 'ex5'}}:
            raise AssertionError(fullname)
sys.meta_path.insert(0, Block())
import kitti_slam
assert 'cv2' not in sys.modules
from kitti_slam import tracking, bundle_adjustment, pose_graph, pipeline, detector_config
from kitti_slam.evaluation import trajectory_errors, projection_errors, ground_truth
from kitti_slam.tracking_database import Link
assert 'detector' not in vars(detector_config)
legacy = b'ctracking_database\\nLink\\n(F5.0\\nF4.0\\nF3.0\\ntR.'
link = pickle.loads(legacy)
assert type(link) is Link and (link.x_left, link.x_right, link.y) == (5., 4., 3.)
assert b'tracking_database' in pickle.dumps(link)
assert {str(ROOT / 'code')!r} not in sys.path
'''
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run([sys.executable, "-I", "-B", "-c", script], cwd=directory,
                                    capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_package_has_no_bare_imports_of_its_own_modules(self):
        local = {file.stem for file in (ROOT / "kitti_slam").glob("*.py")}
        for file in (ROOT / "kitti_slam").rglob("*.py"):
            for node in ast.walk(ast.parse(file.read_text(encoding="utf-8"))):
                if isinstance(node, ast.ImportFrom) and node.level == 0:
                    self.assertNotIn(node.module.split('.')[0], local, str(file))
                elif isinstance(node, ast.Import):
                    for alias in node.names:
                        self.assertNotIn(alias.name.split('.')[0], local, str(file))


if __name__ == "__main__":
    unittest.main()
