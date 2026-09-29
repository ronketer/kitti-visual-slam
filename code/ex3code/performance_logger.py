"""Legacy entry point; implementation: coursework.ex3code.performance_logger."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _compat import forward_module
forward_module(__name__, "coursework.ex3code.performance_logger")
