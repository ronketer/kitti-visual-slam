"""Legacy entry point; implementation: reports.plot_trajectories."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _compat import forward_module
forward_module(__name__, "reports.plot_trajectories")
