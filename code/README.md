# Legacy entry points and artifact archive

This directory now contains forwarding/import compatibility modules and retained
historical artifacts. It has no canonical exercise, report, or SLAM algorithm
implementations.

- [Reusable package](../kitti_slam/): geometry, frontend, tracking, BA, pose graph,
  evaluation and pipeline.
- [Coursework](../coursework/README.md): canonical `ex1.py` through `ex5.py` and
  `ex3code/` implementations.
- [Reports](../reports/README.md): canonical implementations formerly under
  `final/`, plus historical plotting helpers and the endpoint-prior report.
- [Artifact archive](../docs/artifacts.md): existing `output/` and final-project
  figures/checkpoint pointers retained for compatibility and provenance.
- [Migration reconciliation](../docs/migration-status.md): original phases,
  remaining work completed, and justified deferrals.

Existing commands such as `python code/ex3.py` and imports such as
`from ex3code.q3 import q3` continue through forwarding modules. The old
parent-relative detector import issue has been repaired in canonical coursework.
Import aliases share the canonical module object; direct script execution uses
the canonical module as `__main__`. Running historical main functions can still
perform long computations or overwrite historical outputs.

`alg.py`, `utility.py`, and `consts.py` retain old export names. New code should
import the owning concept directly. These facades must not become dependencies
of `kitti_slam`. The installed top-level `tracking_database.py` shim separately
preserves pickle class paths and is required even without this directory.
