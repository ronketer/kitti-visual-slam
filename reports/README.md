# Historical report and experiment source

This directory owns existing figure-generation programs, not estimation stages.
Reusable trajectory/projection metrics live in `kitti_slam.evaluation` and
trajectory extraction lives in `kitti_slam.trajectory`. Estimation imports
neither this directory nor Matplotlib.

| Program | Preserved purpose |
|---|---|
| `plot_absolute_errors.py`, `plot_relative_errors.py` | Historical metric orchestration, sampling and plots |
| `plot_projection_vs_distance.py`, `plot_median_projection_errors.py` | Projection reports using shared residual and aggregation functions |
| `plot_trajectories.py` | Existing four-trajectory comparison |
| `plot_optimization_errors.py` | BA cost comparison |
| `detector_analysis.py` | Detector timing, distribution and repeatability experiment |
| `ransac_pnp_analysis.py` | Diagnostic-count tracking wrapper and historical plots |
| `pose_graph_loop_closure.py` | Historical endpoint-prior experiment and figures; canonical graph computation is in `kitti_slam.pose_graph` |
| `plots.py` | Shared historical plotting and image-patch helpers extracted from `utility.py` |
| `paths.py` | Shared legacy input/output path names, preserving original destinations |
| `legacy_utility.py` | Import-only compatibility export surface; no algorithm bodies |

From a source checkout, module entry points are `python -m reports.<module>`.
Old `code/final/<module>.py` paths and `code/pose_graph_loop_closure.py` forward
to the same modules. No new plotting command or new figure was introduced.

Imports and computation are separate capabilities. Projection/metric report
modules can import without GTSAM after unused top-level imports were removed;
their GTSAM operations and serialized GTSAM inputs still require the real
library. The pose-graph report itself imports GTSAM. Full report reproduction
is deferred until a compatible environment and actual checkpoint payloads are
available. Some scripts historically read experiment-specific caches such as
`tracking_enhanced` or detector data under the plot directory; those contracts
are retained, not reinterpreted as pipeline caches.

See [migration status](../docs/migration-status.md) and
[artifact organization](../docs/artifacts.md) for explicit retained/deferred work.
