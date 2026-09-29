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
| `ransac_pnp_analysis.py` | Historical diagnostic-count plots; generate data using canonical tracking with `collect_diagnostics=True` |
| `pose_graph_analysis.py` | Historical endpoint-prior experiment and figures; canonical graph computation is in `kitti_slam.pose_graph` |
| `plots.py` | Shared plotting and image-patch helpers |
| `paths.py` | Explicit generated outputs, checkpoint inputs and historical cache inputs |

From a source checkout, module entry points are `python -m reports.<module>`.
Programs import reusable computation from `kitti_slam`.

Imports and computation are separate capabilities. Projection/metric report
modules can import without GTSAM;
their GTSAM operations and serialized GTSAM inputs still require the real
library. The pose-graph report itself imports GTSAM. Full report reproduction
requires actual checkpoint payloads; real GTSAM is validated in WSL. Some scripts
read experiment-specific caches such as `tracking_enhanced` or detector data from
`results/historical/final/`; these are historical cache inputs,
not pipeline caches. Outputs go to `artifacts/reports/`, never into the archive.

See [artifact organization](../docs/artifacts.md) for inputs and output locations,
and [validation](../docs/validation.md) for coverage and known limitations.
