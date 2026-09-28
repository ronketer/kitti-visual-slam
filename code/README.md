# Coursework, compatibility and historical reports

The reusable implementation is in [`../kitti_slam/`](../kitti_slam/).
This directory preserves the course's progression and old imports. It is not
the entry point for a new end-to-end run: use `python -m kitti_slam`.

## Responsibilities

| Source | Historical role | Shared implementation today / disposition |
|---|---|---|
| `ex1.py` | Feature detection, descriptor matching and ratio-test examples; its own image reader and plotting setup | Retain demonstrations. Runtime input handling is `kitti_slam.dataset`; ratio filtering is `kitti_slam.stereo.filter_by_ratio_test`. Import also seeds global Python RNG. |
| `ex2.py` | Epipolar filtering and OpenCV versus linear least-squares triangulation comparisons | Uses shared `classify_matches_by_deviation` and handwritten `solveLLST`. Retain comparison/plot wrappers. |
| `ex3.py`, `ex3code/q1.py`, `q2.py` | Stereo/temporal matching demonstrations | Retain historical policy: q1 lacks the runtime disparity filter. Do not silently merge the two behaviors. |
| `ex3code/q3.py` | PnP hypothesis exercise | Alias of canonical `motion.solve_pnp_and_locations`. |
| `ex3code/q4.py`, `q5.py` | Timed supporters and RANSAC demonstrations | Share projection validation, hypothesis generation and refinement; timed orchestration remains historical. |
| `ex3code/q6.py`, `performance_logger.py` | Earlier multi-frame VO experiment with FrameState, GT/plots and timing | Historical, not the tracking database loop. Canonical loop is `tracking.build_tracking_database`. |
| `ex4.py` | Tracking database build, track statistics, patches and reprojection plots | Uses canonical tracking builder, then saves and analyzes. Its main can perform a full tracking run. |
| `ex5.py` | Single-track/first-window BA diagnostics and full BA trajectory report | `q1`, `q3` and plotting stay here; `q4` calls canonical BA then evaluates/plots. Main loads tracking, runs diagnostics and BA, then saves historical output. |
| `pose_graph_loop_closure.py` | Endpoint-prior experiment, plotting and legacy persistence | Wraps `kitti_slam.pose_graph`; the historical name does not imply visual loop detection. |
| `utility.py` | Historical plots, image crops, GT helpers and compatibility exports | Keep report-only callers here. Runtime already uses concept-based modules. Remaining transform helpers require caller/semantic checks before merging. |
| `alg.py`, old motion helper filenames | Original import surfaces | Re-exports, not alternate algorithms. |
| `config.py`, `geometry.py`, `stereo.py`, `motion.py`, `supporters.py`, `tracking.py`, `tracking_database.py`, `window_selector.py`, `dataset.py`, `detector_config.py`, backend/checkpoint/trajectory modules, `evaluation/` | Compatibility imports after extraction | Alias the canonical modules using `_compat.py`; keep until coursework migration and pickle compatibility are verified. |

## Final report scripts

| Source under `final/` | Responsibility and remaining boundary |
|---|---|
| `ransac_pnp_analysis.py` | `updated_create_tracking_db` enables canonical tracking diagnostics. Script orchestration and plots remain historical. |
| `detector_analysis.py` | Detector comparison, timing, image perturbation/repeatability and cached results. Its alternate stereo path is experimental, not the runtime policy. |
| `plot_absolute_errors.py` | Loads old artifacts, extracts trajectories, calls shared metrics and plots. Uses the recomposed BA trajectory variant. |
| `plot_relative_errors.py` | Loads old artifacts, calls shared metrics, makes full/half consecutive and subsequence plots. Uses stored BA endpoints. |
| `plot_trajectories.py` | Four-trajectory presentation; still hardcodes checkpoint and GT paths. |
| `plot_median_projection_errors.py`, `plot_projection_vs_distance.py` | Projection statistics and plot orchestration. Overlapping projection-error helpers still need behavioral comparison before consolidation. |
| `plot_optimization_errors.py` | Read BA graphs/results, compare costs, generate historical figure. |

Shared trajectory metrics and three BA/pose-graph plot helpers live in
`kitti_slam.evaluation`. The remaining report scripts have not been converted
into a common evaluation CLI. They still target `code/output/` and
`code/final/plots/` or explicit relative paths, not the new artifacts directory.

## Execution and preservation rules

- Historical scripts may read data, write plots/checkpoints and run expensive
  computation. They are not import smoke-test targets as a group.
- Prefer the repository root as working directory for legacy scripts, but this
  alone does not resolve every import/path assumption.
- `ex3code/q1.py`, `q2.py`, and `q6.py` use parent-relative detector imports that
  are incompatible with the top-level `ex3code` imports in direct `ex3.py`
  execution. This pre-existing entry-point problem is not fixed by the package
  migration. Treat that demonstration as requiring a focused import repair.
- `ex5.py`, pose-graph reporting and several final scripts require GTSAM and
  real historical checkpoint payloads. LFS pointers cannot reproduce figures.
- No historical scripts, plots, checkpoints or report files were relocated or
  deleted as part of these boundary documents. `results/` remains the curated
  historical figure location; new runtime artifacts go under `artifacts/`.

## Remaining structural work, in order

1. Make one explicit evaluation entry point accept a checkpoint directory and
   plot destination, using the existing metric and trajectory functions. Start
   with trajectory plotting; preserve figure semantics and existing wrappers.
2. Extract remaining projection statistics only after comparing their sampling,
   residual definitions, exception behavior and aggregation.
3. Repair exercise entry-point imports separately, with import-only checks that
   avoid main execution; retain policy differences and timing demonstrations.
4. Move report scripts into a named reporting area only when paths, command
   entry points and compatibility wrappers are tested. Do not move binaries
   merely to make the tree look tidy.
5. Remove compatibility helpers only when no supported callers or saved objects
   depend on them. The installed `tracking_database` pickle shim has a longer
   compatibility lifetime than ordinary source import wrappers.

Algorithm changes are tracked separately in [validation.md](../docs/validation.md).
