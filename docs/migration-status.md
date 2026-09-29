# Architecture migration reconciliation

Reconciled against the accepted audit migration sequence and the live source on
2026-09-29. The starting point for this verification was commit `8ff50c9`.
Its commit title said the refactor was complete; inspection found the last
organization phase was still partial. This document records the remaining work
completed in this pass, without introducing another architecture or algorithm.

## Accepted phases: before and after this pass

| Original migration area | Status at start of this pass | Final disposition and evidence |
|---|---|---|
| Extract shared geometry and coordinate helpers | Complete for active runtime; partial for historical utility functions | **Complete.** `kitti_slam.geometry` also owns the two remaining pure helpers from `utility.py`. Bodies retained; source-to-target transform tested. |
| Make dataset paths/calibration/detector/frame inputs explicit; isolate optional GTSAM imports | Complete for runtime; partial for historical entry points | **Complete.** Existing `ProjectPaths`/injection APIs retained. Historical paths centralized in `reports.paths`; remaining literal GT/checkpoint paths use those constants. Broken parent-relative detector imports in exercise 3 repaired. |
| Extract stereo processing, motion estimation and supporter validation | Complete | **Complete.** Canonical `stereo`, `motion`, `supporters` remain unchanged. Handwritten SVD, custom RANSAC control and project-specific validation remain first-class code. |
| Unify tracking loop and database updates | Complete | **Complete.** Pipeline, exercise 4 and diagnostic wrapper share `tracking.build_tracking_database`. Database class paths/payloads remain compatible. Historical timed VO demonstration remains distinct. |
| Separate BA construction/optimization and window selection from exercises and reporting | Complete structurally; numerical validation deferred | **Complete structurally.** `bundle_adjustment`, `window_selector`, `gtsam_geometry`; exercise 5 retains only its diagnostic/report orchestration around these APIs. Numerical validation remains explicitly deferred. |
| Separate pose graph, trajectory extraction and evaluation | Mostly complete; projection evaluation still duplicated/mixed into scripts | **Complete structurally.** Projection residual and aggregation functions now live in `evaluation.projection_errors`. Duplicate residual bodies were identical and merged; distinct per-window/distance aggregation policies are preserved. GT center helpers live in `evaluation.ground_truth`. |
| Installable package, repository organization, artifacts, compatibility and docs | Package/CLI worked, but organization was **partial** | **Complete with the retention decisions below.** Canonical reusable package, `coursework/`, `reports/`, forwarding-only legacy entry points, explicit package contents, source distribution manifest and documentation. |
| Regression coverage and baseline comparison | Small synthetic tests complete; GTSAM/KITTI baseline work deferred | **Complete for feasible structural validation.** Added import boundaries, projection aggregation, window policy and moved geometry checks. Full numerical validation remains deferred below. |

There is no remaining identified feasible source move or consolidation from
these migration phases. This is structural completion, not certification of the
SLAM mathematics or the historical accuracy numbers.

## What was missing and was finished here

| Previous location/function | Current owner | Action / reason |
|---|---|---|
| `code/ex1.py` through `ex5.py`, `code/ex3code/` | `coursework/`, `coursework/ex3code/` | **MOVE.** Separate teaching progression from runtime. Old paths forward execution/imports. |
| `code/final/*.py` | `reports/*.py` | **MOVE.** Separate report orchestration and experiments from reusable estimation. No new figures or plotting feature added. |
| `code/pose_graph_loop_closure.py` | `reports/pose_graph_loop_closure.py` | **MOVE.** Keep the historical endpoint-prior experiment and its report label distinct from canonical pose-graph computation. |
| `code/utility.py` plotting, patch and figure helpers | `reports/plots.py` | **SPLIT.** Historical visual helpers retain exact function bodies. |
| `utility.compute_camera_to_camera_transform`, `find_transformation` | `kitti_slam/geometry.py` | **MOVE.** Pure math stays independent of Matplotlib. No current internal callers were found, but preserve public compatibility rather than delete speculatively. |
| `utility.parse_gt_line`, `read_ground_truth_poses` | `kitti_slam/evaluation/ground_truth.py` | **MOVE.** Preserve the existing GT interpretation and inclusive frame bound; reader accepts explicit paths. |
| Identical `parse_gt_line` in exercise 5 and pose-graph report | `dataset.parse_gt_line_matrix` | **MERGE.** Identical parse/reshape implementation; the center-returning helper stays distinct. |
| `ex1.read_images` | `dataset.read_images` | **MERGE.** Same grayscale/six-digit image reader; repository-anchored paths replace duplicated CWD construction. |
| Both projection report `calculate_projection_error` functions | `evaluation.projection_errors.calculate_projection_error` | **MERGE.** Both average the left and right 2D residual norms; preserve that formula. |
| PnP/distance/window projection aggregators | `evaluation.projection_errors` | **MOVE.** Preserve track-length filtering, reference frames, iteration order, missing-value/exception behavior and return shapes. GTSAM imports are local. |
| `consts.py`, scattered literal GT/checkpoint paths | `reports.paths` and existing runtime `ProjectPaths` | **SPLIT/CONSOLIDATE.** Legacy defaults stay the same; canonical historical modules no longer need `code/` on `sys.path`. |
| `alg.py`, utility export surface | `coursework._legacy_alg`, `reports.legacy_utility` | **KEEP as compatibility facades.** Import exports only, no duplicate algorithm bodies. |
| Implicit source-distribution contents | `MANIFEST.in`, `pyproject.toml` | **MAKE EXPLICIT.** Wheel includes only reusable package and tracking-pickle shim. Source distribution includes docs, tests, coursework, report source and forwarding modules, not dataset/plot/checkpoint payloads. |

The already-approved flat package layout (`kitti_slam/`) is retained instead of
making a second mechanical move under `src/`. It provides the proposed package
boundaries and installation behavior. The original target tree was a module
organization proposal; adding another directory layer would not close a missing
dependency boundary. Likewise, the original report remains at its linked root
path instead of being moved just to match an illustrative tree.

## Intentional retention and justified deferrals

| Item | Disposition | Why / condition for revisiting |
|---|---|---|
| Real GTSAM BA, pose graph and projection numerical validation | **Deferred** | No compatible stable GTSAM wheel in the checked native Windows environment. Run the real tests in a suitable environment; do not substitute an optimizer or weaken assertions. |
| KITTI/full-pipeline and historical result reproduction | **Deferred** | Historical checkpoints are LFS pointers, not numerical baselines. Recover trusted payloads or establish a labeled baseline in a GTSAM environment before reproducing metrics. |
| Physical relocation/deletion of historical binaries | **Intentionally retained**, not a missing source migration | `code/output/` has 44 PNGs and six LFS pickle pointers; `code/final/` has 28 PNGs and two LFS pickle pointers. Keep original provenance/paths until reproduction is possible. They are catalogued legacy artifacts, excluded from distributions; canonical source has moved out. New runtime outputs already go under ignored `artifacts/`. |
| Legacy script/import wrappers and installed `tracking_database` shim | **Intentionally retained** | Supported commands and pickle class paths still use them. Removing them is not required for clean canonical dependencies. |
| Timed exercise RANSAC, exercise stereo policy and detector-comparison experiment | **Intentionally retained** | Educational/experimental behavior differs from the runtime, so collapsing them would change the work being demonstrated. They now have historical owners. |
| Separate BA trajectory extraction variants and report sampling | **Intentionally retained** | Stored versus recomposed endpoints, half-sequence selection and nearest-keyframe snapping are behavior, not formatting duplication. |
| Algorithmic/numerical issues, misleading legacy result keys | **Outside the structural refactor** | Listed in [validation.md](validation.md). They require separate decisions and numerical evidence. Existing keys remain for compatibility. |

A new unified evaluation CLI, new trajectory figures, richer provenance tracking
and changes to evaluation protocols are not prerequisites to this structural
migration. No such feature was added. Existing historical report programs retain
their output conventions and can still be called explicitly.

## Verification

- Final suite: **56 tests run, 52 passed, four real-GTSAM tests skipped** on
  native Windows. No KITTI sequence or report-generation main was run.
- Source distribution and wheel built successfully. Archive contents were
  inspected: the wheel contains only `kitti_slam` plus the pickle shim; the
  source distribution includes historical source/tests/docs but no dataset,
  plot, pickle, PDF or bytecode payloads. Isolated wheel imports, legacy pickle
  class lookup and CLI help passed with GTSAM, Matplotlib and historical-module
  imports blocked.
- Existing runtime algorithm modules are unchanged in this pass, except adding
  two unchanged legacy helpers to `geometry.py`.
- 78 moved top-level functions/classes match the previous commit's syntax trees
  after normalizing imports and the explicit path substitutions. The duplicate
  matrix parsers and image reader are separately replaced by existing equivalents.
- New synthetic checks exercise projection residual semantics, distance grouping,
  missing values, source-to-target geometry, inclusive window endpoints, tail
  windows, historical fallback and overlap policy.
- Import checks exercise canonical coursework/report modules from another working
  directory without adding `code/` to the import path. Execution dispatch at old
  script paths is checked without launching long-running scripts.
- The real-GTSAM checks remain real tests and are skipped when GTSAM is absent.
  Symbolic objects in projection aggregation checks establish data flow only.

See [architecture.md](architecture.md), [coursework](../coursework/README.md),
[reports](../reports/README.md), and [artifacts](artifacts.md) for the resulting
ownership boundaries.
