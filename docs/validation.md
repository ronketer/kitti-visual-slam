# Validation and known limitations

## Tested environment and results

The latest recorded full suite completed with **54 passed, 0 failed, 0 skipped**.

| Component | Tested version |
|---|---|
| OS | Ubuntu 24.04.3 under WSL2 |
| Python | 3.12.3 |
| GTSAM | 4.3.0, installed from a PyPI wheel |
| NumPy | 2.5.3 |
| OpenCV (`opencv-python`) | 4.14.0.94 |
| Matplotlib | 3.11.1 |
| tqdm | 4.70.1 |

These are the versions used for the recorded run, not a guarantee of compatibility
with every version allowed by the package metadata. The wheel and source
distribution also built successfully; distribution contents, isolated wheel
imports, CLI help and historical tracking-pickle loading were checked.

## Running the checks

From a source checkout in an environment with a compatible GTSAM wheel:

```sh
python -m pip install -e ".[plotting,optimization]"
python -B -m unittest discover -s tests -v
```

To build distributions with [uv](https://docs.astral.sh/uv/):

```sh
uv build
```

Native Windows can run the frontend and tests that do not require GTSAM. Install
`.[plotting]` for that subset. Four numerical tests skip if GTSAM is absent;
confirm that they execute when validating backend numerical behavior. Backend
operations and checkpoints containing GTSAM objects require the real library.

## Coverage and limits

The dataset-independent suite covers geometry; synthetic stereo and seeded
OpenCV RANSAC/PnP; multiframe tracking; window boundaries; backend construction
and orchestration; covariance extraction; checkpoint failures and pickle
compatibility; evaluation policies; import direction; and Exercise 3's distinct
stereo/timing behavior.

The four real-GTSAM numerical tests exercise:

- Stereo BA graph optimization.
- Two-window BA optimization.
- Endpoint-constrained pose-chain optimization.
- Joint-marginal extraction.

Passing these checks does not establish full-sequence KITTI accuracy, validate
every mathematical convention, or reproduce the submitted results. The archived
pickle files are Git-LFS pointer text, not numerical payloads. Coursework and
report programs have not been reproduced end to end with the current code.

For reproducible dataset runs, record image/calibration checksums, frame bounds,
dependency versions and RNG seeds. Compare observations/IDs, transforms, window
endpoints, graph keys, costs and covariance blocks with a recorded reference run.
Checkpoint sidecars hash payloads and upstream checkpoints, not source, dataset,
thresholds or RNG state; use a new run directory or `--force` when these inputs
change.

## Known limitations and open mathematical questions

The table distinguishes source behavior from questions requiring further
validation. Synthetic test success does not resolve the coordinate or covariance
interpretation questions below.

| Location | Observed behavior or unresolved concern | Validation needed |
|---|---|---|
| `motion.solve_pnp_and_locations`, `refine_pose` | Right-camera extrinsics compose the estimated left transform after the baseline transform. The physical ordering needs checking under rotation. | Synthetic rotating stereo-rig test with a known baseline |
| `motion.refine_pose` | Starts iterative PnP with zero rotation/translation, despite a comment referring to the RANSAC estimate. Returns refined poses but `perform_motion_estimation` keeps the old supporters. | Decide initialization and reclassification policy explicitly |
| `stereo.trangulate_inliers`, `tracking.build_tracking_database` | An empty filtered point list can have shape `(0,)` before `shape[1]` is accessed; insufficient temporal matches return `None` poses that the tracking loop does not handle. | Define empty-frame, failed-motion and recovery behavior |
| `motion.perform_ransac_loop` | `MAXITERATIONS` is not the loop bound; the initial outlier assumption can imply a very large adaptive budget. | Bound runtime and test degenerate samples without silently changing historical stopping |
| `supporters.validate_projections_batch`, `stereo.solveLLST` | Division by projected depth or homogeneous scale lacks explicit finite/degeneracy guards. | Numerical and cheirality tests |
| `window_selector` | Overlap compares start-frame tracks with candidate-minus-one tracks. Fallback returns the minimum window even when criteria fail. | Document or revise intended policy separately |
| `WindowSelector.peek_next_window/current_window` | Peek calls nonexistent `find_next_window`; current-window state can have identical endpoints after selection. Neither is used by the BA runner. | Repair or remove only after caller checks and focused tests |
| `bundle_adjustment.run_bundle_adjustment` | Global endpoint chaining uses the optimized local end pose while the local start is constrained by a soft prior. | Check whether start-to-end relative composition is required |
| `pose_graph.extract_relative_pose_covariance` | A conditional endpoint covariance block is used as relative-pose covariance; key/block ordering, tangent-frame interpretation and omitted Jacobians require review. | Derive and test relative-pose uncertainty explicitly |
| `pipeline.run_stage_3` | Uses the last ground-truth pose as a tight estimation prior. | Keep this experiment labeled; visual loop detection would be new algorithmic work |
| GT readers and `gtsam_geometry.create_pose_from_extrinsics` | Parsed KITTI pose matrices are treated as world-to-camera extrinsics and inverted. | Verify the dataset convention before changing transforms or metrics |
| `evaluation.trajectory_errors`, `reports/plot_relative_errors.py` | Subsequence computation uses the first half of available keyframes and nearest-endpoint snapping; the report also plots half-sampled consecutive results. | Establish an explicit evaluation protocol before claiming standard KITTI metrics |
| Historical figures and submitted report | Results have not been reproduced from restored payloads. | Recover input provenance and payloads before making reproducible accuracy claims |
