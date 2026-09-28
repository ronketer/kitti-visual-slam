# Validation status and separate correctness work

## What has been verified

The package migration was checked on native Windows with CPython 3.13.5:
48 tests ran, 44 passed and four real-GTSAM tests were skipped. This is a recorded
refactor validation result, not a permanent promise about other environments.
Editable installation, a built wheel's isolated imports, the console entry point
from another directory, and historical tracking-pickle class loading were checked.

Run the suite after installing the `plotting` extra:

```sh
python -B -m unittest discover -s tests -v
```

| Layer | Existing checks | What it does not establish |
|---|---|---|
| Pure geometry | Transform composition, camera centers, empty arrays, dtype behavior | All dataset coordinate conventions |
| Synthetic frontend | Stereo thresholds, handwritten SVD, correspondence ordering, seeded real OpenCV RANSAC/PnP, right-view outliers | Full-sequence tracking accuracy or failure recovery |
| Synthetic tracking | Three-frame IDs, collisions, lost/new tracks, remapping, pose accumulation, diagnostic counts, pickle compatibility | Long-run memory and data-dependent drift |
| Backend orchestration | Factor/order and result contracts, original error gate, covariance formula, explicit GT endpoint, no plotting/I/O in core operations | Solver accuracy; test doubles are call markers, not numerical solvers |
| Real GTSAM (currently skipped) | Synthetic stereo BA, two windows, endpoint-constrained pose chain, joint marginal extraction | KITTI end-to-end accuracy or historical result reproduction |
| Checkpoint/package boundary | LFS and malformed payload rejection, atomic-write failure, stale upstream hashes, dependency errors, module identity and import direction | Complete schema/type validation, numerical health, source/dataset provenance |
| Evaluation | Rotation metric, historical half-sequence/sampling behavior, stored endpoints and missing PnP poses | Correctness of every historical plot or reported aggregate |

No full KITTI run was performed for the structural refactor. Historical LFS
pointer files are not usable numerical baselines. The GTSAM wheel-only check in
this Windows environment found no compatible stable package; compilation and
environment replacement were intentionally deferred.

## Next validation in a suitable GTSAM environment

1. Run the existing suite with real GTSAM and record Python, OpenCV, NumPy and
   GTSAM versions. Confirm the four gated tests execute rather than skip.
2. Establish a short, explicit KITTI fixture: calibration, a small consecutive
   frame range beginning at zero, and input checksums. Use the Python APIs for
   short frame ranges; the CLI currently retains the full sequence defaults.
3. Compare tracking observations/IDs exactly and transforms numerically against
   the same pre-change implementation. Fix Python and NumPy random seeds for
   comparisons without changing runtime defaults. Solver/platform tolerances
   should be measured, documented and justified.
4. Compare window boundaries, graph sizes, keys, initialization, optimized costs,
   endpoint poses and covariance blocks. Keep tests of wiring separate from
   numerical comparisons.
5. Recover trusted historical checkpoint payloads or produce a clearly labeled
   new baseline. Run full sequence regression only as an explicit validation
   task. Record GT usage and the precise metric sampling and normalization.

Checkpoint sidecars currently hash only the payload and upstream checkpoint.
They do not identify source revision, calibration, image contents, thresholds,
RNG state, or dependency versions. Use a new run directory or `--force` after
changing these inputs; do not interpret a successful resume as provenance proof.

## Correctness concerns excluded from structural cleanup

These observations describe existing source behavior. No fix is implied by the
package move. Each change needs its own expected behavior and regression evidence.

| Location | Observation / unresolved concern | Separate follow-up |
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
| `evaluation.trajectory_errors`, `code/final/plot_relative_errors.py` | Subsequence computation uses the first half of available keyframes and nearest-endpoint snapping; the report also plots half-sampled consecutive results. | Establish an explicit evaluation protocol before claiming standard KITTI metrics |
| Historical README results | Aggregate numbers and image provenance have not been reproduced from restored payloads. | Recover provenance or publish new labeled measurements |

Other historical entry-point and path problems are recorded in the
[coursework map](../code/README.md). They do not block canonical package imports.
