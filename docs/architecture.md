# Architecture of the current implementation

This describes the live `kitti_slam` package. The original submission and
coursework explain its history; source code determines runtime behavior. The
abandoned `archive/opencv-modernization` branch is not a design dependency.

## System in one minute

The project implements offline stereo visual odometry, persistent feature
tracking, independent bundle-adjustment windows, and a pose graph over window
endpoints. The frontend implements correspondence assembly, geometric filtering,
adaptive RANSAC control, and track management. OpenCV supplies feature detection,
matching, triangulation, and PnP solvers. The backend constructs its own GTSAM
graphs and delegates nonlinear optimization and marginal computation to GTSAM.

The final experiment anchors the last pose to ground truth. There is no visual
place recognition, loop-candidate verification, or incremental ISAM2 backend.
Describe the final stage as **pose graph optimization with an endpoint prior**.

## End-to-end flow

```text
python -m kitti_slam / kitti-slam / run_pipeline.py
                         |
                  pipeline.main
                         |
dataset.read_cameras + read_images     detector_config
                         |                  |
                  tracking.build_tracking_database
                         |
     stereo.process_stereo_pair (left/right keypoints, descriptors,
             filtered matches, aligned Nx3 cloud)
                         |
     temporal left-to-left matches + find_stereo_temporal_matches
                         | (l0, r0, l1, r1) keypoint indices
     motion.perform_motion_estimation
       custom adaptive RANSAC -> OpenCV P3P hypotheses
       -> supporters: projections in all four views
       -> OpenCV iterative PnP refinement
                         |
     compact descriptor-row remapping -> TrackingDB.add_frame
     relative pose -> compose_extrinsics -> absolute pose
                         |
     tracking_with_geometric_validation_without_far_tracks.pkl
                         |
     WindowSelector.next_window -> bundle_adjustment.run_bundle_adjustment
       local pose initialization + stereo backprojection
       -> prior and stereo factors -> independent GTSAM LM solves
                         |
                    ba_results.pkl
                         |
     pose_graph.extract_relative_pose_covariance
       -> relative endpoints + regularized conditional covariance
                         |
     build_pose_graph <- explicit final ground-truth pose from pipeline
       -> endpoint priors + BetweenFactors -> batch LM
                         |
               loop_closure_results.pkl

Explicit evaluation: checkpoint data -> trajectory + evaluation metrics/plots
Historical source entry points: reports/, coursework/ex5.py
Legacy paths under code/ forward to these modules.
```

The pipeline writes checkpoints but does not automatically generate plots.
BA and pose graph `.pkl.json` sidecars record payload and upstream SHA-256
digests. See the [README checkpoint contract](../README.md#checkpoints-and-project-artifacts).

## Module boundaries and implementation ownership

All paths below are within [`kitti_slam/`](../kitti_slam/).

| Module / principal functions | Responsibility | External primitives |
|---|---|---|
| `config.ProjectPaths`, `dataset.read_images/read_cameras` | Paths, KITTI filenames, calibration decomposition; no estimation | OpenCV image reads, NumPy matrix inverse |
| `detector_config.create_detector_and_matcher` | AKAZE/ORB Hamming or SIFT L2 component construction | OpenCV detectors and BFMatcher |
| `stereo.classify_matches_by_deviation`, `trangulate_inliers` | Vertical/disparity tests, aligned stereo matches/cloud | `cv2.triangulatePoints` in the runtime |
| `stereo.solveLLST` | Handwritten linear system construction, homogeneous solution and normalization; retained from exercise 2 | NumPy SVD; no higher-level triangulation replacement |
| `stereo.find_stereo_temporal_matches` | Join two stereo match sets with temporal matches using original keypoint indices | Python collections |
| `motion.solve_pnp_and_locations`, `perform_ransac_loop`, `refine_pose` | Four-point sampling, solver retry, best hypothesis, adaptive stopping, refinement orchestration | OpenCV P3P, iterative PnP, Rodrigues; Python random |
| `supporters.validate_projections_batch`, `find_ransac_iteration_supporters` | Transform, project, test pixel residuals and intersect masks across four views | NumPy |
| `geometry` | 3x4 composition, point transformation, camera center | NumPy |
| `tracking.build_tracking_database` | Frame loop, original-index to compact-row mapping, motion calls, pose accumulation | Shared frontend and database; OpenCV DMatch containers |
| `tracking_database.TrackingDB`, `Link` | Track IDs, collisions, observations, leftovers, frame/track lookup, legacy serialization | Python pickle, NumPy, OpenCV containers |
| `window_selector.WindowSelector` | Search window endpoints using count, overlap and translation criteria; fallback policy | Database and camera centers |
| `gtsam_geometry` | Extrinsic-to-Pose3 conversion, stereo calibration, stereo backprojection | GTSAM Pose3, StereoCamera and Cal3_S2Stereo |
| `bundle_adjustment.initialize_pose_keys/initialize_landmarks/add_stereo_factors_to_graph` | Local pose chain, last-observation landmark initialization, observation factors | GTSAM values, stereo factors and backprojection |
| `bundle_adjustment.run_bundle_adjustment` | Window selection, independent solves, error-reduction gate, endpoint chaining | GTSAM Levenberg–Marquardt |
| `pose_graph.extract_relative_pose_covariance/build_pose_graph/optimize_pose_graph` | Endpoint constraints, regularization and covariance block extraction, chain graph and explicit priors | GTSAM marginals, BetweenFactorPose3, LM; NumPy inverses |
| `trajectory` | Extract PnP/BA/pose-graph trajectories; retain distinct stored and recomposed BA variants | GTSAM pose operations where needed |
| `evaluation.trajectory_errors`, `evaluation.plots` | Historical metric formulas and plotting functions | NumPy, OpenCV rotation conversion, GTSAM relative poses, Matplotlib plots |
| `evaluation.projection_errors`, `evaluation.ground_truth` | Shared stereo residual/distance/window aggregations and historical GT-center interpretation; no plots or checkpoint I/O | NumPy; deferred GTSAM projection operations; GT reader accepts paths |
| `checkpoints`, `pipeline` | Persistence validation and atomic publication; stage order, resume and GT endpoint loading | Standard library; stage APIs |

## Dependency direction

- `pipeline` depends on computation, dataset I/O, and checkpoints.
- `tracking` depends on stereo, motion, geometry, inputs and the database.
- `motion` depends on supporters and geometry; neither depends on tracking.
- BA depends on database/window selection and GTSAM geometry. It can read
  calibration when `reference_extrinsics` is omitted; the runner supplies it.
- Pose graph consumes BA result dictionaries; it does not call the BA runner or
  read ground truth itself.
- Evaluation consumes estimation results. Estimation does not import evaluation.
- Coursework imports the package through compatibility shims. The package has
  no imports from `code/`, `alg`, `utility`, or exercise modules.

GTSAM imports occur inside the operations that need it. Core imports therefore
work on native Windows without GTSAM. Importing the package itself initializes
no OpenCV detector. Matplotlib is confined to the plotting module in the package.

## Data and coordinate contracts

| Boundary | Contract in the source |
|---|---|
| Calibration | `read_cameras` returns NumPy `(K, M1, M2)`. Projection matrices are `K @ M1` and `K @ M2`. `read_calibration` returns a different object: GTSAM stereo calibration. |
| Extrinsics | A NumPy `3x4 [R\|t]` maps reference coordinates to camera coordinates: `X_camera = R X_reference + t`. Camera center is `-R.T @ t`. |
| Composition | `compose_extrinsics(second, first)` applies `first`, then `second`. Tracking composes the new relative extrinsic after the previous absolute extrinsic. |
| GTSAM poses | `create_pose_from_extrinsics` inverts the matrix using `R.T` and `-R.T @ t`, producing a camera-to-reference Pose3. |
| Point arrays | Triangulated and transformed clouds are `N x 3`. Despite its docstring, `solveLLST` returns three normalized coordinates, not a four-element homogeneous vector. |
| Index spaces | Original keypoint indices, stereo-cloud row indices and compact database descriptor rows are distinct. Preserve the explicit maps in `tracking` and `motion`. |
| Observation | `Link(x_left, x_right, y)` uses a shared vertical coordinate. GTSAM measurement order is `(u_left, u_right, v)`. |
| Frames | Tracking starts at zero and uses inclusive bounds, by default 0–2599. Relative extrinsics indexed by frame `i` connect the preceding frame to `i`. |
| Windows | `(start, end)` is inclusive; successive windows share an endpoint. Default size parameters 5–20 measure endpoint differences, so ordinary windows include 6–21 frame IDs. The tail may be shorter. |

The code treats parsed ground-truth matrices as extrinsics when computing centers
and creating GTSAM poses. This documents the implemented convention, not a claim
that the convention is correct for KITTI. Resolve that separately before
reinterpreting historical errors; see [validation and open concerns](validation.md).

## Backend and persistence contracts

BA uses one local graph per window, an identity first-pose prior with unit
six-dimensional noise, and unit stereo measurement sigmas. It initializes a
landmark from the last available observation in the window. A factor whose
initial error exceeds 1000 raises; a solve whose optimized error exceeds half
the initial error also raises. These are existing policies, not newly tuned gates.

Each BA dictionary retains `start_kf`, `end_kf`, `graph`, `initialEstimate`,
`pose_keys`, `point_keys`, `tracks`, `frames`, `result`, `abs_start_pose`, and
`abs_end_pose`. These objects include GTSAM values and graphs, so the checkpoint
is not a portable NumPy-only format.

Pose graph constraints contain `start_kf`, `end_kf`, `relative_pose`, and
`conditional_cov`. The implementation computes a regularized 12x12 inverse
joint covariance and inverts its lower-right 6x6 information block with further
regularization (`eps=1e-4`). That operation is preserved; its interpretation as
relative-pose uncertainty needs a separate mathematical review.

The pose graph anchors frame zero and the supplied final pose with sigmas
`1e-6`. It also overwrites the final initial estimate with the endpoint pose.
The legacy result key `poses_without_loop_closure` consequently refers to this
initial estimate, not an unconstrained optimized graph. `poses_with_loop_closure`
holds the batch-optimized result. Key names remain for compatibility.

Tracking pickle keys and the `tracking_database` class module names are retained.
The installed top-level shim supports old pickles without putting `code/` on
`sys.path`. Legacy imports alias canonical module objects, including lazy
detector attributes; they do not maintain separate algorithm implementations.

## Reading order for an interview

1. `pipeline.py`: three stage calls and where ground truth enters.
2. `tracking.py`: the complete frontend loop and its index remapping.
3. `stereo.py`, `motion.py`, `supporters.py`: filtering, custom RANSAC and four-view validation.
4. `tracking_database.py`: how observations become persistent tracks.
5. `window_selector.py`, `bundle_adjustment.py`: policy, graph construction and independent optimization.
6. `pose_graph.py`: uncertainty extraction, chain constraints and explicit endpoint prior.
7. `trajectory.py`, `evaluation/`, and [validation.md](validation.md): what is measured and what remains unverified.

For historical responsibilities, see [coursework](../coursework/README.md) and
[reports](../reports/README.md). See [migration status](migration-status.md) for
the original phase checklist and justified deferrals. `code/` now contains only
forwarding/import modules and retained artifacts.
