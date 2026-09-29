# Coursework

These source-checkout programs preserve the course progression and educational
experiments. Reusable estimation lives in `kitti_slam`.

| Module | Purpose |
|---|---|
| `ex1.py` | Feature detection, descriptor matching and ratio-test demonstrations |
| `ex2.py` | Stereo geometry and handwritten SVD versus OpenCV triangulation |
| `ex3.py` | Stereo preparation through timed sequence visual odometry |
| `ex4.py` | Tracking statistics and diagnostics |
| `ex5.py` | BA diagnostics and exercise/report visualizations |

Exercise 3 is one sectioned module:

| Exercise section | Descriptive implementation |
|---|---|
| 3.1 | `prepare_stereo_matches` |
| 3.2 | `match_temporal_descriptors` |
| 3.3 | Canonical `solve_pnp_and_locations` from `kitti_slam.motion` |
| 3.4 | `classify_timed_supporters` |
| 3.5 | `run_timed_ransac`, `estimate_timed_motion`, canonical `refine_pose` |
| 3.6 | `FrameState`, `process_timed_stereo_pair`, `run_sequence_odometry` |

`OperationTimer` and local plotting functions support the experiment. The exercise
stereo path deliberately has no runtime disparity filter; timed RANSAC/supporter
loops include timing instrumentation for the experiment.

Run from the checkout, for example `python -m coursework.ex3`. This performs the
two-frame demonstration **and a sequence run**; importing does not run that work.
Exercise main programs may load KITTI, take substantial time, and require real
checkpoint payloads/GTSAM. End-to-end reproduction with the current code has not been validated.
Some demonstrations retain detector initialization and RNG seeding at import.

Shared paths/plotting helpers live in `reports`. Figures go to
`artifacts/coursework/`; reusable pipeline checkpoints go to
`artifacts/sequence-05/checkpoints/`. Original output evidence is preserved in
`results/historical/`.
