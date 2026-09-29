# Coursework source

The original progression is preserved here: `ex1.py` feature matching,
`ex2.py` triangulation comparison, `ex3.py`/`ex3code/` motion and timed VO,
`ex4.py` tracking analysis, and `ex5.py` BA diagnostics/reporting.

Reusable algorithms live in [`kitti_slam/`](../kitti_slam/). Exercises call those
implementations directly. `_legacy_alg.py` retains the original wildcard export
surface for exercises 1 and 3; it implements no algorithms. The exercise 3
detector imports now address the canonical package explicitly.

From the checkout, use module entry points such as `python -m coursework.ex2`.
Old commands such as `python code/ex2.py` forward to the same implementation.
These are source-checkout programs, not part of the runtime wheel. Imports have
been checked where optional dependencies permit; running a main can load KITTI,
perform a long run and overwrite historical outputs. Exercise 5 execution
requires real GTSAM and checkpoint payloads. No exercise was run end to end as
part of the source migration.

Historical differences are deliberate: exercise stereo does not apply the same
disparity filter as the runtime, and the timed RANSAC/VO path is an experiment,
not another tracking-database implementation. Their algorithm bodies and
plotting policies have not been replaced. Global detector creation and RNG
seeding in some demonstrations remain historical behavior.

Paths come from `reports.paths`, backed by `ProjectPaths` defaults. Explicit
legacy GT/checkpoint literals now resolve through these paths. Compatibility
artifacts stay in their original directories; see [artifacts](../docs/artifacts.md).
