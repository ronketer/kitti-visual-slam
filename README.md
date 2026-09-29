# Stereo Visual SLAM — KITTI Odometry Sequence 05

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-5C3EE8?logo=opencv&logoColor=white)
![GTSAM](https://img.shields.io/badge/GTSAM-4.x-orange)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)

Stereo visual odometry with custom feature tracking and RANSAC-PnP orchestration, windowed bundle adjustment, and pose graph optimization using a ground-truth endpoint prior — developed on KITTI Sequence 05 over 2,600 frames.

![Trajectory comparison: Ground Truth vs PnP vs Bundle Adjustment vs Pose Graph](results/trajectory_comparison.png)
*Historical camera trajectory comparison on KITTI Sequence 05 (X-Z plane).*

## Historical figures

The figures below are archived project outputs. Their underlying checkpoint
payloads are unavailable in this checkout, so these plots have not been
reproduced with the current code. The final stage uses ground truth as an
estimation constraint, and the evaluation uses project-specific sampling policies;
these figures are not a validated standard KITTI benchmark. See
[validation and known limitations](docs/validation.md).

| | |
|---|---|
| ![Historical relative translation errors](results/relative_errors_subsequence_400.png) | ![Historical feature track length distribution](results/track_length_histogram.png) |
| *Relative translation errors for 400-frame subsequences* | *Feature track length distribution* |

## Pipeline

1. **AKAZE stereo matching** — filter matches by vertical deviation < 2 px and positive disparity ≥ 2 px, then triangulate with OpenCV. The handwritten SVD triangulator is also retained for the coursework comparison.
2. **RANSAC-PnP motion estimation** — 4-point PnP with adaptive iteration count (99.9% confidence), refined on all inliers
3. **Windowed bundle adjustment** — GTSAM stereo factors in independently optimized windows, selected using feature count, track overlap, and translation criteria. Default endpoint separations are 5–20 frames; bounds are inclusive and the final window may be shorter.
4. **Pose graph with endpoint prior** — relative constraints from BA marginals and batch GTSAM Levenberg–Marquardt optimization. The historical classroom experiment supplies a tight prior from the final ground-truth pose; automatic visual loop detection is not implemented.

## Repository

```text
kitti_slam/                 canonical reusable SLAM package
  evaluation/              shared metrics and explicit plotting helpers
coursework/ex1.py–ex5.py     educational exercises and timed experiments
reports/                   report/analysis programs and presentation helpers
tests/                     behavioral, numerical and import-boundary contracts
docs/                      architecture, validation and artifacts
results/                   curated historical figures
  historical/              archived submission outputs and provenance manifest
tracking_database.py       sole compatibility shim: historical pickle class paths
slam_final_submission.pdf  original submitted report
pyproject.toml             package dependencies, extras and console entry point
MANIFEST.in                source-distribution inclusion/exclusion rules
requirements.txt           editable install with plotting dependencies
artifacts/                 ignored generated runs, figures and build checks
dataset/                   ignored local KITTI inputs
```

Import reusable code from `kitti_slam`; run coursework/report modules from the
source checkout. Exercise 3 demonstrates stereo preparation, temporal matching,
supporter classification, custom RANSAC and timed sequence odometry. See
[coursework](coursework/README.md).

Start with [architecture and data flow](docs/architecture.md), then
[validation](docs/validation.md),
[reports](reports/README.md), and [artifact provenance](docs/artifacts.md).

## Setup and supported commands

```sh
pip install -e ".[plotting]"
# In a compatible environment (validated using WSL/Linux):
pip install -e ".[optimization]"

python -m kitti_slam --help
python -m kitti_slam --through tracking
python -m kitti_slam                 # full sequence, resumes valid checkpoints
python -m kitti_slam --force         # recompute
python -m kitti_slam --check         # validate selected checkpoints without estimation
python -m kitti_slam --output-dir artifacts/my-run/checkpoints

python -m coursework.ex1
python -m reports.plot_trajectories  # existing report; requires real checkpoint payloads
```

The installed `kitti-slam` command and `python -m kitti_slam` call the same
pipeline.
The base install requires NumPy, OpenCV and tqdm; plotting is optional. GTSAM is
required only for backend computation, GTSAM data and the full pipeline. Frontend
imports/tests remain usable on native Windows without GTSAM.

Place KITTI grayscale images/calibration in `dataset/sequences/05/` and ground
truth in `dataset/poses/05.txt`. Source/editable installs use repository-relative
defaults even from another working directory. For a wheel, set `KITTI_SLAM_ROOT`
to a data/output workspace; otherwise defaults use the import-time working directory.
Explicit `ProjectPaths` and injected calibration/detectors are supported:

```python
from pathlib import Path
from kitti_slam.config import ProjectPaths
from kitti_slam.detector_config import create_detector_and_matcher
from kitti_slam.tracking import build_tracking_database

paths = ProjectPaths(dataset_root=Path("D:/KITTI"), sequence="05")
detector, matcher = create_detector_and_matcher("AKAZE")
db = build_tracking_database(paths=paths, detector=detector, matcher=matcher,
                             last_frame=20)  # inclusive 0 through 20
```

Default tracking remains frames 0–2599 and AKAZE. Nonzero starting frames are
rejected because tracking IDs assume zero-based frames.

## Checkpoints and project artifacts

New checkpoints go to `artifacts/sequence-05/checkpoints/`, coursework figures to
`artifacts/coursework/`, and report figures to `artifacts/reports/`. Historical
outputs are preserved under `results/historical/`, with original paths and byte
hashes. All eight archived `.pkl` files are Git-LFS pointer text, not usable data.
Experiment-specific report inputs still require recovery of their real payloads.

The root `tracking_database.py`
shim preserves the class paths used in trusted old and new tracking pickles; it
is included in the wheel.

BA/pose-graph `.pkl.json` sidecars contain payload/upstream SHA-256 digests. Resume
rejects missing/stale metadata, malformed payloads and LFS pointers. Atomic writes
protect existing payloads from failed serialization. These checks do not establish
numerical validity or fingerprint source, dataset, thresholds or RNG state. Use
`--force` or another run directory when those inputs change. Only load trusted
pickles; deserialization can execute code.

The wheel contains the runtime package and tracking pickle shim. The sdist also
includes coursework/report source, tests and docs; neither includes datasets,
figures, checkpoint pointers, local environments or the submitted PDF.

## Validation

```sh
python -B -m unittest discover -s tests -v
uv build
```

The test suite passed **54 tests with zero failures and zero skips** on WSL2
Ubuntu 24.04.3, Python 3.12.3 and GTSAM 4.3.0, including four real-GTSAM numerical
checks. [Validation notes](docs/validation.md) describe the tested environment,
coverage and known limitations. The tests do not establish full-sequence KITTI
accuracy or reproduce the historical figures.

## Report

[Original final submission](slam_final_submission.pdf). The live source takes
precedence where the report and implementation differ.
