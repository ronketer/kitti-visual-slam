# Artifacts and distribution boundaries

| Location | Ownership |
|---|---|
| `results/*.png` | Four curated historical figures used for project presentation |
| `results/historical/coursework/` | Original `code/output/`: 44 PNGs, six LFS pointer files |
| `results/historical/final/` | Original `code/final/plots/`: 28 PNGs, two LFS pointer files |
| `results/historical/manifest.json` | Original paths, destination paths, SHA-256, byte sizes and artifact kinds at `057d991` |
| `results/historical/ex4-console-transcript.txt`, `ex5-console-transcript.txt` | Historical console output preserved from exercise source |
| `slam_final_submission.pdf` | Original submitted report; source wins if implementation differs |
| `artifacts/sequence-05/checkpoints/` | Default new tracking/BA/pose-graph payloads and dependency sidecars |
| `artifacts/coursework/` | Generated exercise figures |
| `artifacts/reports/` | Generated report figures, including experiment subdirectories |
| `dataset/` | Local KITTI inputs; ignored, never packaged |

The archive contains 72 PNGs and eight checkpoint pointers. All archived `.pkl`
files contain Git-LFS pointer text, not usable serialized data. Their object IDs
provide provenance; they cannot serve as inputs to numerical regression tests.
Detector and diagnostic-count report programs still expect experiment-specific
cache inputs from the archive; executing those reports requires recovering the
actual payloads. Generic BA/pose/trajectory reports read pipeline checkpoints.

`reports.paths` separates checkpoint inputs, historical experiment inputs and
new output destinations. Module entry points create generated output directories;
imports do not create directories. Functions accepting explicit save paths leave
that choice to callers. The historical report results have not been reproduced
from these inputs.

Checkpoints retain historical filenames, dictionary formats and tracking class
pickle paths. BA/pose-graph `.pkl.json` sidecars hash the payload and upstream
checkpoint. Restored old payloads can be read, but pipeline resume also requires
valid dependency sidecars. Load only trusted pickles. `python -m kitti_slam --check`
checks selected checkpoints without estimation or writes.

The wheel contains only `kitti_slam`, the tracking pickle shim, and package metadata.
The source distribution additionally contains coursework/report source, tests and
docs. Neither distribution includes dataset files, PNGs, checkpoint pointers,
the report PDF, local environments, build directories, or generated run artifacts.
