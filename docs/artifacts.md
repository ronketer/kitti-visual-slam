# Artifact organization

Source implementations and generated/project artifacts have separate owners.
No historical binary was opened for numerical interpretation, regenerated,
deleted, or renamed during the migration reconciliation.

| Location | Ownership and lifecycle |
|---|---|
| `kitti_slam/` | Installable reusable source only |
| `coursework/` | Historical exercise source only |
| `reports/` | Historical report/experiment source only |
| `code/*.py`, `code/ex3code/*.py`, `code/final/*.py` | Compatibility imports and execution forwarding only |
| `code/output/` | Retained historical exercise figures and checkpoint pointers; 44 PNGs and six pickle pointers at reconciliation |
| `code/final/plots/` and other existing non-source files under `code/final/` | Retained historical final-project artifacts; 28 PNGs and two pickle pointers in total |
| `results/` | Four curated historical figures referenced by the README |
| `slam_final_submission.pdf` | Original submission, retained at its existing linked path |
| `dataset/` | Local validation data, ignored and excluded from distributions |
| `artifacts/<run>/checkpoints/` | New runtime results and dependency-hash sidecars; ignored |
| `artifacts/package-check/` | Local build verification artifacts; ignored |
| `build/`, `dist/`, `*.egg-info/` | Generated Python packaging files; ignored |

The legacy artifact directories are compatibility archives, not locations for
new reusable source. Runtime defaults already use
`artifacts/sequence-05/checkpoints/`. `--output-dir` selects a different runtime
checkpoint directory. Existing historical report entry points retain their
original output paths through `reports.paths` so reproduction behavior is not
silently changed. Use the original directories only for deliberate historical
reproduction; they are not loaded automatically by the runtime.

All eight tracked historical pickle files inspected here contain Git LFS
pointer headers. Renaming these files would not restore their payloads. Their
relocation/consolidation should accompany recovered provenance and report
reproduction. Their current paths remain documented rather than masquerading as
working cached results. The checkpoint validator rejects them before unpickling.

The runtime wheel includes neither these artifacts nor coursework/report source.
The source distribution includes historical Python source and documentation but
excludes datasets, binary reports, figures, checkpoints and generated build/run
directories. `MANIFEST.in` makes this separation explicit.
