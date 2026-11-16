# Task 03 — verification, calibration and fixed fusion diagnostic

This follow-up uses the existing DiNomaly2 and ArcAD checkpoints and reuses Task 01/02 numeric maps. It does not train, alter checkpoints, or write into the original result folders. Output run ID: `analysis_task03_20261102_01`.

## Reproduction

From the repository root, using the project environment:

```bash
PY=/media/avlab/1CE6451EE644FA12/2026_10_22_dinomaly/dinomaly2/bin/python
MPLCONFIGDIR=/tmp/mpl-task03 "$PY" analysis_task03_20261102_01/work/phase0_1_audit.py
MPLCONFIGDIR=/tmp/mpl-task03 "$PY" analysis_task03_20261102_01/work/finish_phase0.py
MPLCONFIGDIR=/tmp/mpl-task03 "$PY" analysis_task03_20261102_01/work/phase2_fusion.py
MPLCONFIGDIR=/tmp/mpl-task03 "$PY" analysis_task03_20261102_01/work/phase3_examples.py
MPLCONFIGDIR=/tmp/mpl-task03 "$PY" analysis_task03_20261102_01/work/finalize_task03.py
```

The scripts read the Real-IAD Variety source split JSON and source images under `/media/avlab/7458a474-6ea2-4b90-b232-373c3df8d329/real_iad_variety_1024`, and original ArcAD/D2 results in the paths recorded in `protocol_audit.json`. Dependencies are Python, NumPy, SciPy, scikit-learn, Pillow, Matplotlib, OpenCV, PyTorch.

## Data and interpretation

The original 160-category screen uses full-precision D2 `category_heatmap_metrics.json` and ArcAD `results.csv` values. D2 has 83 and ArcAD 100 categories with I-AUROC strictly below 0.90; no source values equal 0.90. Native test splits differ, so this is a category screen, not a paired model comparison. The Task 02 five-category diagnostic is deliberately selected and is not representative of all 160.

The common pixel field is source `[64:960) × [64:960)`, 224×224, independently rechecked against source masks. Task 02 maps are reused; no new model inference was needed. Existing checkpoints are fingerprinted in `protocol_audit.json`. The held-out calibration objects are 10 normal objects/category from the pre-existing Task 02 split. Their pooled 50-object CDF supports only a fixed, diagnostic 50/50 rank-mean baseline and OR baseline; it is not external deployment calibration. No anomaly-based fusion tuning was performed.

See `analysis_report.md`, `issues.md`, and `calibration_reliability.md` for verified findings and limits. Source labels and earlier Task 01/02 outputs remain unchanged.


## Verified high-level findings

The original 160-category threshold counts reproduce: 83 weak D2 and 100 weak ArcAD, with overlap 78 both / 5 D2-only / 22 ArcAD-only / 55 neither. Forty-one laser_diode common-crop empty masks split into 39 source masks outside the crop and two source-all-zero cases. The fixed 50/50 normal-rank fusion is diagnostic only; see `analysis_report.md` for held-out metrics and calibration limits.
