# Task 04A — ArcAD cropping ablation

Run folder: `analysis_task04A_crop_ablation_20261102_01`. This is a frozen-checkpoint **inference-only** intervention; full-frame ArcAD is out of distribution relative to original training crop. No original outputs, checkpoints, labels, or optimizer state were changed.

## Reproduce

From the workspace root, in the project Python environment and with GPU access:

```bash
PY=/media/avlab/1CE6451EE644FA12/2026_10_22_dinomaly/dinomaly2/bin/python
"$PY" analysis_task04A_crop_ablation_20261102_01/work/preflight.py
"$PY" analysis_task04A_crop_ablation_20261102_01/work/run_arcad_branches.py
MPLCONFIGDIR=/tmp/mpl-task04a "$PY" analysis_task04A_crop_ablation_20261102_01/work/analyze_task04a.py
MPLCONFIGDIR=/tmp/mpl-task04a "$PY" analysis_task04A_crop_ablation_20261102_01/work/finalize_task04a.py
```

`reproduce.sh` runs the same sequence; it writes only under this Task 04A folder. ArcAD original uses resize448→center crop392; full-frame uses resize392; D2 full-frame maps are reused from earlier runs. Dataset and checkpoint paths, SHA256 values, GPU, PyTorch/CUDA, cache provenance and timings appear in `protocol_audit.json`, `checkpoint_integrity.json`, and `inference_runtime.json`.

## Results

See `summary.md` for the overall, border, central/small-defect, object-aggregation, and retraining answers. `geometry_validation.json` documents coordinate-ramp checks and source-aligned map grids. `category_metrics.csv`, `paired_bootstrap_ci.csv`, `border_39_audit.csv`, `region_topk_metrics.csv`, `pixel_mask_visibility_sensitivity.csv`, `visibility_status_counts.csv`, and `maps/` contain machine-readable outputs.
