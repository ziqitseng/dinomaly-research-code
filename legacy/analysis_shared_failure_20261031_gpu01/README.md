# Reproduction and data notes

This folder contains a paired, inference-only analysis of DiNomaly2 (D2) and ArcAD on the exact shared image paths recorded in `shared_manifest.json`. The categories are `lattice_block_plug`, `laser_diode`, `power_jack` (a failure-enriched, selection-biased set), plus `LED_indicator` and the preselected favorable comparison `thyristor`. Results must not be generalized to all 160 categories.

ArcAD's test samples were intersected with D2's native test samples and ArcAD labeled training paths/objects were excluded. The manifest records category, object, camera, source label and mask identity; 10 normal objects per category are held out for normal-only threshold calibration and excluded from thresholded evaluation. Full shared-split threshold-free AUROC/AP use the full eligible shared test.

## Coordinates and maps

ArcAD uses resize 448 and center crop 392, exposing source coordinates `[64,960)` in each axis. D2's 256 map covers the full 1024 frame. Both maps and original masks are aligned to the common source field `[64,960) × [64,960)` and resampled to 224×224; GT uses nearest-neighbor interpolation. The D2 map crop is `[16:240]` in each axis; ArcAD maps are linearly resized 256→224. `alignment_validation.json` records geometry and positive views with empty source masks. `maps/common_*.npz` contain the paired common-grid scores, GT and paths.

Maps use the saved model outputs after 5×5 Gaussian smoothing (sigma 4). View and object scores are the mean of the top 1% map values. Cross-model pixel scores are compared only on the common field. Camera IDs are parsed from original filenames (`C1`–`C5`) and included in paired tables. Map images normalize colors separately for display; color magnitude should not be compared across models.

## Metrics and limits

I and O AUROC/AP use sklearn; O groups five cameras by physical object. P AUROC/AP use exact flattened common-grid pixels. P-AUPRO uses 8-connected GT components, 200 linearly spaced thresholds, `score > threshold`, FPR cap 0.30, trapezoidal integration divided by the cap. Thresholds at normal-object FPR targets 1%, 5%, and 10% use the higher empirical quantile of the maximum view score (or pooled object score) among 10 calibration normal objects. With only 10 calibration objects, these three targets collapse to the same maximum-score threshold (calibration realized FPR 10%), so they do not form three distinct operating points. Pixel operating points use normal background score quantiles. Oracle maximum F1 is marked as oracle and is not a deployable threshold.

Image/object AUROC/AP bootstrap CIs resample paired physical objects, retaining all views (2,000 iterations, seed 20261031). Pixel/region metrics are reported without bootstrap CIs. The paired `laser_diode` set contains 41 positive views with empty source masks; these remain positive for image/object labels but are omitted from mask-based view/region analyses. One ArcAD-labeled object/category overlaps a native D2 test manifest; all such objects are removed from the paired shared set. Small calibration size and the failure-enriched category selection limit inference. GT morphology uses 8-connectivity. No foreground object masks are available; border distance is to the image crop. Luminance contrast is a local proxy affected by texture and illumination. Seen/unseen comparisons are associations only. No score fusion or training was performed.

## Checkpoints and execution

ArcAD and D2 checkpoint SHA-256 hashes before and after inference are in their `*_export_metadata.json`; both are unchanged. RTX 3090 was available to Python. ArcAD all five categories and D2 LED_indicator/thyristor were exported on GPU. D2 low-three numeric maps come from the earlier Task 01 shared-split inference (CPU); their view scores are independently checked against the Task 01 CSV in the analysis script.

Run from the repository root with the project Python environment:

```bash
python analysis_shared_failure_20261031_gpu01/work/prepare_analysis.py
python analysis_shared_failure_20261031_gpu01/work/run_exports.py
python analysis_shared_failure_20261031_gpu01/work/run_analysis.py
python analysis_shared_failure_20261031_gpu01/work/complete_analysis.py
python analysis_shared_failure_20261031_gpu01/work/finalize_outputs.py
```

The scripts expect the original Real-IAD Variety, ArcAD and D2 paths recorded at their top. Inference is checkpoint-only; it does not train or modify model weights.
