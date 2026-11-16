# Task 02 — paired failure-mechanism analysis

This analysis uses the current ArcAD prototype-initialized checkpoint at `ArcAD/out/real_iad_variety out/arcad_realiad_variety_protoinit/model.pth` and the current DiNomaly2 Variety checkpoint. The shared manifest was rebuilt from current ArcAD `sup` and DiNomaly2 native manifests; every ArcAD labeled training object is excluded. `LED_indicator` is included as requested. `aircraft_model_head` was selected as the strong control from current full-run summaries before examining its paired shared-split predictions.

**No ArcAD files or results from any `legacy/` folder are used.** Full-run summaries were used only for category selection; paired metrics come from new numeric inference on the common held-out records. No ArcAD `results.csv` is used as a paired prediction source.

## Pixel coordinate alignment

ArcAD resizes each 1024×1024 input to 448×448 and center-crops 392×392. Its 256×256 map therefore represents the central 87.5% of the source image. DiNomaly2 maps the full image to 256×256. To compare pixels, DiNomaly2 maps are bilinearly sampled onto ArcAD's physical crop frame using pixel-center coordinates `i = 0.875*(j+0.5)+15.5`; both use the same source mask transformed with nearest-neighbor resizing into that common crop. The original image, common GT, and aligned maps are visually audited in `plots/`. Native full-FOV pixel metrics remain separate from common-crop pixel diagnostics.

## Thresholds and uncertainty

Fixed thresholds use model-specific normal-only training-reference scores, without test labels. View thresholds come from each normal calibration object’s maximum of five view scores; object thresholds come from its pooled top-1% score over five maps. D2 calibration maps are aligned to the same common crop as test maps. ArcAD has only 7–8 normal training objects per category and DiNomaly2 has 25; these examples were seen during model training. Thus the resulting thresholds are **in-sample exploratory reference points**, not independent calibration or deployment thresholds. No test-label threshold tuning or ensemble tuning is performed. Bootstrap units are physical objects, retaining all five views and masks together.

The low-three categories were selected because they were DiNomaly2's weakest I-AUROC categories. Their error rates are failure-enriched and are not estimates for all 160 categories. `aircraft_model_head` is a selected strong control, not a random category sample.

## Scope and interpretation

Quantitative error overlap, map/GT geometry, mask morphology, within-view ranking, top-k localization, region recall, view consistency, and ArcAD seen/unseen defect-type associations are reported separately. Associations do not establish that supervision caused a performance difference. Any score union is labeled oracle coverage; no max-map ensemble is treated as a deployable upper bound. Pixel comparisons are restricted to the common physical crop.

## Outputs

- `analysis_report.md`: observations, associations, interpretation limits, and reproduction details.
- `joined_predictions.csv`: one row per model and matched view, with view/object scores and provenance.
- `tables/view_predictions.csv`, `tables/object_diagnostics.csv`, `tables/region_diagnostics.csv`, `tables/seen_unseen_summary.csv`: per-view spatial diagnostics, grouped-object outcomes, top-1%-rank region matching, and defect-type associations.
- `tables/category_metrics.csv`, `tables/fixed_threshold_metrics.csv`, `tables/aggregate_metrics.csv`, `tables/delta_by_category.csv`, `tables/failure_overlap.csv`: per-category and pooled results, fixed operating points, and paired disagreement.
- `tables/task01_shared_protocol_reference.csv`: low-three Task 01 shared-split scores beside Task 02 common-crop scores, with protocol compatibility marked.
- `tables/bootstrap_ci.csv` plus `tables/bootstrap_replicates.csv`: paired object-cluster resamples (2,000 per category, seed 20261103).
- `selected_examples.csv` and `plots/paired_examples/`: evidence-selected matched views for shared misses, one-model successes, normal false alarms, off-target ranking, view dependence, and the good control.
- `plots/`: common-frame audit and paired score, delta, disagreement, size, top-k, and contrast figures.

P-AUROC and pixel AP use a 2,048-bin score histogram approximation; P-AUPRO uses a 200-threshold grid and FPR cap 0.30. Per-pixel values are point estimates; the reported cluster-bootstrap intervals are for image- and object-level AUROC/AP. Region matching uses the top 1% of common-crop pixels as a rank-based diagnostic, not a calibrated pixel threshold.

To reproduce the paired summaries from cached maps, run `bash reproduce.sh`. Map and normal-reference inference scripts are in `work/`; checkpoint hashes, strict loading, and map integrity are recorded in the `*_integrity.json` files.
