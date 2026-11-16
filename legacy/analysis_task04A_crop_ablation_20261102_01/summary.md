# Task 04A — ArcAD cropping ablation results

## Summary

This is an inference-time intervention on the frozen ArcAD checkpoint. ArcAD full-frame inference changes the field of view to full 1024 source coverage resized to 392×392; the original branch remains resize448→center crop392. No retraining occurred. The five categories were fixed from the Task 02 shared manifest before inference.

The primary comparison uses 5550 shared views across 1110 category-object groups in five categories. All groups contain five cameras; the manifest validation found zero ArcAD training-object leakage and zero duplicate test paths. ArcAD original regression max map difference per category is recorded in `arcad_original_regression.csv`.

### (a) Did full-frame ArcAD help overall?

Center-field macro results (macro across five categories):
- **arcad_original:** I_AUROC 0.7048, I_AP 0.9335, O_AUROC 0.8079, O_AP 0.9661, P_AUROC 0.8758, P_AP 0.1812, P_AUPRO 0.7055
- **arcad_full_frame:** I_AUROC 0.6952, I_AP 0.9305, O_AUROC 0.7925, O_AP 0.9644, P_AUROC 0.8751, P_AP 0.1738, P_AUPRO 0.7015
- **dinomaly2_original:** I_AUROC 0.7033, I_AP 0.9273, O_AUROC 0.7944, O_AP 0.9646, P_AUROC 0.8754, P_AP 0.1498, P_AUPRO 0.6618

For full-source pixel localization (only the two branches that observe the border), macro results were: arcad_full_frame: full_field_P_AUROC 0.8915, full_field_P_AP 0.1660, full_field_P_AUPRO 0.7336; dinomaly2_original: full_field_P_AUROC 0.8932, full_field_P_AP 0.1461, full_field_P_AUPRO 0.7184. Original ArcAD full-source results are unavailable because its outside-crop map is NaN, not imputed.

Full-frame minus original ArcAD AUROC by category is listed in `category_delta_summary.csv`; object and view deltas are paired on identical samples. See `paired_bootstrap_ci.csv` for 2,000-object bootstrap 95% intervals, valid draws and excluded single-class draws. A CI spanning zero is inconclusive.

### (b) Did it help the crop-external border cases?

The audit contains 39 positive laser_diode views with source masks wholly outside the original center crop. Their mean per-view top-1% scores were arcad_original 0.179669, arcad_full_frame 0.174137, dinomaly2_original 0.0519841. The 39 views occur across 24 objects. Compared descriptively with manifest-normal views, score-ranking AUROCs were arcad_original: view AUROC 0.638, object AUROC 0.587; arcad_full_frame: view AUROC 0.575, object AUROC 0.528; dinomaly2_original: view AUROC 0.685, object AUROC 0.720. Thus, the full-frame branch did not improve this test-derived border-versus-normal ranking. This is not a calibrated binary decision; original ArcAD has no map values in the GT region outside its crop, and its GT-region score/rank is explicitly marked not observed. Scores alone cannot establish visibility or a false negative. Manual review is required.

### (c) Did central or small defects degrade?

For laser_diode positive views whose masks are entirely inside the crop (n=745), mean per-view scores were arcad_original 0.183596, arcad_full_frame 0.179802, dinomaly2_original 0.0510172. `visibility_size_strata_metrics.csv` gives category-level threshold-free results for center-only, partly outside, wholly outside, and the pooled lowest source-mask-area quartile. Small-defect groups are descriptive and vary by category only through each category’s sample. No thresholded decisions are inferred.

### (d) Did object aggregation alter the conclusion?

Compare `center_I_AUROC` and `center_O_AUROC` in `category_metrics.csv` and the paired object bootstrap deltas. Object scores pool the top 1% of smoothed pixels across all five views; image scores use top 1% per view. These different aggregation levels can rank models differently.

### (e) Is retraining warranted?

This experiment alone cannot establish that retraining with full-frame inputs will improve ArcAD. The aggregate center-field results do not show an overall gain from crop removal, though category-level changes vary. A separate controlled retraining ablation is warranted only if the potential border-coverage benefit is important enough to test against matched training budgets/seeds/splits. The current result estimates only inference-time sensitivity of the existing checkpoint.

## Interpretation and limitations

- ArcAD-full-frame receives a 392×392 full-frame image that differs from its 448-resize/392-center-crop training geometry; this is an out-of-distribution inference intervention.
- ArcAD-original has no prediction outside source `[64:960)×[64:960)`; its full-source aligned map uses NaN outside observed pixels. Full-source localization compares only full-frame ArcAD and D2.
- Masks outside the center field remain in image-level metrics and full-source pixel metrics; center-field positive-region metrics use only nonempty center GT for region analyses.
- Pixel AP depends on positive-pixel prevalence. Maps use bilinear interpolation; masks use nearest-neighbor interpolation.
- No threshold calibration or binary classification decisions were introduced. Detection and localization are reported separately.
- The 39 border cases and two all-zero laser_diode source masks remain unresolved annotation/visibility cases; original labels were preserved.
- ArcAD original numerically regressed against Task 02 maps before color normalization; tolerances and differences are in `arcad_original_regression.csv`.
- D2 maps were reused, not rerun; their exact file/path provenance is in `protocol_audit.json`.
- Runtime is descriptive for this machine and batch size only.

## Recommended next experiment

If the full-frame crop-coverage trade-off remains scientifically important, train a separate full-frame ArcAD variant and compare it against the unchanged original using matched training samples, budgets, seeds, and an untouched object-level evaluation set. Keep the frozen original ArcAD and D2 branches as the primary baseline for the separate Task 04B fusion study.

### Central and small-defect stratum result

Center-only positive views plus all normal views: arcad_original AUROC 0.7028, arcad_full_frame AUROC 0.6901, dinomaly2_original AUROC 0.6921. These are descriptive category-macro rankings with all manifest-normal views included; the lowest-area quartile uses pooled positive-view mask area (cut points in `visibility_size_strata_metrics.csv`).

Lowest pooled source-mask-area quartile plus all normal views: arcad_original AUROC 0.6341, arcad_full_frame AUROC 0.6290, dinomaly2_original AUROC 0.6628. These are descriptive category-macro rankings with all manifest-normal views included; the lowest-area quartile uses pooled positive-view mask area (cut points in `visibility_size_strata_metrics.csv`).


### Empty-mask effect on pixel-negative pools

Positive image-labelled views with no GT pixels in the evaluated field remain in the primary pixel metrics as all-negative masks. Center-grid empty-positive counts by category: laser_diode 41, lattice_block_plug 0, power_jack 0, LED_indicator 0, thyristor 0. Full-source empty-positive counts: laser_diode 2, lattice_block_plug 0, power_jack 0, LED_indicator 0, thyristor 0. Their exact negative-pixel contributions are in `pixel_mask_visibility_sensitivity.csv`; image-level labels and metrics are unchanged.
For laser_diode, the 919 positive views divide into inside_original_crop 745, partly_outside_original_crop 133, wholly_outside_original_crop 39, source_all_zero 2. The `visibility_status_counts.csv` table also lists normal-view counts and all other categories.
