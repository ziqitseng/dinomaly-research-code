# Issues and interpretation limits

- **Inference-only intervention:** ArcAD full-frame inputs are out of distribution relative to ArcAD training preprocessing. This does not estimate retrained full-frame ArcAD.
- **Border visibility:** 39 positive laser_diode views have nonzero masks completely outside the original crop; two source-all-zero masks are also present. All 39 crop-external examples have five-camera contact sheets. Visibility remains `requires_manual_review`; no cause or false-negative label is inferred.
- **Source-coordinate full field:** original ArcAD has no observed border prediction. It is NaN/abstention outside the crop and is excluded from full-source pixel comparison.
- **Thresholding:** no independent calibration was used and no binary decisions are claimed. All detection metrics are threshold free.
- **Inference scope:** five preselected categories from the exact shared Task 02 manifest; this is not a 160-category generalization claim.
- **Uncertainty:** object-cluster bootstrap has 2,000 planned replicates per comparison, retaining sibling views; single-class replicates are excluded and counted. Results are exploratory across multiple categories.
- **Skipped:** optional D2 center-crop inference was not run; it is secondary to the requested ArcAD ablation.
- **Reused scores:** D2 numeric maps are reused from Task 01 (three primary categories) and Task 02 (LED_indicator, thyristor); provenance and source hashes are recorded.
