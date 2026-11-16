# Paired failure analysis: DiNomaly2 vs ArcAD

## Observations

The paired shared test contains five categories: lattice_block_plug, laser_diode, power_jack, LED_indicator, thyristor. Low-three categories were selected as failure-enriched and are not representative of all 160 categories. `thyristor` was preselected as the favorable comparison from archived native I-AUROC before new paired maps were generated. Category I-AUROC delta (D2 − ArcAD) has 1 D2 wins, 4 ArcAD wins, and 0 ties; mean -0.0015, median -0.0159, 10th percentile -0.0560, worst five lattice_block_plug=-0.0707, power_jack=-0.0340, laser_diode=-0.0159, thyristor=-0.0001, LED_indicator=0.1131.

Paired metric values and original native-split context are in `category_metrics.csv`; category deltas in `delta_by_category.csv`; clustered intervals in `bootstrap_ci.csv`. Fixed-threshold confusion tables and overlap are in `confusion_matrices.csv` and `failure_overlap.csv`. `joint_recall_oracle`/union coverage is an oracle diagnostic, not an ensemble result.

## Statistical associations

Bootstrap resamples physical objects, retaining all views and paired models (2,000 draws, seed 20261031). The low-three sample is deliberately enriched for prior failures. Therefore the category summaries are descriptive for this selected subset and must not be extrapolated to the 160-category benchmark. Thresholds use ten held-out normal calibration objects per category; because the higher empirical quantile selects the maximum for all three targets, nominal 1%, 5%, and 10% targets collapse to the same threshold (10% realized calibration FPR). These are not three distinct operating points. Thresholded evaluation excludes these objects. AUROC/AP do not use thresholds.

ArcAD seen/unseen comparisons in `seen_unseen_summary.csv` are associations only; seen labels may differ in intrinsic visibility, size, and prevalence. No causal claim is made.

## Hypotheses

Review `view_diagnostics.csv`, `region_diagnostics.csv`, `object_diagnostics.csv`, and `error_taxonomy_counts.csv` for size, contrast, localization, and multiview patterns. These are exploratory descriptions, not causal attribution. The edge/contrast proxy uses original RGB images in the common crop; object foreground masks were unavailable.

## Unverified explanations and limitations

The common map field is source [64:960) in each axis (87.5% of source width/height): ArcAD's 392/448 center crop and the corresponding D2 center crop are resampled to 224x224; source GT masks use nearest-neighbor resizing. See `alignment_validation.json`; it records 41 positive `laser_diode` views whose source masks are empty (these retain positive image/object labels and are excluded from mask-based view/region summaries). One ArcAD-labeled object/category intersects the D2 native test manifests; those objects were removed before evaluation, and no ArcAD-labeled object remains in the paired shared test. Map colors in figures are for display only. Pixel AP/AUROC/AUPRO follow common-grid definitions saved with the config. D2 low-three maps are cached Task 01 CPU inference, while ArcAD maps and added D2 categories were inferred on RTX 3090; strict checkpoint hashes and map provenance are recorded in metadata.

No training or tuned score fusion was performed. Calibration contains only ten normal objects/category, so operational FPR estimates have low resolution. Object cluster intervals are reported for image/object AUROC/AP; pixel/region metrics have no bootstrap intervals. See `README.md` for reproduction and exact transforms.


## Aggregate summaries and exploratory miss associations

`macro_micro_summary.csv` reports category macro means (equal category weights) and pooled metrics on the five selected categories. The pooled result is descriptive because raw score scales may differ by category. It is not a 160-category benchmark estimate. `mechanism_associations.csv` fits separate category/camera-adjusted logistic models for a miss among positive views with nonempty masks, using log area, GT/background contrast proxy, region count, and border distance; the intervals resample category/object clusters. Coefficients are exploratory associations and are not causal. Camera IDs are parsed from source filenames and retained in joined, view, region, and morphology tables. Confusion matrices include raw counts and row-normalized class rates.


## Aggregate summaries and exploratory miss associations

`macro_micro_summary.csv` reports category macro means (equal category weights) and pooled metrics on the five selected categories. The pooled result is descriptive because raw score scales may differ by category. It is not a 160-category benchmark estimate. `mechanism_associations.csv` fits separate category/camera-adjusted logistic models for a miss among positive views with nonempty masks, using log area, GT/background contrast proxy, region count, and border distance; the intervals resample category/object clusters. Coefficients are exploratory associations and are not causal. `paired_region_overlap.csv` matches each GT component across both models and records fixed pixel-threshold hits and coverage. Camera IDs are parsed from source filenames and retained in joined, view, region, and morphology tables. Confusion matrices include raw counts and row-normalized class rates.
