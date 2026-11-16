# Task 04A — Controlled crop and field-of-view ablation

## (a) Did full-frame ArcAD help overall?

No, on average. On the shared central 224×224 field, equal-category view I-AUROC changed from **0.6993** with ArcAD-original to **0.6898** with ArcAD-full-frame (Δ **−0.0094**). Object O-AUROC changed from **0.8095** to **0.7974** (Δ **−0.0121**). Full-frame view I-AUROC was lower in four categories and higher in power_jack.

| Category | Center I-AUROC: original → full (Δ) | Center O-AUROC: original → full (Δ) | Center P-AUPRO: original → full |
|---|---:|---:|---:|
| lattice_block_plug | 0.574 → 0.544 (−0.030) | 0.866 → 0.833 (−0.033) | 0.671 → 0.623 |
| laser_diode | 0.626 → 0.596 (−0.029) | 0.691 → 0.669 (−0.023) | 0.718 → 0.701 |
| power_jack | 0.645 → 0.679 (+0.033) | 0.789 → 0.807 (+0.018) | 0.699 → 0.641 |
| LED_indicator | 0.652 → 0.631 (−0.021) | 0.702 → 0.678 (−0.024) | 0.526 → 0.518 |
| thyristor | 1.000 → 0.999 (−0.001) | 1.000 → 1.000 (+0.000) | 0.882 → 0.865 |

Paired object-cluster bootstrap 95% intervals for view I-AUROC excluded zero in the three lower categories and power_jack; thyristor's interval included zero. These intervals are unadjusted for multiple comparisons and the five categories were preselected. See `paired_bootstrap_ci.csv` for all I/O AUROC/AP effects.

The ArcAD-original control reproduced Task 02 per-view top-1% scores on the four cached categories to floating-point precision: maximum absolute difference **8.3×10⁻¹⁷**, against a declared 5×10⁻⁵ tolerance. Thyristor had no Task 02 cache, so its original-preprocess branch was newly inferred.

## (b) Did it help the border cases?

Full-frame ArcAD maps now cover all **39** nonempty masks wholly outside the original crop. That restores spatial observability; it does not itself prove detection. The median GT-mean score percentile versus background was **0.823** for full-frame ArcAD and **0.873** for D2 on those 39 views. ArcAD-original has no prediction on these border pixels. No thresholds were applied, and all 39 remain `requires_manual_review`.

## (c) Did it degrade central or small defects?

Central-field P-AUPRO decreased in all five categories; its equal-category mean changed from **0.6990** to **0.6696**. Central P-AUROC was nearly unchanged overall (**0.8755** to **0.8748**), while P-AP changed from **0.2118** to **0.1993**.

For fully in-crop positive views with source masks below 1% of source-image area, mean top-1%-map GT recall was **0.435** for ArcAD-original and **0.399** for full-frame ArcAD (D2: **0.368**). The geometric area bins are in `size_strata_metrics.csv`; these are descriptive size-stratified estimates.

## (d) Did object-level aggregation alter the conclusion?

Object pooling made the macro AUROC drop slightly larger than view pooling (Δ O-AUROC **−0.0121** vs Δ I-AUROC **−0.0094**). Each object score pools the top 1% over all five camera maps; it is not the average of five view scores. Bootstrap resampling used physical objects as clusters and retained all sibling views.

## (e) Is retraining warranted?

This experiment changes inference geometry on a frozen checkpoint and cannot establish the benefit of full-frame training. If full-field coverage is a product requirement, use a separate matched-split, matched-budget, multi-seed retraining ablation to test that question.

## Protocol and limits

The evaluation contains **5,550 views and 1,110 physical objects** across five categories. Positive prevalence is high (view **83.9–88.8%**, object **88.5–88.8%**), so PR baselines equal those prevalence values. ArcAD-original uses resize 448→center-crop 392; full-frame ArcAD resizes directly to 392; D2 retains full-frame 280 input. Only full-frame ArcAD and D2 are compared on the full-source grid. ArcAD-original outside-crop values are unobserved. Pixel AUROC/AP use 2,048 score bins; P-AUPRO uses 200 thresholds and FPR cap 0.30. Pixel metrics are point estimates. No independent calibration set exists, so no deployable thresholds or thresholded detection claims are reported.

Task 02 used `aircraft_model_head` where this brief listed `thyristor`. This run used exact Task 02 membership for the four shared categories and an exact native ArcAD/D2 test-path intersection for thyristor, excluding all ArcAD labeled objects. No substitution was made, and no ArcAD legacy artifacts were used.
