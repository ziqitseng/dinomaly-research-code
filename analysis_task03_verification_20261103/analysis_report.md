# Task 03 verification, calibration, and fusion follow-up

## Verified observations

The current ArcAD protoinit summary contains 160 categories (excluding its `Mean` row): macro I-AUROC 0.8633, 94 categories <0.90 and 94 <=0.90. D2 has macro I-AUROC 0.8859, 83 categories <0.90 and 83 <=0.90. Screening overlap is 75 weak-both, 8 D2-only, 19 ArcAD-only, and 58 neither. Native test lists differ, so this is a screening comparison, not paired inference. The older ArcAD summary cited in Task 03 was not opened because the current prototype-initialized run supersedes it under the standing instruction.

The current Task 02 held-out set has 25 normal physical objects/category (not 10), grouped by five views; their exact object keys are listed in `calibration_split_manifest.json`. The actual categories are `lattice_block_plug`, `laser_diode`, `power_jack`, `LED_indicator`, and `aircraft_model_head`; the Task 03 brief’s prior-run list substitutes `thyristor`. This run uses the current saved five-category maps and performs no new model inference.

Independent 224-grid verification passed for 5550 maps. D2 maps use native crop [16:240]; ArcAD maps are bilinearly resized 256→224; GT masks use raw source crop [64:960] and nearest resize. All path orders, shapes, map ranges, and finite values passed. See `alignment_validation_224.json` and `plots/spotcheck_224.png`.

The 41 `laser_diode` positive views with empty Task 02 common GT resolve to 39 nonzero masks entirely outside [64:960] and 2 genuinely all-zero source masks (both Task 01 OilStain/S0005). None vanished only at 224 downsampling. Both native test manifests agree on all 41 view labels. The object OilStain/S0005 has two labeled anomalous and three normal views; object labels remain positive. All 41 cases require manual visual review.

## Evidence for/against mechanisms

Threshold-free view/object ranking, pixel localization, component metrics, top-k sensitivity, GT size, RGB luminance/gradient contrast, background texture, per-camera scores, and seen/unseen defect-type associations are recorded separately. Defect prevalence is high: positive view prevalence ranges 83.9–88.7%, and positive object prevalence ranges 88.5–88.8%; therefore I-AP/O-AP baselines are correspondingly high and should be read alongside AUROC and the prevalence columns in `threshold_free_metrics.csv`. Model-derived score contrast C/Z/Q is not used as an independent cause of failure. The category split and size-bin boundaries were fixed before the diagnostic calculations. See `mechanism_by_stratum.csv` and `object_camera_consistency.csv`.

## Unverified hypotheses

Small/dispersed defects, low RGB contrast, view-dependent visibility, and representation differences remain hypotheses. The analysis is observational and cannot isolate causal effects. The empty-mask contact sheets do not decide whether a physical defect is visible; manual annotation is required.

## Practical ensemble result

A fixed 50/50 score-average was evaluated after global per-model z-standardization using normal training-reference score distributions. These normal examples were seen during model training, so the baseline is exploratory. Macro view AUROC was 0.7037 for D2, 0.6955 for ArcAD, and 0.7168 for Mean50; macro object AUROC was 0.7955, 0.8095, and 0.8218, respectively. Per-category results show both gains and regressions, and paired physical-object bootstrap intervals are in `fusion_comparison.csv` and `bootstrap_intervals.csv`. No anomaly-labeled validation objects or independent normal calibration population were available; no category-specific weight or threshold was tuned. This is not a deployable calibration result.

Test-normal FPR sweeps and region pixel-FPR sweeps are descriptive only. The test set has 25 normal objects/category, so object FPR resolution is 4%; the curves do not claim distinct 1%, 5%, or 10% calibrated operating points. Thresholded OR and region union costs are tabulated with actual normal-object/pixel false-positive counts. Independent calibration is needed before deploying the OR rule.

## Limitations

ArcAD uses the current protoinit checkpoint; the non-initialized ArcAD result folder is excluded. ArcAD and D2 full-run test manifests contain 192,165 and 178,995 views and differ in split composition. A single access_card object has unparseable camera IDs in both native manifests. Task 02 normal references are in-sample. Pixel AUC/AP use 2,048-bin histograms; P-AUPRO uses 200 thresholds with FPR cap 0.30. Pixel metrics are point estimates; bootstrap intervals cover view/object I/O AUROC/AP.

No 160-category extension inference was launched. Category selection remains a score-ranked plan, and claims do not generalize from the five diagnostic categories.

## Recommended next experiment

Create a larger object-disjoint normal calibration set and anomaly-labeled validation set, ensuring all five cameras of an object remain together and excluding all ArcAD supervised training objects. Resolve the 41 laser diode mask visibility cases by expert review. Then evaluate a pre-registered calibration and fusion rule on untouched shared test objects, and expand to the preselected four extension categories only if compute and storage are approved.

Reproduce with `bash reproduce.sh`; no training or model inference is performed by the analysis scripts. Checkpoint SHA256 values and source split details are in `protocol_audit.json`.
