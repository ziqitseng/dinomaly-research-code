# Task 2: paired failure analysis

## Observations

All 5,550 paired views across 1,110 five-view objects use identical manifest paths. All map pairs and transformed masks passed common-crop geometry validation; see `plots/alignment_audit_all_categories.png`. ArcAD uses its current protoinit checkpoint; no file from an ArcAD `legacy/` folder was used. No model training was run. For 41 laser_diode and 5 aircraft_model_head positive views, the common crop contains no annotated GT pixels; those views remain positive in view/object metrics but are excluded from pixel and region localization metrics.

The low-three categories were selected for prior weak DiNomaly2 performance and are failure-enriched. `LED_indicator` and the preselected `aircraft_model_head` control are reported separately; neither makes this a random sample of the 160-category benchmark. I-AUROC/AP, O-AUROC/AP, common-crop P-AUROC/AP and P-AUPRO, fixed-threshold confusion tables, category deltas, and paired overlap are in `tables/`. Pixel histogram metrics use 2,048 bins; P-AUPRO uses a 200-step threshold grid and FPR cap 0.30.

At view level, D2’s I-AUROC exceeds ArcAD in 3 of 5 categories; mean Δ (D2−ArcAD) is 0.0082, median Δ is 0.0132, and the worst category delta is -0.0251. These are descriptive category comparisons, not an estimate over all categories.

## Paired detection and uncertainty

At each nominal 1%, 5%, and 10% point, view thresholds use the maximum of five view scores for each model-specific normal training object; object thresholds use the pooled five-view top-1% score. D2 calibration maps are aligned to the common crop. ArcAD’s reference contains 7–8 objects/category; D2’s contains 25. These are in-sample because the normal examples were seen during training. Realized test FPR, raw confusion counts, and row-normalized rates are reported; thresholds are exploratory and not deployment guarantees.

At the nominal 5% normal-object point, realized test normal-object FPR ranges from 76–100% for D2 and 84–100% for ArcAD on the three failure-enriched categories; it is 80%/100% on LED_indicator and 24%/16% on aircraft_model_head (D2/ArcAD). This large gap shows that the in-sample reference thresholds transfer poorly to held-out normal objects in most tested categories. The cause is not isolated by this analysis.

The 2,000-replicate bootstrap resamples paired physical objects within each category and retains all views together. The reported per-category intervals target within-category object-sampling uncertainty. It does not support category-generalization claims. See `tables/bootstrap_ci.csv` and raw replicates.

## Spatial behavior and morphology

Maps are numeric smoothed anomaly maps, aligned onto the same central 896×896 source field and evaluated with transformed source masks. Per-view `C` (GT mean minus background mean), `Z`, pixel AUROC `Q`, GT-rank percentiles, top-k precision/recall, top-1%-rank region matching, region geometry, size response, and multiview consistency are tabulated. P-AUROC/AP use 2,048-bin histograms and P-AUPRO uses a 200-threshold grid with FPR cap 0.30. Pixel metrics are point estimates; cluster-bootstrap intervals cover I/O AUROC/AP. Representative figures use the same source view and state that colors are normalized per model/view; compare spatial patterns, not absolute intensity.

ArcAD seen/unseen rows compare test defect-type names with the current ArcAD labeled supervision manifest. These are observational associations: defects and objects were not randomized, and type frequencies/sample sizes differ. No causal mechanism is established.

## Hypotheses and unverified explanations

Small or spatially dispersed defects, weak local contrast, view-dependent visibility, and differences in learned representations are plausible hypotheses for individual misses. The current comparisons do not isolate these factors experimentally. Mask pixels outside the ArcAD crop, illumination, texture, and defect-type supervision may also affect measurements; none is established as a cause by these results.

## Complementarity and limits

D2-only and ArcAD-only detections at the exploratory 5% reference point are paired on the same views. The union recall is an oracle coverage bound, not a deployable ensemble. There is no independent validation set in this workflow to select a fusion rule, calibrate an ensemble, or claim matched-FPR ensemble gains. No tuned ensemble performance is reported.

The original full-run category outputs are not substituted for shared-split results. Any native-run references, if included in metadata, are kept separate. Missing independent calibration and the small ArcAD normal reference set limit fixed-threshold interpretation.

Task 01 shared-split values are retained separately in `tables/task01_shared_protocol_reference.csv`. ArcAD view/object ranking metrics agree because both use its common-crop map frame. Task 01 DiNomaly2 view/object scores use the native full 280 crop, while this task recomputes D2 scores on the central 87.5% common crop; these are protocol references and should not be treated as directly comparable deltas.

## Reproduction

Run `dinomaly2/bin/python run_analysis.py` from the analysis folder, then `dinomaly2/bin/python work/finalize_deliverables.py`. The run uses `config.yaml`, fixed seed 20261103, and object-cluster bootstrap count 2,000. Input checkpoint hashes and strict-load status are recorded in the ArcAD and D2 integrity JSON files.
