# Task 03 — Verification, calibration and fusion follow-up

## Verified observations

The original native-split screen has 160 categories. From full-precision source files, D2 has **83/160** categories with I-AUROC `<0.90` (`<=0.90`: 83; none exactly 0.90), and ArcAD has **100/160** (`<=0.90`: 100; none exactly 0.90). The 2×2 screen is 78 weak on both, 5 D2-only, 22 ArcAD-only, and 55 neither. Macro means are D2 0.885916 and ArcAD 0.859856; these match the reported 0.8859/0.8599. Native test lists and supervision differ, so this overlap is descriptive screening, not paired inference. See `category_160_audit.csv`, `weak_category_overlap.csv`, and `native_split_counts_160.csv`.

On the 41 laser_diode views Task 02 flagged as empty on the common field, source-mask recomputation found **39 nonzero masks entirely outside the shared crop** and **2 all-zero source masks** (OilStain/S0005 C1/C4). No mask vanished only at 224×224. All 41 sheets show the image and five sibling views with source/crop masks. These are unresolved visibility/annotation cases requiring manual review; no labels or masks were changed. `label_sensitivity.csv` reports common-grid pixel metrics before/after excluding the 41 ambiguous views from mask-dependent calculations, plus an explicitly exploratory view-label sensitivity. Object labels remain unchanged.

The coordinate audit independently matched source GT masks to both models' common-grid GT in 15 seeded sample views, and confirms 224×224 maps, finite nonconstant scores, and unique shared paths. The five-camera groups and D2/ArcAD view labels/mask paths match. One ArcAD-labeled object/category appears in each diagnostic category's D2 native test manifest; all those objects are excluded from the shared paired set. There is no ArcAD-labeled or D2-training object in the paired shared evaluation partition.

## Evidence for/against mechanisms

Paired per-view and per-object scores, raw normal/defect score distributions, predeclared top-score-percentage sensitivity, per-camera outcomes, GT area/component/width, RGB local luminance contrast, model-derived anomaly-score contrast, GT/background quantiles, Q, top-k precision/recall, and seen/unseen type counts are saved in their corresponding CSVs. Camera and defect-type tables retain physical object IDs. A separate exploratory category/camera-adjusted miss model is not fitted here; this follow-up uses stratified summaries to avoid over-interpreting sparse category/type effects.

Paired figures show the same source view, common-crop GT, both maps and actual top-1% map pixels. Raw top-1% image scores, map ranges, normal-CDF ranks, thresholds, Q and top-k measurements are printed separately from display-normalized heatmaps. `manual_review_sheet.md` separates visible observations from plausible but unverified explanations. No causal interpretation is assigned to contrast, view, or ArcAD seen/unseen association.

## Unverified hypotheses

The 39 crop-external masks could reflect legitimate defect location outside the common field, crop mismatch, or annotation placement; the 2 all-zero masks could be weak/missing annotation or image-label disagreement. Contact sheets alone do not resolve which explanation is correct. Lighting, texture, specularity, occlusion and supervision effects remain hypotheses requiring expert review or controlled data.

## Practical ensemble result

No independent anomaly-validation set was available, so no weighted fusion was tuned. The only ensemble baseline is a fixed 50/50 mean of model-wise global empirical normal-CDF ranks, with a calibrated OR rule also reported. This uses 50 normal calibration objects (10/category) and excludes them from held-out evaluation; calibration objects are from the same native test manifests, so this is a **diagnostic object-disjoint transfer experiment**, not deployment validation.

On held-out objects, category-macro AUROC was D2 0.7814, ArcAD 0.8003, fixed mean 0.8238; pooled AUROC was D2 0.7805, ArcAD 0.7893, fixed mean 0.7782. The view-level macro AUROCs were D2 0.6900, ArcAD 0.6993, fixed mean 0.7065. See `fusion_comparison.csv` for AP, per-category results and the corresponding prevalence baselines; the positive prevalence is high (91.0% of views, 92.9% of objects), so AP is correspondingly high.

At the pooled 10% operating point, held-out object results are in `operating_point_metrics.csv`; calibration/evaluation FP counts and exact binomial intervals are in `calibration_reliability_counts.csv`. The OR detector's false-positive union cost is reported in both operating-point and failure-overlap tables. Oracle union coverage is separately labeled and is not an implementable ensemble result. Pixel-level fusion and a new region-mask OR recomputation were skipped. `region_overlap_vs_fpr.csv` retains the aligned Task 02 per-model component IoU/coverage and oracle region detection contingency at IoU 0.3 for exploratory 1%/5% pixel-FPR thresholds. The implementable score-level OR false-positive union cost appears in `operating_point_metrics.csv` and `failure_overlap_vs_fpr.csv`; it is not a pixel-region OR estimate.

## Limitations

The 160-category results use different native test/supervision regimes. Five paired categories are selected diagnostics, not a representative benchmark sample; no new category inference was launched. Nominal 1/5/10% thresholds from Task 02's ten normals all collapse to one max-score threshold and were rejected as distinct operating points. The pooled 50-normal calibration improves resolution but is still native-test-derived and category-mixed. Fixed fusion gains may reflect this score-CDF transfer and the selected categories; no 160-category ensemble claim follows.

Bootstrap intervals resample physical objects and retain all five views (2,000 draws; see `bootstrap_intervals.csv` for valid and excluded single-class draws). Pixel/region bootstrap intervals, ROC/PR bootstrap bands, and a new region-mask OR pixel-area cost calculation were skipped; the existing component table and its exploratory 10-normal calibration are retained. Task 02 numeric maps were reused; no new model inference or checkpoint changes were made. Task 01 low-three D2 maps were originally CPU-generated; existing ArcAD and added D2 maps were GPU-generated as recorded in provenance.

## Recommended next experiment

Have an annotator review the 41 laser_diode contact sheets against the original 1024 images and all five cameras, without editing the source labels in place. For a deployable fusion test, reserve a substantially larger, disjoint normal calibration set and an independent anomaly-validation set with object-level separation; freeze the CDF transform and any weight there, then evaluate on untouched objects. Before broadening to all 160 categories, evaluate the four frozen extension categories in `category_selection_plan.md` using matched objects, identical scoring definitions and a prospective holdout.
