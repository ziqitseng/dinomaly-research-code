# Codex Task 02 — Quantitative failure-mechanism analysis and verification for DiNomaly2 vs ArcAD on Real-IAD Variety

**Dependency:** Prefer Task 01's exact shared, held-out test manifest and newly saved DiNomaly2 numeric outputs, plus cached/recomputed ArcAD numeric outputs on **the same** image paths. **Do not train models for this task.** Do not claim a causal reason for failure without evidence. Treat the existing qualitative low-3 figures as illustrations, not the sampling frame for statistical estimates.

## 0. Objective, inputs, and guardrails

Goal: determine **what makes individual defects hard**, whether the two models **miss the same physical objects/views/regions**, and whether different strengths offer a realistic ensemble opportunity.

Required inputs: exact shared manifest; per-view labels/masks; per-object group and five camera IDs; numeric smoothed anomaly maps from both models (not per-image normalized visualization), image scores, object scores; training supervision manifests to identify ArcAD seen defect types; provenance/metric configs.

Starting categories: `lattice_block_plug`, `laser_diode`, `power_jack`; add `LED_indicator` and ≥1 **preselected** good-performing comparison category when manifests and paired predictions are available. Label low-3 as a **failure-enriched, selection-biased subset**. Do not extrapolate its rates to all 160 categories. All analyses should be extensible to 160 categories.

**Before any analysis:** check identical image-path intersection, five-view object membership, labels, mask coordinates, model map shape, mask transformation, and no ArcAD labeled training object in tests. ArcAD maps and masks may have been evaluated after 448→392 center crop, DiNomaly2 after 280 crop; confirm geometric correspondence in original 1024 coordinate space. For pixel-level *cross-model* overlap, align outputs/GT to a common valid physical field of view or explicitly analyze only the jointly visible area. A 256×256 tensor from each model alone does **not** prove pixel alignment. If original transform coordinates are irrecoverable, block cross-model pixel-overlap claims and report it.

## 1. Fixed reproducible evaluation conventions

1. Use distinct levels: **view/image** (`I-*`); **object** (`O-*`, grouped five views); **pixel/region** (`P-*`/PRO). Do not infer object-level false negatives from an image-level AUROC alone.
2. Fix a threshold selection procedure *without test labels*: either (a) a held-out normal-only calibration set to fix a false-positive target, or (b) separately held-out validation with labeled examples when allowed by the learning setting. Disclose ArcAD supervision used in calibration. Present several operating points (e.g. normal-object FPR 1%, 5%, 10%), empirical realized test FPR, and threshold-independent AUROC/AP. **Do not optimize thresholds on test data** for operational precision/recall/F1. Original benchmark maximal F1 may be reproduced only as explicitly marked **oracle_max_F1**.
3. For image/object detection with strong class imbalance, report **AUROC, average precision (AP), positive prevalence, and precision/recall/F1 at the fixed threshold**. Include confusion matrices with raw counts and row-normalized rates at that threshold. Distinguish `per_view` and `per_object` confusion matrices.
4. Set and record P-AUPRO FPR cap (existing ArcAD script says `0.3`), threshold grid, normalization/integration details. Calculate AP from the full precision–recall curve, not trapezoidal PR-AUC unless clearly renamed.
5. Report category macro mean as category-weighted equally; separate overall micro/pooled aggregations. Do not calculate global AUROC by averaging individual AUCs and call it global AUROC.
6. Treat five camera views of one physical object as **clustered observations**, not five independent subjects for uncertainty estimation.

## 2. Required baseline and paired metric table

For each category and model: `n_objects`, `n_views`, `n_positive_objects`, `n_positive_views`, positive prevalence, I-AUROC/AP, O-AUROC/AP, P-AUROC/AP/AUPRO, optionally fixed-threshold precision/recall/F1 for all valid levels. Calculate paired delta `Δ_c = M_D2,c − M_ArcAD,c` on **same test samples and metric protocol**; annotate an incompatibility instead of forcing a delta. Summarize category wins/ties/losses, median and mean delta, 10th percentile, worst five categories; make a sorted delta plot. Keep `original_native_split` columns separate from `shared_split` columns.

**Statistical uncertainty:** cluster-bootstrap physical objects (retain five views and GT together), resample **paired object IDs** 2,000 iterations with fixed RNG seed; calculate CI 2.5–97.5% for each metric and paired delta, skipping/flagging resamples missing a class. For pooled across categories use **hierarchical bootstrap** (resample categories then objects) or fixed-category per-category cluster bootstrap depending on target estimand; document estimand. Category-level paired Wilcoxon across 160 categories may be supplemental, not on just three selected low categories. Report CI and effect sizes rather than interpreting a nonsignificant p-value as equivalence. Avoid uncorrected many-comparison significance fishing.

## 3. Quantify shared vs complementary detection errors (HIGH PRIORITY)

At each **prespecified** image/object threshold define failure sets over positive ground-truth cases: `F_D = {i: D2 misses i}`, `F_A = {i: ArcAD misses i}`. Report:

- Both detect, DiNomaly2 only, ArcAD only, neither (a 2×2 contingency count and percentages).
- `failure_Jaccard = |F_D ∩ F_A| / |F_D ∪ F_A|`; define 0/0 as NA.
- `D2_rescue_by_ArcAD = |F_D \ F_A| / |F_D|` when denominator > 0; reciprocal ArcAD rescue by D2.
- `joint_recall_oracle = 1 − |F_D ∩ F_A|/N_positive`; **explicitly label oracle coverage/union at fixed thresholds**, not an attainable ensemble result.
- Association of binary errors (e.g. phi coefficient), paired discordance count, McNemar test where appropriate; report data counts/CI.
- Stratify by category, defect type, and view. Preserve individual object/view IDs for paired visualization.

False positives: similarly analyze normal-object/view overlap, disagreement and rescue cost. Any proposed score fusion must report whether detection recall gains increase FP burden.

**Caution:** `max(D2_map, ArcAD_map)` is *not* an oracle performance bound. Pixelwise max may increase false positives; upper bound requires access to GT to choose the successful prediction, and must be labeled oracle, never deployable.

## 4. GT-driven defect morphology / difficulty covariates (HIGH PRIORITY)

For every **positive view**, compute geometry from the aligned GT mask, using nearest-neighbor resizing only and recording coordinate resolution:

- Defect area fraction `a = |G|/(H·W)`, and number of connected GT regions `R` (consistent 8-connectivity; report connectivity).
- Per-region area, bounding-box width/height, aspect ratio, perimeter, compactness `4πA/P²` (flag 0 perimeter), fragmentation/connected-component count.
- Defect position: centroid, distance to image or foreground boundary when foreground object mask exists; if unavailable, report just image border distance and do not invent object mask.
- Defect type from manifest (`Scratch`, `MissingPart`, `Abrasion`, etc.); camera/view index. Use original images and GT to calculate optional local luminance contrast (e.g. mean inside GT minus a narrow dilation ring outside GT); note that lighting changes and object texture can confound this proxy.
- Define size bins **in advance**, e.g. `<0.1%`, `0.1–1%`, `1–5%`, `>5%` of evaluated pixels, and report per-bin counts; merge sparse bins without hiding the decision. Also analyze size continuously (log-area).

Break down fixed-threshold defect-region recall, pixel AP/precision/recall, score contrast, top-k coverage, and view detection by size and defect type. Do not compute ROC AUC for groups that contain only one class or use image-positive-only groups to imply image AUROC. Prefer object-cluster bootstrap CIs and avoid strong conclusions for small groups.

## 5. Is the defect identified but outranked by normal pixels? (HIGH PRIORITY)

Let `A(x)` be a **numeric** anomaly map in one view (postprocessed, before figure visualization normalization); `G` its nonempty binary aligned GT mask; `B=¬G` within valid evaluated pixels. For each model/view calculate:

- `μ_G = mean_{x∈G} A(x)`, `μ_B = mean_{x∈B} A(x)`, `σ_B = std_{x∈B} A(x)`.
- Contrast `C = μ_G − μ_B`; and standardized contrast `Z = (μ_G − μ_B)/(σ_B + ε)`. These are **diagnostic measures**, not named benchmark scores. Don't compare raw `C` across models if scales differ; `Z` still depends on score distribution and should be accompanied by rank-based measures.
- **Defect-background ranking probability** `Q = Pr(A(g)>A(b)) + 0.5Pr(A(g)=A(b))` for GT pixels `g` and background pixels `b` in same view; compute exactly via ranks or a documented unbiased sample if too large. Equivalent to within-image pixel AUROC for a nonempty/nonfull mask. A high `Q` with low operational detection suggests threshold/aggregation issue; low `Q` means poor discrimination.
- `top_k_precision = |T_k(A)∩G| / |T_k(A)|`, `top_k_recall = |T_k(A)∩G|/|G|` for k ∈ {0.1%, 0.5%, 1%, 2%, 5%} of pixels. Define tie handling and minimum selected-pixel rounding, save k and GT area fraction. These can behave differently across defect sizes; never interpret top-k precision without mask prevalence.
- `GT_max_percentile` and `GT_mean_percentile`: rank GT pixel responses relative to within-view background distribution, with robust tie handling.
- `off_target_top1_fraction = 1 − top1_precision`; false-positive area at a *fixed calibrated map threshold*: `FP_area_fraction = |{x∈B:A(x)≥τ}|/|valid pixels|` and `FPR_pixels = FP/|B|`. Keep both denominators explicit.

Flag classes for inspection, based on transparent fixed criteria: `no_signal`, `on_target_but_outranked`, `high_FP_normal_texture`, `partial_region`, `view_dependent`. Keep continuous measures alongside labels; criteria are heuristic, not ground truth for causal explanations.

## 6. GT-region sensitivity and localization (HIGH PRIORITY)

For every connected GT component `G_j`, calculate best intersecting predicted connected component at a **fixed calibrated pixel threshold**, IoU, GT-region recall (any overlap), and stricter coverage `|pred∩G_j|/|G_j|`. Report detection at region IoU thresholds 0.1/0.3/0.5 and region recall at controlled pixel FPR (e.g. 1%/5%; choose calibration separately, not from test). Show by region size, count, category and defect type. Unmatched predicted regions are FPs; report FP components per normal view and per defective view. Avoid a metric that trivially labels a full-image positive prediction as perfect detection.

Keep benchmark P-AUPRO separate from fixed-threshold component recall. For physical regions split across multiple views, count GT component per view and object separately; do not assume spatial correspondence across camera views.

## 7. Multiview consistency (HIGH PRIORITY)

For each **defective object**, report: #GT-positive views (may be <5), #successfully detected GT-positive views at the fixed per-view threshold, fraction of positive views detected, mean and spread of within-model image scores, coefficient of variation only if scores positive and stable, and `object_detected` from **native pooled five-view score**. Also report normal views of defective objects separately: high anomaly scores there may indicate spurious object-level cues.

Create histograms of 0/1/2/... successes **conditional on actual GT-positive view count**; it is incorrect to score a camera view without a visible GT defect as a missed defect. Compare D2 and ArcAD on the exact same object/view IDs. Analyze whether the two models rescue **different** camera views on the same objects.

## 8. Supervision and unseen-defect hypothesis

Using ArcAD `labeled` manifest, determine **by category** which `anomaly_class` types had labeled examples and how many *distinct physical objects* and views per type. On the shared test split, partition defect images/objects into `seen defect type` vs `unseen defect type` **relative to ArcAD labeled categories**, ensuring no labeled physical object enters the test. Disclose unequal prevalence/sample counts and differences in intrinsic defect difficulty.

Compare each model within each stratum (fixed-threshold recall, pixel and region measures, score contrast), then examine difference-in-differences:

`(ArcAD − D2)|seen − (ArcAD − D2)|unseen`

**Interpretation:** association only. It does **not** isolate causal benefits of supervision because seen and unseen defects may differ in size, visibility or intrinsic difficulty. For causal evidence later propose separate controlled ablation of ArcAD with 0 / 1 / 2 / k supervised objects per category if supported by its implementation, fixed splits, repeated seeds; do not silently execute training in this task. Do not describe zero-defect ArcAD as an existing valid configuration without verifying code supports it.

## 9. Exploratory causal/mechanism alternatives

Create defect-difficulty covariate vs outcome panels (scatter or grouped distributions): log mask area, local image contrast, #regions, view index, defect type, edge proximity, and optional normal texture complexity. Fit simple interpretable exploratory models (e.g. logistic regression of missed detection or mixed-effects / category-adjusted regression where practical). Cluster by physical object, include category effects, and check sparse strata; report estimated associations, uncertainty and limitations, not definitive causes. Use the same controls for both models. Correlate category-level delta with aggregate morphology only after establishing within-category sample distributions.

## 10. Ensemble feasibility, without training an ensemble yet

Calculate complementary success proportions and **oracle union coverage** only for same-sample fixed-threshold decisions. Test simple **held-out calibrated** baselines if an independent validation subset is available: score percentile calibration from training normals; per-model z-score normalization fit on normals; average or max fusion of calibrated image/object scores; pixel fusion only on aligned maps. Select fusion hyperparameters on validation, **never on the shared test**. Compare to the best single model at **matched false-positive rate**; record actual precision, recall, FPR, AUROC/AP, P-AUPRO and false-positive area. Include independent raw score scales and calibration curves/normal score distributions. If no suitable validation data exist, report complementarity diagnostics only, not tuned ensemble performance.

For eventual multi-decoder DiNomaly2 work, log seed/checkpoint diversity and per-region error correlation before asserting ensembles add complementary information.

## 11. Qualitative example selection: evidence-first, paired

Current `selected_examples_10_per_category` collections are ranked low-confidence cases from each model **independently**. Their example indices need not refer to the same objects; never place them side-by-side as direct comparisons without matching by full image path + camera view.

Select examples programmatically after quantitative analysis, at minimum: shared misses, D2-only success, ArcAD-only success, confidently wrong high-FP normal views, on-target-but-outranked, view-dependent misses, and a well-detected control. Save the selection rule and numerical evidence beside each paired figure.

Figure panel: same raw 1024 source image (or clearly identified common crop), GT mask, D2 heatmap, ArcAD heatmap, top-k selected pixels and/or threshold overlay; include paired scores, GT area, Q, top-k precision/recall, view index, category, defect type. If visuals normalize each heatmap separately, state **“colors normalized per model/view; compare spatial patterns, not absolute intensity.”**

## 12. Deliverables / file schema

Create a new folder `analysis_shared_failure_<runid>/` with:

- `README.md`: tested hypotheses, data splits, checkpoints, preprocessing and metric definitions, limits, no-leakage confirmation.
- `joined_predictions.parquet` or CSV: one record per matched view with `category`, `object_key`, `image_path`, `view_id`, `gt_view_label`, `gt_object_label`, `defect_type`, `model`, `score_view`, `score_object`, `map_path`, `gt_mask_path`, `split_id`.
- `view_diagnostics.csv`: `gt_area_fraction`, `gt_region_count`, `C`, `Z`, `Q`, top-k precision/recall, FP metrics, detected-at-threshold flags, with `valid_mask_coordinates` flag.
- `region_diagnostics.csv`: component-level measures, IoUs and detection flags.
- `object_diagnostics.csv`: GT-positive-view count, detected-positive-view count, pooled score, object success, view score spread, opposite model success.
- `category_metrics.csv`, `delta_by_category.csv`, `bootstrap_ci.csv`, `failure_overlap.csv`, `seen_unseen_summary.csv`, `error_taxonomy_counts.csv`.
- `plots/`: category delta; paired score rank scatter; success/disagreement matrix; size-response curves; contrast vs detection; top-k curves; multiview consistency; breakdown by defect type; representative paired failures.
- `selected_examples.csv` and paired example PNG/SVG figures with reproducible selection rules.
- `analysis_report.md`: clearly separate **observations**, **statistical associations**, **hypotheses**, **unverified explanations**. Include at least one favorable/control category and explain selected-low3 bias.
- `run_analysis.py`, `config.yaml`, `requirements.txt`/environment record and `reproduce.sh` or equivalent reproducible entrypoint.

## 13. Acceptance tests and stop conditions

- Every pair has identical source image and intended GT label; no ArcAD training sample or training object is counted as test.
- Every cached anomaly map can be traced to original checkpoint, file path, transform and view; GT and map spatial alignment validated visually on random examples and programmatically on masks.
- AUROC/AP reproducibility agrees with Task 01 on the common split; any discrepancy is explained.
- No threshold selected using test GT labels except explicitly marked oracle benchmark maximal-F1.
- Missing classes produce `NA` for undefined ROC/region metrics, never silently 0/1.
- Bootstrap resamples at **object level** and maintains pairing; multiple views are not falsely counted as independent objects.
- Oracle-union and tuned fusion results are distinctly labeled; no unfounded causal claims.
- If ArcAD raw maps are unavailable, first write and run an **inference-only** export for the already-trained ArcAD checkpoint on the common manifest, preserving its original transform and checkpoint; do not derive numeric diagnostics from PNG color maps.
- If common pixel coordinate mapping cannot be verified, perform view/object analyses and label pixelwise cross-model comparisons **blocked** until alignment is solved.

**Priority order:** (1) common sample + correct map/GT alignment, (2) paired baseline and failure overlap, (3) contrast/rank/top-k and GT-region diagnostics, (4) multiview, (5) morphology / seen-unseen, (6) uncertainty and ensemble feasibility. Complete rigorously rather than manufacturing every metric at once.
