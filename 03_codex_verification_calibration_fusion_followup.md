# Task 03 for Codex — Verify failure mechanisms, repair operating-point analysis, and test modest fusion baselines

## Mission and scope

Use the **existing trained DiNomaly2 and ArcAD checkpoints** on Real-IAD Variety 1024. The goals are (1) verify anomalies and unresolved limitations from Task 01/02, (2) identify the actual kinds of errors each model makes, (3) decide whether model complementarity offers meaningful benefit at controlled false-positive rates, and (4) audit which of the **160 categories** merit further study. **Do not train either original model, overwrite checkpoints, change original GT labels, alter the original result folders, or present oracle upper bounds as deployable ensemble results.** Work in a new timestamped output directory and make the new experiment reproducible.

This task is a *follow-up to*, not a replacement for, the earlier runs. Ground paths from the lab filesystem; do not assume the paths below are portable.

### Known inputs and provenance to locate

- Original resized source dataset: `/media/avlab/7458a474-6ea2-4b90-b232-373c3df8d329/real_iad_variety_1024`.
- DiNomaly2 checkpoint: `Dinomaly2/out/real_iad_variety_1024 out/full run/variety_160cat/model.pth`; original training image size/crop 280/280, output map 256, Gaussian 5x5 sigma 4.
- ArcAD checkpoint: `ArcAD/out/real_iad_variety out/arcad_realiad_variety/model.pth`; resize 448, center crop 392, map 256, Gaussian 5x5 sigma 4.
- Task 01: `Dinomaly2/out/re_eval_shared_20261031_cpu01/` with saved D2 maps/scores for `lattice_block_plug`, `laser_diode`, `power_jack`.
- Task 02: find `analysis_shared_failure_20261031_gpu01/` with `shared_manifest.json`, `maps/common_*.npz`, `alignment_validation.json`, `category_metrics.csv`, `failure_overlap.csv`, `joined_predictions.csv`, `view_diagnostics.csv`, `region_diagnostics.csv`, `seen_unseen_summary.csv`, `selected_examples.csv`, `analysis_report.md`.
- Five Task-02 categories: `lattice_block_plug`, `laser_diode`, `power_jack`, `LED_indicator`, `thyristor`.
- Task 02 has already confirmed common **source coordinates [64:960) × [64:960)**, common **224x224** grid: D2 map crop `[16:240]` and ArcAD 256->224 interpolation. Preserve this alignment and verify it independently before calculating pixel metrics.
- Task 02 reported **41 positive laser_diode views with empty full-source masks**, compared with Task 01's two `OilStain/S0005` examples. Task 02's 41 include many `MissingPart` views. Treat them as unresolved *view visibility/annotation* cases—not automatically erroneous labels.
- Original reported full-160-category I-AUROC macro means: D2 **0.8859**, ArcAD **0.8599**. User additionally reports **83/160 D2** and **100/160 ArcAD** categories with I-AUROC < 0.90: independently check and document exact comparison strictness and source files.

## Non-negotiable data separation / leakage controls

1. Preserve **object identity**, grouping all five cameras of one physical object. Never split views of one object across train/calibration/validation/test partitions.
2. Keep all ArcAD labeled training objects out of any common held-out test. Validate with actual paths and object keys, not inferred IDs alone.
3. Do not select models, fusion parameters, thresholds, categories, or normalizers by looking at the final test labels. Predefine experiments and distinguish preselected diagnostic categories from unbiased performance estimates.
4. **Important:** Task 02's ten held-out normal objects/category are too few to resolve 1%/5%/10% FPR calibration levels. Do not relabel repeated maximum-score thresholds as distinct FPR operating points. Never tune test thresholds on the test negatives. If an adequate independent calibration/validation split is impossible, report this limitation and use threshold-free ROC/PR curves and explicitly labeled exploratory analysis rather than claiming deployable operating points.
5. A category-specific calibration/fusion rule is allowed only if enough independent calibration/validation objects are available; otherwise prefer a global rule estimated from independent categories/objects, documenting the transfer setting. ArcAD's labeled anomaly training examples must never be repurposed as independent validation or test data.
6. Preserve Task 01/02 scores, output files, metric definitions, and checkpoint hashes. Use CUDA if available for substantial new inference; show GPU, PyTorch, model SHA256, inference device. Fail loudly if new large inference accidentally falls back to CPU; CPU is fine for CSV statistics/plots.

## Phase 0 — Inventory, audit the 160-category claim, and freeze a plan

A. Locate both original 160-category results and parse category-wise I-AUROC. Report count strictly `<0.90` and `<=0.90` separately, handling displayed rounding carefully; verify **100 ArcAD / 83 D2** rather than treating these as unquestioned facts. Reproduce 160 macro averages and note differing native evaluation splits and supervision regimes.

B. Generate `all160_category_screening.csv` with category, both original I-AUROCs, delta, weak_both, weak_D2_only, weak_ArcAD_only, and rank. Create a **2x2 contingency table**: both <0.9, D2 only <0.9, ArcAD only <0.9, neither. This is an overlapping *screening comparison*, not a paired statistical inference because original test lists differ.

C. Build a short, **precommitted** prioritized extension list beyond the existing five: select representatives of shared weakness, D2-only weakness, ArcAD-only weakness, and strong performance; use objective criteria (rank, disagreement magnitude) stated *before* generating new maps or examining per-object outcomes. If compute is limited, keep new inference optional; prioritize Task 02's saved numeric maps for the first pass.

D. Compare source commit/hash, model pipeline, normal/defect counts, mask handling, object and view labels, input transforms, score extraction. Save `protocol_audit.json`.

## Phase 1 — Resolve visibility, mask and geometry caveats

A. Recompute the `laser_diode` **41 empty-source-mask** list from original source PNGs, and distinguish:
- genuinely all-zero source mask on a view labeled anomalous;
- nonzero source mask but fully outside the shared `[64:960)` crop;
- nonzero source mask whose region vanishes during 224x224 downsampling;
- missing or corrupted mask file;
- true image-label / object-label mismatch.

B. For **all 41** exceptions, save a machine-readable audit with object ID, camera, defect type, source-mask pixel count, cropped-mask pixel count, source data shape, image/view/object labels, and status. Generate a contact sheet showing the image, *uncropped source GT*, common-crop GT, and if possible sibling camera views of the same object. If visibility cannot be determined from a photo without expert annotation, mark `requires_manual_review`; do not assert the cause. Do not repair original dataset or invent masks.

C. Conduct one *label sensitivity* calculation on the diagnostic subset: (i) original labels, (ii) remove only ambiguous positive views from **mask-dependent** calculations, (iii) optional transparent exploratory view-label sensitivity, while retaining correct object-level labels. Keep published/original benchmark metrics unchanged.

D. Inspect cross-model maps for the same physical pixel locations. Save spot-check overlays and assert GT/map dimensions, nonempty expected maps, valid score ranges, finite values, and no sample duplication.

## Phase 2 — Make threshold-dependent evidence meaningful

**Primary results:** threshold-free per-category **view I-AUROC/I-AP**, **object O-AUROC/O-AP**, and common-grid **P-AUROC/P-AP/P-AUPRO**, plus ROC/PR curves and confidence bands where feasible. Maintain the existing top-1%-score definitions and separately show sensitivity to top-0.1%, 0.5%, 2% and 5% *without choosing the best setting on test*.

1. Report raw score distributions for normal and defect samples per model/category, separately at view and object level. Note that I-AP is sensitive to the large anomalous-positive prevalence in these test splits (~88-89% of objects); include class prevalence and ROC/PR baseline.
2. Revisit threshold calibration. Document exactly which normal objects were held out and what data remains for independent validation. **If independent negative calibration examples are still only 10/category, do NOT claim meaningful 1% FPR or three distinct FPR targets.** Instead give empirical ROC/PR tradeoffs *descriptively* on the held-out diagnostic data, clearly labeled test-derived descriptive curves—not deployable calibrated performance—and bootstrap physical-object confidence intervals. For genuinely calibrated deployment-like curves, obtain a larger disjoint normal calibration/validation population from allowable original normal training data (but verify potential train overlap and avoid using ArcAD supervised training object views), and then evaluate on untouched test objects. Disclose source split and distribution shift.
3. For any proposed threshold, show calibration count, actual calibration FPR, measured held-out FPR/recall, **binomial uncertainty**, confusion counts, and thresholds on each model's native scale. Use exact FPR counts rather than decimals that imply false precision. Do not optimize F1 thresholds using test labels except in explicitly labeled `oracle_max_F1` diagnostic columns.
4. Repeat failure overlap as a function of threshold/false positive budget. Give contingency for `both detected`, `D2 only`, `ArcAD only`, `neither`, for *views and physical objects*, along with false-positive **union cost** on normal objects. Count sibling views in cluster bootstrap as one physical-object sampling unit.
5. Include region-level comparison at pixel FPR budgets, using same GT region definitions and common grid. Report GT-component recall at fixed IoU thresholds 0.1/0.3/0.5 and coverage; OR of detections is an **oracle upper bound** for finding positive regions, not an achievable ensemble outcome. Report its associated false-positive union cost when evaluating an implementable OR rule.

## Phase 3 — Verify mechanisms, not just heuristic labels

Use predefined strata and paired examples for both models on identical images:

- **Localization priority / on-target-but-outranked:** GT anomaly-score quantile versus background, GT score contrast, GT percentile rank distribution, top-k GT recall and precision (k = 0.1%, 0.5%, 1%, 2%, 5% of common map). Show *actual higher-scoring background areas* on overlays, not just a label inferred from quantiles.
- **Defect area:** GT positive area fraction, connected component size/count and minimum width. Stratify without choosing post-hoc boundaries to maximize an effect. Distinguish image detectability from pixel localization; positive views with empty masks must not enter area analyses.
- **Viewpoint:** for each physical object, per-camera scores, GT visibility/area, per-camera hit at a defined threshold, and variability across C1–C5. Check whether specific camera misses correspond to empty masks, severe occlusion, edge crop, or legitimate view changes.
- **Natural contrast / specularity:** distinguish *input RGB contrast* from *model-derived anomaly-score contrast* (the latter cannot independently explain why the model failed without circularity). RGB local-luminance contrast, gradients/edges and background texture can be exploratory covariates; speculative descriptions like reflective/glossy must be verified visually or labeled as hypotheses.
- **Seen versus unseen defect-type effects:** group by defect type present/absent in ArcAD labeled training, but do not infer a causal training benefit without balanced controlled supervision or ablations; record sample sizes and types.

Select at least 3-5 representative *paired* examples for: shared miss, D2-only success, ArcAD-only success, localization disagreement, threshold-only miss, and false positive on a normal view. Use deterministic objective selection, and show unnormalized scores separately from display-normalized heatmaps. Include object/camera, GT, both anomaly maps, thresholds or percentile ranks and explanatory measurements. Curate a **small manual review sheet** for any visual explanation; explicitly distinguish `observed`, `plausible`, and `unverified`.

## Phase 4 — A fair minimal fusion benchmark

Only after the data/split/calibration checks, evaluate simple fusion rules with **fixed, validation-fitted parameters** and no test-based tuning:

- `D2 alone` and `ArcAD alone` (required baselines).
- View/object score averaging after model-wise calibration using independent normal/validation data (e.g. standardized score or empirical normal quantile; specify precise transform and avoid per-test-image min-max normalization).
- Fixed 50/50 mean of calibrated scores.
- One weighted mean with weight selected on validation only; retain chosen weight and validation criterion.
- A simple calibrated OR rule: each model flags if its independently chosen threshold is exceeded. Report full union false-positive cost; don't compare recall alone.
- OPTIONAL map-level mean or max fusion for pixel localization, but only on precisely aligned common-grid maps with score scales normalized using calibration, and quantify potential extra false-positive area.

Report held-out I/O AUROC, AP, ROC/PR curves, recall at *measured* FPR, P-AUPRO/P-AP if applicable, per-category and defect-type deltas, category macro mean and paired object-bootstrap confidence intervals. Include a category breakdown for when fusion **hurts** as well as helps. If no independent validation anomalies exist for tuning, skip tuning; run a fixed unsupervised calibration/50-50 baseline and explicitly disclaim its limitations. An oracle selection using GT is **not** a realizable fusion baseline. Do not describe elementwise map-max as a theoretical upper bound.

Do not infer 160-category ensemble improvement from the five preselected diagnostic categories. Any generalizable claim requires held-out, prespecified category/object evaluation across a representative subset or all 160 categories.

## Deliverables (all in a new output directory)

- `README.md`: minimal reproduction commands, source files, device, hash, objective and high-level findings.
- `protocol_audit.json`, `category_160_audit.csv`, `weak_category_overlap.csv`, `category_selection_plan.md`.
- `empty_mask_audit.csv`, `empty_mask_contact_sheet/`, `data_quality_issues.md`.
- `threshold_audit.csv`, `calibration_split_manifest.json`, `calibration_reliability.md`.
- `threshold_free_metrics.csv`, `operating_point_metrics.csv`, `failure_overlap_vs_fpr.csv`, `region_overlap_vs_fpr.csv`.
- `mechanism_by_stratum.csv`, `object_camera_consistency.csv`, `paired_example_index.csv`, `paired_examples/`.
- `fusion_comparison.csv`, `fusion_validation_selection.json`, `fusion_failure_cases.csv`, relevant plots including ROC/PR and recall-vs-measured-FPR.
- `bootstrap_intervals.csv` (report number of resamples, seeds, physical-object clustering, and excluded/invalid draws); `analysis_report.md` with separate headings **Verified observations**, **Evidence for/against mechanisms**, **Unverified hypotheses**, **Practical ensemble result**, **Limitations**, **Recommended next experiment**.
- `issues.md`: blockers, unresolved GT cases, insufficient calibration and GPU limitations. Clearly mark skipped deliverables as skipped with a reason, not silently omitted.

## Acceptance criteria

- Existing checkpoints and earlier output files are byte-for-byte unchanged. No training.
- Exact shared object/view identifiers and expected counts verified; zero train/test object leakage.
- 160-category weak-counts independently rechecked, including both/one/neither overlap (or explicit mismatch reported).
- 41 empty-source-mask cases fully accounted for and reconciled with Task 01's two cases. Original benchmark labels remain unchanged.
- Spatial alignment validated by source image coordinates and mask overlays; compare numeric maps before display normalization.
- No claimed 1%/5%/10% calibrated thresholds when calibration resolution cannot distinguish them; no test-label tuning.
- Distinguish oracle maximum F1/union coverage from implementable threshold and fusion results.
- Fusion compared to both models on **the same held-out samples**, with measured FPR and CIs; state if independent validation was unavailable.
- Conclusions do not overgeneralize from five cherry-picked categories or treat model-derived features as independent causes.
- Runnable reproduction scripts, environment/provenance metadata, machine-readable CSVs and a concise research-summary report.

## Execution strategy and priorities

First do **Phase 0 + Phase 1** (cheap, high priority) and produce a short issues report. Then **Phase 2** using saved Task 02 maps, with no unnecessary rerun. Only attempt **Phase 4 fusion** once a valid calibration/validation design is established. If data separation cannot be guaranteed, deliver the descriptive analysis rather than inventing calibrated estimates. For optional expansion to more categories, record compute requirements and a frozen selection plan, then reuse checkpoints and GPU inference; never automatically launch all 160 categories without evaluating necessity and compute/storage cost.
