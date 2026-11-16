# Task 04A — Controlled cropping and field-of-view ablation (Codex execution specification)

## Objective
Test whether ArcAD's original inference preprocessing (resize 1024×1024 source to 448×448, then center crop 392×392) contributes to missed border defects, compared with feeding the **same frozen ArcAD checkpoint** a full-frame 392×392 resize. Keep DiNomaly2's original full-frame 280×280 inference as reference. This is an **inference-time intervention**, not a reproduction of a newly trained model and not proof of what would happen after retraining.

## Starting evidence and provenance (verify locally; don't assume)
- Dataset: Real-IAD Variety source images/masks nominally 1024×1024, with five views C1–C5 per physical object.
- Original ArcAD preprocessing: resize 448, center crop 392; source-equivalent visible region `[64:960) × [64:960)` assuming exact square 1024 input and square resize.
- Original DiNomaly2: full-frame resize to 280×280; maps later resized/smoothed for evaluation.
- Prior paired Task 01/02/03 artifacts under lab workspace `/media/avlab/1CE6451EE644FA12/2026_10_22_dinomaly/` and dataset `/media/avlab/7458a474-6ea2-4b90-b232-373c3df8d329/real_iad_variety_1024` (verify paths).
- Task 03 diagnosed 41 laser_diode *positive views whose common-crop GT is empty*: 39 masks nonempty in full source but entirely outside common center crop, 2 original all-zero masks (`NG/OilStain/S0005` C1/C4). These are **not automatically ArcAD false negatives**. Preserve original labels.
- Models have different native train/test splits; use exact path intersection excluding ArcAD supervised training objects, not merely category-name matching.
- Preserve checkpoint files and hash them before/after. Do **not** train, modify optimizer, or overwrite original artifacts.

## Required conditions
1. **ArcAD-original**: the exact saved model, original resize448→center crop392 normalization/processing, scores/maps. Regression-check against Task 02 outputs (numerical tolerance declared, including CPU/GPU differences).
2. **ArcAD-full-frame-inference**: same frozen checkpoint, resize full 1024×1024 source to **392×392 without crop**, identical normalization, anomaly-map construction, smoothing, object pooling and aggregation as original, except for the explicitly changed image geometry. Verify the model accepts this tensor shape, and use safe interpolation consistent with repo conventions. This branch is an *out-of-distribution inference experiment* relative to original training.
3. **DiNomaly2-original**: frozen checkpoint, original full-frame 280×280 processing; reuse proven, path-aligned Task 02 numeric maps/scores where available, otherwise infer. Do not change its scoring scheme.
4. **Optional, clearly secondary**: DiNomaly2 source-centered crop `[64:960)` resized to 280×280 using same frozen D2 checkpoint. Only run after required conditions; report as intervention, never mix with main comparisons.

## Sample design and anti-leakage
- **Primary categories:** `laser_diode`, `lattice_block_plug`, `power_jack` as already evaluated. Include `LED_indicator` and `thyristor` as preselected comparison categories if runtime/storage allows; do not silently narrow to only the 39 suspicious samples.
- Reuse Task 02 shared manifest, verifying checksum, five-view physical-object grouping, exact image paths, label/mask consistency, and exclusion of ArcAD training objects from every evaluation/test partition.
- Primary metrics should use **entire eligible categories** and unchanged shared membership; the 39 crop-external views are a prespecified *diagnostic stratum*, not the sole test set. Provide separate counts for border, central, all-zero-source-mask, normal.
- Do not select thresholds or branch/parameter choices using test results. No new train/validation split needed to compute threshold-free AUROC/AP; any thresholded result needs independent normal calibration objects excluded from evaluated objects and clearly labeled as diagnostic if drawn from original test manifests.
- Keep existing Task 03 calibration settings in a separate descriptive comparison if relevant; avoid calling them deployment-ready.

## Image geometry: mandatory checks
- Verify actual image shape and source-to-input mapping using test patterns / pixel coordinates, not merely expected dimensions.
- Produce mappings from source (x,y) to: ArcAD-original crop, ArcAD-full-frame, DiNomaly2-full-frame, and each output anomaly-map pixel. Verify interpolation and any padding/crop in patch/token pipelines.
- **Critical:** Full-frame ArcAD maps are *not* the same geometry as original ArcAD maps. Do not directly compare them in 256×256 map coordinates. Project to a common **source-aligned** grid with documented interpolation:
  - **Central field:** source `[64:960)×[64:960)` for all three branches, e.g. 224×224 grid, matching Task 02.
  - **Full source field:** 1024×1024 equivalent (downsample to documented grid, e.g. 256×256) for ArcAD-full-frame and DiNomaly2. ArcAD-original has no prediction outside its observed crop; **mark outside as not observed (NaN/mask) and do not invent anomaly scores or zero-fill**. If full-frame GT metrics are shown for original ArcAD, label them explicitly as coverage-aware/abstention metrics and specify their interpretation; don't pretend it predicted unseen border pixels.
- Preserve both *raw numeric anomaly maps* (float32) and binary GT masks; figures may normalize colors for display, but numerical comparisons must not use display-normalized images.
- Masks: use nearest-neighbor source-aligned resampling and report original positives completely outside original crop, partial overlaps, nonempty center masks, and original-all-zero GT separately.

## Evaluation outputs and metrics
### Primary threshold-free detection
For each category and branch, compute **I-AUROC, I-AP, O-AUROC, O-AP**, using exact shared images and physical objects, with identical score definitions per level. `I` = each camera image; `O` = five-view physical object. Note that changing image preprocessing may legitimately change the numeric top-1% pooled score: retain original aggregation fraction and distinguish per-view top-1% from pooled-object top-1%.

### Pixel-level localization
Compute P-AUROC, P-AP, P-AUPRO using common definitions / common source-aligned center grid for **all three branches**. For full-frame compare **only ArcAD-full-frame and DiNomaly2** on the full source grid; original ArcAD center-crop cannot provide comparable full-frame localization predictions. Report per-GT-region recall/IoU and top-k overlap, stratified by whether GT is inside crop, partly outside, wholly outside, or source-mask empty. Exclude empty-in-evaluation-field GT views from *positive-region localization metrics* but preserve them in image-level metrics and report how they affect pixel-negative computations.

### The 39 border cases
Create one row per view with category/object/camera/path, original-mask area inside/outside crop, original and full-frame ArcAD scores, D2 original scores, thresholds if valid, GT-region map score/rank, and visibility flag `requires_manual_review`. Separate **cannot see annotated GT** from **predicts normal** (not the same). For any binary detection comparison, calibrate thresholds on independent normal objects and indicate held-out measured FPR.

### Paired uncertainty
Use **paired physical-object cluster bootstrap** (retain all camera views), seed and at least 2,000 replicates if tractable. Report delta CIs for ArcAD-full-frame vs ArcAD-original at I and O levels; compare D2 as a reference. Handle single-class resamples without replacing with zeros. Don't characterize a CI including zero as significant; disclose exploratory multiple comparisons.

### Performance diagnostic
Record full model and data preprocessing versions, GPU availability, CUDA/PyTorch, elapsed inference (descriptive), batches, random seed, exact hashes of checkpoint/code/manifest, numerical map finite checks and all reused-cache provenance. Use GPU for new inference if available; if CUDA unavailable, stop with a clear report rather than silently launching massive CPU inference. CPU may be used for aggregation.

## Required deliverables (new isolated Task 04A output folder)
- `README.md` with commands, full provenance, explicit inference-only warning and interpretation limits.
- `protocol_audit.json`, `manifest_validation.json`, `checkpoint_integrity.json`, `geometry_validation.json`, `issues.md`.
- `category_metrics.csv`: per-category and branch I/O/P results, center/full-field tags; no inappropriate comparisons.
- `paired_bootstrap_ci.csv`: paired effects and object-level confidence intervals.
- `border_39_audit.csv` plus `border_39_contact_sheet/` and manual-review sheet; record unresolved visibility labels without fabricating resolutions.
- `per_view_scores.csv`, `per_object_scores.csv`, `maps/` float32 source-aligned outputs + index and GT alignment metadata.
- `plots/`: per-category ROC/PR; ArcAD original vs full-frame score distributions for positive and normal; border vs center comparison; 5-camera paired examples with same color scale **within model/branch when meaningful**, otherwise label per-panel scales.
- `summary.md`: answer (a) did full-frame ArcAD help overall? (b) did it help the border cases? (c) did it degrade central or small defects? (d) did object-level aggregation alter the conclusion? (e) is retraining warranted? Include effect sizes, denominators and caveats.
- `reproduce.sh` that does not modify any previously recorded baseline results.

## Acceptance and stop conditions
- All required sample keys exactly match manifest, no missing/duplicated views, and all five views grouped correctly.
- Frozen checkpoint hashes unchanged; exact input size/normalization verified.
- ArcAD-original regression matches previous outputs within tolerance or explicitly stops and diagnoses divergence.
- Geometry validation passes, including explicit 39 outside-crop and 2 all-zero-source-mask cases; if mismatch investigate before metric claims.
- No training, no data labels changed, no test-informed choice of inference hyperparameters; branch comparison is predeclared.
- Distinguish: (i) input observability, (ii) localization, (iii) image classification, (iv) thresholded decisions. Never infer (iv) solely from (i).
- If full-frame input breaks the architecture/position embeddings or changes required internal geometry unexpectedly, stop and report compatibility rather than silently patching network weights.

## Interpretation / next decision
This experiment can establish the *inference-time sensitivity* of the **existing** ArcAD checkpoint to crop removal, including effects on border vs central objects. It **cannot** isolate the long-term benefit of using full-frame inputs consistently during training. If ArcAD-full-frame performs better or results are mixed due to distribution shift, propose a **separate controlled retraining ablation** using matched training budgets/seeds/splits, not changes to this task's baselines. For Task 04B, keep the frozen original-ArcAD + D2 fusion rule as the primary generalization baseline; any full-frame-ArcAD fusion is an explicitly separate secondary branch.
