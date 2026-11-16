# Codex Task 01 — Re-evaluate the EXISTING DiNomaly2 checkpoint on the ArcAD–DiNomaly2 shared, leakage-free Real-IAD Variety split

**Purpose:** Produce a genuinely paired comparison with ArcAD for the already-trained Real-IAD Variety model. **Inference and evaluation only; absolutely no training, fine-tuning, optimization, checkpoint updates, or split resampling.** Do not overwrite the original run, checkpoint, log, or results.

## 0. Known local context (verify; do not treat as permission to guess)

- Original dataset directory reported in the existing scripts: `/media/avlab/7458a474-6ea2-4b90-b232-373c3df8d329/real_iad_variety_1024`.
- ArcAD supervision manifests: `realiadvariety_jsons/sup/<category>.json` under that root.
- DiNomaly2 native manifests: `realiadvariety_jsons/Real-IAD_Variety_jsons/<category>.json`.
- Existing comparison directory: `out/real_iad_variety out/low3_shared_split_comparison_20261031/` **under the ArcAD repository**, containing `prepare_shared_split.py`, `shared_test_manifest.json`, `evaluate_arcad_shared.py`, `arcad_common_test_metrics.json`, `comparison_metrics.csv`, and `README.md`.
- DiNomaly2 review directory: `review_low3/` in the DiNomaly2 Variety results; its README identifies the checkpoint as `variety_160cat/model.pth`, already trained. **Discover its exact absolute path in the lab filesystem and record it.**
- Initial categories: `lattice_block_plug`, `laser_diode`, `power_jack`. These are the low-three DiNomaly2 I-AUROC categories, NOT necessarily ArcAD's low three; `LED_indicator` is an additional worthwhile later case.
- As documented by existing DiNomaly2 review README: inference uses 280 × 280 input/crop, 256 × 256 maps, top-1% image/object scoring, and 5 × 5 Gaussian smoothing. **Verify the precise implementation from the original evaluation code** (normalization, sigma, interpolation, label logic, object pooling and transforms). The base image files being 1024 × 1024 does *not* mean that network input was 1024 × 1024.
- ArcAD's *separate* evaluation uses 448 resize → 392 center crop → 256 map, Gaussian 5 × 5, sigma 4, mean top-1% image scores. **Do not silently replace DiNomaly2 preprocessing with ArcAD preprocessing** in the primary reproduction. The differing model pre-processing is a documented comparison limitation and can be studied separately.

## 1. Preflight: trace provenance before executing inference

1. Locate DiNomaly2 repo, original training/eval config, exact trained `.pth`, original full-run log, and `review_low3` inference-only script. Print relative and absolute locations.
2. Capture `git rev-parse HEAD` for each repo (if tracked), Python/PyTorch/torchvision versions, CUDA/device info, relevant package versions, original checkpoint SHA-256, and original script/config hashes. If source code has uncommitted modifications, report them without resetting or overwriting anything.
3. Inspect the **original** Variety manifest and the `shared_test_manifest.json` structure. Resolve image paths, mask paths, labels, five-view object IDs. Preserve the source manifest and copy or reference it read-only.
4. Identify whether the old `review_low3` inference implementation reproduces the original DiNomaly2 final category metrics (sanity check). If not, stop and document discrepancy instead of silently proceeding.
5. **Do not claim a training-data leak in DiNomaly2.** DiNomaly2 used normal-only training. The issue is that one defect object used as ArcAD labeled training appears in DiNomaly2's native test population, and original ArcAD's test population includes additional normal objects. Our goal is the **same held-out test image/object population for both models**.

## 2. Validate the shared split (required before any score comparison)

Inspect `prepare_shared_split.py` and regenerate its output in a *new working directory*, then compare against the archived `shared_test_manifest.json`. The selection rule is: exact image-path intersection between ArcAD test and DiNomaly2 native test, with **every ArcAD labeled training image excluded**. Never use any ArcAD labeled image in paired evaluation, even if an original test manifest accidentally includes it.

Required known counts (assert these unless the files have demonstrably changed):

| Category | DiNomaly2 original test objects | ArcAD original test objects | Paired test objects | Paired defect objects | Paired normal objects | Paired images |
|---|---:|---:|---:|---:|---:|---:|
| lattice_block_plug | 224 | 240 | 223 | 198 | 25 | 1,115 |
| laser_diode | 218 | 235 | 217 | 192 | 25 | 1,085 |
| power_jack | 225 | 241 | 224 | 199 | 25 | 1,120 |

For each category assert: unique `(category, image_path)`; 5 unique views per object; no ArcAD labeled path in shared; labels and mask paths match across source manifests; referenced images and masks exist; normal views have empty GT masks; an object's label is positive iff any of its views is anomalous. Verify no object-level overlap with ArcAD labeled training objects, not only image-path overlap. Preserve sample count and excluded-object reports.

**Important:** The already-generated ArcAD shared-test results are fixed and should remain read-only. Recalculate ArcAD only if there is a clear provenance mismatch, and save revised results separately with an explanation.

## 3. DiNomaly2 inference on the exact shared image list

1. Load the **existing** `variety_160cat/model.pth` once in `eval()` with `torch.inference_mode()`. No optimizer, gradients, writes to model parameters, or training call.
2. Reuse the model architecture, data normalization, input processing, feature comparison, anomaly-map generation, interpolation, Gaussian smoothing, top-1% scoring, and multiview aggregation from the original DiNomaly2 Variety evaluator. Preserve the original protocol in a named `original_d2_protocol` config.
3. Iterate in deterministic sorted order over `shared_test_manifest.json` records, **not** over the DiNomaly2 native test loader's full manifest. Ensure every sample processed is a manifest member and every manifest member is processed exactly once.
4. At **every view**, retain: `category`, `object_key`, `image_path`, camera/view ID, GT view label, raw anomaly-map score summary, top-1% image score, and GT-mask path.
5. At **every object**, reproduce the native DiNomaly2 Variety **pooled top-1% across all five views** score (rather than merely averaging the five view scores), with object label `max(view labels)`. Confirm exact pooling formula in original code.
6. Retain unnormalized (not per-image min-max visualization-normalized) **postprocessed 256 × 256 anomaly maps** for Task 02, preferably compressed per-view arrays with an index. If feasible, also preserve pre-smoothing maps, but label them distinctly. Reprojection/GT masks must match the model's map coordinate frame; save transforms.
7. Do **not** normalize heatmaps independently for quantitative cross-model comparisons. For cross-model comparisons use within-model ranks, calibrated thresholds, scale-normalized diagnostics, or ROC/AP, not direct raw score magnitude differences.
8. Save outputs to a **new** timestamped directory, e.g. `out/re_eval_shared_20261031_<runid>/`. On any existing output, create another directory; never overwrite checkpoint/results.

## 4. Metrics: identical sample membership; explicit metric definitions

For each paired category calculate and save:
- **Image/view**: I-AUROC, I-AP, I-F1 (label F1 as `oracle_max_F1` *if* reproducing original test-set maximal-F1 convention; do not represent it as deployable/calibrated F1).
- **Pixel**: P-AUROC, P-AP, P-F1 (same threshold-convention note), P-AUPRO with the **same FPR integration cap**, interpolation, and integration algorithm as the original evaluator; explicitly record the cap (ArcAD script says 0.3).
- **Object**: O-AUROC, O-AP, O-F1 using correctly pooled five-view scores, and object labels.
- Confidence intervals can be generated in Task 02, not required for this reproduction.

**Critical comparability caveat:** Matched samples do not automatically guarantee *identical metric procedures*. The primary report should (a) reproduce each model's original metric protocol on common samples, (b) separately compute a common evaluation-protocol score from cached maps/scores wherever feasible. Document differences due to model processing, score aggregation, Gaussian smoothing, mask transforms, F1 threshold selection, and PRO integration. If same label/mask coordinate systems cannot be reconciled, do not compare pixel metrics as if identical.

## 5. Mandatory regression and integrity tests

- Checkpoint hash **before and after inference identical**.
- Existing checkpoint must not change modification time. Confirm number of loaded parameters and strict weight loading.
- Compare DiNomaly2 original native-test output to recorded reference numbers before restricting to common split (acceptable numerical tolerance must be declared in advance; e.g. AUROC ±0.0005). Known original I-AUROC: lattice `0.5423`, laser `0.6078`, power jack `0.6639`; these are **regression checks, not expected shared-test scores**.
- Compare manifest list to inference output with set difference and row count; assert no duplicates or missing files.
- Recompute selected scores independently from saved arrays (top-1% score, object pooling) and assert consistency.
- Check finite scores, maps of expected dimensions, nonempty GT for positive view, correct normal mask handling, and absence of training-object leakage.
- When making visualization, ensure correct camera view and GT alignment; no independently normalized heatmaps used for metric computation.
- Record deterministic seed and any unavoidable nondeterminism.

## 6. Deliverables

1. `run_metadata.json`: paths, checkpoint + SHA-256, git commits, env, preprocessing, map and mask transforms, metric variants, run command, timestamp.
2. `manifest_validation.json`: exact counts, shared/missing/extra/leakage reports, validation status.
3. `per_view_scores.csv` and `per_object_scores.csv`: one record per view/object with documented schema and unique keys.
4. `maps/` with raw numeric DiNomaly2 anomaly maps and a machine-readable index; no PNG-only substitute.
5. `dinomaly2_shared_metrics.json` and `dinomaly2_shared_metrics.csv`: each category and macro mean (macro = arithmetic category average; do not silently substitute pooled global AUROC).
6. `paired_comparison.csv`: one row/category, ArcAD shared metrics and DiNomaly2 shared metrics plus `D2 - ArcAD` deltas, only after ensuring matching evaluation/metric definitions. Record original native-test metrics in **separate** clearly labeled columns.
7. `reproduce.sh` or documented invocation plus `README.md` explaining what was rerun and what was intentionally unchanged.
8. `issues.md`: unresolved mismatches, code assumptions, limitations, and whether additional categories need the same treatment.

## 7. Completion criteria

**PASS** only when: all three categories are evaluated on exactly the 1,115 / 1,085 / 1,120 images; ArcAD labeled objects are excluded; pre-existing weights unchanged; original D2 baseline metrics reproduced on the original population; common-split metrics generated; and the paired report distinguishes any unresolved differences in metric protocols.

If resources are insufficient, complete one category correctly, report partial status, and do not invent the remaining scores. Do not retrain as a workaround.

## 8. Scope boundary

This task is **not** a new ArcAD training experiment and not a full 160-category redo. After the low-three paired setup passes, provide an optional documented extension plan to all 160 categories and `LED_indicator`. Full-population averaging or statistical claims should not mix original and paired splits.
