# Matched cold-start protocol — v2, Protocol B

## Objective and source of authority

Compare the local ArcAD and Dinomaly2 implementations using the same limited normal pool, frozen backbone, image geometry, normal-update budget, and shared held-out test objects. ArcAD additionally gets one labeled defective object per category, as confirmed by the user. The user's subsequent Protocol B instructions replace v1's missing-mask handling. This is a practical comparison with different supervision budgets; it cannot isolate architecture alone.

Cold start means fresh anomaly heads and task-specific prototypes, not absence of external image pretraining. No old anomaly checkpoint, prototype, score cache, or generated `sup` split may initialize this experiment.

## Data membership, calibration, and support

1. Use the audited raw inventory and source byte checksums. Group by category/class/sample parent folder with separators intact; require exactly cameras 1–5. Parse C03/C04 as 3/4. Resolve actual nested roots without moving files.
2. Read the explicit `config/protocol_b_ng_without_masks.csv`. Its 3,235 IDs must exactly match source-test NG records whose original masks are null. A generated EMPTY mask never changes source-mask availability. Every such view and its complete five-view group is protected test data.
3. Apply the audit quarantine rules before fitting, with the explicit protected-test exception in `DATASET_POLICY.md`. Duplicates are checked globally across categories, both in raw hashes and final RGB/392 input pixels. Nonprotected counterparts of protected duplicates cannot enter fitting. Protected test groups retain quality flags even when excluded counterparts are removed.
4. Per category, use SHA-256 ranking of split seed/category/role/group to select `max(1, floor(0.30 × eligible official normal training objects))`. All five views enter `normal_train`, shared by both methods. Remaining eligible source-OK/train objects become `normal_calibration`, shared and disjoint from fitting and the main test. At least one calibration object is required. Calibration is never used for optimization, prototype fitting, or model selection.
5. Select one eligible source-test defective object per category for ArcAD. A candidate must have all five original masks present, no protected/null-mask view, no group exclusion flags, and at least one reliable nonempty training-resolution mask. ArcAD fits only its reliable nonempty-mask views; all five views are reserved from both models' tests. This draws no labeled anomaly from a null-mask object and prevents partial-object leakage. It is one object per category, not one per defect type.
6. Test is the eligible official test minus support, plus **all protected test groups**. No unselected training normals are moved into test. Object/view membership and labels are identical for both methods.
7. Fix split seed 1 and repeat training seeds 1/2/3. Same-seed pairs receive identical ordered normal batches. Repeats measure training variation on one fixed support choice, not support-selection uncertainty.

Expected scope remains at most 159 categories because `fuse_holder` has no null-mask entries and remains excluded. Exact post-audit counts are deferred until manual preparation. An inventory-only check found fully mask-present NG objects in every source category; final support eligibility after all quality controls is checked by preparation and must not be silently relaxed.

## Preprocessing and masks

Read each retained source payload once, validate those bytes against the audit, and decode from the same buffer. Work is parallelized across four CPU/IO threads by default. On resume, a completed journal entry is reused only if its recipe/audit/config/Pillow fingerprint, source identity/size/timestamps, and encoded-cache hashes match; otherwise that source is revalidated and decoded. Pillow decodes by content, converts all modes to RGB, and resizes the full image directly to 392×392 with bilinear interpolation. No crop or random augmentation. RGB conversion follows the native loaders, including RGBA conversion without invented compositing. This produces a common 28×28 patch grid for patch size 14.

Supplied reliable masks use nearest-neighbor resampling to 392 for training and 256 for metrics. For every listed missing-mask NG, create separate per-image files under `cache/empty_masks/<size>/<declared-parent>/<stem>_mask_EMPTY.png`, at both 392 and 256, containing only zero. Record their pixel checksums and generated status. Do not write beside raw source images or change source annotations.

All 3,235 generated masks must enter main pixel metrics, including quality-flagged protected views. Existing valid empty masks also remain zero pixel targets. A supplied mask with dimension/geometry defects remains pixel-ineligible; a nonempty supplied mask that disappears at 256 is excluded and logged. No guessed geometric repair, dilation, or mask reconstruction is performed. New cache PNGs use faster lossless compression level 1; valid old PNG encodings can be retained. Decoded pixels and pixel-hash filenames are identical to the serial recipe, although encoded byte hashes can differ. Encoded hashes are calculated from bytes written/read during processing; an independent final disk reread is optional with --verify-cache. Training/evaluation continue checking pixel hashes on load. Worker count and compression level do not alter the scientific recipe. Main image Protocol B labels depend on explicit list membership, not foreground size or pixel eligibility.

## Fitting controls

Both use the same local frozen DINOv2 ViT-B/14 with four registers and hashed pretrained weights. Keep the encoder in evaluation mode. Retain native bottlenecks, eight decoder blocks, two fusion groups, reconstruction loss, and Dinomaly2 context-aware recentering. ArcAD retains 500 prototypes, log-space Sinkhorn, denominator protection, synthetic noise 0.015, focal alpha 0.75, and clustering/classification/calibration weights 0.1. To support strict determinism in PyTorch 1.12, its adapter performs the discrete candidate search without autograd and recomputes the chosen normalized candidate with gradients; RNG consumption, selected outputs, and branch gradients are regression-checked against the original method. This avoids differentiable CUDA gather; it does not detach the synthetic loss or disable deterministic checks. Native ArcAD labels all patch features in an eligible defective training view as anomalous for its discriminator; masks guide spatial calibration.

Both train FP32, batch 16 normals, 100,000 normal updates, clipping 0.1, StableAdamW lr 0.002, betas (0.9,0.999), weight decay 0.0001, eps 1e-10. Warm up for 100 updates and cosine-decay to 0.0002; this deliberately harmonizes native schedule differences. Preserve native AMSGrad choices (ArcAD true; Dinomaly2 false). No AMP, test evaluation, test-tuned hyperparameters, or best-epoch selection during fitting.

ArcAD additionally consumes two eligible support views per update using an independent deterministic stream. Backpropagate normal and support losses before one optimizer step, freeing the normal graph first. Both methods therefore see exactly the same normals, while ArcAD uses extra support labels/compute. Do not claim equal FLOPs or runtime.

For each ArcAD seed, fit normalized MiniBatchKMeans prototypes from the **actual initialized training model**, with dropout off during extraction, using every selected normal view. Sample up to 128 spatial tokens per view, reduce that per-view quota uniformly to respect a one-million-patch cap, and exclude special tokens. Save initial model, prototype hashes, seed, normal IDs, and patch counts. No support/calibration/test features may fit prototypes. Reject zero, nonfinite, or non-unit prototype initialization.

Disable xFormers/cuDNN benchmarking, request deterministic CUDA algorithms, and record software/hardware metadata. Full checkpoints include optimizer/model/learner/discriminator, completed step, independent batch streams, and Python/NumPy/Torch/CUDA RNG states. Resume restores exact stream positions. Unsupported kernels or nonfinite values must fail; GPU behavior remains to be checked manually.

## Common inference and metric table

Use only the final fixed-budget model. For each fused group, compute `1 - cosine(encoder, decoder)`, bilinearly interpolate to 392 with align_corners=True, average groups, resize to 256 with align_corners=False, then smooth with a normalized 5×5 Gaussian (sigma 4, zero padding). No per-image normalization or discriminator-score fusion. A view score is the mean of its top floor(1% × 256²) pixels. The supplementary object score pools all five maps and averages the top floor(1% × 5 × 256²) pixels.

| Output | Population/target | Role |
|---|---|---|
| **I_B_AUROC / I_B_AP** | All retained test views; listed NG/null-mask=0, source OK=0, other NG=1 | Main image table, explicitly Protocol B |
| P_AUROC / P_AP / P_AUPRO | All 3,235 generated zero targets plus eligible supplied masks and true-normal zero masks | Main localization table, background from null-mask views included |
| I_source_AUROC / I_source_AP | Same test predictions; all source NG=1 | Sensitivity to treating every NG image as anomalous |
| I_visible_AUROC / I_visible_AP | Source OK plus NG views with reliable nonempty raw masks | Visibility-qualified supplementary subset; excludes missing/empty NG views |
| O_source_AUROC / O_source_AP | Original OK/NG object status, independent of annotation availability | Supplementary source-object result, not Protocol B image targets |

Evaluation rereads the frozen null-mask list and checks cached labels against it. It preserves source object status and class separately, so a fully unannotated defective object has five Protocol B image targets 0 but source object target 1. Existing mask files with zero foreground remain image-positive when the source is NG; they are not added to the 3,235 list.

Image/object AUROC/AP use exact scikit-learn routines. Pixel metrics use a common streaming 4096-bin histogram over [0,2], with ROC trapezoidal integration and AP from recall increments times precision. PRO averages recall per 8-connected foreground region and integrates background FPR [0,0.3], with an interpolated endpoint and division by 0.3. Zero masks contribute background but no foreground regions. Label pixel outputs as histogram approximations; neither repository's differing native pixel implementation is used.

Calculate per-category metrics and unweighted macro averages. Use the same defined categories for every method/seed and disclose undefined coverage. Do not replace one-class metrics with zero, combine detection/localization into a ranking, or tune F1 thresholds on test labels.

## False-alarm calibration and buckets

Before test error accounting, score only `normal_calibration` views. Require original source OK, source train, no quality exclusions, and no missing-mask NG; reject any other record. For each method/seed/category, take the higher empirical quantile at 0.99 (configured false_alarm_target_fpr=0.01); alarm iff `score > threshold`. This is an independent normal-only operating point, not a promise of 1% test FPR. Record threshold, calibration IDs/count, quantile rule, and observed calibration FPR.

At that frozen threshold, distinguish:

- `true_normal`: original OK test images; these define genuine-normal false alarms/FPR.
- `ng_without_mask`: original NG in the explicit list, target 0 only for Protocol B; report alarms/count/rate and their share of all Protocol B false alarms separately.
- `ng_with_mask`: original masked NG; report misses separately.

Never let the 3,235 relabeled negatives influence threshold fitting. Save sorted true-normal and missing-mask alarm CSVs for human review. Preserve source paths/classes and quality flags. A visible defect in the missing-mask bucket is annotation-ambiguous, not automatically a model error. The user's roughly 40%+ observation is prior context only; actual shares are calculated, not assumed.

## Completion, uncertainty, and external comparison

Save view/object scores, calibration scores, thresholds, error analysis, per-category metrics/counts, pixel sufficient statistics, logs, completion markers, and source/model hashes. Full float32 map export is optional and off by default. Final aggregation requires all six complete runs, exactly all listed test IDs with image label 0, correct protocol/list hashes, matched source/image labels, eligibility flags, test membership, update budgets, and all metric denominators.

Report each seed, mean/sample SD, paired differences, and a 10,000-draw paired category bootstrap (seed 20261109) after averaging training seeds. This interval is conditional on one split and does not establish physical-instance independence. Preserve per-run error-bucket statistics rather than pretending different methods share one numerical score threshold.

The main report names Protocol B. Before external comparison, verify the same missing-mask identities, labels, pixels, test/category population, source/support removal, geometry, score definition, and metric implementation. Different protocols can change the reported score; the user's example PTB numbers are not new measurements here. This audited subset with protected quality-flagged test objects is not identical to an unfiltered official 160-category benchmark. Prototype/cache costs and resumed timing segments must not be mislabeled as model-only runtime.
