# Requested Protocol B revision

## Was the cold-start plan already using it?

No. The earlier **new cold-start setup** excluded missing-mask NG from the main image/pixel metrics and kept source object NG labels for its primary object result. It did not implement Protocol B as the main image table, synthetic EMPTY masks, or a distinct false-alarm bucket. The original Dinomaly2 loader did assign null-mask views image label 0, but that alone did not make the whole new experiment comply with these five requirements.

## Required behavior and implementation

1. **Zero masks.** The explicit source-test list is `config/protocol_b_ng_without_masks.csv` (3,235 unique IDs). Manual `prepare.py` will create one `_mask_EMPTY.png` per listed image at 392 and 256, with zero foreground. It preserves `raw_mask: null` and records `synthetic_empty_mask: true`; generated masks never make an image eligible for training.
2. **Main image table: Protocol B.** Evaluation rereads the frozen list and sets only those IDs to image target 0. Other source NG remain 1, including NG with an existing but empty mask. Main fields are `I_B_AUROC` and `I_B_AP`. `I_source_AUROC/AP` is the all-source-NG-positive sensitivity on exactly the same predictions/population; `I_visible` is explicitly supplementary. No PTB or other team's scores are imported as results of this experiment.
3. **Pixels.** Listed images are pixel-eligible with their generated all-zero masks. They contribute normal/background pixels to P-AUROC, P-AP, and P-AUPRO. Supplied misaligned/geometry-flagged masks remain excluded from pixel metrics under the independent audit policy; this never removes any of the 3,235 synthetic masks. Existing valid empty masks remain zero targets.
4. **ArcAD labeled split.** Every listed image remains test. To avoid same-object train/test leakage, every view of its object remains test too. Support candidates must be fully mask-present across all five source views and free of training-exclusion flags; at least one reliable nonempty mask must survive 392 resizing. Only those reliable views enter the support stream. Neither null-mask NG nor its generated EMPTY target enters normal training, prototype fitting, calibration, or supervised anomaly fitting.
5. **False-alarm analysis.** Unselected eligible official source-OK/train objects supply an independent shared normal-calibration pool. Per category/model/seed, use the empirical higher 99th percentile; alarm iff score > threshold. The configured target FPR is 1%, not a guaranteed test-set FPR. Separate `true_normal`, `ng_without_mask`, and `ng_with_mask` buckets. Save both false-alarm lists and the NG/no-mask share of Protocol B false alarms. A high response to a visibly defective, unannotated NG can reflect annotation problems; it is not automatically evidence of model failure.

The user's prior observation of roughly 40%+ missing-mask contribution to false alarms and example PTB scores motivate this accounting. They are not measured values or expected targets for this run. The actual contribution and both labeling-protocol scores remain pending until manual evaluation.

## Resolving conflicts with the old quality exclusions

“All 3,235 stay in test” is stricter than the earlier quarantine plan. All listed images' complete objects are protected test groups. They override content/blank/duplicate exclusions **for test membership only**, while preserving every quality flag. This list contains 33 copied audio-jack Pit views and four white views; silently applying the old exclusions would violate the requested count. These groups can never supply training/support/calibration. Any nonprotected duplicate counterpart is quarantined from fitting and test by the existing conservative duplicate rule. Repeated content may remain between protected test groups and is disclosed.

Do not call this test population fully clean, or numerically identical to a full 160-category benchmark. The category exclusions and cold-start support removal still apply elsewhere. Matching Protocol B is necessary but not sufficient before comparing with another paper/team. Confirm the same null-mask list, image labels, pixel targets, grouping, category population, split/support removal, preprocessing/scoring, and metric implementation.

## Revision provenance and execution state

The user explicitly requested this revision before execution. The previous source/config/docs/lock and status were preserved at `provenance/revisions/pre_protocol_b_v1/` before editing the active setup. A new input lock seals the revised setup. The archived version is historical and not the active run folder.

That initial delivery ran metadata indexing and CPU setup tests only. Subsequently preflight/import checks passed and the user began real serial preparation. The fast-preparation follow-up preserves that cache and adds a tested parallel/resumable path; the assistant has not started the new real-data preparation. Model GPU smoke checks, fitting, inference and final performance reporting remain manual.
