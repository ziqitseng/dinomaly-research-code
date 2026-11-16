# Inspection findings and motivation

Inspected on 2026-11-09: `ArcAD/`, `Dinomaly2/`, the raw integrity audit, and its `run_impact_20261107/` follow-up. Evidence describes this local dataset copy; no claim is made about a repaired upstream release.

## Why the saved runs cannot answer this question

The impact audit identifies the completed ArcAD `arcad_realiad_variety_100k_numerics` run and Dinomaly2 `variety_160cat` run. Their training populations were 6,785 and 19,955 views, and their evaluation populations were 192,165 and 178,995 views. ArcAD used a generated 30%-normal-plus-one-defective-object split and moved unused official normal training objects into evaluation. Dinomaly2 used the official training/test split. Their image transforms also differed: ArcAD 448 resize/392 crop, Dinomaly2 full 280 resize. The old scores therefore combine model, supervision, data membership, crop, and resolution effects.

The audit traced 45 distinct nonblank image contents crossing training and evaluation in ArcAD through copied `button_battery_holder`/`fuse_holder` content. Both runs encountered the copied audio-jack Pit block, white images, questionable annotations, and contradictory duplicate targets. The old checkpoints cannot have their training exposure undone by inference-only filtering. New training and prototype fitting are necessary.

## ArcAD implementation

- `arcad_realiad_uni.py` uses `models.uad.ViTill`, the register-token DINOv2 base encoder, eight decoder blocks, a noisy `bMlp` bottleneck, 500 prototypes, a discriminator, and reconstruction/clustering/classification/calibration losses.
- The encoder layers are frozen for reconstruction. The adapter explicitly freezes the entire encoder and holds it in evaluation mode for both methods.
- `RealIADVarietyDataset` reads `realiadvariety_jsons/sup/`. It resolves nested category roots and converts images to RGB. Its manifest `label` governs the view label; a missing mask otherwise yields a zero target. The existing split generator can assign null-mask NG views label zero, including in training.
- The current local trainer requires prototypes rather than silently proceeding with zeros. It includes log-space Sinkhorn and a positive linear-attention denominator floor. Those numerical fixes are preserved.
- The standalone `gen_protos.py` creates a separate random model and extracts normal bottleneck features. The new adapter fits prototypes inside each training process using its actual initialized model, avoiding a possible coordinate mismatch between independent random bottlenecks. No old prototype tensor is reused.
- Native final evaluation calls `evaluation_batch` on reconstruction maps. The imported discriminator fusion evaluator is not the active final scoring path. The new primary score likewise uses reconstruction maps; it does not introduce unvalidated discriminator fusion or test-dependent per-image normalization.

## Dinomaly2 implementation

- `dinomaly_multiview.py` constructs `models.uad.Dinomaly` using a 768→256→3072→768 noisy bottleneck, eight decoder blocks, two layer-fusion groups, linear attention, context-aware recentering, and the hard-mined cosine reconstruction loss.
- Training is normal-only. Native multi-view evaluation flattens views through the model, then aggregates score maps. Five cameras do not imply a different jointly conditioned multi-view network here.
- `RealIADDatasetv2` changes an NG view's label to false when its mask is null, and native object targets are the maximum view label. This can make a fully unannotated defective object a normal scoring target. The revised main image table now follows the user-specified Protocol B list (missing-mask NG target 0). Source object labels remain independent for explicitly supplementary O_source metrics, and false-alarm thresholds exclude every source NG. The older new-plan exclusion approach has been superseded; see PROTOCOL_B_REVISION.md.
- Both repositories' native evaluators form cosine-discrepancy maps, interpolate, smooth with a 5×5 Gaussian (sigma 4), and use top-1% scores. Their pixel evaluation implementations differ; the new shared metric code removes that difference.

## Audit scope and evidence

The raw audit inventories 160 categories, 198,950 images, 39,790 five-view groups, 155,810 masks, and 155,811 annotation JSONs. It fully decoded all images and masks and found no unreadable or actually missing referenced files after resolving nesting. It found zero official group crossings between training and test, but this establishes neither near-duplicate absence nor physical-instance independence.

The inspection relies on the original [audit report](/media/avlab/7458a474-6ea2-4b90-b232-373c3df8d329/real_iad_variety_raw/integrity_audit/REPORT.md) and impact follow-up; self-contained text copies and relevant small tables are saved under `provenance/audit_snapshot/`. Large `manifest_inventory.csv` and `file_checks.jsonl` remain at their external paths and are pinned by SHA-256. Preparation verifies retained source image bytes and usable mask bytes against that audit before creating the common cache.

These audit findings motivate quality flags and conservative fitting exclusions, with the explicit user-required protected-test exception for all 3,235 missing-mask NG views. The resulting subset is not evidence that all remaining annotations are semantically correct. It also is not numerically interchangeable with the original 160-category benchmark.
