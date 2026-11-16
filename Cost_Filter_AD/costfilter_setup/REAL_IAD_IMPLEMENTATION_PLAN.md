# Real-IAD integration assessment

Assessment performed on 2026-11-13 against the local CostFilter-AD checkout
and the extracted Real-IAD dataset. The initial runner, training adapter, and
30-class head were subsequently implemented and smoke tested. See
`costfilter_setup/REAL_IAD_RUNNER.md` for current commands and remaining
evaluation work. The file-level design below remains the reference for the
broader Real-IAD comparison.

## What already works

Dataset root:
`/media/avlab/1CE6451EE644FA12/2026_10_22_dinomaly/dataset/Real-IAD`.

The existing `Costfilter_Dinomaly/dataset.py:RealIADDataset` correctly reads
the original nested `realiad_jsons/realiad_jsons/<category>.json` layout on
this PC. It needs no image copying, folder conversion, or archive extraction.
CPU smoke checks loaded and transformed one normal training image, one
normal test image, and one anomalous test image/mask from every category:
30 categories / 90 successfully decoded examples. Image tensors were
`[3,392,392]` and test masks `[1,392,392]`. This was a sampled decode check,
not a fresh integrity scan of all 151,050 images.

Full metadata audit of the original split:

| Item | Count |
| --- | ---: |
| Categories | 30 |
| Normal training views | 36,465 |
| Anomalous training views | 0 |
| Training objects | 7,293 |
| Test views | 114,585 |
| Normal test views | 63,256 |
| Anomalous test views | 51,329 |
| Test objects | 22,917 |

All test objects have five views. Using the official sample-ID pattern,
every filename matched and no original-split training/test object IDs
overlapped within any category.

## Protocol must be explicit

The directory also contains `realiad_jsons/sup/`, used by the neighboring
ArcAD work. Those JSONs have `labeled` and `test` keys, not `train`/`test`.
They contain 12,156 labeled training views, including 1,216 real anomalous
views, and 138,894 test views. Object IDs span the two lists in all 30
categories, and some test objects have fewer than five retained views.
This is a different supervised/image-split experiment, not an interchangeable
replacement for the original normal-only, object-disjoint UAD protocol.

Decision: use the original normal-only split for the initial CostFilter
experiment, record exact JSON paths/hashes, and reject non-normal training
entries. This is the split used for the neighboring Dinomaly2 original run,
so the two runs can be compared on the same data protocol. Their model
architectures and training recipes still differ. If comparison with ArcAD's
custom protocol is wanted, define a separate named experiment rather than
silently switching manifests.

## File-level implementation

1. `Costfilter_Dinomaly/dataset.py` or a new `realiad_data.py`:
   add `RealIADCostFilterTrainDataset`. Read the normal training file list
   from JSON and reuse the repaired DTD/structure anomaly generator. Return
   the CostFilter training contract:
   `(image, synthetic_mask, has_anomaly, category_name, category_id)`.
   The existing loader returns only `(image, anomaly_label)` in training;
   its all-zero normal/anomaly label cannot serve as the 30-way category ID.
   Save a stable category-to-ID mapping with checkpoints. Keep sample/view
   IDs available to the evaluator. Share the anomaly generator instead of
   copying its implementation, and keep the reproduced MVTec behavior covered.

2. `unet3d_att_dino_channel_test_48_min.py`:
   expose `cls_classes=15` on `DiscriminativeSubNetwork_3d_att_dino_channel`
   and pass it to `DecoderDiscriminative_att_3D(..., cls_classes=cls_classes)`.
   Real-IAD uses 30; keeping the default at 15 preserves MVTec checkpoint
   key names and shapes. The binary segmentation head stays at two outputs.
   The DINO encoder and cost-volume filtering architecture need no redesign.

3. New `costfilter_dinomaly_realiad_uni_train_test.py`:
   connect the dataset, 30-class head, frozen target-trained Dinomaly,
   and existing CostFilter training/evaluation flow. Expose data/JSON/DTD
   paths, baseline/filter checkpoint paths, output directory, device,
   batch size, worker count, seed, epoch/step budget, and evaluation size.
   Preserve 448 resize / 392 crop, two fused 768-channel feature maps with
   28 x 28 patches, and the 768-channel matching-cost input initially.
   The current implementation hardcodes these dimensions; simply choosing
   a smaller crop is not supported. Save optimizer/scheduler/RNG state and
   periodic checkpoints so longer Real-IAD runs can resume exactly.

4. Training policies:
   make foreground-mask policy and hard-category gamma overrides explicit.
   The current foreground mode and difficult-class lists contain MVTec
   category names. Reuse the class-aware loss calculation, but do not
   implicitly apply those dataset-specific rules to Real-IAD categories.
   Add a generic policy (or training-data-derived configurable policy).
   The existing gamma helper tests the whole batch's `clsname` against a
   string list; a new configurable per-category rule must index each sample.
   Leave the reproduced MVTec path unchanged while introducing this policy.

5. `utils.py` / new `evaluate_realiad.py`:
   separate baseline-only evaluation from CostFilter evaluation. Existing
   `dinomaly_realiad_uni.py` calls `evaluation_batch(model, dataloader,
   device, ...)`, but `utils.py` now requires `(model, model_unet,
   dataloader, device, ...)`. The original baseline implementation is still
   in `utils_origin.py`; avoid confusing the two signatures. Existing
   baseline scripts also import unused `ptflops` and the unified one selects
   `cuda:1`. Remove the unused dependency or declare it, and use configurable
   `cuda:0` for this PC.

   Evaluate the same frozen baseline with and without filtering, preferably
   sharing each feature forward pass. Make smoothing, mask binarization,
   image-score aggregation, and metric resize identical and recorded.
   The inherited Real-IAD baseline uses `resize_mask=256`, whereas our MVTec
   experiment used 392; metric size is separate from the model's fixed crop.
   Preserve per-view labels and add object/sample metrics using the official
   five-view grouping and ADEval reference, without calling this multi-view
   feature fusion. The core model continues to process one RGB view at a time.

   Move predictions/masks to CPU per batch and evaluate per category, with
   disk-backed arrays when needed. All test prediction maps together at
   392 x 392 float32 would be about 65.6 GiB. Current evaluation already
   loops over categories, but retains each category's maps on GPU. The
   largest original test category has 4,470 views. Optimize repeated AU-PRO
   ground-truth region labeling only with numerical checks against the
   existing implementation and the official metric reference. Do not
   silently replace exact pixel metrics with approximate ones.

6. Tests and checkpoint metadata:
   record dataset protocol, JSON hashes, categories, baseline hash, image
   geometry, filtering settings, software versions, and training budget.
   The original split has about ten times MVTec's training images; copying
   17 epochs also multiplies the update budget by about ten. Support a fixed
   step budget and declare the chosen recipe before inspecting test scores.

## Checkpoint choice

The MVTec-trained Dinomaly and CostFilter weights can technically process
Real-IAD images at 392 pixels, but that measures cross-dataset transfer.
The current 15-way auxiliary classifier does not block inference because
the evaluation path discards its output. Target-dataset CostFilter training
does require the 30-way head and an explicit target-trained baseline.

The nearby checkpoint at
`Dinomaly2/out/real_iad out/train/model.pth` is not a drop-in original-Dinomaly
checkpoint. Its loaded state dict has bottleneck weight shapes
`[256,768]`, `[3072,256]`, and `[768,3072]`; the current ViTill baseline
expects `[3072,768]` and `[768,3072]` under different keys. The neighboring
training code also enables context-aware recentering. A Dinomaly2 adapter
would be a separately identified baseline integration.

For the closest extension of our reproduced pipeline, use the original
authors' published Real-IAD Dinomaly-B checkpoint. Their README lists
R448-C392, 100,000 iterations, and 89.7% image AUROC:

- Source: <https://github.com/guojiajeremy/Dinomaly#trained-model-weights>
- Checkpoint link: <https://drive.google.com/file/d/12EZTEtolifMRdpFM8Uoa1cLdVdmAbDSS/view>

The checkpoint is now downloaded to
`Costfilter_Dinomaly/saved_results_realiad/realiad_vitb_it100k.pth`.
All 276 keys strictly loaded into CostFilter's original ViTill architecture,
and a 392-pixel forward pass on the RTX 3090 returned finite 28 x 28 feature
maps. See `costfilter_setup/realiad_weights_manifest.json` for its checksum.
A baseline reference evaluation on the Real-IAD test set is still needed
before freezing it for CostFilter training. Train the new
CostFilter network from scratch for the first target-trained comparison;
any MVTec warm-start should be a separately recorded experiment.

## Validation order

1. Select and fingerprint the original normal-only manifests; sample-load
   every category and check object-disjoint splits and five-view groups.
2. The original Dinomaly Real-IAD checkpoint already strictly loads and
   passes a GPU forward check. Measure a baseline inference run on Real-IAD.
3. Verify the configurable class head still strictly loads the old 15-class
   MVTec weights by default. Run real Real-IAD training batches containing
   category ID 29, to catch an accidentally retained 15-class output head.
4. Exercise DTD and structure augmentation branches, normal/anomalous masks,
   batch-size-one evaluation tails, and sample grouping with mixed per-view
   labels. Check finite loss, gradients and optimizer updates on the 3090.
5. Check exact metric agreement on a small saved-map fixture, including
   object-level aggregation, before full-dataset evaluation.
6. Run one category end to end, then the declared 30-category baseline versus
   CostFilter experiment. Preserve the completed MVTec run as a regression
   reference; no full MVTec retraining is needed for these plumbing changes.

Official Real-IAD setup and sample metric reference:
<https://github.com/Tencent/AnomalyDetection_Real-IAD#3-evaluating>.

## Initial runner status

The original-manifest training adapter, 30-class head, and separate Real-IAD
runner are in place. One full-category-map optimizer step, one `zipper` (ID
29) optimizer step, a four-view paired evaluation with a batch-size-one tail,
and a two-object five-view paired evaluation completed on the RTX 3090.
Checkpoint saving and strict loading passed; the old MVTec 15-class checkpoint
still strictly loads by default. The evaluator now produces matched baseline
and CostFilter metrics plus ADEval-style max-over-five-view object metrics.
Full 30-category training and evaluation remain to be run before interpreting
the outcome as a Real-IAD benchmark.
