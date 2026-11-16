# Handoff prompt for the ARCAD Codex session

Please continue the Variety 1024 anomaly-detection comparison with ARCAD.

## Context from the Dinomaly2 run

- Dataset root: `/media/avlab/7458a474-6ea2-4b90-b232-373c3df8d329/real_iad_variety_1024`
- Dinomaly2 full-run checkpoint: `Dinomaly2/out/real_iad_variety_1024 out/full run/variety_160cat/model.pth`
- Dinomaly2 native Variety manifests are under `realiadvariety_jsons/Real-IAD_Variety_jsons/`.
- The three lowest final Dinomaly2 I-AUROC categories were:
  - `lattice_block_plug`: I-AUROC 0.5423, P-AUROC 0.9338, P-AUPRO 0.7080, O-AUROC 0.7604
  - `laser_diode`: I-AUROC 0.6078, P-AUROC 0.8352, P-AUPRO 0.7161, O-AUROC 0.7233
  - `power_jack`: I-AUROC 0.6639, P-AUROC 0.8991, P-AUPRO 0.6973, O-AUROC 0.7302
- These categories contain 224, 218, and 225 test objects, respectively (667 total; five views each).
- Dinomaly2 inference reproduced its final training-log metrics using 280×280 input/crop, 256×256 maps, top-1% scores, and the native 5×5 Gaussian smoothing.
- Ten representative Dinomaly2 triptychs (image, ground-truth mask, heatmap) are in `Dinomaly2/out/real_iad_variety_1024 out/full run/variety_160cat/review_low3/selected_examples/`.
- Metrics JSON: `Dinomaly2/out/real_iad_variety_1024 out/full run/variety_160cat/review_low3/metrics.json`.

## Requested ARCAD work

1. Inspect ARCAD's Variety 1024 training checkpoint, inference/evaluation code, and the cold-start manifests prepared for ARCAD. Do not assume Dinomaly2's native manifests are ARCAD's split.
2. Confirm which split the ARCAD checkpoint was trained/evaluated against, and identify its held-out test images for the same three categories. If it uses a different test split, report that clearly before making a direct metric comparison.
3. Evaluate the ARCAD checkpoint on those categories with ARCAD's own inference path and record category-level metrics. Use the same held-out images as Dinomaly2 if the manifests allow a valid match; otherwise explain the split difference.
4. Count distinct held-out objects per category. If practical, generate a balanced sample of ten low-performing defective objects across categories, using ARCAD's actual object score/evaluation semantics. For each selected object, save a comparison figure with the original image, ground-truth mask, and ARCAD heatmap.
5. Save the metrics, figures, and a short README in ARCAD's output folder, without overwriting existing results. State the image/map preprocessing and score definition so the results are reproducible.

Do not combine I-AUROC, pixel metrics, and object metrics into a single ranking unless there is a clearly justified weighting scheme. Keep the metric types separate and use I-AUROC to identify weak image-level categories, with pixel metrics as localization context.
