# Dinomaly2 Real-IAD inference

`dinomaly2_inference.py` loads the trained Real-IAD ViT-B/14-register checkpoint
and evaluates the Real-IAD **test split** from the category JSON files. The test
split contains the normal and defective samples and their masks; the training
split is not evaluated.

Activate the local environment from the workspace root, then run from
`Dinomaly2/`:

```bash
conda activate ../dinomaly2
python dinomaly2_inference.py \
  --data-path ../dataset/Real-IAD \
  --checkpoint "out/real_iad out/train/model.pth" \
  --output-dir "out/real_iad out/inference_metrics"
```

Defaults match the training log evaluation: 448 resize, 392 center crop, 256×256
maps, top **1%** of map pixels for image and object scores, and 5×5 Gaussian
smoothing with sigma 4. The script calls Dinomaly2's own multi-view evaluator,
including its pixel metrics and Real-IAD five-view object aggregation. The
output is a category-by-category metric summary and an unweighted category mean,
as in the training log. Adjust `--score-top-percent` or `--map-size` to explore
other settings.

The only output artifact is `metrics.json`; no per-image scores or anomaly-map
images are written. This log-compatible evaluator requires CUDA.

The evaluator also supports Real-IAD Variety. To evaluate the 100-iteration
`2pin_block_plug` smoke checkpoint against only that category's held-out test
split, use:

```bash
python dinomaly2_inference.py \
  --dataset-type realiad-variety \
  --data-path /media/avlab/7458a474-6ea2-4b90-b232-373c3df8d329/real_iad_variety_1024 \
  --category 2pin_block_plug \
  --checkpoint "out/real_iad_1024 out/2pin_block_plug_100iter_rerun/model.pth" \
  --output-dir "out/real_iad_1024 out/2pin_block_plug_100iter_rerun/evaluation"
```

## Inspecting one test view

Use `dinomaly2_visualize.py` to display a selected test view beside its nearest
normal training image, ground-truth mask, and anomaly heatmap. First list the
test objects for a category, then select an object and view index:

```bash
python dinomaly2_visualize.py --category usb --list
python dinomaly2_visualize.py --category usb --object-index 0 --view-index 0 \
  --save-path "out/real_iad out/visualizations/usb_0_0.png"
```

The nearest training image is selected by cosine similarity between pooled,
layer-fused encoder features. The heatmap is normalized to its own 0–1 range
for display; it is an anomaly score map, not a calibrated defect probability.
The reported top-pixel score uses the same 256×256 map, Gaussian smoothing,
and top 1% default as the evaluation script.
