# Real-IAD CostFilter runner

The runner is `costfilter-ad/Costfilter_Dinomaly/costfilter_dinomaly_realiad_uni_train_test.py`.
It reads the original `realiad_jsons/realiad_jsons/*.json` manifests. The
36,465 training views must all be normal; no ArcAD `sup` split is used. All
30 category IDs are stable and alphabetical, including `zipper = 29`.

The frozen Dinomaly-B teacher is the published Real-IAD checkpoint at
`costfilter-ad/Costfilter_Dinomaly/saved_results_realiad/realiad_vitb_it100k.pth`.
CostFilter starts with a fresh 30-class head. Its binary segmentation head,
448 resize / 392 crop, cost volume, loss weights, and 0.05 evaluation blend
follow the reproduced MVTec runner. The synthetic anomaly generator reuses
DTD textures and structure anomalies, with a full-image foreground mask
instead of MVTec category-specific foreground rules. The MVTec runner still
defaults to its original 15-class head and strictly loads its checkpoint.

From the workspace root, after `conda activate cosfilterad`:

```bash
python -B costfilter-ad/Costfilter_Dinomaly/costfilter_dinomaly_realiad_uni_train_test.py \
  --train \
  --data-path /media/avlab/1CE6451EE644FA12/2026_10_22_dinomaly/dataset/Real-IAD \
  --dtd-path datasets/dtd/images \
  --output-dir costfilter_setup/runs/realiad_full_01 \
  --batch-size 12 --epochs 17
```

Each epoch saves a CostFilter checkpoint with model, optimizer, scheduler,
random state, category mapping, manifest hashes, and teacher hash.
`step_losses.csv` records total, segmentation, and classification loss at
every optimizer step; `training.json` records mean loss and elapsed time per
epoch. Plot them with `python -B costfilter_setup/plot_realiad_loss.py
costfilter_setup/runs/realiad_full_01`. Existing checkpoint filenames are
never overwritten.
This is a new Real-IAD training recipe; 17 epochs over 36,465 views is much
more optimization than 17 MVTec epochs. Declare the actual training budget
before interpreting test metrics. Add `--resume-checkpoint PATH` with the
same output directory and recipe to resume after a completed epoch. Any
partial step-loss rows from an interrupted epoch are removed on resume;
mid-epoch optimizer resume is not implemented.

To evaluate a trained checkpoint and the matched frozen Dinomaly-B baseline
on all 30 categories in a single pass:

```bash
python -B costfilter-ad/Costfilter_Dinomaly/costfilter_dinomaly_realiad_uni_train_test.py \
  --data-path /media/avlab/1CE6451EE644FA12/2026_10_22_dinomaly/dataset/Real-IAD \
  --costfilter-checkpoint PATH_TO_TRAINED_CHECKPOINT \
  --output-dir costfilter_setup/runs/realiad_full_01_eval \
  --eval-batch-size 8 --pixel-backend adeval
```

The runner scores the same views with the frozen Dinomaly-B teacher and with
CostFilter's 0.05 blend. Both branches use the same 256-pixel map resize,
top-1% view score, masks, and metric code. It records image AUROC/AP/F1max,
pixel AUROC/AP/F1max/AUPRO, sample AUROC/AP/F1max, and filtered-minus-baseline
deltas in `metrics.json`. Each category also gets `*_views.csv` and
`*_objects.csv`. Sample scores and labels are the maxima over their five
views, following Real-IAD's ADEval convention. The run checks original
manifest hashes, teacher hash, and category map against the filter checkpoint.
The `complete` flag is true only for all 30 categories with all test views.
The `adeval` pixel backend uses the official ADEval 1.1.0 streaming
accumulator with 10,000 score strips, avoiding multi-gigabyte per-category
map arrays. Its pixel AUROC/AP/AUPRO and `P_F1max_approx` are histogram
estimates; image and object metrics remain exact. A 10-view parity check
against the previous exact evaluator differed by less than 0.0006 in each
pixel metric. The default `exact` backend retains the original MVTec-style
per-category arrays and exact pixel F1, but is much more costly at full
Real-IAD scale. Use `--resume-eval` to skip saved categories after an
interruption.

For a teacher-only full run, omit `--costfilter-checkpoint` and add
`--baseline-only`. Add `--skip-pixel-metrics` when only image and five-view
object results are needed; this avoids retaining anomaly maps for pixel
metrics. Full 30-category evaluation has not been run yet.

To smoke test complete five-view objects, add `--categories zipper
--eval-objects-per-label 1` to the evaluation command. This selects one
normal and one anomalous object, including all five views of each. By
contrast, `--eval-per-label` selects individual views and suppresses sample
metrics when five-view groups are incomplete.

## Completed smoke test

- One GPU optimizer step on the 30-category training dataset: finite loss
  `9.217157`; checkpoint in `runs/realiad_runner_smoke_20261113/train/`.
- One GPU optimizer step with `--categories zipper`: labels `[29, 29]`,
  finite loss `8.874806`; checkpoint in `.../train_zipper/`.
- Filtered evaluation of two normal and two anomalous `zipper` views, with
  batch size 3 and a final batch of one, succeeded. Small-sample metrics are
  in `.../eval/metrics.json`; they are a plumbing check, not benchmark scores.
- Original MVTec filter checkpoint still loads strictly into the default
  15-class head. The Real-IAD model exposes 30 outputs.
- Paired evaluation on four individual views reproduced the previous seven
  filtered metrics to within 1e-9.
- Paired evaluation on two complete `zipper` objects (10 views) produced
  per-view/per-object score tables. Their max-view object scores and labels
  were independently checked. The teacher-only command also passed on these
  two objects. These small-subset scores are not benchmark results.
