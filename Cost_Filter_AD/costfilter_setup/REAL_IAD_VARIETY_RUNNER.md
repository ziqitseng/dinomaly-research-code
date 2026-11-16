# CostFilter on Real-IAD Variety

The separate runner is `costfilter-ad/Costfilter_Dinomaly/costfilter_dinomaly_variety_uni_train_test.py`. Run the commands below from this workspace's root with the `cosfilterad` environment. It reads the **original** 160 category JSON manifests under `realiadvariety_jsons/Real-IAD_Variety_jsons`, not the ArcAD split. It trains a 160 class CostFilter head on the normal training views with DTD synthetic anomalies, then evaluates the original test split. The existing 30 class Real-IAD runner and checkpoints are separate.

The original manifests contain 19,955 training views and 178,995 test views. Each test object has five views. Three categories have a repeated category folder, which the loader resolves. The manifest paths have been checked against all images and declared masks. There are 3,235 anomalous test views without a mask path. Those views remain in image and object metrics and are excluded from pixel metrics. Per-category output reports the count.

## Teacher checkpoint

`--dinomaly-checkpoint` must be an **original ViTill Dinomaly-B** checkpoint compatible with the Real-IAD model architecture. `--teacher-description` records its training dataset in outputs. The existing `saved_results_realiad/realiad_vitb_it100k.pth` works as a cross-dataset transfer teacher and was used for the pipeline smoke test. It is **not** a Variety-trained teacher. The local Dinomaly2 Variety checkpoint is architecturally incompatible. A matched Variety benchmark needs a Variety-trained original Dinomaly-B checkpoint (or training that model first).

## Train

```bash
VARIETY_DATA=/media/avlab/7458a474-6ea2-4b90-b232-373c3df8d329/real_iad_variety_raw
TEACHER=costfilter-ad/Costfilter_Dinomaly/saved_results_realiad/realiad_vitb_it100k.pth
RUN=costfilter_setup/runs/realiad_variety_transfer

.conda/envs/cosfilterad/bin/python -B costfilter-ad/Costfilter_Dinomaly/costfilter_dinomaly_variety_uni_train_test.py \
  --train --data-path "$VARIETY_DATA" --dtd-path datasets/dtd/images \
  --dinomaly-checkpoint "$TEACHER" \
  --teacher-description 'Real-IAD Dinomaly-B cross-dataset transfer' \
  --output-dir "$RUN/train" --epochs 17 --batch-size 12
```

Training saves `step_losses.csv`, `training.json`, a protocol record, and a checkpoint after each epoch. `--resume-checkpoint` resumes at the next epoch with the same teacher, manifests, category set, and training recipe. Use a new output directory for a different teacher or experiment.

## Evaluate

Set `FILTER` to the checkpoint to test. The command computes baseline and CostFilter image, five-view object, and pixel metrics, plus per-view and per-object score tables. `--pixel-backend adeval` uses streaming pixel accumulation for the full test set; it reports an approximate maximum pixel F1. `--pixel-backend exact` uses exact pixel metrics and can require substantial RAM on full categories.

```bash
FILTER="$RUN/train/epoch_16_step_REPLACE_variety.pth"
.conda/envs/cosfilterad/bin/python -B costfilter-ad/Costfilter_Dinomaly/costfilter_dinomaly_variety_uni_train_test.py \
  --data-path "$VARIETY_DATA" --dinomaly-checkpoint "$TEACHER" \
  --teacher-description 'Real-IAD Dinomaly-B cross-dataset transfer' \
  --costfilter-checkpoint "$FILTER" --output-dir "$RUN/eval" \
  --pixel-backend adeval --eval-batch-size 8
```

`--resume-eval` skips completed categories in an interrupted evaluation. `--baseline-only` evaluates the teacher without a CostFilter checkpoint. For a short check, use `--categories wireless_receiver_module --eval-objects-per-label 1 --metric-size 64` with a new output directory; its results are not a benchmark.

## Verified smoke run

`costfilter_setup/runs/realiad_variety_smoke_20261115` contains a one-step, batch-two training checkpoint, step loss, paired inference for ten test views, and a separate ten-view check with one anomalous view lacking a mask. Both inference checks completed on the local RTX 3090. The one-step metrics have no statistical or benchmark meaning.
