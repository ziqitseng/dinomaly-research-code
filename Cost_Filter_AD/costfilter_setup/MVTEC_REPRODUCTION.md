# MVTec-AD training and checkpoint comparison

Completed successfully on 2026-11-13 on the RTX 3090. Training took 43 min
38 s for 17 epochs / 5,134 optimizer steps, with zero invalid-loss skips.
Both full evaluations covered all 15 categories and 1,725 test images.
The official checkpoint remained unchanged. The local final checkpoint
closely matched it: image AUROC 99.5826% vs 99.5944%, pixel AUROC 98.6323%
vs 98.6695%, and pixel AP 76.0673% vs 75.6309% (+0.4363 percentage points).
This is one seed-1 run. Full results are in
`runs/mvtec_reproduction_20261113/REPORT.md` and `comparison.csv`.

The run directory is `runs/mvtec_reproduction_20261113/` relative to this file.
`status.json` records stage completion and exact commands. Every stage has a
`console.log`; each evaluation also writes per-category `metrics.json`.

The sequence is:

1. Evaluate the published CostFilter MVTec checkpoint on all 1,725 test images.
2. Check two real training batches with MVTec images and DTD anomaly synthesis.
3. Train CostFilter for all 17 epochs in the released Dinomaly-MVTec script.
4. Evaluate the final epoch (epoch 16) on the same test set.
5. Write `comparison.csv` and `REPORT.md`, including differences from the
   authors' bundled MVTec evaluation log.

The fixed pretrained Dinomaly model is shared between both evaluations and
held frozen during CostFilter training. Training uses all 3,629 normal MVTec
training images, DTD textures, FP32, batch size 12, seed 1, Adam at 0.0001,
and the upstream scheduler/losses. The original architecture, augmentation,
class order, 448-pixel resize, 392-pixel crop, 0.05 blending coefficient,
image-score aggregation, and seven evaluation metrics are preserved.
The final checkpoint is chosen by the fixed training budget, not test scores.

The MVTec dataset is read from:
`/media/avlab/1CE6451EE644FA12/2026_10_22_dinomaly/dataset/MVTec-AD/mvtec_anomaly_detection`.
All 15 categories and every anomalous-image/mask pairing were checked.
Official DTD was downloaded from
<https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz>
and extracted to `../datasets/dtd/` relative to this file; all 5,640 images
are present.

Changes to upstream code expose GPU, dataset, DTD, checkpoint, output,
batch-size, seed, worker, epoch, and short-run options. Training and evaluation
outputs are isolated from the original checkpoint directories. The DTD file
list is cached at dataset initialization instead of enumerated for every
image; sampling order and anomaly generation are unchanged. CPU threads
are set to four. Evaluation uses the unmodified upstream metrics code.

The first real-data smoke run exposed an upstream DTD augmentation dtype
bug: textures were converted to float32 before PIL-based augmenters such
as Autocontrast, which accept uint8 only. Textures now remain uint8 through
augmentation and are converted to float32 afterward. All ten augmenters
passed checks on both real texture and structure inputs (20 cases).
The failed smoke log is retained under `smoke_failed_attempt_1/`.
This training-only fix does not affect the completed official evaluation.

To repeat with a fresh output directory, from the workspace root:

```bash
conda activate cosfilterad
python -B costfilter_setup/run_mvtec_reproduction.py \
  --output-dir costfilter_setup/runs/mvtec_reproduction_repeat
```

The driver refuses an existing run status file unless `--resume` is explicit.
Resume keeps completed stages and preserves failed-stage outputs before
retrying. It verifies that the published CostFilter checkpoint's SHA-256
is unchanged after the comparison. The two-batch smoke checkpoint is separate
from the full training run and is not used for either evaluation.
