# Original Dinomaly-B training on Real-IAD Variety

The separate trainer is `costfilter-ad/Costfilter_Dinomaly/dinomaly_variety_train.py`. It uses the original 160-category normal training manifests from `realiadvariety_jsons/Real-IAD_Variety_jsons`, the original ViTill Dinomaly-B model, and the local DINOv2-R ViT-B/14 backbone. The image geometry is resize 448, center crop 392. The Variety paper specifies 50,000 steps for Dinomaly on this dataset ([paper, page 8](https://arxiv.org/pdf/2511.00540)).

The isolated environment is `.conda/envs/dinomaly_variety`, cloned from the working `cosfilterad` environment. From this workspace root:

```bash
DATA=/media/avlab/7458a474-6ea2-4b90-b232-373c3df8d329/real_iad_variety_raw
RUN=costfilter_setup/runs/dinomaly_variety_50k_20261116
.conda/envs/dinomaly_variety/bin/python -B costfilter-ad/Costfilter_Dinomaly/dinomaly_variety_train.py \
  --data-path "$DATA" --output-dir "$RUN" \
  --total-steps 50000 --batch-size 16 --num-workers 4
```

The trainer records each step in `step_losses.csv`, writes a resumable training checkpoint every 5,000 steps, and writes a plain `model.pth` at step 50,000. The plain checkpoint has the state-dict shape expected by the separate CostFilter Variety runner. To resume after interruption, pass `--resume-checkpoint "$RUN/trainer_step_N.pth"` with the same output directory and training settings. The trainer truncates the loss log back to the checkpoint step before continuing.

The baseline run should be evaluated on the same original Variety test split before comparing to a newly trained CostFilter model. The earlier CostFilter Variety run used a Real-IAD-trained teacher and is a separate cross-dataset transfer experiment.
