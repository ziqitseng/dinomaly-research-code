# Phase 2 implementation report

## Scope and files

The study is isolated at `/media/avlab/1CE6451EE644FA12/2026_10_22_dinomaly/convergence_study_20261112` on Ubuntu. The local editable mirror is this folder. The original `comparison_cold_start_20261109` experiment is read only. The study uses the same prepared RGB/masks and frozen input lock, and resolves model repositories and backbone paths relative to the original experiment.

Added/changed in the study: `run.sh`; `config/experiment_100k.json`, `config/experiment_50k.json`; `scripts/common.py`, `scripts/train.py`, `scripts/evaluate.py`, `scripts/checkpointing.py`, `scripts/evaluation_setup.py`, `scripts/validation_manifest.py`; `tests/test_infrastructure.py`; this report and `VALIDATION_PROTOCOL.md`. Unchanged copies of `model_adapter.py`, `metrics.py`, and `error_analysis.py` support the study evaluator. Other copied baseline files in the local mirror are reference material and are not exposed by the study runner.

## Training and milestones

The 100K study config enables full-state immutable snapshots at completed optimizer updates 20K, 40K, 50K, 70K, and 100K. The dedicated 50K config enables 20K, 40K, and 50K snapshots. A milestone is named `milestone_020000.pth`, etc. `latest.pth` retains the old 1,000-update replacement behavior. Snapshot creation uses a temporary file and atomic hard link; an existing target is fully loaded and compared rather than overwritten. Each full state includes model, optimizer, normal/support sampler state, learner/discriminator state for ArcAD, Python/NumPy/Torch/CUDA RNG, completed update count, config/prepared fingerprints, and scheduler formula metadata (total updates, warmup, last LR). Resume still uses `latest.pth` in the same run directory. The milestone save is after `optimizer.step()`, so its number denotes completed updates.

The 100K and 50K configs differ only in `total_steps` and `milestone_updates`. Both retain 100 warmup updates, architecture, cold-start samples, 392×392 RGB input, 256×256 map, seeds, and all other experiment settings. The prepared-cache fingerprint is checked against the unchanged original config; the study training config has its own fingerprint. Both configs use the original cosine formula, now in a tested helper.

| Completed update | LR, 100K schedule | LR, dedicated 50K schedule |
|---:|---:|---:|
| 100 | 0.0020000 | 0.0020000 |
| 20,000 | 0.0018295 | 0.0013814 |
| 40,000 | 0.0013797 | 0.0003726 |
| 50,000 | 0.0011014 | 0.0002000 |
| 70,000 | 0.0005717 | — |
| 100,000 | 0.0002000 | — |

**Experiment A** is the 50K milestone in a 100K schedule. **Experiment B** is the final 50K milestone in a dedicated 50K schedule. Their learning rates and training trajectories differ.

## Evaluation

`run.sh evaluate SCHEDULE MODEL SEED` retains the completed-run default: it requires `TRAIN_COMPLETE.json`, checks the final model hash, and writes the fixed `evaluation/` directory. With `--checkpoint PATH --output-dir NEW_DIR`, it accepts a full-state checkpoint before training completion, validates model, seed, step, config and prepared hashes, and refuses an existing output directory. `--split validation --manifest PATH` evaluates only selected validation objects; `--split test --manifest PATH` evaluates the corresponding revised final test. Without a manifest, `test` means the original full benchmark test. `COMPLETE.json` records model, seed, checkpoint path/hash/iteration, training config/hash/total steps, evaluation split, selection-manifest hash, and scoring metadata. The evaluator reuses the original map scoring, Protocol B labels, normal-only threshold fitting, five-view object pooling, category metrics, and pixel metrics. `metrics.py` is byte-identical to the original and F1 was not added.

See `VALIDATION_PROTOCOL.md` for object eligibility, population effects, and the fixed checkpoint-selection rule. The optional manifest generator is **inspect-only by default** and has not written a split.

## Lightweight validation performed

- Four CPU-only unit tests passed on Ubuntu: schedule/config scope; immutable checkpoint naming, save, reload, and mismatch refusal on the study filesystem; explicit checkpoint identity and output-dir isolation; deterministic split and protected partition behavior.
- `bash run.sh plan`, `bash run.sh train 100k --help`, and `bash run.sh evaluate 50k --help` passed. No training or evaluation was launched.
- The read-only manifest scan found 159 categories, 1,590 proposed validation objects, 33,744 remaining test objects, and all 3,235 protected missing-mask views retained in final test.
- Original `run.sh`, `scripts/train.py`, `scripts/evaluate.py`, `scripts/metrics.py`, and `config/experiment.json` SHA-256 values matched before and after deployment. The study `metrics.py` hash matches the original. ArcAD seed 2 was still running (PID 1747119) at the final process check; it was not touched.

## Future commands (manual; do not run during Phase 2)

From the Ubuntu study directory:

```bash
bash run.sh inspect-validation
mkdir -p manifests
bash run.sh inspect-validation --write manifests/validation.json

# Experiment A: new 100K trajectories with permanent intermediate states.
bash run.sh train 100k arcad 1 --device cuda:0
bash run.sh train 100k dinomaly2 1 --device cuda:0

# Experiment B: full 50K cosine trajectories.
bash run.sh train 50k arcad 1 --device cuda:0
bash run.sh train 50k dinomaly2 1 --device cuda:0

# A/50K validation (repeat for models, seeds, and predeclared milestones).
bash run.sh evaluate 100k arcad 1 \
  --checkpoint runs/100k/arcad/seed_1/milestone_050000.pth \
  --output-dir evaluations/validation/A/arcad/seed_1/step_050000 \
  --split validation --manifest manifests/validation.json

# B/50K validation.
bash run.sh evaluate 50k dinomaly2 1 \
  --checkpoint runs/50k/dinomaly2/seed_1/milestone_050000.pth \
  --output-dir evaluations/validation/B/dinomaly2/seed_1/step_050000 \
  --split validation --manifest manifests/validation.json

# After selection is frozen, evaluate chosen full-state checkpoints on revised final test.
bash run.sh evaluate 100k arcad 1 \
  --checkpoint runs/100k/arcad/seed_1/milestone_100000.pth \
  --output-dir evaluations/final_test/A/arcad/seed_1/step_100000 \
  --split test --manifest manifests/validation.json
```

Repeat commands for declared seeds 2 and 3 when GPU capacity is available; run only one heavy stage at a time on the shared GPU. An existing study run directory cannot be reused without `--resume`, and existing evaluation directories cannot be overwritten.

## Timing and limitations

At recorded single-run rates, a 100K ArcAD plus DiNomaly2 pair is about 19.5 + 16.8 = **36.3 GPU hours per seed**. The dedicated 50K pair is roughly **18.15 GPU hours per seed** if time scales linearly. Both conditions together are about **54.45 hours per seed**, or **163.35 hours for three seeds**, excluding evaluations and contention. A full evaluation was about **1.3 hours per checkpoint**; 18 such runs (100K, A/50K, B/50K for two models and three seeds) would be about **23.4 hours** at that rate. Additional 20K/40K/70K validation evaluations increase the total. Validation is smaller than the original full test but has not been timed; none of these are guaranteed runtimes.

No GPU checkpoint evaluation or full training was performed in Phase 2. The new checkpoints are available only from *future* study runs; old intermediate weights cannot be recovered. The proposed validation changes the final-test population, so original full-test 100K scores cannot be used as direct revised-test comparators. The fixed holdout is one split; conclusions remain conditional on that split and on the planned training seeds.
