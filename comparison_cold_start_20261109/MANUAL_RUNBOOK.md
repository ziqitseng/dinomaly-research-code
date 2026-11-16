# Manual execution runbook — Protocol B

The original setup launched no model/data stages. The user has since started the serial preparation; switch it to the faster version using FAST_PREPARE_COMMANDS.txt. Model fitting/evaluation remain manual. Commands are provided for later manual use. `bash run.sh` defaults to printing a plan and exits without preparing data or loading models.

## 1. Review the frozen experiment

Read `PROTOCOL_B_REVISION.md`, `PROTOCOL.md`, and `DATASET_POLICY.md`. Check paths/budgets in `config/experiment.json`. The agreed setting is shared 30% normals plus one defective object per retained category for ArcAD. No warm-start checkpoint, old prototype, or generated `sup` manifest is used.

The explicit 3,235-entry `config/protocol_b_ng_without_masks.csv` is already indexed from audit metadata. Real EMPTY masks are generated as preparation advances; completion is checked using prepared/COMPLETE.json. Source code, metadata, this list, config, and the shared backbone are pinned in `provenance/input_lock.json`. The user-requested Protocol B revision of the unexecuted setup is archived under `provenance/revisions/pre_protocol_b_v1/`; the current lock supersedes it. Do not silently edit the active protocol after fitting begins. If a budget, environment-driven numerical choice, source correction, or repair policy must change, create a sibling experiment with a new lock before any fitting. Merely selecting a compatible `PYTHON_BIN` or CUDA device does not change model/data choices; preflight records the actual environment.

## 2. Check environment and resources

```bash
cd /media/avlab/1CE6451EE644FA12/2026_10_22_dinomaly/comparison_cold_start_20261109
bash run.sh preflight
bash run.sh test
```

The default interpreter is `/home/avlab/anaconda3/envs/arcad/bin/python`. Its inspected package metadata is Python 3.8.12, Torch 1.12.0+cu113, torchvision 0.13.0+cu113, timm 0.9.12, NumPy 1.21.6, SciPy 1.4.1, scikit-learn 0.22.2.post1, and Pillow 9.0.1. The code supports the older Pillow interpolation constants. Preflight imports the actual runtime components from both repositories in separate subprocesses and validates their differing backbone-loader signatures without constructing a model. Only dependencies used by this PNG comparison are required: natsort and tifffile belong to unused Dinomaly2 dataset/video paths and are not preflight requirements. Full GPU model compatibility is checked later by the explicit smoke commands.

If another **existing compatible environment** is desired, set the same interpreter for both methods and repeat preflight:

```bash
export PYTHON_BIN=/absolute/path/to/compatible/python
bash run.sh preflight
```

No package installation or download is automated. The same existing local ArcAD backbone file is used for both models and is hashed; Dinomaly2's separate backbone copy is not needed. xFormers is disabled, cuDNN benchmarking is off, deterministic algorithms are requested, and package/CUDA/GPU metadata are logged. Matplotlib/Torch caches are redirected inside `.runtime/` in this experiment.

Preflight requires 200 GiB free in the experiment filesystem. This is a conservative budget for the full RGB/mask cache, six model runs and resumable states, metrics, and logs. Full-map export is off. Turning it on would add roughly 47 GB of uncompressed float32 maps per ~180,000-view run before compression, across six runs, and requires a separately planned storage budget/configuration. Reserve approximately 16 GiB or more host RAM for audit joining and prototype clustering; actual peak GPU memory must be measured by both manual smoke checks rather than inferred from model names. No run-duration estimate is claimed before measuring this hardware.

Use the same GPU model for paired runs. Avoid competing jobs when assessing capacity or recording timing. The default is `cuda:0`; a later command can pass `--device cuda:1` where available.

## 3. Prepare the shared data once

```bash
bash run.sh prepare --resume --workers 4
```

If an old preparation is running, press Ctrl+C there and wait for the prompt first; keep prepared/. This is a CPU/data-I/O stage using four concurrent workers by default: it streams the audit inventories, verifies retained raw bytes against their hashes, decodes retained source images and usable masks, and creates 392 RGB and 392/256 binary-mask PNG caches. For every listed NG/null-mask image it writes per-image `_mask_EMPTY.png` targets containing only zero, preserves original `raw_mask: null`, and keeps the full object in test. Generated masks cannot qualify a support image. It does **not** change the raw or 1024 datasets and does not run a network. Using raw data avoids inheriting the 1024 conversion's independent stretching of mismatched masks. Retained raw reads can approach the original audit's hundreds-of-GB I/O scale.

Inspect these outputs before training:

- `prepared/summary.json`: source/retained counts, quality flags versus actual exclusions, protected-group override counts, exactly 3,235 NG/null-mask test views, partition totals, and object-only test groups.
- `prepared/category_counts.csv`: category-level training-normal/calibration/support/test/view/pixel coverage; maximum expected scope 159 categories.
- `prepared/decisions.csv` and `prepared/records.jsonl`: every source record's source split/class, camera/group, audit flags, metric eligibility, final partition, and cache hashes/paths. `partition=arcad_support` identifies reserved fully mask-present support objects. `partition=normal_calibration` holds only reviewed source-OK/train objects, disjoint from fitting and test. A protected test row may retain nonempty `exclude_reasons` as quality flags; `partition` is authoritative. Also inspect `prepared/protocol_b_ng_without_masks.csv` for every generated mask path.
- `prepared/cache_checksums.jsonl`, `prepared/COMPLETE.json`: cache checksums and immutable membership fingerprints.
- `prepared/progress.json`, `prepared/.prepare_journal.sqlite3`: live progress and per-image resume state; these are not frozen artifacts. `prepared/preparation_runtime.json` records final worker settings and work counters.

The exact retained counts remain pending until this stage runs. Any listed NG/null-mask image missing from test is a hard failure. The null-mask groups explicitly override test quality exclusions while keeping flags. A category with insufficient valid normal/calibration/fully annotated support/test data triggers failure; the script does not silently relax quarantine or drop that category. Partial `prepared/` output has no valid completion marker. Keep the incomplete folder and use --resume. Valid legacy PNGs are reused after pixel verification; the old version still requires source decoding once to identify its cached outputs. Future resumes validate source metadata and cache hashes against a per-image journal and skip raw decoding for completed entries. Damaged cache files are selectively repaired. A process check and advisory lock reject concurrent preparation writers. See FAST_PREPARE.md for progress, compression, independent cache verification and operational details.

## 4. Manually validate both GPU paths

```bash
bash run.sh smoke arcad --device cuda:0
bash run.sh smoke dinomaly2 --device cuda:0
```

These commands use synthetic tensors and random unit prototypes only, run two optimizer updates at the configured 16-normal/2-support batch budget, then check five-view inference. They measure allocated peak memory and write `logs/smoke_*.json`. They do not read real training images, fit real prototypes, or produce benchmark scores. A smoke pass does not validate convergence or semantic annotation quality.

The known PyTorch 1.12 CUDA gather-backward error in ArcAD is handled by the equivalent selected-candidate recomputation described in DETERMINISM_FIX.md; strict deterministic mode remains on. Expected xFormers-disabled warnings are harmless. Stop to inspect any other dependency, deterministic-kernel, nonfinite, or OOM error. Do not silently change just one method's batch size, image size, data policy, or numerical settings. A protocol change needs a new frozen experiment. Preflight and smoke are intentionally separate from the setup tests already run.

## 5. Train fresh paired runs

Run sequentially on the selected device, or schedule manually on equivalent GPUs with adequate resources. These are explicit launch commands, not commands executed by setup:

```bash
bash run.sh train arcad 1 --device cuda:0
bash run.sh train dinomaly2 1 --device cuda:0
bash run.sh train arcad 2 --device cuda:0
bash run.sh train dinomaly2 2 --device cuda:0
bash run.sh train arcad 3 --device cuda:0
bash run.sh train dinomaly2 3 --device cuda:0
```

Each run writes `runs/<model>/seed_<n>/`. The ArcAD command first fits that seed's normal-only prototypes using the initialized training model. Its `prototype_provenance.json` lists every normal view and patch count. Neither model evaluates test data during training. Each stops after exactly 100,000 updates and writes `TRAIN_COMPLETE.json` only after hashing the final model.

`training.jsonl` records finite losses, gradient norms, LR, sampled identities, and prototype norms where applicable. `initial_model.pth` records the actual fresh initialization; `latest.pth` atomically retains full state every 1,000 steps. `model.pth` is the final model used for evaluation. On interruption, preserve the directory and explicitly resume:

```bash
bash run.sh train arcad 1 --device cuda:0 --resume
```

The normal/support stream offsets and RNG states resume from the latest completed checkpoint. Uncheckpointed updates are replayed; later log entries from the interrupted segment may therefore repeat step IDs. Stage stdout logs remain separate. A completed run refuses another training invocation. A failure before the first resumable checkpoint requires preserving/moving that incomplete run directory before intentionally starting it again. No automatic output deletion is provided.

## 6. Evaluate every final checkpoint

```bash
bash run.sh evaluate arcad 1 --device cuda:0
bash run.sh evaluate dinomaly2 1 --device cuda:0
bash run.sh evaluate arcad 2 --device cuda:0
bash run.sh evaluate dinomaly2 2 --device cuda:0
bash run.sh evaluate arcad 3 --device cuda:0
bash run.sh evaluate dinomaly2 3 --device cuda:0
```

Each writes `runs/<model>/seed_<n>/evaluation/` with `view_scores.csv`, `object_scores.csv`, `category_metrics.csv/json`, and compressed `pixel_statistics/<category>.npz`. Evaluation rereads the frozen 3,235-entry list and uses Protocol B image targets. Per-view rows preserve source object target, B image target, missing-mask bucket, visibility-qualified auxiliary eligibility, pixel eligibility, score, threshold/alarm, and quality flags. Object rows use original source status and are supplementary. All listed zero masks enter pixel statistics.

The same evaluation stage first scores only `normal_calibration` source-OK views, fits each category's higher empirical 99th percentile (target FPR 1%; not a guaranteed test FPR), and freezes it before test error accounting. It rejects source NG and source-test calibration inputs. Inspect `calibration_scores.csv`, `false_alarm_thresholds.json`, `error_analysis.json`, `true_normal_false_alarms.csv`, and `ng_without_mask_false_alarms.csv`. The two alarm CSVs retain source paths/flags for review. The missing-mask share of Protocol B false alarms is measured per run, not preset to a prior 40%+ example.

An evaluation `COMPLETE.json` certifies completion and records checkpoint/input/artifact hashes. An existing or partial evaluation directory is never overwritten; preserve/move it before a deliberate restart. Evaluation batches the five views of each object and streams per-category pixel sufficient statistics, limiting metric memory. No full map cache is generated by the default configuration.

## 7. Write the final comparison in this folder

```bash
bash run.sh summarize
```

Aggregation requires all six completed evaluations. It checks shared expected test identities, source/B labels, the exact 3,235 list and protocol hashes, pixel eligibility/denominators, category membership, data fingerprints, and training budgets. It writes:

- `results/RESULTS.md`: Protocol B main I-AUROC/AP table, pixel metrics including generated-zero targets, explicitly supplementary source/visibility/object results, seed SD, paired intervals, and limitations.
- `results/comparison.json`: all summary values, coverage/exclusions/quality overrides, Protocol B list/source fingerprints, skipped metric categories, paired seed differences, and per-run false-alarm bucket analysis.
- `results/per_seed_macro.csv`, `paired_category_differences.csv`: machine-readable paired results.
- `results/COMPLETE.json`, `STATUS.json`: final hashes and completion state.

Before this command completes successfully, `results/RESULTS.md` remains a “not run” placeholder. Training logs, final scores, dataset decisions, and provenance all stay under this experiment folder; synthetic tests use system temporary directories. `logs/<stage>_<timestamp>.log` captures each manual stage's stdout/stderr and exit code. Do not report a fabricated number, a smoke result, a partial run, or an old native score as this experiment's final result.

Before comparing external numbers, verify the other result uses Protocol B on the same null-mask identities and also match population, category exclusions, support removal, geometry, scoring and pixel implementation. Matching the name alone is insufficient. No provided PTB example is entered as a measured result here.
