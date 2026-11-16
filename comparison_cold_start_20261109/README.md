# Real-IAD Variety: matched cold start — Protocol B

**The SSD preparation is complete; real model training/evaluation remain unstarted.** Its completion marker and fingerprints remain unchanged by the ArcAD deterministic-backward fix. See [DETERMINISM_FIX.md](DETERMINISM_FIX.md). Synthetic smoke checks validate runtime compatibility only; they are not benchmark results.

The earlier unexecuted plan did **not** use the requested scheme: it omitted NG/no-mask views from the main image/pixel metrics, allowed partially annotated support objects, and had no dedicated false-alarm bucket. This folder has been revised to the user's Protocol B requirements. The previous setup and input lock are archived in `provenance/revisions/pre_protocol_b_v1/`; this current setup supersedes it.

## Protocol B now specified

| Component | Current behavior |
|---|---|
| Explicit list | `config/protocol_b_ng_without_masks.csv`: exactly 3,235 unique source-test NG IDs; hashed and read again at scoring time |
| Preprocessing | Per listed image, write a matching all-zero `_mask_EMPTY.png` at cached-image and metric resolutions, inside this experiment |
| Main image metric | `I_B_AUROC` / `I_B_AP`: listed NG images target 0; other source NG images target 1; source OK target 0 |
| Main pixel metric | All 3,235 synthetic zero masks contribute background pixels; valid existing empty masks also stay zero |
| ArcAD support | No listed view or other view of its object enters training. Choose one nonprotected, fully mask-present defective object per category; train only on its reliable nonempty-mask views |
| False-alarm threshold | Per-method/seed/category empirical 99th-percentile threshold from reviewed source-OK/train calibration views, held out of fitting; never calibrate with source NG or test data |
| Error analysis | Separate source-OK false alarms, NG/no-mask Protocol B false alarms, and masked-NG misses; measure the missing-mask share from actual predictions |
| Source identity | Preserve `object_label`, source class, and audit flags separately from Protocol B scoring labels |

## Matched model comparison

Both methods get the same `max(1, floor(0.30 × eligible official normal training objects))` per category, all five views, with identical ordered normal batches of 16 at every update. Remaining eligible official training normals are **normal calibration**, not model training or the main test set. ArcAD additionally receives two eligible support views per update from its one selected defective object per category. All support views are excluded from both models' tests.

Use the same local frozen DINOv2 ViT-B/14 register backbone, RGB full-image 392×392 inputs without cropping, 256×256 maps, 100,000 updates, split seed 1, and training seeds 1/2/3. Both architectures are fresh; ArcAD prototypes are fitted from the same actual initialized training model and selected normals only. Native architecture/loss differences and ArcAD's additional supervision remain explicit. This is not an architecture-only or equal-label-budget comparison.

## Audit handling and the mandatory test exception

The prior quality controls remain for normal training/calibration, support, and nonprotected test objects: exclude `fuse_holder`, wrong audio-jack Pit content, white-image objects, the tire conflict, and remaining raw/resized duplicates. **The new requirement to retain every one of the 3,235 views takes priority in test:** their complete five-view groups remain test even if quality-flagged. Flags are retained, not described as repaired or clean. Nonprotected duplicates of these inputs are quarantined, preventing known duplicate input from entering training/calibration/support.

This exception matters: the list includes 33 views in the copied audio-jack Pit block and four white images. The main result therefore includes documented annotation/content problems. Other category/exclusion choices and geometry rules still differ from the full official benchmark. Matching Protocol B alone is insufficient for external numerical comparison; also match splits, support budget, preprocessing, category population, exclusions, and pixel metric implementation.

## Documents and artifacts

- [PROTOCOL_B_REVISION.md](PROTOCOL_B_REVISION.md): direct answer to each new instruction and changes from v1.
- [PROTOCOL.md](PROTOCOL.md): complete sampling, fitting, metric, calibration, and reporting protocol.
- [DATASET_POLICY.md](DATASET_POLICY.md): all audit issues, priority rules, and repair roadmap.
- [MANUAL_RUNBOOK.md](MANUAL_RUNBOOK.md): explicit commands for later execution.
- [INSPECTION.md](INSPECTION.md): original model/audit inspection context.
- [results/RESULTS.md](results/RESULTS.md): future final report, currently NOT RUN.

Scripts provide parallel/resumable CPU preparation with a per-image journal, separate-process architecture adapters, resumable training, shared evaluation, source-OK threshold calibration, three error buckets, final paired aggregation, preflight, and manual synthetic GPU smoke checks. `provenance/input_lock.json` seals current code/config/list/evidence/backbone. `logs/` retains CPU setup tests and later manual stage output.

`bash run.sh` prints the plan and exits. Default Python is the existing ArcAD environment; runtime repository imports and both backbone-loader signatures are now checked by preflight; full model/GPU smoke checks remain manual. Original repositories, raw images/masks/manifests, previous run results, and external audit files are untouched.
