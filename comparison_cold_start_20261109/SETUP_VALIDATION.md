# Protocol B setup validation

The revised setup is validated with CPU-only synthetic data. The user has started real preparation with the prior serial script. Its cache was left untouched by this acceleration work. No model construction/weight loading, GPU model forward/backward, prototype fitting, training, inference, or benchmark evaluation was launched by this work. The 3,235-entry list was indexed from existing audit metadata, not from a new data-processing run.

Sixteen synthetic tests cover grouping, whole-object split isolation, deterministic/resumable normal streams, list-based B labels (including existing-empty versus absent masks), protected quality-flagged objects staying in test, and full preparation on invented pixels. The latter checks per-image zero `_mask_EMPTY.png` files at both resolutions, all listed views in test with image target 0 and pixel eligibility, no protected support views, source NG identity preservation, quality flags surviving overrides, C03 parsing, and raw/resized duplicate controls.

Metric tests check exact separated-score AUROC/AP agreement, tied/perfect curves, PRO endpoint integration, top-pixel scoring, and undefined one-class metrics. Error-analysis tests ensure thresholds reject source NG and test images, and verify true-normal versus missing-mask false-alarm counts/shares and masked-NG misses.

An additional synthetic reporting fixture exercises the six-run completion gate and rejects a missing pair or a wrong Protocol B label even if its CSV checksum is updated. It verifies primary I_B ordering, paired differences, and a calculated missing-mask false-alarm share in a temporary report. It runs no models and creates no real experiment results.

Python sources/tests are statically compiled under Python 3.8.12; `bash -n run.sh` and the no-stage plan-only launcher are checked. `logs/test_*.log` retains each synthetic run and exit status; earlier logs belong to the archived v1 design. The legacy NumPy/SciPy warnings do not indicate failed assertions.

An inventory-only review confirmed exactly 3,235 source-test NG/null-mask rows, including 33 in the copied audio-jack Pit block and four white images. Every source category has fully mask-present NG objects before quality filtering; final counts and support eligibility remain deferred to manual preparation.

The old unexecuted source/config/docs/lock are preserved in `provenance/revisions/pre_protocol_b_v1/`. The new lock pins the revised list, code, config, protocol, original model sources, audit metadata, and shared backbone. Original repositories and external audit/data files are unchanged. The preflight dependency-fix follow-up verified actual runtime imports for both repositories without natsort and added regression checks for the ArcAD name-only and Dinomaly2 WEIGHTS_DIR loader APIs. Full GPU model smoke checks and final real six-run aggregation remain manual.

## Faster preparation validation

The full invented-audit fixture is compared against the hash-pinned pre-optimization serial implementation. Records, category/split/decision manifests, decoded RGB pixels and binary masks match exactly, including grayscale/RGBA and JPEG-in-.png cases. Encoded byte hashes are independently checked on the new output. The test also verifies legacy PNG adoption without rewriting valid files, journal resume without any raw reads, selective regeneration of one damaged cache, recovery after an injected mid-run worker failure with no false completion marker, and rejection of a changed recipe. Additional tests cover raw-source changes and detection of another legacy prepare process.

The 20-image, 1536-to-392 synthetic warm-cache benchmark measured 1.75 s serial versus 0.48 s optimized (about 3.66x), with matching output pixel hashes. This uses invented RGB images in /tmp and excludes real-drive, full audit, journal and mask costs; it is not a promised whole-dataset speedup. Full evidence is in logs/prepare_synthetic_benchmark.json. The real fast preparation awaits the user's manual restart.

## ArcAD deterministic selection fix

The user completed preparation on NVMe and then hit PyTorch 1.12's unsupported deterministic CUDA gather backward in the ArcAD smoke test. The comparison adapter replaces that discrete-selection implementation with a no-grad candidate search and differentiable selected-candidate recomputation. Strict checks remain enabled in both smoke and fitting. The new CPU test compares four FP32/FP64 cases against the exact original pinned function, checking values, gradients, RNG consumption and strict repeated outputs. Targeted CUDA regression/smoke outcomes are written to logs/ with the active source-lock fingerprint. No real fitting or evaluation is launched. Prepared artifacts and their completion fingerprint are preserved.
