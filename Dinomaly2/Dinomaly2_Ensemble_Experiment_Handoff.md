# Dinomaly 2 — Shared-Encoder / Multi-Decoder Experiment Plan

**Status:** Proposed research/engineering plan; no experiments have been run under this plan.  
**Prepared:** 2026-10-28  
**Primary benchmark:** Real-IAD (30 categories, 5 views/object).  
**Audience:** Codex CLI operating inside a locally checked-out Dinomaly 2 repository.

> **Codex instruction:** First inspect the local repository, environment, data, and available checkpoints. Do **not** assume the upstream preview repository exactly matches the paper. Make the smallest reversible modifications required. Run a baseline and smoke tests before training ensembles. Do not claim experimental improvements without measured results.

## 0. Executive brief

We want to test whether **diverse, independently trained bottleneck–decoder branches**, sharing **one frozen pretrained ViT encoder**, make complementary anomaly-detection errors on Real-IAD. The eventual hope is better defect detection/localization, particularly where the baseline misses certain defect types, **without tripling the encoder computation**.

The research question is *not simply* “Does a bigger ensemble improve AUROC?” Rather:

> Can we deliberately create **complementary reconstruction behavior** through different initialization, data selection, augmentation, selected encoder layers, and reconstruction capacity, and do these differences improve Real-IAD performance relative to a well-controlled single Dinomaly 2 model?

Separate the source of improvement from the source of compute overhead. Test both per-image and **five-view object-level** metrics; include anomaly localization and false positives. Three branches are the proposed initial size, not a fixed architectural requirement.

**Core architecture:**

```text
                       test image / training image
                                  │
                            FROZEN ViT
                       (single encoder pass)
                                  │
                 selected intermediate feature maps
                       ┌──────────┼──────────┐
                       ▼          ▼          ▼
                 bottleneck A  bottleneck B  bottleneck C
                 decoder A     decoder B     decoder C
                       │          │          │
                  anomaly map A  map B       map C
                       └──────────┼──────────┘
                                  ▼
                 aligned & calibrated map fusion
                                  ▼
                    image / view / object scores
```

**Important:** In Dinomaly 2, the encoder is frozen; **both the bottleneck and decoder are trainable**. Do not implement “three bottlenecks feeding one shared trained decoder” by accident. The proposed branches consist of **three independent bottleneck+decoder pairs**, unless a particular ablation explicitly specifies otherwise.

## 1. Sources and scope of verified information

### Papers supplied by the researcher

- Dinomaly, *The Less Is More Philosophy in Multi-Class Unsupervised Anomaly Detection*, CVPR 2025 / arXiv:2405.14325v5.
- Dinomaly 2, *One Dinomaly 2 Detect Them All: A Unified Framework for Full-Spectrum Unsupervised Anomaly Detection*, arXiv:2510.17611v2 (2025).

### Official code

- Dinomaly 2: https://github.com/guojiajeremy/Dinomaly2
- Original Dinomaly: https://github.com/guojiajeremy/Dinomaly

At handoff, the Dinomaly 2 repository README calls its release a **preview** and warns of possible bugs and configurations different from the paper. Pin the actual local Git commit and report any differences. The README lists:

- Real-IAD data directory: `../Real-IAD` with `realiad_1024` and `realiad_jsons` subfolders.
- Real-IAD entry point: `dinomaly_multiview.py`.
- Typical example: `python dinomaly_multiview.py --data_path ../Real-IAD --image_size 448 --crop_size 392 --save_name dinomaly2_realiad`.
- Other relevant files: `dinomaly_2D.py`, `dinomaly_fewshot.py`, `dataset.py`, `models/`, `utils.py`, `adeval/`.

The *paper* reports, among other settings, a DINOv2-with-registers ViT encoder, a noisy bottleneck, a transformer decoder, loosened reconstruction, and (for Real-IAD) five-view evaluation. The exact code behavior **must** be confirmed locally. Paper Real-IAD multi-class image-level AUROC: **92.1% for Dinomaly2-L** (Table 3); this is a *reference*, **not** a guaranteed reproduction target for a different model size/split/script. The same table reports **94.9% object-level AUROC** for Dinomaly2-L. Do not confuse image-level with object-level.

## 2. Hypotheses

- **H0 (control):** Three independent seeds with otherwise identical settings may improve stability or performance, but may make strongly correlated errors.
- **H1 (augmentation diversity):** Mild, *physically plausible* transformations of normal training images generate complementary decoder behaviors.
- **H2 (feature-depth diversity):** Different subsets of the *same frozen encoder's intermediate layers* capture different defect types (e.g., surface texture vs. component structure).
- **H3 (capacity diversity):** Decoders with different dropout rates or sizes balance normal reconstruction and copying anomalies differently.
- **H4 (data-sampling diversity):** Stratified bootstrap / subsampling of normal data generates useful disagreement, if it does not omit essential modes of normal variation.
- **H5 (fusion):** Calibrated fusion of sufficiently complementary anomaly maps outperforms the strongest individual branch on a locked test set.

**Potential failure modes:** identical errors across branches; increased false positives; unfavorable precision/recall trade-offs; too much dropout hurts normal reconstruction; augmentation erases tiny defects or makes impossible normal samples; averaging washes out a defect detected by only one branch; test-set tuning masquerades as gains.

## 3. Experimental design — staged and controlled

Use **E0 → E1 → E3** as the shortest initial path, followed by E2 and E4. E5/E6 are optional. Keep dataset splits, preprocessing, model size, training budget, checkpoint selection, and evaluation protocol identical unless the experiment explicitly changes them.

| ID | Branch A | Branch B | Branch C | Variable intentionally changed | Priority |
|---|---|---|---|---|---|
| **E0** | Official single Dinomaly 2 | — | — | None: reproduction baseline | **Required first** |
| **E1** | Seed 1 | Seed 2 | Seed 3 | Random initialization / stochastic training only | **High** |
| **E3** | Earlier-layer feature subset | Paper/default middle layers | Later-layer feature subset | Selected encoder layers | **High** |
| **E2** | Official/default transforms | Mild geometric transforms | Carefully chosen photometric or plausible orientation transforms | Training augmentation | Medium |
| **E4a** | Dropout 0.2 | Dropout 0.4 | Dropout 0.6 | Noisy bottleneck strength | Medium |
| **E4b** | Smaller decoder | Default decoder | Other constrained decoder | Decoder capacity | Optional |
| **E5** | Bootstrap sample A | Bootstrap sample B | Bootstrap sample C | Training sample exposure, stratified | Optional |
| **E6** | Alternative loss A | Official loss | Alternative loss B | Reconstruction objective | Later / research-intensive |

Values like **0.2/0.4/0.6** are *suggested experimental settings*, not paper optima. Note that the Dinomaly 2 paper reports default dropout 0.2 or 0.4 depending on dataset/search; verify the Real-IAD configuration in code. A branch that is much worse than baseline is not automatically useful to the ensemble.

### E0 — Reproduce the single-model reference

1. Inventory actual local dependencies, CUDA and GPU compatibility, dataset paths, checkpoints, config, commit hash.
2. Identify and run the official **Real-IAD multiview** training/evaluation path, **without modifying upstream logic** if possible.
3. Confirm which encoder (Base/Large), selected layers, learning rate schedule, dropout, losses, image preprocessing, training steps, seed, model-selection policy, evaluation and view aggregation are used.
4. Run a tiny forward/backward smoke test and a minimal evaluation before full training.
5. Record image-level, object-level, and pixel-level results; sample anomaly maps; runtime; peak VRAM; checkpoint path.
6. If reproduction is not comparable to a reported paper row, clearly explain why (e.g., ViT-Base vs Large, model size, image size, steps, view aggregation, preprocessing, version).

**Do not proceed to comparisons until E0 outputs have been independently sanity-checked.**

### E1 — Independent-seed ensemble (proper control)

- Use identical model architecture, dataset, preprocessing, optimizer, training schedule and augmentation policy.
- Vary only random seed, including trainable weight initialization and data-order/dropout RNG.
- Train three independent bottleneck+decoder pairs; verify encoder weights never update.
- Evaluate A/B/C alone, pairwise A+B etc., and A+B+C.
- Primary goal: estimate baseline ensemble gain and error correlation before introducing deliberate diversity.
- Ideally rerun a control ensemble for variance estimates, budget permitting.

### E3 — Diversity from selected intermediate ViT layers

This preserves **one ViT** and changes which of its intermediate activations each branch reconstructs. It tests the researcher's original intuition without adding 3 independent backbones.

- Inspect the actual encoder API and determine whether layer indices are 0- or 1-based; verify which layers the official implementation uses. The paper's Base example uses eight middle layers (often represented as 3–10 in 1-based numbering).
- Propose **three valid feature sets only after verifying code compatibility**. Example placeholders for a 12-layer backbone, **1-based notation**:
  - Earlier-leaning: `[1,2,3,4,5,6,7,8]`
  - Default: `[3,4,5,6,7,8,9,10]`
  - Later-leaning: `[5,6,7,8,9,10,11,12]`
- **These are examples, not validated settings.** Check feature extraction and whether final-layer/class-token/register-token behavior makes them unsuitable. Change to safe subsets if necessary; log exact indices.
- Ensure corresponding decoder targets and loose-reconstruction groups remain coherent; do not merely change an index list while retaining incompatible target alignment.
- Compare branches individually and fused. Analyze defect-type/category breakdown and whether predictions are meaningfully complementary.
- Note: selected layer groups overlap significantly; if there is little diversity, test less-overlapping sets as a follow-up, without silently changing primary E3.

### E2 — Data-augmentation diversity

- Feed all branches the **same underlying Real-IAD normal training split**, but use distinct train-time transform policies.
- A: official default (resize/crop or code equivalent).
- B: mild random translation and carefully bounded scale jitter; avoid cropping away most of an object.
- C: mild brightness/contrast changes **or** 90° rotations/flips **only where physically plausible**; do not mix both into one experiment if effects should be distinguished.
- Preserve original validation/test preprocessing and masks; no random test-time augmentation in the initial study.
- Important: augmented images are passed through the frozen encoder, and reconstruction targets derive from those *same augmented inputs*; do not reconstruct features of geometrically mismatched images.
- Dinomaly 2 explicitly uses stronger geometric augmentation in its *few-shot* regime; do not assume that is the default Real-IAD full-shot policy.
- Small scratches and fine defects may be destroyed by downsampling, blur, and large crops; avoid strong blur and aggressive rescaling at first.

### E4 — Regularization or capacity diversity

- E4a: independently train branches with different **bottleneck Dropout** values, otherwise equal; include official baseline value and monitor normal-image reconstruction.
- E4b (separate run): vary decoder width/depth or bottleneck dimensionality **one factor at a time**, only after E4a.
- Too much reconstruction ability may reproduce anomalies; too little may create false alarms everywhere. Neither extreme is guaranteed to help.
- Log normal reconstruction losses and defect/normal anomaly score distributions, not merely final AUROC.

### E5 — Stratified bootstrap / bagging (optional)

- Sample *normal training images only*, independently for each branch, with replacement or controlled subsampling.
- Maintain representation across product categories and, where possible, camera views; preserve group identifiers.
- Report effective unique sample counts and coverage. Do not bootstrap validation/test data or introduce leakage.

### E6 — Different losses (later)

- Explore variations on the original loose-reconstruction objective only after baseline and simpler diversity tests are stable.
- Change exactly one reconstruction-loss mechanism at a time (grouping, weights, hard mining, etc.).
- Test gradient behavior and explain the intended effect; avoid claiming a changed loss faithfully implements the paper if it does not.

## 4. Dataset integrity and evaluation protocol

### Real-IAD specifics

- Prioritize **full Real-IAD (30 categories; five views/object)** using the official supplied splits.
- Confirm how view IDs and object IDs are represented in the actual JSON; do not assume filenames alone correctly link views.
- **Never split by individual view when the same physical object can appear across views**; keep all views of an object together in any newly created train/validation subdivision.
- Train on **normal** training data only (unless a future separate supervised experiment is explicitly declared).
- Preserve official test data untouched; reserve a group-disjoint validation/calibration subset from the official training portion where feasible. If there is no admissible validation protocol for selecting fusion on abnormal samples, use simple predeclared fusion rules or a separately labeled development dataset, and explain limitations.
- Keep image/mask geometry consistent after crop/resize; track coordinate frame when saving heatmaps.

### Primary metrics

Report **both** standard category-average and pooled/unified results, and be explicit about which is which:

1. **Image/view level:** AUROC, AP, F1-max (threshold-selected metric; do not treat test-selected F1-max as deployment performance).
2. **Object level (five-view):** object AUROC and available object-level AP/F1; use the official view aggregation first.
3. **Pixel level:** AUROC, AP, F1-max, AUPRO where masks and official evaluator permit.
4. **Breakdowns:** per category, per defect type if metadata permits, per view, small-vs-large defect where defensibly annotated.
5. **Efficiency:** training GPU-hours, inference images/sec, time/object, peak VRAM, parameters and shared-encoder vs independent-model computation.

For Dinomaly 2, paper describes image score from mean of the top `z%` anomaly-map pixels (generally 1%; **Real-IAD uses 0.1% in paper implementation details**) and multi-view object scoring from aggregated maps. **Inspect code before assuming those formulas are implemented identically.** Reproduce official metrics first, then mark any alternative scoring as a separate experiment.

### Calibration and fusion

Keep the initial fusion deliberately simple and auditable:

- **F0:** Single-branch outputs (required baseline).
- **F1:** Arithmetic mean of aligned raw anomaly maps, **only if score scales are comparable**.
- **F2:** Calibrate each branch's scores/maps using **training-normal or held-out-normal data only** (e.g., fixed robust median/IQR per model), then average. Fit calibration without official test labels. Avoid per-test-image normalization for AUROC scoring.
- **F3:** Per-pixel max of calibrated maps (tests whether averaging suppresses branch-specific defects). Expect possible higher false positives.
- **F4 (optional):** Fixed weights selected on a separate admissible development set; avoid searching weights using final test anomalies.

Do not use GT anomaly masks or test labels to choose fusion/thresholds. Document if normal-only calibration cannot align abnormal score distributions reliably. Apply consistent aggregation from fused per-view maps to image/object scores. Report both fused maps and fused scalar-score variants if testing both.

### Complementarity analyses

- For each branch, save per-sample scores, heatmap paths, category/view/object metadata and labels (labels for *evaluation only*).
- Measure **error overlap** at a fixed, predeclared/calibrated operating point, not only AUROC: anomalies detected by only A, only B, only C, all, none; false-positive overlap on normal objects.
- Pairwise rank correlations of sample-level scores, pixel-map correlations (where useful), ensemble gain relative to **best** single branch and E0 baseline.
- Highlight visually verified cases where one branch detects a defect missed by another; include counterexamples where fusion makes things worse.
- If possible, provide confidence intervals by **resampling independent object groups** (not individual views). Use consistent groups across systems for paired comparisons.

## 5. Software implementation proposal (inspect existing APIs first)

**Do not rewrite the project from scratch.** Implement small wrappers around the official model/data/evaluator if possible.

Suggested modular shape (exact file names flexible):

```text
experiments/
  README.md
  configs/
    e0_baseline.yaml
    e1_seeds.yaml
    e3_layer_sets.yaml
    e2_augmentations.yaml
    e4_dropout.yaml
  train_branch.py            # train 1 independent bottleneck+decoder
  infer_branches.py          # one frozen encoder forward, N branch forwards
  fuse_maps.py               # alignment, calibration, fusion
  eval_ensemble.py           # official evaluator + grouped analyses
  validate_splits.py
  summarize_runs.py
  tests/
results/
  <experiment>/<branch_or_fusion>/
    config_resolved.yaml
    git_commit.txt
    environment.txt
    train_log.csv
    metrics.json
    sample_scores.csv
    object_scores.csv
    checkpoint.pt
    anomaly_maps/             # optional / compressed
```

**Design considerations:**

- Freeze encoder parameters and switch to appropriate inference/eval behavior to avoid unwanted encoder stochasticity; verify gradients do not enter frozen encoder weights.
- A shared encoder can produce a dictionary of needed intermediate layers in one forward pass. Different branches select from that dictionary.
- Ensure each independent branch has its **own optimizer, random seed, learned parameters and checkpoint**. No accidental parameter sharing.
- For lower VRAM, train branches **sequentially** first. At inference, run one encoder forward, then branches sequentially and fuse results; keep cached features only as long as needed.
- Do not assume 3 independent models necessarily cost 3x wall-clock at inference; benchmark both naive independent-model and shared-encoder alternatives.
- Check batch norm/layer norm, Dropout behavior and train/eval mode for consistency.
- Check feature tensor shapes, selected-layer count, register/class-token handling, reconstruction-group definitions and output map alignment for every branch variant.
- Avoid silent patching of upstream scoring, preprocessing or masks. If changes are needed, compare against an untouched baseline or clearly document them.
- Support resume-from-checkpoint and reproducible logging; save a manifest of data files and grouped IDs.

### Suggested config contract (illustrative, not existing code)

```yaml
experiment_id: E1
source_commit: VERIFY_LOCAL_GIT_HASH
benchmark: real_iad
split: official
seed: 1
encoder:
  name: dinov2_register_vit_base
  freeze: true
  selected_layers: verify_in_repo
model:
  train_bottleneck: true
  train_decoder: true
  dropout: verify_official_realiad
training:
  num_steps: verify_official_realiad
  batch_size: verify_official_realiad
  augmentation: official
inference:
  preserve_five_view_grouping: true
  scoring: official
fusion:
  strategy: none
```

**Do not blindly run this YAML**: field names and values are a proposed interface for Codex to implement only if useful.

## 6. Recommended Codex workflow and acceptance gates

### Gate A — Audit, no training

- [ ] Locate or clone the **official Dinomaly 2** repo, record commit and any local changes.
- [ ] Report Python, PyTorch, CUDA, GPU, VRAM and installed dependencies; avoid replacing working environments unnecessarily.
- [ ] Determine whether Real-IAD data exist, and enumerate split JSONs / object IDs / view IDs.
- [ ] Identify exact encoder, bottleneck, decoder, optimizer and loss definitions.
- [ ] Identify standard vs few-shot augmentations in code and compare them to paper descriptions.
- [ ] Identify *official* image, pixel and five-view object scoring implementation.
- [ ] Write a short `audit.md` listing paper–code discrepancies and a minimal implementation plan.

### Gate B — Single-model smoke + baseline

- [ ] One-image and one-batch forward pass with expected feature/map shapes.
- [ ] One or several train steps: encoder unchanged; bottleneck and decoder update; no NaN.
- [ ] One small evaluation including consistent object grouping and pixel masks.
- [ ] Full E0 training/evaluation if data and compute permit; save results and config. **If unavailable, stop and explain rather than invent baseline metrics.**

### Gate C — Minimal E1 prototype

- [ ] Two independently initialized branches train end-to-end and save separately.
- [ ] Inference reuses a single frozen encoder pass.
- [ ] All individual branch scores match independent single-branch inference numerically (within floating-point tolerance).
- [ ] Fusion outputs have correct shape/metadata, and `mean(A,A,A) == A` numerically.
- [ ] No encoder weights updated; no overlap between test and training groups.

### Gate D — Experiments and comparisons

- [ ] Finish E1 three-seed control before interpreting more complex ensembles.
- [ ] Finish E3 layer selection, with explicit documentation of layer numbering and compatible reconstruction targets.
- [ ] Only then run E2/E4 as controlled ablations.
- [ ] For each experiment, report single models, ensemble, computational cost, false positives and examples of complementary misses.
- [ ] Create summary CSV and readable Markdown table, with results marked as **measured**, **not yet run**, or **failed**.

## 7. Immediate next actions for Codex CLI

1. **Inspect, don't assume:** `git status`, `git rev-parse HEAD`, README, `dinomaly_multiview.py`, model definitions, `dataset.py`, and evaluation code.
2. Run environment / GPU / dataset diagnostics. Report blockers and choose the smallest viable smoke test.
3. Produce `experiments/audit.md` and exact, reproducible baseline commands based on *the local checkout*.
4. Implement a clean branch-training abstraction and **shared-encoder branch inference** without altering official single-branch outputs.
5. Add tests for tensor alignment, encoder freezing, view grouping, deterministic seed behavior and fusion identity.
6. Run E0 smoke test, then E1 two-branch smoke test. Do **not** launch expensive full-dataset training before those pass.
7. Present the plan/results and estimated resource cost; proceed with full E0/E1/E3 runs if feasible in the current session and environment.

## 8. Decision criteria / research interpretation

We regard the ensemble as promising **only if** improvement over the strongest single branch is reproducible, not confined to a few cherry-picked images, and does not create disproportionate false positives or prohibitive inference cost.

Interpret outcomes:

- **E1 helps as much as E3:** basic training stochasticity is sufficient; extra feature-layer complexity may not be justified.
- **E3 helps more than E1:** feature-depth diversity may be valuable; inspect which defect types benefit.
- **E2 helps but individual branches suffer:** test whether ensemble is masking weak detectors; check normal-score calibration.
- **All branch errors correlate strongly:** focus on creating genuinely different representations or revisiting whether reconstruction is the limiting factor.
- **Higher AUROC, worse localization / object AUROC:** not an unequivocal success; report trade-offs.
- **Ensemble no better than best branch:** valuable negative result; avoid claiming novelty on ensemble size alone.

**Potential publishable framing (hypothesis, not claim):** *Can multi-depth, diversity-controlled reconstruction branches improve multi-class anomaly detection while using a single frozen vision foundation model?*

## 9. Constraints, provenance, and deferred work

- **Hardware context supplied previously:** Ubuntu workstation, NVIDIA RTX 3090 24GB, Python 3.10/PyTorch/CUDA stack available. Verify actual current environment rather than assuming these versions are still active.
- Do not depend on the original Dinomaly repo when Dinomaly 2 code suffices; use original version only for reference or explicit comparison.
- Real-IAD is the target; MVTec-AD and VisA are secondary sanity checks, not substitutes for Real-IAD.
- Real-IAD_Variety (160 classes) and other datasets are **deferred** until a dependable baseline and at least one controlled ensemble study exist.
- No need to invent specialized defect synthesis or use anomalous images during training in the first round.
- Do not automatically add test-time augmentation, extra ViT backbones, learnable fusion networks, or supervised thresholds; these would confound the initial hypothesis.
- An ablation using *three independently pretrained ViTs* may be considered later as a cost/representation-diversity reference, but it is **not** the primary proposal.
- Keep commands, configs, git hash, checkpoints, seed, training sample coverage, raw predictions and evaluation scripts for reproducibility.

---

### Brief handoff prompt to paste into Codex CLI

> Read `Dinomaly2_Ensemble_Experiment_Handoff.md`. First audit the existing Dinomaly 2 checkout, local GPU environment, Real-IAD files, and evaluation pipeline. Record the actual code and paper differences. Do not begin large training jobs until a single-model smoke test and baseline reproduction path are validated. Implement a minimal shared-frozen-ViT / independent-bottleneck-plus-decoder experiment framework with reproducible configs, independent-branch metrics, calibrated fusion, five-view object-level evaluation, and tests. Prioritize E0 baseline, E1 independent seeds, then E3 different encoder layer selections. Report measured outcomes, blockers, and exact commands; do not invent results or silently change the official evaluation protocol.
