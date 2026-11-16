# Local CostFilter-AD setup

Full MVTec-AD reproduction is now complete: 17 real-data training epochs
and two full evaluations passed on the RTX 3090. See `MVTEC_REPRODUCTION.md`
and `runs/mvtec_reproduction_20261113/REPORT.md` for results and the required
DTD augmentation dtype repair. The local model reached 76.0673% pixel AP,
compared with 75.6309% from the published checkpoint on this PC.

The RTX 3090 24 GB can run the Costfilter_Dinomaly variant. The initial
hardware check passed with the repository's DINOv2 ViT-B/14 encoder,
eight-block Dinomaly decoder, matching cost construction, and CostFilter
network. Inputs were random 392 x 392 RGB images and weights were randomly
initialized. This verifies execution and memory feasibility, not anomaly
detection quality or published benchmark accuracy.

Verified on 2026-11-13 in `cosfilterad`, in FP32 with gradients disabled:

| Batch size | Peak allocated VRAM | Peak reserved VRAM | Result |
| --- | --- | --- | --- |
| 1 | 0.823 GiB | 0.910 GiB | Passed |
| 12 | 1.974 GiB | 2.303 GiB | Passed |

The MVTec-AD inference entry point's `--help` command passed, including
all its imports. `pip check` reported no broken requirements. The optional
xFormers package is absent; the repository's default attention implementation
passed these checks without it.

The environment is named `cosfilterad`, with Python 3.8.20 and
PyTorch 1.13.1+cu117, torchvision 0.14.1+cu117, and torchaudio 0.13.1+cu117.
It is stored at `../.conda/envs/cosfilterad` relative to this file because
the system drive had only 4 GB free. Its parent directory was appended to
the user's Conda `envs_dirs`; existing environment directories remain available.

```bash
conda activate cosfilterad
python costfilter_setup/verify_inference.py
```

Run the verification command from the workspace directory containing
`costfilter-ad/` and `costfilter_setup/`. It performs FP32 forward passes
at batch sizes 1 and 12 and reports PyTorch peak allocated/reserved VRAM.
These figures exclude the desktop, CUDA context, and other processes.
Its timings are diagnostic and are not a throughput benchmark.

`requirements-dinomaly.txt` contains the selected runtime dependencies,
using the upstream pinned versions for direct dependencies. The upstream
full requirements file also covers other baselines and CUDA 12 libraries;
this installation is scoped to Dinomaly. Other baseline variants have not
been validated. `requirements.lock.txt` records all installed Python package
versions; use the extra PyTorch index in `requirements-dinomaly.txt` when
reinstalling that lock file.

The three published MVTec-AD checkpoints were downloaded on 2026-11-13.
The files are installed at the exact paths used by the upstream MVTec script:

- `backbones/weights/dinov2_vitb14_reg4_pretrain.pth` (346,393,545 bytes).
- `saved_results_mvtec/vitill_mvtec_uni_dinov2br_c392_en29_bn4dp2_de8_laelu_md2_i1_it10k_sams2e3_wd1e4_w1hcosa2e4_ghmp09f01w01_b16_s1/model_10000.pth`
  (592,137,369 bytes).
- `checkpoint_paths/mvtecad0.0001/costfilter_dinaomaly_mvtecad.pth`
  (174,467,764 bytes).

These paths are relative to `costfilter-ad/Costfilter_Dinomaly/`. All three
checkpoints passed strict state-dictionary loading with every key and shape
matched. A 392 x 392 FP32 GPU forward pass using the published weights
produced a finite anomaly map. This does not measure dataset accuracy.
`weights_manifest.json` records source URLs, sizes, SHA-256 digests, and
verification results. The digests are local records, not publisher-signed
checksums. `download_mvtec_weights.py` can repeat the download; it requires
`gdown==5.2.0`, now installed in `cosfilterad` and recorded in the lock file.
`verify_weights.py` repeats checkpoint loading and the GPU forward check.

Before pretrained evaluation:

- The MVTec-AD checkpoints are installed. For other datasets, download their
  corresponding checkpoints from the upstream README first.
- Supply the dataset path through `--data_path`.
- The local MVTec-AD script defaults to `cuda:0` and accepts `--device`.
  Its original upstream version selected `cuda:2`, which this single-GPU PC
  does not have. Other upstream dataset scripts may still need adjustment.
- Run from `costfilter-ad/Costfilter_Dinomaly/` so relative checkpoint paths
  resolve correctly. The local MVTec-AD script fails immediately if its
  selected CostFilter checkpoint is absent.

DTD is used for synthetic anomaly generation during training; this
random-input inference verification does not require it.

The full MVTec-AD train/evaluate comparison uses the local entry point's
`--dtd_path`, `--output_dir`, and `--costfilter_checkpoint` options. See
`MVTEC_REPRODUCTION.md` for its protocol and exact run locations. DTD is
installed under `../datasets/dtd/images` relative to this file.

## Training feasibility verification

On 2026-11-13, `verify_training.py` passed three synthetic training steps
at each of batch sizes 8 and 12 in `cosfilterad` on the RTX 3090. It compiles
the upstream MVTec-AD `train` function in memory, replacing dataset loading,
pretrained checkpoint loading, checkpoint output, and epoch count. It keeps
the real model architecture, frozen Dinomaly forward pass, matching cost
construction, CostFilter training mode, all training losses, backward pass,
Adam updates, scheduler and gradient clipping. Random weights are used;
no useful trained checkpoint is produced. Finite nonzero gradients, finite
parameters/loss, and initialized Adam state were checked.

| Batch size | Peak allocated VRAM | Peak reserved VRAM | Later synthetic step times |
| --- | --- | --- | --- |
| 8 | 2.756 GiB | 3.209 GiB | 0.327–0.330 s |
| 12 | 3.524 GiB | 4.412 GiB | 0.425–0.428 s |

These are FP32 measurements at 392 x 392, with `CUDA_LAUNCH_BLOCKING=1`
as in the upstream entry point. VRAM figures exclude the CUDA context,
desktop and other processes. Timings cover the model computation through
the optimizer update; they exclude real image loading, anomaly synthesis,
checkpoint I/O, full-epoch evaluation, and diagnostic gradient checks. They
are not an estimate of total dataset training time or proof of convergence.

```bash
conda activate cosfilterad
python costfilter_setup/verify_training.py
```

The [CostFilter-AD paper](https://arxiv.org/pdf/2505.01476), section 4,
states a general training recipe of 40 epochs, batch size 8, Adam and
initial learning rate 0.001. Its A100 40 GB statement is explicitly attached
to the batch-size-1 efficiency/inference measurements around Table 7.
I found no wall-clock training duration or unambiguous GPU assignment for
each training run. The [official README](https://github.com/ZHE-SAPI/CostFilter-AD#environments)
lists A100 40/80 GB and GV100 32 GB for experiments. The released Dinomaly
MVTec-AD training script instead uses 17 epochs, batch size 12, and Adam
learning rate 0.0001; its settings should not be described as an exact
copy of the general paper recipe.

The provided integration targets original Dinomaly (CVPR 2025), using a
DINOv2 ViT-B/14 encoder with four register tokens. DINOv2 is the feature
backbone name. [Dinomaly2](https://github.com/guojiajeremy/Dinomaly2) is a
separate later extension; its integration and checkpoint compatibility with
CostFilter-AD have not been verified here.
