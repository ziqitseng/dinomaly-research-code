# Task 04A crop ablation

This is a frozen-checkpoint inference-time comparison of ArcAD's original center crop and a full-frame 392×392 resize. It does not train either model and does not show what full-frame training would achieve.

## Reproduce

From the workspace root, run `bash analysis_task04A_crop_ablation_20261103/reproduce.sh` in an environment where the RTX 3090 and NVIDIA driver are visible to PyTorch. The script refuses CPU inference. CPU-only map analysis runs after inference.

Inputs are the Real-IAD Variety 1024 dataset, the Task 02 shared manifest/caches, and the Task 01 full-frame D2 maps. Four categories reuse path-checked saved maps; thyristor maps are inferred for the original ArcAD and D2 branches. New full-frame ArcAD maps are inferred for all five categories. No ArcAD legacy data or results are used.

## Provenance

- ArcAD checkpoint: current prototype-initialized model; SHA256 `81c2b11c2942b8fe318ca62a5dd6a189e5754471a5ede48aa75d5f55d1bf664c`.
- D2 checkpoint: SHA256 `1148d6da28b18eedb6cb8137b432ff70f9df792c31b95613455140d35057e6fd`.
- Device: NVIDIA GeForce RTX 3090, 24 GiB; PyTorch 2.7.1+cu118, CUDA 11.8. GPU smoke and checkpoint integrity audits are saved in this folder.
- The exact shared and thyristor manifests, source split checks, code hashes, map-cache hashes, and transforms are recorded in `protocol_audit.json` and `manifest_validation.json`.

## Findings

Full-frame ArcAD reduced macro central-field view and object AUROC across the five selected categories, although power_jack improved. Central pixel AUPRO also decreased in all five categories. Full-frame maps make the 39 crop-external annotated regions observable; region scores remain descriptive, and all cases still need manual visibility review. See `summary.md` and `issues.md`.

