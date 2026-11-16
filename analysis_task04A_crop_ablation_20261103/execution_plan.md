# Task 04A execution plan and preflight

## Current status

No inference or experiment has been launched. The machine-level GPU check returned: `nvidia-smi` could not communicate with the NVIDIA driver. Task 04A requires stopping when CUDA is unavailable rather than running this workload on CPU. This plan records the intended run once a working CUDA device is available.

The current workspace is `/media/avlab/1CE6451EE644FA12/2026_10_22_dinomaly`. The output directory for any eventual run is this new isolated directory. No files from `legacy/` or any ArcAD legacy output will be inputs.

## Inputs already located

- Source dataset: `/media/avlab/7458a474-6ea2-4b90-b232-373c3df8d329/real_iad_variety_1024`.
- Frozen ArcAD checkpoint: `ArcAD/out/real_iad_variety out/arcad_realiad_variety_protoinit/model.pth`, SHA256 `81c2b11c2942b8fe318ca62a5dd6a189e5754471a5ede48aa75d5f55d1bf664c`.
- Frozen D2 checkpoint: `Dinomaly2/out/real_iad_variety_1024 out/full run/variety_160cat/model.pth`, SHA256 `1148d6da28b18eedb6cb8137b432ff70f9df792c31b95613455140d35057e6fd`.
- Shared held-out manifest and split/leakage validation: `analysis_shared_failure_20261103/shared_test_manifest.json` and `manifest_validation.json`.
- ArcAD-original aligned maps exist in the current Task 02 cache for five categories. D2 full-frame maps exist for the three primary categories in `Dinomaly2/out/re_eval_shared_20261031_cpu01/maps/`; those maps will be reused after exact path and hash/order checks. Task 02's common 224 maps alone are center-cropped and are not a substitute for full-frame maps.
- The 39 nonempty source masks outside ArcAD's original crop and two all-zero source masks are recorded by Task 03; labels will remain unchanged and all 41 visibility cases will remain `requires_manual_review`.

## Frozen analysis design

1. Revalidate the exact shared path membership, five-view object groups, mask labels, and ArcAD labeled-object exclusions from the Task 02 manifest. Primary categories are `lattice_block_plug`, `laser_diode`, and `power_jack`.
2. Reuse path-verified Task 02 ArcAD-original maps for regression checks and Task 01 D2 full-frame maps for the primary categories. Compare ArcAD-original scores/maps with Task 02 using a declared numerical tolerance before accepting the baseline.
3. Infer only the ArcAD-full-frame branch with the same prototype-initialized checkpoint, resizing each full 1024×1024 image directly to 392×392 and preserving ImageNet normalization, map construction, 5×5 sigma-4 smoothing, and top-1% image score. Input tensor shape remains 392×392, divisible by patch size 14; no network or checkpoint changes are permitted.
4. If a working GPU has sufficient memory and time, add the preselected comparison categories `LED_indicator` and `thyristor`, only after building and validating exact ArcAD/D2 shared path intersections and ArcAD-training exclusions. The present Task 02 set has `aircraft_model_head`, not `thyristor`; it will not be silently substituted.
5. Preserve raw float32 maps. Evaluate the three branches on the source-aligned central 224×224 field. Evaluate ArcAD-full-frame and D2 on a common full-source 256×256 field; represent ArcAD-original's unobserved outside-crop field as missing, never zero. Use nearest-neighbor mask resampling.
6. Report view and pooled five-view object I/O AUROC/AP, common-field pixel metrics, region/top-k localization, GT visibility strata, and per-view rows. Keep crop-external positive views in image-level metrics while excluding empty evaluation-field masks from positive-region localization metrics.
7. Run 2,000 paired physical-object bootstrap replicates with all five views clustered together and seed `20261103`. Report invalid single-class draws explicitly. No thresholds will be tuned on test labels; thresholded results, if any, will be diagnostic-only and retain Task 03's calibration limitations.
8. Hash both checkpoints before and after inference; record code/manifest hashes, versions, device, batch size, and elapsed time. Write outputs only under this new directory.

## Geometry and interpretation controls

- ArcAD-original input: resize 1024→448, center crop 392 (nominal source field `[64:960)×[64:960)`), ImageNet normalization.
- ArcAD-full-frame input: resize 1024→392, no crop, same normalization. This changes inference geometry and is out-of-distribution relative to training; it does not test retraining.
- D2 reference input: original full-frame 280×280 processing; use saved full-frame maps when exact path alignment is proven.
- Validate source-to-input and map coordinates with synthetic coordinate/checkerboard patterns and landmark positions, not dimensions alone. Confirm the ArcAD encoder accepts 392×392 and the same 28×28 patch grid.
- Never compare the three branches directly in native map coordinates. Project to declared source-aligned grids. For ArcAD-original, full-source border pixels are unobserved.

## Stop conditions

- Do not launch any inference while CUDA is unavailable.
- Stop if ArcAD-original regression exceeds the declared tolerance, any checkpoint hash changes, path membership differs, or any category has missing/duplicate views or train/test object leakage.
- Stop if the full-frame input produces a changed internal token/grid geometry or requires modifying weights or architecture.
- If the GPU is still unavailable, produce a blocker report and do not substitute CPU inference.

