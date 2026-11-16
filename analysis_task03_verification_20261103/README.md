# Task 03 analysis package

This package audits the current 160-category summaries, resolves the saved Task 02 empty-mask cases, independently aligns the five current Task 02 categories on a common 224×224 grid, and computes threshold-free diagnostics plus a fixed 50/50 score-level fusion baseline.

## Reproduction

From the workspace root, run:

```bash
bash analysis_task03_verification_20261103/reproduce.sh
```

The scripts use the saved maps in `analysis_shared_failure_20261103`; they do not train models or perform model inference. They require the local `dinomaly2` Python environment and the source dataset/checkpoints documented in `protocol_audit.json`. Phase 2 takes several minutes and writes analysis artifacts into this directory. The saved output of this run was generated on 2026-11-03. Source scripts are in `work/`.

## Provenance and findings

- ArcAD: initialized checkpoint `ArcAD/out/real_iad_variety out/arcad_realiad_variety_protoinit/model.pth`. Artifacts in `legacy/` and the obsolete non-initialized ArcAD run are excluded.
- D2: `Dinomaly2/out/real_iad_variety_1024 out/full run/variety_160cat/model.pth`, SHA256 `1148d6da28b18eedb6cb8137b432ff70f9df792c31b95613455140d35057e6fd`.
- ArcAD protoinit SHA256: `81c2b11c2942b8fe318ca62a5dd6a189e5754471a5ede48aa75d5f55d1bf664c`.
- No new inference was run. CUDA/NVML was unavailable (`torch 2.7.1+cu118`); CPU analysis was used on existing maps. Detailed input and split hashes are in `protocol_audit.json`.
- Current category set: `lattice_block_plug`, `laser_diode`, `power_jack`, `LED_indicator`, `aircraft_model_head`. This differs from the older Task 03 brief, which listed `thyristor`.
- The 41 empty common-crop laser-diode masks resolve to 39 source masks wholly outside the shared crop and two all-zero source masks. Expert review is still needed to assess physical visibility.
- Test-normal thresholds and pixel-FPR sweeps are descriptive only. The fixed 50/50 baseline uses in-sample normal references and is exploratory, not deployment calibration.
- Full results, limitations, and next steps are in `analysis_report.md`; blockers and skipped deliverables are in `issues.md`.

