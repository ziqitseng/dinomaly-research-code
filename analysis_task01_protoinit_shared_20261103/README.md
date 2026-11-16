# Task 01 — shared split evaluation with initialized ArcAD run

This folder contains a fresh low-three-category shared-split evaluation. The ArcAD side uses the prototype-initialized run at `ArcAD/out/real_iad_variety out/arcad_realiad_variety_protoinit`. No ArcAD files, metrics, scripts, or manifests from any `legacy/` directory were used.

## Procedure

1. `work/build_shared_manifest.py` intersects the current ArcAD Variety `sup` test manifest with the official DiNomaly2 native test manifest and excludes all ArcAD labeled images. It writes `shared_test_manifest.json` and `manifest_validation.json`.
2. ArcAD inference uses the new run's `model.pth`, strictly loaded into the source architecture, with the original 448 resize / 392 center crop, 256 map, 5×5 Gaussian (`sigma=4`), and top-1% view scoring.
3. The DiNomaly2 numeric map arrays from its prior inference-only output were reused only after every image path, object key, camera, label, mask path, and stored view score matched this newly generated manifest. No ArcAD comparison values from that prior output were reused. DiNomaly2 object scores are independently pooled from its map arrays.
4. `work/finalize_comparison.py` computes paired image and object metrics and common pixel metric formulas. Pixel deltas are omitted because the models use different input crop transforms and the map frames are not identical.

## Results

See `shared_metrics.json`, `shared_metrics.csv`, and `paired_comparison.csv`. F1 values are oracle maximum F1 over the evaluated labels and are not calibrated deployment thresholds. Macro means are unweighted arithmetic averages over the three categories.

Numeric maps are bundled under `arcad_maps/` and `dinomaly2_maps/`; each folder has an `index.csv` that maps records to array positions. The ArcAD arrays contain raw postprocessed score maps and transformed GT masks; the DiNomaly2 arrays retain the inference output and aligned masks.

## Validation and limits

`d2_map_reuse_validation.json` confirms map membership, dimensions, finite values, labels, mask paths, and independently recomputed DiNomaly2 view scores. Two positive laser diode views have all-zero source PNG masks; ArcAD's native resize/crop transform also drops small annotations in 37 further positive views. Those records remain in the split, and both effects are documented in `issues.md`.

The DiNomaly2 native-test reference I-AUROC values are recorded separately from shared-split values. The native test inference was not rerun in this update. The ArcAD full-run `results.csv` was not used in paired metrics because it does not represent this exact paired population.

## Reproduction

From the workspace root, run `bash analysis_task01_protoinit_shared_20261103/reproduce.sh`. It creates a new timestamped sibling folder for each run, copies the workflow scripts there, regenerates the split, runs inference and validation, and leaves this result folder untouched. The evaluator can resume from a completed per-category ArcAD map cache inside that new folder; checkpoint hashes are checked before and after inference.
