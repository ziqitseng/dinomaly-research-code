# Proposed checkpoint-selection protocol

This proposal does **not** create or apply a split. The official prepared manifest and the original 159-category test remain untouched until a separate, explicit `--write` invocation creates a study selection manifest.

## Eligibility and object isolation

The original split generator (`prepare.py`) assigns all five views of an object together. It places reviewed source-OK train objects in `normal_train` or `normal_calibration`, takes one ArcAD anomalous support object per category from source test, and sends remaining source-test objects to `test`. Every object containing a Protocol B missing-mask view is protected in test.

Select, by a fixed SHA-256 rank with seed 20261112, **five source-test normal objects and five source-test anomalous objects per category** from the existing `test` partition. Each selected object must have five views, no exclusion flags or protected missing-mask views, and pixel eligibility on all views. Anomalous objects must have supplied, visibly positive masks on **all five views**. Select the same objects for both models and every training seed. Do not use `normal_calibration` for checkpoint choice: it has no labeled anomalies and is reserved for normal-only false-alarm thresholds.

The read-only feasibility scan of the current prepared records found 159 eligible categories, at least 19 eligible normal and 56 eligible fully annotated anomalous test objects per category. The proposed validation contains 1,590 objects / 7,950 views. The remaining final test contains 33,744 objects / 168,720 views. All 2,447 protected objects and all 3,235 missing-mask views remain in final test. Both classes and all 159 categories remain in each split. Validation objects do not overlap existing `normal_train`, `normal_calibration`, or `arcad_support` groups because they come only from the already disjoint `test` partition.

## Use and comparability

Use only validation evaluations for checkpoint choice. Primary selection score: arithmetic mean of the 159 category-level `I_B_AUROC` values, with every category weighted equally. Inspect category coverage and `I_B_AP`, `P_AUROC`, and `P_AUPRO` as diagnostics; do not create a new F1 metric or use final-test numbers to break ties. Predeclare a deterministic tie rule: choose the earlier completed update when primary means are equal to the reported precision. Compare Experiment A (milestone 50K within the 100K schedule) with Experiment B (final 50K in a dedicated 50K schedule), separately for each model, using the same validation manifest.

The selected validation objects came from the original final-test population. Thus the revised final test is smaller, and its results are **not directly comparable** to the already published full-test 100K scores. For a controlled A/B comparison, reevaluate the 100K reference checkpoint and selected 50K checkpoint on the same revised final-test manifest after choices are frozen. Retain original full-test results as their own benchmark, labeled with their original population. The 5+5 validation design lowers selection variance compared with one object per class, but it remains a single fixed split, and its performance is an estimate, not an independent dataset replication.

## Explicit creation

Run `bash run.sh inspect-validation` to inspect only. When the protocol is accepted for a future controlled experiment, create a new manifest explicitly with `bash run.sh inspect-validation --write manifests/validation.json` after creating the `manifests/` directory. The utility uses exclusive file creation and refuses to replace an existing manifest. The evaluator checks the prepared-data hash, object partition, protected set, disjointness, class balance, and category coverage before using it.
