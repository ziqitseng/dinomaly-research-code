# Task 03 issues and skipped work

## Blockers and unresolved cases

- No independent, object-disjoint normal calibration set or anomaly-labeled validation set was available. No calibrated operating point or validation-fitted fusion weight is claimed.
- The 41 positive laser-diode views with empty common-crop masks remain `requires_manual_review`: 39 masks are outside the common crop and two source masks are all zero. These observations do not establish whether the defect is physically visible in each image.
- CUDA/NVML initialization was unavailable. No new model inference was launched; the analysis uses saved Task 02 maps.
- The current Task 02 category set has `aircraft_model_head` in place of the older brief's `thyristor`; only the actual five-category saved maps were analyzed.

## Skipped deliverables or analyses

- A validation-fitted weighted fusion model and calibrated OR rule were skipped because there are no independent validation anomalies and no independent normal calibration population. The fixed 50/50 z-score mean is reported as exploratory only.
- Independent calibration curves at 1%, 5%, and 10% FPR were skipped. The 25 held-out normal objects/category resolve object FPR only in increments of 4%; reported sweeps are test-derived descriptive diagnostics.
- Optional map-level fusion, 160-category map inference, and extension-category inference were skipped. The category extension list is frozen in `category_selection_plan.md`; no inference was launched.
- Manual expert adjudication and physical visibility labeling were skipped; the contact sheets and audit identify the cases for review without changing labels.
- Confidence intervals were computed for view/object AUROC and AP using 2,000 paired physical-object bootstrap draws/category. Pixel and component metrics are point estimates; no pixel bootstrap interval is provided.
- Paired example figures contain one deterministic representative per requested case type (seven total). A larger 3–5-example set per case type was not curated in this run.

