# Calibration reliability audit

No independent calibration split exists in the current workflow. Task 02’s ArcAD and D2 normal score references are training normals already seen by their models. ArcAD has 7–9 normal objects/category and D2 has 25; these are in-sample and cannot support deployment calibration claims.

The current Task 02 held-out test manifest has 25 normal objects/category, not 10. Its empirical object-FPR resolution is 1/25 = 4 percentage points. It cannot resolve distinct 1%, 5%, and 10% points. Curves in `failure_overlap_vs_fpr.csv` and `region_overlap_vs_fpr.csv` are test-derived descriptive sweeps, not selected operating points. Exact normal-object/pixel FP counts and realized rates are included.

The fixed 50/50 fusion uses global normal-score z-scores from the in-sample training references, solely as an exploratory unsupervised baseline. No category-specific thresholds or fusion weights were fit, and no independent anomaly validation objects exist. A held-out, object-disjoint normal calibration set and anomaly-labeled validation set are needed for deployment-like operating points.
