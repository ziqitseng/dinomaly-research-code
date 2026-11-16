# Calibration reliability

The Task 02 per-category normal-only calibration set has only 10 objects/category. The empirical higher quantile selects the same maximum view-score threshold for nominal 1%, 5%, and 10% targets; these are **one threshold, not three calibrated operating points**. `threshold_audit.csv` records the exact native-scale thresholds and realized calibration FPR.

For this fixed-baseline follow-up, 50 normal objects (10/category across the five preselected diagnostic categories) are pooled to fit a global per-model empirical normal-score CDF and fixed 50/50 mean. Calibration object IDs are disjoint from evaluation IDs, but they are reserved objects from the same native test manifests, not an external deployment calibration set. A 10% point corresponds to approximately 5/50 calibration normal objects; 20% to approximately 10/50. Exact counts and 95% Clopper-Pearson intervals appear in `operating_point_metrics.csv`. Scores/thresholds are descriptive for this diagnostic transfer setting. No anomalies were used to fit or select fusion weights.

## Actual held-out results

See `calibration_reliability_counts.csv` for each system, level and nominal 10%/20% point: exact false-positive object counts, denominators and Clopper-Pearson 95% intervals on both calibration and held-out normal objects. The pooled calibration has 50 normal objects; thresholds are based on higher empirical quantiles, so ties can make the realized calibration rate differ from the nominal target. The held-out pool has 75 normal objects across the five selected categories, giving wide binomial intervals.
