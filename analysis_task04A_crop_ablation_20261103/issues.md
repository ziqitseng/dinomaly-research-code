# Task 04A limitations and unresolved items

- The 39 nonempty masks wholly outside the original ArcAD crop remain `requires_manual_review`. The two all-zero source masks from the 41 Task 03 empty-center cases remain unresolved. No GT labels were changed.
- No independent normal calibration or anomaly-validation set is available. This analysis reports no calibrated thresholds or thresholded detection claims.
- Full-frame ArcAD is an out-of-distribution inference intervention on the original frozen checkpoint. It cannot establish full-frame training performance or, by itself, justify retraining.
- Task 02's five-category set had aircraft_model_head instead of the Task 04A brief's thyristor. This run used the exact Task 02 membership for the four shared categories and built a separate exact native ArcAD/D2 test intersection for thyristor, excluding every ArcAD labeled-training object. It did not substitute aircraft_model_head.
- Native ArcAD and D2 test splits differ. The reported comparison uses their path intersection and groups all five cameras by physical object.
- Pixel AUROC/AP are 2,048-bin point estimates. P-AUPRO uses 200 thresholds with FPR cap 0.30. Bootstrap intervals cover view/object AUROC/AP, not pixel metrics.
- The optional D2 source-centered crop branch was skipped; it is not needed for the required comparison.
- Full-field ArcAD-original predictions do not exist outside its observed crop; those pixels are represented as unobserved, not zero-filled.
- The five categories are preselected diagnostics; results do not generalize to all 160 categories.

