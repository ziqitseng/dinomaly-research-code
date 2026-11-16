# Data quality issues and visibility audit

Recomputed the 41 positive `laser_diode` views that had empty Task 02 common-grid GT. Full-source mask and crop pixel counts are in `empty_mask_audit.csv`; all source files were read-only.

## Classification


- `genuinely_all_zero_source_mask`: 2
- `nonzero_mask_outside_[64:960]_crop`: 39

Task 01’s two `OilStain/S0005` paths are both present and confirmed to have genuinely all-zero source masks. The remaining cases are classified independently from raw 1024 masks and a direct [64:960] crop followed by nearest-neighbor 224 downsampling.

ArcAD and D2 view labels agree for all 41 audited cases (disagreements: 0). The positive object `NG/OilStain/S0005` has mixed view labels: two anomalous views and three normal views in both manifests. This is retained as a view/object-label distinction, not treated as proof of an incorrect label.

A zero or crop-invisible mask is not treated as proof that an anomaly label is wrong. Every case is marked `requires_manual_review`; this automated audit does not establish physical defect visibility.

No source mask, image label, object label, checkpoint, or original output was modified.

## Contact sheets

Pages under `empty_mask_contact_sheet/` show each target image, uncropped source mask, common-crop 224 mask, and all five sibling camera views. The uncropped mask uses max pooling for display visibility only; all reported pixel counts are from the source mask. Masks are displayed independently; no annotation is invented.
