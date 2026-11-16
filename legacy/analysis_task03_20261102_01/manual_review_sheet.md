# Manual review guide

Review the paired PNGs listed in `paired_example_index.csv` at the original source path and common crop. Each figure shows the same camera image and GT for D2/ArcAD, calibrated-normal ranks and native raw image score values. Cyan outlines mark actual top-1% anomaly-map pixels; red shows common-grid GT. Heatmap display colors are normalized independently; compare spatial positions and consult listed numeric scores for magnitude.

## What is directly observed

- The saved source mask pixel counts and common-crop geometry are recorded in `empty_mask_audit.csv`; 39 masks have no pixels in the common crop and two are source-all-zero.
- The paired figure shows model score maps and GT at the same source crop. Ranking/coverage values are numeric in the index.

## What remains a hypothesis or needs review

- Occlusion, view-dependent defect visibility, image-label error, specularity, material reflectance, and causal effects of ArcAD supervision require expert review or controlled experiments. No visual appearance alone is treated here as proof of a label error or physical cause. Cases marked for manual review should be checked against original 1024 images and all five siblings.
