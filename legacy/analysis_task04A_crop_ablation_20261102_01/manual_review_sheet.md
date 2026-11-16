# Manual review sheet — crop-external laser_diode masks

**Observed:** the audited rows have positive source GT area but zero source-mask pixels in `[64:960)×[64:960)`. Their original ArcAD input therefore excludes the annotated region. The contact sheets show all five source views, source masks, center masks and ArcAD maps.

**Requires expert review:** whether the annotated defect is visible in each photograph, whether the mask location is correct, and whether the view should be considered an expected positive for the crop-limited model.

**Not inferred:** a crop-external GT region does not establish that ArcAD predicted normal; no visibility or causal label was fabricated. See `border_39_audit.csv`.
