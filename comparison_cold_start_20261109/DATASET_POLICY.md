# Dataset policy — Protocol B with audit flags

## Priority and provenance

The user's latest requirements govern the main image/pixel table. Read the explicit 3,235-entry `config/protocol_b_ng_without_masks.csv`, preserve original OK/NG identity, generate zero masks only in the experiment cache, set listed image targets to 0, and keep all listed views in test. Do not reinterpret these images as genuinely normal content. The old unknown-label/exclusion approach is archived and superseded.

A whole-object partition is still required: if any view appears in that list, **all five views form a protected test group**. It cannot become training, calibration, or ArcAD support. Group protection overrides earlier quality-based test exclusions, but preserves every quality reason. Other objects follow conservative audit quarantine. This is needed because 33 listed views belong to copied audio-jack Pit content and four listed views are white. Dropping them would contradict the required 3,235 test population.

`prepared/records.jsonl` records all source views, source labels, Protocol B labels, original mask availability, synthetic mask status, quality flags, protected-group status, partitions, eligibility, and cache hashes. `exclude_reasons` remains an audit-flag list even on a protected retained test row; actual membership is determined by `partition`. `decisions.csv` exposes both fields. Summary separately counts excluded reasons and all quality flags; overlapping reasons are not independent images.

## Issue-by-issue decisions

| Audit finding | Fitting / ordinary test handling | Protected Protocol B test handling |
|---|---|---|
| 250 wrong `fuse_holder/OK` copies of battery holders | Exclude the entire fuse-holder category; retain legitimate battery-holder data after removal | No fuse-holder entries exist in the 3,235 list; category stays excluded |
| 250 copied `audio_jack_socket/NG/Pit` views | Exclude wrong audio-jack Pit groups; keep other defect classes | Keep groups containing any listed view, retaining copied-content flags. This does not repair the category substitution |
| 565 raw exact-content groups (500 cross-category, 65 within-category); 56 contents with different anomaly classes | After known-block removal, quarantine remaining repeated-content objects; repeat check after RGB/392 transform. Nonprotected counterparts of retained protected duplicates are quarantined too | Retain protected groups even if repeated. Duplicates may remain within test; none may enter normal fitting/calibration or support through a nonprotected counterpart |
| 25 white images, including six official training views | Quarantine whole five-view object | Keep a protected test object, retain blank flags; its listed null-mask views still contribute zero targets |
| Tire `toy_tire/OK/S0005` has a Damage annotation on C4 | Quarantine all five views before normal selection/calibration | It is source OK and not protected by the NG-only list |
| 184 image/mask dimension disagreements, 173 in thyristor | Do not stretch into assumed alignment. Supplied flagged masks are excluded from pixel metrics/support; source image labels remain available | Listed null masks are generated zeros and stay pixel-eligible. Other genuinely misaligned supplied masks in the same group remain excluded from pixel metrics |
| 3,235 source-test NG/null masks | Never train, fit prototypes, calibrate, or sample labeled anomalies from them or their object | Keep all in test; generate per-image `_mask_EMPTY.png`; image label 0 and all-zero pixel truth in main Protocol B |
| 65 supplied empty masks | Original mask presence is preserved. Source NG image label remains 1 under B; reliable empty raster is zero pixel truth, never positive mask supervision | These are distinct from the 3,235 null-mask images; existing-empty status never silently adds an ID to the list |
| Three fully maskless NG objects; fourth object with no foreground including an empty file | Cannot supply reliable labeled support | Full maskless objects have five image targets 0 under B; source object label remains NG. Existing-mask views still follow mask-presence rule |
| 12 images with degenerate polygons; 26 with outside-canvas points; annotation canvas mismatches | Do not regenerate masks. Exclude flagged supplied masks from support/pixel/visibility-qualified auxiliary metrics | Main image B labels still depend on list membership; a synthetic zero target is not dropped because of another audit flag |
| 3,735 image and 2,912 mask references need nested category root | Use audited actual paths, retain declared path as stable ID; verify retained raw bytes | Same mechanical path handling |
| Two access-card camera names use C03/C04 | Group from source manifest parent, parse numeric camera and require exactly 1–5 | Same grouping; never split a null-mask object |
| 113 JPEGs named `.png`, all decoded successfully in audit | Decode by content, verify audit byte checksums, save real PNG cache | Same, no extension-based exclusion |
| 2,476 gray and 171 RGBA images | Convert explicitly to RGB, identically for both methods | Same |
| 155,811 stale annotation imagePath values; label spelling/casing varies | Pair from manifest/audit, not stale JSON filenames; preserve source strings and binary OK/NG separately | Same |
| Supplied nonempty mask vanishes at 256 | Log and exclude that supplied pixel target; retain source image target. Require surviving foreground at 392 for support | Synthetic zero masks are intentional and do not trigger a vanished-foreground exclusion |

The remaining dimension-mismatch counts are retaining_ring 3; pot_core and toy_tire 2 each; VR_joystick, common_mode_filter, lego_reel, and vertical_adjustable_resistor 1 each. Report pixel coverage per category rather than silently treating stretched masks as corrected.

## Explicit label example

An NG object has two nonempty masks, two absent masks, and one present empty mask. It is a protected test object because of the two absent masks, so none of its views can supply ArcAD supervision.

- Main Protocol B image targets are 1, 1, 0, 0, 1 in that corresponding order.
- Pixel targets are the two reliable binary masks, two generated all-zero masks, and the supplied empty raster if geometrically valid.
- Source NG labels remain 1 for all views and the source object target stays 1 for supplementary O_source metrics.
- The two listed null-mask views enter the separate `ng_without_mask` error bucket. An alarm there counts as a Protocol B false alarm, while its interpretation may be an annotation issue.
- Thresholds are fitted solely from reviewed source-OK calibration objects, never from any of these five views.

## What the score can establish

Protocol B implements the requested reporting convention; it does not prove every zero-mask NG view is defect-free. Some may visibly contain defects. Preserve source identities, quality flags, two image-label protocols on identical predictions, and distinct false-alarm buckets. The final report must measure the missing-mask alarm share; do not hard-code a prior 40%+ observation.

Matching Protocol B alone does not make numbers comparable with an external model. The current experiment still removes a category, uses cold-start support removal, retains protected problematic objects, differs in geometry from older native runs, and uses one shared histogram pixel evaluator. Record exact populations and settings before comparison. Conservative exclusions elsewhere can alter difficulty; protected test duplicates can alter independence.

## Repair roadmap for a future version

1. Recover authoritative fuse-holder normals and audio-jack Pit images, with release/file checksums and reviewed source correspondence. Renaming copied content is not a repair.
2. Review white replacements, the tire conflict, wholly unannotated NG objects, and empty-foreground objects as five-view sets. Save original and reviewed annotations, reviewer/date, rationale, and confidence in a versioned overlay.
3. Inspect raw images/masks/JSON canvases for all 184 mismatches, prioritizing thyristor. Apply a coordinate transform only with trusted registration evidence. Review clipping/degenerate polygons before any raster regeneration; check overlays and foreground alignment. Generic resizing cannot recover missing content.
4. Distinguish truly invisible defects from omitted annotations by review/authoritative visibility data. Add a corrected-label sensitivity only in a separately frozen protocol; do not silently change Protocol B's fixed published-comparison list after observing scores.
5. Re-audit corrected inputs, freeze new metadata/overlay/list, regenerate partitions, refit normal-only prototypes, and retrain both models. Filtering old scores cannot undo contaminated fitting.

The audit's 800 rasterized JSON/mask pairs matched exactly, but this verifies sampled raster consistency, not complete semantic correctness. Exact hashes do not detect all near-duplicates or shared physical objects. The current benchmark is documented and reproducible, not a fully corrected official dataset.
