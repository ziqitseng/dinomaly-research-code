# Current split review — 2026-11-10

The current ArcAD and Dinomaly2 runs pass the known exact-content and object split-leakage checks. The evaluation dataset is not fully curated.

The independent metadata review found zero intersections between every pair of normal training, ArcAD support, normal calibration, and test partitions for stable image IDs, five-view object IDs, recorded source-pixel hashes, and recorded resized model-input pixel hashes. All five views of each object remain together. Both run configurations match the same frozen prepared dataset. Logged training samples belong to their permitted partitions, and ArcAD's prototype provenance matches the shared normal training set.

| Partition | Views |
| --- | ---: |
| Shared normal training | 5,550 |
| Additional ArcAD labeled support | 795 |
| Normal calibration | 14,250 |
| Shared test | 176,670 |

No fitting inputs carry the known content-exclusion flags. `fuse_holder` and the known `toy_tire/OK/S0005` conflict are excluded from all active partitions. All 3,235 listed missing-mask NG views remain in test, with the requested Protocol B image labels.

## Remaining evaluation problems

The requirement to retain every listed missing-mask NG view, together with its whole five-view object, takes priority over quality quarantine for those test objects. It leaves 205 content-flagged test views across 41 objects, including 110 misplaced `audio_jack_socket/Pit` views and five actual white images. Flags overlap: object-level duplicate and blank flags do not mean every flagged view is itself a duplicate or blank.

There are 16 groups of identical resized test inputs involving 35 views. Two such groups, involving seven views, have conflicting Protocol B image labels. These repetitions occur within test; they do not cross into training, support, or calibration.

Separately, 217 test views have annotation issues and are excluded from pixel metrics. There are also 65 source-NG test views with existing empty masks, distinct from the 3,235 missing-mask list. Their image labels remain anomalous under the current list-based Protocol B rule; an empty mask alone does not establish that the source NG label is wrong.

## Interpretation and later cleanup

The existing runs support a controlled comparison under the documented frozen Protocol B policy. They should not be described as a fully corrected benchmark. Near-duplicates, physical-instance independence, and all semantic annotations have not been exhaustively verified. This review uses the frozen metadata and recorded pixel hashes; it does not reread all cached image payloads.

Retain the current Protocol B results for reproducibility. A later, separately versioned curated evaluation should adjudicate misplaced objects, blanks, duplicates and conflicting labels, and mask geometry/annotation problems. Apply exactly the same curated test selection to both models and report its exclusions and denominators. That supplementary evaluation changes the all-3,235-retained benchmark rule and must be identified separately.

Test-only cleanup can reuse existing checkpoints for reevaluation. If cleanup changes fitting data, training labels, or the category definition, review retraining both models and rebuilding ArcAD prototypes as appropriate.

The review did not change the frozen split, scientific configuration, model code, checkpoints, or running training process. Machine-readable evidence and affected-view lists are alongside this file in `report.json` and the three CSV files. Run progress in that report is a timestamped snapshot, not live status.
