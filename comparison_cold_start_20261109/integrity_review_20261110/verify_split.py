"""Independent metadata review of the current frozen dataset and run exposure.

No models/checkpoints loaded, no cache payload rewritten, no training interrupted.
"""
import collections
import itertools
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from common import check_prepared, config, now, protocol_b_ids, read_json, rows, sha256, write_csv, write_json


def main():
    check_prepared()
    cfg = config()
    listed = protocol_b_ids(cfg)
    with (ROOT / 'prepared/records.jsonl').open() as stream:
        records = [json.loads(line) for line in stream]
    identity = {r['id']: r for r in records}
    if len(identity) != len(records):
        raise RuntimeError('Duplicate stable image IDs')
    active_names = ('normal_train', 'arcad_support', 'normal_calibration', 'test')
    parts = {name: [r for r in records if r['partition'] == name] for name in active_names}
    keys = ('id', 'group', 'raw_pixel_sha256', 'input_pixel_sha256')
    sets = {name: {key: {r[key] for r in rs} for key in keys} for name, rs in parts.items()}
    overlaps = []
    for left, right in itertools.combinations(active_names, 2):
        overlaps.append(dict(left=left, right=right,
            **{key: len(sets[left][key] & sets[right][key]) for key in keys}))
    groups = collections.defaultdict(list)
    for r in records:
        groups[r['group']].append(r)
    split_groups = [g for g, rs in groups.items() if len({r['partition'] for r in rs}) != 1]
    incomplete = [g for g, rs in groups.items() if sorted(r['camera'] for r in rs) != [1, 2, 3, 4, 5]]
    actual_support = [r for r in parts['arcad_support'] if r['support_eligible']]
    fitting = parts['normal_train'] + actual_support
    fit_flags = [r['id'] for r in fitting if r['exclude_reasons']]
    nulls_outside_test = [r['id'] for r in records if r['id'] in listed and r['partition'] != 'test']
    bad_b_targets = [r['id'] for r in parts['test'] if r['protocol_b_image_label'] != (0 if r['id'] in listed else r['object_label'])]
    support_without_masks = [r['id'] for r in actual_support if not r['raw_mask'] or r['protected_test_group']]
    source_normals_wrong = [r['id'] for r in parts['normal_train'] + parts['normal_calibration'] if r['object_label'] != 0 or r['source_split'] != 'train']
    verified_no_overlap = not any(row[key] for row in overlaps for key in keys)
    passed = verified_no_overlap and not any((split_groups, incomplete, fit_flags, nulls_outside_test,
                                              bad_b_targets, support_without_masks, source_normals_wrong))

    white_paths = {r['path'] for r in rows(Path(cfg['audit_root']) / 'blank_or_nearly_uniform_images.csv')}
    test = parts['test']
    content_flagged = [r for r in test if r['exclude_reasons']]
    annotation_flagged = [r for r in test if r['annotation_issues']]
    pixel_excluded = [r for r in test if not r['pixel_eligible']]
    raw_duplicates, input_duplicates = collections.defaultdict(list), collections.defaultdict(list)
    for r in test:
        raw_duplicates[r['raw_pixel_sha256']].append(r)
        input_duplicates[r['input_pixel_sha256']].append(r)
    duplicated_inputs = {h: rs for h, rs in input_duplicates.items() if len(rs) > 1}
    conflicting_inputs = {h: rs for h, rs in duplicated_inputs.items()
                          if len({r['protocol_b_image_label'] for r in rs}) > 1}
    residual = dict(content_flagged_test_views=len(content_flagged),
        content_flagged_test_objects=len({r['group'] for r in content_flagged}),
        all_content_flagged_test_objects_protected=all(r['protected_test_group'] for r in content_flagged),
        content_flags_overlapping=dict(collections.Counter(reason for r in content_flagged for reason in r['exclude_reasons'])),
        audio_jack_pit_test_views=sum(r['category'] == 'audio_jack_socket' and r['source_class'].lower() == 'pit' for r in test),
        white_test_images=sum(r['raw_image'] in white_paths for r in test),
        white_image_objects_retained_views=sum('blank_image_in_object' in r['exclude_reasons'] for r in test),
        annotation_flagged_test_views=len(annotation_flagged),
        annotation_flags_overlapping=dict(collections.Counter(issue for r in annotation_flagged for issue in r['annotation_issues'])),
        pixel_ineligible_test_views=len(pixel_excluded),
        source_ng_present_empty_masks_test_views=sum(r['object_label'] == 1 and r['raw_mask'] is not None and r['mask_foreground_raw'] == 0 for r in test),
        repeated_raw_content_test_groups=sum(len(rs) > 1 for rs in raw_duplicates.values()),
        repeated_model_input_test_groups=len(duplicated_inputs),
        repeated_model_input_test_views=sum(len(rs) for rs in duplicated_inputs.values()),
        model_input_groups_with_conflicting_protocol_b_labels=len(conflicting_inputs),
        conflicting_protocol_b_input_views=sum(len(rs) for rs in conflicting_inputs.values()))

    prepared_hash = sha256(ROOT / 'prepared/COMPLETE.json')
    run_reports = []
    normal_ids = sets['normal_train']['id']
    support_ids = {r['id'] for r in actual_support}
    for model in ('arcad', 'dinomaly2'):
        for directory in sorted((ROOT / 'runs' / model).glob('seed_*')):
            run_cfg = read_json(directory / 'run_config.json')
            entry = dict(model=model, seed=run_cfg['seed'], directory=str(directory.relative_to(ROOT)),
                matches_current_prepared_hash=run_cfg['prepared_sha256'] == prepared_hash,
                normal_views=run_cfg['normal_views'], support_views=run_cfg['support_views'])
            logged = 0
            violations = []
            last_step = None
            log_path = directory / 'training.jsonl'
            if log_path.exists():
                snapshot = log_path.read_bytes().splitlines(keepends=True)
                for line in snapshot:
                    if not line.endswith(b'\n'):
                        continue  # A concurrently appended final row may be incomplete.
                    record = json.loads(line)
                    logged += 1
                    last_step = record['step']
                    violations.extend(i for i in record.get('normal_ids', []) if i not in normal_ids)
                    allowed_support = support_ids if model == 'arcad' else set()
                    violations.extend(i for i in record.get('support_ids', []) if i not in allowed_support)
            entry.update(sampled_log_records_checked=logged, last_logged_step=last_step,
                         logged_input_membership_violations=len(violations))
            if (directory / 'prototype_provenance.json').exists():
                proto = read_json(directory / 'prototype_provenance.json')
                entry['prototype_normal_ids_exactly_match_normal_train'] = set(proto['normal_ids']) == normal_ids and len(proto['normal_ids']) == len(normal_ids)
            if (directory / 'TRAIN_COMPLETE.json').exists():
                complete = read_json(directory / 'TRAIN_COMPLETE.json')
                entry.update(training_complete_steps=complete['steps'],
                    completion_matches_prepared_hash=complete['prepared_sha256'] == prepared_hash)
            if (directory / 'evaluation/COMPLETE.json').exists():
                complete = read_json(directory / 'evaluation/COMPLETE.json')
                evaluated_ids = {r['id'] for r in rows(directory / 'evaluation/view_scores.csv')}
                entry.update(evaluation_matches_prepared_hash=complete['prepared_sha256'] == prepared_hash,
                             evaluation_view_ids_match_shared_test=evaluated_ids == sets['test']['id'])
            run_reports.append(entry)
            passed = passed and entry['matches_current_prepared_hash'] and not violations
            passed = passed and entry.get('prototype_normal_ids_exactly_match_normal_train', True)
            passed = passed and entry.get('completion_matches_prepared_hash', True)
            passed = passed and entry.get('evaluation_matches_prepared_hash', True)
            passed = passed and entry.get('evaluation_view_ids_match_shared_test', True)

    output = Path(__file__).resolve().parent
    fields = ['id', 'group', 'category', 'source_class', 'protocol_b_image_label', 'is_ng_without_mask',
              'pixel_eligible', 'quality_flags', 'annotation_issues']
    def export(rs):
        return [dict(r, quality_flags='|'.join(r['exclude_reasons']), annotation_issues='|'.join(r['annotation_issues'])) for r in rs]
    write_csv(output / 'retained_content_flagged_test_views.csv', export(content_flagged), fields)
    write_csv(output / 'retained_annotation_flagged_test_views.csv', export(annotation_flagged), fields)
    conflict_rows = [dict(r, input_pixel_sha256=h) for h, rs in conflicting_inputs.items() for r in rs]
    write_csv(output / 'test_identical_inputs_with_conflicting_B_labels.csv', export(conflict_rows), ['input_pixel_sha256'] + fields)
    report = dict(checked_at=now(), known_partition_leakage_checks_passed=bool(passed),
        prepared_sha256=prepared_hash, records_sha256=sha256(ROOT / 'prepared/records.jsonl'),
        scope='Verified frozen metadata fingerprints, source and cached-input hashes, group identities, run configurations, prototype provenance and sampled training logs; no full cache payload reread or near-duplicate/physical-object audit',
        partitions_views={name: len(rs) for name, rs in parts.items()}, actual_arcad_support_views=len(actual_support),
        pairwise_partition_intersections=overlaps, groups_crossing_partitions=split_groups,
        incomplete_five_view_groups=incomplete, fitting_views_with_quality_flags=fit_flags,
        listed_ng_without_mask_outside_test=nulls_outside_test, protocol_b_target_errors=bad_b_targets,
        invalid_support_inputs=support_without_masks, invalid_source_normal_inputs=source_normals_wrong,
        fuse_holder_active_views=sum(r['category'] == 'fuse_holder' for rs in parts.values() for r in rs),
        toy_tire_conflict_active_views=sum(r['group'] == 'toy_tire/OK/S0005' for rs in parts.values() for r in rs),
        ng_without_mask_test_views=len(listed), runs=run_reports, remaining_test_quality=residual,
        interpretation='No detected exact-content/group train-test leakage if passed; test set is not fully curated. Protocol B protection intentionally retains quality-flagged test-only objects. No claim of absent near-duplicates, physical-instance overlap, or perfect semantic labels.')
    write_json(output / 'report.json', report)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
