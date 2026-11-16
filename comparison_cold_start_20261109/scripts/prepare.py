"""MANUAL stage: audit join, immutable RGB/mask cache, and matched split.

Reads raw data; writes only this experiment. No model imports or GPU use.
"""
import argparse
import collections
import hashlib
import json
import math
import os
import re
from pathlib import Path

from common import ROOT, check_lock, check_prepared, config, group_id, now, protocol_b_ids, protocol_b_label, rows, sha256, write_csv, write_json


def rank(values, seed, namespace):
    return sorted(values, key=lambda value: hashlib.sha256(
        ('%s/%s/%s' % (seed, namespace, value)).encode()).hexdigest())


def assign_partitions(records, cfg):
    """Pure split logic, also exercised with synthetic records in tests."""
    groups = collections.defaultdict(list)
    for record in records:
        groups[record['group']].append(record)
    for rs in groups.values():
        protected = any(r['is_ng_without_mask'] for r in rs)
        for r in rs:
            r['protected_test_group'] = protected
    categories = sorted({r['category'] for r in records if not r['exclude_reasons'] or r['protected_test_group']})
    counts = []
    for category in categories:
        candidates = {g: rs for g, rs in groups.items()
                      if rs[0]['category'] == category and (not rs[0]['exclude_reasons'] or rs[0]['protected_test_group'])}
        normals = [g for g, rs in candidates.items()
                   if rs[0]['source_split'] == 'train' and rs[0]['object_label'] == 0]
        # Source test provides a support pool because no official anomalous train exists.
        support_pool = [g for g, rs in candidates.items()
                        if rs[0]['source_split'] == 'test' and rs[0]['object_label'] == 1
                        and not rs[0]['protected_test_group']
                        and any(r['support_eligible'] for r in rs)]
        n = max(1, math.floor(len(normals) * cfg['normal_fraction']))
        if n > len(normals) or len(support_pool) < cfg['support_objects_per_category']:
            raise RuntimeError('Insufficient reviewed training data: ' + category)
        if n == len(normals):
            raise RuntimeError('No independent source-OK calibration objects: ' + category)
        selected_normals = set(rank(normals, cfg['split_seed'], category + '/normal')[:n])
        selected_support = set(rank(support_pool, cfg['split_seed'], category + '/support')
                               [:cfg['support_objects_per_category']])
        for group, rs in candidates.items():
            if rs[0]['protected_test_group']:
                partition = 'test'
            elif group in selected_normals:
                partition = 'normal_train'
            elif group in selected_support:
                partition = 'arcad_support'
            elif rs[0]['source_split'] == 'test':
                partition = 'test'
            else:
                partition = 'normal_calibration'
            for r in rs:
                r['partition'] = partition
        tested = [rs for rs in candidates.values() if rs[0]['partition'] == 'test']
        if {rs[0]['object_label'] for rs in tested} != {0, 1}:
            raise RuntimeError('Test requires both object classes: ' + category)
        counts.append({'category': category, 'eligible_train_normal_objects': len(normals),
                       'normal_train_objects': n, 'support_objects': len(selected_support),
                       'normal_train_views': sum(len(candidates[g]) for g in selected_normals),
                       'arcad_support_visible_views': sum(r['support_eligible'] for g in selected_support for r in candidates[g]),
                       'normal_calibration_objects': len(normals) - n,
                       'test_objects': len(tested),
                       'test_normal_objects': sum(rs[0]['object_label'] == 0 for rs in tested),
                       'test_defective_objects': sum(rs[0]['object_label'] == 1 for rs in tested),
                       'test_views': sum(len(rs) for rs in tested),
                       'test_ng_without_masks': sum(r['is_ng_without_mask'] for rs in tested for r in rs),
                       'test_visible_views': sum(r['visible_eligible'] for rs in tested for r in rs),
                       'test_pixel_views': sum(r['pixel_eligible'] for rs in tested for r in rs)})
    return counts


def prepare(cfg, output, workers=4, resume=False, compression=1, progress_seconds=5, verify_written=False):
    from prepare_cache import build_cache, verify_cache
    output = Path(output)
    print('phase=load_audit_metadata', flush=True)

    audit = Path(cfg['audit_root'])
    raw = Path(cfg['raw_root'])
    listed_ids = protocol_b_ids(cfg)
    # Source NG identity is preserved separately from the Protocol B scoring target.
    records = []
    for row in rows(audit / 'manifest_inventory.csv'):
        path = row['actual_image']
        match = re.search(r'_C0?([1-5])_', Path(path).name)
        if not match:
            raise RuntimeError('Unrecognized camera: ' + path)
        records.append({'id': row['declared_image'], 'raw_image': path,
                        'raw_mask': row['actual_mask'] or None,
                        'category': row['category'], 'source_split': row['split'],
                        'source_class': row['anomaly_class'], 'group': group_id(row['declared_image']),
                        'camera': int(match.group(1)), 'object_label': int(row['anomaly_class'] != 'OK'),
                        'exclude_reasons': [], 'annotation_issues': [], 'partition': 'excluded'})
    by_path = {r['raw_image']: r for r in records}
    if len(by_path) != len(records) or len({r['id'] for r in records}) != len(records):
        raise RuntimeError('Repeated manifest entries')
    observed = {r['id'] for r in records if r['object_label'] and not r['raw_mask']}
    if observed != listed_ids:
        raise RuntimeError('Audited null-mask membership differs from the explicit Protocol B list')
    protected_groups = {r['group'] for r in records if r['id'] in listed_ids}
    for r in records:
        r['is_ng_without_mask'] = r['id'] in listed_ids
        r['protocol_b_image_label'] = protocol_b_label(r, listed_ids)
        r['protected_test_group'] = r['group'] in protected_groups
        if r['is_ng_without_mask'] and r['source_split'] != 'test':
            raise RuntimeError('Protocol B null-mask view outside source test')
    checks = {}
    with (audit / 'file_checks.jsonl').open() as stream:
        for line in stream:
            value = json.loads(line)
            if value['path'] in by_path:
                checks[value['path']] = value
    if set(checks) != set(by_path):
        raise RuntimeError('Incomplete audit coverage')
    print('phase=quality_and_group_checks | source_views=%d' % len(records), flush=True)
    blank_paths = {r['path'] for r in rows(audit / 'blank_or_nearly_uniform_images.csv')}
    dimension_paths = {r['path'] for r in rows(audit / 'dimension_mismatches.csv')}
    if len(records) != cfg['expected_source_views'] or len({r['category'] for r in records}) != cfg['expected_source_categories']:
        raise RuntimeError('Source inventory differs from frozen expected coverage')
    for row in rows(audit / 'pixel_issues.csv'):
        if row['issue'] in ('degenerate_polygon', 'annotation_points_outside_canvas',
                            'annotation_dimension_mismatch', 'mask_dimension_mismatch'):
            by_path[row['path']]['annotation_issues'].append(row['issue'])
    for r in records:
        check = checks[r['raw_image']]
        r['raw_pixel_sha256'] = check['pixel_sha256']
        r['mask_foreground_raw'] = int(check.get('mask_nonzero', 0))
        if r['category'] in cfg['exclude_categories']:
            r['exclude_reasons'].append('substituted_normal_category')
        if r['category'] == 'audio_jack_socket' and r['source_class'].lower() == 'pit':
            r['exclude_reasons'].append('copied_audio_jack_pit_block')
        if r['raw_image'] in blank_paths:
            r['exclude_reasons'].append('blank_image_in_object')
        if r['group'] == 'toy_tire/OK/S0005':
            r['exclude_reasons'].append('normal_tire_annotation_conflict')
        if r['raw_image'] in dimension_paths:
            r['annotation_issues'].append('image_mask_dimension_mismatch')
        r['annotation_issues'] = sorted(set(r['annotation_issues']))

    def propagate():
        reasons = collections.defaultdict(set)
        for r in records:
            reasons[r['group']].update(r['exclude_reasons'])
        for r in records:
            r['exclude_reasons'] = sorted(reasons[r['group']])

    propagate()
    # Remove known wrong blocks first: retain the trusted battery-holder and switch copies.
    repeated = collections.defaultdict(list)
    for r in records:
        if not r['exclude_reasons'] or r['protected_test_group']:
            repeated[r['raw_pixel_sha256']].append(r)
    for same in repeated.values():
        if len(same) > 1:
            for r in same:
                r['exclude_reasons'].append('remaining_exact_duplicate_content')
    propagate()
    groups = collections.defaultdict(list)
    for r in records:
        groups[r['group']].append(r)
    for group, rs in groups.items():
        if len(rs) != 5 or sorted(r['camera'] for r in rs) != [1, 2, 3, 4, 5]:
            raise RuntimeError('Incomplete/repeated five-view group: ' + group)
        if len({(r['source_split'], r['object_label'], r['category']) for r in rs}) != 1:
            raise RuntimeError('Group crosses source membership: ' + group)

    checksums, runtime = build_cache(records, checks, cfg, output, workers=workers, resume=resume,
                                      compression=compression, progress_seconds=progress_seconds)
    processed_hashes = collections.defaultdict(list)
    for r in records:
        if 'input_pixel_sha256' in r:
            processed_hashes[r['input_pixel_sha256']].append(r)
    print('phase=duplicate_checks_and_splits', flush=True)
    # New duplicates can arise after RGB conversion/downsampling. Quarantine every involved object.
    for same in processed_hashes.values():
        if len(same) > 1:
            for r in same:
                r['exclude_reasons'].append('duplicate_model_input_pixels')
    propagate()
    for r in records:
        if r['exclude_reasons'] and not r['protected_test_group']:
            r.update(visible_eligible=False, pixel_eligible=False, support_eligible=False)
    counts = assign_partitions(records, cfg)
    if {r['id'] for r in records if r['partition'] == 'test' and r['is_ng_without_mask']} != listed_ids:
        raise RuntimeError('Not every listed Protocol B view is retained in test')
    if any(r['partition'] != 'test' for r in records if r['protected_test_group']):
        raise RuntimeError('Null-mask object split across training and test')
    if any(not r['pixel_eligible'] or r['protocol_b_image_label'] != 0 for r in records if r['is_ng_without_mask']):
        raise RuntimeError('Protocol B null-mask image/pixel scoring policy violated')
    records.sort(key=lambda r: (r['category'], r['group'], r['camera']))
    with (output / 'records.jsonl').open('w') as stream:
        for r in records:
            stream.write(json.dumps(r, sort_keys=True) + '\n')
    print('phase=write_manifests | cache_files=%d' % len(checksums), flush=True)
    if verify_written:
        verify_cache(output, checksums, progress_seconds=progress_seconds)
    with (output / 'cache_checksums.jsonl').open('w') as stream:
        for name, digest in sorted(checksums.items()):
            stream.write(json.dumps({'path': name, 'sha256': digest}) + '\n')
    write_csv(output / 'category_counts.csv', counts, list(counts[0]))
    exclusion_rows = [{'id': r['id'], 'group': r['group'], 'partition': r['partition'],
                       'reasons': '|'.join(r['exclude_reasons']),
                       'protocol_b_image_label': r['protocol_b_image_label'],
                       'is_ng_without_mask': r['is_ng_without_mask'],
                       'protected_test_group': r['protected_test_group'],
                       'annotation_issues': '|'.join(r['annotation_issues'])} for r in records]
    write_csv(output / 'decisions.csv', exclusion_rows, list(exclusion_rows[0]))
    write_csv(output / 'protocol_b_ng_without_masks.csv',
              [r for r in records if r['is_ng_without_mask']],
              ['id', 'group', 'category', 'partition', 'protocol_b_image_label', 'cache_train_mask', 'cache_eval_mask'])
    summary = {'created_at': now(), 'source_views': len(records), 'retained_categories': len(counts),
               'partitions_views': dict(collections.Counter(r['partition'] for r in records)),
               'partitions_objects': dict(collections.Counter(rs[0]['partition'] for rs in groups.values())),
               'scoring_protocol': 'B', 'ng_without_masks_in_test': len(listed_ids),
               'protocol_b_list_sha256': sha256(cfg['protocol_b_null_mask_list']),
               'protected_test_groups': len(protected_groups),
               'protocol_b_retained_quality_flagged_views': sum(bool(r['exclude_reasons']) for r in records if r['partition'] == 'test'),
               'quality_flags_views_overlapping': dict(collections.Counter(reason for r in records for reason in r['exclude_reasons'])),
               'exclusion_reasons_views_overlapping': dict(collections.Counter(
                   reason for r in records if r['partition'] == 'excluded' for reason in r['exclude_reasons'])),
               'object_only_test_groups': sorted(g for g, rs in groups.items()
                   if rs[0]['partition'] == 'test' and rs[0]['object_label'] == 1
                   and not any(r['visible_eligible'] for r in rs))}
    write_json(output / 'summary.json', summary)
    runtime.update(completed_at=now(), independent_disk_verification=verify_written)
    write_json(output / 'preparation_runtime.json', runtime)
    # Mutable resume/lock/progress files are operational state, never frozen manifests.
    artifact_names = ('records.jsonl', 'cache_checksums.jsonl', 'category_counts.csv', 'decisions.csv',
                      'protocol_b_ng_without_masks.csv', 'summary.json', 'preparation_runtime.json')
    artifacts = {name: sha256(output / name) for name in artifact_names}
    write_json(output / 'COMPLETE.json', {'created_at': now(), 'config_sha256': sha256(ROOT / 'config/experiment.json'),
                                        'artifacts': artifacts})
    write_json(output / 'progress.json', dict(status='COMPLETE', phase='complete', timestamp=now(), **runtime))
    print('Preparation COMPLETE: ' + str(output / 'COMPLETE.json'), flush=True)


def main():
    import fcntl
    from prepare_cache import other_prepare_pids, verify_cache
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4, help='Parallel CPU/IO workers (default: 4; try 2 for a busy disk)')
    parser.add_argument('--resume', action='store_true', help='Resume journal or adopt verified legacy cache; stop the old process first')
    parser.add_argument('--png-compress-level', type=int, default=1, choices=range(10), help='Lossless PNG compression; pixels are unchanged')
    parser.add_argument('--progress-seconds', type=float, default=5)
    parser.add_argument('--verify-cache', action='store_true', help='Independently reread every encoded cache file after writing (slower)')
    args = parser.parse_args()
    if not 1 <= args.workers <= 32 or args.progress_seconds <= 0:
        parser.error('--workers must be 1..32 and --progress-seconds must be positive')
    check_lock()
    active = other_prepare_pids(Path(__file__))
    if active:
        parser.error('Another preparation process is active (PID %s). Stop it in its original terminal before restarting.' % ', '.join(map(str, active)))
    cfg = config()
    target = ROOT / 'prepared'
    if (target / 'COMPLETE.json').exists():
        check_prepared()
        if args.verify_cache:
            entries = map(json.loads, (target / 'cache_checksums.jsonl').read_text().splitlines())
            verify_cache(target, {r['path']: r['sha256'] for r in entries}, args.progress_seconds)
        print('Preparation is already complete. No image transforms were run.')
        return
    if target.exists() and not args.resume:
        parser.error('Partial preparation exists. Stop the old process, then use: bash run.sh prepare --resume --workers 4')
    target.mkdir(exist_ok=True)
    # One writer per output directory. Keep the same inode while holding flock.
    with (target / '.prepare.lock').open('a+') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            parser.error('Another preparation writer holds this output folder')
        lock.seek(0)
        lock.truncate()
        lock.write(json.dumps(dict(pid=os.getpid(), started_at=now())) + '\n')
        lock.flush()
        try:
            prepare(cfg, target, workers=args.workers, resume=args.resume, compression=args.png_compress_level,
                    progress_seconds=args.progress_seconds, verify_written=args.verify_cache)
        except KeyboardInterrupt:
            print('Interrupted. Cache/journal preserved. Resume with: bash run.sh prepare --resume --workers 4', flush=True)
            raise SystemExit(130)


if __name__ == '__main__':
    main()
