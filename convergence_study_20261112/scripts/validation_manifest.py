"""Inspect or explicitly create one fixed object-disjoint validation holdout."""
import argparse
import collections
import hashlib
import json
from pathlib import Path

from common import PREPARED, check_prepared, load_records, now, sha256


def rank(group, seed):
    return hashlib.sha256(('%d/%s' % (seed, group)).encode()).hexdigest()


def propose(records, prepared_hash, seed=20261112, per_class=5):
    if per_class < 1:
        raise ValueError('per_class must be positive')
    groups = collections.defaultdict(list)
    for row in records:
        groups[row['group']].append(row)
    test = {g: rs for g, rs in groups.items() if rs[0]['partition'] == 'test'}
    categories = sorted({rs[0]['category'] for rs in test.values()})
    if len(categories) != 159:
        raise RuntimeError('Original 159-category coverage changed')
    chosen, counts = [], []
    for category in categories:
        candidates = [rs for rs in test.values() if rs[0]['category'] == category]
        eligible = {0: [], 1: []}
        for rs in candidates:
            label = rs[0]['object_label']
            if (len(rs) != 5 or sorted(r['camera'] for r in rs) != [1, 2, 3, 4, 5]
                    or any(r['protected_test_group'] or r['is_ng_without_mask'] or r['exclude_reasons']
                           or r['source_split'] != 'test' or not r['pixel_eligible'] for r in rs)):
                continue
            if label == 1 and (any(not r['raw_mask'] or not r['visible_eligible']
                                   or r['visible_label'] != 1 or r['mask_foreground_raw'] <= 0
                                   for r in rs)):
                continue
            eligible[label].append(rs[0]['group'])
        if (len(eligible[0]) < per_class or len(eligible[1]) < per_class
                or sum(rs[0]['object_label'] == 0 for rs in candidates) <= per_class
                or sum(rs[0]['object_label'] == 1 for rs in candidates) <= per_class):
            raise RuntimeError('Insufficient disjoint validation and final-test classes: ' + category)
        selected = [g for label in (0, 1) for g in
                    sorted(eligible[label], key=lambda g: rank(g, seed))[:per_class]]
        chosen.extend(selected)
        counts.append(dict(category=category, eligible_normal=len(eligible[0]),
                           eligible_anomalous=len(eligible[1]), selected_groups=selected))
    selected = set(chosen)
    final = sorted(set(test) - selected)
    protected = {g for g, rs in test.items() if any(r['protected_test_group'] for r in rs)}
    if not protected <= set(final):
        raise RuntimeError('Protected object moved out of final test')
    return dict(created_at=now(), method='%d source-test normal and %d fully annotated anomaly objects per category; SHA256 deterministic rank' % (per_class, per_class),
                selection_seed=seed, prepared_sha256=prepared_hash,
                objects_per_class_per_category=per_class,
                validation_groups=sorted(selected), test_groups=final,
                categories=categories, candidate_counts=counts,
                protected_test_groups=len(protected),
                protected_missing_mask_views=sum(r['is_ng_without_mask'] for g in protected for r in test[g]),
                validation_objects=len(selected), final_test_objects=len(final))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=20261112)
    parser.add_argument('--write', type=Path, help='Explicit new output path; omitted means inspect only')
    args = parser.parse_args()
    check_prepared()
    proposal = propose(load_records(), sha256(PREPARED / 'COMPLETE.json'), args.seed)
    print(json.dumps({k: proposal[k] for k in ('selection_seed', 'validation_objects',
        'final_test_objects', 'protected_test_groups', 'protected_missing_mask_views')}, indent=2))
    print('eligible_normal_min=%d eligible_anomalous_min=%d' % (
        min(r['eligible_normal'] for r in proposal['candidate_counts']),
        min(r['eligible_anomalous'] for r in proposal['candidate_counts'])))
    if args.write:
        # Exclusive creation: no prior split or result can be silently replaced.
        with args.write.expanduser().open('x') as stream:
            json.dump(proposal, stream, indent=2, sort_keys=True)
            stream.write('\n')
        print('Wrote ' + str(args.write))


if __name__ == '__main__':
    main()
