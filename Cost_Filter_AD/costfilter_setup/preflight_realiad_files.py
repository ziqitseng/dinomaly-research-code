"""Audit every original Real-IAD manifest path before a long run."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path


SAMPLE = re.compile(r'([a-zA-Z][a-zA-Z0-9_]*_[0-9]{4}_[A-Z][A-Z_]*[A-Z])_C[0-9]_')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-path', type=Path, required=True)
    parser.add_argument('--dtd-path', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest_dir = args.data_path / 'realiad_jsons/realiad_jsons'
    files = sorted(manifest_dir.glob('*.json'))
    if len(files) != 30:
        raise ValueError('Expected 30 original category manifests; got %d' % len(files))
    textures = list(args.dtd_path.glob('*/*.jpg'))
    if not textures:
        raise FileNotFoundError('No DTD textures found in %s' % args.dtd_path)
    summary = {'protocol': 'original Real-IAD normal-only split',
               'dtd_textures': len(textures), 'categories': {},
               'missing_files': [], 'complete': False}
    for manifest in files:
        category = manifest.stem
        entries = json.loads(manifest.read_text())
        if set(entries) != {'meta', 'train', 'test'}:
            raise ValueError('Unexpected manifest keys in %s' % manifest)
        train_ids, test_ids = set(), set()
        test_group_sizes = Counter()
        counts = {'train_views': 0, 'test_views': 0,
                  'normal_test_views': 0, 'anomalous_test_views': 0}
        image_dir = args.data_path / 'realiad_1024' / category
        for phase in ('train', 'test'):
            for sample in entries[phase]:
                match = SAMPLE.search(sample['image_path'])
                if not match:
                    raise ValueError('Invalid Real-IAD filename: %s' % sample['image_path'])
                group = match.group(1)
                (train_ids if phase == 'train' else test_ids).add(group)
                if phase == 'train' and sample['anomaly_class'] != 'OK':
                    raise ValueError('Anomalous training view in original split: %s' % manifest)
                counts[phase + '_views'] += 1
                image_path = image_dir / sample['image_path']
                if not image_path.is_file():
                    summary['missing_files'].append(str(image_path))
                if phase == 'test':
                    test_group_sizes[group] += 1
                    if sample['anomaly_class'] == 'OK':
                        counts['normal_test_views'] += 1
                    else:
                        counts['anomalous_test_views'] += 1
                        mask_path = image_dir / sample['mask_path']
                        if not mask_path.is_file():
                            summary['missing_files'].append(str(mask_path))
        if train_ids & test_ids:
            raise ValueError('Train/test object IDs overlap for %s' % category)
        if set(test_group_sizes.values()) != {5}:
            raise ValueError('Incomplete five-view object in %s' % category)
        counts['train_objects'] = len(train_ids)
        counts['test_objects'] = len(test_ids)
        summary['categories'][category] = counts
        print(category, counts, 'missing_files:', len(summary['missing_files']), flush=True)
    summary['totals'] = {key: sum(category[key] for category in summary['categories'].values())
                         for key in next(iter(summary['categories'].values()))}
    summary['complete'] = not summary['missing_files']
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2) + '\n')
    print('Total:', summary['totals'], 'DTD textures:', len(textures),
          'missing files:', len(summary['missing_files']), flush=True)
    if summary['missing_files']:
        raise FileNotFoundError('%d dataset files are missing' % len(summary['missing_files']))


if __name__ == '__main__':
    main()
