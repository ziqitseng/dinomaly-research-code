"""Shared metadata utilities. This module never imports a model or starts work."""
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE_ROOT = Path(os.environ.get('COLD_START_BASE_ROOT',
    ROOT.parent / 'comparison_cold_start_20261109')).resolve()
PREPARED = BASE_ROOT / 'prepared'


def config_path():
    schedule = os.environ.get('STUDY_SCHEDULE', '100k')
    if schedule not in ('100k', '50k'):
        raise ValueError('STUDY_SCHEDULE must be 100k or 50k')
    return ROOT / 'config' / ('experiment_' + schedule + '.json')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')
    temp.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def rows(path):
    with Path(path).open(newline='') as stream:
        yield from csv.DictReader(stream)


def write_csv(path, data, fields):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(data)


def config():
    result = read_json(config_path())
    for key in ('arcad_repo', 'dinomaly2_repo', 'backbone_file', 'protocol_b_null_mask_list'):
        result[key] = str((BASE_ROOT / result[key]).resolve())
    return result


def check_lock():
    lock = read_json(BASE_ROOT / 'provenance/input_lock.json')
    for entry in lock['files']:
        path = Path(entry['path'])
        if not path.is_absolute():
            path = BASE_ROOT / path
        if not path.is_file() or sha256(path) != entry['sha256']:
            raise RuntimeError('Frozen input changed or missing: ' + str(path))
    return sha256(BASE_ROOT / 'provenance/input_lock.json')


def check_prepared():
    check_lock()
    meta = read_json(PREPARED / 'COMPLETE.json')
    if meta['config_sha256'] != sha256(BASE_ROOT / 'config/experiment.json'):
        raise RuntimeError('Frozen baseline configuration differs from prepared data')
    for name, digest in meta['artifacts'].items():
        if sha256(PREPARED / name) != digest:
            raise RuntimeError('Prepared artifact changed: ' + name)
    return meta


def group_id(path):
    """Keep separators, class, category, and sample id; never concatenate components."""
    return str(Path(path).parent).replace('\\', '/')


def load_records():
    with (PREPARED / 'records.jsonl').open() as stream:
        return [json.loads(line) for line in stream]


def protocol_b_ids(cfg):
    listed = list(rows(cfg['protocol_b_null_mask_list']))
    ids = {r['id'] for r in listed}
    if len(ids) != len(listed) or len(ids) != cfg['expected_ng_without_masks']:
        raise RuntimeError('Protocol B null-mask list has duplicates or incorrect coverage')
    return ids


def protocol_b_label(record, listed_ids):
    return 0 if record['id'] in listed_ids else record['object_label']


def load_cached_mask(record, key):
    import numpy as np
    from PIL import Image
    with Image.open(PREPARED / record[key]) as mask:
        array = np.asarray(mask).copy()
    if hashlib.sha256(array.tobytes()).hexdigest() != record[key + '_pixel_sha256']:
        raise RuntimeError('Mask cache changed: ' + record['id'])
    return array > 0


def run_dir(model, seed):
    return ROOT / 'runs' / os.environ.get('STUDY_SCHEDULE', '100k') / model / ('seed_%d' % seed)
