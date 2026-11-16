"""Shared metadata utilities. This module never imports a model or starts work."""
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
    result = read_json(ROOT / 'config/experiment.json')
    for key in ('arcad_repo', 'dinomaly2_repo', 'backbone_file', 'protocol_b_null_mask_list'):
        result[key] = str((ROOT / result[key]).resolve())
    return result


def check_lock():
    lock = read_json(ROOT / 'provenance/input_lock.json')
    for entry in lock['files']:
        path = Path(entry['path'])
        if not path.is_absolute():
            path = ROOT / path
        if not path.is_file() or sha256(path) != entry['sha256']:
            raise RuntimeError('Frozen input changed or missing: ' + str(path))
    return sha256(ROOT / 'provenance/input_lock.json')


def check_prepared():
    check_lock()
    meta = read_json(ROOT / 'prepared/COMPLETE.json')
    if meta['config_sha256'] != sha256(ROOT / 'config/experiment.json'):
        raise RuntimeError('Configuration differs from prepared data')
    for name, digest in meta['artifacts'].items():
        if sha256(ROOT / 'prepared' / name) != digest:
            raise RuntimeError('Prepared artifact changed: ' + name)
    return meta


def group_id(path):
    """Keep separators, class, category, and sample id; never concatenate components."""
    return str(Path(path).parent).replace('\\', '/')


def load_records():
    with (ROOT / 'prepared/records.jsonl').open() as stream:
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
    with Image.open(ROOT / 'prepared' / record[key]) as mask:
        array = np.asarray(mask).copy()
    if hashlib.sha256(array.tobytes()).hexdigest() != record[key + '_pixel_sha256']:
        raise RuntimeError('Mask cache changed: ' + record['id'])
    return array > 0


def run_dir(model, seed):
    return ROOT / 'runs' / model / ('seed_%d' % seed)
