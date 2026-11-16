import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path('/media/avlab/7458a474-6ea2-4b90-b232-373c3df8d329/real_iad_variety_1024')
OUT = Path(__file__).resolve().parents[1]
CATEGORIES = ['lattice_block_plug', 'laser_diode', 'power_jack', 'LED_indicator', 'aircraft_model_head']
KNOWN = {'lattice_block_plug': (1115, 223, 198, 25), 'laser_diode': (1085, 217, 192, 25), 'power_jack': (1120, 224, 199, 25)}


def rel(r): return r['image_path'].replace('\\', '/')


def object_key(path):
    p = path.split('/')
    return '/'.join(p[:2] if p[0] == 'OK' else p[:3])


def resolve(cat, path):
    a = ROOT / cat / path
    if a.is_file(): return a
    return ROOT / cat / cat / path


rows, report, errors = [], {'categories': {}, 'source_root': str(ROOT)}, []
for cat in CATEGORIES:
    ap = ROOT / 'realiadvariety_jsons/sup' / f'{cat}.json'
    dp = ROOT / 'realiadvariety_jsons/Real-IAD_Variety_jsons' / f'{cat}.json'
    a, d = json.loads(ap.read_text()), json.loads(dp.read_text())
    at, dt = {rel(x): x for x in a['test']}, {rel(x): x for x in d['test']}
    labeled_by_path = {rel(x): x for x in a['labeled']}
    if len(at) != len(a['test']) or len(dt) != len(d['test']) or len(labeled_by_path) != len(a['labeled']):
        errors.append(f'{cat}: duplicate source path')
    labeled_objects = {object_key(p) for p in labeled_by_path}
    common = sorted(set(at) & set(dt))
    shared = []
    excluded = []
    for p in common:
        ak = object_key(p)
        if ak in labeled_objects:
            excluded.append(p)
            continue
        ar, dr = at[p], dt[p]
        label = int(ar['label'])
        if label != int(bool(dr.get('mask_path'))): errors.append(f'{cat}: source label mismatch {p}')
        if ar.get('mask_path') != dr.get('mask_path'): errors.append(f'{cat}: source mask mismatch {p}')
        img_path = resolve(cat, p)
        mask_path = resolve(cat, ar['mask_path']) if ar.get('mask_path') else None
        if not img_path.is_file(): errors.append(f'{cat}: missing image {img_path}')
        if label and (mask_path is None or not mask_path.is_file()): errors.append(f'{cat}: missing positive mask {mask_path}')
        camera = re.search(r'_C([1-5])_', Path(p).name)
        if not camera: errors.append(f'{cat}: camera parse failed {p}')
        pieces = p.split('/')
        row = {'category': cat, 'object_key': ak, 'image_path': p,
               'mask_path': ar.get('mask_path'), 'label': label,
               'defect_type': pieces[1] if pieces[0] == 'NG' else 'OK',
               'anomaly_class': ar.get('source_anomaly_class', ar.get('anomaly_class', '')),
               'camera_view_id': int(camera.group(1)) if camera else -1,
               'image_abs': str(img_path), 'mask_abs': str(mask_path) if mask_path else '',
               'split_id': 'shared_split_task02_v1'}
        shared.append(row)
    grouped = {}
    for r in shared: grouped.setdefault(r['object_key'], []).append(r)
    bad = {o: len(v) for o, v in grouped.items() if len(v) != 5 or len({x['camera_view_id'] for x in v}) != 5}
    if bad: errors.append(f'{cat}: invalid five-view objects {list(bad.items())[:5]}')
    pos_obj = sum(max(x['label'] for x in v) for v in grouped.values())
    neg_obj = len(grouped) - pos_obj
    counts = (len(shared), len(grouped), pos_obj, neg_obj)
    if cat in KNOWN and counts != KNOWN[cat]: errors.append(f'{cat}: known counts mismatch {counts}')
    rows.extend(shared)
    report['categories'][cat] = {'arcad_manifest': str(ap), 'd2_manifest': str(dp),
        'arcad_labeled_images': len(a['labeled']), 'arcad_labeled_objects': len(labeled_objects),
        'arcad_test_images': len(a['test']), 'd2_test_images': len(d['test']),
        'shared_images': counts[0], 'shared_objects': counts[1],
        'positive_objects': pos_obj, 'normal_objects': neg_obj,
        'excluded_labeled_objects': sorted(labeled_objects), 'excluded_shared_path_count': len(excluded),
        'checks': {'five_views': not bool(bad), 'no_labeled_object_overlap': not bool(set(grouped) & labeled_objects)}}

rows.sort(key=lambda r: (CATEGORIES.index(r['category']), r['image_path']))
if errors: report['status'] = 'FAIL'
else: report['status'] = 'PASS'
OUT.joinpath('shared_test_manifest.json').write_text(json.dumps(rows, indent=2) + '\n')
report['total_views'] = len(rows)
report['total_objects'] = sum(x['shared_objects'] for x in report['categories'].values())
report['errors'] = errors
report['manifest_sha256'] = hashlib.sha256(OUT.joinpath('shared_test_manifest.json').read_bytes()).hexdigest()
OUT.joinpath('manifest_validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'status': report['status'], 'categories': {k: {q:v[q] for q in ['shared_images','shared_objects','positive_objects','normal_objects']} for k,v in report['categories'].items()}, 'errors': errors}, indent=2))
if errors: raise SystemExit(1)
