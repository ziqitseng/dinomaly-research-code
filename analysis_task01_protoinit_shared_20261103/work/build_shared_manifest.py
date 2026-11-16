import hashlib
import json
import os
import re
from pathlib import Path

ROOT = Path('/media/avlab/7458a474-6ea2-4b90-b232-373c3df8d329/real_iad_variety_1024')
OUT = Path(os.environ.get('TASK01_OUTPUT_DIR', Path(__file__).resolve().parents[1])).resolve()
CATEGORIES = ['lattice_block_plug', 'laser_diode', 'power_jack']


def image_rel(rec):
    return rec['image_path'].replace('\\', '/')


def object_key(rec):
    p = image_rel(rec).split('/')
    return '/'.join(p[:2] if p[0] == 'OK' else p[:3])


def resolve(category, rel):
    p = ROOT / category / rel
    if p.is_file():
        return p
    p = ROOT / category / category / rel
    return p


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


report = {'status': 'PASS', 'root': str(ROOT), 'categories': {}, 'errors': []}
manifest = []
for category in CATEGORIES:
    arc_path = ROOT / 'realiadvariety_jsons' / 'sup' / f'{category}.json'
    d2_path = ROOT / 'realiadvariety_jsons' / 'Real-IAD_Variety_jsons' / f'{category}.json'
    arc = json.loads(arc_path.read_text())
    d2 = json.loads(d2_path.read_text())
    arc_test = {image_rel(x): x for x in arc['test']}
    arc_train = {image_rel(x): x for x in arc['labeled']}
    d2_test = {image_rel(x): x for x in d2['test']}
    train_objects = {object_key(x) for x in arc['labeled']}
    common = sorted((set(arc_test) & set(d2_test)) - set(arc_train))
    rows = []
    for rel in common:
        a, d = arc_test[rel], d2_test[rel]
        # The source variety protocol treats null-mask views as normal even if
        # their folder/anomaly_class says NG; this is also D2's loader rule.
        d_label = int(bool(d.get('mask_path')))
        if int(a['label']) != d_label:
            report['errors'].append(f'{category}: label mismatch {rel}')
        if a.get('mask_path') != d.get('mask_path'):
            report['errors'].append(f'{category}: mask mismatch {rel}')
        ip = resolve(category, rel)
        if not ip.is_file():
            report['errors'].append(f'{category}: missing image {ip}')
        mp = resolve(category, a['mask_path']) if a.get('mask_path') else None
        if int(a['label']) and (mp is None or not mp.is_file()):
            report['errors'].append(f'{category}: missing positive mask {mp}')
        rows.append({
            'category': category, 'object_key': object_key(a), 'image_path': rel,
            'mask_path': a.get('mask_path'), 'label': int(a['label']),
            'anomaly_class': a.get('anomaly_class'), 'image_abs': str(ip),
            'mask_abs': str(mp) if mp else '', 'camera': (re.search(r'_C([1-5])_', Path(rel).name).group(1) if re.search(r'_C([1-5])_', Path(rel).name) else ''),
        })
    by_obj = {}
    for row in rows:
        by_obj.setdefault(row['object_key'], []).append(row)
    bad_views = [k for k, v in by_obj.items() if len(v) != 5 or len({r['camera'] for r in v}) != 5]
    bad_labels = [k for k, v in by_obj.items() if max(r['label'] for r in v) != int(any(r['label'] for r in v))]
    overlap = sorted(set(by_obj) & train_objects)
    exp = {'lattice_block_plug': (1115, 223, 198, 25), 'laser_diode': (1085, 217, 192, 25), 'power_jack': (1120, 224, 199, 25)}[category]
    observed = (len(rows), len(by_obj), sum(max(r['label'] for r in v) == 1 for v in by_obj.values()), sum(max(r['label'] for r in v) == 0 for v in by_obj.values()))
    if observed != exp:
        report['errors'].append(f'{category}: expected counts {exp}, got {observed}')
    if bad_views or bad_labels or overlap:
        report['errors'].append(f'{category}: bad_views={bad_views[:3]}, bad_object_labels={bad_labels[:3]}, training_overlap={overlap}')
    report['categories'][category] = {
        'arcad_manifest': str(arc_path), 'd2_manifest': str(d2_path),
        'arcad_labeled_images': len(arc_train), 'arcad_labeled_objects': len(train_objects),
        'd2_native_test_images': len(d2_test), 'arcad_test_images': len(arc_test),
        'paired_images': observed[0], 'paired_objects': observed[1],
        'paired_defect_objects': observed[2], 'paired_normal_objects': observed[3],
        'excluded_arcad_labeled_images_in_d2_test': sorted(set(d2_test) & set(arc_train)),
        'excluded_arcad_labeled_objects': sorted(set(object_key(x) for rel, x in d2_test.items() if rel in arc_train)),
        'checks': {'5_views_per_object': not bad_views, 'labels_consistent_per_object': not bad_labels,
                   'no_arcad_labeled_object_overlap': not overlap, 'source_labels_masks_match': not any(e.startswith(category + ': label') or e.startswith(category + ': mask') for e in report['errors'])}
    }
    manifest.extend(rows)

if report['errors']:
    report['status'] = 'FAIL'
OUT.joinpath('shared_test_manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
report['manifest_path'] = str(OUT / 'shared_test_manifest.json')
report['manifest_sha256'] = sha(OUT / 'shared_test_manifest.json')
OUT.joinpath('manifest_validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'status': report['status'], 'counts': {k: v['paired_images'] for k, v in report['categories'].items()}, 'errors': report['errors']}, indent=2))
if report['errors']:
    raise SystemExit(1)
