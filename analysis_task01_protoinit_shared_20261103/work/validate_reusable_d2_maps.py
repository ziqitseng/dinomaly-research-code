import csv
import json
import os
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get('TASK01_OUTPUT_DIR', Path(__file__).resolve().parents[1])).resolve()
D2 = ROOT / 'Dinomaly2/out/re_eval_shared_20261031_cpu01'
manifest = json.loads((OUT / 'shared_test_manifest.json').read_text())
with open(D2 / 'maps/index.csv', newline='') as f:
    map_index = list(csv.DictReader(f))
with open(D2 / 'per_view_scores.csv', newline='') as f:
    old_views = list(csv.DictReader(f))
with open(D2 / 'per_object_scores.csv', newline='') as f:
    old_objects = list(csv.DictReader(f))

fresh = {(r['category'], r['image_path']): r for r in manifest}
idx = {(r['category'], r['image_path']): r for r in map_index}
view = {(r['category'], r['image_path']): r for r in old_views}
errors, cats = [], {}
if len(fresh) != len(manifest): errors.append('fresh_manifest_duplicate_category_image')
if len(idx) != len(map_index): errors.append('d2_map_index_duplicate_category_image')
if set(idx) != set(fresh):
    errors.append(f'map_membership_mismatch missing={len(set(fresh)-set(idx))} extra={len(set(idx)-set(fresh))}')
if set(view) != set(fresh):
    errors.append(f'view_score_membership_mismatch missing={len(set(fresh)-set(view))} extra={len(set(view)-set(fresh))}')
if len(old_objects) != 664:
    errors.append(f'expected 664 D2 object rows, got {len(old_objects)}')

for category in ('lattice_block_plug', 'laser_diode', 'power_jack'):
    p = D2 / 'maps' / f'{category}.npz'
    with np.load(p, allow_pickle=False) as z:
        maps = z['postprocessed_maps']
        masks = z['gt_masks_256']
        paths = z['image_paths'].tolist()
    rows = [r for r in manifest if r['category'] == category]
    if maps.shape != (len(rows), 256, 256) or masks.shape != maps.shape:
        errors.append(f'{category}: unexpected map/mask shape {maps.shape}/{masks.shape}')
    if len(set(paths)) != len(paths): errors.append(f'{category}: duplicate paths in map NPZ')
    if set(paths) != {r['image_path'] for r in rows}:
        errors.append(f'{category}: map NPZ path membership mismatch')
    if not np.isfinite(maps).all(): errors.append(f'{category}: nonfinite map values')
    pos_empty = []
    for i, path in enumerate(paths):
        r = fresh.get((category, path))
        ix = idx.get((category, path)); vr = view.get((category, path))
        if r is None or ix is None or vr is None: continue
        if ix['object_key'] != r['object_key'] or ix['camera_view_id'] != r['camera']:
            errors.append(f'{category}/{path}: D2 object/view key mismatch')
        if int(vr['gt_view_label']) != r['label']:
            errors.append(f'{category}/{path}: D2 label mismatch')
        if (vr['gt_mask_path'] or '') != (r['mask_path'] or ''):
            errors.append(f'{category}/{path}: D2 mask-path mismatch')
        flat = maps[i].reshape(-1)
        k = int(flat.size * .01)
        score = float(np.partition(flat, -k)[-k:].mean())
        if abs(score - float(vr['top1pct_view_score'])) > 1e-6:
            errors.append(f'{category}/{path}: D2 view score does not recompute')
        if r['label'] == 0 and masks[i].any(): errors.append(f'{category}/{path}: normal mask is nonempty')
        if r['label'] == 1 and not masks[i].any(): pos_empty.append(path)
    cats[category] = {'views': len(rows), 'objects': len({r['object_key'] for r in rows}),
                      'map_shape': list(maps.shape), 'paths_match_fresh_source_manifest': set(paths) == {r['image_path'] for r in rows},
                      'per_view_scores_recomputed': True, 'empty_positive_masks': pos_empty}

report = {'status': 'FAIL' if errors else 'PASS_WITH_SOURCE_MASK_EXCEPTIONS' if any(v['empty_positive_masks'] for v in cats.values()) else 'PASS',
          'fresh_manifest': str(OUT / 'shared_test_manifest.json'),
          'reused_D2_map_source': str(D2 / 'maps'),
          'restriction': 'Only DiNomaly2 anomaly-map arrays and their score rows were reused; no ArcAD files or results from legacy were read.',
          'categories': cats, 'errors': errors}
(OUT / 'd2_map_reuse_validation.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'status': report['status'], 'errors': errors, 'categories': cats}, indent=2))
if errors: raise SystemExit(1)
