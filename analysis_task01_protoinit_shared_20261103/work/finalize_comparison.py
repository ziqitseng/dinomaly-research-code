import csv
import hashlib
import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get('TASK01_OUTPUT_DIR', Path(__file__).resolve().parents[1])).resolve()
D2 = ROOT / 'Dinomaly2/out/re_eval_shared_20261031_cpu01'
ARCAD = ROOT / 'ArcAD'
CATEGORIES = ['lattice_block_plug', 'laser_diode', 'power_jack']
METRICS = ['I-AUROC', 'I-AP', 'I-F1_oracle_max', 'P-AUROC', 'P-AP', 'P-F1_oracle_max', 'P-AUPRO', 'O-AUROC', 'O-AP', 'O-F1_oracle_max']
NATIVE = {'lattice_block_plug': 0.5423, 'laser_diode': 0.6078, 'power_jack': 0.6639}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def f1max(y, s):
    p, r, _ = precision_recall_curve(y, s)
    f = 2 * p * r / (p + r + 1e-7)
    return float(np.nanmax(f[:-1]))


def metrics(y_img, s_img, masks, maps, obj_y, obj_s, compute_pro):
    ypx = masks.reshape(-1)
    spx = maps.reshape(-1)
    return {
        'I-AUROC': float(roc_auc_score(y_img, s_img)),
        'I-AP': float(average_precision_score(y_img, s_img)),
        'I-F1_oracle_max': f1max(y_img, s_img),
        'P-AUROC': float(roc_auc_score(ypx, spx)),
        'P-AP': float(average_precision_score(ypx, spx)),
        'P-F1_oracle_max': f1max(ypx, spx),
        'P-AUPRO': float(compute_pro(masks.astype(np.uint8), maps.astype(np.float32), num_th=200)),
        'O-AUROC': float(roc_auc_score(obj_y, obj_s)),
        'O-AP': float(average_precision_score(obj_y, obj_s)),
        'O-F1_oracle_max': f1max(obj_y, obj_s),
    }


def main():
    spec = importlib.util.spec_from_file_location('arcad_eval', Path(__file__).resolve().parent / 'evaluate_arcad_protoinit.py')
    arcad_eval = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(arcad_eval)
    compute_pro = arcad_eval.fast_compute_pro
    manifest = json.loads((OUT / 'shared_test_manifest.json').read_text())
    split_report = json.loads((OUT / 'manifest_validation.json').read_text())
    reuse_report = json.loads((OUT / 'd2_map_reuse_validation.json').read_text())
    if split_report.get('status') != 'PASS' or reuse_report.get('status') == 'FAIL':
        raise RuntimeError('Fresh split or DiNomaly2 map membership validation did not pass')
    fresh = {(r['category'], r['image_path']): r for r in manifest}
    d2_score_rows, arc_score_rows = [], []
    for f in [D2 / 'per_view_scores.csv', OUT / 'arcad_per_view_scores.csv']:
        with open(f, newline='') as h: rows = list(csv.DictReader(h))
        if len(rows) != len(manifest): raise RuntimeError(f'{f}: expected {len(manifest)} rows, got {len(rows)}')
        if {(r['category'], r['image_path']) for r in rows} != set(fresh):
            raise RuntimeError(f'{f}: sample keys differ from fresh manifest')
        (d2_score_rows if f.parent == D2 else arc_score_rows).extend(rows)

    arc_maps = json.loads((OUT / 'arcad_shared_metrics.json').read_text())
    arc_obj_rows = list(csv.DictReader(open(OUT / 'arcad_per_object_scores.csv', newline='')))
    d2_meta = json.loads((D2 / 'run_metadata.json').read_text())
    d2_checkpoint = Path(d2_meta['checkpoint']['path'])
    if sha(d2_checkpoint) != d2_meta['checkpoint']['sha256']:
        raise RuntimeError('D2 checkpoint no longer matches the hash recorded for its reusable maps')

    model_metrics = {'ArcAD_protoinit_shared': {}, 'DiNomaly2_shared': {}}
    d2_view_map = {(r['category'], r['image_path']): r for r in d2_score_rows}
    arc_view_map = {(r['category'], r['image_path']): r for r in arc_score_rows}
    arc_obj_map = {(r['category'], r['object_key']): r for r in arc_obj_rows}
    d2_obj_rows = []
    output_validation = {'status': 'PASS', 'categories': {}, 'errors': []}
    # Bundle the validated D2 numeric maps with this independent result folder.
    d2_bundle = OUT / 'dinomaly2_maps'
    d2_bundle.mkdir(exist_ok=True)
    for cat in CATEGORIES:
        shutil.copy2(D2 / 'maps' / f'{cat}.npz', d2_bundle / f'{cat}.npz')
    shutil.copy2(D2 / 'maps/index.csv', d2_bundle / 'index.csv')
    with open(OUT / 'arcad_maps/index.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['category', 'index', 'image_path', 'object_key', 'camera_view_id', 'label', 'mask_path', 'map_file'])
        for cat in CATEGORIES:
            cat_rows = [r for r in manifest if r['category'] == cat]
            for i, r in enumerate(cat_rows):
                w.writerow([cat, i, r['image_path'], r['object_key'], r['camera'], r['label'], r['mask_path'] or '', f'{cat}.npz'])
    for cat in CATEGORIES:
        rows = [r for r in manifest if r['category'] == cat]
        d2_path = D2 / 'maps' / f'{cat}.npz'
        with np.load(d2_path, allow_pickle=False) as z:
            d2_maps = z['postprocessed_maps'].astype(np.float32, copy=False)
            d2_masks = z['gt_masks_256'].astype(np.uint8, copy=False)
            d2_paths = z['image_paths'].tolist()
        if set(d2_paths) != {r['image_path'] for r in rows}:
            raise RuntimeError(f'{cat}: D2 map membership mismatch')
        with np.load(OUT / 'arcad_maps' / f'{cat}.npz', allow_pickle=False) as z:
            arc_maps_np, arc_masks_np = z['maps'], z['masks']
        if arc_maps_np.shape != (len(rows), 256, 256) or arc_masks_np.shape != arc_maps_np.shape:
            raise RuntimeError(f'{cat}: unexpected ArcAD map/mask shape {arc_maps_np.shape}/{arc_masks_np.shape}')
        if not np.isfinite(arc_maps_np).all(): raise RuntimeError(f'{cat}: nonfinite ArcAD maps')
        d2_order = {p: i for i, p in enumerate(d2_paths)}
        d2_y, d2_s = [], []
        arc_y, arc_s = [], []
        arc_score_err = 0.0
        transformed_empty, raw_empty = [], []
        for ai, r in enumerate(rows):
            key = (cat, r['image_path'])
            dv, av = d2_view_map[key], arc_view_map[key]
            if int(dv['gt_view_label']) != r['label'] or int(av['label']) != r['label']:
                raise RuntimeError(f'{cat}/{r["image_path"]}: label mismatch')
            d2_y.append(r['label']); d2_s.append(float(dv['top1pct_view_score']))
            arc_y.append(r['label']); arc_s.append(float(av['top1pct_image_score']))
            if (av['mask_path'] or '') != (r['mask_path'] or ''):
                raise RuntimeError(f'{cat}/{r["image_path"]}: ArcAD mask path mismatch')
            k_view = int(arc_maps_np[ai].size * .01)
            recomputed = float(np.partition(arc_maps_np[ai].reshape(-1), -k_view)[-k_view:].mean())
            arc_score_err = max(arc_score_err, abs(recomputed - float(av['top1pct_image_score'])))
            if r['label'] == 0 and arc_masks_np[ai].any():
                raise RuntimeError(f'{cat}/{r["image_path"]}: ArcAD normal mask is nonempty')
            if r['label'] == 1 and not arc_masks_np[ai].any():
                transformed_empty.append(r['image_path'])
                if not np.asarray(Image.open(r['mask_abs']).convert('L')).any():
                    raw_empty.append(r['image_path'])
        d2_y, d2_s = np.asarray(d2_y), np.asarray(d2_s)
        arc_y, arc_s = np.asarray(arc_y), np.asarray(arc_s)
        groups = {}
        for i, r in enumerate(rows): groups.setdefault(r['object_key'], []).append(i)
        d2_obj_y, d2_obj_s, arc_obj_y, arc_obj_s = [], [], [], []
        for obj, indices in sorted(groups.items()):
            if len(indices) != 5: raise RuntimeError(f'{cat}/{obj}: expected five views')
            d2_idx = [d2_order[rows[i]['image_path']] for i in indices]
            dpool = d2_maps[d2_idx].reshape(-1)
            k = int(dpool.size * .01)
            dscore = float(np.partition(dpool, -k)[-k:].mean())
            arc_obj = arc_obj_map[(cat, obj)]
            apool = arc_maps_np[indices].reshape(-1)
            ak = int(apool.size * .01)
            ascore = float(np.partition(apool, -ak)[-ak:].mean())
            if abs(ascore - float(arc_obj['pooled_top1pct_5view_score'])) > 1e-6:
                raise RuntimeError(f'{cat}/{obj}: ArcAD pooled object score mismatch')
            label = max(rows[i]['label'] for i in indices)
            d2_obj_y.append(label); d2_obj_s.append(dscore)
            arc_obj_y.append(label); arc_obj_s.append(ascore)
            d2_obj_rows.append({'category': cat, 'object_key': obj, 'label': label,
                                'pooled_top1pct_5view_score': dscore, 'view_count': 5})
        d2_metrics = metrics(d2_y, d2_s, d2_masks, d2_maps, np.asarray(d2_obj_y), np.asarray(d2_obj_s), compute_pro)
        arc_provided = arc_maps['categories'][cat]
        # ArcAD evaluation already computed these metrics from its raw numeric maps;
        # compare its independently saved view/object scores and protocol outputs.
        arc_metrics = {**arc_provided['image_view'], **arc_provided['pixel'], **arc_provided['object_pooled_5view']}
        model_metrics['DiNomaly2_shared'][cat] = d2_metrics
        model_metrics['ArcAD_protoinit_shared'][cat] = arc_metrics
        output_validation['categories'][cat] = {
            'arcad_map_shape': list(arc_maps_np.shape), 'd2_map_shape': list(d2_maps.shape),
            'all_maps_finite': True, 'arcad_view_scores_recomputed_max_abs_error': arc_score_err,
            'positive_views_with_empty_masks_after_arcad_transform': transformed_empty,
            'positive_views_with_empty_source_png_masks': raw_empty,
        }

    for model_name in model_metrics:
        model_metrics[model_name]['macro_mean'] = {
            m: float(np.mean([model_metrics[model_name][c][m] for c in CATEGORIES])) for m in METRICS
        }
    (OUT / 'shared_metrics.json').write_text(json.dumps(model_metrics, indent=2) + '\n')
    if any(v['positive_views_with_empty_masks_after_arcad_transform'] for v in output_validation['categories'].values()):
        output_validation['status'] = 'PASS_WITH_SOURCE_MASK_EXCEPTIONS'
    (OUT / 'output_validation.json').write_text(json.dumps(output_validation, indent=2) + '\n')
    with open(OUT / 'shared_metrics.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['model', 'category'] + METRICS)
        for name in ('ArcAD_protoinit_shared', 'DiNomaly2_shared'):
            for cat in CATEGORIES + ['macro_mean']:
                w.writerow([name, cat] + [model_metrics[name][cat][m] for m in METRICS])
    with open(OUT / 'paired_comparison.csv', 'w', newline='') as f:
        fields = ['category']
        for m in METRICS: fields += [f'ArcAD_protoinit_{m}', f'D2_{m}', f'D2_minus_ArcAD_{m}']
        fields += ['D2_original_native_test_I-AUROC_reference']
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for cat in CATEGORIES:
            row = {'category': cat, 'D2_original_native_test_I-AUROC_reference': NATIVE[cat]}
            for m in METRICS:
                a, d = model_metrics['ArcAD_protoinit_shared'][cat][m], model_metrics['DiNomaly2_shared'][cat][m]
                row[f'ArcAD_protoinit_{m}'] = a; row[f'D2_{m}'] = d
                row[f'D2_minus_ArcAD_{m}'] = '' if m.startswith('P-') else d - a
            w.writerow(row)

    # Replace generic D2 object table with rows whose membership and label were
    # checked against the freshly built source manifest and scores recomputed from maps.
    with open(OUT / 'd2_per_object_scores.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(d2_obj_rows[0])); w.writeheader(); w.writerows(d2_obj_rows)
    env = subprocess.run([sys.executable, '-c', 'import torch, torchvision, numpy, sklearn; print(torch.__version__); print(torchvision.__version__); print(numpy.__version__); print(sklearn.__version__)'], capture_output=True, text=True)
    git = {}
    for name, path in [('ArcAD', ARCAD), ('DiNomaly2', ROOT / 'Dinomaly2')]:
        g = subprocess.run(['git', '-C', str(path), 'rev-parse', 'HEAD'], capture_output=True, text=True)
        git[name] = g.stdout.strip() if g.returncode == 0 else None
    out_meta = {
        'status': 'completed_with_documented_source_mask_exception',
        'timestamp_local': datetime.now().astimezone().isoformat(),
        'scope': 'Task 01 low-three shared-split inference-only comparison',
        'legacy_exclusion': 'No file under ArcAD/.../legacy or the top-level legacy folder was used. ArcAD paired results were inferred from the new protoinit model.pth.',
        'fresh_manifest': str(OUT / 'shared_test_manifest.json'),
        'manifest_sha256': sha(OUT / 'shared_test_manifest.json'),
        'ArcAD': {'checkpoint': str(ARCAD / 'out/real_iad_variety out/arcad_realiad_variety_protoinit/model.pth'),
                  'checkpoint_sha256': json.loads((OUT / 'arcad_integrity.json').read_text())['sha256_after'],
                  'log': str(ARCAD / 'out/real_iad_variety out/arcad_realiad_variety_protoinit/log.txt'),
                  'prototypes_loaded_confirmed_in_log': True,
                  'full_run_results_csv_not_used_for_paired_metrics': str(ARCAD / 'out/real_iad_variety out/arcad_realiad_variety_protoinit/results.csv'),
                  'preprocess': '448x448 resize, tensor, center crop 392, ImageNet normalize; 256x256 bilinear map, 5x5 Gaussian sigma=4; top 1% mean per view'},
        'DiNomaly2': {'checkpoint': str(d2_checkpoint), 'checkpoint_sha256': sha(d2_checkpoint),
                      'map_arrays_reused_from': str(D2 / 'maps'),
                      'map_arrays_bundled_at': str(d2_bundle),
                      'reuse_verified_against_fresh_manifest': True,
                      'preprocess': '280x280 input/crop; 256x256 map, 5x5 Gaussian sigma=4; top 1% view and pooled five-view object scores'},
        'sample_counts': {c: len([r for r in manifest if r['category'] == c]) for c in CATEGORIES},
        'metric_protocol': {'I/O': 'sklearn AUROC/AP and oracle max-F1; object score pooled top 1% across concatenated five-view postprocessed maps',
                            'pixel': 'sklearn exact curves and same ArcAD compute_pro implementation for both maps, 200 thresholds, FPR cap 0.3; pixel deltas suppressed because preprocessing/map-frame transforms differ',
                            'macro': 'unweighted arithmetic mean of the three category metrics'},
        'git_commits': git, 'environment_versions': env.stdout.strip().splitlines(),
        'platform': platform.platform(), 'device': 'CPU' if not torch.cuda.is_available() else 'CUDA',
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in [
            ARCAD / 'arcad_realiad_uni.py', ARCAD / 'dataset.py', ARCAD / 'utils.py',
            ARCAD / 'out/real_iad_variety out/arcad_realiad_variety_protoinit/log.txt',
            ARCAD / 'out/real_iad_variety out/prototype_reproduction_20261102/prototypes/prototypes_init.pth',
            ROOT / 'Dinomaly2/dinomaly_multiview.py', ROOT / 'Dinomaly2/utils.py',
            OUT / 'work/build_shared_manifest.py', OUT / 'work/evaluate_arcad_protoinit.py',
            OUT / 'work/validate_reusable_d2_maps.py', OUT / 'work/finalize_comparison.py']},
        'd2_native_reference': NATIVE,
    }
    (OUT / 'run_metadata.json').write_text(json.dumps(out_meta, indent=2) + '\n')
    print(json.dumps({'status': 'finalized', 'metrics': model_metrics}, indent=2))


if __name__ == '__main__': main()
