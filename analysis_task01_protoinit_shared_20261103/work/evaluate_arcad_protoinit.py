import csv
import hashlib
import json
import os
import re
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score, auc
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(os.environ.get('TASK01_OUTPUT_DIR', Path(__file__).resolve().parents[1])).resolve()
ARCAD = ROOT / 'ArcAD'
CHECKPOINT = ARCAD / 'out/real_iad_variety out/arcad_realiad_variety_protoinit/model.pth'
sys.path.insert(0, str(ARCAD))

from models.uad import ViTill
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from utils import cal_anomaly_maps, compute_pro, f1_score_max, get_gaussian_kernel


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def build_model(device):
    # Instantiate the exact backbone architecture without fetching pretrained
    # weights: the full trained model checkpoint strictly replaces all weights.
    from dinov2.models import vision_transformer as v2_vit
    encoder = v2_vit.vit_base(patch_size=14, img_size=518, block_chunks=0,
                              init_values=1e-8, num_register_tokens=4,
                              interpolate_antialias=False, interpolate_offset=0.1)
    embed_dim, num_heads = 768, 12
    bottleneck = nn.ModuleList([bMlp(embed_dim, embed_dim * 4, embed_dim, drop=0.4)])
    decoder = nn.ModuleList([
        VitBlock(dim=embed_dim, num_heads=num_heads, mlp_ratio=4., qkv_bias=True,
                 norm_layer=partial(nn.LayerNorm, eps=1e-8), attn_drop=0., attn=LinearAttention2)
        for _ in range(8)
    ])
    model = ViTill(encoder=encoder, bottleneck=bottleneck, decoder=decoder,
                   target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
                   fuse_layer_encoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
                   fuse_layer_decoder=[[0, 1, 2, 3], [4, 5, 6, 7]],
                   mask_neighbor_size=0)
    before = sha(CHECKPOINT)
    mtime_before = CHECKPOINT.stat().st_mtime_ns
    state = torch.load(CHECKPOINT, map_location='cpu', weights_only=False)
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(f'non-strict checkpoint load: {result}')
    if CHECKPOINT.stat().st_mtime_ns != mtime_before:
        raise RuntimeError('ArcAD checkpoint modification time changed while loading')
    return model.to(device).eval(), before, mtime_before, sum(p.numel() for p in model.parameters())


def topk_mean(x, ratio=0.01):
    flat = x.reshape(x.shape[0], -1)
    k = int(flat.shape[1] * ratio)
    return torch.sort(flat, dim=1, descending=True)[0][:, :k].mean(dim=1)


def fast_compute_pro(masks, amaps, num_th=200):
    """Equivalent to ArcAD/utils.py::compute_pro with cached components/bins."""
    masks = (masks > 0.5).astype(np.uint8, copy=False)
    lo, hi = float(amaps.min()), float(amaps.max())
    norm = (amaps - lo) / (hi - lo) if hi - lo > 1e-8 else np.zeros_like(amaps)
    thresholds = np.linspace(0, 1, num_th)
    structure = np.ones((3, 3), dtype=np.uint8)
    region_hists, neg_hist = [], np.zeros(num_th + 1, dtype=np.int64)
    total_neg = 0
    for mask, score in zip(masks, norm):
        labels, count = ndimage.label(mask, structure=structure)
        flat_l, flat_m, flat_s = labels.ravel(), mask.ravel(), score.ravel()
        neg_s = flat_s[~flat_m.astype(bool)]
        total_neg += neg_s.size
        neg_bins = np.searchsorted(thresholds, neg_s, side='left')
        neg_hist += np.bincount(neg_bins, minlength=num_th + 1)
        if count:
            pos = flat_l > 0
            bins = np.searchsorted(thresholds, flat_s[pos], side='left')
            ids = flat_l[pos].astype(np.int64) - 1
            region_hists.append(np.bincount(ids * (num_th + 1) + bins,
                                             minlength=count * (num_th + 1)).reshape(count, num_th + 1))
    if not region_hists or total_neg == 0:
        return 0.0
    hist = np.concatenate(region_hists, axis=0)
    region_area = hist.sum(axis=1)
    pro = np.asarray([(hist[:, j + 1:].sum(axis=1) / region_area).mean() for j in range(num_th)])
    fpr = np.asarray([neg_hist[j + 1:].sum() / total_neg for j in range(num_th)])
    order = np.argsort(fpr)
    fpr, pro = fpr[order], pro[order]
    keep = fpr <= 0.3
    fpr, pro = fpr[keep], pro[keep]
    return float(auc(fpr / 0.3, pro)) if fpr.size > 1 and fpr.max() > 0 else 0.0


def main():
    os.chdir(ARCAD)
    records = json.loads((OUT / 'shared_test_manifest.json').read_text())
    if len(records) != 3320:
        raise RuntimeError(f'Expected 3320 records, got {len(records)}')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model, checkpoint_sha_before, checkpoint_mtime_before, parameter_count = build_model(device)
    image_tf = transforms.Compose([
        transforms.Resize((448, 448)), transforms.ToTensor(), transforms.CenterCrop(392),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    mask_tf = transforms.Compose([transforms.Resize((448, 448)), transforms.CenterCrop(392), transforms.ToTensor()])
    gaussian = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    by_cat = {}
    for r in records:
        by_cat.setdefault(r['category'], []).append(r)
    all_view_rows = []
    all_object_rows = []
    metrics = {}
    batch_size = 4 if device.type == 'cuda' else 2

    for category, cat_records in by_cat.items():
        maps, masks, labels, scores, map_rows = [], [], [], [], []
        cache_path = OUT / 'arcad_maps' / f'{category}.npz'
        resumed = False
        if cache_path.exists():
            with np.load(cache_path, allow_pickle=False) as z:
                cached_maps = z['maps']
            if cached_maps.shape != (len(cat_records), 256, 256):
                raise RuntimeError(f'{category}: invalid cached map shape {cached_maps.shape}')
            maps = [x for x in cached_maps]
            # Rebuild GT masks from the current source manifest using ArcAD's
            # original mask transform and bool() semantics (any nonzero pixel).
            for r in cat_records:
                if r['label'] and r['mask_abs']:
                    g = mask_tf(Image.open(r['mask_abs']).convert('L')).unsqueeze(0)
                    g = F.interpolate(g, size=256, mode='nearest')[0, 0].numpy()
                    masks.append((g > 0).astype(np.uint8))
                else:
                    masks.append(np.zeros((256, 256), dtype=np.uint8))
            map_rows = list(cat_records)
            labels = [int(r['label']) for r in cat_records]
            scores = [float(np.partition(x.reshape(-1), -int(x.size * .01))[-int(x.size * .01):].mean()) for x in cached_maps]
            for r, s in zip(cat_records, scores):
                all_view_rows.append({'category': category, 'object_key': r['object_key'],
                    'image_path': r['image_path'], 'camera_view_id': r['camera'], 'label': int(r['label']),
                    'top1pct_image_score': s, 'mask_path': r['mask_path'] or ''})
            cached_masks = np.stack(masks)
            np.savez_compressed(cache_path, maps=cached_maps, masks=cached_masks)
            resumed = True
        for start in range(0, 0 if resumed else len(cat_records), batch_size):
            batch = cat_records[start:start + batch_size]
            imgs, gts = [], []
            for r in batch:
                imgs.append(image_tf(Image.open(r['image_abs']).convert('RGB')))
                if r['label'] and r['mask_abs']:
                    gts.append(mask_tf(Image.open(r['mask_abs']).convert('L')))
                else:
                    gts.append(torch.zeros(1, 392, 392))
            x = torch.stack(imgs).to(device)
            gt = torch.stack(gts).to(device)
            with torch.inference_mode():
                en, de, _ = model(x)
                amap, _ = cal_anomaly_maps(en, de, x.shape[-1])
                amap = F.interpolate(amap, size=256, mode='bilinear', align_corners=False)
                gt = F.interpolate(gt, size=256, mode='nearest')
                amap = gaussian(amap)
                sc = topk_mean(amap).cpu().numpy()
            a_np = amap[:, 0].float().cpu().numpy()
            g_np = gt[:, 0].cpu().numpy().astype(bool).astype(np.uint8)
            for r, a, g, s in zip(batch, a_np, g_np, sc):
                if not np.isfinite(a).all() or not np.isfinite(s):
                    raise RuntimeError(f'nonfinite map or score: {r["image_path"]}')
                maps.append(a.astype(np.float32, copy=False))
                masks.append(g)
                labels.append(int(r['label']))
                scores.append(float(s))
                map_rows.append(r)
                all_view_rows.append({'category': category, 'object_key': r['object_key'],
                    'image_path': r['image_path'], 'camera_view_id': r['camera'],
                    'label': int(r['label']), 'top1pct_image_score': float(s),
                    'mask_path': r['mask_path'] or ''})
            if start % 100 == 0:
                print(f'{category}: {min(start + batch_size, len(cat_records))}/{len(cat_records)}', flush=True)

        maps_np = np.stack(maps)
        masks_np = np.stack(masks)
        labels_np, scores_np = np.asarray(labels), np.asarray(scores)
        np.savez_compressed(OUT / 'arcad_maps' / f'{category}.npz', maps=maps_np, masks=masks_np)
        # Match D2's pooled top-1% over the concatenated five-view map, for the
        # common object metric. The model-native ArcAD results.csv has no object metric.
        groups = {}
        for i, r in enumerate(map_rows):
            groups.setdefault(r['object_key'], []).append(i)
        obj_y, obj_s = [], []
        for obj, idxs in sorted(groups.items()):
            if len(idxs) != 5:
                raise RuntimeError(f'{category}/{obj}: expected 5 views, got {len(idxs)}')
            pooled = maps_np[idxs].reshape(1, -1)
            k = int(pooled.shape[1] * 0.01)
            pooled_score = float(np.partition(pooled, -k, axis=1)[:, -k:].mean())
            y = max(labels_np[idxs])
            obj_y.append(y); obj_s.append(pooled_score)
            all_object_rows.append({'category': category, 'object_key': obj, 'label': y,
                                    'pooled_top1pct_5view_score': pooled_score, 'view_count': 5})
        gtpx, prpx = masks_np.ravel(), maps_np.ravel()
        metrics[category] = {
            'image_view': {'I-AUROC': float(roc_auc_score(labels_np, scores_np)),
                           'I-AP': float(average_precision_score(labels_np, scores_np)),
                           'I-F1_oracle_max': float(f1_score_max(labels_np, scores_np))},
            'pixel': {'P-AUROC': float(roc_auc_score(gtpx, prpx)),
                      'P-AP': float(average_precision_score(gtpx, prpx)),
                      'P-F1_oracle_max': float(f1_score_max(gtpx, prpx)),
                      'P-AUPRO': fast_compute_pro(masks_np, maps_np, num_th=200)},
            'object_pooled_5view': {'O-AUROC': float(roc_auc_score(obj_y, obj_s)),
                                    'O-AP': float(average_precision_score(obj_y, obj_s)),
                                    'O-F1_oracle_max': float(f1_score_max(obj_y, obj_s))},
            'view_count': len(cat_records), 'object_count': len(groups),
            'empty_masks_on_positive_views': int(sum(y == 1 and m.sum() == 0 for y, m in zip(labels_np, masks_np)))
        }
        print(f'{category}: metrics complete', flush=True)

    (OUT / 'arcad_shared_metrics.json').write_text(json.dumps({
        'source_checkpoint': str(CHECKPOINT), 'source_checkpoint_sha256': checkpoint_sha_before,
        'protocol': 'new protoinit checkpoint; original ArcAD preprocessing and evaluation_batch scoring',
        'categories': metrics
    }, indent=2) + '\n')
    for filename, rows in [('arcad_per_view_scores.csv', all_view_rows), ('arcad_per_object_scores.csv', all_object_rows)]:
        with open(OUT / filename, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader(); w.writerows(rows)
    after = sha(CHECKPOINT)
    checkpoint_mtime_after = CHECKPOINT.stat().st_mtime_ns
    if checkpoint_sha_before != after or checkpoint_mtime_before != checkpoint_mtime_after:
        raise RuntimeError('ArcAD checkpoint changed during inference')
    (OUT / 'arcad_integrity.json').write_text(json.dumps({
        'checkpoint': str(CHECKPOINT), 'sha256_before': checkpoint_sha_before,
        'sha256_after': after, 'mtime_ns_before': checkpoint_mtime_before,
        'mtime_ns_after': checkpoint_mtime_after,
        'unchanged': checkpoint_sha_before == after and checkpoint_mtime_before == checkpoint_mtime_after,
        'strict_load': True, 'parameter_count': parameter_count,
        'device': str(device), 'batch_size': batch_size
    }, indent=2) + '\n')
    print('ArcAD inference complete')


if __name__ == '__main__':
    main()
