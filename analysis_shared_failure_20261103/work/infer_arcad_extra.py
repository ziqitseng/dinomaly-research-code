import csv
import hashlib
import json
import os
import sys
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1]
ARCAD = ROOT / 'ArcAD'
CKPT = ARCAD / 'out/real_iad_variety out/arcad_realiad_variety_protoinit/model.pth'
sys.path.insert(0, str(ARCAD))
from models.uad import ViTill
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from utils import cal_anomaly_maps, get_gaussian_kernel

CATEGORIES = ['LED_indicator', 'aircraft_model_head']


def sha(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def build_model(device):
    from dinov2.models import vision_transformer as v2_vit
    encoder = v2_vit.vit_base(patch_size=14, img_size=518, block_chunks=0,
                              init_values=1e-8, num_register_tokens=4,
                              interpolate_antialias=False, interpolate_offset=0.1)
    bottle = nn.ModuleList([bMlp(768, 768 * 4, 768, drop=0.4)])
    decoder = nn.ModuleList([VitBlock(dim=768, num_heads=12, mlp_ratio=4., qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-8), attn_drop=0., attn=LinearAttention2) for _ in range(8)])
    model = ViTill(encoder=encoder, bottleneck=bottle, decoder=decoder,
        target_layers=[2,3,4,5,6,7,8,9], fuse_layer_encoder=[[0,1,2,3],[4,5,6,7]],
        fuse_layer_decoder=[[0,1,2,3],[4,5,6,7]], mask_neighbor_size=0)
    state = torch.load(CKPT, map_location='cpu', weights_only=False)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), sum(p.numel() for p in model.parameters())


def main():
    os.chdir(ARCAD)
    manifest = json.loads((OUT / 'shared_test_manifest.json').read_text())
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    hash_before, mtime_before = sha(CKPT), CKPT.stat().st_mtime_ns
    model, nparams = build_model(device)
    img_tf = transforms.Compose([transforms.Resize((448,448)), transforms.ToTensor(),
        transforms.CenterCrop(392), transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    mask_tf = transforms.Compose([transforms.Resize((448,448)), transforms.CenterCrop(392), transforms.ToTensor()])
    gaussian = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)
    outdir = OUT / 'maps/arcad'; outdir.mkdir(parents=True, exist_ok=True)
    all_scores = []
    for cat in CATEGORIES:
        rows = [r for r in manifest if r['category'] == cat]
        rows.sort(key=lambda r:r['image_path'])
        maps, masks, scores = [], [], []
        for start in range(0, len(rows), 2):
            batch = rows[start:start+2]
            xs, gs = [], []
            for r in batch:
                xs.append(img_tf(Image.open(r['image_abs']).convert('RGB')))
                if r['label'] and r['mask_abs']:
                    gs.append(mask_tf(Image.open(r['mask_abs']).convert('L')))
                else:
                    gs.append(torch.zeros(1,392,392))
            x, gt = torch.stack(xs).to(device), torch.stack(gs).to(device)
            with torch.inference_mode():
                en, de, _ = model(x)
                amap, _ = cal_anomaly_maps(en, de, x.shape[-1])
                amap = F.interpolate(amap, size=256, mode='bilinear', align_corners=False)
                gt = F.interpolate(gt, size=256, mode='nearest')
                amap = gaussian(amap)
                flat = amap.flatten(1)
                k = int(flat.shape[1] * .01)
                sc = torch.sort(flat, dim=1, descending=True)[0][:,:k].mean(1).cpu().numpy()
            maps.extend(amap[:,0].cpu().numpy().astype(np.float32))
            masks.extend(gt[:,0].bool().cpu().numpy().astype(np.uint8))
            scores.extend(sc.tolist())
            if start % 100 == 0: print(f'{cat}: {min(start+2,len(rows))}/{len(rows)}', flush=True)
        mp, gp = np.stack(maps), np.stack(masks)
        np.savez_compressed(outdir/f'{cat}.npz', maps=mp, masks=gp,
                            image_paths=np.asarray([r['image_path'] for r in rows]))
        for r, s in zip(rows, scores):
            all_scores.append({'category':cat,'object_key':r['object_key'],'image_path':r['image_path'],
                'camera_view_id':r['camera_view_id'],'gt_view_label':r['label'],
                'gt_object_label':max(x['label'] for x in rows if x['object_key']==r['object_key']),
                'score_view':s,'gt_mask_path':r['mask_path'] or ''})
    with open(OUT/'arcad_extra_per_view.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(all_scores[0])); w.writeheader(); w.writerows(all_scores)
    after=sha(CKPT); mtime_after=CKPT.stat().st_mtime_ns
    if hash_before != after or mtime_before != mtime_after: raise RuntimeError('ArcAD checkpoint changed')
    (OUT/'arcad_extra_integrity.json').write_text(json.dumps({'checkpoint':str(CKPT),'sha256_before':hash_before,
        'sha256_after':after,'mtime_ns_before':mtime_before,'mtime_ns_after':mtime_after,
        'strict_load':True,'parameter_count':nparams,'device':str(device)},indent=2)+'\n')
    print('ArcAD extra-category inference complete')


if __name__=='__main__': main()
