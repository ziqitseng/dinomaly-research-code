import csv
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1]
D2 = ROOT / 'Dinomaly2'
DATA = Path('/media/avlab/7458a474-6ea2-4b90-b232-373c3df8d329/real_iad_variety_1024')
CKPT = D2 / 'out/real_iad_variety_1024 out/full run/variety_160cat/model.pth'
CATEGORIES = ['LED_indicator', 'aircraft_model_head']
sys.path.insert(0, str(D2))
from dinomaly2_inference import make_model
from utils import cal_anomaly_maps, get_gaussian_kernel


def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()


def main():
    os.chdir(D2)
    torch.set_num_threads(4)
    manifest=json.loads((OUT/'shared_test_manifest.json').read_text())
    device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    before,mtime=sha(CKPT),CKPT.stat().st_mtime_ns
    model=make_model(device)
    state=torch.load(CKPT,map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True); model.eval()
    img_tf=transforms.Compose([transforms.Resize((280,280)),transforms.ToTensor(),
        transforms.CenterCrop(280),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    mask_tf=transforms.Compose([transforms.Resize((280,280)),transforms.CenterCrop(280),transforms.ToTensor()])
    gaussian=get_gaussian_kernel(kernel_size=5,sigma=4).to(device)
    outdir=OUT/'maps/d2'; outdir.mkdir(parents=True,exist_ok=True)
    all_scores=[]
    for cat in CATEGORIES:
        rows=sorted([r for r in manifest if r['category']==cat],key=lambda r:r['image_path'])
        maps,masks,scores=[],[],[]
        for start in range(0,len(rows),2):
            batch=rows[start:start+2]; xs=[]; gs=[]
            for r in batch:
                xs.append(img_tf(Image.open(r['image_abs']).convert('RGB')))
                if r['label'] and r['mask_abs']:
                    gs.append(mask_tf(Image.open(r['mask_abs']).convert('L')))
                else: gs.append(torch.zeros(1,280,280))
            x,gt=torch.stack(xs).to(device),torch.stack(gs).to(device)
            with torch.inference_mode():
                en,de=model(x)
                amap,_=cal_anomaly_maps(en,de,x.shape[-1])
                amap=F.interpolate(amap,size=256,mode='bilinear',align_corners=False)
                gt=F.interpolate(gt,size=256,mode='nearest')
                amap=gaussian(amap)
                flat=amap.flatten(1); k=int(flat.shape[1]*.01)
                sc=torch.sort(flat,dim=1,descending=True)[0][:,:k].mean(1).cpu().numpy()
            maps.extend(amap[:,0].cpu().numpy().astype(np.float32))
            masks.extend(gt[:,0].bool().cpu().numpy().astype(np.uint8))
            scores.extend(sc.tolist())
            if start%100==0: print(f'{cat}: {min(start+2,len(rows))}/{len(rows)}',flush=True)
        np.savez_compressed(outdir/f'{cat}.npz',maps=np.stack(maps),masks=np.stack(masks),
                            image_paths=np.asarray([r['image_path'] for r in rows]))
        object_labels={o:max(r['label'] for r in rows if r['object_key']==o) for o in {r['object_key'] for r in rows}}
        for r,s in zip(rows,scores):
            all_scores.append({'category':cat,'object_key':r['object_key'],'image_path':r['image_path'],
                'camera_view_id':r['camera_view_id'],'gt_view_label':r['label'],
                'gt_object_label':object_labels[r['object_key']],'score_view':s,'gt_mask_path':r['mask_path'] or ''})
    with open(OUT/'d2_extra_per_view.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(all_scores[0]));w.writeheader();w.writerows(all_scores)
    after,mtime_after=sha(CKPT),CKPT.stat().st_mtime_ns
    if before!=after or mtime!=mtime_after: raise RuntimeError('D2 checkpoint changed')
    (OUT/'d2_extra_integrity.json').write_text(json.dumps({'checkpoint':str(CKPT),'sha256_before':before,
        'sha256_after':after,'mtime_ns_before':mtime,'mtime_ns_after':mtime_after,'strict_load':True,
        'parameter_count':sum(p.numel() for p in model.parameters()),'device':str(device)},indent=2)+'\n')
    print('DiNomaly2 extra-category inference complete')


if __name__=='__main__': main()
