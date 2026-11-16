#!/usr/bin/env python3
"""Inference-only numeric map export for one already-trained model."""
import argparse,csv,hashlib,json,sys,time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score,average_precision_score
from torchvision import transforms

HERE=Path(__file__).resolve().parent
OUT=HERE.parent
DATA=Path('/media/avlab/7458a474-6ea2-4b90-b232-373c3df8d329/real_iad_variety_1024')
D2REPO=Path('/media/avlab/1CE6451EE644FA12/2026_10_22_dinomaly/Dinomaly2')
ARCADREPO=Path('/media/avlab/1CE6451EE644FA12/2026_10_22_dinomaly/ArcAD')
MANIFEST=OUT/'shared_manifest.json'
ARC_CKPT=ARCADREPO/'out/real_iad_variety out/arcad_realiad_variety/model.pth'
D2_CKPT=D2REPO/'out/real_iad_variety_1024 out/full run/variety_160cat/model.pth'
CATS=('lattice_block_plug','laser_diode','power_jack','LED_indicator','thyristor')
TOP=0.01

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',choices=('arcad','d2'),required=True)
    ap.add_argument('--categories',nargs='+',default=None)
    ap.add_argument('--batch-size',type=int,default=8)
    a=ap.parse_args()
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable in this process')
    device=torch.device('cuda:0'); torch.manual_seed(0); np.random.seed(0)
    cats=a.categories or list(CATS)
    if set(cats)-set(CATS): raise ValueError('Unknown categories '+str(set(cats)-set(CATS)))
    manifest=json.loads(MANIFEST.read_text())
    if a.model=='arcad':
        sys.path.insert(0,str(ARCADREPO))
        from dataset import get_data_transforms
        from models import vit_encoder
        from models.uad import ViTill
        from models.vision_transformer import Block as VitBlock,bMlp,LinearAttention2
        from functools import partial
        from utils import cal_anomaly_maps,get_gaussian_kernel,compute_pro,f1_score_max
        size,crop,map_size=448,392,256
        data_t,gt_t=get_data_transforms(size,crop)
        encoder=vit_encoder.load('dinov2reg_vit_base_14')
        emb,heads=768,12
        bottleneck=torch.nn.ModuleList([bMlp(emb,emb*4,emb,drop=0.4)])
        decoder=torch.nn.ModuleList([VitBlock(dim=emb,num_heads=heads,mlp_ratio=4.0,qkv_bias=True,norm_layer=partial(torch.nn.LayerNorm,eps=1e-8),attn_drop=0.0,attn=LinearAttention2) for _ in range(8)])
        model=ViTill(encoder=encoder,bottleneck=bottleneck,decoder=decoder,target_layers=[2,3,4,5,6,7,8,9],mask_neighbor_size=0,fuse_layer_encoder=[[0,1,2,3],[4,5,6,7]],fuse_layer_decoder=[[0,1,2,3],[4,5,6,7]])
        ckpt=ARC_CKPT
        state=torch.load(ckpt,map_location='cpu',weights_only=True)
        model.load_state_dict(state,strict=True)
        gaussian=get_gaussian_kernel(kernel_size=5,sigma=4).to(device)
        metric_pixel='sklearn exact AUROC/AP/max-F1; ArcAD compute_pro (200 thresholds, FPR cap .3)'
    else:
        sys.path.insert(0,str(D2REPO))
        from dinomaly2_inference import make_model
        from dataset import get_data_transforms
        from utils import cal_anomaly_maps,get_gaussian_kernel,f1_score_max
        size,crop,map_size=280,280,256
        data_t,gt_t=get_data_transforms(size,crop)
        model=make_model(device)
        ckpt=D2_CKPT
        state=torch.load(ckpt,map_location='cpu',weights_only=True)
        model.load_state_dict(state,strict=True)
        gaussian=get_gaussian_kernel(kernel_size=5,sigma=4).to(device)
        from adeval.iterative import EvalAccumulator
        metric_pixel='CPU EvalAccumulator 1000-strip histogram (intended mirror of D2 CUDA accumulator)'
    model=model.to(device).eval()
    checkpoint_hash_before=sha(ckpt); checkpoint_mtime_before=ckpt.stat().st_mtime_ns
    mapdir=OUT/'maps'; mapdir.mkdir(exist_ok=True)
    all_rows=[]; cat_metrics={}; start=time.time()
    mean=[0.485,0.456,0.406]; std=[0.229,0.224,0.225]
    transform=transforms.Compose([transforms.Resize((size,size)),transforms.ToTensor(),transforms.CenterCrop(crop),transforms.Normalize(mean,std)])
    gt_transform=transforms.Compose([transforms.Resize((size,size)),transforms.CenterCrop(crop),transforms.ToTensor()])
    for cat in cats:
        data=manifest['categories'][cat]
        rows=sorted(data['shared_test_images'],key=lambda r:r['image_path'])
        if len(rows)!=data['counts']['shared_views']: raise ValueError('Manifest count mismatch '+cat)
        map_chunks=[]; mask_chunks=[]; view_rows=[]
        for start_i in range(0,len(rows),a.batch_size):
            chunk=rows[start_i:start_i+a.batch_size]
            images=[]; masks=[]
            for r in chunk:
                ip=DATA/cat/r['image_path']
                with Image.open(ip) as im: images.append(transform(im.convert('RGB')))
                if r['mask_path']:
                    with Image.open(DATA/cat/r['mask_path']) as m: masks.append(gt_transform(m.convert('L')))
                else: masks.append(torch.zeros((1,crop,crop),dtype=torch.float32))
            x=torch.stack(images).to(device,non_blocking=True)
            gt=torch.stack(masks).to(device,non_blocking=True)
            with torch.inference_mode():
                result=model(x)
                if a.model=='arcad': en,de=result[0],result[1]
                else: en,de=result[0],result[1]
                amap,_=cal_anomaly_maps(en,de,x.shape[-1])
                amap=F.interpolate(amap,size=(map_size,map_size),mode='bilinear',align_corners=False)
                amap=gaussian(amap)
                gtmap=F.interpolate(gt,size=(map_size,map_size),mode='nearest').bool()
            amap=amap[:,0].cpu().numpy().astype(np.float32)
            gtmap=gtmap[:,0].cpu().numpy().astype(np.uint8)
            map_chunks.append(amap); mask_chunks.append(gtmap)
            for j,r in enumerate(chunk):
                flat=amap[j].ravel(); k=max(1,int(flat.size*TOP)); score=float(np.partition(flat,flat.size-k)[-k:].mean())
                view_rows.append({'category':cat,'model':a.model,'object_key':r['object_key'],'image_path':r['image_path'],'view_id':r['view_id'],'gt_view_label':r['label'],'gt_object_label':next(o['label'] for o in data['shared_test_objects'] if o['object_key']==r['object_key']),'defect_type':r['anomaly_class'],'gt_mask_path':r['mask_path'] or '', 'score_view':score, 'map_path':str((mapdir/f'{a.model}_{cat}.npz').resolve()),'map_index':start_i+j,'model_map_shape':'256x256','map_coordinates':f'{size}->{map_size}; model_input_crop={crop}'})
        maps=np.concatenate(map_chunks); masks=np.concatenate(mask_chunks)
        paths=np.asarray([r['image_path'] for r in rows])
        np.savez_compressed(mapdir/f'{a.model}_{cat}.npz',smoothed_maps=maps,model_masks=masks,image_paths=paths,map_size=np.asarray([map_size,map_size]),input_size=np.asarray([size,size]),crop_size=np.asarray([crop,crop]),gaussian=np.asarray([5,4]))
        obj_groups={}
        for i,r in enumerate(rows): obj_groups.setdefault(r['object_key'],[]).append(i)
        yv=np.asarray([r['label'] for r in rows],dtype=np.uint8); sv=np.asarray([r['score_view'] for r in view_rows],dtype=np.float64)
        yo=[]; so=[]; object_rows=[]
        for key,idx in sorted(obj_groups.items()):
            pooled=maps[idx].reshape(-1); k=max(1,int(pooled.size*TOP)); score=float(np.partition(pooled,pooled.size-k)[-k:].mean())
            label=max(rows[i]['label'] for i in idx); yo.append(label); so.append(score)
            object_rows.append({'category':cat,'model':a.model,'object_key':key,'gt_object_label':label,'score_object':score,'view_indices':'|'.join(map(str,idx)),'view_paths':'|'.join(rows[i]['image_path'] for i in idx)})
            for i in idx: view_rows[i]['score_object']=score
        if a.model=='arcad':
            py=np.concatenate(masks).ravel(); ps=maps.ravel()
            cat_metrics[cat]={'I-AUROC':float(roc_auc_score(yv,sv)),'I-AP':float(average_precision_score(yv,sv)),'I-oracle_max_F1':float(f1_score_max(yv,sv)),'O-AUROC':float(roc_auc_score(yo,so)),'O-AP':float(average_precision_score(yo,so)),'O-oracle_max_F1':float(f1_score_max(yo,so)),'n_views':len(rows),'n_objects':len(yo),'P-AUROC':float(roc_auc_score(masks.ravel(),ps)),'P-AP':float(average_precision_score(masks.ravel(),ps)),'P-oracle_max_F1':float(f1_score_max(masks.ravel(),ps)),'P-AUPRO':float(compute_pro(masks,maps))}
        else:
            from adeval.iterative import EvalAccumulator
            lo,hi=float(maps.min()),float(maps.max()); hi=hi if hi>lo else lo+1e-6
            acc=EvalAccumulator(lo,hi,lo,hi); acc.add_anomap_batch(maps,masks); pm=acc.summary()
            cat_metrics[cat]={'I-AUROC':float(roc_auc_score(yv,sv)),'I-AP':float(average_precision_score(yv,sv)),'I-oracle_max_F1':float(f1_score_max(yv,sv)),'O-AUROC':float(roc_auc_score(yo,so)),'O-AP':float(average_precision_score(yo,so)),'O-oracle_max_F1':float(f1_score_max(yo,so)),'n_views':len(rows),'n_objects':len(yo),'P-AUROC':float(pm['p_auroc']),'P-AP':float(pm['p_aupr']),'P-oracle_max_F1':float(pm['p_f1max']),'P-AUPRO':float(pm['p_aupro'])}
        cat_metrics[cat]['pixel_protocol']=metric_pixel
        all_rows.extend(view_rows)
        with open(OUT/f'{a.model}_per_view_scores.csv','w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(all_rows[0]));w.writeheader();w.writerows([r for r in all_rows if r['model']==a.model])
        with open(OUT/f'{a.model}_per_object_scores.csv','w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=list(object_rows[0]));w.writeheader();w.writerows(object_rows)
        print(a.model,cat,len(rows),len(yo),cat_metrics[cat],flush=True)
    after=sha(ckpt)
    report={'model':a.model,'checkpoint':str(ckpt.resolve()),'checkpoint_sha256_before':checkpoint_hash_before,'checkpoint_sha256_after':after,'checkpoint_mtime_ns_before':checkpoint_mtime_before,'checkpoint_mtime_ns_after':ckpt.stat().st_mtime_ns,'checkpoint_unchanged':checkpoint_hash_before==after and checkpoint_mtime_before==ckpt.stat().st_mtime_ns,'device':torch.cuda.get_device_name(0),'input_resize':size,'center_crop':crop,'map_size':map_size,'gaussian':{'kernel':5,'sigma':4},'score_top_percent':1.0,'category_metrics':cat_metrics,'elapsed_seconds':time.time()-start}
    (OUT/f'{a.model}_export_metadata.json').write_text(json.dumps(report,indent=2)+'\n')

if __name__=='__main__':main()
