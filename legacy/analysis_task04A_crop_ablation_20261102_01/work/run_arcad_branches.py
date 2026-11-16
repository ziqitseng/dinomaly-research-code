#!/usr/bin/env python3
"""Run the two predeclared ArcAD inference geometries on the shared manifest."""
import csv, hashlib, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
ARCAD = ROOT / 'ArcAD'
sys.path.insert(0, str(ARCAD))
from models import vit_encoder
from models.uad import ViTill
from models.vision_transformer import Block as VitBlock, bMlp, LinearAttention2
from utils import cal_anomaly_maps, get_gaussian_kernel

DATA = Path('/media/avlab/7458a474-6ea2-4b90-b232-373c3df8d329/real_iad_variety_1024')
MANIFEST = ROOT / 'analysis_shared_failure_20261031_gpu01/shared_manifest.json'
CHECKPOINT = ARCAD / 'out/real_iad_variety out/arcad_realiad_variety/model.pth'
OUT = ROOT / 'analysis_task04A_crop_ablation_20261102_01'
CATEGORIES = ('laser_diode', 'lattice_block_plug', 'power_jack', 'LED_indicator', 'thyristor')
BATCH = 8
WORKERS = 4
MAP_SIZE = 256
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()

def build_model(device):
    encoder=vit_encoder.load('dinov2reg_vit_base_14')
    bottleneck=nn.ModuleList([bMlp(768,768*4,768,drop=0.4)])
    decoder=nn.ModuleList([VitBlock(dim=768,num_heads=12,mlp_ratio=4.0,qkv_bias=True,
        norm_layer=partial(nn.LayerNorm,eps=1e-8),attn_drop=0.0,attn=LinearAttention2) for _ in range(8)])
    model=ViTill(encoder=encoder,bottleneck=bottleneck,decoder=decoder,
        target_layers=[2,3,4,5,6,7,8,9],mask_neighbor_size=0,
        fuse_layer_encoder=[[0,1,2,3],[4,5,6,7]],fuse_layer_decoder=[[0,1,2,3],[4,5,6,7]])
    model.load_state_dict(torch.load(CHECKPOINT,map_location='cpu'),strict=True)
    return model.to(device).eval()

class Rows(Dataset):
    def __init__(self, rows, transform): self.rows,self.transform=rows,transform
    def __len__(self): return len(self.rows)
    def __getitem__(self,i):
        r=self.rows[i]; p=DATA/r['category']/r['image_path']
        if not p.is_file(): p=DATA/r['category']/r['category']/r['image_path']
        with Image.open(p) as im:
            image=im.convert('RGB')
            if image.size != (1024,1024): raise ValueError(f'Unexpected image size {image.size}: {p}')
            tensor=self.transform(image)
        return tensor, r['image_path']

def main():
    if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable; refusing large CPU inference.')
    device=torch.device('cuda:0')
    props=torch.cuda.get_device_properties(0)
    shared=json.loads(MANIFEST.read_text())
    start=time.time(); ckpt_before=sha(CHECKPOINT)
    original=transforms.Compose([transforms.Resize((448,448)),transforms.CenterCrop(392),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    full=transforms.Compose([transforms.Resize((392,392)),transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    gaussian=get_gaussian_kernel(kernel_size=5,sigma=4).to(device)
    model=build_model(device)
    model_inputs={'arcad_original':original,'arcad_full_frame':full}
    OUT.joinpath('maps').mkdir(parents=True,exist_ok=True)
    index=[]; timing={}
    for cat in CATEGORIES:
        rows=shared['categories'][cat]['shared_test_images']
        expected=shared['categories'][cat]['counts']['shared_views']
        if len(rows)!=expected: raise ValueError(f'Manifest count changed for {cat}')
        for branch,transform in model_inputs.items():
            t0=time.time(); maps=[]; paths=[]; scores=[]; shapes={}
            loader=DataLoader(Rows(rows,transform),batch_size=BATCH,shuffle=False,num_workers=WORKERS,pin_memory=True)
            with torch.inference_mode():
                for batch, batchpaths in loader:
                    batch=batch.to(device,non_blocking=True)
                    en,de,_=model(batch)
                    amap,_=cal_anomaly_maps(en,de,batch.shape[-1])
                    amap=F.interpolate(amap,size=MAP_SIZE,mode='bilinear',align_corners=False)
                    amap=gaussian(amap)
                    if not torch.isfinite(amap).all(): raise FloatingPointError(f'Nonfinite maps: {cat}/{branch}')
                    flattened=amap.flatten(1); k=max(1,int(flattened.shape[1]*.01))
                    scores.extend(flattened.topk(k,dim=1,sorted=False).values.mean(1).cpu().tolist())
                    maps.extend(amap[:,0].float().cpu().numpy())
                    paths.extend(batchpaths)
                    shapes[str(tuple(batch.shape[-2:]))]=shapes.get(str(tuple(batch.shape[-2:])),0)+len(batchpaths)
            maps=np.asarray(maps,dtype=np.float32)
            if len(paths)!=expected or len(set(paths))!=expected: raise ValueError(f'Path mismatch/duplicates {cat}/{branch}')
            f=OUT/'maps'/f'{branch}_{cat}.npz'
            np.savez_compressed(f,smoothed_maps=maps,image_paths=np.asarray(paths),view_top1_scores=np.asarray(scores,dtype=np.float32),input_size=np.asarray([392,392]),map_size=np.asarray([MAP_SIZE,MAP_SIZE]),gaussian=np.asarray([5,4]))
            timing[f'{cat}/{branch}']={'images':len(paths),'seconds':time.time()-t0,'input_shapes':shapes,'map_shape':list(maps.shape),'finite':bool(np.isfinite(maps).all()),'map_file':str(f.relative_to(OUT))}
            index.append({'category':cat,'branch':branch,'map_file':str(f.relative_to(OUT)),'views':len(paths),'first_path':paths[0],'last_path':paths[-1]})
            del maps
            torch.cuda.empty_cache()
            print(cat,branch,'views',expected,'seconds',round(timing[f'{cat}/{branch}']['seconds'],1),flush=True)
    ckpt_after=sha(CHECKPOINT)
    if ckpt_before!=ckpt_after: raise RuntimeError('ArcAD checkpoint changed during inference')
    (OUT/'arcad_map_index.json').write_text(json.dumps(index,indent=2)+'\n')
    (OUT/'inference_runtime.json').write_text(json.dumps({'device':str(device),'gpu':props.name,'gpu_memory_bytes':props.total_memory,'torch':torch.__version__,'cuda':torch.version.cuda,'elapsed_seconds':time.time()-start,'branches':timing,'checkpoint_sha256_before':ckpt_before,'checkpoint_sha256_after':ckpt_after,'batch_size':BATCH,'workers':WORKERS,'inference_random_seed':'not applicable; eval mode, deterministic transforms, no shuffling','analysis_bootstrap_seed':20261102,'inference_only':True},indent=2)+'\n')
    print('DONE',round(time.time()-start,1),'seconds',flush=True)

if __name__=='__main__': main()
