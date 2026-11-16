import hashlib,json,os,sys
from functools import partial
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parents[1]
ARCAD=ROOT/'ArcAD'; CKPT=ARCAD/'out/real_iad_variety out/arcad_realiad_variety_protoinit/model.pth'
sys.path.insert(0,str(ARCAD))
from models.uad import ViTill
from models.vision_transformer import Block as VitBlock,bMlp,LinearAttention2
from utils import cal_anomaly_maps,get_gaussian_kernel
CATS=['lattice_block_plug','laser_diode','power_jack','LED_indicator','aircraft_model_head']

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def model_make(device):
 from dinov2.models import vision_transformer as v2
 enc=v2.vit_base(patch_size=14,img_size=518,block_chunks=0,init_values=1e-8,num_register_tokens=4,interpolate_antialias=False,interpolate_offset=.1)
 bottle=nn.ModuleList([bMlp(768,3072,768,drop=.4)])
 dec=nn.ModuleList([VitBlock(dim=768,num_heads=12,mlp_ratio=4.,qkv_bias=True,norm_layer=partial(nn.LayerNorm,eps=1e-8),attn_drop=0.,attn=LinearAttention2) for _ in range(8)])
 m=ViTill(encoder=enc,bottleneck=bottle,decoder=dec,target_layers=[2,3,4,5,6,7,8,9],fuse_layer_encoder=[[0,1,2,3],[4,5,6,7]],fuse_layer_decoder=[[0,1,2,3],[4,5,6,7]],mask_neighbor_size=0)
 m.load_state_dict(torch.load(CKPT,map_location='cpu',weights_only=False),strict=True)
 return m.to(device).eval()

def main():
 os.chdir(ARCAD); cal=json.load(open(OUT/'calibration_manifest.json'))
 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
 before,mtime=sha(CKPT),CKPT.stat().st_mtime_ns; model=model_make(device)
 it=transforms.Compose([transforms.Resize((448,448)),transforms.ToTensor(),transforms.CenterCrop(392),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
 gauss=get_gaussian_kernel(5,4).to(device); dest=OUT/'calibration_maps/arcad';dest.mkdir(parents=True,exist_ok=True)
 for cat in CATS:
  rs=cal['categories'][cat]['ArcAD_train_normals']['records']; maps=[]; scores=[]
  for s in range(0,len(rs),2):
   batch=rs[s:s+2]; x=torch.stack([it(Image.open(r['image_abs']).convert('RGB')) for r in batch]).to(device)
   with torch.inference_mode():
    en,de,_=model(x); amap,_=cal_anomaly_maps(en,de,x.shape[-1]); amap=F.interpolate(amap,size=256,mode='bilinear',align_corners=False); amap=gauss(amap)
    flat=amap.flatten(1);k=int(flat.shape[1]*.01); sc=torch.sort(flat,dim=1,descending=True)[0][:,:k].mean(1).cpu().numpy()
   maps.extend(amap[:,0].cpu().numpy().astype(np.float32));scores.extend(sc.tolist())
  np.savez_compressed(dest/f'{cat}.npz',maps=np.stack(maps),image_paths=np.asarray([r['image_path'] for r in rs]),object_keys=np.asarray([r['object_key'] for r in rs]),view_scores=np.asarray(scores))
 after,mtime2=sha(CKPT),CKPT.stat().st_mtime_ns
 if before!=after or mtime!=mtime2:raise RuntimeError('ArcAD checkpoint changed')
 (OUT/'arcad_calibration_integrity.json').write_text(json.dumps({'sha256':after,'mtime_ns_before':mtime,'mtime_ns_after':mtime2,'strict_load':True,'device':str(device),'status':'in-sample training-normal references'},indent=2)+'\n')
 print('ArcAD normal-reference inference complete')

if __name__=='__main__':main()
