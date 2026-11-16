import hashlib,json,os,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parents[1];D2=ROOT/'Dinomaly2'
CKPT=D2/'out/real_iad_variety_1024 out/full run/variety_160cat/model.pth'
sys.path.insert(0,str(D2));from dinomaly2_inference import make_model;from utils import cal_anomaly_maps,get_gaussian_kernel
CATS=['lattice_block_plug','laser_diode','power_jack','LED_indicator','aircraft_model_head']

def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def main():
 os.chdir(D2);torch.set_num_threads(4);cal=json.load(open(OUT/'calibration_manifest.json'))
 device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu');before,mtime=sha(CKPT),CKPT.stat().st_mtime_ns
 model=make_model(device);model.load_state_dict(torch.load(CKPT,map_location='cpu',weights_only=True),strict=True);model.eval()
 it=transforms.Compose([transforms.Resize((280,280)),transforms.ToTensor(),transforms.CenterCrop(280),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
 gauss=get_gaussian_kernel(5,4).to(device);dest=OUT/'calibration_maps/d2';dest.mkdir(parents=True,exist_ok=True)
 for cat in CATS:
  rs=cal['categories'][cat]['D2_train_normals']['records'];maps=[];scores=[]
  for s in range(0,len(rs),2):
   batch=rs[s:s+2];x=torch.stack([it(Image.open(r['image_abs']).convert('RGB')) for r in batch]).to(device)
   with torch.inference_mode():
    en,de=model(x);amap,_=cal_anomaly_maps(en,de,x.shape[-1]);amap=F.interpolate(amap,size=256,mode='bilinear',align_corners=False);amap=gauss(amap)
    flat=amap.flatten(1);k=int(flat.shape[1]*.01);sc=torch.sort(flat,dim=1,descending=True)[0][:,:k].mean(1).cpu().numpy()
   maps.extend(amap[:,0].cpu().numpy().astype(np.float32));scores.extend(sc.tolist())
  np.savez_compressed(dest/f'{cat}.npz',maps=np.stack(maps),image_paths=np.asarray([r['image_path'] for r in rs]),object_keys=np.asarray([r['object_key'] for r in rs]),view_scores=np.asarray(scores))
 after,mtime2=sha(CKPT),CKPT.stat().st_mtime_ns
 if before!=after or mtime!=mtime2:raise RuntimeError('D2 checkpoint changed')
 (OUT/'d2_calibration_integrity.json').write_text(json.dumps({'sha256':after,'mtime_ns_before':mtime,'mtime_ns_after':mtime2,'strict_load':True,'device':str(device),'status':'in-sample training-normal references'},indent=2)+'\n')
 print('DiNomaly2 normal-reference inference complete')

if __name__=='__main__':main()
