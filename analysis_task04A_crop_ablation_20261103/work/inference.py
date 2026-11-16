#!/usr/bin/env python3
"""GPU-only frozen-checkpoint inference/cache preparation for Task 04A."""
import argparse,csv,hashlib,json,os,sys,time
from pathlib import Path
from functools import partial
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parents[1]
ARCAD=ROOT/'ArcAD';D2=ROOT/'Dinomaly2'
ARCAD_CKPT=ARCAD/'out/real_iad_variety out/arcad_realiad_variety_protoinit/model.pth'
D2_CKPT=D2/'out/real_iad_variety_1024 out/full run/variety_160cat/model.pth'
CATS=['lattice_block_plug','laser_diode','power_jack','LED_indicator','thyristor']
REUSE_CATS=set(CATS)-{'thyristor'}
BATCH=2;TOP=.01;SEED=20261103

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()
def top1(a):
    flat=np.asarray(a,dtype=np.float32).reshape(-1);k=max(1,int(flat.size*TOP))
    return float(np.partition(flat,-k)[-k:].mean())
def manifest():return json.loads((OUT/'shared_test_manifest.json').read_text())
def rows_for(cat):return sorted([r for r in manifest() if r['category']==cat],key=lambda r:r['image_path'])
def load_npz(path,key):
    with np.load(path,allow_pickle=False) as z:
        selected=key if key in z.files else ('maps' if 'maps' in z.files else 'postprocessed_maps')
        if selected not in z.files:raise KeyError(f'{path} missing map array; keys={z.files}')
        return z[selected].astype(np.float32),z['image_paths'].astype(str).tolist()
def isolate_repo(repo):
    prefixes=('models.','dinov2.','dinov2','beit.','beit')
    for name in list(sys.modules):
        if name in ('dataset','utils','models','beit','dinov2') or name.startswith(prefixes):
            del sys.modules[name]
    sys.path[:]=[p for p in sys.path if p not in (str(ARCAD),str(D2))]
    sys.path.insert(0,str(repo))
    os.chdir(repo)
def check_aligned(cat,arr,paths,expected,tag):
    if paths!=expected:raise RuntimeError(f'{cat} {tag}: exact path order/membership mismatch')
    if arr.shape!=(len(expected),256,256):raise RuntimeError(f'{cat} {tag}: unexpected map shape {arr.shape}')
    if arr.dtype!=np.float32 or not np.isfinite(arr).all():raise RuntimeError(f'{cat} {tag}: invalid dtype/nonfinite values')
    return arr

def build_arcad(device):
    isolate_repo(ARCAD)
    from models.uad import ViTill
    from models.vision_transformer import Block as VitBlock,bMlp,LinearAttention2
    from utils import cal_anomaly_maps,get_gaussian_kernel
    from dinov2.models import vision_transformer as v2_vit
    encoder=v2_vit.vit_base(patch_size=14,img_size=518,block_chunks=0,init_values=1e-8,
        num_register_tokens=4,interpolate_antialias=False,interpolate_offset=0.1)
    bottle=nn.ModuleList([bMlp(768,768*4,768,drop=0.4)])
    decoder=nn.ModuleList([VitBlock(dim=768,num_heads=12,mlp_ratio=4.,qkv_bias=True,
        norm_layer=partial(nn.LayerNorm,eps=1e-8),attn_drop=0.,attn=LinearAttention2) for _ in range(8)])
    model=ViTill(encoder=encoder,bottleneck=bottle,decoder=decoder,target_layers=[2,3,4,5,6,7,8,9],
        fuse_layer_encoder=[[0,1,2,3],[4,5,6,7]],fuse_layer_decoder=[[0,1,2,3],[4,5,6,7]],mask_neighbor_size=0)
    state=torch.load(ARCAD_CKPT,map_location='cpu',weights_only=False);model.load_state_dict(state,strict=True)
    return model.to(device).eval(),cal_anomaly_maps,get_gaussian_kernel

def build_d2(device):
    isolate_repo(D2)
    from dinomaly2_inference import make_model
    from utils import cal_anomaly_maps,get_gaussian_kernel
    model=make_model(device);state=torch.load(D2_CKPT,map_location='cpu',weights_only=True)
    model.load_state_dict(state,strict=True);return model.eval(),cal_anomaly_maps,get_gaussian_kernel

def transforms_for():
    norm=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    return {
      'arcad_original':transforms.Compose([transforms.Resize((448,448)),transforms.ToTensor(),transforms.CenterCrop(392),norm]),
      'arcad_full':transforms.Compose([transforms.Resize((392,392)),transforms.ToTensor(),norm]),
      'd2_full':transforms.Compose([transforms.Resize((280,280)),transforms.ToTensor(),transforms.CenterCrop(280),norm])}

def forward_maps(model,cal,gauss,x,dim):
    with torch.inference_mode():
        result=model(x)
        en,de=(result[0],result[1]) if isinstance(result,(tuple,list)) else (result,None)
        if de is None:raise RuntimeError('model returned no decoder/reconstruction feature')
        amap,_=cal(en,de,x.shape[-1])
        amap=F.interpolate(amap,size=256,mode='bilinear',align_corners=False)
        amap=gauss(amap)
    out=amap[:,0].float().cpu().numpy().astype(np.float32)
    if out.shape[1:]!=(256,256) or not np.isfinite(out).all():raise RuntimeError(f'{dim} map output invalid: {out.shape}')
    return out

def smoke():
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; Task 04A inference is forbidden on CPU')
    dev=torch.device('cuda:0');free,total=torch.cuda.mem_get_info(dev)
    rows=rows_for('laser_diode')
    im=Image.open(rows[0]['image_abs']).convert('RGB');tf=transforms_for()
    torch.cuda.reset_peak_memory_stats(dev)
    arc,cal,gk=build_arcad(dev);gauss=gk(kernel_size=5,sigma=4).to(dev)
    xo=tf['arcad_original'](im).unsqueeze(0).to(dev);xf=tf['arcad_full'](im).unsqueeze(0).to(dev)
    if xo.shape!=xf.shape or tuple(xo.shape[-2:])!=(392,392):raise RuntimeError('ArcAD 392 input shape mismatch')
    mo=forward_maps(arc,cal,gauss,xo,'ArcAD-original-smoke');mf=forward_maps(arc,cal,gauss,xf,'ArcAD-full-smoke')
    torch.cuda.synchronize();arc_peak=torch.cuda.max_memory_allocated(dev)
    del arc,mo,mf,xo,xf,gauss;torch.cuda.empty_cache()
    d2,cal2,gk2=build_d2(dev);g2=gk2(kernel_size=5,sigma=4).to(dev)
    xd=tf['d2_full'](im).unsqueeze(0).to(dev)
    if tuple(xd.shape[-2:])!=(280,280):raise RuntimeError('D2 280 input shape mismatch')
    md=forward_maps(d2,cal2,g2,xd,'D2-full-smoke');torch.cuda.synchronize();d2_peak=torch.cuda.max_memory_allocated(dev)
    del d2,md,xd,g2;torch.cuda.empty_cache()
    result={'status':'PASS_GPU_SMOKE','device':torch.cuda.get_device_name(dev),'torch_version':torch.__version__,
       'cuda_version':torch.version.cuda,'free_vram_bytes_at_start':int(free),'total_vram_bytes':int(total),
       'ArcAD_original_and_fullframe_input_shape':[1,3,392,392],'D2_input_shape':[1,3,280,280],
       'ArcAD_peak_allocated_bytes':int(arc_peak),'D2_peak_allocated_bytes':int(d2_peak),
       'maps_finite_shape_256':True,'checkpoint_sha256':{'ArcAD':sha(ARCAD_CKPT),'D2':sha(D2_CKPT)},
       'synthetic_landmark_check':'same 392x392 network grid for ArcAD branches; distinct preprocessing verified by source pixel mapping in geometry_validation.json'}
    (OUT/'gpu_smoke.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))

def infer_branch(model_name,branch,device):
    model,cal,gk=(build_arcad(device) if model_name=='ArcAD' else build_d2(device))
    gauss=gk(kernel_size=5,sigma=4).to(device);tf=transforms_for();allmeta={}
    h0=sha(ARCAD_CKPT if model_name=='ArcAD' else D2_CKPT);m0=(ARCAD_CKPT if model_name=='ArcAD' else D2_CKPT).stat().st_mtime_ns
    for cat in CATS:
        if model_name=='ArcAD' and branch=='arcad_full':todo=True
        elif cat=='thyristor':todo=True
        else:todo=False
        if not todo:continue
        rs=rows_for(cat);maps=[];started=time.time();tr=tf[branch]
        for start in range(0,len(rs),BATCH):
            batch=rs[start:start+BATCH]
            xs=[tr(Image.open(r['image_abs']).convert('RGB')) for r in batch]
            x=torch.stack(xs).to(device,non_blocking=True)
            if tuple(x.shape[-2:])!=(392,392) if model_name=='ArcAD' else tuple(x.shape[-2:])!=(280,280):
                raise RuntimeError(f'{model_name}/{branch}: unexpected tensor shape {tuple(x.shape)}')
            maps.extend(forward_maps(model,cal,gauss,x,f'{model_name}-{branch}-{cat}'))
            if start%200==0:
                torch.cuda.synchronize();print(f'{model_name}/{branch}/{cat}: {min(start+BATCH,len(rs))}/{len(rs)}; elapsed={time.time()-started:.1f}s',flush=True)
        arr=np.stack(maps).astype(np.float32)
        if not np.isfinite(arr).all():raise RuntimeError(f'{cat}/{branch}: nonfinite map')
        target_branch='d2_original' if model_name=='D2' else branch
        target=OUT/'maps'/target_branch;target.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(target/f'{cat}.npz',maps=arr,image_paths=np.asarray([r['image_path'] for r in rs]),
            object_keys=np.asarray([r['object_key'] for r in rs]),labels=np.asarray([r['label'] for r in rs],np.uint8),
            camera_view_ids=np.asarray([r['camera_view_id'] for r in rs],np.uint8))
        allmeta[cat]={'views':len(rs),'shape':list(arr.shape),'elapsed_seconds':time.time()-started,
           'score_top1_first':top1(arr[0]),'min':float(arr.min()),'max':float(arr.max())}
        print(json.dumps({'branch':branch,'category':cat,**allmeta[cat]}),flush=True)
    ckpt=ARCAD_CKPT if model_name=='ArcAD' else D2_CKPT
    if sha(ckpt)!=h0 or ckpt.stat().st_mtime_ns!=m0:raise RuntimeError(f'{model_name} checkpoint changed during inference')
    del model,gauss;torch.cuda.empty_cache()
    return allmeta

def cache_existing():
    """Save exact path-validated current Task 02 ArcAD/D2 maps for four shared categories."""
    out={}
    for cat in sorted(REUSE_CATS):
        rs=rows_for(cat);exp=[r['image_path'] for r in rs]
        for branch,src,key in [('arcad_original',ROOT/'analysis_shared_failure_20261103/maps/common'/f'{cat}.npz','arcad_maps'),
                               ('d2_original',ROOT/'analysis_shared_failure_20261103/maps/d2'/f'{cat}.npz','postprocessed_maps')]:
            arr,paths=load_npz(src,key);check_aligned(cat,arr,paths,exp,branch)
            dest=OUT/'maps'/branch;dest.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(dest/f'{cat}.npz',maps=arr,image_paths=np.asarray(paths),
                object_keys=np.asarray([r['object_key'] for r in rs]),labels=np.asarray([r['label'] for r in rs],np.uint8),
                camera_view_ids=np.asarray([r['camera_view_id'] for r in rs],np.uint8),
                provenance=np.asarray([str(src)]))
            out[f'{cat}/{branch}']={'source':str(src),'source_sha256':sha(src),'views':len(paths),'shape':list(arr.shape),'path_order_match':True}
    (OUT/'cache_reuse_provenance.json').write_text(json.dumps(out,indent=2)+'\n')

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--smoke',action='store_true');ap.add_argument('--run',action='store_true')
    args=ap.parse_args()
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; refusing CPU inference per Task 04A stop condition')
    if json.loads((OUT/'manifest_validation.json').read_text())['status']!='PASS':raise RuntimeError('manifest did not validate')
    torch.manual_seed(SEED);np.random.seed(SEED);torch.set_num_threads(4)
    if args.smoke:smoke();return
    if not args.run:raise SystemExit('specify --smoke or --run')
    dev=torch.device('cuda:0');torch.cuda.init()
    if not (OUT/'gpu_smoke.json').exists() or json.loads((OUT/'gpu_smoke.json').read_text()).get('status')!='PASS_GPU_SMOKE':
        raise RuntimeError('successful CUDA smoke check required before full inference')
    cache_existing()
    timing={'device':torch.cuda.get_device_name(dev),'torch_version':torch.__version__,'cuda_version':torch.version.cuda,
       'batch_size':BATCH,'seed':SEED,'top_fraction':TOP,'start_unix':time.time(),'branches':{}}
    # Full-frame ArcAD is the only new ArcAD inference for already cached categories.
    timing['branches']['arcad_fullframe']=infer_branch('ArcAD','arcad_full',dev)
    # The optional thyristor category has no Task 02 ArcAD-original cache; infer original branch too.
    timing['branches']['arcad_original_thyristor']=infer_branch('ArcAD','arcad_original',dev)
    # D2 full-frame cache exists for the four Task 02 categories. Infer it for thyristor.
    timing['branches']['d2_original_thyristor']=infer_branch('D2','d2_full',dev)
    timing['end_unix']=time.time();timing['elapsed_seconds']=timing['end_unix']-timing['start_unix']
    timing['status']='PASS_INFERENCE_COMPLETE'
    (OUT/'inference_run.json').write_text(json.dumps(timing,indent=2)+'\n')
    print(json.dumps({'status':timing['status'],'elapsed_seconds':timing['elapsed_seconds'],'categories':CATS},indent=2))

if __name__=='__main__':main()
