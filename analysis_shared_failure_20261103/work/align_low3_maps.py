import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw
from torchvision.transforms import InterpolationMode
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1]
T1 = ROOT / 'analysis_task01_protoinit_shared_20261103'
CATS = ['lattice_block_plug','laser_diode','power_jack']


def load_npz(path, map_key, fallback_paths=None):
    with np.load(path,allow_pickle=False) as z:
        maps=z[map_key].astype(np.float32)
        paths=z['image_paths'].tolist() if 'image_paths' in z.files else fallback_paths
    return maps,paths


def common_mask(r):
    if not r['mask_abs']: return np.zeros((256,256),dtype=np.uint8)
    tf=transforms.Compose([transforms.Resize((448,448),interpolation=InterpolationMode.NEAREST),
        transforms.CenterCrop(392),transforms.Resize((256,256),interpolation=InterpolationMode.NEAREST)])
    m=Image.open(r['mask_abs']).convert('L')
    return (np.asarray(tf(m))>0).astype(np.uint8)


def main():
    manifest=json.loads((OUT/'shared_test_manifest.json').read_text())
    aligned_report={'status':'PASS','coordinate_system':'common central 87.5% FOV of original 1024x1024 image; mapped to 256x256',
      'transforms':{'ArcAD':'Resize 448, CenterCrop 392, maps 256; map remains in common crop frame',
                    'DiNomaly2':'280 resize/full crop, map 256; grid-sampled bilinearly onto ArcAD crop using physical pixel-center coordinates',
                    'GT':'original source mask, nearest-neighbor Resize 448, CenterCrop 392, nearest-neighbor Resize 256'},
      'formula':'For common output pixel j, D2 map coordinate i = 0.875*(j+0.5)+15.5; grid_sample align_corners=False.',
      'categories':{},'errors':[]}
    all_d2=[]; all_arc=[]; all_gt=[]; all_rows=[]
    yy,xx=torch.meshgrid(torch.arange(256,dtype=torch.float32),torch.arange(256,dtype=torch.float32),indexing='ij')
    src=0.875*(xx+0.5)+15.5
    gx=2*(src+0.5)/256-1; gy=2*(0.875*(yy+0.5)+15.5+0.5)/256-1
    grid=torch.stack([gx,gy],dim=-1).unsqueeze(0)
    for cat in CATS:
        rows=sorted([r for r in manifest if r['category']==cat],key=lambda r:r['image_path'])
        expected_paths=[r['image_path'] for r in rows]
        arc,apaths=load_npz(OUT/f'maps/arcad/{cat}.npz','maps',expected_paths)
        d2,dpaths=load_npz(OUT/f'maps/d2/{cat}.npz','postprocessed_maps',expected_paths)
        if apaths!=[r['image_path'] for r in rows]:
            aligned_report['errors'].append(f'{cat}: ArcAD map path/index ordering mismatch')
        if set(dpaths)!={r['image_path'] for r in rows}:
            aligned_report['errors'].append(f'{cat}: D2 map path membership mismatch')
        d_index={p:i for i,p in enumerate(dpaths)}
        d2rows=[]
        for r in rows:
            raw=torch.from_numpy(d2[d_index[r['image_path']]]).view(1,1,256,256)
            aligned=F.grid_sample(raw,grid,mode='bilinear',padding_mode='border',align_corners=False)[0,0].numpy()
            d2rows.append(aligned.astype(np.float32))
        arc_ix={p:i for i,p in enumerate(apaths)}
        arc_rows=np.stack([arc[arc_ix[r['image_path']]] for r in rows])
        gt_rows=np.stack([common_mask(r) for r in rows])
        all_d2.extend(d2rows);all_arc.extend(arc_rows);all_gt.extend(gt_rows);all_rows.extend(rows)
        aligned_report['categories'][cat]={'views':len(rows),'map_shape':[len(rows),256,256],
            'source_image_dimensions':sorted({tuple(Image.open(r['image_abs']).size) for r in rows[:5]}),
            'arcad_native_crop_fraction':0.875,'d2_crop_fraction':1.0,
            'common_crop_original_pixels':'center crop [64,960) x [64,960) of 1024x1024'}
    # Write numeric paired arrays, including the common mask used by all spatial comparisons.
    np.savez_compressed(OUT/'maps/common_low3.npz',d2_maps=np.stack(all_d2),arcad_maps=np.stack(all_arc),
        gt_masks=np.stack(all_gt),image_paths=np.asarray([r['image_path'] for r in all_rows]),
        categories=np.asarray([r['category'] for r in all_rows]),labels=np.asarray([r['label'] for r in all_rows],dtype=np.uint8))
    aligned_report['sample_count']=len(all_rows)
    aligned_report['status']='FAIL' if aligned_report['errors'] else 'PASS_GEOMETRIC_MAPPING'
    (OUT/'alignment_validation.json').write_text(json.dumps(aligned_report,indent=2)+'\n')
    # Paired visual audit. Color scales are independently normalized for visibility.
    rng=np.random.default_rng(20261103)
    panel_w,panel_h=400,400
    canvas=Image.new('RGB',(panel_w*4,panel_h*len(CATS)+80),'white')
    draw=ImageDraw.Draw(canvas)
    draw.text((10,8),'Alignment audit: raw crop | GT | D2 aligned | ArcAD (heatmaps normalized per view for display)',fill='black')
    for row_idx,cat in enumerate(CATS):
        candidates=[i for i,r in enumerate(all_rows) if r['category']==cat and r['label']==1 and all_gt[i].any()]
        i=int(rng.choice(candidates))
        r=all_rows[i]
        image=Image.open(r['image_abs']).convert('RGB')
        crop=transforms.Compose([transforms.Resize((448,448)),transforms.CenterCrop(392)])(image).resize((panel_w,panel_h))
        mask=Image.fromarray(all_gt[i]*255).resize((panel_w,panel_h),Image.Resampling.NEAREST)
        base=crop.copy(); overlay=Image.new('RGBA',base.size,(255,0,0,0)); overlay.putalpha(mask.point(lambda v:int(v*.35)))
        base=Image.alpha_composite(base.convert('RGBA'),overlay).convert('RGB')
        d2v=all_d2[i]; arv=all_arc[i]
        def heat(a):
            z=(a-a.min())/(a.max()-a.min()+1e-12)
            return Image.fromarray(np.uint8(np.clip(z,0,1)*255)).resize((panel_w,panel_h)).convert('RGB')
        panels=[crop,base,heat(d2v),heat(arv)]
        for j,panel in enumerate(panels):
            x=j*panel_w;y=row_idx*panel_h+45
            canvas.paste(panel,(x,y));
        draw.text((8,row_idx*panel_h+28),f'{cat} — {r["object_key"]} view C{r["camera_view_id"]}',fill='black')
    canvas.save(OUT/'plots/alignment_audit_low3.png')
    if aligned_report['errors']: raise SystemExit(1)
    print(json.dumps({'status':aligned_report['status'],'samples':len(all_rows),'figure':str(OUT/'plots/alignment_audit_low3.png')},indent=2))


if __name__=='__main__': main()
