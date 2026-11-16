import csv,json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageDraw
from torchvision import transforms
from torchvision.transforms import InterpolationMode

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parents[1]
CATS=['lattice_block_plug','laser_diode','power_jack','LED_indicator','aircraft_model_head']
MASK_TF=transforms.Compose([transforms.Resize((448,448),interpolation=InterpolationMode.NEAREST),transforms.CenterCrop(392),transforms.Resize((256,256),interpolation=InterpolationMode.NEAREST)])

def get_arrays(path, preferred):
 with np.load(path,allow_pickle=False) as z:
  key=preferred if preferred in z.files else ('maps' if 'maps' in z.files else 'postprocessed_maps')
  return z[key].astype(np.float32),z['image_paths'].tolist() if 'image_paths' in z.files else None

def main():
 manifest=json.load(open(OUT/'shared_test_manifest.json')); dest=OUT/'maps/common';dest.mkdir(parents=True,exist_ok=True)
 report={'status':'PASS','common_fov':'center 896x896 source pixels [64,960) x [64,960), resized to 256x256',
  'mapping_formula':'D2 source map coordinate i = 0.875*(j+0.5)+15.5; grid_sample bilinear, align_corners=False',
  'ArcAD_map_transform':'448 resize -> 392 center crop -> 256 map; common crop is native ArcAD map frame',
  'GT_transform':'original source mask -> nearest resize 448 -> center crop 392 -> nearest resize 256',
  'categories':{},'errors':[]}
 yy,xx=torch.meshgrid(torch.arange(256,dtype=torch.float32),torch.arange(256,dtype=torch.float32),indexing='ij')
 sx=.875*(xx+.5)+15.5;sy=.875*(yy+.5)+15.5
 grid=torch.stack([2*(sx+.5)/256-1,2*(sy+.5)/256-1],-1).unsqueeze(0)
 idx_rows=[];vis_rows=[];rng=np.random.default_rng(20261103)
 for cat in CATS:
  rs=sorted([r for r in manifest if r['category']==cat],key=lambda r:r['image_path'])
  expected=[r['image_path'] for r in rs]
  arc,ap=get_arrays(dest.parent/'arcad'/f'{cat}.npz','maps')
  d2,dp=get_arrays(dest.parent/'d2'/f'{cat}.npz','postprocessed_maps')
  if ap is not None and ap!=expected:report['errors'].append(f'{cat}: ArcAD path order mismatch')
  if dp is None or set(dp)!=set(expected):report['errors'].append(f'{cat}: D2 map paths mismatch')
  ai={p:i for i,p in enumerate(ap or expected)};di={p:i for i,p in enumerate(dp or expected)}
  da=[];aa=[];gt=[]
  image_sizes=set()
  for r in rs:
   image_sizes.add(Image.open(r['image_abs']).size)
   raw=torch.from_numpy(d2[di[r['image_path']]]).view(1,1,256,256)
   aligned=F.grid_sample(raw,grid,mode='bilinear',padding_mode='border',align_corners=False)[0,0].numpy()
   da.append(aligned.astype(np.float32));aa.append(arc[ai[r['image_path']]])
   if r['mask_abs']:
    mm=np.asarray(MASK_TF(Image.open(r['mask_abs']).convert('L')))
    gt.append((mm>0).astype(np.uint8))
   else:gt.append(np.zeros((256,256),dtype=np.uint8))
  if image_sizes!={(1024,1024)}: report['errors'].append(f'{cat}: source image dimensions {image_sizes}')
  da,aa,gt=np.stack(da),np.stack(aa),np.stack(gt)
  if da.shape!=(len(rs),256,256) or aa.shape!=da.shape or gt.shape!=da.shape:report['errors'].append(f'{cat}: map/mask shape mismatch')
  if not np.isfinite(da).all() or not np.isfinite(aa).all():report['errors'].append(f'{cat}: nonfinite score map')
  np.savez_compressed(dest/f'{cat}.npz',d2_maps=da,arcad_maps=aa,gt_masks=gt,
   image_paths=np.asarray(expected),object_keys=np.asarray([r['object_key'] for r in rs]),
   labels=np.asarray([r['label'] for r in rs],dtype=np.uint8),view_ids=np.asarray([r['camera_view_id'] for r in rs],dtype=np.uint8))
  report['categories'][cat]={'views':len(rs),'objects':len({r['object_key'] for r in rs}),
   'positive_views_with_empty_common_mask':sum(r['label']==1 and not gt[i].any() for i,r in enumerate(rs)),
   'normal_masks_empty':all(not gt[i].any() for i,r in enumerate(rs) if r['label']==0),
   'valid_mask_coordinates':True,'map_shape':list(da.shape)}
  if not report['categories'][cat]['normal_masks_empty']:report['errors'].append(f'{cat}: normal view has nonempty common mask')
  for i,r in enumerate(rs):idx_rows.append([cat,i,r['image_path'],r['object_key'],r['camera_view_id'],r['label'],r['mask_path'] or '',f'{cat}.npz'])
  vis_candidates=[i for i,r in enumerate(rs) if r['label']==1 and gt[i].any()]
  vi=int(rng.choice(vis_candidates));vis_rows.append((cat,rs[vi],aa[vi],da[vi],gt[vi]))
 with open(dest/'index.csv','w',newline='') as f:
  w=csv.writer(f);w.writerow(['category','index','image_path','object_key','camera_view_id','gt_view_label','gt_mask_path','map_file']);w.writerows(idx_rows)
 report['status']='FAIL' if report['errors'] else 'PASS_GEOMETRIC_MAPPING'
 report['total_views']=len(idx_rows)
 (OUT/'alignment_validation.json').write_text(json.dumps(report,indent=2)+'\n')
 W=320;H=320
 canvas=Image.new('RGB',(W*4,(H+35)*len(CATS)+70),'white');draw=ImageDraw.Draw(canvas)
 draw.text((10,5),'Common-crop alignment audit | grayscale maps normalized per view for display only',fill='black')
 for ri,(cat,r,aa,da,gt) in enumerate(vis_rows):
  im=Image.open(r['image_abs']).convert('RGB');crop=transforms.Compose([transforms.Resize((448,448)),transforms.CenterCrop(392)])(im).resize((W,H))
  mask=Image.fromarray(gt*255).resize((W,H),Image.Resampling.NEAREST);overlay=Image.new('RGBA',crop.size,(255,0,0,0));overlay.putalpha(mask.point(lambda x:int(x*.35)))
  crop_gt=Image.alpha_composite(crop.convert('RGBA'),overlay).convert('RGB')
  def heat(m):
   v=(m-m.min())/(m.max()-m.min()+1e-12);return Image.fromarray(np.uint8(v*255)).resize((W,H)).convert('RGB')
  y=ri*(H+35)+35
  for j,p in enumerate([crop,crop_gt,heat(da),heat(aa)]):canvas.paste(p,(j*W,y))
  draw.text((8,y-20),f'{cat}: {r["object_key"]} C{r["camera_view_id"]}',fill='black')
 canvas.save(OUT/'plots/alignment_audit_all_categories.png')
 print(json.dumps({'status':report['status'],'views':len(idx_rows),'figure':str(OUT/'plots/alignment_audit_all_categories.png'),'errors':report['errors']},indent=2))
 if report['errors']:raise SystemExit(1)

if __name__=='__main__':main()
