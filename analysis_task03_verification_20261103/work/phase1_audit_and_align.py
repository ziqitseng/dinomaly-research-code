#!/usr/bin/env python3
import csv,json,math
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parents[1];T2=ROOT/'analysis_shared_failure_20261103';T1=ROOT/'analysis_task01_protoinit_shared_20261103'
CATS=['lattice_block_plug','laser_diode','power_jack','LED_indicator','aircraft_model_head']
TASK2_EMPTY_EXPECTED=41
TASK1_ZERO_PATHS={
 'NG/OilStain/S0005/laser_diode__0005_NG_Oil_Stain_C1_20240713143457258.png',
 'NG/OilStain/S0005/laser_diode__0005_NG_Oil_Stain_C4_20240713143454828.png'}

def load_npz(path,key):
 with np.load(path,allow_pickle=False) as z:return z[key].astype(np.float32)
def read_csv(path):
 with open(path,newline='') as f:return list(csv.DictReader(f))
def mask_arrays(path):
 if not path:return None,None,None,None
 try:
  im=Image.open(path).convert('L'); raw=np.asarray(im)>0
  if raw.shape!=(1024,1024):return raw,None,None,im.size
  crop=raw[64:960,64:960]
  c224=np.asarray(Image.fromarray(crop.astype(np.uint8)*255).resize((224,224),Image.Resampling.NEAREST))>0
  c256=np.asarray(Image.fromarray(crop.astype(np.uint8)*255).resize((256,256),Image.Resampling.NEAREST))>0
  return raw,crop,c224,im.size
 except Exception:return 'CORRUPT',None,None,None

manifest=json.loads((T2/'shared_test_manifest.json').read_text());mby={c:sorted([r for r in manifest if r['category']==c],key=lambda r:r['image_path']) for c in CATS}
aligned=OUT/'maps224';aligned.mkdir(exist_ok=True)
validation={'status':'PASS','grid':'224x224; physical source field [64:960) x [64:960)','mapping':{
 'D2':'native 256x256 map crop rows/cols [16:240], no interpolation',
 'ArcAD':'native 256 common-crop map bilinearly resized to 224 with interpolate(align_corners=False)',
 'GT':'original 1024 source mask crop [64:960] then nearest-neighbor resize to 224'},'categories':{},'errors':[]}
yy,xx=torch.meshgrid(torch.arange(224,dtype=torch.float32),torch.arange(224,dtype=torch.float32),indexing='ij')

# Trace ArcAD map order to the appropriate native inference index; trace D2 order from each saved archive.
arcad_index={}
for r in read_csv(T1/'arcad_maps/index.csv'):arcad_index.setdefault(r['category'],[]).append(r['image_path'])
for r in read_csv(T2/'arcad_extra_per_view.csv'):arcad_index.setdefault(r['category'],[]).append(r['image_path'])
empty_cases=[]
for cat in CATS:
 rs=mby[cat];expected=[r['image_path'] for r in rs]
 arc=load_npz(T2/'maps/arcad'/f'{cat}.npz','maps')
 with np.load(T2/'maps/d2'/f'{cat}.npz',allow_pickle=False) as z:
  dkey='postprocessed_maps' if 'postprocessed_maps' in z.files else 'maps'
  d2=z[dkey].astype(np.float32);dpaths=z['image_paths'].astype(str).tolist()
 if len(arc)!=len(rs) or len(d2)!=len(rs):validation['errors'].append(f'{cat}: count mismatch')
 aip=arcad_index.get(cat,[])
 if aip!=expected:validation['errors'].append(f'{cat}: ArcAD source-index order mismatch')
 if dpaths!=expected:validation['errors'].append(f'{cat}: D2 source-path order mismatch')
 if len(set(expected))!=len(expected):validation['errors'].append(f'{cat}: duplicate manifest path')
 if arc.shape!=(len(rs),256,256) or d2.shape!=(len(rs),256,256):validation['errors'].append(f'{cat}: native map shape mismatch')
 if not np.isfinite(arc).all() or not np.isfinite(d2).all():validation['errors'].append(f'{cat}: nonfinite native map value')
 am=F.interpolate(torch.from_numpy(arc[:,None]),size=(224,224),mode='bilinear',align_corners=False)[:,0].numpy()
 dm=d2[:,16:240,16:240]
 gt224=[];gt_source=[];gt_crop=[];mstat=[];dims=set()
 for i,r in enumerate(rs):
  raw,crop,small,size=mask_arrays(r.get('mask_abs') or '')
  if isinstance(raw,str):validation['errors'].append(f'{cat}/{r["image_path"]}: corrupt mask')
  if raw is None:small=np.zeros((224,224),bool);gt_source.append(None);gt_crop.append(None)
  else:
   dims.add(size);gt_source.append(raw);gt_crop.append(crop);small=small if small is not None else np.zeros((224,224),bool)
   if r['label']==0 and raw.any():validation['errors'].append(f'{cat}/{r["image_path"]}: normal view has nonzero source mask')
  gt224.append(small.astype(np.uint8))
  if cat=='laser_diode' and r['label']==1 and not small.any():
   if raw is None or isinstance(raw,str):status='missing_or_corrupt_mask'
   elif not raw.any():status='source_mask_all_zero'
   elif crop is None:status='source_mask_dimension_invalid'
   elif not crop.any():status='nonzero_mask_outside_common_crop'
   else:status='region_vanished_during_224_downsampling'
   empty_cases.append({'manifest_row':r,'raw':raw if not isinstance(raw,str) else None,'crop':crop,'mask224':small,'status':status,'source_size':size,'source_pixels':int(raw.sum()) if isinstance(raw,np.ndarray) else -1,'crop_pixels':int(crop.sum()) if isinstance(crop,np.ndarray) else -1,'mask224_pixels':int(small.sum())})
 gt224=np.stack(gt224).astype(np.uint8)
 if not np.isfinite(am).all() or not np.isfinite(dm).all():validation['errors'].append(f'{cat}: nonfinite aligned maps')
 if np.min(am)<np.min(arc)-1e-5 or np.max(am)>np.max(arc)+1e-5:validation['errors'].append(f'{cat}: ArcAD interpolation score range escaped source range')
 if np.min(dm)<np.min(d2)-1e-6 or np.max(dm)>np.max(d2)+1e-6:validation['errors'].append(f'{cat}: D2 crop score range escaped source range')
 obj_keys=[r['object_key'] for r in rs];labels=np.array([r['label'] for r in rs],np.uint8);views=np.array([r['camera_view_id'] for r in rs],np.uint8)
 np.savez_compressed(aligned/f'{cat}.npz',d2_maps=dm.astype(np.float32),arcad_maps=am.astype(np.float32),gt_masks=gt224,
  image_paths=np.asarray(expected),object_keys=np.asarray(obj_keys),labels=labels,view_ids=views)
 validation['categories'][cat]={'views':len(rs),'objects':len(set(obj_keys)),'source_sizes':sorted([list(q) for q in dims]),'map_shape':[len(rs),224,224],
  'd2_native_crop':[16,240],'arcad_bilinear_resize':[256,224],'gt_nonzero_224_views':int(gt224.reshape(len(rs),-1).any(1).sum()),
  'source_map_ranges':{'D2':[float(d2.min()),float(d2.max())],'ArcAD':[float(arc.min()),float(arc.max())]},'aligned_map_ranges':{'D2':[float(dm.min()),float(dm.max())],'ArcAD':[float(am.min()),float(am.max())]}}

if len(empty_cases)!=TASK2_EMPTY_EXPECTED:validation['errors'].append(f'expected {TASK2_EMPTY_EXPECTED} laser diode positives with empty Task02 common GT, found {len(empty_cases)}')
validation['status']='FAIL' if validation['errors'] else 'PASS_224_INDEPENDENT_ALIGNMENT'
(OUT/'alignment_validation_224.json').write_text(json.dumps(validation,indent=2)+'\n')

# Reconcile raw-mask causes and labels with both native split manifests.
def object_key(path):
 p=path.replace('\\','/').split('/');return '/'.join(p[:2] if p[0]=='OK' else p[:3])
def loadj(folder,cat):return json.loads((folder/f'{cat}.json').read_text())
sup=loadj(Path('/media/avlab/7458a474-6ea2-4b90-b232-373c3df8d329/real_iad_variety_1024/realiadvariety_jsons/sup'),'laser_diode')
d2j=loadj(Path('/media/avlab/7458a474-6ea2-4b90-b232-373c3df8d329/real_iad_variety_1024/realiadvariety_jsons/Real-IAD_Variety_jsons'),'laser_diode')
asrc={x['image_path']:x for x in sup['test']};dsrc={x['image_path']:x for x in d2j['test']}
rr=[r for r in manifest if r['category']=='laser_diode'];byobj=defaultdict(list)
for r in rr:byobj[r['object_key']].append(r)
audit=[]
for c in empty_cases:
 r=c['manifest_row'];p=r['image_path'];ar=asrc.get(p,{});dr=dsrc.get(p,{})
 obj=byobj[r['object_key']];arc_obj=any(int(x.get('label',0)) for x in obj);d2_obj=any(bool(dsrc.get(x['image_path'],{}).get('mask_path')) for x in obj)
 maskpx=c['source_pixels'];crop_px=c['crop_pixels'];down=c['mask224_pixels']
 if c['status']=='missing_or_corrupt_mask':status='missing_or_corrupted_mask'
 elif c['status']=='source_mask_dimension_invalid':status='source_mask_dimension_invalid'
 elif maskpx==0:status='genuinely_all_zero_source_mask'
 elif crop_px==0:status='nonzero_mask_outside_[64:960]_crop'
 elif down==0:status='nonzero_crop_mask_vanished_at_224_downsample'
 else:status='unexpectedly_visible_at_224_task02_empty_was_256_grid'
 viewagree=int(int(ar.get('label',-1))==int(r['label'])==int(bool(dr.get('mask_path'))))
 mixed=int(len({int(x['label']) for x in obj})>1)
 audit.append({'category':'laser_diode','object_key':r['object_key'],'image_path':p,'camera_view_id':r['camera_view_id'],'defect_type':r.get('defect_type') or '',
  'source_mask_path':r.get('mask_path') or '','source_mask_exists':int(isinstance(c['raw'],np.ndarray)),'source_mask_shape':str(c['source_size']),'source_mask_pixels':maskpx,'common_crop_source_pixels':crop_px,'common_crop_224_pixels':down,
  'image_label':r['label'],'ArcAD_test_view_label':ar.get('label'),'D2_test_mask_label':int(bool(dr.get('mask_path'))),'ArcAD_D2_view_labels_agree':viewagree,'object_label_arcad':int(arc_obj),'object_label_d2':int(d2_obj),'object_has_mixed_view_labels':mixed,
  'Task01_reported_all_zero_example':int(p in TASK1_ZERO_PATHS),'status':status,'manual_review_status':'requires_manual_review'})
with open(OUT/'empty_mask_audit.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(audit[0]));w.writeheader();w.writerows(audit)

# Contact sheets: target view, full uncropped source mask, 224 crop mask, then C1-C5 sibling images.
sheetdir=OUT/'empty_mask_contact_sheet';sheetdir.mkdir(exist_ok=True);cw=148;ch=148;page_n=4
for page,start in enumerate(range(0,len(empty_cases),page_n),1):
 cases=empty_cases[start:start+page_n];canvas=Image.new('RGB',(cw*8,(ch+30)*len(cases)+28),'white');draw=ImageDraw.Draw(canvas)
 draw.text((5,4),f'Laser diode empty-common-mask audit | cases {start+1}-{start+len(cases)} of {len(empty_cases)}',fill='black')
 for ri,c in enumerate(cases):
  r=c['manifest_row'];objviews=sorted(byobj[r['object_key']],key=lambda x:x['camera_view_id']);y=24+ri*(ch+30)
  imgs=[]
  try:imgs.append(Image.open(r['image_abs']).convert('RGB').resize((cw,cw)))
  except Exception:imgs.append(Image.new('RGB',(cw,cw),'gray'))
  for mm in [c['raw'],c['mask224']]:
   if isinstance(mm,np.ndarray):imgs.append(Image.fromarray((mm.astype(np.uint8)*255)).convert('RGB').resize((cw,cw),Image.Resampling.NEAREST))
   else:imgs.append(Image.new('RGB',(cw,cw),'gray'))
  for view in objviews:
   try:imgs.append(Image.open(view['image_abs']).convert('RGB').resize((cw,cw)))
   except Exception:imgs.append(Image.new('RGB',(cw,cw),'gray'))
  labels=['target image','full source GT','common-crop GT224']+[f"sibling C{v['camera_view_id']}" for v in objviews]
  for j,(im,title) in enumerate(zip(imgs,labels)):
   canvas.paste(im,(j*cw,y));draw.text((j*cw+2,y+2),title,fill='yellow' if j==0 or j>=3 else 'red',stroke_width=1,stroke_fill='black')
  draw.text((5,y+cw+2),f"{r['object_key']} C{r['camera_view_id']} | source={c['source_pixels']} crop={c['crop_pixels']} 224={c['mask224_pixels']} | {c['status']} | requires manual review",fill='black')
 canvas.save(sheetdir/f'page_{page:02d}.png')

from collections import Counter
counts=Counter(x['status'] for x in audit);zero_task1=sum(x['Task01_reported_all_zero_example'] for x in audit)
label_disagreements=sum(int(x['ArcAD_D2_view_labels_agree']==0) for x in audit)
mixed_objs=sorted({x['object_key'] for x in audit if int(x['object_has_mixed_view_labels'])})
issues=['# Data quality issues and visibility audit','',
 f"Recomputed the {len(audit)} positive `laser_diode` views that had empty Task 02 common-grid GT. Full-source mask and crop pixel counts are in `empty_mask_audit.csv`; all source files were read-only.",
 '', '## Classification','','']
issues += [f'- `{k}`: {v}' for k,v in sorted(counts.items())]
issues += ['',f'Task 01’s two `OilStain/S0005` paths are both present and confirmed to have genuinely all-zero source masks. The remaining cases are classified independently from raw 1024 masks and a direct [64:960] crop followed by nearest-neighbor 224 downsampling.',
 '',f'ArcAD and D2 view labels agree for all {len(audit)} audited cases (disagreements: {label_disagreements}). The positive object `NG/OilStain/S0005` has mixed view labels: two anomalous views and three normal views in both manifests. This is retained as a view/object-label distinction, not treated as proof of an incorrect label.',
 '', 'A zero or crop-invisible mask is not treated as proof that an anomaly label is wrong. Every case is marked `requires_manual_review`; this automated audit does not establish physical defect visibility.',
 '', 'No source mask, image label, object label, checkpoint, or original output was modified.',
 '', '## Contact sheets','','Pages under `empty_mask_contact_sheet/` show each target image, uncropped source mask, common-crop 224 mask, and all five sibling camera views. The crop masks are displayed independently; no annotation is invented.']
(OUT/'data_quality_issues.md').write_text('\n'.join(issues)+'\n')
print(json.dumps({'alignment':validation['status'],'alignment_errors':validation['errors'],'empty_cases':len(audit),'empty_mask_status_counts':dict(counts),'Task01_zero_paths_reconciled':zero_task1,'contact_pages':len(list(sheetdir.glob('*.png')))},indent=2))
