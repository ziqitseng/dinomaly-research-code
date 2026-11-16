#!/usr/bin/env python3
import csv,json
from pathlib import Path
from collections import defaultdict
import numpy as np,pandas as pd
from PIL import Image
from scipy import ndimage
from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve,precision_recall_curve,auc
from skimage import measure
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[2]; OUT=Path(__file__).resolve().parents[1]
T2=ROOT/'analysis_shared_failure_20261103'; T3=ROOT/'analysis_task03_verification_20261103'
CATS=['lattice_block_plug','laser_diode','power_jack','LED_indicator','thyristor']
BR=['ArcAD_original','ArcAD_fullframe','D2_original']; BOOT=2000; SEED=20261103
torch.set_num_threads(2)
def writecsv(p,rows):
 rows=list(rows)
 if not rows:p.write_text('');return
 with open(p,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),extrasaction='ignore');w.writeheader();w.writerows(rows)
def top(a,f=.01):
 x=np.asarray(a,np.float32).ravel();k=max(1,int(x.size*f));return float(np.partition(x,-k)[-k:].mean())
def cls(y,s):
 y=np.asarray(y,np.uint8);s=np.asarray(s,float)
 if len(np.unique(y))<2:return {'AUROC':np.nan,'AP':np.nan}
 return {'AUROC':float(roc_auc_score(y,s)),'AP':float(average_precision_score(y,s))}
def pix(m,g):
 x=np.asarray(m,np.float32).ravel();y=np.asarray(g,bool).ravel();lo=float(x.min());hi=float(x.max())
 e=np.linspace(lo,hi+max(1e-12,abs(hi)*1e-12),2049)
 p=np.histogram(x[y],bins=e)[0][::-1].cumsum();n=np.histogram(x[~y],bins=e)[0][::-1].cumsum()
 rec=np.r_[0,p/max(1,y.sum())];fpr=np.r_[0,n/max(1,(~y).sum())];pr=p/np.maximum(1,p+n)
 return {'P_AUROC':float(np.trapz(rec,fpr)),'P_AP':float(np.sum(np.diff(rec)*pr)),
         'positive_pixels':int(y.sum()),'negative_pixels':int((~y).sum()),'histogram_bins':2048}
def pro(masks,maps,nth=200,cap=.3):
 masks=(np.asarray(masks)>0).astype(np.uint8);maps=np.asarray(maps,np.float32)
 lo=float(maps.min());hi=float(maps.max());z=(maps-lo)/(hi-lo) if hi-lo>1e-8 else np.zeros_like(maps)
 th=np.linspace(0,1,nth);nh=np.zeros(nth+1,np.int64);nn=0;rh=[]
 for g,a in zip(masks,z):
  lab,k=ndimage.label(g,structure=np.ones((3,3),np.uint8));gb=g.ravel().astype(bool);lf=lab.ravel();af=a.ravel()
  nh+=np.bincount(np.searchsorted(th,af[~gb],side='left'),minlength=nth+1);nn+=int((~gb).sum())
  if k:
   q=lf>0;ids=lf[q].astype(np.int64)-1;bi=np.searchsorted(th,af[q],side='left')
   rh.append(np.bincount(ids*(nth+1)+bi,minlength=k*(nth+1)).reshape(k,nth+1))
 if not rh or not nn:return float('nan')
 h=np.concatenate(rh);ar=h.sum(1);ps=np.array([(h[:,j+1:].sum(1)/ar).mean() for j in range(nth)])
 fs=np.array([nh[j+1:].sum()/nn for j in range(nth)]);ix=np.argsort(fs);fs,ps=fs[ix],ps[ix];ok=fs<=cap
 return float(auc(fs[ok]/cap,ps[ok])) if ok.sum()>1 and fs[ok].max()>0 else 0.
def reg(g,a,f):
 g=np.asarray(g,bool);a=np.asarray(a,np.float32);k=max(1,int(a.size*f));p=np.zeros(a.size,bool);p[np.argpartition(a.ravel(),-k)[-k:]]=1;p=p.reshape(a.shape)
 gl=measure.label(g,connectivity=2);pl=measure.label(p,connectivity=2);ng=int(gl.max());npred=int(pl.max())
 if ng==0:return {'top_precision':float(p[g].sum()/max(1,p.sum())),'top_recall':float(p[g].sum()/max(1,g.sum())),
  'components':0,'recall_iou_0_1':np.nan,'recall_iou_0_3':np.nan,'recall_iou_0_5':np.nan,'mean_best_iou':np.nan,'mean_coverage':np.nan}
 ga=np.bincount(gl.ravel(),minlength=ng+1);pa=np.bincount(pl.ravel(),minlength=npred+1);stride=npred+1
 active=(gl>0)&(pl>0);pairs=gl[active].astype(np.int64)*stride+pl[active].astype(np.int64)
 if pairs.size:
  code,count=np.unique(pairs,return_counts=True);gi=code//stride;pi=code%stride
  iou=count/(ga[gi]+pa[pi]-count);coverage=count/ga[gi];bi=np.zeros(ng+1);bc=np.zeros(ng+1)
  np.maximum.at(bi,gi,iou);np.maximum.at(bc,gi,coverage)
 else:bi=np.zeros(ng+1);bc=np.zeros(ng+1)
 ious=bi[1:];cov=bc[1:]
 return {'top_precision':float(p[g].sum()/max(1,p.sum())),'top_recall':float(p[g].sum()/max(1,g.sum())),
 'components':len(ious),'recall_iou_0_1':float(np.mean(ious>=.1)),
 'recall_iou_0_3':float(np.mean(ious>=.3)),'recall_iou_0_5':float(np.mean(ious>=.5)),
 'mean_best_iou':float(np.mean(ious)),'mean_coverage':float(np.mean(cov))}

if not (OUT/'inference_run.json').is_file():raise RuntimeError('GPU inference incomplete')
manifest=json.loads((OUT/'shared_test_manifest.json').read_text())
allx={};metrics=[];views=[];objects_out=[];regions=[];strata_rows=[];index=[];regress=[]
mapout=OUT/'maps/source_aligned';gtout=OUT/'maps/ground_truth';mapout.mkdir(parents=True,exist_ok=True);gtout.mkdir(exist_ok=True)
for c in CATS:
 rs=sorted([r for r in manifest if r['category']==c],key=lambda r:r['image_path']);paths=[r['image_path'] for r in rs];n=len(rs);raw={}
 for b,directory in [('ArcAD_original','arcad_original'),('ArcAD_fullframe','arcad_full'),('D2_original','d2_original')]:
  with np.load(OUT/'maps'/directory/f'{c}.npz',allow_pickle=False) as z:a=z['maps'].astype(np.float32);pp=z['image_paths'].astype(str).tolist()
  if pp!=paths or a.shape!=(n,256,256) or not np.isfinite(a).all():raise RuntimeError(f'{c}/{b}: invalid map paths/shape/range')
  raw[b]=a
 gf=[];gc=[];total=[];inside=[];outside=[]
 for r in rs:
  m=np.asarray(Image.open(r['mask_abs']).convert('L'))>0 if r['mask_abs'] else np.zeros((1024,1024),bool)
  if m.shape!=(1024,1024):raise RuntimeError(f'{c}: mask shape failure {r["image_path"]}')
  q=m[64:960,64:960];gf.append(np.asarray(Image.fromarray(m.astype('uint8')*255).resize((256,256),Image.Resampling.NEAREST))>0)
  gc.append(np.asarray(Image.fromarray(q.astype('uint8')*255).resize((224,224),Image.Resampling.NEAREST))>0)
  total.append(int(m.sum()));inside.append(int(q.sum()));outside.append(int(m.sum()-q.sum()))
 gf=np.asarray(gf,np.uint8);gc=np.asarray(gc,np.uint8);total=np.array(total);inside=np.array(inside);outside=np.array(outside)
 if not (gtout/f'{c}.npz').exists():np.savez_compressed(gtout/f'{c}.npz',gt_full256=gf,gt_center224=gc,image_paths=np.asarray(paths),
  source_mask_pixels=total,inside_crop_pixels=inside,outside_crop_pixels=outside)
 aligned={};scores={}
 for b,a in raw.items():
  if b=='ArcAD_original':cm=F.interpolate(torch.from_numpy(a[:,None]),size=(224,224),mode='bilinear',align_corners=False)[:,0].numpy().astype(np.float32);fm=None
  else:cm=a[:,16:240,16:240].astype(np.float32);fm=a
  aligned[b]={'center':cm,'full':fm};scores[b]=np.array([top(x) for x in a])
  dest=mapout/b;dest.mkdir(exist_ok=True)
  fields={'map_center224':cm,'image_paths':np.asarray(paths),'object_keys':np.asarray([r['object_key'] for r in rs]),
   'labels':np.asarray([r['label'] for r in rs],np.uint8),'camera_view_ids':np.asarray([r['camera_view_id'] for r in rs],np.uint8)}
  if fm is not None:fields['map_full256']=fm
  if not (dest/f'{c}.npz').exists():np.savez_compressed(dest/f'{c}.npz',**fields)
  for i,r in enumerate(rs):index.append({'category':c,'branch':b,'index':i,'image_path':r['image_path'],'object_key':r['object_key'],
   'camera_view_id':r['camera_view_id'],'label':r['label'],'map_file':str(dest/f'{c}.npz'),'center_grid':'source [64:960] to 224x224','full_grid_available':fm is not None})
 if c in ['lattice_block_plug','laser_diode','power_jack','LED_indicator']:
  ref=pd.read_csv(T2/'tables/view_predictions.csv');ref=ref[ref.category==c].set_index('image_path')
  ds=[abs(scores['ArcAD_original'][i]-float(ref.loc[r['image_path'],'ArcAD_common_top1'])) for i,r in enumerate(rs)]
  mx=float(max(ds));regress.append({'category':c,'views':n,'max_abs_score_delta':mx,'mean_abs_score_delta':float(np.mean(ds)),
   'tolerance':5e-5,'status':'PASS' if mx<=5e-5 else 'FAIL'})
  if mx>5e-5:raise RuntimeError(f'{c}: ArcAD-original Task 02 score regression failed ({mx})')
 groups=defaultdict(list)
 for i,r in enumerate(rs):groups[r['object_key']].append(i)
 ol=np.array([int(any(rs[i]['label'] for i in ix)) for ix in groups.values()],np.uint8);onames=list(groups)
 oi={o:j for j,o in enumerate(onames)};voi=np.array([oi[r['object_key']] for r in rs]);labels=np.array([r['label'] for r in rs],np.uint8)
 oscores={b:np.array([top(raw[b][ix]) for ix in groups.values()]) for b in BR}
 emptyc=(labels==1)&(~gc.reshape(n,-1).any(1));emptyf=(labels==1)&(~gf.reshape(n,-1).any(1));st=[]
 for i,r in enumerate(rs):
  s='normal' if not r['label'] else 'source_mask_empty' if total[i]==0 else 'wholly_outside_original_crop' if inside[i]==0 else 'inside_original_crop' if outside[i]==0 else 'partly_outside_original_crop'
  st.append(s);views.append({'category':c,'image_path':r['image_path'],'object_key':r['object_key'],'camera_view_id':r['camera_view_id'],
   'view_label':r['label'],'object_label':int(ol[oi[r['object_key']]]),'defect_type':r['defect_type'],'source_mask_pixels':int(total[i]),
   'mask_inside_crop_pixels':int(inside[i]),'mask_outside_crop_pixels':int(outside[i]),'source_gt_stratum':s,
   'center_gt_pixels':int(gc[i].sum()),'full_gt_pixels':int(gf[i].sum()),'ArcAD_original_score_top1':scores['ArcAD_original'][i],
   'ArcAD_fullframe_score_top1':scores['ArcAD_fullframe'][i],'D2_original_score_top1':scores['D2_original'][i],
   'original_GT_region_observed':bool(gc[i].any())})
 st=np.asarray(st);allx[c]={'rs':rs,'raw':raw,'aligned':aligned,'gf':gf,'gc':gc,'total':total,'inside':inside,'outside':outside,
  'labels':labels,'groups':groups,'objects':onames,'object_labels':ol,'view_object_idx':voi,'view_scores':scores,'object_scores':oscores,'strata':st}
 for b in BR:
  cm=aligned[b]['center'];visible=(labels==0)|((labels==1)&gc.reshape(n,-1).any(1));p=pix(cm[visible],gc[visible]);p['P_AUPRO']=pro(gc[visible],cm[visible])
  pa=pix(cm,gc);pa['P_AUPRO']=pro(gc,cm)
  metrics.append({'category':c,'branch':b,'field':'central_224','views':n,'objects':len(groups),
   'positive_view_prevalence':float(labels.mean()),'positive_object_prevalence':float(ol.mean()),
   'positive_empty_center_masks':int(emptyc.sum()),'positive_empty_full_masks':int(emptyf.sum()),
   'pixel_policy':'normal views plus positive views with visible GT in this field',**p,
   'P_AUROC_including_empty_positive_as_pixel_negative':pa['P_AUROC'],'P_AP_including_empty_positive_as_pixel_negative':pa['P_AP'],
   'P_AUPRO_including_empty_positive_as_pixel_negative':pa['P_AUPRO'],
   **{f'I_{k}':v for k,v in cls(labels,scores[b]).items()},**{f'O_{k}':v for k,v in cls(ol,oscores[b]).items()}})
  for i in range(n):
   if not labels[i] or not gc[i].any():continue
   rg=reg(gc[i],cm[i],.01);rr={'category':c,'branch':b,'field':'central_224','image_path':rs[i]['image_path'],
    'object_key':rs[i]['object_key'],'camera_view_id':rs[i]['camera_view_id'],'source_gt_stratum':st[i],
    'region_eval_top_fraction':.01,**{k:v for k,v in rg.items() if k not in ['top_precision','top_recall']}}
   flat=cm[i].ravel();gm=gc[i].ravel()
   for f in [.001,.005,.01,.02,.05]:
    k=max(1,int(flat.size*f));ix=np.argpartition(flat,-k)[-k:]
    rr[f'top{f:g}_precision']=float(gm[ix].mean());rr[f'top{f:g}_recall']=float(gm[ix].sum()/max(1,gm.sum()))
   regions.append(rr)
  if b in ['ArcAD_fullframe','D2_original']:
   fm=aligned[b]['full'];vf=(labels==0)|((labels==1)&gf.reshape(n,-1).any(1));q=pix(fm[vf],gf[vf]);q['P_AUPRO']=pro(gf[vf],fm[vf])
   qa=pix(fm,gf);qa['P_AUPRO']=pro(gf,fm)
   metrics.append({'category':c,'branch':b,'field':'full_source_256','views':n,'objects':len(groups),
    'positive_view_prevalence':float(labels.mean()),'positive_object_prevalence':float(ol.mean()),
    'positive_empty_center_masks':int(emptyc.sum()),'positive_empty_full_masks':int(emptyf.sum()),
    'pixel_policy':'normal views plus positive views with visible full-source GT',**q,
    'P_AUROC_including_empty_positive_as_pixel_negative':qa['P_AUROC'],'P_AP_including_empty_positive_as_pixel_negative':qa['P_AP'],
    'P_AUPRO_including_empty_positive_as_pixel_negative':qa['P_AUPRO']})
   for i in range(n):
    if not labels[i] or not gf[i].any():continue
    rg=reg(gf[i],fm[i],.01);rr={'category':c,'branch':b,'field':'full_source_256','image_path':rs[i]['image_path'],
     'object_key':rs[i]['object_key'],'camera_view_id':rs[i]['camera_view_id'],'source_gt_stratum':st[i],
     'region_eval_top_fraction':.01,**{k:v for k,v in rg.items() if k not in ['top_precision','top_recall']}}
    flat=fm[i].ravel();gm=gf[i].ravel()
    for f in [.001,.005,.01,.02,.05]:
     k=max(1,int(flat.size*f));ix=np.argpartition(flat,-k)[-k:]
     rr[f'top{f:g}_precision']=float(gm[ix].mean());rr[f'top{f:g}_recall']=float(gm[ix].sum()/max(1,gm.sum()))
    regions.append(rr)
 for o,ix in groups.items():objects_out.append({'category':c,'object_key':o,'object_label':int(any(labels[ix])),'views':len(ix),
  'ArcAD_original_score_top1_pooled5':top(raw['ArcAD_original'][ix]),'ArcAD_fullframe_score_top1_pooled5':top(raw['ArcAD_fullframe'][ix]),
  'D2_original_score_top1_pooled5':top(raw['D2_original'][ix])})
 for s in sorted(set(st)):
  ix=np.where(st==s)[0];strata_rows.append({'category':c,'source_gt_stratum':s,'views':len(ix),'positive_views':int(labels[ix].sum()),
   'median_source_mask_pixels':float(np.median(total[ix])),'median_inside_crop_pixels':float(np.median(inside[ix])),'median_outside_crop_pixels':float(np.median(outside[ix]))})

rng=np.random.default_rng(SEED);boot=[];invalid=defaultdict(int)
for c,d in allx.items():
 n=len(d['objects']);voi=d['view_object_idx'];y=d['labels'];oy=d['object_labels']
 for rep in range(BOOT):
  draw=rng.integers(0,n,n);cnt=np.bincount(draw,minlength=n);oi=np.flatnonzero(cnt);vi=np.flatnonzero(cnt[voi]>0)
  if len(np.unique(y[vi]))<2 or len(np.unique(oy[oi]))<2:invalid[c]+=1;continue
  vw=cnt[voi[vi]].astype(float);ow=cnt[oi].astype(float);z={'category':c,'replicate':rep}
  for b in BR:
   for lev,yy,ss,w in [('I',y[vi],d['view_scores'][b][vi],vw),('O',oy[oi],d['object_scores'][b][oi],ow)]:
    z[f'{b}_{lev}_AUROC']=roc_auc_score(yy,ss,sample_weight=w);z[f'{b}_{lev}_AP']=average_precision_score(yy,ss,sample_weight=w)
  for lev in ['I','O']:
   for m in ['AUROC','AP']:
    z[f'delta_full_minus_original_{lev}_{m}']=z[f'ArcAD_fullframe_{lev}_{m}']-z[f'ArcAD_original_{lev}_{m}']
    z[f'delta_full_minus_D2_{lev}_{m}']=z[f'ArcAD_fullframe_{lev}_{m}']-z[f'D2_original_{lev}_{m}']
  boot.append(z)
cis=[]
for c in CATS:
 for e in ['delta_full_minus_original_I_AUROC','delta_full_minus_original_I_AP','delta_full_minus_original_O_AUROC','delta_full_minus_original_O_AP',
           'delta_full_minus_D2_I_AUROC','delta_full_minus_D2_I_AP','delta_full_minus_D2_O_AUROC','delta_full_minus_D2_O_AP']:
  x=np.asarray([r[e] for r in boot if r['category']==c]);cis.append({'category':c,'effect':e,'valid_replicates':len(x),'invalid_single_class_draws':invalid[c],
   'requested_replicates':BOOT,'seed':SEED,'cluster_unit':'physical object; all five views retained',
   'ci95_low':float(np.quantile(x,.025)) if len(x) else np.nan,'ci95_high':float(np.quantile(x,.975)) if len(x) else np.nan})

# Required 39 full-source GT cases: original ArcAD has no observed border score.
empty=pd.read_csv(T3/'empty_mask_audit.csv');border=empty[empty.status=='nonzero_mask_outside_[64:960]_crop'];border_rows=[]
for _,e in border.iterrows():
 c=e.category;d=allx[c];i=next(i for i,r in enumerate(d['rs']) if r['image_path']==e.image_path);g=d['gf'][i].astype(bool)
 z={'category':c,'object_key':e.object_key,'camera_view_id':int(e.camera_view_id),'image_path':e.image_path,'defect_type':e.defect_type,
  'source_mask_pixels':int(d['total'][i]),'mask_area_inside_crop':int(d['inside'][i]),'mask_area_outside_crop':int(d['outside'][i]),
  'visibility_flag':'requires_manual_review','ArcAD_original_full_field_observed':False,
  'ArcAD_original_view_score':d['view_scores']['ArcAD_original'][i],'ArcAD_fullframe_view_score':d['view_scores']['ArcAD_fullframe'][i],
  'D2_original_view_score':d['view_scores']['D2_original'][i],'ArcAD_original_GT_region_score':'unobserved outside crop',
  'ArcAD_fullframe_GT_region_mean':float(d['aligned']['ArcAD_fullframe']['full'][i][g].mean()),
  'ArcAD_fullframe_GT_region_max':float(d['aligned']['ArcAD_fullframe']['full'][i][g].max()),
  'D2_GT_region_mean':float(d['aligned']['D2_original']['full'][i][g].mean()),'D2_GT_region_max':float(d['aligned']['D2_original']['full'][i][g].max()),
  'threshold_status':'none; no independent calibration'}
 for b,k in [('ArcAD_fullframe','ArcAD_fullframe_GT_mean_percentile'),('D2_original','D2_GT_mean_percentile')]:
  a=d['aligned'][b]['full'][i];z[k]=float(np.mean(a[~g]<=a[g].mean())) if g.any() else np.nan
 border_rows.append(z)
if len(border_rows)!=39:raise RuntimeError(f'expected 39 crop-external nonempty masks, got {len(border_rows)}')

# Four cases per contact-sheet page; visibility decisions remain pending.
sheet=OUT/'border_39_contact_sheet';sheet.mkdir(exist_ok=True)
for page,start in enumerate(range(0,39,4),1):
 batch=border_rows[start:start+4];fig,ax=plt.subplots(len(batch),4,figsize=(14,3.5*len(batch)),squeeze=False)
 for j,z in enumerate(batch):
  d=allx[z['category']];i=next(i for i,r in enumerate(d['rs']) if r['image_path']==z['image_path']);r=d['rs'][i]
  im=np.asarray(Image.open(r['image_abs']).convert('RGB'));g=np.asarray(Image.open(r['mask_abs']).convert('L'))>0
  crop=Image.fromarray(im).crop((64,64,960,960)).resize((224,224),Image.Resampling.BILINEAR)
  cg=np.asarray(Image.fromarray(g.astype('uint8')*255).crop((64,64,960,960)).resize((224,224),Image.Resampling.NEAREST))>0
  ax[j,0].imshow(im);ax[j,0].set_title('Source image');ax[j,1].imshow(g,cmap='gray');ax[j,1].set_title('Uncropped GT')
  ax[j,2].imshow(crop);ax[j,2].imshow(cg,cmap='Reds',alpha=.7);ax[j,2].set_title('Crop + GT')
  ax[j,3].imshow(d['aligned']['ArcAD_fullframe']['full'][i],cmap='magma');ax[j,3].set_title('ArcAD full-frame map')
  for a in ax[j]:a.axis('off')
  ax[j,0].set_ylabel(f'{z["object_key"]} C{z["camera_view_id"]}; manual review',fontsize=8)
 fig.tight_layout();fig.savefig(sheet/f'page_{page:02d}.png',dpi=110);plt.close(fig)
writecsv(sheet/'manual_review.csv',[{**x,'review_decision':'pending expert review'} for x in border_rows])

# Five camera paired panels, using a branch-shared display scale across all views in that object.
pair=OUT/'plots/paired_five_camera';pair.mkdir(parents=True,exist_ok=True)
for c,d in allx.items():
 cand=[(o,ix) for o,ix in d['groups'].items() if len(ix)==5 and any(d['labels'][ix])]
 if c=='laser_diode':
  ed=[x for x in cand if any(d['outside'][i]>0 and d['inside'][i]==0 for i in x[1])]
  if ed:cand=ed
 o,ix=sorted(cand,key=lambda x:x[0])[0];ix=sorted(ix,key=lambda i:d['rs'][i]['camera_view_id'])
 fig,ax=plt.subplots(5,5,figsize=(15,14));titles=['Image crop','GT','ArcAD original','ArcAD full-frame','D2 original']
 for j,t in enumerate(titles):ax[0,j].set_title(t)
 mats={b:np.stack([d['aligned'][b]['center'][i] for i in ix]) for b in BR}
 for rix,i in enumerate(ix):
  r=d['rs'][i];im=np.asarray(Image.open(r['image_abs']).convert('RGB').crop((64,64,960,960)).resize((224,224),Image.Resampling.BILINEAR))
  ax[rix,0].imshow(im);ax[rix,0].set_ylabel(f'C{r["camera_view_id"]}',rotation=0,labelpad=18);ax[rix,1].imshow(d['gc'][i],cmap='gray')
  for col,b in zip([2,3,4],BR):ax[rix,col].imshow(mats[b][rix],cmap='magma',vmin=float(mats[b].min()),vmax=float(mats[b].max()))
  for a in ax[rix]:a.axis('off')
 fig.suptitle(f'{c} | {o}; within-branch color scales shared across C1-C5');fig.tight_layout();fig.savefig(pair/f'{c}.png',dpi=120);plt.close(fig)

# Per-category ROC/PR, score distributions, and border/center score plots.
plot=OUT/'plots';plot.mkdir(exist_ok=True);colors={'ArcAD_original':'#1976d2','ArcAD_fullframe':'#ef6c00','D2_original':'#388e3c'};curves=[]
for c,d in allx.items():
 y=d['labels'];oy=d['object_labels'];fig,ax=plt.subplots(2,2,figsize=(10,8))
 for b in BR:
  for lev,yy,ss,ar,ap in [('view',y,d['view_scores'][b],ax[0,0],ax[0,1]),('object',oy,d['object_scores'][b],ax[1,0],ax[1,1])]:
   fpr,tpr,_=roc_curve(yy,ss);pr,re,_=precision_recall_curve(yy,ss);ar.plot(fpr,tpr,color=colors[b],label=b);ap.plot(re,pr,color=colors[b],label=b)
   curves.extend([{'category':c,'level':lev,'branch':b,'curve':'ROC','x':float(x),'y':float(q),'prevalence':float(yy.mean())} for x,q in zip(fpr,tpr)])
   curves.extend([{'category':c,'level':lev,'branch':b,'curve':'PR','x':float(x),'y':float(q),'prevalence':float(yy.mean())} for x,q in zip(re,pr)])
 for a,t in zip(ax.flat,['View ROC','View PR','Object ROC','Object PR']):a.set_title(t);a.grid(alpha=.2);a.legend(fontsize=7)
 fig.suptitle(f'{c}; PR baselines equal prevalence');fig.tight_layout();fig.savefig(plot/f'roc_pr_{c}.png',dpi=130);plt.close(fig)
 fig,a=plt.subplots(figsize=(9,5))
 for b in BR:
  a.hist(d['view_scores'][b][y==0],bins=30,density=True,histtype='step',ls='--',color=colors[b],label=b+' normal')
  a.hist(d['view_scores'][b][y==1],bins=30,density=True,histtype='step',color=colors[b],label=b+' positive')
 a.set_title(f'{c} view top-1% scores');a.set_xlabel('Native anomaly score');a.set_ylabel('Density');a.grid(alpha=.2);a.legend(fontsize=7,ncol=2)
 fig.tight_layout();fig.savefig(plot/f'score_distributions_{c}.png',dpi=130);plt.close(fig)
 order=['normal','inside_original_crop','partly_outside_original_crop','wholly_outside_original_crop','source_mask_empty']
 fig,a=plt.subplots(figsize=(9,4));a.boxplot([d['view_scores']['ArcAD_original'][d['strata']==s] for s in order],labels=order,showfliers=False)
 a.tick_params(axis='x',rotation=25);a.set_ylabel('ArcAD-original top-1% score');a.set_title(f'{c}: border vs center');a.grid(axis='y',alpha=.2)
 fig.tight_layout();fig.savefig(plot/f'border_center_{c}.png',dpi=130);plt.close(fig)
writecsv(OUT/'roc_pr_curves.csv',curves)

writecsv(OUT/'category_metrics.csv',metrics);writecsv(OUT/'per_view_scores.csv',views);writecsv(OUT/'per_object_scores.csv',objects_out)
writecsv(OUT/'per_view_region_metrics.csv',regions);writecsv(OUT/'gt_source_mask_strata.csv',strata_rows);writecsv(OUT/'maps/index.csv',index)
writecsv(OUT/'paired_bootstrap_ci.csv',cis);writecsv(OUT/'paired_bootstrap_replicates.csv',boot)
writecsv(OUT/'arcad_original_regression.csv',regress);writecsv(OUT/'border_39_audit.csv',border_rows)

md=['# Task 04A summary','',
 '## (a) Did full-frame ArcAD help overall?',
 'See per-category I/O AUROC/AP and center/full-source pixel results in category_metrics.csv; paired physical-object CIs for full-frame minus original ArcAD are in paired_bootstrap_ci.csv. This is an exploratory five-category comparison.',
 '',
 '## (b) Did it help the border cases?',
 f'border_39_audit.csv has {len(border_rows)} crop-external views. ArcAD-original has no border prediction; the full-frame ArcAD and D2 rows provide continuous GT-region scores, not thresholded decisions. All 39 require manual visibility review.',
 '',
 '## (c) Did it degrade central or small defects?',
 'Central pixel, top-k, and component summaries are in category_metrics.csv and per_view_region_metrics.csv, with source area strata in gt_source_mask_strata.csv. Effects are descriptive.',
 '',
 '## (d) Did object-level aggregation alter the conclusion?',
 'View and object metrics are separate. Object scores pool the top 1% over all five maps, rather than averaging view scores. Bootstrap resamples physical objects and retains all sibling views.',
 '',
 '## (e) Is retraining warranted?',
 'This frozen-checkpoint intervention cannot establish training-time benefit. If motivated by these results, run a separate matched-budget, matched-split, multi-seed training ablation.',
 '',
 '## Design and limits',
 f'{len(manifest)} views from five preselected categories were evaluated. ArcAD full-frame is resize-to-392, no crop; ArcAD-original is resize448 then crop392; D2 remains full-frame 280. No training or test-fitted threshold was used. Pixel metrics exclude positive views whose masks are empty in the evaluated field, with companion columns including them as pixel negatives. Calibration is unavailable. ArcAD-original outside-crop pixels are unobserved. No ArcAD legacy artifacts were used.']
(OUT/'summary.md').write_text('\n'.join(md)+'\n')
print(json.dumps({'categories':CATS,'views':len(manifest),'border_39':len(border_rows),'bootstrap_replicates':len(boot),
 'invalid_draws':dict(invalid),'arcad_original_regression':regress,'metric_rows':len(metrics)},indent=2))

