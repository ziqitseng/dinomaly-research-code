#!/usr/bin/env python3
"""Task 03 threshold-free, object-clustered and fixed 50/50 fusion follow-up.
No model inference, training, or test-label threshold fitting occurs here.
Test-normal sweeps are descriptive diagnostics only.
"""
import csv,json,sys,math
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve,precision_recall_curve
from skimage import measure
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parents[1]
T2=ROOT/'analysis_shared_failure_20261103';CATS=['lattice_block_plug','laser_diode','power_jack','LED_indicator','aircraft_model_head']
MODELS=['D2','ArcAD','Mean50'];SEED=20261103;BOOT=2000;FPR_OBJECT_BUDGETS=[0,1,2,3,5,10]
sys.path.insert(0,str(T2));import run_analysis as task2

def writecsv(path,rows):
 rows=list(rows)
 if not rows:path.write_text('');return
 with open(path,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),extrasaction='ignore');w.writeheader();w.writerows(rows)
def score_top(a,f=.01):
 x=np.asarray(a,dtype=np.float32).reshape(-1);k=max(1,int(x.size*f));return float(np.partition(x,-k)[-k:].mean())
def cls(y,s):
 y=np.asarray(y,dtype=np.uint8);s=np.asarray(s,dtype=float)
 if len(np.unique(y))<2:return {'AUROC':float('nan'),'AP':float('nan')}
 return {'AUROC':float(roc_auc_score(y,s)),'AP':float(average_precision_score(y,s))}
def hist_pixel_metrics(maps,masks,bins_n=2048):
 maps=np.asarray(maps,dtype=np.float32);masks=np.asarray(masks,dtype=bool);lo=float(maps.min());hi=float(maps.max());edges=np.linspace(lo,hi+1e-12,bins_n+1)
 pos=np.histogram(maps[masks],bins=edges)[0][::-1].cumsum().astype(float);neg=np.histogram(maps[~masks],bins=edges)[0][::-1].cumsum().astype(float)
 P=max(1.,float(masks.sum()));N=max(1.,float((~masks).sum()));rec=pos/P;fpr=neg/N;prec=pos/np.maximum(1,pos+neg)
 ap=float(np.sum(np.diff(np.r_[0.,rec])*prec));auc=float(np.trapezoid(np.r_[0.,rec],np.r_[0.,fpr])) if hasattr(np,'trapezoid') else float(np.trapz(np.r_[0.,rec],np.r_[0.,fpr]))
 return auc,ap
def descriptive_threshold(normal_scores,k):
 x=np.asarray(normal_scores,dtype=float)
 if k<=0:return float(np.nextafter(np.max(x),np.inf)) if len(x) else float('inf')
 return float(np.sort(x)[::-1][min(k,len(x))-1])
def component_summary(gt,pred):
 gl=measure.label(np.asarray(gt)>0,connectivity=2);pl=measure.label(np.asarray(pred)>0,connectivity=2);hits=[];cov=[]
 for rp in measure.regionprops(gl):
  gm=gl==rp.label;cand=np.unique(pl[gm]);cand=cand[cand>0];best=0.;bestcov=0.
  for pid in cand:
   pm=pl==pid;inter=int((gm&pm).sum());union=int((gm|pm).sum());best=max(best,inter/max(1,union));bestcov=max(bestcov,inter/max(1,int(gm.sum())))
  hits.append(best);cov.append(bestcov)
 return {'gt_regions':len(hits),'recall_iou_0_1':float(np.mean(np.array(hits)>=.1)) if hits else float('nan'),
  'recall_iou_0_3':float(np.mean(np.array(hits)>=.3)) if hits else float('nan'),'recall_iou_0_5':float(np.mean(np.array(hits)>=.5)) if hits else float('nan'),
  'mean_best_iou':float(np.mean(hits)) if hits else float('nan'),'mean_coverage':float(np.mean(cov)) if cov else float('nan')}
def gt_geom(gt):
 lab=measure.label(gt,connectivity=2);rp=measure.regionprops(lab)
 widths=[min(p.bbox[2]-p.bbox[0],p.bbox[3]-p.bbox[1]) for p in rp]
 return len(rp),min(widths) if widths else 0
def display_crop(path):
 return Image.open(path).convert('RGB').crop((64,64,960,960)).resize((224,224),Image.Resampling.BILINEAR)

manifest=json.loads((T2/'shared_test_manifest.json').read_text());groups_by_cat={c:sorted([r for r in manifest if r['category']==c],key=lambda r:r['image_path']) for c in CATS}
validation=json.loads((OUT/'alignment_validation_224.json').read_text())
if validation['status']!='PASS_224_INDEPENDENT_ALIGNMENT':raise RuntimeError('224 geometry validation failed')

# In-sample normal-only score references, aligned identically to test maps. They are
# used only to make a prespecified exploratory equal-weight score-average baseline.
calraw=json.loads((T2/'calibration_manifest.json').read_text());cal_scores={m:{'view':[],'object':[]} for m in ['D2','ArcAD']};cal_counts={}
for cat in CATS:
 for model in ['D2','ArcAD']:
  p=T2/'calibration_maps'/model.lower()/f'{cat}.npz'
  with np.load(p,allow_pickle=False) as z:maps=z['maps'].astype(np.float32);objs=z['object_keys'].astype(str)
  if model=='D2':maps=maps[:,16:240,16:240]
  else:maps=F.interpolate(torch.from_numpy(maps[:,None]),size=(224,224),mode='bilinear',align_corners=False)[:,0].numpy()
  cdata=calraw['categories'][cat][f'{model}_train_normals'];cal_counts[f'{cat}_{model}']={'objects':len(set(objs)),'views':len(objs),'records':len(cdata['records'])}
  object_scores=[]
  for obj in sorted(set(objs)):
   ix=np.where(objs==obj)[0];group=maps[ix]
   for im in group:cal_scores[model]['view'].append(score_top(im,.01))
   object_scores.append(score_top(group,.01))
  cal_scores[model]['object'].extend(object_scores)
norm_params={}
for m in ['D2','ArcAD']:
 for level in ['view','object']:
  vals=np.asarray(cal_scores[m][level],float);norm_params[f'{m}_{level}']={'mean':float(vals.mean()),'std':float(vals.std(ddof=0)) or 1.,'n_views_or_objects':len(vals)}

metrics=[];sens=[];view_rows=[];object_rows=[];camera_rows=[];stratum_rows=[];score_dists=[];curves=[];failure=[];op_rows=[];region_rows=[];all_category_data={}
size_bins=[0,.001,.01,.05,1.01];size_names=['<0.1%','0.1–1%','1–5%','≥5%']
for cat in CATS:
 rs=groups_by_cat[cat];paths=[r['image_path'] for r in rs]
 z=np.load(OUT/'maps224'/f'{cat}.npz',allow_pickle=False);d=z['d2_maps'];a=z['arcad_maps'];g=z['gt_masks'].astype(bool)
 if z['image_paths'].tolist()!=paths:raise RuntimeError(f'{cat}: maps224 manifest order mismatch')
 y=np.asarray([r['label'] for r in rs],np.uint8);objkeys=np.asarray([r['object_key'] for r in rs]);views=np.asarray([r['camera_view_id'] for r in rs],int)
 labels={m:(d if m=='D2' else a) for m in ['D2','ArcAD']}
 view_scores={m:{f:np.asarray([score_top(mp,f) for mp in labels[m]],float) for f in [.001,.005,.01,.02,.05]} for m in ['D2','ArcAD']}
 # Apply fixed global z transforms learned only from model-normal training references.
 view_scores['Mean50']=.5*((view_scores['D2'][.01]-norm_params['D2_view']['mean'])/norm_params['D2_view']['std']+ (view_scores['ArcAD'][.01]-norm_params['ArcAD_view']['mean'])/norm_params['ArcAD_view']['std'])
 objlist=sorted(set(objkeys));oidx={o:np.where(objkeys==o)[0] for o in objlist};oy=np.asarray([int(y[ix].max()) for ix in [oidx[o] for o in objlist]],np.uint8)
 obj_scores={}
 for m in ['D2','ArcAD']:
  maps=labels[m];obj_scores[m]=np.asarray([score_top(maps[oidx[o]],.01) for o in objlist],float)
 obj_scores['Mean50']=.5*((obj_scores['D2']-norm_params['D2_object']['mean'])/norm_params['D2_object']['std']+(obj_scores['ArcAD']-norm_params['ArcAD_object']['mean'])/norm_params['ArcAD_object']['std'])
 view_idx={o:i for i,o in enumerate(objlist)}
 for oi,obj in enumerate(objlist):
  ix=oidx[obj];object_rows.append({'category':cat,'object_key':obj,'gt_object_label':int(oy[oi]),'views':len(ix),'positive_views':int(y[ix].sum()),
   'D2_object_top1':float(obj_scores['D2'][oi]),'ArcAD_object_top1':float(obj_scores['ArcAD'][oi]),'Mean50_object_score_z':float(obj_scores['Mean50'][oi]),
   'D2_view_score_mean':float(view_scores['D2'][.01][ix].mean()),'D2_view_score_sd':float(view_scores['D2'][.01][ix].std()),
   'ArcAD_view_score_mean':float(view_scores['ArcAD'][.01][ix].mean()),'ArcAD_view_score_sd':float(view_scores['ArcAD'][.01][ix].std()),
   'D2_score_camera_C1':float(view_scores['D2'][.01][ix[np.argmin(views[ix])]]),'D2_score_camera_C5':float(view_scores['D2'][.01][ix[np.argmax(views[ix])]])})
 all_category_data[cat]={'records':rs,'d2':d,'arcad':a,'gt':g,'labels':y,'objects':objkeys,'object_list':objlist,'object_indices':oidx,'object_labels':oy,'view_scores':view_scores,'object_scores':obj_scores}
 nobj=len(objlist);pview=int(y.sum());pobj=int(oy.sum());prev=float(y.mean());oprev=float(oy.mean());empty_pos=int(sum(int(y[i] and not g[i].any()) for i in range(len(rs))))
 pixel_valid=(~y.astype(bool))|g.reshape(len(g),-1).any(axis=1)
 category_rec={'category':cat,'n_views':len(rs),'n_objects':nobj,'n_positive_views':pview,'n_positive_objects':pobj,'I_positive_prevalence':prev,'O_positive_prevalence':oprev,'I_AP_prevalence_baseline':prev,'O_AP_prevalence_baseline':oprev,'positive_views_empty_common_gt':empty_pos,'positive_views_excluded_pixel_metrics':empty_pos}
 scores_all={'D2':view_scores['D2'][.01],'ArcAD':view_scores['ArcAD'][.01],'Mean50':view_scores['Mean50']}
 for m in MODELS:
  im=cls(y,scores_all[m]);os=cls(oy,obj_scores[m]);category_rec.update({f'{m}_I_AUROC':im['AUROC'],f'{m}_I_AP':im['AP'],f'{m}_O_AUROC':os['AUROC'],f'{m}_O_AP':os['AP']})
  for level,yy,ss in [('view',y,scores_all[m]),('object',oy,obj_scores[m])]:
   for status,mask in [('normal',yy==0),('defect',yy==1)]:
    vals=ss[mask]
    score_dists.append({'category':cat,'model':m,'level':level,'sample_class':status,'n':len(vals),'mean':float(vals.mean()) if len(vals) else float('nan'),
     'std':float(vals.std(ddof=0)) if len(vals) else float('nan'),'q05':float(np.quantile(vals,.05)) if len(vals) else float('nan'),
     'median':float(np.quantile(vals,.5)) if len(vals) else float('nan'),'q95':float(np.quantile(vals,.95)) if len(vals) else float('nan')})
  # Operating-point thresholds are descriptive test-normal score ranks, never calibrated.
  normals=np.asarray([i for i,o in enumerate(objlist) if not oy[i]],int)
  for budget in FPR_OBJECT_BUDGETS:
   for m in ['D2','ArcAD']:
    normal_obj_view=np.asarray([float(view_scores[m][.01][oidx[o]].max()) for o in objlist if oy[view_idx[o]]==0])
    tv=descriptive_threshold(normal_obj_view,budget)
    viewflag=view_scores[m][.01]>=tv
    normal_view_obj_fp=sum(bool(np.any(viewflag[oidx[o]])) for o in objlist if oy[view_idx[o]]==0)
    normal_views_fp=int(sum(viewflag[i] for i in range(len(rs)) if y[i]==0))
    posdet=int(sum(viewflag[i] for i in range(len(rs)) if y[i]==1))
    opred= obj_scores[m]>=descriptive_threshold(obj_scores[m][normals],budget)
    normal_obj_fp=int(sum(opred[i] for i in normals));pos_obj_det=int(sum(opred[i] for i in range(len(objlist)) if oy[i]))
    op_rows.append({'category':cat,'model':m,'level':'view','normal_object_false_positive_budget':budget,'calibration_source':'held-out test normals, descriptive only','test_normal_objects':len(normals),'threshold_native_score':tv,
      'actual_normal_object_fp_count':normal_view_obj_fp,'actual_normal_object_fpr':normal_view_obj_fp/max(1,len(normals)),'normal_fp_views':normal_views_fp,'positive_views_detected':posdet,'positive_views':pview,
      'positive_view_recall':posdet/max(1,pview),'positive_object_recall_any_view':sum(bool(np.any(viewflag[oidx[o]])) for o in objlist if oy[view_idx[o]])/max(1,pobj)})
    op_rows.append({'category':cat,'model':m,'level':'object','normal_object_false_positive_budget':budget,'calibration_source':'held-out test normals, descriptive only','test_normal_objects':len(normals),
      'threshold_native_score':descriptive_threshold(obj_scores[m][normals],budget),'actual_normal_object_fp_count':normal_obj_fp,'actual_normal_object_fpr':normal_obj_fp/max(1,len(normals)),
      'normal_fp_views':int(sum(opred[view_idx[o]] for o in objlist if oy[view_idx[o]]==0)),'positive_views_detected':int(sum(opred[view_idx[o]] for o in objlist if oy[view_idx[o]])),'positive_views':pobj,
      'positive_view_recall':pos_obj_det/max(1,pobj),'positive_object_recall_any_view':pos_obj_det/max(1,pobj)})
   # paired detection overlap on view and object units
   tvs={m:descriptive_threshold(np.asarray([float(view_scores[m][.01][oidx[o]].max()) for o in objlist if oy[view_idx[o]]==0]),budget) for m in ['D2','ArcAD']}
   fviews={m:view_scores[m][.01]>=tvs[m] for m in ['D2','ArcAD']}
   opth={m:descriptive_threshold(obj_scores[m][normals],budget) for m in ['D2','ArcAD']};fo={m:obj_scores[m]>=opth[m] for m in ['D2','ArcAD']}
   for level,lab,flags,groups in [('view',y,fviews,oidx),('object',oy,fo,{o:np.array([view_idx[o]]) for o in objlist})]:
    posidx=np.where(lab==1)[0];negidx=np.where(lab==0)[0];both=int(np.sum(flags['D2'][posidx]&flags['ArcAD'][posidx]));donly=int(np.sum(flags['D2'][posidx]&~flags['ArcAD'][posidx]));aonly=int(np.sum(flags['ArcAD'][posidx]&~flags['D2'][posidx]));neither=int(np.sum(~flags['D2'][posidx]&~flags['ArcAD'][posidx]))
    fpD=sum(bool(np.any(flags['D2'][groups[o]])) for o in objlist if oy[view_idx[o]]==0);fpA=sum(bool(np.any(flags['ArcAD'][groups[o]])) for o in objlist if oy[view_idx[o]]==0);fpU=sum(bool(np.any(flags['D2'][groups[o]]) or np.any(flags['ArcAD'][groups[o]])) for o in objlist if oy[view_idx[o]]==0)
    failure.append({'category':cat,'level':level,'normal_object_fp_budget':budget,'nominal_fpr_fraction':budget/max(1,len(normals)),'test_normal_objects':len(normals),
      'D2_threshold':tvs['D2'] if level=='view' else opth['D2'],'ArcAD_threshold':tvs['ArcAD'] if level=='view' else opth['ArcAD'],
      'positive_units':len(posidx),'both_detected':both,'D2_only':donly,'ArcAD_only':aonly,'neither':neither,'D2_positive_recall':(both+donly)/max(1,len(posidx)),
      'ArcAD_positive_recall':(both+aonly)/max(1,len(posidx)),'OR_positive_recall_descriptive':(both+donly+aonly)/max(1,len(posidx)),
      'D2_false_positive_normal_objects':fpD,'ArcAD_false_positive_normal_objects':fpA,'OR_false_positive_normal_object_union':fpU,'OR_normal_object_fpr':fpU/max(1,len(normals)),
      'interpretation':'test-normal rank sweep; descriptive, not an independent calibration/operating point'})
  # Score sensitivity to top-pixel fraction; no best fraction is selected.
  for m in ['D2','ArcAD']:
   for frac in [.001,.005,.01,.02,.05]:
    met=cls(y,view_scores[m][frac]);sens.append({'category':cat,'model':m,'top_pixel_fraction':frac,'I_AUROC':met['AUROC'],'I_AP':met['AP'],'view_prevalence_baseline':prev})
 # Pixel metrics exclude positive views with empty common-crop masks.
 for m,mp in [('D2',d),('ArcAD',a)]:
  pixm=mp[pixel_valid];pixg=g[pixel_valid];pauc,pap=hist_pixel_metrics(pixm,pixg,2048);pro=task2.fast_pro(pixg,pixm,nth=200,cap=.3)
  category_rec.update({f'{m}_P_AUROC_hist2048':pauc,f'{m}_P_AP_hist2048':pap,f'{m}_P_AUPRO_FPR0.3':pro})
 # Per-view maps: localization stats and independent RGB covariates.
 view_threshold={m:descriptive_threshold(np.asarray([view_scores[m][.01][oidx[o]].max() for o in objlist if oy[view_idx[o]]==0]),1) for m in ['D2','ArcAD']}
 for i,r in enumerate(rs):
  row={'category':cat,'object_key':r['object_key'],'image_path':r['image_path'],'view_id':r['camera_view_id'],'gt_view_label':int(y[i]),'gt_object_label':int(oy[view_idx[r['object_key']]]),
    'defect_type':r.get('defect_type') or '', 'gt_area_px':int(g[i].sum()),'gt_area_fraction':float(g[i].mean()),'gt_visible_in_common_crop':int(g[i].any()),
    'D2_score_top1':float(view_scores['D2'][.01][i]),'ArcAD_score_top1':float(view_scores['ArcAD'][.01][i]),'Mean50_score_z':float(view_scores['Mean50'][i]),
    'D2_test_descriptive_hit_budget1':int(view_scores['D2'][.01][i]>=view_threshold['D2']),'ArcAD_test_descriptive_hit_budget1':int(view_scores['ArcAD'][.01][i]>=view_threshold['ArcAD'])}
  if g[i].any():
   nreg,minw=gt_geom(g[i]);row.update({'gt_region_count':nreg,'gt_min_component_width_px':minw})
   for m,mp in [('D2',d),('ArcAD',a)]:
    q=mp[i];gp=g[i];bg=~gp;mg=float(q[gp].mean());mb=float(q[bg].mean());sd=float(q[bg].std())
    row[m+'_C']=mg-mb;row[m+'_Z']=(mg-mb)/(sd+1e-12);row[m+'_Q']=float(roc_auc_score(gp.ravel(),q.ravel()))
    bg_sort=np.sort(q[bg]);row[m+'_GT_max_percentile']=float(np.searchsorted(bg_sort,q[gp].max(),side='right')/len(bg_sort));row[m+'_GT_mean_percentile']=float(np.searchsorted(bg_sort,q[gp],side='right').mean()/len(bg_sort))
    for f in [.001,.005,.01,.02,.05]:
     k=max(1,int(q.size*f));inds=np.argpartition(q.ravel(),-k)[-k:];prec=float(gp.ravel()[inds].mean());rec=float(gp.ravel()[inds].sum()/gp.sum())
     row[f'{m}_top{f:g}_precision']=prec;row[f'{m}_top{f:g}_recall']=rec
    row[m+'_off_target_top1_fraction']=1-row[m+'_top0.01_precision']
   # Input RGB contrast/gradient/texture are separate from model-derived anomaly score contrast.
   gray=np.asarray(display_crop(r['image_abs']).convert('L'),dtype=np.float32)/255.;ring=ndimage.binary_dilation(g[i],iterations=5)&~g[i]
   gy=ndimage.sobel(gray,axis=0);gx=ndimage.sobel(gray,axis=1);grad=np.hypot(gx,gy)
   if ring.any():
    row.update({'rgb_luminance_contrast_gt_minus_ring':float(gray[g[i]].mean()-gray[ring].mean()),'rgb_gradient_contrast_gt_minus_ring':float(grad[g[i]].mean()-grad[ring].mean()),'rgb_background_texture_std_ring':float(gray[ring].std())})
  else:row.update({'gt_region_count':0,'gt_min_component_width_px':0,'D2_Q':np.nan,'ArcAD_Q':np.nan})
  view_rows.append(row)
  camera_rows.append({'category':cat,'object_key':r['object_key'],'camera_view_id':r['camera_view_id'],'gt_view_label':int(y[i]),'gt_object_label':int(oy[view_idx[r['object_key']]]),'gt_area_px':int(g[i].sum()),'gt_area_fraction':float(g[i].mean()),
    'D2_score_top1':float(view_scores['D2'][.01][i]),'ArcAD_score_top1':float(view_scores['ArcAD'][.01][i]),'D2_hit_descriptive_one_normal_object_fp':int(view_scores['D2'][.01][i]>=view_threshold['D2']),'ArcAD_hit_descriptive_one_normal_object_fp':int(view_scores['ArcAD'][.01][i]>=view_threshold['ArcAD'])})
  for frac,name in zip([.001,.01,.05],['<0.1%','0.1–1%','1–5%']):pass
 # Category × defect-area mechanism summary (fixed bins from config).
 vr=pd.DataFrame([x for x in view_rows if x['category']==cat]);visible=vr[(vr.gt_view_label==1)&(vr.gt_visible_in_common_crop==1)]
 visible['size_bin']=pd.cut(visible.gt_area_fraction,bins=size_bins,labels=size_names,right=False)
 for model in ['D2','ArcAD']:
  for size_name,part in visible.groupby('size_bin',observed=False):
   if pd.isna(size_name):continue
   for key,threshold in [('budget1',view_threshold[model])]:
    flags=part[f'{model}_score_top1']>=threshold
    mechanism_rows=part
    stratum_rows.append({'category':cat,'model':model,'gt_size_bin':str(size_name),'positive_visible_views':len(part),'detected_at_test_descriptive_one_normal_object_fp':int(flags.sum()),
     'recall_descriptive':float(flags.mean()) if len(part) else np.nan,'median_gt_area_fraction':float(part.gt_area_fraction.median()) if len(part) else np.nan,
     'median_gt_component_count':float(part.gt_region_count.median()) if len(part) else np.nan,'median_min_component_width_px':float(part.gt_min_component_width_px.median()) if len(part) else np.nan,
     'median_Q':float(part[f'{model}_Q'].median()) if len(part) else np.nan,'median_C_model_score_contrast':float(part[f'{model}_C'].median()) if len(part) else np.nan,
     'median_rgb_luminance_contrast':float(part.rgb_luminance_contrast_gt_minus_ring.median()) if len(part) and 'rgb_luminance_contrast_gt_minus_ring' in part else np.nan,
     'median_rgb_gradient_contrast':float(part.rgb_gradient_contrast_gt_minus_ring.median()) if len(part) and 'rgb_gradient_contrast_gt_minus_ring' in part else np.nan,
     'median_rgb_background_texture_std':float(part.rgb_background_texture_std_ring.median()) if len(part) and 'rgb_background_texture_std_ring' in part else np.nan,
     'threshold_status':'test-derived descriptive only'})
 # Per category threshold-free tables.
 metrics.append(category_rec)
 for m in ['D2','ArcAD','Mean50']:
  mm=cls(y,view_scores[m][.01] if m!='Mean50' else view_scores[m]);oo=cls(oy,obj_scores[m])
  for level,ret,positive,prev in [('view',mm,pview,prev),('object',oo,pobj,oprev)]:
   for metric,value in ret.items():
    curve_type='roc' if metric=='AUROC' else 'pr'
    if m!='Mean50' or level in ['view','object']:
     if metric=='AUROC':fx,ty,_=roc_curve(y if level=='view' else oy,view_scores[m][.01] if level=='view' and m!='Mean50' else view_scores[m] if level=='view' else obj_scores[m])
     else:ty,fx,_=precision_recall_curve(y if level=='view' else oy,view_scores[m][.01] if level=='view' and m!='Mean50' else view_scores[m] if level=='view' else obj_scores[m])
     for xx,yyv in zip(fx,ty):curves.append({'category':cat,'model':m,'level':level,'curve':curve_type,'x':float(xx),'y':float(yyv),'positive_prevalence_baseline':prev})

 # Region overlap by empirical test-normal pixel FPR budgets; GT regions are identical.
 normal_idx=np.where(y==0)[0];positive_idx=[i for i in range(len(rs)) if y[i]==1 and g[i].any()]
 thresholds={}
 for model,mp in [('D2',d),('ArcAD',a)]:
  normal_pixels=mp[normal_idx].ravel()
  for budget in [.01,.05,.10]:
   th=float(np.quantile(normal_pixels,1-budget,method='higher'));actual=float(np.mean(normal_pixels>=th));thresholds[(model,budget)]=(th,actual)
 for budget in [.01,.05,.10]:
  preds={m:(d>=thresholds[('D2',budget)][0],a>=thresholds[('ArcAD',budget)][0]) for m in []}
  pred_d=d>=thresholds[('D2',budget)][0];pred_a=a>=thresholds[('ArcAD',budget)][0]
  records_by_model={'D2':[],'ArcAD':[],'OR_descriptive':[]}
  for i in positive_idx:
   for name,pred in [('D2',pred_d[i]),('ArcAD',pred_a[i]),('OR_descriptive',pred_d[i]|pred_a[i])]:records_by_model[name].append(component_summary(g[i],pred))
  for name,recs in records_by_model.items():
   total=sum(x['gt_regions'] for x in recs)
   row={'category':cat,'map_pixel_fpr_budget':budget,'model_or_rule':name,'positive_views_with_visible_gt':len(positive_idx),'gt_components':total,
    'mean_region_recall_iou_0_1':float(np.average([x['recall_iou_0_1'] for x in recs],weights=[x['gt_regions'] for x in recs])) if total else np.nan,
    'mean_region_recall_iou_0_3':float(np.average([x['recall_iou_0_3'] for x in recs],weights=[x['gt_regions'] for x in recs])) if total else np.nan,
    'mean_region_recall_iou_0_5':float(np.average([x['recall_iou_0_5'] for x in recs],weights=[x['gt_regions'] for x in recs])) if total else np.nan,
    'mean_best_iou':float(np.average([x['mean_best_iou'] for x in recs],weights=[x['gt_regions'] for x in recs])) if total else np.nan,
    'mean_gt_coverage':float(np.average([x['mean_coverage'] for x in recs],weights=[x['gt_regions'] for x in recs])) if total else np.nan,
    'D2_native_pixel_threshold':thresholds[('D2',budget)][0],'D2_realized_normal_pixel_fpr':thresholds[('D2',budget)][1],
    'ArcAD_native_pixel_threshold':thresholds[('ArcAD',budget)][0],'ArcAD_realized_normal_pixel_fpr':thresholds[('ArcAD',budget)][1],
    'OR_normal_pixel_union_fpr':float(np.mean((pred_d[normal_idx]|pred_a[normal_idx]))),'threshold_source':'test normal pixels; descriptive only; no deployable calibration'}
   region_rows.append(row)
 # paired ROC/PR data are shared for macro plotting; preserve objects for bootstrap.

# Global category-macro and pooled micro metrics, plus fusion deltas.
metric_df=pd.DataFrame(metrics);aggregate=[]
for m in MODELS:
 for level in ['I','O']:
  for metric in ['AUROC','AP']:
   col=f'{m}_{level}_{metric}';aggregate.append({'model':m,'level':level,'metric':metric,'aggregation':'category_macro_equal_weight','value':float(metric_df[col].mean())})
for model,viewcol,objcol in [('D2','D2_score_top1','D2_object_top1'),('ArcAD','ArcAD_score_top1','ArcAD_object_top1'),('Mean50','Mean50_score_z','Mean50_object_score_z')]:
 allv=pd.DataFrame(view_rows);allo=pd.DataFrame(object_rows)
 for level,df,scorecol,labelcol in [('I',allv,viewcol,'gt_view_label'),('O',allo,objcol,'gt_object_label')]:
  for metric,value in cls(df[labelcol].to_numpy(int),df[scorecol].to_numpy(float)).items():aggregate.append({'model':model,'level':level,'metric':metric,'aggregation':'pooled_micro','value':value})

# Paired object-cluster bootstrap, shared object IDs per replicate and all five views retained.
rng=np.random.default_rng(SEED);boot=[];invalid=Counter()
for cat in CATS:
 data=all_category_data[cat];objs=data['object_list'];viewgroups=data['object_indices'];omap={r['object_key']:r for r in object_rows if r['category']==cat}
 for rep in range(BOOT):
  draw=rng.choice(objs,size=len(objs),replace=True);cnt=Counter(draw);vw=[];ow=[]
  for o,n in cnt.items():
   vw.extend(viewgroups[o].tolist());ow.append(o)
  if not len(np.unique(data['labels'][vw]))>1 or not len(np.unique([int(omap[o]['gt_object_label']) for o in ow]))>1:invalid[cat]+=1;continue
  b={'category':cat,'replicate':rep,'objects_drawn':len(draw)}
  for level,ids,yy,scoremap in [('view',np.asarray(vw,int),data['labels'][vw],data['view_scores']),('object',ow,np.asarray([int(omap[o]['gt_object_label']) for o in ow]),data['object_scores'])]:
   wt=np.asarray([cnt[o] for o in data['objects'][ids]],float) if level=='view' else np.asarray([cnt[o] for o in ids],float)
   # Do not expand duplicated objects: use physical-object multiplicity as sample weights.
   for model in MODELS:
    if level=='view':ss=scoremap[model][.01] if model in ['D2','ArcAD'] else scoremap['Mean50'];ss=ss[ids]
    else:ss=scoremap[model][[data['object_list'].index(o) for o in ids]]
    try:
     au=float(roc_auc_score(yy,ss,sample_weight=wt));ap=float(average_precision_score(yy,ss,sample_weight=wt))
    except ValueError:au=ap=float('nan')
    b[f'{model}_{level}_AUROC']=au;b[f'{model}_{level}_AP']=ap
  for level in ['view','object']:
   for metric in ['AUROC','AP']:
    b[f'delta_D2_minus_ArcAD_{level}_{metric}']=b[f'D2_{level}_{metric}']-b[f'ArcAD_{level}_{metric}']
    b[f'delta_Mean50_minus_best_single_{level}_{metric}']=b[f'Mean50_{level}_{metric}']-max(b[f'D2_{level}_{metric}'],b[f'ArcAD_{level}_{metric}'])
  boot.append(b)
ci=[]
for cat in CATS:
 for metric in ['delta_D2_minus_ArcAD_view_AUROC','delta_D2_minus_ArcAD_view_AP','delta_D2_minus_ArcAD_object_AUROC','delta_D2_minus_ArcAD_object_AP',
               'delta_Mean50_minus_best_single_view_AUROC','delta_Mean50_minus_best_single_view_AP','delta_Mean50_minus_best_single_object_AUROC','delta_Mean50_minus_best_single_object_AP']:
  vals=np.asarray([r[metric] for r in boot if r['category']==cat and np.isfinite(r[metric])]);ci.append({'category':cat,'metric':metric,'valid_replicates':len(vals),'invalid_replicates':invalid[cat],
   'ci95_low':float(np.quantile(vals,.025)) if len(vals) else np.nan,'ci95_high':float(np.quantile(vals,.975)) if len(vals) else np.nan,'seed':SEED,'cluster_unit':'physical object; all camera views and masks retained'})

# Per-view/object metrics and calibration reliability artifacts.
writecsv(OUT/'threshold_free_metrics.csv',metrics);writecsv(OUT/'threshold_sensitivity.csv',sens);writecsv(OUT/'score_distributions.csv',score_dists)
writecsv(OUT/'view_diagnostics.csv',view_rows);writecsv(OUT/'object_camera_consistency.csv',camera_rows);writecsv(OUT/'mechanism_by_stratum.csv',stratum_rows)
writecsv(OUT/'operating_point_metrics.csv',op_rows);writecsv(OUT/'failure_overlap_vs_fpr.csv',failure);writecsv(OUT/'region_overlap_vs_fpr.csv',region_rows)
writecsv(OUT/'roc_pr_curves.csv',curves);writecsv(OUT/'bootstrap_intervals.csv',ci);writecsv(OUT/'bootstrap_replicates.csv',boot);writecsv(OUT/'aggregate_metrics.csv',aggregate)
writecsv(OUT/'seen_unseen_summary.csv',pd.read_csv(T2/'tables/seen_unseen_summary.csv').to_dict('records'))
label_sens=[]
for cat,data in all_category_data.items():
 y=data['labels'];visible=np.asarray([m.any() for m in data['gt']]);amb=(y==1)&~visible;oy=data['object_labels']
 for m in ['D2','ArcAD']:
  s=data['view_scores'][m][.01];orig=cls(y,s);keep=~amb;clean=cls(y[keep],s[keep])
  obj=cls(oy,data['object_scores'][m])
  label_sens.append({'category':cat,'model':m,'metric_level':'view','label_policy':'all original view labels','n_views':len(y),'excluded_ambiguous_positive_views':0,'positive_views':int(y.sum()),'AUROC':orig['AUROC'],'AP':orig['AP']})
  label_sens.append({'category':cat,'model':m,'metric_level':'view','label_policy':'exclude positive views with empty common-crop mask for sensitivity only','n_views':int(keep.sum()),'excluded_ambiguous_positive_views':int(amb.sum()),'positive_views':int(y[keep].sum()),'AUROC':clean['AUROC'],'AP':clean['AP']})
  label_sens.append({'category':cat,'model':m,'metric_level':'object','label_policy':'original object label retained as any positive view','n_views':len(oy),'excluded_ambiguous_positive_views':0,'positive_views':int(oy.sum()),'AUROC':obj['AUROC'],'AP':obj['AP']})
writecsv(OUT/'label_sensitivity.csv',label_sens)

# No operating threshold is claimed as calibrated. This audit records resolution and provenance.
threshold_audit=[]
calmanifest=json.loads((T2/'calibration_manifest.json').read_text())
for cat in CATS:
 data=all_category_data[cat];nnormal=int(sum(data['object_labels']==0))
 for model in ['D2','ArcAD']:
  cm=calmanifest['categories'][cat][f'{model}_train_normals']
  for level in ['view','object']:
   threshold_audit.append({'category':cat,'model':model,'level':level,'independent_normal_calibration_objects':0,'in_sample_training_normal_objects':cm['objects'],'in_sample_training_normal_views':cm['views'],
    'heldout_test_normal_objects_for_descriptive_curve':nnormal,'test_object_fpr_resolution':1/max(1,nnormal),'distinct_1_5_10_percent_calibration_supported':False,
    'deployment_threshold_claim':'none','descriptive_threshold_curve_source':'held-out test normals; descriptive only'})
writecsv(OUT/'threshold_audit.csv',threshold_audit)
caltext=['# Calibration reliability audit','',
 'No independent calibration split exists in the current workflow. Task 02’s ArcAD and D2 normal score references are training normals already seen by their models. ArcAD has 7–9 normal objects/category and D2 has 25; these are in-sample and cannot support deployment calibration claims.',
 '', 'The current Task 02 held-out test manifest has 25 normal objects/category, not 10. Its empirical object-FPR resolution is 1/25 = 4 percentage points. It cannot resolve distinct 1%, 5%, and 10% points. Curves in `failure_overlap_vs_fpr.csv` and `region_overlap_vs_fpr.csv` are test-derived descriptive sweeps, not selected operating points. Exact normal-object/pixel FP counts and realized rates are included.',
 '', 'The fixed 50/50 fusion uses global normal-score z-scores from the in-sample training references, solely as an exploratory unsupervised baseline. No category-specific thresholds or fusion weights were fit, and no independent anomaly validation objects exist. A held-out, object-disjoint normal calibration set and anomaly-labeled validation set are needed for deployment-like operating points.']
(OUT/'calibration_reliability.md').write_text('\n'.join(caltext)+'\n')

# Independent spot-check montage on the 224 common grid.
rng=np.random.default_rng(SEED);fig,axs=plt.subplots(len(CATS),4,figsize=(12,3*len(CATS)))
for ri,cat in enumerate(CATS):
 data=all_category_data[cat];opts=[i for i,r in enumerate(data['records']) if r['label'] and data['gt'][i].any()]
 i=int(rng.choice(opts));im=np.asarray(display_crop(data['records'][i]['image_abs']));g0=data['gt'][i]
 axs[ri,0].imshow(im);axs[ri,1].imshow(g0,cmap='gray');axs[ri,2].imshow(data['d2'][i],cmap='magma');axs[ri,3].imshow(data['arcad'][i],cmap='magma')
 axs[ri,0].set_ylabel(f"{cat}\n{data['records'][i]['object_key']} C{data['records'][i]['camera_view_id']}")
 for ax in axs[ri]:ax.axis('off')
for ax,title in zip(axs[0],['Original common crop','Source GT 224','D2 native map [16:240]','ArcAD map 256→224']):ax.set_title(title)
plt.tight_layout();plt.savefig(OUT/'plots/spotcheck_224.png',dpi=150);plt.close()

# ROC/PR and measured-FPR diagnostic plots.
fig,axs=plt.subplots(1,2,figsize=(12,5));cols=dict(zip(CATS,plt.cm.tab10(np.linspace(0,1,len(CATS)))))
for cat in CATS:
 for model,ls in [('D2','-'),('ArcAD','--')]:
  q=[r for r in curves if r['category']==cat and r['model']==model and r['level']=='view' and r['curve']=='roc']
  axs[0].plot([r['x'] for r in q],[r['y'] for r in q],color=cols[cat],linestyle=ls,label=f'{cat} {model}')
  q=[r for r in curves if r['category']==cat and r['model']==model and r['level']=='view' and r['curve']=='pr']
  axs[1].plot([r['x'] for r in q],[r['y'] for r in q],color=cols[cat],linestyle=ls,label=f'{cat} {model}')
axs[0].set(xlabel='False positive rate',ylabel='True positive rate',title='View ROC; solid D2, dashed ArcAD');axs[1].set(xlabel='Recall',ylabel='Precision',title='View precision-recall')
for ax in axs:ax.grid(alpha=.2);ax.legend(fontsize=6,ncol=2)
fig.tight_layout();fig.savefig(OUT/'plots/roc_pr_curves.png',dpi=160);plt.close(fig)
fdf=pd.DataFrame(failure);q=fdf[(fdf.level=='object')&(fdf.normal_object_fp_budget>0)]
fig,ax=plt.subplots(figsize=(7,5))
for cat in CATS:
 c=q[q.category==cat];actual=c.OR_false_positive_normal_object_union/c.test_normal_objects
 ax.plot(actual,c.OR_positive_recall_descriptive,marker='o',label=cat)
ax.set(xlabel='Measured normal-object FP union rate (test-derived)',ylabel='Positive-object OR recall',title='Descriptive thresholded OR; not deployable calibration');ax.grid(alpha=.2);ax.legend(fontsize=7);fig.tight_layout();fig.savefig(OUT/'plots/recall_vs_measured_fpr.png',dpi=160);plt.close(fig)

# Fusion metric comparison and ranked object-level failure cases.
fusion=[];fcases=[]
for cat in CATS:
 data=all_category_data[cat];y=data['labels'];oy=data['object_labels'];
 for m in MODELS:
  vs=data['view_scores'][m][.01] if m in ['D2','ArcAD'] else data['view_scores']['Mean50'];oscore=data['object_scores'][m]
  vm=cls(y,vs);om=cls(oy,oscore)
  fusion.append({'category':cat,'model_or_fusion':m,'I_AUROC':vm['AUROC'],'I_AP':vm['AP'],'O_AUROC':om['AUROC'],'O_AP':om['AP'],'P_metrics':'NA; score-level fusion only','fusion_weight_D2':.5 if m=='Mean50' else np.nan,'fusion_weight_ArcAD':.5 if m=='Mean50' else np.nan})
 rows=[r for r in object_rows if r['category']==cat]
 rank_by={}
 fields={'D2':'D2_object_top1','ArcAD':'ArcAD_object_top1','Mean50':'Mean50_object_score_z'}
 for m,field in fields.items():rank_by[m]=rankdata(np.asarray([r[field] for r in rows]),method='average')/len(rows)
 for i,r in enumerate(rows):
  if not r['gt_object_label']:continue
  best=max(rank_by['D2'][i],rank_by['ArcAD'][i]);rk=rank_by['Mean50'][i]
  fcases.append({'category':cat,'object_key':r['object_key'],'gt_object_label':1,'Mean50_object_score_z':float(r['Mean50_object_score_z']),'Mean50_rank_percentile':float(rk),
    'D2_object_rank_percentile':float(rank_by['D2'][i]),'ArcAD_object_rank_percentile':float(rank_by['ArcAD'][i]),'best_single_rank_percentile':float(best),
    'Mean50_rank_drop_vs_best_single':float(best-rk),'diagnostic':'positive object with worst mean50 rank drop vs best single; descriptive, no model/weight tuning'})
writecsv(OUT/'fusion_comparison.csv',fusion)
mf=pd.DataFrame(fusion);fig,ax=plt.subplots(figsize=(9,4));x=np.arange(len(CATS));width=.24
for j,m in enumerate(['D2','ArcAD','Mean50']):ax.bar(x+(j-1)*width,mf[mf.model_or_fusion==m].set_index('category').loc[CATS].I_AUROC,width,label=m)
ax.set_xticks(x,CATS,rotation=25,ha='right');ax.set_ylabel('View I-AUROC');ax.set_title('Fixed score-level 50/50 mean; no test-fitted weight');ax.legend();fig.tight_layout();fig.savefig(OUT/'plots/fusion_iauROC.png',dpi=160);plt.close(fig)
fcases=sorted(fcases,key=lambda r:r['Mean50_rank_drop_vs_best_single'],reverse=True)
writecsv(OUT/'fusion_failure_cases.csv',fcases[:100])
selinfo={'status':'fixed_unsupervised_mean50_only','weights':{'D2':.5,'ArcAD':.5},'fusion_rule':'Global per-model z-score using mean/std of each model’s in-sample normal training-reference view/object scores aligned to 224 common crop; arithmetic mean of the two z-scores.',
 'normal_reference_parameters':norm_params,'normal_reference_counts':cal_counts,'independent_calibration_objects':0,'independent_anomaly_validation_objects':0,
 'weighted_mean_tuning':'skipped: no independent validation anomaly objects','test_labels_used_for_fusion_parameter_selection':False,
 'interpretation':'exploratory score-average baseline only; normal references were seen during training, the global calibrator may shift across categories, and no deployable calibration claim is made.'}
(OUT/'fusion_validation_selection.json').write_text(json.dumps(selinfo,indent=2)+'\n')

# Evidence-first paired examples at the descriptive one-normal-object FP point.
examples=[];per_threshold={}
for cat in CATS:
 data=all_category_data[cat];objlist=data['object_list'];oidx=data['object_indices'];oy=data['object_labels'];rs=data['records'];vsc=data['view_scores']
 nnormal=sum(oy==0)
 th={m:descriptive_threshold(np.asarray([vsc[m][.01][oidx[o]].max() for o in objlist if oy[objlist.index(o)]==0]),1) for m in ['D2','ArcAD']}
 flags={m:vsc[m][.01]>=th[m] for m in ['D2','ArcAD']}
 for i,r in enumerate(rs):
  g0=data['gt'][i];
  row={q['image_path']:q for q in rs}[r['image_path']]
  ev={'category':cat,'image_path':r['image_path'],'object_key':r['object_key'],'view_id':r['camera_view_id'],'label':int(r['label']),'defect_type':r.get('defect_type') or '',
   'gt_area_px':int(g0.sum()),'D2_score':float(vsc['D2'][.01][i]),'ArcAD_score':float(vsc['ArcAD'][.01][i]),'D2_threshold_descriptive':th['D2'],'ArcAD_threshold_descriptive':th['ArcAD'],
   'D2_Q':float(roc_auc_score(g0.ravel(),data['d2'][i].ravel())) if g0.any() else np.nan,'ArcAD_Q':float(roc_auc_score(g0.ravel(),data['arcad'][i].ravel())) if g0.any() else np.nan}
  if g0.any():
   for m,mp in [('D2',data['d2'][i]),('ArcAD',data['arcad'][i])]:
    k=max(1,int(.01*mp.size));ix=np.argpartition(mp.ravel(),-k)[-k:];ev[m+'_top1_precision']=float(g0.ravel()[ix].mean());ev[m+'_top1_recall']=float(g0.ravel()[ix].sum()/g0.sum())
  view_rows_i=next((q for q in view_rows if q['category']==cat and q['image_path']==r['image_path']),None)
  if view_rows_i:ev['rgb_luminance_contrast']=view_rows_i.get('rgb_luminance_contrast_gt_minus_ring',np.nan)
  examples.append(ev)
# selection with deterministic conditions, de-duplicated across types
chosen=[]
def choose(candidates,kind,rule,keyscore=None,descending=True):
 candidates=[x for x in candidates if all((x['category'],x['image_path'])!=(y['category'],y['image_path']) for y in chosen)]
 if not candidates:return
 candidates.sort(key=lambda x:(x.get(keyscore,0) if keyscore else 0),reverse=descending);q=dict(candidates[0]);q.update({'example_type':kind,'selection_rule':rule});chosen.append(q)
pos=[x for x in examples if x['label']==1]
choose([x for x in pos if x['D2_score']<x['D2_threshold_descriptive'] and x['ArcAD_score']<x['ArcAD_threshold_descriptive']],'shared_miss','both per-view scores below model thresholds derived from one highest-scoring held-out normal object; descriptive only','D2_score',True)
choose([x for x in pos if x['D2_score']>=x['D2_threshold_descriptive'] and x['ArcAD_score']<x['ArcAD_threshold_descriptive']],'D2_only_success','D2 flags, ArcAD does not at the descriptive one-normal-object FP point','D2_score',True)
choose([x for x in pos if x['ArcAD_score']>=x['ArcAD_threshold_descriptive'] and x['D2_score']<x['D2_threshold_descriptive']],'ArcAD_only_success','ArcAD flags, D2 does not at the descriptive one-normal-object FP point','ArcAD_score',True)
visible=[x for x in pos if x['gt_area_px']>0 and np.isfinite(x.get('D2_Q',np.nan)) and np.isfinite(x.get('ArcAD_Q',np.nan))]
for x in visible:x['_Q_gap']=abs(x['D2_Q']-x['ArcAD_Q'])
choose(visible,'localization_disagreement','positive visible view maximizing absolute difference in D2 vs ArcAD within-view pixel AUROC Q','_Q_gap')
for m in ['D2','ArcAD']:
 q=[x for x in visible if x['label']==1 and x[f'{m}_score']<x[f'{m}_threshold_descriptive'] and x.get(f'{m}_top1_recall',0)>.2]
 choose(q,'threshold_only_miss',f'{m} below descriptive score threshold while top-1% pixels recover >20% of visible GT','D2_score' if m=='D2' else 'ArcAD_score',True)
normal=[x for x in examples if x['label']==0 and (x['D2_score']>=x['D2_threshold_descriptive'] or x['ArcAD_score']>=x['ArcAD_threshold_descriptive'])]
choose(normal,'normal_false_positive','normal view flagged by at least one model at descriptive one-normal-object FP point','D2_score',True)

# Paired same-source figures with independent heatmap display normalization and actual raw scores.
figdir=OUT/'paired_examples';figdir.mkdir(exist_ok=True)
for n,e in enumerate(chosen,1):
 cat=e['category'];data=all_category_data[cat];i=next(i for i,r in enumerate(data['records']) if r['image_path']==e['image_path']);im=np.asarray(display_crop(data['records'][i]['image_abs']));g0=data['gt'][i].astype(bool);d=data['d2'][i];a=data['arcad'][i]
 panels=[im]
 ov=im.copy();ov[g0]=(.45*ov[g0]+.55*np.array([255,30,30])).astype(np.uint8);panels.append(ov)
 def heat(x):
  u=(x-x.min())/(x.max()-x.min()+1e-12);return plt.get_cmap('magma')(u)[...,:3]
 panels += [heat(d),heat(a)]
 def top_overlay(x,color):
  k=max(1,int(.01*x.size));mask=np.zeros(x.size,bool);mask[np.argpartition(x.ravel(),-k)[-k:]]=True;mask=mask.reshape(x.shape);o=im.copy();o[mask]=(.35*o[mask]+.65*np.asarray(color)).astype(np.uint8);o[g0]=(.5*o[g0]+.5*np.array([255,30,30])).astype(np.uint8);return o
 panels += [top_overlay(d,[0,120,255]),top_overlay(a,[255,150,0])]
 titles=['Common crop','GT overlay','D2 heat','ArcAD heat','D2 top 1% + GT','ArcAD top 1% + GT']
 fig,axs=plt.subplots(1,6,figsize=(18,4.3))
 for ax,p,t in zip(axs,panels,titles):ax.imshow(p);ax.set_title(t);ax.axis('off')
 fig.suptitle(f"{e['example_type']} | {cat} {e['object_key']} C{e['view_id']} {e['defect_type']} | GT={e['gt_area_px']} px\nD2 score={e['D2_score']:.5g}, descriptive threshold={e['D2_threshold_descriptive']:.5g}, Q={e['D2_Q']:.3f}; ArcAD score={e['ArcAD_score']:.5g}, threshold={e['ArcAD_threshold_descriptive']:.5g}, Q={e['ArcAD_Q']:.3f}\nHeat colors normalized independently for display; scores printed in native map scale. Test-normal thresholds are descriptive only.",fontsize=9)
 fig.tight_layout(rect=(0,0,1,.83));fn=f'{n:02d}_{e["example_type"]}.png';fig.savefig(figdir/fn,dpi=120);plt.close(fig);e['figure']=f'paired_examples/{fn}'
writecsv(OUT/'paired_example_index.csv',chosen)

# Report category 160 macro overlap and detailed Task 03 report with explicit limits.
screen=pd.read_csv(OUT/'category_160_audit.csv');weak=pd.read_csv(OUT/'weak_category_overlap.csv')
report=['# Task 03 verification, calibration, and fusion follow-up','',
 '## Verified observations','',
 f"The current ArcAD protoinit summary contains 160 categories (excluding its `Mean` row): macro I-AUROC {screen.ArcAD_I_AUROC_protoinit.mean():.4f}, {int((screen.ArcAD_I_AUROC_protoinit<.9).sum())} categories <0.90 and {int((screen.ArcAD_I_AUROC_protoinit<=.9).sum())} <=0.90. D2 has macro I-AUROC {screen.D2_I_AUROC.mean():.4f}, {int((screen.D2_I_AUROC<.9).sum())} categories <0.90 and {int((screen.D2_I_AUROC<=.9).sum())} <=0.90. Screening overlap is 75 weak-both, 8 D2-only, 19 ArcAD-only, and 58 neither. Native test lists differ, so this is a screening comparison, not paired inference. The older ArcAD summary cited in Task 03 was not opened because the current prototype-initialized run supersedes it under the standing instruction.",
 '', 'The current Task 02 held-out set has 25 normal physical objects/category (not 10), grouped by five views. The actual categories are `lattice_block_plug`, `laser_diode`, `power_jack`, `LED_indicator`, and `aircraft_model_head`; the Task 03 brief’s prior-run list substitutes `thyristor`. This run uses the current saved five-category maps and performs no new model inference.',
 '', f"Independent 224-grid verification passed for {sum(len(x['records']) for x in all_category_data.values())} maps. D2 maps use native crop [16:240]; ArcAD maps are bilinearly resized 256→224; GT masks use raw source crop [64:960] and nearest resize. All path orders, shapes, map ranges, and finite values passed. See `alignment_validation_224.json` and `plots/spotcheck_224.png`.",
 '', 'The 41 `laser_diode` positive views with empty Task 02 common GT resolve to 39 nonzero masks entirely outside [64:960] and 2 genuinely all-zero source masks (both Task 01 OilStain/S0005). None vanished only at 224 downsampling. Both native test manifests agree on all 41 view labels. The object OilStain/S0005 has two labeled anomalous and three normal views; object labels remain positive. All 41 cases require manual visual review.',
 '', '## Evidence for/against mechanisms','',
 'Threshold-free view/object ranking, pixel localization, component metrics, top-k sensitivity, GT size, RGB luminance/gradient contrast, background texture, per-camera scores, and seen/unseen defect-type associations are recorded separately. Model-derived score contrast C/Z/Q is not used as an independent cause of failure. The category split and size-bin boundaries were fixed before the diagnostic calculations. See `mechanism_by_stratum.csv` and `object_camera_consistency.csv`.',
 '', '## Unverified hypotheses','',
 'Small/dispersed defects, low RGB contrast, view-dependent visibility, and representation differences remain hypotheses. The analysis is observational and cannot isolate causal effects. The empty-mask contact sheets do not decide whether a physical defect is visible; manual annotation is required.',
 '', '## Practical ensemble result','',
 'A fixed 50/50 score-average was evaluated after global per-model z-standardization using normal training-reference score distributions. These normal examples were seen during model training, so the baseline is exploratory. No anomaly-labeled validation objects or independent normal calibration population were available; no category-specific weight or threshold was tuned. The mean fusion improves some ranking metrics and reduces others; per-category results and paired object-bootstrap intervals are in `fusion_comparison.csv` and `bootstrap_intervals.csv`. It is not a deployable calibration result.',
 '', 'Test-normal FPR sweeps and region pixel-FPR sweeps are descriptive only. The test set has 25 normal objects/category, so object FPR resolution is 4%; the curves do not claim distinct 1%, 5%, or 10% calibrated operating points. Thresholded OR and region union costs are tabulated with actual normal-object/pixel false-positive counts. Independent calibration is needed before deploying the OR rule.',
 '', '## Limitations','',
 'ArcAD uses the current protoinit checkpoint; the non-initialized ArcAD result folder is excluded. ArcAD and D2 full-run test manifests contain 192,165 and 178,995 views and differ in split composition. A single access_card object has unparseable camera IDs in both native manifests. Task 02 normal references are in-sample. Pixel AUC/AP use 2,048-bin histograms; P-AUPRO uses 200 thresholds with FPR cap 0.30. Pixel metrics are point estimates; bootstrap intervals cover view/object I/O AUROC/AP.',
 '', 'No 160-category extension inference was launched. Category selection remains a score-ranked plan, and claims do not generalize from the five diagnostic categories.',
 '', '## Recommended next experiment','',
 'Create a larger object-disjoint normal calibration set and anomaly-labeled validation set, ensuring all five cameras of an object remain together and excluding all ArcAD supervised training objects. Resolve the 41 laser diode mask visibility cases by expert review. Then evaluate a pre-registered calibration and fusion rule on untouched shared test objects, and expand to the preselected four extension categories only if compute and storage are approved.',
 '', 'Reproduce with `bash reproduce.sh`; no training or model inference is performed by the analysis scripts. Checkpoint SHA256 values and source split details are in `protocol_audit.json`.']
(OUT/'analysis_report.md').write_text('\n'.join(report)+'\n')
print(json.dumps({'categories':len(CATS),'views':sum(len(x['records']) for x in all_category_data.values()),'examples':len(chosen),'bootstrap_replicates':len(boot),'invalid_bootstrap_by_category':dict(invalid),'fusion_rows':len(fusion),'region_rows':len(region_rows)},indent=2))
