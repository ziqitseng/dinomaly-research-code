#!/usr/bin/env python3
"""Align cached D2 and new ArcAD maps, calculate metrics and create review outputs."""
import csv,json,hashlib,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
from PIL import Image
from scipy import ndimage
from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve,precision_recall_curve
ROOT=Path(__file__).resolve().parents[2]; OUT=ROOT/'analysis_task04A_crop_ablation_20261102_01'
DATA=Path('/media/avlab/7458a474-6ea2-4b90-b232-373c3df8d329/real_iad_variety_1024')
TASK2=ROOT/'analysis_shared_failure_20261031_gpu01'; TASK1=ROOT/'Dinomaly2/out/re_eval_shared_20261031_cpu01'
CATS=('laser_diode','lattice_block_plug','power_jack','LED_indicator','thyristor')
BRANCHES=('arcad_original','arcad_full_frame','dinomaly2_original')
def readcsv(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
def writecsv(p,rows):
 rows=list(rows)
 if not rows:return
 with open(p,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def metric(y,s):
 y=np.asarray(y).astype(np.uint8);s=np.asarray(s,dtype=np.float64)
 if len(np.unique(y))<2:return np.nan,np.nan
 return float(roc_auc_score(y,s)),float(average_precision_score(y,s))
def pro_fast(masks,amaps,num_th=200):
 """Vectorized equivalent of ArcAD utils.compute_pro (same 200 thresholds/FPR cap)."""
 from skimage.measure import label as cc_label
 masks=np.asarray(masks)>0;amaps=np.asarray(amaps,dtype=np.float32)
 lo=float(amaps.min());hi=float(amaps.max())
 if hi-lo<=1e-8:return 0.0
 thresholds=np.linspace(lo,hi,num_th,dtype=np.float64)
 neg=np.sort(amaps[~masks].astype(np.float64))
 fpr=(len(neg)-np.searchsorted(neg,thresholds,side='right'))/max(1,len(neg))
 pro_sum=np.zeros(num_th,dtype=np.float64);nregions=0
 for mask,amap in zip(masks,amaps):
  if not mask.any():continue
  labels=cc_label(mask,connectivity=2)
  for rid in range(1,int(labels.max())+1):
   vals=np.sort(amap[labels==rid].astype(np.float64))
   pro_sum += (len(vals)-np.searchsorted(vals,thresholds,side='right'))/len(vals)
   nregions+=1
 pros=pro_sum/max(1,nregions);order=np.argsort(fpr);fpr=fpr[order];pros=pros[order]
 keep=fpr<=.3;fpr=fpr[keep];pros=pros[keep]
 if len(fpr)>1 and fpr.max()>0:return float(np.trapz(pros,fpr/.3))
 return 0.0
def top_score(a,pct=1.0):
 x=np.asarray(a).reshape(-1);n=max(1,int(len(x)*pct/100));return float(np.partition(x,len(x)-n)[-n:].mean())
def mapload(cat,branch):
 if branch.startswith('arcad_'):
  z=np.load(OUT/'maps'/f'{branch}_{cat}.npz');return z['smoothed_maps'].astype(np.float32),z['image_paths'].astype(str).tolist()
 f=(TASK1/'maps'/f'{cat}.npz') if cat in ('laser_diode','lattice_block_plug','power_jack') else TASK2/'maps'/f'd2_{cat}.npz'
 z=np.load(f);k='postprocessed_maps' if 'postprocessed_maps' in z.files else 'smoothed_maps';return z[k].astype(np.float32),z['image_paths'].astype(str).tolist()
def getpath(cat,rel):
 p=DATA/cat/rel
 if not p.is_file():p=DATA/cat/cat/rel
 return p
def getmask(cat,row):
 if not row.get('mask_path') or not int(row['label']):return np.zeros((1024,1024),np.uint8)
 p=getpath(cat,row['mask_path'])
 with Image.open(p) as im: a=np.asarray(im.convert('L'))
 if a.shape!=(1024,1024):raise ValueError(f'bad mask geometry {p} {a.shape}')
 return (a>0).astype(np.uint8)
def ci_boot(groups, branch_scores, ref_scores, seed=20261102, nrep=2000):
 rng=np.random.default_rng(seed); keys=sorted(groups); out=[]; invalid=0
 for b in range(nrep):
  sel=rng.integers(0,len(keys),len(keys)); ys=[];bs_i=[];rs_i=[];bs_o=[];rs_o=[]
  for ix in sel:
   k=keys[ix]; rows=groups[k]; ys.extend([r['label'] for r in rows]);bs_i.extend([branch_scores[r['image_path']] for r in rows]);rs_i.extend([ref_scores[r['image_path']] for r in rows]);
   # Object metrics are scores already pooled across all five views.
   bs_o.append(branch_scores['object::'+k]);rs_o.append(ref_scores['object::'+k])
  ai,ap=metric(ys,bs_i);ri,rp=metric(ys,rs_i)
  oy=[max(int(r['label']) for r in groups[k]) for k in [keys[ix] for ix in sel]]
  ao,_=metric(oy,bs_o);ro,_=metric(oy,rs_o)
  if not np.isfinite(ai+ri+ao+ro):invalid+=1;continue
  out.append((ai-ri,ao-ro))
 a=np.asarray(out)
 return {'valid_replicates':len(out),'invalid_single_class_replicates':invalid,'n_replicates':nrep,'view_delta_mean':float(np.mean(a[:,0])),'view_delta_ci95_low':float(np.quantile(a[:,0],.025)),'view_delta_ci95_high':float(np.quantile(a[:,0],.975)),'object_delta_mean':float(np.mean(a[:,1])),'object_delta_ci95_low':float(np.quantile(a[:,1],.025)),'object_delta_ci95_high':float(np.quantile(a[:,1],.975))}
def main():
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 sys.path.insert(0,str(ROOT/'ArcAD'))
 m=json.loads((ROOT/'analysis_shared_failure_20261031_gpu01/shared_manifest.json').read_text())
 out_metrics=[];view_rows=[];object_rows=[];bootstrap=[];border=[];region_rows=[];regression=[];dist_rows=[];map_index=[]
 (OUT/'paired_examples').mkdir(exist_ok=True)
 for cat in CATS:
  rows=m['categories'][cat]['shared_test_images']; paths=[r['image_path'] for r in rows]
  bypath={r['image_path']:r for r in rows}
  branch_source={};branch_paths={}
  for b in BRANCHES:
   arr,ps=mapload(cat,b)
   if ps!=paths:raise ValueError(f'{cat}/{b}: saved map path sequence mismatch')
   branch_source[b]=arr;branch_paths[b]=ps
  # Source masks are read once, then projected to the prescribed grids with nearest-neighbour sampling.
  full_gt=[];center_gt=[];maskstats={};region_component_cache={}
  for r in rows:
   src=getmask(cat,r); total=int(src.sum()); crop=src[64:960,64:960]; inside=int(crop.sum())
   src_image=Image.fromarray((src*255).astype(np.uint8))
   gfull=(np.asarray(src_image.resize((256,256),Image.Resampling.NEAREST))>0).astype(np.uint8)
   gc=(np.asarray(src_image.crop((64,64,960,960)).resize((224,224),Image.Resampling.NEAREST))>0).astype(np.uint8)
   full_gt.append(gfull);center_gt.append(gc)
   status='source_all_zero' if total==0 else ('wholly_outside_original_crop' if inside==0 else ('partly_outside_original_crop' if inside<total else 'inside_original_crop'))
   maskstats[r['image_path']]={'source_area':total,'inside_area':inside,'outside_area':total-inside,'status':status,'center_area_224':int(gc.sum()),'full_area_256':int(gfull.sum())}
   comps,n=ndimage.label(src)
   region_component_cache[r['image_path']]=(comps,n)
  full_gt=np.asarray(full_gt,dtype=np.uint8);center_gt=np.asarray(center_gt,dtype=np.uint8)
  task2_gt=np.load(TASK2/'maps'/f'common_d2_{cat}.npz')
  if task2_gt['image_paths'].astype(str).tolist()!=paths or not np.array_equal(center_gt,task2_gt['gt_masks'].astype(np.uint8)):
   raise ValueError(f'{cat}: source-to-common-grid GT failed exact Task02 alignment check')
  # Source-grid projections: original ArcAD covers only the center crop; full ArcAD/D2 cover all source pixels.
  center_maps={}
  for b,a in branch_source.items():
   if b=='arcad_original': center_maps[b]=np.stack([cv2.resize(x,(224,224),interpolation=cv2.INTER_LINEAR) for x in a]).astype(np.float32)
   else:center_maps[b]=a[:,16:240,16:240].astype(np.float32)
  # Regression against Task 02's original ArcAD maps, before display normalization.
  prior=np.load(TASK2/'maps'/f'arcad_{cat}.npz');prior_paths=prior['image_paths'].astype(str).tolist()
  if prior_paths!=paths:raise ValueError(f'Task02 ArcAD path mismatch for {cat}')
  diff=np.abs(branch_source['arcad_original']-prior['smoothed_maps'].astype(np.float32))
  regression.append({'category':cat,'views':len(paths),'map_max_abs_diff':float(diff.max()),'map_mean_abs_diff':float(diff.mean()),'score_max_abs_diff':float(max(abs(top_score(x)-top_score(y)) for x,y in zip(branch_source['arcad_original'],prior['smoothed_maps']))),'tolerance_max_abs':1e-4,'pass':bool(diff.max()<=1e-4)})
  # Per-view and object-level anomaly scores keep top-1% aggregation unchanged.
  groups=defaultdict(list);scores={b:{} for b in BRANCHES}
  for i,r in enumerate(rows):
   groups[r['object_key']].append(r)
   for b in BRANCHES:scores[b][r['image_path']]=top_score(branch_source[b][i])
  for key,rr in sorted(groups.items()):
   idx=[paths.index(x['image_path']) for x in rr]
   for b in BRANCHES:scores[b]['object::'+key]=top_score(branch_source[b][idx])
  ys=np.asarray([int(r['label']) for r in rows],np.uint8)
  okeys=sorted(groups);oy=np.asarray([max(int(r['label']) for r in groups[k]) for k in okeys],np.uint8)
  branch_obj_score={b:np.asarray([scores[b]['object::'+k] for k in okeys]) for b in BRANCHES}
  center_labels=center_gt.reshape(-1)
  # Full-source masks for full-frame ArcAD and full-frame D2 only.
  full_labels=full_gt.reshape(-1)
  # Exact view/object threshold-free scores.
  for b in BRANCHES:
   vi=np.asarray([scores[b][r['image_path']] for r in rows]); oscores=branch_obj_score[b]
   ia,ip=metric(ys,vi);oa,op=metric(oy,oscores)
   cmap=center_maps[b]; flat_y=center_labels;flat_s=cmap.reshape(-1)
   pa,pp=metric(flat_y,flat_s)
   try:pro=pro_fast(center_gt,cmap)
   except Exception as e:pro=float('nan'); print('P-AUPRO failed',cat,b,repr(e),flush=True)
   row={'category':cat,'branch':b,'views':len(rows),'objects':len(groups),'positive_views':int(ys.sum()),'normal_views':int((ys==0).sum()),'positive_objects':int(oy.sum()),'normal_objects':int((oy==0).sum()),'center_I_AUROC':ia,'center_I_AP':ip,'center_O_AUROC':oa,'center_O_AP':op,'center_P_AUROC':pa,'center_P_AP':pp,'center_P_AUPRO':pro,'center_pixels':int(center_gt.size),'center_positive_pixel_prevalence':float(center_gt.mean()),'full_field_P_AUROC':np.nan,'full_field_P_AP':np.nan,'full_field_P_AUPRO':np.nan,'full_field_tag':'not_applicable_unobserved_border' if b=='arcad_original' else 'full_source_256'}
   if b!='arcad_original':
    ma,mp=metric(full_labels,branch_source[b].reshape(-1));row.update(full_field_P_AUROC=ma,full_field_P_AP=mp)
    try:row['full_field_P_AUPRO']=pro_fast(full_gt,branch_source[b])
    except Exception as e:print('full P-AUPRO failed',cat,b,repr(e),flush=True)
   out_metrics.append(row)
   for kind,y,sc in [('view',ys,vi),('object',oy,oscores)]:
    fpr,tpr,_=roc_curve(y,sc);prec,rec,_=precision_recall_curve(y,sc)
    plt.figure(figsize=(5.4,4.5))
    if kind=='view':plt.plot(fpr,tpr);plt.plot([0,1],[0,1],'k--',lw=.7);plt.xlabel('False positive rate');plt.ylabel('True positive rate')
    else:plt.plot(rec,prec);plt.xlabel('Recall');plt.ylabel('Precision')
    plt.title(f'{cat} — {b} {kind}\nAUROC={metric(y,sc)[0]:.4f}, AP={metric(y,sc)[1]:.4f}')
    plt.tight_layout();plt.savefig(OUT/'plots'/f'{cat}_{b}_{kind}_curve.png',dpi=140);plt.close()
   # Score distributions and paired bootstrap preparation.
   for cls,nm in [(0,'normal'),(1,'positive')]:
    vals=vi[ys==cls]
    dist_rows.append({'category':cat,'branch':b,'level':'view','class':nm,'n':len(vals),'mean':float(vals.mean()),'std':float(vals.std()),'median':float(np.median(vals))})
    vals= oscores[oy==cls]
    dist_rows.append({'category':cat,'branch':b,'level':'object','class':nm,'n':len(vals),'mean':float(vals.mean()),'std':float(vals.std()),'median':float(np.median(vals))})
   # Paired object-cluster bootstrap: original compared with full frame and D2 reference.
   if b=='arcad_full_frame':
    bootstrap.append({'category':cat,'comparison':'arcad_full_frame_minus_arcad_original',**ci_boot(groups,scores['arcad_full_frame'],scores['arcad_original'])})
   elif b=='arcad_original':
    pass
  bootstrap.append({'category':cat,'comparison':'arcad_full_frame_minus_dinomaly2_reference',**ci_boot(groups,scores['arcad_full_frame'],scores['dinomaly2_original'],seed=20261103)})
  # Save source-aligned maps: original branch has NaN outside its observed source crop.
  aligned_orig=np.full_like(branch_source['arcad_original'],np.nan,dtype=np.float32)
  aligned_orig[:,16:240,16:240]=np.stack([cv2.resize(x,(224,224),interpolation=cv2.INTER_LINEAR) for x in branch_source['arcad_original']])
  np.savez_compressed(OUT/'maps'/f'aligned_{cat}.npz',arcad_original_source256=aligned_orig,arcad_full_frame_source256=branch_source['arcad_full_frame'].astype(np.float32),dinomaly2_source256=branch_source['dinomaly2_original'].astype(np.float32),gt_source256=full_gt,gt_center224=center_gt,image_paths=np.asarray(paths),original_arcad_observed=np.isfinite(aligned_orig),coordinate_note=np.asarray('source grid 256x256; ArcAD original observed only crop [16:240] on this grid; source crop [64:960]') )
  map_index.append({'category':cat,'file':f'maps/aligned_{cat}.npz','coordinate_frame':'source-aligned full 256 plus center 224 GT','arcad_original_outside_crop':'NaN/not observed','d2_original_source':'reused existing numeric maps; provenance in protocol_audit.json'})
  # Per-view output and mask visibility strata; per-object rows pool top 1% across five views.
  for i,r in enumerate(rows):
   st=maskstats[r['image_path']]
   vr={'category':cat,'object_key':r['object_key'],'camera_id':next((f'C{i}' for i in range(1,6) if f'_C{i}_' in Path(r['image_path']).name),''),'image_path':r['image_path'],'view_label':int(r['label']),'object_label':int(max(int(x['label']) for x in groups[r['object_key']])),'defect_type':r.get('anomaly_class',''),'mask_status':st['status'],'source_mask_pixels':st['source_area'],'mask_pixels_inside_original_crop':st['inside_area'],'mask_pixels_outside_original_crop':st['outside_area'],'gt_center_224_pixels':st['center_area_224']}
   for b in BRANCHES:vr[b+'_view_top1_score']=scores[b][r['image_path']]
   view_rows.append(vr)
   if cat=='laser_diode' and int(r['label']) and st['source_area']>0 and st['inside_area']==0:
    border.append({'category':cat,'object_key':r['object_key'],'camera_id':vr['camera_id'],'image_path':r['image_path'],'mask_path':r['mask_path'],'source_mask_pixels':st['source_area'],'inside_crop_pixels':st['inside_area'],'outside_crop_pixels':st['outside_area'],'arcad_original_view_score':scores['arcad_original'][r['image_path']],'arcad_full_frame_view_score':scores['arcad_full_frame'][r['image_path']],'dinomaly2_original_view_score':scores['dinomaly2_original'][r['image_path']],'visibility_flag':'requires_manual_review','cannot_observe_gt_in_original_crop':True,'predicts_normal_claim':'not inferred from observability'})
  for k in okeys:
   rr=groups[k]; ob={'category':cat,'object_key':k,'object_label':int(max(int(r['label']) for r in rr)),'n_views':len(rr)}
   for b in BRANCHES:ob[b+'_pooled_top1_score']=scores[b]['object::'+k]
   object_rows.append(ob)
  # Top-k region localization rows on center field (three branches) and full field (full ArcAD/D2).
  for i,r in enumerate(rows):
   comps,nc=ndimage.label(center_gt[i])
   for rid in range(1,nc+1):
    gtreg=comps==rid; area=int(gtreg.sum())
    for b in BRANCHES:
     amap=center_maps[b][i]; flat=amap.ravel()
     for pct in (.1,.5,1.,2.,5.):
      k=max(1,int(flat.size*pct/100)); ix=np.argpartition(flat,-k)[-k:]; pred=np.zeros(flat.shape,bool);pred[ix]=True;pred=pred.reshape(amap.shape)
      inter=int((pred&gtreg).sum());union=int((pred|gtreg).sum());iou=inter/max(1,union);cov=inter/area
      region_rows.append({'category':cat,'object_key':r['object_key'],'camera_id':next((f'C{j}' for j in range(1,6) if f'_C{j}_' in Path(r['image_path']).name),''),'image_path':r['image_path'],'field':'center224','region_id':rid,'gt_area_pixels':area,'branch':b,'top_k_percent':pct,'best_region_iou':iou,'gt_component_coverage':cov,'detected_iou_0.1':iou>=.1,'detected_iou_0.3':iou>=.3,'detected_iou_0.5':iou>=.5})
    # Full-grid components evaluated only for full-frame branches; components empty after resampling are excluded.
   comps,nc=ndimage.label(full_gt[i])
   for rid in range(1,nc+1):
    gtreg=comps==rid;area=int(gtreg.sum())
    for b in ('arcad_full_frame','dinomaly2_original'):
     amap=branch_source[b][i];flat=amap.ravel()
     for pct in (.1,.5,1.,2.,5.):
      k=max(1,int(flat.size*pct/100));ix=np.argpartition(flat,-k)[-k:];pred=np.zeros(flat.shape,bool);pred[ix]=1;pred=pred.reshape(amap.shape)
      inter=int((pred&gtreg).sum());union=int((pred|gtreg).sum());iou=inter/max(1,union);cov=inter/area
      region_rows.append({'category':cat,'object_key':r['object_key'],'camera_id':next((f'C{j}' for j in range(1,6) if f'_C{j}_' in Path(r['image_path']).name),''),'image_path':r['image_path'],'field':'full_source256','region_id':rid,'gt_area_pixels':area,'branch':b,'top_k_percent':pct,'best_region_iou':iou,'gt_component_coverage':cov,'detected_iou_0.1':iou>=.1,'detected_iou_0.3':iou>=.3,'detected_iou_0.5':iou>=.5})
  print(cat,'metrics complete; center labels',int(center_gt.sum()),'border flags so far',len(border),flush=True)
  # One paired five-camera sheet for each laser-diode object with at least one border mask.
  border_objects=sorted({x['object_key'] for x in border if x['category']==cat})
  for key in border_objects:
   rr=groups[key];fig,axes=plt.subplots(5,5,figsize=(16,15),squeeze=False)
   for j,r in enumerate(sorted(rr,key=lambda x:x['image_path'])):
    ix=paths.index(r['image_path']);srcp=getpath(cat,r['image_path'])
    with Image.open(srcp) as im:src=np.asarray(im.convert('RGB'))
    axes[j,0].imshow(src);axes[j,0].axvline(64,color='yellow');axes[j,0].axvline(959,color='yellow');axes[j,0].set_title('Source '+next((f'C{i}' for i in range(1,6) if f'_C{i}_' in srcp.name),''))
    axes[j,1].imshow(getmask(cat,r),cmap='gray',vmin=0,vmax=1);axes[j,1].set_title('Source GT')
    axes[j,2].imshow(center_gt[ix],cmap='gray',vmin=0,vmax=1);axes[j,2].set_title('Center GT 224')
    axes[j,3].imshow(center_maps['arcad_original'][ix],cmap='magma');axes[j,3].set_title('ArcAD original center map')
    axes[j,4].imshow(branch_source['arcad_full_frame'][ix],cmap='magma');axes[j,4].set_title('ArcAD full frame map')
    for ax in axes[j]:ax.axis('off')
   fig.suptitle(f'laser_diode {key} — five views; each heatmap uses its own map scale. Visibility requires expert review.',fontsize=12)
   fig.tight_layout();fig.savefig(OUT/'border_39_contact_sheet'/f'{key.replace("/","_")}.png',dpi=120);plt.close(fig)
  # Deterministic central example sheets: highest original crop GT area and one smallest nonempty component.
  if cat=='laser_diode':
   candidates=[r for r in rows if maskstats[r['image_path']]['inside_area']>0]
   def rankarea(r):return maskstats[r['image_path']]['inside_area']
   selected=[]
   if candidates:selected=[max(candidates,key=rankarea),min(candidates,key=rankarea)]
   for rank,r in enumerate(selected,1):
    key=r['object_key'];rr=sorted(groups[key],key=lambda x:x['image_path']);fig,axes=plt.subplots(5,5,figsize=(16,15),squeeze=False)
    for j,rv in enumerate(rr):
     ix=paths.index(rv['image_path']);srcp=getpath(cat,rv['image_path'])
     with Image.open(srcp) as im:src=np.asarray(im.convert('RGB'))
     axes[j,0].imshow(src);axes[j,0].set_title('Source')
     axes[j,1].imshow(center_gt[ix],cmap='gray',vmin=0,vmax=1);axes[j,1].set_title('Center GT')
     axes[j,2].imshow(center_maps['arcad_original'][ix],cmap='magma');axes[j,2].set_title('ArcAD original')
     axes[j,3].imshow(center_maps['arcad_full_frame'][ix],cmap='magma');axes[j,3].set_title('ArcAD full frame')
     axes[j,4].imshow(center_maps['dinomaly2_original'][ix],cmap='magma');axes[j,4].set_title('D2 reference')
     for ax in axes[j]:ax.axis('off')
    fig.suptitle(f'laser_diode paired five-camera example {rank}: {key}; panel scales independent');fig.tight_layout();fig.savefig(OUT/'paired_examples'/f'laser_diode_paired_{rank}.png',dpi=120);plt.close(fig)
  # Per-category paired score distributions by label and field strata.
  if cat=='laser_diode':
   d=defaultdict(list)
   for r in rows:
    st=maskstats[r['image_path']]['status'];
    for b in BRANCHES:d[(b,st)].append(scores[b][r['image_path']])
   plt.figure(figsize=(11,5));labels=[];vals=[]
   for b in BRANCHES:
    for st in ('inside_original_crop','partly_outside_original_crop','wholly_outside_original_crop','source_all_zero'):
     if d[(b,st)]:labels.append(b.replace('_','\n')+'\n'+st.replace('_',' '));vals.append(d[(b,st)])
   plt.boxplot(vals,labels=labels,showfliers=False);plt.xticks(rotation=35,ha='right');plt.ylabel('Per-view top-1% score');plt.title('Laser diode score distributions by source-mask visibility');plt.tight_layout();plt.savefig(OUT/'plots'/'laser_diode_border_vs_center.png',dpi=150);plt.close()
  del branch_source,center_maps,full_gt,center_gt
 writecsv(OUT/'category_metrics.csv',out_metrics);writecsv(OUT/'per_view_scores.csv',view_rows);writecsv(OUT/'per_object_scores.csv',object_rows);writecsv(OUT/'paired_bootstrap_ci.csv',bootstrap);writecsv(OUT/'border_39_audit.csv',border);writecsv(OUT/'region_topk_metrics.csv',region_rows);writecsv(OUT/'score_distributions.csv',dist_rows);writecsv(OUT/'arcad_original_regression.csv',regression)
 (OUT/'map_index.json').write_text(json.dumps(map_index,indent=2)+'\n')
 (OUT/'manual_review_sheet.md').write_text('# Manual review sheet — crop-external laser_diode masks\n\n**Observed:** the audited rows have positive source GT area but zero source-mask pixels in `[64:960)×[64:960)`. Their original ArcAD input therefore excludes the annotated region. The contact sheets show all five source views, source masks, center masks and ArcAD maps.\n\n**Requires expert review:** whether the annotated defect is visible in each photograph, whether the mask location is correct, and whether the view should be considered an expected positive for the crop-limited model.\n\n**Not inferred:** a crop-external GT region does not establish that ArcAD predicted normal; no visibility or causal label was fabricated. See `border_39_audit.csv`.\n')
 print('Wrote outputs:',len(out_metrics),'category rows,',len(view_rows),'views,',len(object_rows),'objects,',len(bootstrap),'bootstrap comparisons,',len(border),'border rows',flush=True)

if __name__=='__main__':main()

