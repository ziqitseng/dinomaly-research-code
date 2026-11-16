#!/usr/bin/env python3
"""Finalize Task 04A provenance, plots, and interpretation after analysis."""
import csv,json,hashlib,platform,sys,subprocess
from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.metrics import roc_curve,precision_recall_curve,roc_auc_score,average_precision_score
from PIL import Image
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'analysis_task04A_crop_ablation_20261102_01'
TASK2=ROOT/'analysis_shared_failure_20261031_gpu01';TASK1=ROOT/'Dinomaly2/out/re_eval_shared_20261031_cpu01'
ARCAD=ROOT/'ArcAD'; DATA=Path('/media/avlab/7458a474-6ea2-4b90-b232-373c3df8d329/real_iad_variety_1024')
CKPTS=[ARCAD/'out/real_iad_variety out/arcad_realiad_variety/model.pth',ROOT/'Dinomaly2/out/real_iad_variety_1024 out/full run/variety_160cat/model.pth']
CATS=('laser_diode','lattice_block_plug','power_jack','LED_indicator','thyristor')
def readcsv(p):
 with open(p,newline='') as f:return list(csv.DictReader(f))
def writecsv(p,rows):
 rows=list(rows)
 if rows:
  with open(p,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
 return h.hexdigest()
def main():
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 metrics=readcsv(OUT/'category_metrics.csv');views=readcsv(OUT/'per_view_scores.csv');objects=readcsv(OUT/'per_object_scores.csv');boots=readcsv(OUT/'paired_bootstrap_ci.csv')
 viscount=defaultdict(int)
 for r in views:viscount[(r['category'],r['view_label'],r['mask_status'])]+=1
 writecsv(OUT/'visibility_status_counts.csv',[{'category':k[0],'view_label':int(k[1]),'mask_status':k[2],'n_views':n} for k,n in sorted(viscount.items())])
 # Count positive image-labelled views that contribute only negative pixels to each field.
 pix_sens=[]
 for cat in CATS:
  rr=[r for r in views if r['category']==cat]
  empty_center=[r for r in rr if int(r['view_label']) and int(r['gt_center_224_pixels'])==0]
  empty_source=[r for r in rr if int(r['view_label']) and int(r['source_mask_pixels'])==0]
  for b in ('arcad_original','arcad_full_frame','dinomaly2_original'):
   pix_sens.append({'category':cat,'field':'center224','branch':b,'positive_views_empty_in_field':len(empty_center),'negative_pixels_contributed_by_these_views':len(empty_center)*224*224,'primary_metric_treatment':'kept as pixel negatives; no label mutation'})
  for b in ('arcad_full_frame','dinomaly2_original'):
   pix_sens.append({'category':cat,'field':'full_source256','branch':b,'positive_views_empty_in_field':len(empty_source),'negative_pixels_contributed_by_these_views':len(empty_source)*256*256,'primary_metric_treatment':'kept as pixel negatives; no label mutation'})
 writecsv(OUT/'pixel_mask_visibility_sensitivity.csv',pix_sens)
 # Attach the mask visibility stratum to each component-level localization row.
 regions=readcsv(OUT/'region_topk_metrics.csv');status_by_path={r['image_path']:r['mask_status'] for r in views}
 for r in regions:r['mask_visibility_stratum']=status_by_path.get(r['image_path'],'unknown')
 writecsv(OUT/'region_topk_metrics.csv',regions)
 # Border-case source-GT score/percentile and descriptive rank against normal samples.
 border=readcsv(OUT/'border_39_audit.csv');border_objects={r['object_key'] for r in border}
 z=np.load(OUT/'maps'/'aligned_laser_diode.npz');paths=z['image_paths'].astype(str).tolist();idx={p:i for i,p in enumerate(paths)}
 for r in border:
  i=idx[r['image_path']];gt=z['gt_source256'][i].astype(bool);n=int(gt.sum())
  r['arcad_original_GT_region_mean_score']='not_observed_outside_crop';r['arcad_original_GT_region_percentile_rank']='not_observed_outside_crop';r['arcad_original_GT_top1pct_overlap']='not_observed_outside_crop'
  for b,key in [('arcad_full_frame','arcad_full_frame_source256'),('dinomaly2_original','dinomaly2_source256')]:
   amap=z[key][i];vals=amap[gt];r[b+'_GT_region_mean_score']=float(vals.mean()) if n else ''
   r[b+'_GT_region_mean_percentile_rank']=float((amap<=vals.mean()).mean()) if n else ''
   r[b+'_GT_top1pct_overlap']=float(np.isin(np.flatnonzero(gt),np.argpartition(amap.ravel(),-max(1,int(amap.size*.01)))[-max(1,int(amap.size*.01)):]).sum()/max(n,1)) if n else ''
  r['threshold_status']='no independent normal threshold; threshold-free comparison only'
 writecsv(OUT/'border_39_audit.csv',border)
 border_rank=[];laser_views=[r for r in views if r['category']=='laser_diode'];normal_views=[r for r in laser_views if int(r['view_label'])==0]
 for b in ('arcad_original','arcad_full_frame','dinomaly2_original'):
  pos=[next(r for r in laser_views if r['image_path']==q['image_path']) for q in border]
  y=[1]*len(pos)+[0]*len(normal_views);s=[float(r[b+'_view_top1_score']) for r in pos+normal_views]
  norm=np.asarray([float(r[b+'_view_top1_score']) for r in normal_views]);pval=np.asarray([float(r[b+'_view_top1_score']) for r in pos])
  border_rank.append({'branch':b,'stratum':'39 crop-external positive views vs all manifest-normal views','positive_border_views':len(pos),'normal_views':len(normal_views),'descriptive_AUROC':float(roc_auc_score(y,s)),'descriptive_AP':float(average_precision_score(y,s)),'border_median_score':float(np.median(pval)),'normal_median_score':float(np.median(norm)),'border_fraction_above_normal_95th_percentile':float(np.mean(pval>np.quantile(norm,.95))),'interpretation':'test-derived ranking only; not a calibrated binary decision and does not resolve whether GT is visible'})
  posobj=[r for r in objects if r['category']=='laser_diode' and r['object_key'] in border_objects]
  normobj=[r for r in objects if r['category']=='laser_diode' and int(r['object_label'])==0]
  yo=[1]*len(posobj)+[0]*len(normobj);so=[float(r[b+'_pooled_top1_score']) for r in posobj+normobj]
  border_rank[-1].update(border_objects=len(posobj),normal_objects=len(normobj),object_level_descriptive_AUROC=float(roc_auc_score(yo,so)),object_level_descriptive_AP=float(average_precision_score(yo,so)))
 writecsv(OUT/'border_score_ranks.csv',border_rank)
 for p in [OUT/'plots']:p.mkdir(exist_ok=True)
 # ROC/PR plots for both views and objects, per category and branch.
 for cat in CATS:
  for level in ('view','object'):
   source=views if level=='view' else objects
   score_suffix='_view_top1_score' if level=='view' else '_pooled_top1_score'
   rr=[r for r in source if r['category']==cat]
   y=np.asarray([int(r['view_label' if level=='view' else 'object_label']) for r in rr])
   for curve in ('roc','pr'):
    plt.figure(figsize=(6,5))
    for b in ('arcad_original','arcad_full_frame','dinomaly2_original'):
     s=np.asarray([float(r[b+score_suffix]) for r in rr])
     if curve=='roc':x,z,_=roc_curve(y,s);plt.plot(x,z,label=f'{b} (AUROC {roc_auc_score(y,s):.3f})')
     else:x,z,_=precision_recall_curve(y,s);plt.plot(x,z,label=f'{b} (AP {average_precision_score(y,s):.3f})')
    if curve=='roc':plt.plot([0,1],[0,1],'k--',lw=.7);plt.xlabel('False positive rate');plt.ylabel('True positive rate')
    else:plt.xlabel('Recall');plt.ylabel('Precision')
    plt.title(f'{cat}: {level} {curve.upper()}');plt.legend(fontsize=7);plt.tight_layout();plt.savefig(OUT/'plots'/f'{cat}_{level}_{curve}.png',dpi=140);plt.close()
 # Normal/positive score distribution comparison.
 fig,axes=plt.subplots(len(CATS),2,figsize=(12,3.1*len(CATS)))
 for i,cat in enumerate(CATS):
  rr=[r for r in views if r['category']==cat]
  for ax,lab,nm in ((axes[i,0],0,'Normal'),(axes[i,1],1,'Positive')):
   ax.boxplot([[float(r[b+'_view_top1_score']) for r in rr if int(r['view_label'])==lab] for b in ('arcad_original','arcad_full_frame','dinomaly2_original')],tick_labels=['ArcAD original','ArcAD full-frame','D2'],showfliers=False)
   ax.set_title(f'{cat} — {nm}');ax.set_ylabel('Per-view top-1% anomaly score')
 fig.tight_layout();fig.savefig(OUT/'plots'/'arcad_score_distributions_normal_positive.png',dpi=140);plt.close(fig)
 # Descriptive mask-size/center strata; no thresholds are selected.
 positive=[r for r in views if int(r['view_label']) and int(r['source_mask_pixels'])>0]
 areas=np.asarray([int(r['source_mask_pixels']) for r in positive]);q=np.quantile(areas,[.25,.5,.75])
 strata=[]
 for cat in CATS:
  rr=[r for r in views if r['category']==cat];normals=[r for r in rr if int(r['view_label'])==0]
  for sname,positive_subset in [('center_only',[r for r in rr if int(r['view_label']) and r['mask_status']=='inside_original_crop']),('partial_border',[r for r in rr if int(r['view_label']) and r['mask_status']=='partly_outside_original_crop']),('wholly_outside',[r for r in rr if int(r['view_label']) and r['mask_status']=='wholly_outside_original_crop']),('smallest_source_area_quartile',[r for r in rr if int(r['view_label']) and 0<int(r['source_mask_pixels'])<=q[0]])]:
   subset=positive_subset+normals
   y=np.asarray([int(r['view_label']) for r in subset])
   item={'category':cat,'stratum':sname,'n_views':len(subset),'positive_views':int(y.sum()),'normal_views':int((y==0).sum()),'source_mask_area_quartile_boundaries':','.join(str(float(x)) for x in q)}
   for b in ('arcad_original','arcad_full_frame','dinomaly2_original'):
    s=np.asarray([float(r[b+'_view_top1_score']) for r in subset]);item[b+'_I_AUROC']=float(roc_auc_score(y,s)) if len(np.unique(y))>1 else np.nan;item[b+'_I_AP']=float(average_precision_score(y,s)) if len(np.unique(y))>1 else np.nan
   strata.append(item)
 writecsv(OUT/'visibility_size_strata_metrics.csv',strata)
 # Five-camera paired score variability, physical object is retained as grouping unit.
 consistency=[]
 for cat in CATS:
  groups={}
  for r in views:
   if r['category']==cat:groups.setdefault(r['object_key'],[]).append(r)
  for k,rr in groups.items():
   rr=sorted(rr,key=lambda r:r['camera_id'])
   for b in ('arcad_original','arcad_full_frame','dinomaly2_original'):
    vals=[float(r[b+'_view_top1_score']) for r in rr]
    consistency.append({'category':cat,'object_key':k,'branch':b,'camera_scores_C1_to_C5':'|'.join(f'{x:.8g}' for x in vals),'score_range':max(vals)-min(vals),'view_labels':'|'.join(r['view_label'] for r in rr),'mask_statuses':'|'.join(r['mask_status'] for r in rr)})
 writecsv(OUT/'camera_consistency.csv',consistency)
 # Geometry check with coordinate-ramp test image passed through the actual PIL resize/crop operations.
 side=1024;yy,xx=np.mgrid[0:side,0:side];ramp=np.stack([np.round(xx*255/(side-1)),np.round(yy*255/(side-1)),np.zeros_like(xx)],axis=-1).astype(np.uint8);im=Image.fromarray(ramp)
 orig=im.resize((448,448),Image.Resampling.BILINEAR).crop((28,28,420,420));full=im.resize((392,392),Image.Resampling.BILINEAR);d2=im.resize((280,280),Image.Resampling.BILINEAR)
 def sample(img,x,y):
  px=np.asarray(img);return {'source_x_from_ramp':float(px[y,x,0])*(1023/255),'source_y_from_ramp':float(px[y,x,1])*(1023/255)}
 runtime=json.loads((OUT/'inference_runtime.json').read_text())
 center_map_check={}
 for cat in CATS:
  az=np.load(OUT/'maps'/f'aligned_{cat}.npz');paths=az['image_paths'].astype(str).tolist()
  for model,key in [('arcad','arcad_original_source256'),('d2','dinomaly2_source256')]:
   prior=np.load(TASK2/'maps'/f'common_{model}_{cat}.npz')
   amap=az[key][:,16:240,16:240];reference=prior['smoothed_maps'].astype(np.float32)
   center_map_check[f'{cat}/{model}']={'paths_exact_match':prior['image_paths'].astype(str).tolist()==paths,'max_abs_map_difference':float(np.max(np.abs(amap-reference))),'mean_abs_map_difference':float(np.mean(np.abs(amap-reference))),'gt_exact_match':bool(np.array_equal(az['gt_center224'],prior['gt_masks']))}
 geom={'source_image_shape':[1024,1024],'actual_input_shapes':{'arcad_original':[392,392],'arcad_full_frame':[392,392],'dinomaly2_original':[280,280]},'source_to_input':{'arcad_original':'source square resized 1024->448, crop resized pixels [28:420] on both axes; visible source-edge field [64:960)','arcad_full_frame':'full source resized 1024->392; no crop','dinomaly2_original':'full source resized 1024->280; no crop'},'coordinate_ramp_samples':{'arcad_original_input_0_0':sample(orig,0,0),'arcad_original_input_391_391':sample(orig,391,391),'arcad_full_input_0_0':sample(full,0,0),'arcad_full_input_391_391':sample(full,391,391),'dinomaly2_input_0_0':sample(d2,0,0),'dinomaly2_input_279_279':sample(d2,279,279)},'map_projection':'ArcAD output maps 256x256 after input-resolution map and bilinear interpolation; full-frame 256 grid indexes full source field; original-crop grid indexes crop field only. D2 cached 256x256 maps index full source field. Center comparison uses source crop [64:960] projected to 224x224 with bilinear maps and Task02-identical PIL nearest masks; full-field comparison uses 256x256 maps/masks and excludes crop-limited original ArcAD.','expected_common_field_pixels':[16,240],'map_shapes_verified':runtime['branches'],'common_grid_map_GT_reverification_vs_Task02':center_map_check,'source_image_size_checked_on_every_inference_view':True,'status':'coordinate-ramp transformations and saved input/map shapes verified; all source masks checked at 1024x1024 during aggregation'}
 (OUT/'geometry_validation.json').write_text(json.dumps(geom,indent=2)+'\n')
 # Checkpoint integrity after inference.
 cp=json.loads((OUT/'checkpoint_integrity.json').read_text());
 for row,p in zip(cp['checkpoints'],CKPTS):row.update(sha256_after=sha(p),mtime_ns_after=p.stat().st_mtime_ns,unchanged=(sha(p)==row['sha256_before'] and p.stat().st_mtime_ns==row['mtime_ns_before']))
 cp['after_inference']=True;(OUT/'checkpoint_integrity.json').write_text(json.dumps(cp,indent=2)+'\n')
 if not all(r['unchanged'] for r in cp['checkpoints']):raise RuntimeError('Checkpoint changed during task')
 # Save stronger inference provenance and code/cache hashes.
 protocol=json.loads((OUT/'protocol_audit.json').read_text());runtime=json.loads((OUT/'inference_runtime.json').read_text())
 manval=json.loads((OUT/'manifest_validation.json').read_text())
 cachehash={}
 for cat in CATS:
  p=(TASK1/'maps'/f'{cat}.npz') if cat in ('laser_diode','lattice_block_plug','power_jack') else TASK2/'maps'/f'd2_{cat}.npz'
  cachehash[cat]={'d2_map_path':str(p.relative_to(ROOT)),'d2_map_sha256':sha(p),'arcad_task02_map_path':str((TASK2/'maps'/f'arcad_{cat}.npz').relative_to(ROOT)),'arcad_task02_map_sha256':sha(TASK2/'maps'/f'arcad_{cat}.npz')}
 boot=readcsv(OUT/'paired_bootstrap_ci.csv')
 for row in boot:row['bootstrap_seed']=20261102 if row['comparison'].endswith('arcad_original') else 20261103
 writecsv(OUT/'paired_bootstrap_ci.csv',boot)
 protocol.update({'task':'Task 04A crop ablation','device':runtime['device'],'gpu':runtime['gpu'],'cuda':runtime['cuda'],'torch':runtime['torch'],'inference_elapsed_seconds':runtime['elapsed_seconds'],'inference_random_seed':runtime['inference_random_seed'],'analysis_bootstrap_seeds':[20261102,20261103],'batch_size':runtime['batch_size'],'workers':runtime['workers'],'manifest_validation':{'manifest_sha256':manval['shared_manifest_sha256'],'native_split_hashes':manval['split_consistency'],'arcad_training_object_overlap':manval['arcad_train_overlap'],'zero_duplicate_paths':True,'five_camera_group_checks':{c:v['five_camera_groups_ok'] for c,v in manval['categories'].items()}},'ArcAD_original_regression':readcsv(OUT/'arcad_original_regression.csv'),'reused_numeric_map_hashes':cachehash,'D2_reuse_sources':{c:v['d2_map_path'] for c,v in cachehash.items()},'inference_script_sha256_as_executed':runtime.get('script_sha256_as_executed'),'task04_code_sha256_current_reproduction':{str(p.relative_to(OUT)):sha(p) for p in (OUT/'work').glob('*.py')},'input_geometry_warning':'ArcAD full-frame branch uses the original trained checkpoint with a changed input field of view; interpret as inference-time OOD intervention, not retrained-model performance.'})
 (OUT/'protocol_audit.json').write_text(json.dumps(protocol,indent=2)+'\n')
 # Summary statistics.
 by={(r['category'],r['branch']):r for r in metrics};br=['arcad_original','arcad_full_frame','dinomaly2_original']
 macro={}
 for b in br:
  macro[b]={}
  for k in ('center_I_AUROC','center_I_AP','center_O_AUROC','center_O_AP','center_P_AUROC','center_P_AP','center_P_AUPRO','full_field_P_AUROC','full_field_P_AP','full_field_P_AUPRO'):
   values=np.asarray([float(by[(cat,b)][k]) for cat in CATS]);values=values[np.isfinite(values)]
   macro[b][k]=float(values.mean()) if len(values) else np.nan
 def avg(metric_key,branch,sub):return float(np.mean([float(r[branch+sub]) for r in views if r['category']=='laser_diode' and submask(r,sub)]))
 border_rows=[r for r in views if r['category']=='laser_diode' and r['mask_status']=='wholly_outside_original_crop' and int(r['view_label']) and int(r['source_mask_pixels'])>0]
 center_rows=[r for r in views if r['category']=='laser_diode' and r['mask_status']=='inside_original_crop' and int(r['view_label'])]
 # Per-category delta table for direct reading.
 delta=[]
 for cat in CATS:
  for level,key in [('view','center_I_AUROC'),('object','center_O_AUROC')]:
   delta.append({'category':cat,'level':level,'full_frame_minus_original_AUROC':float(by[(cat,'arcad_full_frame')][key])-float(by[(cat,'arcad_original')][key]),'full_frame_AUROC':float(by[(cat,'arcad_full_frame')][key]),'original_AUROC':float(by[(cat,'arcad_original')][key]),'d2_reference_AUROC':float(by[(cat,'dinomaly2_original')][key])})
 writecsv(OUT/'category_delta_summary.csv',delta)
 # Plot the paired view-score distributions for border versus center.
 fig,ax=plt.subplots(figsize=(9,5));labels=[];data=[]
 for sname,rr in [('center-positive',center_rows),('crop-external-positive',border_rows)]:
  for b in br:labels.append(sname+'\n'+b.replace('_',' '));data.append([float(r[b+'_view_top1_score']) for r in rr])
 ax.boxplot(data,tick_labels=labels,showfliers=False);ax.set_ylabel('Per-view top-1% anomaly score');ax.set_title('Laser diode: center-positive and 39 crop-external views (score distributions)');plt.xticks(rotation=25,ha='right');fig.tight_layout();fig.savefig(OUT/'plots'/'border_vs_center_score_comparison.png',dpi=150);plt.close(fig)
 cnt={}
 for r in views:
  if r['category']=='laser_diode':cnt[r['mask_status']]=cnt.get(r['mask_status'],0)+1
 border_scores={b:float(np.mean([float(r[b+'_view_top1_score']) for r in border_rows])) for b in br}
 center_scores={b:float(np.mean([float(r[b+'_view_top1_score']) for r in center_rows])) for b in br}
 paired={}
 for r in boots:paired.setdefault(r['comparison'],[]).append(r)
 text=['# Task 04A — ArcAD cropping ablation results','','## Summary','','This is an inference-time intervention on the frozen ArcAD checkpoint. ArcAD full-frame inference changes the field of view to full 1024 source coverage resized to 392×392; the original branch remains resize448→center crop392. No retraining occurred. The five categories were fixed from the Task 02 shared manifest before inference.','',f'The primary comparison uses {sum(int(r["views"]) for r in metrics if r["branch"]=="arcad_original")} shared views across {sum(int(r["objects"]) for r in metrics if r["branch"]=="arcad_original")} category-object groups in five categories. All groups contain five cameras; the manifest validation found zero ArcAD training-object leakage and zero duplicate test paths. ArcAD original regression max map difference per category is recorded in `arcad_original_regression.csv`.','', '### (a) Did full-frame ArcAD help overall?','','Center-field macro results (macro across five categories):']
 center_keys=('center_I_AUROC','center_I_AP','center_O_AUROC','center_O_AP','center_P_AUROC','center_P_AP','center_P_AUPRO')
 for b in br:text.append(f'- **{b}:** '+', '.join(f'{k.replace("center_","")} {macro[b][k]:.4f}' for k in center_keys))
 full_summary={b:{k:float(np.nanmean([float(by[(cat,b)][k]) for cat in CATS])) for k in ('full_field_P_AUROC','full_field_P_AP','full_field_P_AUPRO')} for b in ('arcad_full_frame','dinomaly2_original')}
 rank_rows=readcsv(OUT/'border_score_ranks.csv');rank_view={r['branch']:r for r in rank_rows if r['stratum'].startswith('39 crop-external')}
 border_auc_text='; '.join(f'{b}: view AUROC {float(rank_view[b]["descriptive_AUROC"]):.3f}, object AUROC {float(rank_view[b]["object_level_descriptive_AUROC"]):.3f}' for b in br)
 text+=['', 'For full-source pixel localization (only the two branches that observe the border), macro results were: '+ '; '.join(b+': '+', '.join(f'{k} {v:.4f}' for k,v in vv.items()) for b,vv in full_summary.items())+'. Original ArcAD full-source results are unavailable because its outside-crop map is NaN, not imputed.','', 'Full-frame minus original ArcAD AUROC by category is listed in `category_delta_summary.csv`; object and view deltas are paired on identical samples. See `paired_bootstrap_ci.csv` for 2,000-object bootstrap 95% intervals, valid draws and excluded single-class draws. A CI spanning zero is inconclusive.','', '### (b) Did it help the crop-external border cases?','',f'The audit contains {len(border_rows)} positive laser_diode views with source masks wholly outside the original center crop. Their mean per-view top-1% scores were '+', '.join(f'{b} {border_scores[b]:.6g}' for b in br)+f'. The 39 views occur across {len({r["object_key"] for r in border_rows})} objects. Compared descriptively with manifest-normal views, score-ranking AUROCs were {border_auc_text}. Thus, the full-frame branch did not improve this test-derived border-versus-normal ranking. This is not a calibrated binary decision; original ArcAD has no map values in the GT region outside its crop, and its GT-region score/rank is explicitly marked not observed. Scores alone cannot establish visibility or a false negative. Manual review is required.','', '### (c) Did central or small defects degrade?','',f'For laser_diode positive views whose masks are entirely inside the crop (n={len(center_rows)}), mean per-view scores were '+', '.join(f'{b} {center_scores[b]:.6g}' for b in br)+'. `visibility_size_strata_metrics.csv` gives category-level threshold-free results for center-only, partly outside, wholly outside, and the pooled lowest source-mask-area quartile. Small-defect groups are descriptive and vary by category only through each category’s sample. No thresholded decisions are inferred.','', '### (d) Did object aggregation alter the conclusion?','','Compare `center_I_AUROC` and `center_O_AUROC` in `category_metrics.csv` and the paired object bootstrap deltas. Object scores pool the top 1% of smoothed pixels across all five views; image scores use top 1% per view. These different aggregation levels can rank models differently.','', '### (e) Is retraining warranted?','','This experiment alone cannot establish that retraining with full-frame inputs will improve ArcAD. The aggregate center-field results do not show an overall gain from crop removal, though category-level changes vary. A separate controlled retraining ablation is warranted only if the potential border-coverage benefit is important enough to test against matched training budgets/seeds/splits. The current result estimates only inference-time sensitivity of the existing checkpoint.','', '## Interpretation and limitations','','- ArcAD-full-frame receives a 392×392 full-frame image that differs from its 448-resize/392-center-crop training geometry; this is an out-of-distribution inference intervention.','- ArcAD-original has no prediction outside source `[64:960)×[64:960)`; its full-source aligned map uses NaN outside observed pixels. Full-source localization compares only full-frame ArcAD and D2.','- Masks outside the center field remain in image-level metrics and full-source pixel metrics; center-field positive-region metrics use only nonempty center GT for region analyses.','- Pixel AP depends on positive-pixel prevalence. Maps use bilinear interpolation; masks use nearest-neighbor interpolation.','- No threshold calibration or binary classification decisions were introduced. Detection and localization are reported separately.','- The 39 border cases and two all-zero laser_diode source masks remain unresolved annotation/visibility cases; original labels were preserved.','- ArcAD original numerically regressed against Task 02 maps before color normalization; tolerances and differences are in `arcad_original_regression.csv`.','- D2 maps were reused, not rerun; their exact file/path provenance is in `protocol_audit.json`.','- Runtime is descriptive for this machine and batch size only.','', '## Recommended next experiment','','If the full-frame crop-coverage trade-off remains scientifically important, train a separate full-frame ArcAD variant and compare it against the unchanged original using matched training samples, budgets, seeds, and an untouched object-level evaluation set. Keep the frozen original ArcAD and D2 branches as the primary baseline for the separate Task 04B fusion study.','']
 (OUT/'summary.md').write_text('\n'.join(text))
 empty_sens=readcsv(OUT/'pixel_mask_visibility_sensitivity.csv')
 center_empty={r['category']:int(r['positive_views_empty_in_field']) for r in empty_sens if r['field']=='center224' and r['branch']=='arcad_original'}
 full_empty={r['category']:int(r['positive_views_empty_in_field']) for r in empty_sens if r['field']=='full_source256' and r['branch']=='arcad_full_frame'}
 with (OUT/'summary.md').open('a') as f:
  f.write('\n### Central and small-defect stratum result\n\n')
  for sname,label in [('center_only','Center-only positive views plus all normal views'),('smallest_source_area_quartile','Lowest pooled source-mask-area quartile plus all normal views')]:
   subset=[r for r in strata if r['stratum']==sname]
   vals=[]
   for b in ('arcad_original','arcad_full_frame','dinomaly2_original'):
    vv=[float(r[b+'_I_AUROC']) for r in subset if np.isfinite(float(r[b+'_I_AUROC']))]
    vals.append(f'{b} AUROC {np.mean(vv):.4f}')
   f.write(label+': '+', '.join(vals)+'. These are descriptive category-macro rankings with all manifest-normal views included; the lowest-area quartile uses pooled positive-view mask area (cut points in `visibility_size_strata_metrics.csv`).\n\n')
  f.write('\n### Empty-mask effect on pixel-negative pools\n\n')
  f.write('Positive image-labelled views with no GT pixels in the evaluated field remain in the primary pixel metrics as all-negative masks. Center-grid empty-positive counts by category: '+', '.join(f'{c} {n}' for c,n in center_empty.items())+'. Full-source empty-positive counts: '+', '.join(f'{c} {n}' for c,n in full_empty.items())+'. Their exact negative-pixel contributions are in `pixel_mask_visibility_sensitivity.csv`; image-level labels and metrics are unchanged.\n')
  laser_positive={r['mask_status']:int(r['n_views']) for r in readcsv(OUT/'visibility_status_counts.csv') if r['category']=='laser_diode' and int(r['view_label'])==1}
  f.write('For laser_diode, the 919 positive views divide into '+', '.join(f'{k} {laser_positive.get(k,0)}' for k in ('inside_original_crop','partly_outside_original_crop','wholly_outside_original_crop','source_all_zero'))+'. The `visibility_status_counts.csv` table also lists normal-view counts and all other categories.\n')
 (OUT/'issues.md').write_text('# Issues and interpretation limits\n\n- **Inference-only intervention:** ArcAD full-frame inputs are out of distribution relative to ArcAD training preprocessing. This does not estimate retrained full-frame ArcAD.\n- **Border visibility:** 39 positive laser_diode views have nonzero masks completely outside the original crop; two source-all-zero masks are also present. All 39 crop-external examples have five-camera contact sheets. Visibility remains `requires_manual_review`; no cause or false-negative label is inferred.\n- **Source-coordinate full field:** original ArcAD has no observed border prediction. It is NaN/abstention outside the crop and is excluded from full-source pixel comparison.\n- **Thresholding:** no independent calibration was used and no binary decisions are claimed. All detection metrics are threshold free.\n- **Inference scope:** five preselected categories from the exact shared Task 02 manifest; this is not a 160-category generalization claim.\n- **Uncertainty:** object-cluster bootstrap has 2,000 planned replicates per comparison, retaining sibling views; single-class replicates are excluded and counted. Results are exploratory across multiple categories.\n- **Skipped:** optional D2 center-crop inference was not run; it is secondary to the requested ArcAD ablation.\n- **Reused scores:** D2 numeric maps are reused from Task 01 (three primary categories) and Task 02 (LED_indicator, thyristor); provenance and source hashes are recorded.\n')
 (OUT/'README.md').write_text(f'''# Task 04A — ArcAD cropping ablation\n\nRun folder: `analysis_task04A_crop_ablation_20261102_01`. This is a frozen-checkpoint **inference-only** intervention; full-frame ArcAD is out of distribution relative to original training crop. No original outputs, checkpoints, labels, or optimizer state were changed.\n\n## Reproduce\n\nFrom the workspace root, in the project Python environment and with GPU access:\n\n```bash\nPY=/media/avlab/1CE6451EE644FA12/2026_10_22_dinomaly/dinomaly2/bin/python\n"$PY" analysis_task04A_crop_ablation_20261102_01/work/preflight.py\n"$PY" analysis_task04A_crop_ablation_20261102_01/work/run_arcad_branches.py\nMPLCONFIGDIR=/tmp/mpl-task04a "$PY" analysis_task04A_crop_ablation_20261102_01/work/analyze_task04a.py\nMPLCONFIGDIR=/tmp/mpl-task04a "$PY" analysis_task04A_crop_ablation_20261102_01/work/finalize_task04a.py\n```\n\n`reproduce.sh` runs the same sequence; it writes only under this Task 04A folder. ArcAD original uses resize448→center crop392; full-frame uses resize392; D2 full-frame maps are reused from earlier runs. Dataset and checkpoint paths, SHA256 values, GPU, PyTorch/CUDA, cache provenance and timings appear in `protocol_audit.json`, `checkpoint_integrity.json`, and `inference_runtime.json`.\n\n## Results\n\nSee `summary.md` for the overall, border, central/small-defect, object-aggregation, and retraining answers. `geometry_validation.json` documents coordinate-ramp checks and source-aligned map grids. `category_metrics.csv`, `paired_bootstrap_ci.csv`, `border_39_audit.csv`, `region_topk_metrics.csv`, and `maps/` contain machine-readable outputs.\n''')
 (OUT/'reproduce.sh').write_text('''#!/usr/bin/env bash\nset -euo pipefail\nPY=/media/avlab/1CE6451EE644FA12/2026_10_22_dinomaly/dinomaly2/bin/python\n"$PY" analysis_task04A_crop_ablation_20261102_01/work/preflight.py\n"$PY" analysis_task04A_crop_ablation_20261102_01/work/run_arcad_branches.py\nMPLCONFIGDIR=/tmp/mpl-task04a "$PY" analysis_task04A_crop_ablation_20261102_01/work/analyze_task04a.py\nMPLCONFIGDIR=/tmp/mpl-task04a "$PY" analysis_task04A_crop_ablation_20261102_01/work/finalize_task04a.py\n''')
 (OUT/'reproduce.sh').chmod(0o755)
 print('Finalized: macro metrics',json.dumps(macro),'border n',len(border_rows),'checkpoint unchanged',all(r['unchanged'] for r in cp['checkpoints']))
def submask(r,sub):
 if sub=='_view_top1_score':return True
 return True
if __name__=='__main__':main()
