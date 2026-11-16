#!/usr/bin/env python3
"""Paired evidence figures, manual-review index, and mechanism strata."""
import csv,json,re
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path('/media/avlab/1CE6451EE644FA12/2026_10_22_dinomaly');OUT=ROOT/'analysis_task03_20261102_01';T2=ROOT/'analysis_shared_failure_20261031_gpu01';DATA=Path('/media/avlab/7458a474-6ea2-4b90-b232-373c3df8d329/real_iad_variety_1024')
def rd(p):
 with open(p,encoding='utf8') as f:return list(csv.DictReader(f))
def wr(p,rows):
 with open(p,'w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),extrasaction='ignore');w.writeheader();w.writerows(rows)
def num(x):
 try:return float(x)
 except:return float('nan')
def truth(x):return x is True or str(x).strip().lower() in ('1','true','yes')
def main():
 (OUT/'paired_examples').mkdir(exist_ok=True)
 diag=rd(T2/'view_diagnostics.csv'); bydiag={(r['category'],r['model'],r['image_path']):r for r in diag}
 cams=rd(OUT/'object_camera_consistency.csv'); bycam={(r['category'],r['image_path']):r for r in cams}
 op=rd(OUT/'operating_point_metrics.csv');thmap={(r['system'],r['level']):r['threshold'] for r in op if r['category']=='ALL_FIVE' and r['normal_object_fpr_target']=='0.1'}
 joined=rd(T2/'joined_predictions.csv'); pairs=defaultdict(dict)
 for r in joined:pairs[(r['category'],r['image_path'])][r['model']]=r
 cand=defaultdict(list)
 for (cat,path),p in pairs.items():
  if set(p)!={'d2','arcad'}:continue
  d,a=p['d2'],p['arcad'];v=bycam[(cat,path)];positive=int(d['gt_view_label'])==1
  qd=num(v['d2_Q']);qa=num(v['arcad_Q']);rd1=num(v['d2_top1_recall']);ra1=num(v['arcad_top1_recall']);
  if positive:
   if not truth(v['d2_threshold_hit_at_10pct']) and not truth(v['arcad_threshold_hit_at_10pct']):cand['shared_miss'].append((cat,path,p,v))
   if truth(v['d2_threshold_hit_at_10pct']) and not truth(v['arcad_threshold_hit_at_10pct']):cand['D2_only_success'].append((cat,path,p,v))
   if truth(v['arcad_threshold_hit_at_10pct']) and not truth(v['d2_threshold_hit_at_10pct']):cand['ArcAD_only_success'].append((cat,path,p,v))
   if bydiag[(cat,'d2',path)]['mask_nonempty']=='True' and bydiag[(cat,'arcad',path)]['mask_nonempty']=='True' and np.isfinite(rd1) and np.isfinite(ra1):cand['localization_disagreement'].append((abs(rd1-ra1),cat,path,p,v))
   # A localized heatmap whose view-level anomaly score does not pass its heldout-normal operating threshold.
   good=[(model,num(v[f'{model}_Q']),num(v[f'{model}_top1_recall']),num(v[f'{model}_raw_score_top1pct']),truth(v[f'{model}_threshold_hit_at_10pct'])) for model in ('d2','arcad')]
   viable=[x for x in good if x[1]>=.80 and x[2]>=.20 and not x[4]]
   if viable:cand['threshold_only_miss'].append((max(x[1]+x[2] for x in viable),cat,path,p,v))
  else:
   if truth(v['d2_threshold_hit_at_10pct']) or truth(v['arcad_threshold_hit_at_10pct']):cand['normal_false_positive'].append((max(num(v['d2_global_normal_rank']),num(v['arcad_global_normal_rank'])),cat,path,p,v))
 selected=[]
 for kind in ('shared_miss','D2_only_success','ArcAD_only_success','localization_disagreement','threshold_only_miss','normal_false_positive'):
  rows=cand[kind]
  if not rows:continue
  if kind=='shared_miss':rows=sorted(rows,key=lambda x:(x[0],x[1]))
  elif kind in ('D2_only_success','ArcAD_only_success'):rows=sorted(rows,key=lambda x:(x[0],x[1]))
  elif kind in ('localization_disagreement','threshold_only_miss','normal_false_positive'):rows=sorted(rows,key=lambda x:(-x[0],x[1],x[2]))
  item=rows[0];
  if kind in ('shared_miss','D2_only_success','ArcAD_only_success'):cat,path,p,v=item
  else:_,cat,path,p,v=item
  rec=p['d2']; d=bydiag[(cat,'d2',path)];a=bydiag[(cat,'arcad',path)]
  # Verify current source mask transform for figure against the common-grid GT tensor.
  with np.load(T2/'maps'/f'common_d2_{cat}.npz') as z:
   paths=z['image_paths'].astype(str).tolist();idx=paths.index(path);gt=z['gt_masks'][idx].astype(bool);dm=z['smoothed_maps'][idx].astype(np.float32)
  with np.load(T2/'maps'/f'common_arcad_{cat}.npz') as z:am=z['smoothed_maps'][z['image_paths'].astype(str).tolist().index(path)].astype(np.float32)
  rgb=np.asarray(Image.open(DATA/cat/path).convert('RGB').crop((64,64,960,960)).resize((224,224),Image.Resampling.BILINEAR))
  fig,ax=plt.subplots(2,3,figsize=(12,7))
  ax[0,0].imshow(rgb);ax[0,0].imshow(gt,cmap='Reds',alpha=.55);ax[0,0].set_title('Same source view + common-grid GT')
  for axis,arr,title in ((ax[0,1],dm,'D2 heatmap'),(ax[0,2],am,'ArcAD heatmap')):
   axis.imshow(arr,cmap='magma',vmin=np.percentile(arr,1),vmax=np.percentile(arr,99));axis.contour(gt,levels=[.5],colors='cyan',linewidths=.7);axis.set_title(title+' (display colors normalized)')
  for axis,arr,title in ((ax[1,0],dm,'D2 top 1% pixels'),(ax[1,1],am,'ArcAD top 1% pixels')):
   k=max(1,int(arr.size*.01));cut=np.partition(arr.ravel(),arr.size-k)[arr.size-k];top=arr>=cut
   axis.imshow(rgb);axis.imshow(gt,cmap='Reds',alpha=.35);axis.contour(top,levels=[.5],colors='cyan',linewidths=.75);axis.set_title(title+'; cyan = actual selected map pixels')
  ax[1,2].axis('off')
  cam=str(v['camera_id']);cam=cam if cam.startswith('C') else 'C'+cam
  ax[1,2].text(0,1,f"Category: {cat}\nObject: {rec['object_key']}\nCamera: {cam}\nLabel/view: {rec['gt_view_label']}\nDefect type: {rec['defect_type']}\n\nRaw top-1% view scores\nD2: {rec['score_view']}\nArcAD: {p['arcad']['score_view']}\n\nGlobal normal-rank scores\nD2: {v['d2_global_normal_rank']}\nArcAD: {v['arcad_global_normal_rank']}\n50/50: {v['fixed_half_mean_rank']}\n\nCommon-map raw range\nD2 min/max/mean: {dm.min():.5g} / {dm.max():.5g} / {dm.mean():.5g}\nArcAD min/max/mean: {am.min():.5g} / {am.max():.5g} / {am.mean():.5g}\n\nFixed 10% diagnostic thresholds\nD2 native: {thmap.get(('d2','view'),'NA')}\nArcAD native: {thmap.get(('arcad','view'),'NA')}\nFusion rank: {thmap.get(('fixed_50_50_mean','view'),'NA')}\n\nHits at fixed point\nD2: {v['d2_threshold_hit_at_10pct']}\nArcAD: {v['arcad_threshold_hit_at_10pct']}\nFusion: {v['fusion_threshold_hit_at_10pct']}\n\nGT area: {d['gt_area_fraction']}\nQ (D2 / ArcAD): {v['d2_Q']} / {v['arcad_Q']}\nTop1% GT recall: {v['d2_top1_recall']} / {v['arcad_top1_recall']}\nTop1% GT precision: {d['top1_precision']} / {a['top1_precision']}\nTop1% off-target: {1-num(d['top1_precision']):.4g} / {1-num(a['top1_precision']):.4g}\n\nColors normalized per model/view; compare locations, not color magnitudes.",va='top',fontsize=8,wrap=True)
  for axis in ax.ravel():axis.axis('off')
  fig.suptitle(f'{kind} | {cat} | {rec["object_key"]} | {cam}',fontsize=11);fig.tight_layout(rect=(0,0,1,.96));fname=f'{kind}.png';fig.savefig(OUT/'paired_examples'/fname,dpi=140);plt.close(fig)
  selected.append({'example_type':kind,'selection_rule':'fixed deterministic rule using heldout normal calibration thresholds and localization statistics; see script','category':cat,'object_key':rec['object_key'],'image_path':path,'camera_id':cam,'view_label':rec['gt_view_label'],'object_label':rec['gt_object_label'],'defect_type':rec['defect_type'],'d2_raw_top1_score':rec['score_view'],'arcad_raw_top1_score':p['arcad']['score_view'],'d2_normal_rank':v['d2_global_normal_rank'],'arcad_normal_rank':v['arcad_global_normal_rank'],'d2_Q':v['d2_Q'],'arcad_Q':v['arcad_Q'],'d2_top1_recall':v['d2_top1_recall'],'arcad_top1_recall':v['arcad_top1_recall'],'d2_top1_precision':d['top1_precision'],'arcad_top1_precision':a['top1_precision'],'d2_hit':v['d2_threshold_hit_at_10pct'],'arcad_hit':v['arcad_threshold_hit_at_10pct'],'fusion_hit':v['fusion_threshold_hit_at_10pct'],'figure':str((OUT/'paired_examples'/fname).relative_to(OUT)),'display':'heatmap colors normalized per model/view; raw scores and empirical-normal ranks listed separately'})
 wr(OUT/'paired_example_index.csv',selected)
 # Summarize fixed, prespecified strata from Task02 diagnostics and morphology.
 morph=rd(T2/'morphology_view.csv');morphmap={(r['category'],r['image_path']):r for r in morph};regions=rd(T2/'morphology_region.csv');byview=defaultdict(list)
 for r in regions:byview[(r['category'],r['image_path'])].append(r)
 g=defaultdict(list)
 for r in diag:
  if r['gt_view_label']!='1' or r['mask_nonempty']!='True':continue
  m=morphmap.get((r['category'],r['image_path']))
  if not m:continue
  for kind,name in (('size_bin',m['size_bin']),('defect_type',r['defect_type']),('camera',r['camera_id'])):
   g[(r['category'],r['model'],kind,name)].append((r,m,byview[(r['category'],r['image_path'])]))
 out=[]
 for (cat,model,kind,name),items in g.items():
  r0=[x[0] for x in items];m0=[x[1] for x in items];rg=[z for x in items for z in x[2]];widths=[int(x['bbox_width']) for x in rg]
  def mean(k):
   v=[num(x[k]) for x in r0];v=[x for x in v if np.isfinite(x)];return float(np.mean(v)) if v else ''
  out.append({'category':cat,'model':model,'stratum_type':kind,'stratum':name,'n_positive_nonempty_views':len(items),'n_objects':len(set(x['object_key'] for x in r0)),'mean_GT_area_fraction':float(np.mean([num(x['gt_area_fraction']) for x in r0])),'mean_GT_component_count':float(np.mean([int(x['gt_region_count']) for x in r0])),'mean_min_component_width':float(np.mean([min(int(y['bbox_width']) for y in byview[(x['category'],x['image_path'])]) for x in r0 if byview[(x['category'],x['image_path'])]])) if any(byview[(x['category'],x['image_path'])] for x in r0) else '','mean_Q':mean('Q_within_view_pixel_AUROC'),'mean_RGB_local_luminance_contrast':mean('luminance_contrast_gt_minus_7px_ring'),'mean_model_score_C_muG_minus_muB':mean('C_muG_minus_muB'),'mean_model_score_Z':mean('Z_contrast'),'mean_top0.1_recall':mean('top0.1_recall'),'mean_top0.5_recall':mean('top0.5_recall'),'mean_top1_recall':mean('top1_recall'),'mean_top2_recall':mean('top2_recall'),'mean_top5_recall':mean('top5_recall'),'interpretation':'descriptive stratum; RGB contrast is separate from model-derived anomaly-score contrast; no causal assertion; fixed area bins were predeclared'})
 wr(OUT/'mechanism_by_stratum.csv',out)
 (OUT/'manual_review_sheet.md').write_text('# Manual review guide\n\nReview the paired PNGs listed in `paired_example_index.csv` at the original source path and common crop. Each figure shows the same camera image and GT for D2/ArcAD, calibrated-normal ranks and native raw image score values. Cyan outlines mark actual top-1% anomaly-map pixels; red shows common-grid GT. Heatmap display colors are normalized independently; compare spatial positions and consult listed numeric scores for magnitude.\n\n## What is directly observed\n\n- The saved source mask pixel counts and common-crop geometry are recorded in `empty_mask_audit.csv`; 39 masks have no pixels in the common crop and two are source-all-zero.\n- The paired figure shows model score maps and GT at the same source crop. Ranking/coverage values are numeric in the index.\n\n## What remains a hypothesis or needs review\n\n- Occlusion, view-dependent defect visibility, image-label error, specularity, material reflectance, and causal effects of ArcAD supervision require expert review or controlled experiments. No visual appearance alone is treated here as proof of a label error or physical cause. Cases marked for manual review should be checked against original 1024 images and all five siblings.\n')
 print('paired examples',[(x['example_type'],x['category'],x['camera_id']) for x in selected], 'strata rows',len(out))
if __name__=='__main__':main()
