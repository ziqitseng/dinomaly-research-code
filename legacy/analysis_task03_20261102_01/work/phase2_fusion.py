#!/usr/bin/env python3
"""Threshold-free diagnostics, globally calibrated fixed fusion, CIs, ROC/PR."""
import csv,json,hashlib,re
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import beta
from sklearn.metrics import roc_auc_score,average_precision_score,roc_curve,precision_recall_curve

ROOT=Path('/media/avlab/1CE6451EE644FA12/2026_10_22_dinomaly');OUT=ROOT/'analysis_task03_20261102_01';T2=ROOT/'analysis_shared_failure_20261031_gpu01';T1=ROOT/'Dinomaly2/out/re_eval_shared_20261031_cpu01'
CATS=('lattice_block_plug','laser_diode','power_jack','LED_indicator','thyristor');MODELS=('d2','arcad');TARGETS=(.10,.20);BOOT=2000;SEED=20261102
def rd(p):
 with open(p,encoding='utf8') as f:return list(csv.DictReader(f))
def wr(p,rows):
 if not rows:return
 with open(p,'w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),extrasaction='ignore');w.writeheader();w.writerows(rows)
def auc(y,s):
 y=np.asarray(y,dtype=np.uint8);s=np.asarray(s,float)
 if len(np.unique(y))<2:return float('nan'),float('nan')
 return float(roc_auc_score(y,s)),float(average_precision_score(y,s))
def qtile(x,q):return float(np.quantile(np.asarray(x,float),1-q,method='higher'))
def ecdf(cal,x):
 z=np.sort(np.asarray(cal,float));x=np.asarray(x,float);l=np.searchsorted(z,x,'left');r=np.searchsorted(z,x,'right');return ((l+r)/2)/len(z)
def cp(k,n,conf=.95):
 if n==0:return (float('nan'),float('nan'))
 a=(1-conf)/2;lo=0 if k==0 else beta.ppf(a,k,n-k+1);hi=1 if k==n else beta.ppf(1-a,k+1,n-k)
 return float(lo),float(hi)
def cm(y,p):
 y=np.asarray(y,bool);p=np.asarray(p,bool);tn=int(np.sum(~y&~p));fp=int(np.sum(~y&p));fn=int(np.sum(y&~p));tp=int(np.sum(y&p));return {'TN':tn,'FP':fp,'FN':fn,'TP':tp,'TN_rate':tn/max(1,tn+fp),'FP_rate':fp/max(1,tn+fp),'FN_rate':fn/max(1,fn+tp),'TP_rate':tp/max(1,fn+tp),'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'F1':2*tp/max(1,2*tp+fp+fn)}
def ftable(y,d,a):
 y=np.asarray(y,bool);d=np.asarray(d,bool);a=np.asarray(a,bool);p=y
 return {'positive_n':int(p.sum()),'both_detected':int(np.sum(p&d&a)),'D2_only':int(np.sum(p&d&~a)),'ArcAD_only':int(np.sum(p&~d&a)),'neither':int(np.sum(p&~d&~a)),'failure_intersection':int(np.sum(p&~d&~a)),'D2_misses':int(np.sum(p&~d)),'ArcAD_misses':int(np.sum(p&~a)),'oracle_union_coverage':float(np.sum(p&(d|a))/max(1,p.sum())),'oracle_label':'same-sample union coverage; not a learned fusion'}
def main():
 (OUT/'plots').mkdir(exist_ok=True)
 manifest=json.load(open(T2/'shared_manifest.json')); joined=rd(T2/'joined_predictions.csv'); metrics=rd(T2/'category_metrics.csv');
 # Keep exact joined pairs and enumerate calibration/test objects.
 bypair=defaultdict(dict)
 for r in joined:
  if r['category'] in CATS:bypair[(r['category'],r['image_path'])][r['model']]=r
 if any(set(v)!={'d2','arcad'} for v in bypair.values()):raise RuntimeError('unpaired view rows in Task02 cache')
 pairs=[]; objrows=defaultdict(dict);cal_keys={c:set(manifest['categories'][c]['calibration_object_keys']) for c in CATS}
 for (cat,path),p in sorted(bypair.items()):
  d,a=p['d2'],p['arcad']
  if d['object_key']!=a['object_key'] or d['gt_view_label']!=a['gt_view_label']:raise RuntimeError('paired identity or label mismatch')
  row={'category':cat,'object_key':d['object_key'],'image_path':path,'view_id':int(d['view_id']),'camera_id':'C'+re.search(r'_C([1-5])_',path).group(1),'y_view':int(d['gt_view_label']),'y_object':int(d['gt_object_label']),'defect_type':d['defect_type'],'d2':float(d['score_view']),'arcad':float(a['score_view']),'d2_object':float(d['score_object']),'arcad_object':float(a['score_object']),'split':'calibration_normal' if d['object_key'] in cal_keys[cat] else 'heldout_test'}
  pairs.append(row);objrows[(cat,row['object_key'])]=row
 # all five views in each physical group; labels agree and object labels preserved.
 groups=defaultdict(list)
 for r in pairs:groups[(r['category'],r['object_key'])].append(r)
 for k,rs in groups.items():
  if len(rs)!=5 or sorted(x['view_id'] for x in rs)!=[1,2,3,4,5]:raise RuntimeError(f'bad five-camera membership {k}')
  if max(x['y_view'] for x in rs)!=rs[0]['y_object']:raise RuntimeError('object label mismatch')
 cals=[rs for k,rs in groups.items() if k[1] in cal_keys[k[0]]]
 if len(cals)!=50 or any(max(x['y_object'] for x in rs)!=0 for rs in cals):raise RuntimeError('expected exactly 50 normal calibration objects')
 eval_groups={k:rs for k,rs in groups.items() if k[1] not in cal_keys[k[0]]};evalpairs=[r for r in pairs if r['split']=='heldout_test']
 # Model-wise global empirical normal-score CDFs, fit on 50 independent object IDs.
 viewcal={m:np.asarray([r[m] for rs in cals for r in rs]) for m in MODELS}
 objcal={m:np.asarray([rs[0][m+'_object'] for rs in cals]) for m in MODELS}
 calmax={m:np.asarray([max(r[m] for r in rs) for rs in cals]) for m in MODELS}
 for r in pairs:
  r['d2_rank']=float(ecdf(viewcal['d2'],[r['d2']])[0]);r['arcad_rank']=float(ecdf(viewcal['arcad'],[r['arcad']])[0]);r['fusion_mean']=.5*(r['d2_rank']+r['arcad_rank'])
  r['d2_object_rank']=float(ecdf(objcal['d2'],[r['d2_object']])[0]);r['arcad_object_rank']=float(ecdf(objcal['arcad'],[r['arcad_object']])[0]);r['fusion_object_mean']=.5*(r['d2_object_rank']+r['arcad_object_rank'])
 # Calibration score distribution for global fused mean, object and max-view protocols.
 calfusion_obj=np.asarray([.5*(ecdf(objcal['d2'],[rs[0]['d2_object']])[0]+ecdf(objcal['arcad'],[rs[0]['arcad_object']])[0]) for rs in cals])
 calfusion_viewmax=np.asarray([max(.5*(ecdf(viewcal['d2'],[r['d2']])[0]+ecdf(viewcal['arcad'],[r['arcad']])[0]) for r in rs) for rs in cals])
 thresholds={}
 for alpha in TARGETS:
  thresholds[alpha]={
   'd2_view':qtile(calmax['d2'],alpha),'arcad_view':qtile(calmax['arcad'],alpha),'fusion_view':qtile(calfusion_viewmax,alpha),
   'd2_object':qtile(objcal['d2'],alpha),'arcad_object':qtile(objcal['arcad'],alpha),'fusion_object':qtile(calfusion_obj,alpha)}
 # Output calibration manifest with identity-level separation and provenance.
 calmanifest={'calibration_design':'Task02 preselected normal-only object keys, SHA256 order; 10 per category, pooled global transform across five diagnostic categories. Exact same object keys are excluded from all held-out evaluation rows. This is object-disjoint but uses a reserved portion of native test manifests, not an external deployment calibration set.','n_calibration_normal_objects':len(cals),'n_calibration_normal_views':sum(len(x) for x in cals),'n_heldout_objects':len(eval_groups),'n_heldout_normal_objects':sum(rs[0]['y_object']==0 for rs in eval_groups.values()),'n_heldout_positive_objects':sum(rs[0]['y_object']==1 for rs in eval_groups.values()),'categories':{},'calibration_object_keys':{c:sorted(cal_keys[c]) for c in CATS},'arcad_labeled_object_overlap_with_calibration':0,'arcad_labeled_object_overlap_with_heldout':0,'threshold_levels_used':[.10,.20],'not_claimed':'No deployment-grade calibration claim; validation objects come from native test manifests and only five preselected categories.'}
 for c in CATS:
  ce=[k for k in eval_groups if k[0]==c];calmanifest['categories'][c]={'calibration_objects':len(cal_keys[c]),'heldout_objects':len(ce),'heldout_normal_objects':sum(eval_groups[k][0]['y_object']==0 for k in ce),'heldout_positive_objects':sum(eval_groups[k][0]['y_object']==1 for k in ce)}
 (OUT/'calibration_split_manifest.json').write_text(json.dumps(calmanifest,indent=2)+'\n')
 (OUT/'fusion_validation_selection.json').write_text(json.dumps({'validation_anomalies_available':False,'tuned_weighted_mean':'skipped','reason':'No independent validation anomaly set; selecting the weight on the heldout diagnostic labels is prohibited.','fixed_baseline':'50/50 mean of model-wise global empirical-normal CDF ranks, fit only on 50 reserved normal objects (10/category), heldout object keys excluded.','additional_baseline':'Calibrated OR: each model independently flags at the pooled normal-object empirical threshold; report union false-positive cost.','calibration_transfer':'Global score CDF estimated across the five already preselected categories and applied to remaining objects in those categories; diagnostic transfer only, not 160-category generalization.','fusion_weight':0.5,'no_test_label_tuning':True,'pixel_fusion':'skipped; optional, because common-grid score calibration is not independently validated and existing aligned region outputs suffice for this diagnostic.'},indent=2)+'\n')
 # Describe source thresholds that collapse; never claim separate 1/5/10% targets.
 oldthr=rd(T2/'thresholds.csv'); threshold_audit=[]
 for c in CATS:
  for m in MODELS:
   x=[r for r in oldthr if r['category']==c and r['model']==m and r.get('threshold_level')=='view']
   vals=[float(r['threshold']) for r in x]
   threshold_audit.append({'category':c,'model':m,'prior_nominal_targets':'|'.join(r['target_normal_object_fpr'] for r in x),'prior_thresholds_native_scale':'|'.join(r['threshold'] for r in x),'distinct_threshold_count':len(set(vals)),'calibration_normal_objects':10,'realized_calibration_normal_object_fpr_each':x[0].get('calibration_realized_fpr','') if x else '', 'task03_status':'rejected as distinct operating points; 10 normals collapse all nominal 1/5/10% targets to one max-score threshold'})
 wr(OUT/'threshold_audit.csv',threshold_audit)
 (OUT/'calibration_reliability.md').write_text('# Calibration reliability\n\nThe Task 02 per-category normal-only calibration set has only 10 objects/category. The empirical higher quantile selects the same maximum view-score threshold for nominal 1%, 5%, and 10% targets; these are **one threshold, not three calibrated operating points**. `threshold_audit.csv` records the exact native-scale thresholds and realized calibration FPR.\n\nFor this fixed-baseline follow-up, 50 normal objects (10/category across the five preselected diagnostic categories) are pooled to fit a global per-model empirical normal-score CDF and fixed 50/50 mean. Calibration object IDs are disjoint from evaluation IDs, but they are reserved objects from the same native test manifests, not an external deployment calibration set. A 10% point corresponds to approximately 5/50 calibration normal objects; 20% to approximately 10/50. Exact counts and 95% Clopper-Pearson intervals appear in `operating_point_metrics.csv`. Scores/thresholds are descriptive for this diagnostic transfer setting. No anomalies were used to fit or select fusion weights.\n')
 # Metric tables and raw score distributions.
 tfr=[]
 for r in metrics:
  if r['category'] in CATS:tfr.append({'category':r['category'],'selection_group':r['selection_group'],**r})
 wr(OUT/'threshold_free_metrics.csv',tfr)
 dist=[]
 for c in CATS:
  for m in ('d2','arcad'):
   rr=[r for r in pairs if r['category']==c]
   for level in ('view','object'):
    items=[]
    if level=='view':items=[(r['y_view'],r[m],r['split']) for r in rr]
    else:items=[(rs[0]['y_object'],rs[0][m+'_object'],rs[0]['split']) for k,rs in groups.items() if k[0]==c]
    for split in ('all_shared','heldout_test','calibration_normal'):
     subset=[(y,s) for y,s,sp in items if split=='all_shared' or sp==split]
     for y0,name in ((0,'normal'),(1,'defect')):
      vals=np.asarray([s for y,s in subset if y==y0],float)
      if len(vals):dist.append({'category':c,'model':m,'level':level,'split':split,'class':name,'n':len(vals),'prevalence':sum(y for y,s in subset)/len(subset),'mean':float(vals.mean()),'std':float(vals.std()),'q05':float(np.quantile(vals,.05)),'q25':float(np.quantile(vals,.25)),'median':float(np.median(vals)),'q75':float(np.quantile(vals,.75)),'q95':float(np.quantile(vals,.95))})
 wr(OUT/'raw_score_distributions.csv',dist)
 # Per-category thresholds, fixed 10/20% pooled normal budgets. Model-score thresholds
 # are native units; fusion thresholds are calibrated percentile scales.
 op=[]; fov=[]; fail=[]; fp_cases=[]
 for alpha,ths in thresholds.items():
  # Define the prediction systems on every view and object in the held-out set.
  systems={'d2':{},'arcad':{},'fixed_50_50_mean':{},'calibrated_OR':{}}
  for r in evalpairs:
   systems['d2'].setdefault('view',{})[r['image_path']]=r['d2']>=ths['d2_view'];systems['arcad'].setdefault('view',{})[r['image_path']]=r['arcad']>=ths['arcad_view'];systems['fixed_50_50_mean'].setdefault('view',{})[r['image_path']]=r['fusion_mean']>=ths['fusion_view'];systems['calibrated_OR'].setdefault('view',{})[r['image_path']]=(r['d2']>=ths['d2_view'] or r['arcad']>=ths['arcad_view'])
  for k,rs in eval_groups.items():
   r=rs[0];systems['d2'].setdefault('object',{})[k]=r['d2_object']>=ths['d2_object'];systems['arcad'].setdefault('object',{})[k]=r['arcad_object']>=ths['arcad_object'];systems['fixed_50_50_mean'].setdefault('object',{})[k]=r['fusion_object_mean']>=ths['fusion_object'];systems['calibrated_OR'].setdefault('object',{})[k]=(r['d2_object']>=ths['d2_object'] or r['arcad_object']>=ths['arcad_object'])
  for level in ('view','object'):
   if level=='view': sample=evalpairs; y=np.asarray([r['y_view'] for r in sample]); keys=[r['image_path'] for r in sample]; normgroups={k:rs for k,rs in eval_groups.items() if rs[0]['y_object']==0}
   else: sample=[rs[0] for k,rs in eval_groups.items()];y=np.asarray([r['y_object'] for r in sample]);keys=[(r['category'],r['object_key']) for r in sample];normgroups={k:rs for k,rs in eval_groups.items() if rs[0]['y_object']==0}
   for system in systems:
    pred=np.asarray([systems[system][level][k] for k in keys],bool);stats=cm(y,pred)
    normal_fp_obj=0
    for k,rs in normgroups.items():
     flag=any(systems[system]['view'][r['image_path']] for r in rs) if level=='view' else systems[system]['object'][k]
     normal_fp_obj+=bool(flag)
    nnormal=len(normgroups);lo,hi=cp(normal_fp_obj,nnormal)
    category_values=sorted(set(r['category'] for r in sample))+['ALL_FIVE']
    for cat in category_values:
     inds=np.asarray([True]*len(sample) if cat=='ALL_FIVE' else [r['category']==cat for r in sample]); st=cm(y[inds],pred[inds]);
     cal_flags=[]
     for rs in cals:
      if level=='view':
       if system=='d2':flag=any(r['d2']>=ths['d2_view'] for r in rs)
       elif system=='arcad':flag=any(r['arcad']>=ths['arcad_view'] for r in rs)
       elif system=='fixed_50_50_mean':flag=any(r['fusion_mean']>=ths['fusion_view'] for r in rs)
       else:flag=any(r['d2']>=ths['d2_view'] or r['arcad']>=ths['arcad_view'] for r in rs)
      else:
       r0=rs[0]
       if system=='d2':flag=r0['d2_object']>=ths['d2_object']
       elif system=='arcad':flag=r0['arcad_object']>=ths['arcad_object']
       elif system=='fixed_50_50_mean':flag=r0['fusion_object_mean']>=ths['fusion_object']
       else:flag=r0['d2_object']>=ths['d2_object'] or r0['arcad_object']>=ths['arcad_object']
      cal_flags.append(flag)
     nh=sum(1 for k in normgroups if cat=='ALL_FIVE' or k[0]==cat); kh=sum((any(systems[system]['view'][r['image_path']] for r in eval_groups[k]) if level=='view' else systems[system]['object'][k]) for k in normgroups if cat=='ALL_FIVE' or k[0]==cat);ci0,ci1=cp(kh,nh)
     op.append({'category':cat,'aggregation':'pooled' if cat=='ALL_FIVE' else 'category','system':system,'level':level,'normal_object_fpr_target':alpha,'calibration_normal_objects':len(cals),'threshold':f'{ths["d2_"+level]} OR {ths["arcad_"+level]}' if system=='calibrated_OR' else ths[('fusion' if system=='fixed_50_50_mean' else system)+('_view' if level=='view' else '_object')],'threshold_scale':'native D2' if system=='d2' else 'native ArcAD' if system=='arcad' else 'empirical-normal-rank mean / OR native thresholds','calibration_object_fp_count':sum(cal_flags),'calibration_realized_object_fpr':sum(cal_flags)/len(cals),'heldout_normal_object_fp_count':kh,'heldout_normal_objects':nh,'heldout_normal_object_fpr_ci95_low':ci0,'heldout_normal_object_fpr_ci95_high':ci1,'positive_recall':st['recall'],'precision':st['precision'],'F1':st['F1'],'confusion_TN':st['TN'],'confusion_FP':st['FP'],'confusion_FN':st['FN'],'confusion_TP':st['TP'],'row_TN_rate':st['TN_rate'],'row_FP_rate':st['FP_rate'],'row_FN_rate':st['FN_rate'],'row_TP_rate':st['TP_rate'],'normal_object_fpr_target_status':'empirical pooled calibration operating point; diagnostic transfer only'})
    # Overall union-FP and paired model failure contingencies.
   for cat in (*CATS,'ALL_FIVE'):
    vr=[r for r in evalpairs if cat=='ALL_FIVE' or r['category']==cat];yy=np.asarray([r['y_view'] for r in vr]);pd=np.asarray([systems['d2']['view'][r['image_path']] for r in vr]);pa=np.asarray([systems['arcad']['view'][r['image_path']] for r in vr]);o=ftable(yy,pd,pa); ng=[rs for k,rs in eval_groups.items() if (cat=='ALL_FIVE' or k[0]==cat) and rs[0]['y_object']==0]; fp_d=sum(any(systems['d2']['view'][r['image_path']] for r in rs) for rs in ng);fp_a=sum(any(systems['arcad']['view'][r['image_path']] for r in rs) for rs in ng);fp_u=sum(any(systems['calibrated_OR']['view'][r['image_path']] for r in rs) for rs in ng)
    fail.append({'category':cat,'level':'view','target':alpha,'threshold_protocol':'global normal-only empirical CDF, fixed alpha; heldout object-disjoint','false_positive_union_objects_d2_arcad_or':fp_u,'normal_objects':len(ng),'D2_false_positive_objects':fp_d,'ArcAD_false_positive_objects':fp_a,**o})
    # object-level failure overlap
    objs=[(k,rs[0]) for k,rs in eval_groups.items() if cat=='ALL_FIVE' or k[0]==cat];yy=np.asarray([r['y_object'] for k,r in objs]);pd=np.asarray([systems['d2']['object'][k] for k,r in objs]);pa=np.asarray([systems['arcad']['object'][k] for k,r in objs]);o=ftable(yy,pd,pa);ng=[k for k,r in objs if r['y_object']==0];fp_d=sum(systems['d2']['object'][k] for k in ng);fp_a=sum(systems['arcad']['object'][k] for k in ng);fp_u=sum(systems['calibrated_OR']['object'][k] for k in ng)
    fail.append({'category':cat,'level':'object','target':alpha,'threshold_protocol':'global normal-only empirical CDF, fixed alpha; heldout object-disjoint','false_positive_union_objects_d2_arcad_or':fp_u,'normal_objects':len(ng),'D2_false_positive_objects':fp_d,'ArcAD_false_positive_objects':fp_a,**o})
  if alpha==.10:
   for r in evalpairs:
    dd=systems['d2']['view'][r['image_path']];aa=systems['arcad']['view'][r['image_path']];ff=systems['fixed_50_50_mean']['view'][r['image_path']];oo=systems['calibrated_OR']['view'][r['image_path']]
    if (r['y_view'] and (not dd or not aa or not ff or dd!=aa or ff!=dd)) or (not r['y_view'] and (dd or aa or ff or oo)):
     fp_cases.append({'level':'view','category':r['category'],'object_key':r['object_key'],'camera_id':r['camera_id'],'image_path':r['image_path'],'gt_label':r['y_view'],'defect_type':r['defect_type'],'d2_score_native':r['d2'],'arcad_score_native':r['arcad'],'d2_normal_rank':r['d2_rank'],'arcad_normal_rank':r['arcad_rank'],'fusion_mean_rank':r['fusion_mean'],'d2_hit':dd,'arcad_hit':aa,'fusion_50_50_hit':ff,'calibrated_OR_hit':oo,'target':'10% pooled normal-object FPR, descriptive transfer calibration'})
   for k,rs in eval_groups.items():
    r=rs[0];dd=systems['d2']['object'][k];aa=systems['arcad']['object'][k];ff=systems['fixed_50_50_mean']['object'][k];oo=systems['calibrated_OR']['object'][k]
    if (r['y_object'] and (not dd or not aa or not ff or dd!=aa or ff!=dd)) or (not r['y_object'] and (dd or aa or ff or oo)):
     fp_cases.append({'level':'object','category':k[0],'object_key':k[1],'gt_label':r['y_object'],'d2_score_native':r['d2_object'],'arcad_score_native':r['arcad_object'],'d2_normal_rank':r['d2_object_rank'],'arcad_normal_rank':r['arcad_object_rank'],'fusion_mean_rank':r['fusion_object_mean'],'d2_hit':dd,'arcad_hit':aa,'fusion_50_50_hit':ff,'calibrated_OR_hit':oo,'target':'10% pooled normal-object FPR, descriptive transfer calibration'})
  # End alpha loop
 wr(OUT/'operating_point_metrics.csv',op)
 wr(OUT/'failure_overlap_vs_fpr.csv',fail)
 wr(OUT/'fusion_failure_cases.csv',fp_cases)
 # Calibration/external heldout exact counts and CP intervals by system/point/level.
 rel=[]
 for alpha,ths in thresholds.items():
  for model in MODELS:
   for level in ('view','object'):
    calpred=[];evalpred=[]
    if level=='view':
     for rs in cals:calpred.append(any(r[model]>=ths[model+'_view'] for r in rs))
     for rs in eval_groups.values():
      if rs[0]['y_object']==0:evalpred.append(any(r[model]>=ths[model+'_view'] for r in rs))
    else:
     calpred=[rs[0][model+'_object']>=ths[model+'_object'] for rs in cals]
     evalpred=[rs[0][model+'_object']>=ths[model+'_object'] for rs in eval_groups.values() if rs[0]['y_object']==0]
    for split,preds in [('calibration',calpred),('heldout_test',evalpred)]:
     k=sum(preds);lo,hi=cp(k,len(preds));rel.append({'model':model,'level':level,'target_normal_object_fpr':alpha,'split':split,'false_positive_object_count':k,'normal_objects':len(preds),'empirical_object_fpr':k/max(1,len(preds)),'binomial_ci95_low':lo,'binomial_ci95_high':hi,'threshold_native':ths[model+('_view' if level=='view' else '_object')],'interpretation':'empirical binomial interval; object is sampling unit'})
  for sysname,calp,evp,th in [('fixed_50_50_mean',calfusion_viewmax,[],ths['fusion_view']),('fixed_50_50_mean_object',calfusion_obj,[],ths['fusion_object'])]:
   level='view' if sysname=='fixed_50_50_mean' else 'object'
   calpred=[]
   if level=='view':calpred=[max(.5*(ecdf(viewcal['d2'],[r['d2']])[0]+ecdf(viewcal['arcad'],[r['arcad']])[0]) for r in rs)>=th for rs in cals];evp=[any(r['fusion_mean']>=th for r in rs) for rs in eval_groups.values() if rs[0]['y_object']==0]
   else:calpred=[v>=th for v in calfusion_obj];evp=[rs[0]['fusion_object_mean']>=th for rs in eval_groups.values() if rs[0]['y_object']==0]
   for split,preds in [('calibration',calpred),('heldout_test',evp)]:
    k=sum(preds);lo,hi=cp(k,len(preds));rel.append({'model':'fixed_50_50_mean','level':level,'target_normal_object_fpr':alpha,'split':split,'false_positive_object_count':k,'normal_objects':len(preds),'empirical_object_fpr':k/max(1,len(preds)),'binomial_ci95_low':lo,'binomial_ci95_high':hi,'threshold_native':th,'interpretation':'rank mean from 50/50 fixed calibration; empirical binomial interval'})
 wr(OUT/'calibration_reliability_counts.csv',rel)
 # Region overlap is inherited from Task02's aligned common grid and 10-normal pixel calibration.
 preg=rd(T2/'paired_region_overlap.csv'); wr(OUT/'region_overlap_vs_fpr.csv',[{'category':r['category'],'image_path':r['image_path'],'object_key':r['object_key'],'camera_id':r['camera_id'],'region_id':r['region_id'],'pixel_fpr_target':r['pixel_fpr_target'],'d2_best_iou':r['d2_best_iou'],'arcad_best_iou':r['arcad_best_iou'],'d2_coverage':r['d2_coverage'],'arcad_coverage':r['arcad_coverage'],'both_iou_0.3':r['both_detect_iou_0.3'],'d2_only_iou_0.3':r['d2_only_detect_iou_0.3'],'arcad_only_iou_0.3':r['arcad_only_detect_iou_0.3'],'neither_iou_0.3':r['neither_detect_iou_0.3'],'protocol_note':'Inherited aligned common-grid region pairing; per-category pixel threshold estimated from 10 normal objects, therefore exploratory FPR budget, not deployment calibration.'} for r in preg])
 # Fixed model baselines vs 50/50 calibrated mean, same object-disjoint heldout records.
 comp=[]
 for cat in (*CATS,'ALL_FIVE'):
  for level in ('view','object'):
   sample=evalpairs if level=='view' else [rs[0] for k,rs in eval_groups.items() if rs[0]['y_object']==1 or rs[0]['y_object']==0]
   if cat!='ALL_FIVE':sample=[r for r in sample if r['category']==cat]
   for system,scorekey,labelkey in [('d2','d2','y_view' if level=='view' else 'y_object'),('arcad','arcad','y_view' if level=='view' else 'y_object'),('fixed_50_50_mean','fusion_mean' if level=='view' else 'fusion_object_mean','y_view' if level=='view' else 'y_object')]:
    y=np.asarray([r[labelkey] for r in sample]);s=np.asarray([r[scorekey] for r in sample]);aa,ap=auc(y,s);comp.append({'category':cat,'level':level,'system':system,'n_samples':len(y),'positive_n':int(y.sum()),'positive_prevalence':float(y.mean()),'AUROC':aa,'AP':ap,'PR_random_baseline_prevalence':float(y.mean()),'fusion_rule':'fixed 50/50 mean after model-specific global empirical normal CDF' if system.startswith('fixed') else 'single model baseline','tuning':'No anomaly-based tuning'})
 for level in ('view','object'):
  for system in ('d2','arcad','fixed_50_50_mean'):
   rr=[r for r in comp if r['category'] in CATS and r['level']==level and r['system']==system]
   comp.append({'category':'MACRO_FIVE','level':level,'system':system,'n_samples':sum(int(r['n_samples']) for r in rr),'positive_n':sum(int(r['positive_n']) for r in rr),'positive_prevalence':float(np.mean([r['positive_prevalence'] for r in rr])),'AUROC':float(np.mean([r['AUROC'] for r in rr])),'AP':float(np.mean([r['AP'] for r in rr])),'PR_random_baseline_prevalence':'category-specific baseline; AP macro mean','fusion_rule':'equal category mean','tuning':'No anomaly-based tuning'})
 wr(OUT/'fusion_comparison.csv',comp)
 # Paired object-cluster bootstrap for category-level view/object AUROC/AP and fusion deltas.
 rng=np.random.default_rng(SEED); bootrows=[]
 for cat in CATS:
  ckeys=sorted(k for k in eval_groups if k[0]==cat)
  for level in ('view','object'):
   for metric in ('AUROC','AP'):
    for sysname,key in [('d2','d2' if level=='view' else 'd2_object'),('arcad','arcad' if level=='view' else 'arcad_object'),('fixed_50_50_mean','fusion_mean' if level=='view' else 'fusion_object_mean')]:
     y0=[];s0=[]
     for k in ckeys:
      rs=eval_groups[k];
      if level=='view':y0.extend(r['y_view'] for r in rs);s0.extend(r[key] for r in rs)
      else:y0.append(rs[0]['y_object']);s0.append(rs[0][key])
     fn=roc_auc_score if metric=='AUROC' else average_precision_score;estimate=fn(y0,s0);vals=[]
     for _ in range(BOOT):
      chosen=rng.choice(len(ckeys),len(ckeys),replace=True); yy=[];ss=[]
      for ix in chosen:
       rs=eval_groups[ckeys[ix]]
       if level=='view':yy.extend(r['y_view'] for r in rs);ss.extend(r[key] for r in rs)
       else:yy.append(rs[0]['y_object']);ss.append(rs[0][key])
      if len(set(yy))==2:vals.append(float(fn(yy,ss)))
     bootrows.append({'category':cat,'level':level,'metric':metric,'system':sysname,'estimate':float(estimate),'ci95_low':float(np.percentile(vals,2.5)),'ci95_high':float(np.percentile(vals,97.5)),'resamples_requested':BOOT,'valid_draws':len(vals),'excluded_single_class_draws':BOOT-len(vals),'seed':SEED,'sampling_unit':'physical object; paired system data and all five views retained'})
    # paired fusion minus both individual models (same resampled objects)
    for baseline,bkey in [('d2','d2' if level=='view' else 'd2_object'),('arcad','arcad' if level=='view' else 'arcad_object')]:
     fn=roc_auc_score if metric=='AUROC' else average_precision_score
     obs=[]
     for syskey in [bkey,'fusion_mean' if level=='view' else 'fusion_object_mean']:
      yy=[];ss=[]
      for k in ckeys:
       rs=eval_groups[k]
       if level=='view':yy.extend(r['y_view'] for r in rs);ss.extend(r[syskey] for r in rs)
       else:yy.append(rs[0]['y_object']);ss.append(rs[0][syskey])
      obs.append(float(fn(yy,ss)))
     vals=[]
     for _ in range(BOOT):
      chosen=rng.choice(len(ckeys),len(ckeys),replace=True);yy=[];s1=[];s2=[]
      for ix in chosen:
       rs=eval_groups[ckeys[ix]]
       if level=='view':yy.extend(r['y_view'] for r in rs);s1.extend(r[bkey] for r in rs);s2.extend(r['fusion_mean'] for r in rs)
       else:yy.append(rs[0]['y_object']);s1.append(rs[0][bkey]);s2.append(rs[0]['fusion_object_mean'])
      if len(set(yy))==2:vals.append(float(fn(yy,s2)-fn(yy,s1)))
     bootrows.append({'category':cat,'level':level,'metric':metric,'system':f'fusion_minus_{baseline}','estimate':obs[1]-obs[0],'ci95_low':float(np.percentile(vals,2.5)),'ci95_high':float(np.percentile(vals,97.5)),'resamples_requested':BOOT,'valid_draws':len(vals),'excluded_single_class_draws':BOOT-len(vals),'seed':SEED,'sampling_unit':'paired physical object; all views retained'})
 wr(OUT/'bootstrap_intervals.csv',bootrows)
 # Top-percent native map score sensitivity without test-based selection.
 tops=[]
 for cat in CATS:
  records=manifest['categories'][cat]['shared_test_images'];
  for model in MODELS:
   if model=='arcad':p=T2/'maps'/f'arcad_{cat}.npz'
   elif cat in ('lattice_block_plug','laser_diode','power_jack'):p=T1/'maps'/f'{cat}.npz'
   else:p=T2/'maps'/f'd2_{cat}.npz'
   with np.load(p) as z:
    mkey='smoothed_maps' if 'smoothed_maps' in z else 'postprocessed_maps';maps=z[mkey].astype(np.float32);paths=z['image_paths'].astype(str).tolist();idx={v:i for i,v in enumerate(paths)}
    if any(r['image_path'] not in idx for r in records):raise RuntimeError(f'missing native map paths {cat} {model}')
    arr=maps[[idx[r['image_path']] for r in records]]
   for pct in (.1,.5,1.,2.,5.):
    scores=[]
    for am in arr:
     v=am.reshape(-1);k=max(1,int(len(v)*pct/100));scores.append(float(np.partition(v,len(v)-k)[-k:].mean()))
    y=np.asarray([r['label'] for r in records]);aa,ap=auc(y,scores);tops.append({'category':cat,'model':model,'score_top_percent':pct,'level':'view','n_views':len(y),'positive_prevalence':float(y.mean()),'I-AUROC':aa,'I-AP':ap,'selection':'prespecified sensitivity grid; top1% remains primary'})
    byob=defaultdict(list)
    for r,s in zip(records,scores):byob[r['object_key']].append((r['label'],s))
    oy=np.asarray([max(x[0] for x in vs) for _,vs in sorted(byob.items())]);oscore=[]
    for key,vs in sorted(byob.items()):
     z=arr[[idx[x['image_path']] for x in records if x['object_key']==key]].reshape(-1);k=max(1,int(len(z)*pct/100));oscore.append(float(np.partition(z,len(z)-k)[-k:].mean()))
    oa,oap=auc(oy,oscore);tops.append({'category':cat,'model':model,'score_top_percent':pct,'level':'object','n_views':len(byob),'positive_prevalence':float(oy.mean()),'O-AUROC':oa,'O-AP':oap,'selection':'prespecified sensitivity grid; top1% remains primary'})
 wr(OUT/'score_topk_sensitivity.csv',tops)
 # Mechanism and camera tables; raw D2/ArcAD score, GT visibility, fixed global scores.
 vd=rd(T2/'view_diagnostics.csv');vlookup={(r['category'],r['model'],r['image_path']):r for r in vd}; cameras=[];strata=defaultdict(list)
 for r in pairs:
  dd=vlookup[(r['category'],'d2',r['image_path'])];aa=vlookup[(r['category'],'arcad',r['image_path'])];
  source=audit_status='';
  cameras.append({'category':r['category'],'object_key':r['object_key'],'camera_id':r['camera_id'],'image_path':r['image_path'],'view_id':r['view_id'],'object_label':r['y_object'],'view_label':r['y_view'],'defect_type':r['defect_type'],'gt_area_fraction':dd['gt_area_fraction'],'mask_nonempty':dd['mask_nonempty'],'d2_raw_score_top1pct':r['d2'],'arcad_raw_score_top1pct':r['arcad'],'d2_global_normal_rank':r['d2_rank'],'arcad_global_normal_rank':r['arcad_rank'],'fixed_half_mean_rank':r['fusion_mean'],'d2_threshold_hit_at_10pct':r['d2']>=thresholds[.1]['d2_view'],'arcad_threshold_hit_at_10pct':r['arcad']>=thresholds[.1]['arcad_view'],'fusion_threshold_hit_at_10pct':r['fusion_mean']>=thresholds[.1]['fusion_view'],'d2_Q':dd['Q_within_view_pixel_AUROC'],'arcad_Q':aa['Q_within_view_pixel_AUROC'],'d2_top1_recall':dd['top1_recall'],'arcad_top1_recall':aa['top1_recall']})
  if r['y_view'] and dd['mask_nonempty']=='True':
   area=float(dd['gt_area_fraction']);size='<0.1%' if area<.001 else '0.1-1%' if area<.01 else '1-5%' if area<.05 else '>5%'
   for model,row in (('d2',dd),('arcad',aa)):
    strata[(r['category'],'size',size,model)].append(row);strata[(r['category'],'defect_type',r['defect_type'],model)].append(row);strata[(r['category'],'camera',r['camera_id'],model)].append(row)
 wr(OUT/'object_camera_consistency.csv',cameras)
 srows=[]
 for (cat,kind,name,model),rs in strata.items():
  srows.append({'category':cat,'stratum_type':kind,'stratum':name,'model':model,'positive_nonempty_views':len(rs),'physical_objects':len(set(r['object_key'] for r in rs)),'mean_area_fraction':float(np.nanmean([float(r['gt_area_fraction']) for r in rs])),'mean_Q':float(np.nanmean([float(r['Q_within_view_pixel_AUROC']) for r in rs])),'mean_C_contrast':float(np.nanmean([float(r['C_muG_minus_muB']) for r in rs])),'mean_Z_contrast':float(np.nanmean([float(r['Z_contrast']) for r in rs])),'mean_top0.1_recall':float(np.nanmean([float(r['top0.1_recall']) for r in rs])),'mean_top0.5_recall':float(np.nanmean([float(r['top0.5_recall']) for r in rs])),'mean_top1_recall':float(np.nanmean([float(r['top1_recall']) for r in rs])),'mean_top2_recall':float(np.nanmean([float(r['top2_recall']) for r in rs])),'mean_top5_recall':float(np.nanmean([float(r['top5_recall']) for r in rs])),'threshold_label':'threshold-free localization strata; calibration target details in operating_point_metrics.csv'})
 wr(OUT/'mechanism_by_stratum.csv',srows)
 # Curves and score distributions.
 fig,axs=plt.subplots(5,2,figsize=(12,19))
 for i,cat in enumerate(CATS):
  for j,level in enumerate(('view','object')):
   ax=axs[i,j]
   for system in ('d2','arcad','fixed_50_50_mean'):
    ss=evalpairs if level=='view' else [rs[0] for k,rs in eval_groups.items() if rs[0]['y_object']==1 or rs[0]['y_object']==0]
    ss=[r for r in ss if r['category']==cat];key={'d2':'d2' if level=='view' else 'd2_object','arcad':'arcad' if level=='view' else 'arcad_object','fixed_50_50_mean':'fusion_mean' if level=='view' else 'fusion_object_mean'}[system]
    y=np.asarray([r['y_view'] if level=='view' else r['y_object'] for r in ss]);s=np.asarray([r[key] for r in ss]);fpr,tpr,_=roc_curve(y,s);ax.plot(fpr,tpr,label=system)
   ax.plot([0,1],[0,1],':',color='gray');ax.set_title(f'{cat} {level} ROC');ax.set_xlabel('FPR');ax.set_ylabel('TPR');ax.legend(fontsize=7)
 fig.tight_layout();fig.savefig(OUT/'plots/roc_by_category.png',dpi=140);plt.close(fig)
 fig,axs=plt.subplots(5,2,figsize=(12,19))
 for i,cat in enumerate(CATS):
  for j,level in enumerate(('view','object')):
   ax=axs[i,j]
   for system in ('d2','arcad','fixed_50_50_mean'):
    ss=evalpairs if level=='view' else [rs[0] for k,rs in eval_groups.items() if rs[0]['y_object']==1 or rs[0]['y_object']==0]
    ss=[r for r in ss if r['category']==cat];key={'d2':'d2' if level=='view' else 'd2_object','arcad':'arcad' if level=='view' else 'arcad_object','fixed_50_50_mean':'fusion_mean' if level=='view' else 'fusion_object_mean'}[system]
    y=np.asarray([r['y_view'] if level=='view' else r['y_object'] for r in ss]);s=np.asarray([r[key] for r in ss]);pr,rc,_=precision_recall_curve(y,s);ax.plot(rc,pr,label=system)
   prevalence=np.mean([r['y_view'] if level=='view' else r['y_object'] for r in ss]);ax.axhline(prevalence,ls=':',color='gray',label='prevalence baseline');ax.set_title(f'{cat} {level} PR');ax.set_xlabel('Recall');ax.set_ylabel('Precision');ax.legend(fontsize=7)
 fig.tight_layout();fig.savefig(OUT/'plots/pr_by_category.png',dpi=140);plt.close(fig)
 # Recall vs measured heldout object FPR at two fixed calibration points.
 rows=rd(OUT/'operating_point_metrics.csv');fig,ax=plt.subplots(figsize=(7,5))
 for system in ('d2','arcad','fixed_50_50_mean','calibrated_OR'):
  rr=[r for r in rows if r['aggregation']=='pooled' and r['category']=='ALL_FIVE' and r['level']=='object' and r['system']==system]
  if rr:
   xx=np.asarray([int(r['heldout_normal_object_fp_count'])/max(1,int(r['heldout_normal_objects'])) for r in rr]);lo=np.asarray([float(r['heldout_normal_object_fpr_ci95_low']) for r in rr]);hi=np.asarray([float(r['heldout_normal_object_fpr_ci95_high']) for r in rr]);yy=np.asarray([float(r['positive_recall']) for r in rr]);ax.errorbar(xx,yy,xerr=np.vstack((xx-lo,hi-xx)),marker='o',capsize=3,label=system)
 ax.set_xlabel('Measured held-out normal-object FPR (95% exact binomial CI)');ax.set_ylabel('Positive object recall');ax.legend();fig.tight_layout();fig.savefig(OUT/'plots/recall_vs_measured_fpr.png',dpi=150);plt.close(fig)
 print('Fusion/metrics complete; calibration objects',len(cals),'heldout objects',len(eval_groups),'bootstrap rows',len(bootrows))
if __name__=='__main__':main()
