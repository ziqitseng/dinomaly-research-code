#!/usr/bin/env python3
"""Paired, object-clustered failure analysis for Task 02 (no training)."""
import csv, hashlib, json, math, os, re, sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from scipy import ndimage
from scipy.stats import rankdata, binomtest
from sklearn.metrics import (roc_auc_score, average_precision_score, precision_recall_curve,
    precision_score, recall_score, f1_score, confusion_matrix, auc)
from skimage import measure

ROOT=Path(__file__).resolve().parents[1]
CATS=['lattice_block_plug','laser_diode','power_jack','LED_indicator','aircraft_model_head']
MODELS=['D2','ArcAD']
TARGETS=[.01,.05,.10]
BOOT=2000
SEED=20261103

def dump_csv(path, rows, fields=None):
    rows=list(rows)
    if not rows:
        path.write_text('')
        return
    fields=fields or list(dict.fromkeys(k for r in rows for k in r))
    with open(path,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(rows)

def f1max(y,s):
    if len(np.unique(y))<2:return float('nan')
    p,r,_=precision_recall_curve(y,s); z=2*p*r/(p+r+1e-12)
    return float(np.nanmax(z[:-1])) if len(z)>1 else float('nan')

def cls_metrics(y,s):
    y=np.asarray(y,dtype=np.uint8);s=np.asarray(s,dtype=float)
    if len(np.unique(y))<2:return {'auroc':float('nan'),'ap':float('nan'),'oracle_max_F1':float('nan'),'positive_prevalence':float(y.mean()) if len(y) else float('nan')}
    return {'auroc':float(roc_auc_score(y,s)),'ap':float(average_precision_score(y,s)),
        'oracle_max_F1':f1max(y,s),'positive_prevalence':float(y.mean())}

def fixed_metrics(y,s,t):
    y=np.asarray(y,dtype=np.uint8); pred=np.asarray(s)>=t
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'threshold':float(t),'TN':int(tn),'FP':int(fp),'FN':int(fn),'TP':int(tp),
        'precision':float(tp/(tp+fp)) if tp+fp else 0.,'recall':float(tp/(tp+fn)) if tp+fn else 0.,
        'F1':float(2*tp/(2*tp+fp+fn)) if 2*tp+fp+fn else 0.,
        'specificity':float(tn/(tn+fp)) if tn+fp else 0.,
        'FPR':float(fp/(tn+fp)) if tn+fp else 0.,
        'row_normalized_TNR':float(tn/(tn+fp)) if tn+fp else 0.,
        'row_normalized_FPR':float(fp/(tn+fp)) if tn+fp else 0.,
        'row_normalized_FNR':float(fn/(fn+tp)) if fn+tp else 0.,
        'row_normalized_TPR':float(tp/(fn+tp)) if fn+tp else 0.}

def topk_score(a):
    a=np.asarray(a,dtype=np.float32).reshape(-1);k=max(1,int(a.size*.01))
    return float(np.partition(a,-k)[-k:].mean())

def fast_pro(masks,maps,nth=200,cap=.3):
    """Histogram-equivalent acceleration of ArcAD compute_pro; 8-connected regions."""
    masks=(np.asarray(masks)>0).astype(np.uint8);maps=np.asarray(maps,dtype=np.float32)
    lo,hi=float(maps.min()),float(maps.max());norm=(maps-lo)/(hi-lo) if hi-lo>1e-8 else np.zeros_like(maps)
    th=np.linspace(0,1,nth);neg_hist=np.zeros(nth+1,dtype=np.int64);totalneg=0;region_hists=[]
    for m,a in zip(masks,norm):
        lab,n=ndimage.label(m,structure=np.ones((3,3),dtype=np.uint8));mf=m.ravel().astype(bool);lf=lab.ravel();af=a.ravel()
        nb=np.searchsorted(th,af[~mf],side='left');neg_hist+=np.bincount(nb,minlength=nth+1);totalneg+=int((~mf).sum())
        if n:
            pos=lf>0;bins=np.searchsorted(th,af[pos],side='left');ids=lf[pos].astype(np.int64)-1
            region_hists.append(np.bincount(ids*(nth+1)+bins,minlength=n*(nth+1)).reshape(n,nth+1))
    if not region_hists or not totalneg:return float('nan')
    hist=np.concatenate(region_hists,axis=0);areas=hist.sum(axis=1)
    pros=np.array([(hist[:,j+1:].sum(axis=1)/areas).mean() for j in range(nth)])
    fprs=np.array([neg_hist[j+1:].sum()/totalneg for j in range(nth)])
    order=np.argsort(fprs);fprs,pros=fprs[order],pros[order];keep=fprs<=cap
    if keep.sum()<2 or fprs[keep].max()<=0:return 0.
    return float(auc(fprs[keep]/cap,pros[keep]))

def map_grid(device='cpu'):
    y,x=torch.meshgrid(torch.arange(256,dtype=torch.float32),torch.arange(256,dtype=torch.float32),indexing='ij')
    sx=.875*(x+.5)+15.5;sy=.875*(y+.5)+15.5
    return torch.stack([2*(sx+.5)/256-1,2*(sy+.5)/256-1],-1).unsqueeze(0).to(device)

def align_d2(a,grid):
    t=torch.from_numpy(np.asarray(a,dtype=np.float32)).view(1,1,256,256)
    return F.grid_sample(t,grid,mode='bilinear',padding_mode='border',align_corners=False)[0,0].numpy()

def common_mask(path):
    if not path:return np.zeros((256,256),dtype=np.uint8)
    from torchvision import transforms
    from torchvision.transforms import InterpolationMode
    tf=transforms.Compose([transforms.Resize((448,448),interpolation=InterpolationMode.NEAREST),
        transforms.CenterCrop(392),transforms.Resize((256,256),interpolation=InterpolationMode.NEAREST)])
    return (np.asarray(tf(Image.open(path).convert('L')))>0).astype(np.uint8)

def local_luminance(image_path,gt):
    im=Image.open(image_path).convert('L')
    from torchvision import transforms
    crop=transforms.Compose([transforms.Resize((448,448)),transforms.CenterCrop(392),transforms.Resize((256,256))])(im)
    lum=np.asarray(crop,dtype=np.float32)/255.
    ring=ndimage.binary_dilation(gt.astype(bool),iterations=5)&~gt.astype(bool)
    if not gt.any() or not ring.any():return float('nan')
    return float(lum[gt>0].mean()-lum[ring].mean())

def percentile(v,bg):
    bg=np.sort(np.asarray(bg,dtype=float));v=np.asarray(v,dtype=float)
    left=np.searchsorted(bg,v,side='left');right=np.searchsorted(bg,v,side='right')
    return (left+.5*(right-left))/max(1,len(bg))

def map_diagnostics(a,gt):
    a=np.asarray(a,dtype=np.float32);g=np.asarray(gt,dtype=bool);n=a.size;area=int(g.sum());bg=~g
    if not area or not bg.any():
        return {'gt_area_fraction':area/n,'gt_region_count':int(measure.label(g,connectivity=2).max()),
            'C':float('nan'),'Z':float('nan'),'Q':float('nan'),'GT_max_percentile':float('nan'),
            'GT_mean_percentile':float('nan'),'top_k':{},'off_target_top1_fraction':float('nan')}
    mg=float(a[g].mean());mb=float(a[bg].mean());sb=float(a[bg].std())
    q=float(roc_auc_score(g.ravel(),a.ravel()))
    flat=a.ravel();order=np.argsort(-flat,kind='stable');top={};
    for frac in [.001,.005,.01,.02,.05]:
        k=max(1,int(round(n*frac)));ids=order[:k];prec=float(g.ravel()[ids].mean());rec=float(g.ravel()[ids].sum()/area)
        top[f'{frac:g}']={'k':k,'precision':prec,'recall':rec}
    maxp=float(percentile(a[g].max(),a[bg]))
    meanp=float(percentile(a[g],a[bg]).mean())
    return {'gt_area_fraction':area/n,'gt_region_count':int(measure.label(g,connectivity=2).max()),
        'C':mg-mb,'Z':(mg-mb)/(sb+1e-12),'Q':q,'GT_max_percentile':maxp,
        'GT_mean_percentile':meanp,'top_k':top,'off_target_top1_fraction':1-top['0.01']['precision']}

def region_geometry(gt):
    lab=measure.label(np.asarray(gt)>0,connectivity=2)
    out=[]
    for rp in measure.regionprops(lab):
        minr,minc,maxr,maxc=rp.bbox;area=int(rp.area);per=float(rp.perimeter)
        out.append({'region_id':int(rp.label),'area_px':area,'bbox_width':int(maxc-minc),
            'bbox_height':int(maxr-minr),'aspect_ratio':float((maxc-minc)/max(1,maxr-minr)),
            'perimeter':per,'compactness':float(4*np.pi*area/(per*per)) if per else float('nan'),
            'centroid_y':float(rp.centroid[0]),'centroid_x':float(rp.centroid[1]),
            'border_distance_px':float(min(minr,minc,255-maxr,255-maxc))})
    return out

def region_eval(gt,pred):
    gl=measure.label(np.asarray(gt)>0,connectivity=2);pl=measure.label(np.asarray(pred)>0,connectivity=2)
    out=[]
    for rp in measure.regionprops(gl):
        gm=gl==rp.label;best_iou=0.;bestcov=0.;anyhit=False
        candidates=np.unique(pl[gm]);candidates=candidates[candidates>0]
        for pid in candidates:
            pm=pl==pid;inter=int(np.logical_and(gm,pm).sum());union=int(np.logical_or(gm,pm).sum());iou=inter/union if union else 0.
            cov=inter/int(gm.sum());best_iou=max(best_iou,iou);bestcov=max(bestcov,cov);anyhit|=inter>0
        out.append({'gt_region_id':int(rp.label),'best_pred_iou':best_iou,'coverage':bestcov,
            'any_overlap':int(anyhit),'detected_iou_0_1':int(best_iou>=.1),'detected_iou_0_3':int(best_iou>=.3),
            'detected_iou_0_5':int(best_iou>=.5),'gt_area_px':int(rp.area)})
    fp_regions=max(0,int(pl.max())-len(np.unique(pl[gl>0][pl[gl>0]>0])))
    return out,fp_regions

def main():
    out=Path(__file__).resolve().parent; (out/'tables').mkdir(exist_ok=True)
    manifest=json.loads((out/'shared_test_manifest.json').read_text())
    bycat={c:sorted([r for r in manifest if r['category']==c],key=lambda r:r['image_path']) for c in CATS}
    view_rows=[]; object_rows=[]; region_rows=[]
    for cat in CATS:
        z=np.load(out/'maps/common'/f'{cat}.npz',allow_pickle=False)
        a=z['arcad_maps'];d=z['d2_maps'];g=z['gt_masks'].astype(bool); paths=z['image_paths'].tolist()
        rs=bycat[cat]
        if paths != [r['image_path'] for r in rs]: raise ValueError(f'{cat} common map manifest mismatch')
        for i,r in enumerate(rs):
            scores={'D2':topk_score(d[i]),'ArcAD':topk_score(a[i])}
            diag={m:map_diagnostics(v,g[i]) for m,v in [('D2',d[i]),('ArcAD',a[i])]} if g[i].any() else {}
            row={'category':cat,'image_path':r['image_path'],'object_key':r['object_key'],'view_id':r['camera_view_id'],'label':int(r['label']),
                 'defect_type':r.get('anomaly_class','') or '', 'gt_area_px':int(g[i].sum()),
                 'D2_common_top1':scores['D2'],'ArcAD_common_top1':scores['ArcAD'],
                 'D2_common_gt_region_count':diag.get('D2',{}).get('gt_region_count',0),
                 'ArcAD_common_gt_region_count':diag.get('ArcAD',{}).get('gt_region_count',0)}
            for model,mp in [('D2',d[i]),('ArcAD',a[i])]:
                row[model+'_common_gt_max']=float(mp[g[i]].max()) if g[i].any() else float('nan')
                row[model+'_common_gt_mean']=float(mp[g[i]].mean()) if g[i].any() else float('nan')
                if g[i].any():
                    for k,v in diag[model].items():
                        if k!='top_k' and np.isscalar(v): row[model+'_'+k]=v
                    for frac,v in diag[model]['top_k'].items():
                        row[f'{model}_top{frac}_precision']=v['precision']; row[f'{model}_top{frac}_recall']=v['recall']
            view_rows.append(row)
            if g[i].any():
                for j,geo in enumerate(region_geometry(g[i])):
                    region_rows.append({'category':cat,'image_path':r['image_path'],'object_key':r['object_key'],'view_id':r['camera_view_id'],**geo})
        # pooled object score is top 1% over all five maps, not mean view score
        objs=defaultdict(list)
        for i,r in enumerate(rs): objs[r['object_key']].append(i)
        for obj,ix in objs.items():
            pos=any(rs[i]['label'] for i in ix)
            row={'category':cat,'object_key':obj,'positive':int(pos),'views':len(ix),'positive_views':sum(rs[i]['label'] for i in ix),
                 'D2_object_top1':topk_score(d[ix]),'ArcAD_object_top1':topk_score(a[ix]),
                 'D2_view_score_mean':float(np.mean([topk_score(d[i]) for i in ix])),
                 'ArcAD_view_score_mean':float(np.mean([topk_score(a[i]) for i in ix]))}
            object_rows.append(row)
    # In-sample normal thresholds at view/object level; pixel thresholds use calibration maps if present.
    calib_path=out/'calibration_scores.json'
    if not calib_path.exists(): raise RuntimeError('Missing model-specific in-sample training-normal calibration distributions')
    cal=json.loads(calib_path.read_text())
    threshold_rows=[]
    for cat in CATS:
        for model in MODELS:
            # Normal-object operating point for applying one threshold to all constituent views.
            vals=np.asarray(cal[cat][model]['max_view_scores'],float)
            for target in TARGETS:
                q=float(np.quantile(vals,1-target,method='higher'))
                threshold_rows.append({'category':cat,'model':model,'level':'view','target_normal_fpr':target,'threshold':q,'calibration_objects':len(vals),'calibration_views':cal[cat][model]['views'],'source':'max view score per native training-normal object, common crop; in-sample'})
            vals=np.asarray(cal[cat][model]['object_scores'],float)
            for target in TARGETS:
                q=float(np.quantile(vals,1-target,method='higher'))
                threshold_rows.append({'category':cat,'model':model,'level':'object','target_normal_fpr':target,'threshold':q,'calibration_objects':len(vals),'calibration_views':cal[cat][model]['views'],'source':'pooled five-view common-crop score per native training-normal object; in-sample'})
    dump_csv(out/'tables/view_predictions.csv',view_rows);dump_csv(out/'tables/object_predictions.csv',object_rows);dump_csv(out/'tables/gt_regions.csv',region_rows);dump_csv(out/'tables/thresholds.csv',threshold_rows)
    # Category metrics and fixed threshold failure overlap.
    metric_rows=[]; overlap_rows=[]
    for cat in CATS:
        vr=[r for r in view_rows if r['category']==cat]; y=np.array([r['label'] for r in vr]);
        for model in MODELS:
            s=np.array([r[model+'_common_top1'] for r in vr]); metric_rows.append({'category':cat,'model':model,**cls_metrics(y,s)})
        for target in TARGETS:
            th={m:next(x['threshold'] for x in threshold_rows if x['category']==cat and x['model']==m and x['target_normal_fpr']==target) for m in MODELS}
            yd=y.astype(bool); pd2=np.array([r['D2_common_top1']>=th['D2'] for r in vr]); pa=np.array([r['ArcAD_common_top1']>=th['ArcAD'] for r in vr])
            b=int(np.sum(pd2&yd));a=int(np.sum(pa&yd));both=int(np.sum(pd2&pa&yd));donly=int(np.sum(pd2&~pa&yd));aonly=int(np.sum(pa&~pd2&yd));miss=int(np.sum(~pd2&~pa&yd))
            discord=donly+aonly; pval=float(binomtest(min(donly,aonly),discord,.5).pvalue) if discord else 1.
            overlap_rows.append({'category':cat,'target_normal_fpr':target,'positive_views':int(yd.sum()),'D2_detected':b,'ArcAD_detected':a,'both_detected':both,'D2_only':donly,'ArcAD_only':aonly,'both_missed':miss,'oracle_union_detected':b+a-both,'oracle_union_recall':(b+a-both)/max(1,int(yd.sum())),'failure_jaccard':miss/max(1,miss+donly+aonly),'mcnemar_exact_p':pval,'D2_normal_fp':int(np.sum(pd2&~yd)),'ArcAD_normal_fp':int(np.sum(pa&~yd)),'normal_fp_overlap':int(np.sum(pd2&pa&~yd))})
    dump_csv(out/'tables/classification_metrics.csv',metric_rows);dump_csv(out/'tables/detection_overlap.csv',overlap_rows)
    # Paired object bootstrap for AUROC/AP and model deltas; common object IDs sampled within category.
    rng=np.random.default_rng(SEED); boot_rows=[]
    for cat in CATS:
        vr=[r for r in view_rows if r['category']==cat]; ids=sorted({r['object_key'] for r in vr}); groups={o:[r for r in vr if r['object_key']==o] for o in ids}
        for rep in range(BOOT):
            sampled=rng.choice(ids,len(ids),replace=True); sample=[r for o in sampled for r in groups[o]]; yy=np.array([r['label'] for r in sample])
            if len(np.unique(yy))<2: continue
            met={m:cls_metrics(yy,np.array([r[m+'_common_top1'] for r in sample])) for m in MODELS}
            ogroups={r['object_key']:r for r in object_rows if r['category']==cat}
            osample=[ogroups[o] for o in sampled]; oy=np.array([r['positive'] for r in osample])
            omet={m:cls_metrics(oy,np.array([r[m+'_object_top1'] for r in osample])) for m in MODELS}
            boot_rows.append({'category':cat,'replicate':rep,'objects_drawn':len(sampled),
                'D2_auroc':met['D2']['auroc'],'ArcAD_auroc':met['ArcAD']['auroc'],'delta_auroc_D2_minus_ArcAD':met['D2']['auroc']-met['ArcAD']['auroc'],
                'D2_ap':met['D2']['ap'],'ArcAD_ap':met['ArcAD']['ap'],'delta_ap_D2_minus_ArcAD':met['D2']['ap']-met['ArcAD']['ap'],
                'D2_object_auroc':omet['D2']['auroc'],'ArcAD_object_auroc':omet['ArcAD']['auroc'],'delta_object_auroc_D2_minus_ArcAD':omet['D2']['auroc']-omet['ArcAD']['auroc'],
                'D2_object_ap':omet['D2']['ap'],'ArcAD_object_ap':omet['ArcAD']['ap'],'delta_object_ap_D2_minus_ArcAD':omet['D2']['ap']-omet['ArcAD']['ap']})
    dump_csv(out/'tables/bootstrap_replicates.csv',boot_rows)
    ci=[]
    for cat in CATS:
        rr=[r for r in boot_rows if r['category']==cat]
        for metric in ['delta_auroc_D2_minus_ArcAD','delta_ap_D2_minus_ArcAD','delta_object_auroc_D2_minus_ArcAD','delta_object_ap_D2_minus_ArcAD']:
            v=[r[metric] for r in rr];ci.append({'category':cat,'metric':metric,'replicates':len(v),'ci95_low':float(np.quantile(v,.025)),'ci95_high':float(np.quantile(v,.975))})
    dump_csv(out/'tables/bootstrap_ci.csv',ci)
    # Per-category object/view consistency and GT region summary.
    consistency=[]
    for cat in CATS:
        vr=[r for r in view_rows if r['category']==cat]
        for model in MODELS:
            p=[r for r in vr if r['label'] and r[model+'_common_top1']>=next(x['threshold'] for x in threshold_rows if x['category']==cat and x['model']==model and x['target_normal_fpr']==.05)]
            by=defaultdict(list)
            for r in p:by[r['object_key']].append(r)
            consistency.append({'category':cat,'model':model,'positive_views_detected_at_5pct':len(p),'distinct_positive_objects_detected':len(by),'objects_with_multiple_detected_views':sum(len(v)>1 for v in by.values()),'mean_detected_views_per_detected_object':float(np.mean([len(v) for v in by.values()])) if by else 0.})
    dump_csv(out/'tables/multiview_consistency.csv',consistency)
    # Compact, fully paired examples ranked by disagreement and miss confidence.
    rank=[]
    for r in view_rows:
        rank.append({**r,'score_gap_abs':abs(r['D2_common_top1']-r['ArcAD_common_top1']),'example_type':('shared_miss' if r['label'] and r['D2_common_top1']<0 and r['ArcAD_common_top1']<0 else 'positive')})
    dump_csv(out/'tables/paired_view_rows.csv',view_rows)
    report=['# Task 2: paired failure analysis','', '## Scope and validity','',
      'All test comparisons use the shared manifest and aligned central 896×896 source field. D2 maps are bilinearly mapped to the ArcAD 392-crop frame; ground truth uses nearest-neighbor masks. No training was run. ArcAD inputs are the newly initialized protoinit model only; no content from `legacy/` was used.', '',
      'Normal thresholds are exploratory in-sample references from each model’s own training normals; ArcAD has only 7–8 normal objects per category, while D2 has 25. They are not independent calibration and must not be treated as deployment FPR guarantees. No ensemble threshold or tuned ensemble performance is reported. The oracle union is an upper bound only.', '', '## Paired ranking results','',
      '|Category|D2 I-AUROC|ArcAD I-AUROC|D2 AP|ArcAD AP|Positive views|', '|---|---:|---:|---:|---:|---:|']
    for cat in CATS:
        mm={r['model']:r for r in metric_rows if r['category']==cat}; report.append(f"|{cat}|{mm['D2'].get('auroc',float('nan')):.4f}|{mm['ArcAD'].get('auroc',float('nan')):.4f}|{mm['D2'].get('ap',float('nan')):.4f}|{mm['ArcAD'].get('ap',float('nan')):.4f}|{sum(r['category']==cat and r['label'] for r in view_rows)}|")
    report+=['','## Detection overlap at exploratory 5% normal threshold','','|Category|D2 only|ArcAD only|Both detected|Both missed|Union recall|','|---|---:|---:|---:|---:|---:|']
    for r in overlap_rows:
        if r['target_normal_fpr']==.05:report.append(f"|{r['category']}|{r['D2_only']}|{r['ArcAD_only']}|{r['both_detected']}|{r['both_missed']}|{r['oracle_union_recall']:.3f}|")
    report+=['','## Interpretation limits','','The low-three categories were selected for prior failure, so category-level findings are selection-biased. LED_indicator and aircraft_model_head were included as broader controls; their outcomes are associations, not evidence of generalization. Per-view scores are pooled top-1% map values. Anomaly maps are normalized independently in illustrative panels and are not comparable by color intensity. The pixel and region analyses describe spatial behavior within the common field, not an external localization benchmark.','', '## Reproducibility','','Run `python run_analysis.py` from this folder after the map and calibration artifacts exist. Bootstrap resamples are clustered by object (2,000 replicates); view observations from one object remain together.']
    (out/'analysis_report.md').write_text('\n'.join(report)+'\n')
    print(json.dumps({'status':'complete','views':len(view_rows),'objects':len(object_rows),'regions':len(region_rows),'bootstrap_rows':len(boot_rows),'report':str(out/'analysis_report.md')},indent=2))

if __name__=='__main__': main()

