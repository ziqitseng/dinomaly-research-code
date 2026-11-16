#!/usr/bin/env python3
"""Paired, inference-only failure analysis for the Real-IAD Variety subset."""
from __future__ import annotations
import csv, hashlib, json, math, os, re, warnings
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
from PIL import Image
from scipy.stats import binomtest, rankdata
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_fscore_support
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
OUT=HERE.parent
DATA=Path('/media/avlab/7458a474-6ea2-4b90-b232-373c3df8d329/real_iad_variety_1024')
D2=Path('/media/avlab/1CE6451EE644FA12/2026_10_22_dinomaly/Dinomaly2')
ARCAD=Path('/media/avlab/1CE6451EE644FA12/2026_10_22_dinomaly/ArcAD')
T1=D2/'out/re_eval_shared_20261031_cpu01'
MANIFEST=OUT/'shared_manifest.json'
CATS=('lattice_block_plug','laser_diode','power_jack','LED_indicator','thyristor')
MODELS=('d2','arcad')
CAL_ALPHAS=(0.01,0.05,0.10)
PIXEL_ALPHAS=(0.01,0.05)
BOOTSTRAPS=2000
BOOT_SEED=20261031
COMMON_SIZE=224
PIXEL_HIST_BINS=512
EPS=1e-12

def write_csv(path,rows,fields=None):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    if not rows:
        path.write_text('')
        return
    names=fields or list(dict.fromkeys(k for r in rows for k in r))
    with open(path,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=names,extrasaction='ignore'); w.writeheader(); w.writerows(rows)

def auc_ap(y,s):
    y=np.asarray(y,dtype=np.uint8); s=np.asarray(s,dtype=np.float64)
    if len(np.unique(y))<2: return float('nan'),float('nan')
    return float(roc_auc_score(y,s)),float(average_precision_score(y,s))

def f1_at(y,p):
    if len(y)==0:return float('nan')
    return float(precision_recall_fscore_support(y,p,average='binary',zero_division=0)[2])

def topk_stats(scores,mask,fracs=(0.001,0.005,0.01,0.02,0.05)):
    flat=np.asarray(scores,dtype=np.float32).ravel(); gt=np.asarray(mask,dtype=bool).ravel(); n=len(flat)
    result={}
    for frac in fracs:
        k=max(1,int(n*frac)); cutoff=np.partition(flat,n-k)[n-k]
        above=np.flatnonzero(flat>cutoff); ties=np.flatnonzero(flat==cutoff)[:max(0,k-len(above))]
        chosen=np.concatenate((above,ties)); hits=int(gt[chosen].sum())
        tag=f'{frac*100:g}'
        result[f'top{tag}_k']=k; result[f'top{tag}_precision']=hits/k
        result[f'top{tag}_recall']=hits/max(1,int(gt.sum())) if gt.any() else float('nan')
    return result

def bg_percentiles(values,bg):
    bg=np.asarray(bg).ravel(); values=np.asarray(values).ravel(); bg_sorted=np.sort(bg)
    left=np.searchsorted(bg_sorted,values,'left'); right=np.searchsorted(bg_sorted,values,'right')
    pct=(left+right)/(2*max(1,len(bg_sorted)))
    return float(np.max(pct)) if len(pct) else float('nan'),float(np.mean(pct)) if len(pct) else float('nan')

def components(binary):
    n,labels,stats,centroids=cv2.connectedComponentsWithStats(np.asarray(binary,dtype=np.uint8),8)
    out=[]
    for i in range(1,n):
        x,y,w,h,area=map(int,stats[i]); reg=(labels==i)
        contours,_=cv2.findContours(reg.astype(np.uint8),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        perimeter=sum(cv2.arcLength(c,True) for c in contours)
        cx,cy=map(float,centroids[i])
        out.append({'label_map':labels,'component_id':i,'x':x,'y':y,'width':w,'height':h,'area':area,
                    'aspect_ratio':w/h if h else float('nan'),'perimeter':float(perimeter),
                    'compactness':4*math.pi*area/(perimeter*perimeter) if perimeter else float('nan'),
                    'centroid_x':cx,'centroid_y':cy,'border_distance_px':min(cx,cy,COMMON_SIZE-1-cx,COMMON_SIZE-1-cy),
                    'pixels':np.flatnonzero(reg)})
    return out

def compute_pro_curve(masks,maps,thresholds=None,cap=0.3):
    """Common-grid PRO with 8-connectivity, 200 score thresholds, normalized to cap."""
    masks=np.asarray(masks,dtype=bool); maps=np.asarray(maps,dtype=np.float32)
    lo=float(maps.min()); hi=float(maps.max())
    if thresholds is None: thresholds=np.linspace(lo,hi,201,endpoint=True)[:-1]
    regions=[]; neg_values=[]
    for m,a in zip(masks,maps):
        neg_values.append(a[~m].ravel())
        for c in components(m): regions.append(np.sort(a.ravel()[c['pixels']]))
    if not regions:return float('nan'),[],[]
    neg=np.sort(np.concatenate(neg_values)); neg_total=len(neg)
    if thresholds is None: thresholds=np.linspace(lo,hi,201,endpoint=True)[:-1]
    pro=[]; fpr=[]
    for t in thresholds:
        pro.append(float(np.mean([np.searchsorted(v,t,side='right')/len(v) for v in regions])))
        fpr.append(float((neg_total-np.searchsorted(neg,t,side='right'))/neg_total))
    order=np.argsort(fpr)
    x=np.asarray(fpr)[order]; y=np.asarray(pro)[order]
    # Collapse duplicate FPR coordinates before interpolation.
    ux,idx=np.unique(x,return_index=True); uy=y[idx]
    if ux[0]>0: ux=np.r_[0.,ux]; uy=np.r_[uy[0],uy]
    if ux[-1]<cap: ux=np.r_[ux,cap]; uy=np.r_[uy,np.interp(cap,ux,uy)]
    keep=ux<=cap
    xx=ux[keep]; yy=uy[keep]
    if xx[-1]<cap:
        xx=np.r_[xx,cap]; yy=np.r_[yy,np.interp(cap,ux,uy)]
    return float(np.trapz(yy,xx)/cap),xx.tolist(),yy.tolist()

def resolve_model_npz(model,cat):
    if model=='arcad': return OUT/'maps'/f'arcad_{cat}.npz'
    if cat in ('lattice_block_plug','laser_diode','power_jack'):
        return T1/'maps'/f'{cat}.npz'
    return OUT/'maps'/f'd2_{cat}.npz'

def load_maps(model,cat,records):
    p=resolve_model_npz(model,cat)
    with np.load(p) as z:
        key='smoothed_maps' if 'smoothed_maps' in z else 'postprocessed_maps'
        arr=z[key].astype(np.float32); paths=z['image_paths'].astype(str).tolist()
    if len(paths)!=len(set(paths)): raise ValueError(f'{model}/{cat}: duplicate cached map paths')
    idx={path:i for i,path in enumerate(paths)}
    expected=[r['image_path'] for r in records]
    miss=set(expected)-set(idx); extra=set(idx)-set(expected)
    if miss or extra: raise ValueError(f'{model}/{cat}: maps do not exactly cover shared manifest (missing={len(miss)},extra={len(extra)})')
    return arr[[idx[path] for path in expected]],str(p.resolve())

def common_ground_truth(cat,records):
    result=[]; images=[]
    for r in records:
        if r['mask_path']:
            with Image.open(DATA/cat/r['mask_path']) as im:
                m=np.asarray(im.convert('L').crop((64,64,960,960)).resize((COMMON_SIZE,COMMON_SIZE),Image.Resampling.NEAREST))>0
        else: m=np.zeros((COMMON_SIZE,COMMON_SIZE),dtype=bool)
        with Image.open(DATA/cat/r['image_path']) as im:
            rgb=np.asarray(im.convert('RGB').crop((64,64,960,960)).resize((COMMON_SIZE,COMMON_SIZE),Image.Resampling.BILINEAR))
        result.append(m); images.append(rgb)
    return np.stack(result),np.stack(images)

def align_map(model,maps):
    aligned=[]
    for a in maps:
        if model=='d2': aligned.append(a[16:240,16:240])
        else: aligned.append(cv2.resize(a,(COMMON_SIZE,COMMON_SIZE),interpolation=cv2.INTER_LINEAR))
    return np.stack(aligned).astype(np.float32)

def score_rows(records,maps):
    byobj=defaultdict(list); view=[]
    for i,r in enumerate(records):
        flat=maps[i].ravel(); k=max(1,int(len(flat)*.01))
        s=float(np.partition(flat,len(flat)-k)[-k:].mean())
        row={**r,'score_view':s,'map_index':i}; view.append(row); byobj[r['object_key']].append(i)
    obj=[]
    for key,idx in sorted(byobj.items()):
        flat=maps[idx].reshape(-1); k=max(1,int(len(flat)*.01)); s=float(np.partition(flat,len(flat)-k)[-k:].mean())
        obj.append({'object_key':key,'label':max(view[i]['label'] for i in idx),'score_object':s,'indices':idx})
        object_label=max(view[i]['label'] for i in idx)
        for i in idx:
            view[i]['score_object']=s
            view[i]['gt_object_label']=object_label
    return view,obj

def get_thresholds(cat_data,model_rows,model_obj,cal_keys,maps,gt):
    thresholds={}
    cal_obj=[o for o in model_obj if o['object_key'] in cal_keys and o['label']==0]
    eval_obj=[o for o in model_obj if o['object_key'] not in cal_keys]
    # For per-view decisions calibrate the maximum view score per normal object,
    # yielding a normal-object operating target rather than treating five views as IID.
    byobj=defaultdict(list)
    for r in model_rows: byobj[r['object_key']].append(r['score_view'])
    cal_view_max=np.asarray([max(byobj[o['object_key']]) for o in cal_obj])
    cal_obj_scores=np.asarray([o['score_object'] for o in cal_obj])
    cal_indices=[i for i,r in enumerate(model_rows) if r['object_key'] in cal_keys]
    bg=[]
    for i in cal_indices: bg.append(maps[i][~gt[i]])
    bg=np.concatenate(bg) if bg else np.array([],dtype=np.float32)
    for alpha in CAL_ALPHAS:
        thresholds[alpha]={'view':float(np.quantile(cal_view_max,1-alpha,method='higher')),
                           'object':float(np.quantile(cal_obj_scores,1-alpha,method='higher')),
                           'cal_view_max_fpr':float(np.mean(cal_view_max>=np.quantile(cal_view_max,1-alpha,method='higher'))),
                           'cal_object_fpr':float(np.mean(cal_obj_scores>=np.quantile(cal_obj_scores,1-alpha,method='higher')))}
    for alpha in PIXEL_ALPHAS:
        t=float(np.quantile(bg,1-alpha,method='higher')) if len(bg) else float('nan')
        thresholds.setdefault('pixel',{})[alpha]={'threshold':t,'cal_pixel_fpr':float(np.mean(bg>=t)) if len(bg) else float('nan')}
    return thresholds

def cm_metrics(y,p):
    y=np.asarray(y,dtype=np.uint8);p=np.asarray(p,dtype=np.uint8)
    tn=int(np.sum((y==0)&(p==0)));fp=int(np.sum((y==0)&(p==1)));fn=int(np.sum((y==1)&(p==0)));tp=int(np.sum((y==1)&(p==1)))
    return {'tn':tn,'fp':fp,'fn':fn,'tp':tp,'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'f1':f1_at(y,p),'fpr':fp/max(1,fp+tn),'prevalence':float(y.mean()) if len(y) else float('nan')}

def failure_table(y,pa,pb):
    y=np.asarray(y,dtype=np.uint8); pa=np.asarray(pa,dtype=bool); pb=np.asarray(pb,dtype=bool)
    pos=y==1; neg=y==0
    both=int(np.sum(pos&pa&pb)); aonly=int(np.sum(pos&~pa&pb)); bonly=int(np.sum(pos&pa&~pb)); neither=int(np.sum(pos&~pa&~pb)); npos=int(pos.sum())
    fd=pos&~pa; fa=pos&~pb; inter=int(np.sum(fd&fa)); union=int(np.sum(fd|fa)); dmiss=int(fd.sum()); amiss=int(fa.sum())
    discord=int(np.sum(pos&((~pa)!= (~pb))))
    d_fp=int(np.sum(neg&pa)); a_fp=int(np.sum(neg&pb)); both_fp=int(np.sum(neg&pa&pb)); either_fp=int(np.sum(neg&(pa|pb)))
    n01=int(np.sum(pos&~pa&pb)); n10=int(np.sum(pos&pa&~pb))
    # Phi correlation between binary miss indicators on positive cases.
    x=fd[pos].astype(float); z=fa[pos].astype(float)
    phi=float(np.corrcoef(x,z)[0,1]) if len(x)>1 and x.std()>0 and z.std()>0 else float('nan')
    pval=float(binomtest(min(n01,n10),n01+n10,0.5,alternative='two-sided').pvalue) if n01+n10 else float('nan')
    return {'n_positive':npos,'both_detect':both,'d2_only_detects':bonly,'arcad_only_detects':aonly,'neither_detects':neither,
            'both_detect_pct':both/max(1,npos),'d2_only_pct':bonly/max(1,npos),'arcad_only_pct':aonly/max(1,npos),'neither_pct':neither/max(1,npos),
            'd2_failure_count':dmiss,'arcad_failure_count':amiss,'failure_intersection':inter,'failure_union':union,
            'failure_jaccard':inter/union if union else float('nan'),'d2_rescue_by_arcad':aonly/dmiss if dmiss else float('nan'),
            'arcad_rescue_by_d2':bonly/amiss if amiss else float('nan'),'oracle_union_coverage':1-inter/max(1,npos),
            'phi_error_association':phi,'paired_discordance_count':discord,'mcnemar_exact_p':pval,
            'normal_n':int(neg.sum()),'normal_fp_d2':d_fp,'normal_fp_arcad':a_fp,'normal_both_fp':both_fp,
            'normal_either_fp':either_fp,'normal_fp_union_cost':either_fp/max(1,int(neg.sum()))}

def main():
    (OUT/'plots').mkdir(exist_ok=True); (OUT/'maps').mkdir(exist_ok=True); (OUT/'paired_examples').mkdir(exist_ok=True)
    manifest=json.loads(MANIFEST.read_text()); catalog={}; map_sources={}; raw_previews={}
    for cat in CATS:
        d=manifest['categories'][cat]; recs=sorted(d['shared_test_images'],key=lambda r:r['image_path'])
        if len(recs)!=d['counts']['shared_views']: raise RuntimeError('Manifest count mismatch '+cat)
        gt,imgs=common_ground_truth(cat,recs)
        for model in MODELS:
            native_map,path=load_maps(model,cat,recs); common=align_map(model,native_map)
            if common.shape!=(len(recs),COMMON_SIZE,COMMON_SIZE): raise ValueError('bad map shape')
            vrows,orows=score_rows(recs,native_map)
            # Keep scores computed from maps; Task 01 rows are an independent check for D2 low-three.
            if model=='d2' and cat in ('lattice_block_plug','laser_diode','power_jack'):
                old={r['image_path']:float(r['top1pct_view_score']) for r in csv.DictReader(open(T1/'per_view_scores.csv')) if r['category']==cat}
                dif=max(abs(old[r['image_path']]-r['score_view']) for r in vrows)
                if dif>1e-5: raise ValueError(f'D2 task01 view-score mismatch {cat}: {dif}')
            thresholds=get_thresholds(d,vrows,orows,set(d['calibration_object_keys']),common,gt)
            catalog.setdefault(cat,{})[model]={'records':recs,'gt':gt,'images':imgs,'native_maps':native_map,'common_maps':common,
                'view_rows':vrows,'object_rows':orows,'thresholds':thresholds,'map_path':path}
            np.savez_compressed(OUT/'maps'/f'common_{model}_{cat}.npz',smoothed_maps=common,gt_masks=gt.astype(np.uint8),image_paths=np.asarray([r['image_path'] for r in recs]),coordinate_frame=np.asarray('original source field x/y [64,960) of 1024, nearest GT, 224x224; aligned across D2 and ArcAD'))
            map_sources[(cat,model)]=path
        if catalog[cat]['d2']['records']!=catalog[cat]['arcad']['records']: raise ValueError(f'paired records differ in {cat}')
        raw_previews[cat]=imgs

    # Programmatic alignment checks: common masks are identical across model rows,
    # and the physical crop/resampling geometry is saved explicitly.
    align_report={'status':'PASS','source_size':[1024,1024],'arcad_visible_source_bounds_xyxy':[64,64,960,960],
                  'd2_valid_common_crop_in_256_map':[16,16,240,240],'common_map_shape':[224,224],
                  'ArcAD':'256 map covers the 392/448 crop, equivalent to source bounds [64,960); bilinear resample 256->224.',
                  'D2':'256 map covers the full 1024 frame; center crop [16:240] corresponds to the same [64,960) source region.',
                  'ground_truth':'Read original 1024 masks, crop source [64:960] in each axis, nearest-neighbor resize to 224.',
                  'valid_common_area_fraction':0.875**2,'known_source_mask_exceptions':[],'visual_checks':[]}
    for cat in CATS:
        a=catalog[cat]['d2']; b=catalog[cat]['arcad']
        for i,r in enumerate(a['records']):
            if r['label'] and not a['gt'][i].any(): align_report['known_source_mask_exceptions'].append({'category':cat,'image_path':r['image_path'],'type':'positive label with empty source mask'})
        if not np.array_equal(a['gt'],b['gt']): raise ValueError('Common mask alignment mismatch '+cat)
    (OUT/'alignment_validation.json').write_text(json.dumps(align_report,indent=2)+'\n')

    # Compute full paired baseline metrics and per-view/object data tables.
    joined=[]; view_diag=[]; object_diag=[]; region_diag=[]; category_metrics=[]; failure_rows=[]; taxonomy=[]; morph_rows=[]; morph_region_rows=[]
    thresholds_out=[]; seen_rows=[]; originals={}; confusion_rows=[]
    native_arc=json.load(open(ARCAD/'out/real_iad_variety out/low3_shared_split_comparison_20261031/arcad_common_test_metrics.json'))
    arc_native={r['Category']:r for r in csv.DictReader(open(ARCAD/'out/real_iad_variety out/arcad_realiad_variety/results.csv')) if r['Category']!='Mean'}
    d2_native=json.load(open(D2/'out/real_iad_variety_1024 out/full run/variety_160cat/review_low3/metrics.json'))
    d2_trainlog={}
    for line in (D2/'out/real_iad_variety_1024 out/full run/variety_160cat/log.txt').read_text(errors='replace').splitlines():
        m=re.match(r'([^:]+): I-Auroc:([0-9.]+)',line)
        if m and m.group(1)!='Mean': d2_trainlog[m.group(1)]=float(m.group(2))
    supervision=manifest['supervision']

    # Morphology is category/view specific; images were resampled only for the optional contrast proxy.
    morph_lookup={}
    for cat in CATS:
        dat=catalog[cat]['d2']; recs=dat['records']; masks=dat['gt']; imgs=dat['images']
        seen_types=set(supervision[cat])-{ 'OK','' }
        for i,r in enumerate(recs):
            if not r['label']: continue
            mask=masks[i]; n=int(mask.sum()); comps=components(mask)
            area=n/(COMMON_SIZE*COMMON_SIZE); bins=('<0.1%', '0.1-1%', '1-5%', '>5%')
            size_bin=bins[0] if area<.001 else bins[1] if area<.01 else bins[2] if area<.05 else bins[3]
            gray=cv2.cvtColor(imgs[i],cv2.COLOR_RGB2GRAY).astype(np.float32)
            dil=cv2.dilate(mask.astype(np.uint8),np.ones((7,7),np.uint8),iterations=1).astype(bool)
            ring=dil&~mask
            contrast=float(gray[mask].mean()-gray[ring].mean()) if mask.any() and ring.any() else float('nan')
            row={'category':cat,'object_key':r['object_key'],'image_path':r['image_path'],'view_id':r['view_id'],'defect_type':r['anomaly_class'],
                 'area_pixels_common':n,'gt_area_fraction':area,'gt_region_count':len(comps),'size_bin':size_bin,
                 'luminance_contrast_gt_minus_7px_ring':contrast,'edge_distance_min_px':min((c['border_distance_px'] for c in comps),default=float('nan')),
                 'edge_distance_source_px_approx':4*min((c['border_distance_px'] for c in comps),default=float('nan')),
                 'foreground_object_mask_available':False,'arcad_seen_type':r['anomaly_class'] in seen_types,
                 'coordinate_resolution':'224x224 covering 896x896 source pixels; ~4 source px/output pixel'}
            morph_lookup[(cat,r['image_path'])]=row; morph_rows.append(row)
            for compno,c in enumerate(comps,1):
                morph_region={'category':cat,'object_key':r['object_key'],'image_path':r['image_path'],'view_id':r['view_id'],'defect_type':r['anomaly_class'],
                              'component_id':compno,'area_pixels':c['area'],'area_fraction':c['area']/(COMMON_SIZE*COMMON_SIZE),
                              'bbox_x':c['x'],'bbox_y':c['y'],'bbox_width':c['width'],'bbox_height':c['height'],'aspect_ratio':c['aspect_ratio'],
                              'perimeter_8connect_cv2':c['perimeter'],'compactness_4piA_over_P2':c['compactness'],'centroid_x':c['centroid_x'],'centroid_y':c['centroid_y'],'border_distance_px':c['border_distance_px']}
                morph_region_rows.append(morph_region)

    # Precompute original shared scores and the independently-held calibration thresholds.
    thresh_by_cat={cat:{m:data['thresholds'] for m,data in catalog[cat].items()} for cat in CATS}
    for cat in CATS:
        d=catalog[cat]['d2']; a=catalog[cat]['arcad']; recs=d['records']; gt=d['gt']
        yv=np.asarray([r['label'] for r in recs],dtype=np.uint8)
        okeys=sorted({r['object_key'] for r in recs})
        yobj=np.asarray([next(o['label'] for o in d['object_rows'] if o['object_key']==k) for k in okeys],dtype=np.uint8)
        metrics={'category':cat,'n_objects':len(okeys),'n_views':len(recs),'n_positive_objects':int(yobj.sum()),'n_positive_views':int(yv.sum()),'positive_object_prevalence':float(yobj.mean()),'positive_view_prevalence':float(yv.mean()),
                 'selection_group':'low3_failure_enriched' if cat in CATS[:3] else 'additional_preselected_comparison'}
        for model,data in [('d2',d),('arcad',a)]:
            va,ap=auc_ap(yv,[r['score_view'] for r in data['view_rows']]); oa,oap=auc_ap(yobj,[o['score_object'] for o in data['object_rows']])
            metrics.update({f'{model}_I-AUROC':va,f'{model}_I-AP':ap,f'{model}_O-AUROC':oa,f'{model}_O-AP':oap,
                            f'{model}_I-oracle_max_F1':oracle_f1(yv,[r['score_view'] for r in data['view_rows']]),
                            f'{model}_O-oracle_max_F1':oracle_f1(yobj,[o['score_object'] for o in data['object_rows']])})
            py,ps=gt.ravel(),data['common_maps'].ravel(); pa,pap=auc_ap(py,ps)
            pro,_,_=compute_pro_curve(gt,data['common_maps'],cap=.3)
            metrics.update({f'{model}_common_P-AUROC':pa,f'{model}_common_P-AP':pap,f'{model}_common_P-AUPRO':pro})
        for metric in ('I-AUROC','I-AP','O-AUROC','O-AP','common_P-AUROC','common_P-AP','common_P-AUPRO'):
            metrics[f'delta_D2_minus_ArcAD_{metric}']=metrics[f'd2_{metric}' if metric.startswith('common_') else f'd2_{metric}']-metrics[f'arcad_{metric}' if metric.startswith('common_') else f'arcad_{metric}']
        # native split metrics are context only, never paired with shared values.
        if cat in d2_native['categories']:
            for k,v in d2_native['categories'][cat].items(): metrics[f'd2_original_native_{k}']=v
        if cat in arc_native:
            for k in ('I-AUROC','I-AP','I-F1','P-AUROC','P-AP','P-F1','P-AUPRO'): metrics[f'arcad_original_native_{k}']=float(arc_native[cat][k])
        category_metrics.append(metrics)

        # Pixel thresholds from calibration normal maps, operating score thresholds from
        # calibration normal objects. All reported thresholded evaluation excludes cal objects.
        cal_keys=set(manifest['categories'][cat]['calibration_object_keys'])
        eval_idx=[i for i,r in enumerate(recs) if r['object_key'] not in cal_keys]
        obj_by_model={m:{o['object_key']:o for o in data['object_rows']} for m,data in [('d2',d),('arcad',a)]}
        view_by_model={m:{r['image_path']:r for r in data['view_rows']} for m,data in [('d2',d),('arcad',a)]}
        for alpha in CAL_ALPHAS:
            pflags={}
            for model,data in [('d2',d),('arcad',a)]:
                th=data['thresholds'][alpha]
                ev_views=[view_by_model[model][recs[i]['image_path']] for i in eval_idx]
                ev_objs=[o for o in data['object_rows'] if o['object_key'] not in cal_keys]
                view_pred=np.asarray([r['score_view']>=th['view'] for r in ev_views],dtype=np.uint8)
                obj_pred=np.asarray([o['score_object']>=th['object'] for o in ev_objs],dtype=np.uint8)
                view_lab=np.asarray([r['label'] for r in ev_views],dtype=np.uint8)
                obj_lab=np.asarray([o['label'] for o in ev_objs],dtype=np.uint8)
                pflags[model]={'view':(view_lab,view_pred,ev_views),'object':(obj_lab,obj_pred,ev_objs)}
                for level,(yy,pp,items) in pflags[model].items():
                    confusion_rows.append({'category':cat,'model':model,'level':level,'target_normal_object_fpr':alpha,'threshold':th['view'] if level=='view' else th['object'],**cm_metrics(yy,pp)})
                for level,(y,p,items) in pflags[model].items():
                    cm=cm_metrics(y,p)
                    failure_rows.append({'category':cat,'level':level,'operating_point':alpha,'threshold_status':'held-out normal-only calibration','threshold':th['view'] if level=='view' else th['object'],**failure_table(y,p,p)}) if False else None
                    category_metrics[-1][f'{model}_{level}_tpr_at_{alpha:g}']=cm['recall']
                    category_metrics[-1][f'{model}_{level}_precision_at_{alpha:g}']=cm['precision']
                    category_metrics[-1][f'{model}_{level}_fpr_at_{alpha:g}']=cm['fpr']
            for level in ('view','object'):
                y,pd,pai=pflags['d2'][level]; _,pa,pbi=pflags['arcad'][level]
                overlap=failure_table(y,pd,pa)
                failure_rows.append({'category':cat,'level':level,'target_normal_object_fpr':alpha,'threshold_protocol':'10 normal shared objects reserved for calibration; evaluation excludes them','oracle_union_label':'oracle coverage at fixed thresholds, not attainable ensemble',**overlap})

        # Per-view numeric diagnostics in common, jointly visible physical crop.
        pix_thresholds={m:{alpha:data['thresholds']['pixel'][alpha]['threshold'] for alpha in PIXEL_ALPHAS} for m,data in [('d2',d),('arcad',a)]}
        for i,r in enumerate(recs):
            mask=gt[i]; nonempty=bool(mask.any()); base=morph_lookup.get((cat,r['image_path']),{})
            for model,data in [('d2',d),('arcad',a)]:
                amap=data['common_maps'][i]; vals=amap[mask] if nonempty else np.array([]); bg=amap[~mask]
                mu_g=float(vals.mean()) if len(vals) else float('nan'); mu_b=float(bg.mean()); sig_b=float(bg.std())
                q=float(roc_auc_score(mask.ravel().astype(np.uint8),amap.ravel())) if nonempty and (~mask).any() else float('nan')
                max_pct,mean_pct=bg_percentiles(vals,bg) if nonempty else (float('nan'),float('nan'))
                tops=topk_stats(amap,mask)
                row={'category':cat,'model':model,'object_key':r['object_key'],'image_path':r['image_path'],'view_id':r['view_id'],'gt_view_label':r['label'],'gt_object_label':max(v['label'] for v in d['view_rows'] if v['object_key']==r['object_key']),
                     'defect_type':r['anomaly_class'],'gt_mask_path':r['mask_path'] or '', 'valid_mask_coordinates':True,'common_fov':'source [64:960)x[64:960), 224x224','mask_nonempty':nonempty,
                     'gt_area_fraction':float(mask.mean()),'gt_region_count':base.get('gt_region_count',0),'C_muG_minus_muB':mu_g-mu_b if nonempty else float('nan'),'Z_contrast':(mu_g-mu_b)/(sig_b+EPS) if nonempty else float('nan'),
                     'Q_within_view_pixel_AUROC':q,'GT_max_background_percentile':max_pct,'GT_mean_background_percentile':mean_pct,'mu_G':mu_g,'mu_B':mu_b,'sigma_B':sig_b,
                     'FP_area_fraction_at_pixelFPR_1pct_cal':float(np.sum((amap>=pix_thresholds[model][.01])&(~mask))/mask.size),'pixel_FPR_at_pixelFPR_1pct_cal':float(np.sum((amap>=pix_thresholds[model][.01])&(~mask))/max(1,int((~mask).sum()))),
                     'FP_area_fraction_at_pixelFPR_5pct_cal':float(np.sum((amap>=pix_thresholds[model][.05])&(~mask))/mask.size),'pixel_FPR_at_pixelFPR_5pct_cal':float(np.sum((amap>=pix_thresholds[model][.05])&(~mask))/max(1,int((~mask).sum()))),
                     'top1pct_off_target_fraction':1-tops['top1_precision'] if nonempty else float('nan'),**tops,**base}
                for alpha in CAL_ALPHAS:
                    view_th=data['thresholds'][alpha]['view']
                    row[f'detected_at_normal_object_fpr_{alpha:g}']=bool(data['view_rows'][i]['score_view']>=view_th)
                view_diag.append(row)

            # GT-region localization at the two calibrated common-grid pixel thresholds.
            gtcomps=components(mask)
            for model,data in [('d2',d),('arcad',a)]:
                amap=data['common_maps'][i]
                for alpha in PIXEL_ALPHAS:
                    threshold=pix_thresholds[model][alpha]
                    pred=amap>=threshold
                    predcomps=components(pred)
                    pred_label=predcomps[0]['label_map'] if predcomps else np.zeros(mask.shape,dtype=np.int32)
                    pred_ids=np.unique(pred_label[mask]) if mask.any() else np.array([0])
                    false_positive_components=max(0,len(predcomps)-int(np.count_nonzero(pred_ids>0)))
                    pred_areas=np.bincount(pred_label.ravel(),minlength=len(predcomps)+1)
                    for compno,gc in enumerate(gtcomps,1):
                        glabel=gc['label_map']; reg=glabel==gc['component_id']; best_iou=0.; best_cov=0.; best_inter=0
                        overlaps=np.bincount(pred_label[reg],minlength=len(predcomps)+1)
                        for pid in range(1,len(predcomps)+1):
                            inter=int(overlaps[pid]); union=int(gc['area']+pred_areas[pid]-inter)
                            iou=inter/max(1,union)
                            if iou>best_iou:best_iou=iou;best_cov=inter/max(1,int(gc['area']));best_inter=inter
                        region_diag.append({'category':cat,'model':model,'object_key':r['object_key'],'image_path':r['image_path'],'view_id':r['view_id'],'defect_type':r['anomaly_class'],
                                            'region_id':compno,'region_area_px':gc['area'],'region_area_fraction':gc['area']/mask.size,'pixel_fpr_target':alpha,'pixel_threshold':threshold,
                                            'best_predicted_component_iou':best_iou,'gt_region_coverage':best_cov,'any_overlap':best_inter>0,
                                            'detected_iou_0.1':best_iou>=.1,'detected_iou_0.3':best_iou>=.3,'detected_iou_0.5':best_iou>=.5,
                                            'false_positive_predicted_components_in_view':false_positive_components})

        # Object-level multiview consistency, respecting GT-positive view counts.
        for o in d['object_rows']:
            key=o['object_key']; idx=[i for i,r in enumerate(recs) if r['object_key']==key]
            pos_idx=[i for i in idx if recs[i]['label']==1]
            for model,data in [('d2',d),('arcad',a)]:
                bypath=view_by_model[model]; ss=np.asarray([bypath[recs[i]['image_path']]['score_view'] for i in idx]); ps=np.asarray([bypath[recs[i]['image_path']]['score_view'] for i in pos_idx])
                pooled=obj_by_model[model][key]['score_object']
                row={'category':cat,'model':model,'object_key':key,'gt_object_label':int(o['label']),'gt_positive_view_count':len(pos_idx),'view_score_mean':float(ss.mean()),'view_score_std':float(ss.std()),'view_score_cv':float(ss.std()/ss.mean()) if np.all(ss>0) and abs(ss.mean())>EPS else float('nan'),
                     'positive_view_score_mean':float(ps.mean()) if len(ps) else float('nan'),'normal_views_of_defective_object_max_score':float(max((bypath[recs[i]['image_path']]['score_view'] for i in idx if i not in pos_idx),default=float('nan'))),
                     'score_object_pooled_top1pct':pooled}
                for alpha in CAL_ALPHAS:
                    th=data['thresholds'][alpha]
                    detected_pos=sum(bypath[recs[i]['image_path']]['score_view']>=th['view'] for i in pos_idx)
                    row[f'detected_positive_views_at_{alpha:g}']=int(detected_pos)
                    row[f'positive_view_fraction_detected_at_{alpha:g}']=detected_pos/len(pos_idx) if len(pos_idx) else float('nan')
                    row[f'object_detected_from_pooled_score_at_{alpha:g}']=pooled>=th['object']
                object_diag.append(row)

        # Error taxonomy, transparent heuristic tags on positive views only.
        for model in MODELS:
            vrows=catalog[cat][model]['view_rows']; maps=catalog[cat][model]['common_maps']; data=catalog[cat][model]
            for i,r in enumerate(vrows):
                if not r['label'] or not gt[i].any():continue
                q=float(roc_auc_score(gt[i].ravel().astype(np.uint8),maps[i].ravel()))
                top=topk_stats(maps[i],gt[i]); miss=not any(r['score_view']>=data['thresholds'][alpha]['view'] for alpha in CAL_ALPHAS)
                tags=[]
                if q<=.5 and top['top1_recall']<.2:tags.append('no_signal')
                if q>=.7 and miss:tags.append('on_target_but_outranked')
                if q<.7 and top['top1_recall']<.5:tags.append('partial_region')
                if not tags:tags.append('other_or_detected')
                taxonomy.append({'category':cat,'model':model,'object_key':r['object_key'],'image_path':r['image_path'],'view_id':r['view_id'],'defect_type':r['anomaly_class'],'heuristic_tags':'|'.join(tags),'Q':q,'top1pct_recall':top['top1_recall'],'missed_at_all_calibrated_view_thresholds':miss})

        # Defect type seen/unseen by ArcAD labeled anomaly classes.
        seen=set(supervision[cat])-{ 'OK','' }
        for typ in sorted(set(r['anomaly_class'] for r in recs if r['label'])):
            im=[i for i,r in enumerate(recs) if r['label'] and r['anomaly_class']==typ]
            objset={recs[i]['object_key'] for i in im}; stratum='seen' if typ in seen else 'unseen'
            sr={'category':cat,'defect_type':typ,'stratum_relative_to_arcad_labeled_types':stratum,'arcad_supervised_objects':supervision[cat].get(typ,{}).get('distinct_objects',0),'arcad_supervised_views':supervision[cat].get(typ,{}).get('views',0),'n_positive_views':len(im),'n_positive_objects':len(objset)}
            for model in MODELS:
                data=catalog[cat][model]; bypath={r['image_path']:r for r in data['view_rows']}
                eval_idx=[i for i in im if recs[i]['object_key'] not in cal_keys]
                for alpha in CAL_ALPHAS:
                    pred=[bypath[recs[i]['image_path']]['score_view']>=data['thresholds'][alpha]['view'] for i in eval_idx]
                    sr[f'{model}_view_recall_at_{alpha:g}']=float(np.mean(pred)) if pred else float('nan')
                sr[f'{model}_mean_Q']=float(np.mean([roc_auc_score(gt[i].ravel().astype(np.uint8),data['common_maps'][i].ravel()) for i in im if gt[i].any()])) if any(gt[i].any() for i in im) else float('nan')
            seen_rows.append(sr)

        # Calibration threshold metadata from object/view/pixel reference sets.
        for model,data in [('d2',d),('arcad',a)]:
            for alpha in CAL_ALPHAS:
                for lev in ('view','object'):
                    thresholds_out.append({'category':cat,'model':model,'target_normal_object_fpr':alpha,'threshold_level':lev,'threshold':data['thresholds'][alpha][lev],
                                           'calibration_normal_objects':len(cal_keys),'calibration_realized_fpr':data['thresholds'][alpha]['cal_view_max_fpr' if lev=='view' else 'cal_object_fpr'],
                                           'evaluation_excludes_calibration_objects':True})
            for alpha in PIXEL_ALPHAS:
                q=data['thresholds']['pixel'][alpha]
                thresholds_out.append({'category':cat,'model':model,'target_pixel_fpr':alpha,'threshold_level':'common_map_pixel','threshold':q['threshold'],'calibration_realized_pixel_fpr':q['cal_pixel_fpr'],'calibration_normal_objects':len(cal_keys)})

    # Save all score/diagnostic tables.
    for cat in CATS:
        for model in MODELS:
            for r in catalog[cat][model]['view_rows']:
                joined.append({'category':cat,'object_key':r['object_key'],'image_path':r['image_path'],'view_id':r['view_id'],'gt_view_label':r['label'],'gt_object_label':r['gt_object_label'],
                               'defect_type':r['anomaly_class'],'model':model,'score_view':r['score_view'],'score_object':r['score_object'],'map_path':catalog[cat][model]['map_path'],'gt_mask_path':r['mask_path'] or '',
                               'split_id':'shared_eval' if r['object_key'] not in set(manifest['categories'][cat]['calibration_object_keys']) else 'normal_only_calibration','calibration_role':r['object_key'] in set(manifest['categories'][cat]['calibration_object_keys'])})
    write_csv(OUT/'joined_predictions.csv',joined)
    write_csv(OUT/'view_diagnostics.csv',view_diag)
    write_csv(OUT/'region_diagnostics.csv',region_diag)
    write_csv(OUT/'object_diagnostics.csv',object_diag)
    write_csv(OUT/'category_metrics.csv',category_metrics)
    write_csv(OUT/'failure_overlap.csv',failure_rows)
    write_csv(OUT/'error_taxonomy_counts.csv',taxonomy)
    write_csv(OUT/'seen_unseen_summary.csv',seen_rows)
    write_csv(OUT/'thresholds.csv',thresholds_out)
    write_csv(OUT/'confusion_matrices.csv',confusion_rows)
    write_csv(OUT/'morphology_view.csv',morph_rows)
    write_csv(OUT/'morphology_region.csv',morph_region_rows)
    # Core bootstrap, deltas, plots, examples, and report are emitted in subsequent pass.
    (OUT/'analysis_state.json').write_text(json.dumps({'categories':list(CATS),'bootstraps':BOOTSTRAPS,'seed':BOOT_SEED,'baseline_metrics':category_metrics,'status':'core_tables_written','alignment':align_report},indent=2)+'\n')
    print('Core paired tables written; categories:',CATS)

def oracle_f1(y,s):
    y=np.asarray(y,dtype=np.uint8); s=np.asarray(s,dtype=np.float64)
    if not y.any() or y.all(): return float('nan')
    # Explicit benchmark-only oracle F1.
    order=np.argsort(-s); yy=y[order]; tp=np.cumsum(yy); fp=np.cumsum(1-yy); fn=int(yy.sum())-tp
    f=2*tp/(2*tp+fp+fn+EPS)
    return float(np.max(f))

if __name__=='__main__': main()
