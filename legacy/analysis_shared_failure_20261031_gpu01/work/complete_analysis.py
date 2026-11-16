#!/usr/bin/env python3
"""Cluster bootstrap summaries, plots, examples, provenance, and report."""
import csv,json,hashlib,platform,subprocess,sys
import re
from collections import defaultdict
from sklearn.linear_model import LogisticRegression
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from sklearn.metrics import roc_auc_score, average_precision_score
HERE=Path(__file__).resolve().parent; OUT=HERE.parent
def readcsv(p):
    with open(p,encoding='utf-8') as f:return list(csv.DictReader(f))
def writecsv(p,rows):
    if not rows:return
    with open(p,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),extrasaction='ignore');w.writeheader();w.writerows(rows)
def add_camera_ids():
    for name in ('joined_predictions.csv','view_diagnostics.csv','region_diagnostics.csv','morphology_view.csv','morphology_region.csv'):
        p=OUT/name; rows=readcsv(p)
        for r in rows:
            m=re.search(r'_C([1-5])_',r.get('image_path',''))
            r['camera_id']=f'C{m.group(1)}' if m else ''
        writecsv(p,rows)
def ci_rows():
    rng=np.random.default_rng(20261031); joined=readcsv(OUT/'joined_predictions.csv'); result=[]
    for cat in sorted(set(r['category'] for r in joined)):
        # Threshold-free I/O uncertainty matches the full shared-split point estimates;
        # calibration objects are excluded only from thresholded operating-point tables.
        dat=[r for r in joined if r['category']==cat]
        objkeys=sorted(set(r['object_key'] for r in dat)); by={m:defaultdict(list) for m in ('d2','arcad')}
        for r in dat:by[r['model']][r['object_key']].append(r)
        metrics={}
        for level in ('I','O'):
            for model in ('d2','arcad'):
                if level=='I': y=np.array([int(r['gt_view_label']) for r in dat if r['model']==model]);s=np.array([float(r['score_view']) for r in dat if r['model']==model])
                else:
                    rows=[by[model][k][0] for k in objkeys];y=np.array([max(int(r['gt_view_label']) for r in by[model][k]) for k in objkeys]);s=np.array([float(r['score_object']) for r in rows])
                metrics[(level,model,'AUROC')]=(y,s,lambda a,b:roc_auc_score(a,b) if len(np.unique(a))==2 else np.nan)
                metrics[(level,model,'AP')]=(y,s,lambda a,b:average_precision_score(a,b) if len(np.unique(a))==2 else np.nan)
        for metrickey,(y,s,fn) in metrics.items():
            level,model,metric=metrickey; point=fn(y,s); vals=[]
            for _ in range(2000):
                chosen=rng.choice(objkeys,len(objkeys),replace=True)
                if level=='O':
                    # Use one score/label per sampled object, retaining duplicate clusters.
                    yy=np.array([max(int(x['gt_view_label']) for x in by[model][k]) for k in chosen]); ss=np.array([float(by[model][k][0]['score_object']) for k in chosen])
                else:
                    yy=np.concatenate([[int(x['gt_view_label']) for x in by[model][k]] for k in chosen]);ss=np.concatenate([[float(x['score_view']) for x in by[model][k]] for k in chosen])
                if len(np.unique(yy))==2:vals.append(fn(yy,ss))
            result.append({'category':cat,'metric_level':level,'metric':metric,'model':model,'estimate':point,'ci_2.5':float(np.nanpercentile(vals,2.5)) if vals else float('nan'),'ci_97.5':float(np.nanpercentile(vals,97.5)) if vals else float('nan'),'n_bootstrap_valid':len(vals),'bootstrap_unit':'paired physical objects; all five views retained','seed':20261031})
        # Paired delta resamples for view and object score metrics.
        for level in ('I','O'):
            for metric in ('AUROC','AP'):
                fn=roc_auc_score if metric=='AUROC' else average_precision_score
                point=next(v for k,v in [(k,fn(y,s) if len(np.unique(y))==2 else np.nan) for k,(y,s,_) in metrics.items()] if k==(level,'d2',metric))-next(fn(y,s) if len(np.unique(y))==2 else np.nan for k,(y,s,_) in metrics.items() if k==(level,'arcad',metric))
                vals=[]
                for _ in range(2000):
                    chosen=rng.choice(objkeys,len(objkeys),replace=True); yy=[]; sd=[];sa=[]
                    for key in chosen:
                        rowsd=by['d2'][key]; rowsa=by['arcad'][key]
                        if level=='I':
                            yy.extend(int(x['gt_view_label']) for x in rowsd);sd.extend(float(x['score_view']) for x in rowsd);sa.extend(float(x['score_view']) for x in rowsa)
                        else:
                            yy.append(max(int(x['gt_view_label']) for x in rowsd));sd.append(float(rowsd[0]['score_object']));sa.append(float(rowsa[0]['score_object']))
                    if len(set(yy))==2: vals.append(fn(yy,sd)-fn(yy,sa))
                result.append({'category':cat,'metric_level':level,'metric':metric,'model':'D2_minus_ArcAD_paired_delta','estimate':point,'ci_2.5':float(np.percentile(vals,2.5)) if vals else float('nan'),'ci_97.5':float(np.percentile(vals,97.5)) if vals else float('nan'),'n_bootstrap_valid':len(vals),'bootstrap_unit':'paired physical objects; all five views retained','seed':20261031})
    writecsv(OUT/'bootstrap_ci.csv',result)
def mechanism_model():
    rows=[r for r in readcsv(OUT/'view_diagnostics.csv') if r['mask_nonempty']=='True']
    out=[]; rng=np.random.default_rng(2091)
    cats=sorted(set(r['category'] for r in rows)); cams=['C1','C2','C3','C4','C5']
    for model in ('d2','arcad'):
        rs=[r for r in rows if r['model']==model]
        def matrix(items):
            cont=[]; y=[]; groups=[]
            for r in items:
                z=float(r['Z_contrast']) if r['Z_contrast'] not in ('','nan') else 0.
                edge=float(r['edge_distance_min_px']) if r['edge_distance_min_px'] not in ('','nan') else 0.
                cont.append([np.log10(max(float(r['gt_area_fraction']),1e-8)),z,float(r['gt_region_count']),edge])
                y.append(int(r['detected_at_normal_object_fpr_0.1']!='True'));groups.append((r['category'],r['object_key']))
            x=np.asarray(cont,dtype=float); mu=x.mean(0);sd=x.std(0);sd[sd==0]=1;x=(x-mu)/sd
            one=np.asarray([[float(r['category']==c) for c in cats]+[float(r.get('camera_id','')==cam) for cam in cams] for r in items])
            return np.c_[x,one],np.asarray(y),np.asarray(groups,dtype=object)
        X,y,g=matrix(rs); names=['log10_area_z','Z_contrast_z','region_count_z','image_border_distance_z']+[f'category={c}' for c in cats]+[f'camera={c}' for c in cams]
        base=LogisticRegression(max_iter=500,C=1.0).fit(X,y); coefs=[base.coef_[0]]; keys=sorted(set(map(tuple,g)))
        group_indices={key:np.flatnonzero(np.all(g==np.asarray(key,dtype=object),axis=1)) for key in keys}
        for _ in range(200):
            chosen=rng.integers(0,len(keys),len(keys)); sample=[]
            for j in chosen:sample.extend(group_indices[keys[j]])
            if len(set(y[sample]))<2:continue
            try:coefs.append(LogisticRegression(max_iter=500,C=1.0).fit(X[sample],y[sample]).coef_[0])
            except Exception:pass
        arr=np.asarray(coefs)
        for j,name in enumerate(names):out.append({'model':model,'outcome':'missed view at held-out-normal target 10%; positive nonempty masks only','feature':name,'log_odds_coefficient':float(base.coef_[0,j]),'odds_ratio':float(np.exp(np.clip(base.coef_[0,j],-20,20))),'object_cluster_bootstrap_ci_2.5':float(np.percentile(arr[:,j],2.5)),'object_cluster_bootstrap_ci_97.5':float(np.percentile(arr[:,j],97.5)),'bootstrap_valid':len(arr),'cluster_unit':'category/object','interpretation':'exploratory association; category/camera adjusted; not causal'})
    writecsv(OUT/'mechanism_associations.csv',out)
def plots_and_report():
    metrics=readcsv(OUT/'category_metrics.csv'); names=[r['category'] for r in metrics]; delta=[float(r['delta_D2_minus_ArcAD_I-AUROC']) for r in metrics]
    order=np.argsort(delta); fig,ax=plt.subplots(figsize=(9,4));ax.barh(np.array(names)[order],np.array(delta)[order],color=['#3288bd' if x>0 else '#d53e4f' for x in np.array(delta)[order]]);ax.axvline(0,color='black',lw=.8);ax.set_xlabel('D2 − ArcAD shared-split I-AUROC');fig.tight_layout();fig.savefig(OUT/'plots/category_delta_i_auroc.png',dpi=160);plt.close(fig)
    joined=readcsv(OUT/'joined_predictions.csv');fig,axes=plt.subplots(1,5,figsize=(17,3.5))
    for ax,cat in zip(axes, names):
        rows=[r for r in joined if r['category']==cat and r['split_id']=='shared_eval']; dd={r['image_path']:r for r in rows if r['model']=='d2'}; aa={r['image_path']:r for r in rows if r['model']=='arcad'}; keys=sorted(set(dd)&set(aa)); y=[int(dd[k]['gt_view_label']) for k in keys]
        ax.scatter([float(dd[k]['score_view']) for k in keys],[float(aa[k]['score_view']) for k in keys],c=y,s=5,cmap='coolwarm',alpha=.5);ax.set_title(cat,fontsize=8);ax.set_xlabel('D2');ax.set_ylabel('ArcAD')
    fig.tight_layout();fig.savefig(OUT/'plots/paired_score_rank_scatter.png',dpi=150);plt.close(fig)
    v=readcsv(OUT/'view_diagnostics.csv');fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,model in zip(axes,('d2','arcad')):
        rows=[r for r in v if r['model']==model and r['mask_nonempty']=='True']; ax.scatter([float(r['gt_area_fraction']) for r in rows],[float(r['Q_within_view_pixel_AUROC']) for r in rows],s=5,alpha=.25);ax.set_xscale('log');ax.set_xlabel('GT area fraction');ax.set_ylabel('Within-view pixel AUROC Q');ax.set_title(model)
    fig.tight_layout();fig.savefig(OUT/'plots/size_response.png',dpi=150);plt.close(fig)
    # Fixed-threshold contrast association, top-k coverage and category/type summaries.
    diag=readcsv(OUT/'view_diagnostics.csv'); fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,model in zip(axes,('d2','arcad')):
        rows=[r for r in diag if r['model']==model and r['mask_nonempty']=='True']
        ax.scatter([float(r['Z_contrast']) for r in rows if r['Z_contrast'] not in ('','nan')],[int(r['detected_at_normal_object_fpr_0.1']=='True') for r in rows if r['Z_contrast'] not in ('','nan')],s=5,alpha=.2);ax.set_xlabel('GT/background Z contrast');ax.set_ylabel('Detected at 10% normal object FPR');ax.set_title(model)
    fig.tight_layout();fig.savefig(OUT/'plots/contrast_vs_detection.png',dpi=150);plt.close(fig)
    fracs=('0.1','0.5','1','2','5'); fig,ax=plt.subplots(figsize=(6,4))
    for model in ('d2','arcad'):
        rows=[r for r in diag if r['model']==model and r['mask_nonempty']=='True']; vals=[np.mean([float(r[f'top{f}_recall']) for r in rows]) for f in fracs];ax.plot([float(x) for x in fracs],vals,marker='o',label=model)
    ax.set_xlabel('Top-k map area (%)');ax.set_ylabel('Mean GT pixel recall');ax.legend();fig.tight_layout();fig.savefig(OUT/'plots/topk_coverage.png',dpi=150);plt.close(fig)
    # Per-type score localization summary; omit sparse groups with fewer than 5 views.
    grouped=defaultdict(list)
    for r in diag:
        if r['mask_nonempty']=='True':grouped[(r['category'],r['model'],r['defect_type'])].append(r)
    typ_rows=[]
    for (cat,model,typ),rows in grouped.items():
        if len(rows)<5:continue
        typ_rows.append({'category':cat,'model':model,'defect_type':typ,'positive_views':len(rows),'mean_Q':float(np.nanmean([float(r['Q_within_view_pixel_AUROC']) for r in rows])),'mean_top1pct_recall':float(np.nanmean([float(r['top1_recall']) for r in rows])),'recall_at_normal_object_fpr_0.1':float(np.mean([r['detected_at_normal_object_fpr_0.1']=='True' for r in rows]))})
    writecsv(OUT/'defect_type_summary.csv',typ_rows)
    if typ_rows:
        top=sorted(typ_rows,key=lambda r:(r['category'],r['defect_type'],r['model'])); labels=[f"{r['category']}\n{r['defect_type']}\n{r['model']}" for r in top]; fig,ax=plt.subplots(figsize=(max(9,len(labels)*.28),4));ax.bar(range(len(top)),[r['recall_at_normal_object_fpr_0.1'] for r in top]);ax.set_xticks(range(len(top)),labels,rotation=90,fontsize=6);ax.set_ylabel('View recall at 10% normal object FPR');fig.tight_layout();fig.savefig(OUT/'plots/defect_type_breakdown.png',dpi=150);plt.close(fig)
    # Conditional multiview histogram: denominator is the actual GT-positive camera views.
    od=readcsv(OUT/'object_diagnostics.csv'); fig,ax=plt.subplots(figsize=(6,4))
    for model in ('d2','arcad'):
        rows=[r for r in od if r['model']==model and int(r['gt_positive_view_count'])>0]
        frac=[float(r['positive_view_fraction_detected_at_0.1']) for r in rows];ax.hist(frac,bins=np.linspace(0,1,6),alpha=.5,label=model)
    ax.set_xlabel('Fraction of GT-positive views detected (10% target)');ax.set_ylabel('Objects');ax.legend();fig.tight_layout();fig.savefig(OUT/'plots/multiview_consistency.png',dpi=150);plt.close(fig)
    fm=readcsv(OUT/'failure_overlap.csv');delt=sorted(delta); wins=sum(x>0 for x in delta); losses=sum(x<0 for x in delta); ties=sum(x==0 for x in delta)
    report=f'''# Paired failure analysis: DiNomaly2 vs ArcAD

## Observations

The paired shared test contains five categories: {', '.join(names)}. Low-three categories were selected as failure-enriched and are not representative of all 160 categories. `thyristor` was preselected as the favorable comparison from archived native I-AUROC before new paired maps were generated. Category I-AUROC delta (D2 − ArcAD) has {wins} D2 wins, {losses} ArcAD wins, and {ties} ties; mean {np.mean(delta):.4f}, median {np.median(delta):.4f}, 10th percentile {np.percentile(delta,10):.4f}, worst five {', '.join(f'{names[i]}={delta[i]:.4f}' for i in order[:5])}.

Paired metric values and original native-split context are in `category_metrics.csv`; category deltas in `delta_by_category.csv`; clustered intervals in `bootstrap_ci.csv`. Fixed-threshold confusion tables and overlap are in `confusion_matrices.csv` and `failure_overlap.csv`. `joint_recall_oracle`/union coverage is an oracle diagnostic, not an ensemble result.

## Statistical associations

Bootstrap resamples physical objects, retaining all views and paired models (2,000 draws, seed 20261031). The low-three sample is deliberately enriched for prior failures. Therefore the category summaries are descriptive for this selected subset and must not be extrapolated to the 160-category benchmark. Thresholds use ten held-out normal calibration objects per category; 1%, 5%, and 10% tail estimates are coarse at this calibration size. Thresholded evaluation excludes these objects. AUROC/AP do not use thresholds.

ArcAD seen/unseen comparisons in `seen_unseen_summary.csv` are associations only; seen labels may differ in intrinsic visibility, size, and prevalence. No causal claim is made.

## Hypotheses

Review `view_diagnostics.csv`, `region_diagnostics.csv`, `object_diagnostics.csv`, and `error_taxonomy_counts.csv` for size, contrast, localization, and multiview patterns. These are exploratory descriptions, not causal attribution. The edge/contrast proxy uses original RGB images in the common crop; object foreground masks were unavailable.

## Unverified explanations and limitations

The common map field is source [64:960) in each axis (87.5% of source width/height): ArcAD's 392/448 center crop and the corresponding D2 center crop are resampled to 224x224; source GT masks use nearest-neighbor resizing. See `alignment_validation.json`; it records known positive views with empty source GT masks. Map colors in figures are for display only. Pixel AP/AUROC/AUPRO follow common-grid definitions saved with the config. D2 low-three maps are cached Task 01 CPU inference, while ArcAD maps and added D2 categories were inferred on RTX 3090; strict checkpoint hashes and map provenance are recorded in metadata.

No training or tuned score fusion was performed. Calibration contains only ten normal objects/category, so operational FPR estimates have low resolution. Object cluster intervals are reported for image/object AUROC/AP; other image-level group summaries remain descriptive. See `README.md` for reproduction and exact transforms.
'''
    (OUT/'analysis_report.md').write_text(report)
    # Summary deltas
    writecsv(OUT/'delta_by_category.csv',[{'category':r['category'],'selection_group':r['selection_group'],**{k:v for k,v in r.items() if k.startswith('delta_')}} for r in metrics])
def paired_examples():
    manifest=json.loads((OUT/'shared_manifest.json').read_text()); joined=readcsv(OUT/'joined_predictions.csv'); diag=readcsv(OUT/'view_diagnostics.csv'); tax=readcsv(OUT/'error_taxonomy_counts.csv')
    categories=sorted(manifest['categories']); by={}
    for r in joined:by.setdefault((r['category'],r['image_path']),{})[r['model']]=r
    vd={(r['category'],r['model'],r['image_path']):r for r in diag}; tx={(r['category'],r['model'],r['image_path']):r for r in tax}
    candidates=defaultdict(list)
    for (cat,path),pair in by.items():
        if set(pair)!={'d2','arcad'}:continue
        d,a=pair['d2'],pair['arcad']; ispos=int(d['gt_view_label'])==1
        if ispos:
            dd=vd.get((cat,'d2',path),{}); aa=vd.get((cat,'arcad',path),{})
            detd=dd.get('detected_at_normal_object_fpr_0.1')=='True'; deta=aa.get('detected_at_normal_object_fpr_0.1')=='True'
            if not detd and not deta:candidates['shared_miss'].append((cat,path,pair))
            if detd and not deta:candidates['d2_only_success'].append((cat,path,pair))
            if deta and not detd:candidates['arcad_only_success'].append((cat,path,pair))
            if tx.get((cat,'d2',path),{}).get('heuristic_tags','').find('on_target_but_outranked')>=0:candidates['on_target_but_outranked'].append((cat,path,pair))
            if detd and deta:candidates['detected_control'].append((cat,path,pair))
            # View-dependent: object has both detected and missed positive camera views.
            obj=d['object_key']; op=[x for x in joined if x['category']==cat and x['object_key']==obj and x['model']=='d2' and int(x['gt_view_label'])==1]
            statuses=[vd.get((cat,'d2',x['image_path']),{}).get('detected_at_normal_object_fpr_0.1')=='True' for x in op]
            if any(statuses) and not all(statuses):candidates['view_dependent'].append((cat,path,pair))
        elif int(d['gt_view_label'])==0:
            fd=vd.get((cat,'d2',path),{}).get('detected_at_normal_object_fpr_0.1')=='True'
            fa=vd.get((cat,'arcad',path),{}).get('detected_at_normal_object_fpr_0.1')=='True'
            if fd or fa:
                sc=max(float(d['score_view']),float(a['score_view'])); candidates['high_score_normal'].append((cat,path,pair,sc))
    selected=[]
    for kind,items in candidates.items():
        if kind=='high_score_normal': items=sorted(items,key=lambda x:-x[3])
        elif kind=='detected_control': items=sorted(items,key=lambda x:(x[0],x[1]))
        else: items=sorted(items,key=lambda x:(x[0],x[1]))
        if not items:continue
        cat,path,pair=items[0][:3]; d,a=pair['d2'],pair['arcad']; rec=next(r for r in manifest['categories'][cat]['shared_test_images'] if r['image_path']==path)
        source=Path('/media/avlab/7458a474-6ea2-4b90-b232-373c3df8d329/real_iad_variety_1024')/cat/path
        image=np.asarray(Image.open(source).convert('RGB').crop((64,64,960,960)).resize((224,224),Image.Resampling.BILINEAR))
        with np.load(OUT/'maps'/f'common_d2_{cat}.npz') as z:
            paths=z['image_paths'].astype(str).tolist();idx=paths.index(path);gt=z['gt_masks'][idx].astype(bool);md=z['smoothed_maps'][idx]
        with np.load(OUT/'maps'/f'common_arcad_{cat}.npz') as z:ma=z['smoothed_maps'][z['image_paths'].astype(str).tolist().index(path)]
        fig,axs=plt.subplots(1,5,figsize=(15,3.2));axs[0].imshow(image);axs[0].imshow(gt,cmap='Reds',alpha=.5);axs[0].set_title('Common crop + GT')
        for ax,arr,name in zip(axs[1:3],[md,ma],['D2 heatmap','ArcAD heatmap']):
            ax.imshow(arr,cmap='magma',vmin=np.percentile(arr,1),vmax=np.percentile(arr,99));ax.set_title(name+'\ncolors normalized per model/view')
        axs[3].imshow(image);axs[3].imshow(gt,cmap='Reds',alpha=.35);axs[3].contour(md>=np.quantile(md,.99),levels=[.5],colors='cyan',linewidths=.8);axs[3].set_title('D2 top 1% + GT')
        axs[4].imshow(image);axs[4].imshow(gt,cmap='Reds',alpha=.35);axs[4].contour(ma>=np.quantile(ma,.99),levels=[.5],colors='cyan',linewidths=.8);axs[4].set_title('ArcAD top 1% + GT')
        for ax in axs:ax.axis('off')
        qd=vd.get((cat,'d2',path),{});qa=vd.get((cat,'arcad',path),{})
        fig.suptitle(f'{kind}: {cat} / {rec["object_key"]} / {rec["view_id"]} | label={rec["label"]} type={rec["anomaly_class"]}\nD2 score={float(d["score_view"]):.4g}, Q={qd.get("Q_within_view_pixel_AUROC","NA")}, top1 recall={qd.get("top1_recall","NA")} | ArcAD score={float(a["score_view"]):.4g}, Q={qa.get("Q_within_view_pixel_AUROC","NA")}, top1 recall={qa.get("top1_recall","NA")}',fontsize=8)
        fig.tight_layout(rect=(0,0,1,.84));dest=OUT/'paired_examples'/f'{kind}.png';fig.savefig(dest,dpi=150);plt.close(fig)
        cam=re.search(r'_C([1-5])_',path)
        rule='normal view with highest model score among cases where at least one model is a false positive at the fixed 10% normal-object target' if kind=='high_score_normal' else 'first lexicographic matched record satisfying the paired 10% normal-object operating-point rule'
        selected.append({'example_type':kind,'selection_rule':rule,'category':cat,'object_key':rec['object_key'],'image_path':path,'view_id':rec['view_id'],'camera_id':f'C{cam.group(1)}' if cam else '','gt_label':rec['label'],'defect_type':rec['anomaly_class'],'d2_score':d['score_view'],'arcad_score':a['score_view'],'d2_Q':qd.get('Q_within_view_pixel_AUROC'),'arcad_Q':qa.get('Q_within_view_pixel_AUROC'),'figure':str(dest.relative_to(OUT)),'colors':'normalized per model/view; compare spatial patterns, not absolute intensity'})
    writecsv(OUT/'selected_examples.csv',selected)
def main():
    (OUT/'plots').mkdir(exist_ok=True); add_camera_ids(); ci_rows();mechanism_model();plots_and_report();paired_examples();print('Bootstrap, plots, examples and report complete')
if __name__=='__main__':main()
