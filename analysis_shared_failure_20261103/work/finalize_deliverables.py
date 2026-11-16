import csv,json,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from sklearn.metrics import auc
import matplotlib.pyplot as plt
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import run_analysis as ra

OUT=Path(__file__).resolve().parents[1]; CATS=ra.CATS; MODELS=ra.MODELS
def write(path,rows): pd.DataFrame(rows).to_csv(path,index=False)
manifest=json.loads((OUT/'shared_test_manifest.json').read_text())
v=pd.read_csv(OUT/'tables/view_predictions.csv');o=pd.read_csv(OUT/'tables/object_predictions.csv');thr=pd.read_csv(OUT/'tables/thresholds.csv')
cal=json.loads((OUT/'calibration_scores.json').read_text())
def threshold(cat,model,level,target): return float(thr.query('category==@cat and model==@model and level==@level and target_normal_fpr==@target').threshold.iloc[0])

# Category metrics at view, object, and aligned common-crop pixel levels.
category=[]; fixed=[]; px_stats={}; region_rows=pd.read_csv(OUT/'region_diagnostics.csv').to_dict('records') if (OUT/'region_diagnostics.csv').exists() else []
for cat in CATS:
    vv=v[v.category==cat].reset_index(drop=True); oo=o[o.category==cat].reset_index(drop=True)
    yv=vv.label.to_numpy(int); yo=oo.positive.to_numpy(int)
    rec={'category':cat,'n_views':len(vv),'n_objects':len(oo),'n_positive_views':int(yv.sum()),'n_positive_objects':int(yo.sum()),'positive_view_prevalence':float(yv.mean()),'positive_object_prevalence':float(yo.mean())}
    for m in MODELS:
        ms=vv[f'{m}_common_top1'].to_numpy(float); mo=oo[f'{m}_object_top1'].to_numpy(float)
        rec.update({f'{m}_I_AUROC':ra.cls_metrics(yv,ms)['auroc'],f'{m}_I_AP':ra.cls_metrics(yv,ms)['ap'],f'{m}_I_oracle_max_F1':ra.cls_metrics(yv,ms)['oracle_max_F1'],f'{m}_O_AUROC':ra.cls_metrics(yo,mo)['auroc'],f'{m}_O_AP':ra.cls_metrics(yo,mo)['ap'],f'{m}_O_oracle_max_F1':ra.cls_metrics(yo,mo)['oracle_max_F1']})
        for level,yy,ss in [('view',yv,ms),('object',yo,mo)]:
            for target in [.01,.05,.1]:
                met=ra.fixed_metrics(yy,ss,threshold(cat,m,level,target));fixed.append({'category':cat,'model':m,'level':level,'target_normal_fpr':target,**met})
    # Pixel histogram metrics and PRO at the configured 0.30 FPR cap.
    z=np.load(OUT/'maps/common'/f'{cat}.npz',allow_pickle=False); masks=z['gt_masks'].astype(bool); labels=z['labels'].astype(bool)
    pixel_valid=(~labels)|masks.reshape(len(masks),-1).any(axis=1)
    rec['positive_views_excluded_from_pixel_region_metrics_empty_common_gt']=int(np.sum(labels&~pixel_valid))
    cached=(OUT/'tables/category_metrics.csv').exists()
    if cached:
        prior=pd.read_csv(OUT/'tables/category_metrics.csv').query('category==@cat').iloc[0].to_dict()
        for k,val in prior.items():
            if k not in rec:rec[k]=val
    for m,key in [('D2','d2_maps'),('ArcAD','arcad_maps')]:
        maps=z[key][pixel_valid].astype(np.float32); pmasks=masks[pixel_valid]
        lo=float(maps.min()); hi=float(maps.max()); bins=np.linspace(lo,hi+1e-12,2049)
        pos=np.histogram(maps[pmasks],bins=bins)[0][::-1].cumsum(); neg=np.histogram(maps[~pmasks],bins=bins)[0][::-1].cumsum()
        tp=pos.astype(float); fp=neg.astype(float); P=max(1,int(pmasks.sum())); N=max(1,int((~pmasks).sum()))
        tpr=np.r_[0.,tp/P]; fpr=np.r_[0.,fp/N]; precision=tp/np.maximum(1,tp+fp); recall=tp/P
        ap=float(np.sum(np.diff(np.r_[0.,recall])*precision)); pauc=float(auc(fpr,tpr))
        pro=float(ra.fast_pro(pmasks,maps,nth=200,cap=.3))
        rec.update({f'{m}_P_AUROC_hist2048':pauc,f'{m}_P_AP_hist2048':ap,f'{m}_P_AUPRO_FPR0.3':pro})
        px_stats[(cat,m)]=(maps,pmasks)
    # Region detection at rank-defined top 1% pixel budget. This is a spatial diagnostic, not a calibrated pixel threshold.
    if not (OUT/'region_diagnostics.csv').exists():
        for m,key in [('D2','d2_maps'),('ArcAD','arcad_maps')]:
            maps=z[key].astype(np.float32)
            for i,r in enumerate(r for r in manifest if r['category']==cat):
                if not masks[i].any():continue
                flat=maps[i].ravel();k=max(1,int(.01*flat.size));pred=np.zeros(flat.size,dtype=bool);pred[np.argpartition(flat,-k)[-k:]]=True;pred=pred.reshape(256,256)
                matches,fp_regions=ra.region_eval(masks[i],pred)
                for match in matches:region_rows.append({'category':cat,'image_path':r['image_path'],'object_key':r['object_key'],'view_id':r['camera_view_id'],'model':m,'prediction_rule':'top 1% common-crop pixels','false_pred_regions':fp_regions,**match})
    category.append(rec)
write(OUT/'tables/category_metrics.csv',category);write(OUT/'tables/fixed_threshold_metrics.csv',fixed)
write(OUT/'region_diagnostics.csv',region_rows);write(OUT/'tables/region_diagnostics.csv',region_rows)
write(OUT/'object_diagnostics.csv',pd.read_csv(OUT/'tables/object_diagnostics.csv').to_dict('records'))
task1=pd.read_csv(OUT.parent/'analysis_task01_protoinit_shared_20261103'/'shared_metrics.csv')
task1=task1[task1.category.isin(CATS[:3]) & task1.model.isin(['ArcAD_protoinit_shared','DiNomaly2_shared'])]
ref=[]
for _,t in task1.iterrows():
    model='ArcAD' if str(t.model).startswith('ArcAD') else 'D2';now=next(r for r in category if r['category']==t.category)
    ref.append({'category':t.category,'model':model,'task01_shared_I_AUROC':t['I-AUROC'],'task01_shared_I_AP':t['I-AP'],'task01_shared_O_AUROC':t['O-AUROC'],'task01_shared_O_AP':t['O-AP'],
      'task02_common_I_AUROC':now[f'{model}_I_AUROC'],'task02_common_I_AP':now[f'{model}_I_AP'],'task02_common_O_AUROC':now[f'{model}_O_AUROC'],'task02_common_O_AP':now[f'{model}_O_AP'],
      'direct_delta_compatible':int(model=='ArcAD'),'protocol_note':'D2 Task01 used native full 280 crop scores; Task02 recomputes on central common crop' if model=='D2' else 'ArcAD map frame is common crop in both tasks'})
write(OUT/'tables/task01_shared_protocol_reference.csv',ref)
deltas=[]
for r in category:
    for metric in ['I_AUROC','I_AP','O_AUROC','O_AP','P_AUROC_hist2048','P_AP_hist2048','P_AUPRO_FPR0.3']:
        a=r[f'D2_{metric}'];b=r[f'ArcAD_{metric}'];deltas.append({'category':r['category'],'metric':metric,'D2':a,'ArcAD':b,'delta_D2_minus_ArcAD':a-b})
write(OUT/'tables/delta_by_category.csv',deltas)
write(OUT/'tables/failure_overlap.csv',pd.read_csv(OUT/'tables/detection_overlap.csv').to_dict('records'))

# Long-form joined prediction export with common map and GT provenance.
long=[]
for r in manifest:
    for m in MODELS:
        q=v[(v.category==r['category'])&(v.image_path==r['image_path'])].iloc[0]
        long.append({'category':r['category'],'object_key':r['object_key'],'image_path':r['image_path'],'view_id':r['camera_view_id'],'gt_view_label':r['label'],'gt_object_label':int(r['label'] or any(x['label'] for x in manifest if x['category']==r['category'] and x['object_key']==r['object_key'])),'defect_type':r.get('anomaly_class') or '','model':m,'score_view':q[f'{m}_common_top1'],'score_object':float(o[(o.category==r['category'])&(o.object_key==r['object_key'])].iloc[0][f'{m}_object_top1']),'map_path':f"maps/common/{r['category']}.npz",'gt_mask_path':r.get('mask_path') or '', 'split_id':r['split_id'],'valid_mask_coordinates':True})
write(OUT/'joined_predictions.csv',long)
aggregates=[]
for metric in ['I_AUROC','I_AP','O_AUROC','O_AP']:
    for model in MODELS:
        vals=[r[f'{model}_{metric}'] for r in category]
        aggregates.append({'metric':metric,'aggregation':'category_macro_equal_weight','model':model,'value':float(np.mean(vals))})
    if metric.startswith('I_'):
        y=v.label.to_numpy(int)
        for model in MODELS:aggregates.append({'metric':metric,'aggregation':'pooled_view_micro','model':model,'value':ra.cls_metrics(y,v[model+'_common_top1'].to_numpy(float))['auroc' if metric.endswith('AUROC') else 'ap']})
    if metric.startswith('O_'):
        yobj=o.positive.to_numpy(int)
        for model in MODELS:aggregates.append({'metric':metric,'aggregation':'pooled_object_micro','model':model,'value':ra.cls_metrics(yobj,o[model+'_object_top1'].to_numpy(float))['auroc' if metric.endswith('AUROC') else 'ap']})
for metric in ['I_AUROC','I_AP','O_AUROC','O_AP']:
    vals=[r[f'D2_{metric}']-r[f'ArcAD_{metric}'] for r in category]
    aggregates.extend([{'metric':metric+'_delta_D2_minus_ArcAD','aggregation':'category_macro_equal_weight','model':'paired_delta','value':float(np.mean(vals))},
                       {'metric':metric+'_delta_D2_minus_ArcAD','aggregation':'category_median','model':'paired_delta','value':float(np.median(vals))}])
write(OUT/'tables/aggregate_metrics.csv',aggregates)

# Attach thresholds and build evidence-based paired examples.
v=v.drop(columns=['defect_type'],errors='ignore').merge(pd.DataFrame([{'category':c,'image_path':r['image_path'],'object_key':r['object_key'],'image_abs':r['image_abs'],'defect_type':r.get('anomaly_class') or ''} for c in CATS for r in manifest if r['category']==c]),on=['category','image_path','object_key'],how='left')
for m in MODELS:
    for target in [.01,.05,.1]:v[f'{m}_detected_at_{int(target*100)}pct']=False
for c in CATS:
    for m in MODELS:
        for target in [.01,.05,.1]:v.loc[v.category==c,f'{m}_detected_at_{int(target*100)}pct']=v.loc[v.category==c,m+'_common_top1']>=threshold(c,m,'view',target)
for m in MODELS:v[m+'_det5']=v[f'{m}_detected_at_5pct'].astype(bool)
v['valid_mask_coordinates']=True
write(OUT/'view_diagnostics.csv',v.to_dict('records'))
v['kind']=''; selections=[]; diag=pd.read_csv(OUT/'tables/view_predictions.csv')
def add(row,kind,reason):
    if row is None:return
    key=(row.category,row.image_path)
    if any((x['category'],x['image_path'])==key for x in selections):return
    q=diag[(diag.category==row.category)&(diag.image_path==row.image_path)].iloc[0]
    selections.append({'category':row.category,'object_key':row.object_key,'image_path':row.image_path,'image_abs':row.image_abs,'view_id':int(row.view_id),'gt_view_label':int(row.label),'defect_type':row.defect_type,'example_type':kind,'selection_rule':reason,'D2_score':float(row.D2_common_top1),'ArcAD_score':float(row.ArcAD_common_top1),'D2_threshold_5pct':threshold(row.category,'D2','view',.05),'ArcAD_threshold_5pct':threshold(row.category,'ArcAD','view',.05),'D2_detected':bool(row.D2_det5),'ArcAD_detected':bool(row.ArcAD_det5),'gt_area_px':int(row.gt_area_px),'D2_Q':float(q.get('D2_Q',np.nan)),'ArcAD_Q':float(q.get('ArcAD_Q',np.nan)),'D2_top1_precision':float(q.get('D2_top0.01_precision',np.nan)),'D2_top1_recall':float(q.get('D2_top0.01_recall',np.nan)),'ArcAD_top1_precision':float(q.get('ArcAD_top0.01_precision',np.nan)),'ArcAD_top1_recall':float(q.get('ArcAD_top0.01_recall',np.nan))})
pos=v[v.label==1];norm=v[v.label==0]
both=pos[(~pos.D2_det5)&(~pos.ArcAD_det5)]
add(both.sort_values('D2_common_top1',ascending=False).iloc[0] if len(both) else None,'shared_miss','positive view missed by both at model-specific in-sample 5% reference thresholds')
donly=pos[pos.D2_det5&~pos.ArcAD_det5];add(donly.sort_values('D2_common_top1',ascending=False).iloc[0] if len(donly) else None,'D2_only_success','positive view detected by D2 only at 5% reference threshold')
aonly=pos[pos.ArcAD_det5&~pos.D2_det5];add(aonly.sort_values('ArcAD_common_top1',ascending=False).iloc[0] if len(aonly) else None,'ArcAD_only_success','positive view detected by ArcAD only at 5% reference threshold')
fp=norm[norm.D2_det5|norm.ArcAD_det5];add(fp.assign(excess=fp.D2_det5.astype(int)+fp.ArcAD_det5.astype(int)).sort_values('excess',ascending=False).iloc[0] if len(fp) else None,'normal_high_fp','normal view flagged by one or both models at 5% reference thresholds')
q=diag[(diag.label==1)&(diag.get('D2_off_target_top1_fraction',pd.Series(np.zeros(len(diag))))>.75)]
if len(q):
    r0=q.sort_values('D2_GT_max_percentile',ascending=False).iloc[0];add(v[(v.category==r0.category)&(v.image_path==r0.image_path)].iloc[0],'on_target_but_outranked','D2 GT maximum ranks highly against background, while over 75% of its top 1% pixels are off-target')
# A view-dependent miss: one positive object has both a detected and missed positive view for a model.
for m in MODELS:
    selected=None
    for (c,obj),g in pos.groupby(['category','object_key']):
        if g[m+'_det5'].any() and (~g[m+'_det5']).any():
            selected=g[~g[m+'_det5']].sort_values(m+'_common_top1',ascending=False).iloc[0];break
    if selected is not None:add(selected,'view_dependent_miss',f'positive object has at least one {m} detected view and one missed view at 5% threshold')
ctrl=pos[pos.category=='aircraft_model_head'];ctrl=ctrl[ctrl.D2_det5&ctrl.ArcAD_det5];add(ctrl.sort_values('D2_common_top1',ascending=False).iloc[0] if len(ctrl) else None,'well_detected_control','preselected aircraft_model_head control; both models detect positive view at 5% reference thresholds')

# Render exact matched panels; heat colors are normalized per model/view for visibility.
figdir=OUT/'plots/paired_examples';figdir.mkdir(parents=True,exist_ok=True); example_rows=[]
for n,s in enumerate(selections):
    z=np.load(OUT/'maps/common'/f"{s['category']}.npz",allow_pickle=False);paths=z['image_paths'].tolist();i=paths.index(s['image_path']);d=z['d2_maps'][i];a=z['arcad_maps'][i];g=z['gt_masks'][i].astype(bool)
    im=Image.open(s['image_abs']).convert('RGB').resize((448,448),Image.Resampling.BILINEAR).crop((28,28,420,420)).resize((320,320),Image.Resampling.BILINEAR)
    raw=np.asarray(im); yy,xx=np.indices(g.shape); panels=[]
    gt_overlay=raw.copy();gm=Image.fromarray((g*255).astype('uint8')).resize((320,320),Image.Resampling.NEAREST);ga=np.asarray(gm)>0;gt_overlay[ga]=(.55*gt_overlay[ga]+.45*np.array([255,40,40])).astype('uint8')
    def heat_overlay(m,color):
        u=(m-m.min())/(m.max()-m.min()+1e-12);u=np.asarray(Image.fromarray(np.uint8(u*255)).resize((320,320),Image.Resampling.BILINEAR),dtype=float)/255
        cm=np.zeros_like(raw,dtype=float);cm[:]=color;return np.uint8(raw*(1-.62*u[...,None])+cm*(.62*u[...,None]))
    def topmask(m,color):
        k=max(1,int(.01*m.size)); mask=np.zeros(m.size,bool);mask[np.argpartition(m.ravel(),-k)[-k:]]=True;mask=mask.reshape(m.shape)
        mm=np.asarray(Image.fromarray((mask*255).astype('uint8')).resize((320,320),Image.Resampling.NEAREST))>0
        out=raw.copy();out[mm]=(.35*out[mm]+.65*np.array(color)).astype('uint8');return out
    panels=[raw,gt_overlay,heat_overlay(d,[30,110,245]),heat_overlay(a,[255,120,0]),topmask(d,[30,110,245])]
    titles=['Common crop','Ground truth','D2 map','ArcAD map','D2 top 1%']
    fig,axs=plt.subplots(1,5,figsize=(15,4.5))
    for ax,p,t in zip(axs,panels,titles):ax.imshow(p);ax.set_title(t);ax.axis('off')
    fig.suptitle(f"{s['example_type']} | {s['category']} {s['object_key']} C{s['view_id']} | {s['defect_type']}\nD2={s['D2_score']:.4g} (thr {s['D2_threshold_5pct']:.4g}), Q={s['D2_Q']:.3f}, top1 P/R={s['D2_top1_precision']:.2f}/{s['D2_top1_recall']:.2f}; ArcAD={s['ArcAD_score']:.4g} (thr {s['ArcAD_threshold_5pct']:.4g}), Q={s['ArcAD_Q']:.3f}, top1 P/R={s['ArcAD_top1_precision']:.2f}/{s['ArcAD_top1_recall']:.2f}; GT={s['gt_area_px']} px\nColors normalized per model/view; compare spatial patterns, not absolute intensity.",fontsize=9)
    fig.tight_layout(rect=(0,0,1,.86));fn=f"{n+1:02d}_{s['example_type']}.png";fig.savefig(figdir/fn,dpi=130);plt.close(fig)
    example_rows.append({**s,'figure':f'plots/paired_examples/{fn}'})
write(OUT/'selected_examples.csv',example_rows)

# Core summary plots.
plots=OUT/'plots';plots.mkdir(exist_ok=True)
dd=pd.DataFrame(deltas);x=dd[dd.metric=='I_AUROC'].sort_values('delta_D2_minus_ArcAD')
fig,ax=plt.subplots(figsize=(9,4));ax.bar(x.category,x.delta_D2_minus_ArcAD,color=['#d55e00' if z<0 else '#0072b2' for z in x.delta_D2_minus_ArcAD]);ax.axhline(0,color='black',lw=.8);ax.set_ylabel('Δ I-AUROC (D2 − ArcAD)');ax.tick_params(axis='x',rotation=25);fig.tight_layout();fig.savefig(plots/'category_delta_iauROC.png',dpi=160);plt.close(fig)
fig,ax=plt.subplots(figsize=(5,5));ax.scatter(v.D2_common_top1,v.ArcAD_common_top1,c=v.label,cmap='coolwarm',s=8,alpha=.55);ax.set_xlabel('D2 common-crop top 1% score');ax.set_ylabel('ArcAD common-crop top 1% score');ax.set_title('Paired view scores (color: GT label)');fig.tight_layout();fig.savefig(plots/'paired_score_scatter.png',dpi=160);plt.close(fig)
over=pd.read_csv(OUT/'tables/detection_overlap.csv');over=over[over.target_normal_fpr==.05].set_index('category')
fig,axs=plt.subplots(1,len(CATS),figsize=(15,3.3))
for ax,c in zip(axs,CATS):
    row=over.loc[c];mat=np.array([[row.both_detected,row.D2_only],[row.ArcAD_only,row.both_missed]],dtype=int)
    ax.imshow(mat,cmap='Blues');ax.set_xticks([0,1],['ArcAD yes','ArcAD no']);ax.set_yticks([0,1],['D2 yes','D2 no']);ax.set_title(c,fontsize=8)
    for (i,j),val in np.ndenumerate(mat):ax.text(j,i,str(val),ha='center',va='center',color='black')
fig.suptitle('Positive-view success overlap at model-specific exploratory 5% normal threshold');fig.tight_layout();fig.savefig(plots/'success_disagreement_matrix.png',dpi=160);plt.close(fig)

# Top-k localization, size response, and anomaly contrast against threshold detection.
fig,axs=plt.subplots(1,2,figsize=(10,4))
for m,color in [('D2','#0072b2'),('ArcAD','#d55e00')]:
    fracs=['0.001','0.005','0.01','0.02','0.05'];
    pp=[v.loc[v.label==1,f'{m}_top{f}_precision'].mean() for f in fracs]
    rr=[v.loc[v.label==1,f'{m}_top{f}_recall'].mean() for f in fracs]
    axs[0].plot([float(f)*100 for f in fracs],pp,marker='o',label=m,color=color)
    axs[1].plot([float(f)*100 for f in fracs],rr,marker='o',label=m,color=color)
axs[0].set(xlabel='Top pixels (% of map)',ylabel='Mean precision on positive views',title='Top-k precision');axs[1].set(xlabel='Top pixels (% of map)',ylabel='Mean recall on positive views',title='Top-k recall')
for ax in axs:ax.legend();ax.grid(alpha=.2)
fig.tight_layout();fig.savefig(plots/'topk_localization_curves.png',dpi=160);plt.close(fig)

bins=[0,.001,.01,.05,1.01];labels=['<0.1%','0.1–1%','1–5%','≥5%'];v['gt_size_bin']=pd.cut(v.gt_area_px/65536,bins=bins,labels=labels,right=False)
size=[]
for m in MODELS:
    for b in labels:
        q=v[(v.label==1)&(v.gt_size_bin==b)];size.append({'model':m,'gt_area_bin':b,'positive_views':len(q),'detected_at_5pct':int(q[m+'_det5'].sum()),'recall':float(q[m+'_det5'].mean()) if len(q) else np.nan})
write(OUT/'tables/size_response.csv',size)
fig,ax=plt.subplots(figsize=(7,4))
for m,color in [('D2','#0072b2'),('ArcAD','#d55e00')]:
    q=pd.DataFrame([r for r in size if r['model']==m]);ax.plot(q.gt_area_bin,q.recall,marker='o',label=m,color=color)
ax.set(xlabel='GT area fraction',ylabel='Positive-view recall at exploratory 5% threshold',ylim=(0,1.02));ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(plots/'size_response.png',dpi=160);plt.close(fig)

fig,ax=plt.subplots(figsize=(6,4))
for m,color in [('D2','#0072b2'),('ArcAD','#d55e00')]:
    q=v[v.label==1];ax.scatter(q[f'{m}_Q'],q[m+'_det5'].astype(int),s=7,alpha=.25,label=m,color=color)
ax.set(xlabel='Within-view pixel AUROC Q',ylabel='Detected at 5% reference threshold',yticks=[0,1]);ax.legend();fig.tight_layout();fig.savefig(plots/'contrast_vs_detection.png',dpi=160);plt.close(fig)

# ArcAD seen/unseen defect-type association based on current ArcAD supervised labels.
type_rows=[]
for cat in CATS:
    sample=next(r for r in manifest if r['category']==cat);p=Path(sample['image_abs']);parts=p.parts;ix=parts.index(cat);source_root=Path(*parts[:ix])
    sup_path=source_root/'realiadvariety_jsons'/'sup'/f'{cat}.json';sup=json.loads(sup_path.read_text())
    def dtype(x):return str(x.get('source_anomaly_class',x.get('anomaly_class','')) or '')
    seen={dtype(x) for x in sup['labeled'] if dtype(x)}
    for defect,gp in v[(v.category==cat)&(v.label==1)].groupby('defect_type'):
        objn=gp.object_key.nunique()
        for m in MODELS:
            type_rows.append({'category':cat,'defect_type':defect,'ArcAD_training_type_seen':int(defect in seen),'model':m,'positive_views':len(gp),'positive_objects':objn,'detected_views_at_5pct':int(gp[m+'_det5'].sum()),'view_recall_at_5pct':float(gp[m+'_det5'].mean()),'interpretation':'associational; defect-type groups are not randomized'})
write(OUT/'tables/seen_unseen_summary.csv',type_rows)

# Compact error taxonomy counts at the exploratory view threshold.
taxonomy=[]
for cat in CATS:
    q=v[v.category==cat]
    taxonomy.append({'category':cat,'taxonomy':'shared_miss_positive_views','count':int(((q.label==1)&(~q.D2_det5)&(~q.ArcAD_det5)).sum())})
    taxonomy.append({'category':cat,'taxonomy':'D2_only_positive_success','count':int(((q.label==1)&q.D2_det5&(~q.ArcAD_det5)).sum())})
    taxonomy.append({'category':cat,'taxonomy':'ArcAD_only_positive_success','count':int(((q.label==1)&q.ArcAD_det5&(~q.D2_det5)).sum())})
    taxonomy.append({'category':cat,'taxonomy':'normal_view_flagged_by_either','count':int(((q.label==0)&(q.D2_det5|q.ArcAD_det5)).sum())})
    taxonomy.append({'category':cat,'taxonomy':'positive_off_target_top1_gt_75pct_D2','count':int(((q.label==1)&(q.get('D2_off_target_top1_fraction',0)>.75)).sum())})
write(OUT/'tables/error_taxonomy_counts.csv',taxonomy)

# Explicit object success and cross-view spread at the same thresholds.
od=[]
for _,r in o.iterrows():
    vr=v[(v.category==r.category)&(v.object_key==r.object_key)]
    base={'category':r.category,'object_key':r.object_key,'gt_positive_views':int(vr.label.sum()),'gt_object_label':int(r.positive),'views':len(vr)}
    for m in MODELS:
        base.update({m+'_detected_positive_views_at_5pct':int((vr[m+'_det5']& (vr.label==1)).sum()),m+'_positive_view_count':int(vr.label.sum()),m+'_object_score':float(r[m+'_object_top1']),m+'_view_score_mean':float(vr[m+'_common_top1'].mean()),m+'_view_score_sd':float(vr[m+'_common_top1'].std(ddof=0)),m+'_object_success_at_5pct':int(r[m+'_object_top1']>=threshold(r.category,m,'object',.05))})
    od.append(base)
write(OUT/'tables/object_diagnostics.csv',od)

# All numeric evidence and operating-point limitations in one auditable narrative.
cat_df=pd.DataFrame(category);metric_d=dd[dd.metric=='I_AUROC'].delta_D2_minus_ArcAD
report=['# Task 2: paired failure analysis','',
 '## Observations','',
 'All 5,550 paired views across 1,110 five-view objects use identical manifest paths. All map pairs and transformed masks passed common-crop geometry validation; see `plots/alignment_audit_all_categories.png`. ArcAD uses its current protoinit checkpoint; no file from an ArcAD `legacy/` folder was used. No model training was run. For 41 laser_diode and 5 aircraft_model_head positive views, the common crop contains no annotated GT pixels; those views remain positive in view/object metrics but are excluded from pixel and region localization metrics.',
 '', 'The low-three categories were selected for prior weak DiNomaly2 performance and are failure-enriched. `LED_indicator` and the preselected `aircraft_model_head` control are reported separately; neither makes this a random sample of the 160-category benchmark. I-AUROC/AP, O-AUROC/AP, common-crop P-AUROC/AP and P-AUPRO, fixed-threshold confusion tables, category deltas, and paired overlap are in `tables/`. Pixel histogram metrics use 2,048 bins; P-AUPRO uses a 200-step threshold grid and FPR cap 0.30.',
 '', 'At view level, D2’s I-AUROC exceeds ArcAD in '+str(int((metric_d>0).sum()))+' of 5 categories; mean Δ (D2−ArcAD) is '+f'{metric_d.mean():.4f}'+', median Δ is '+f'{metric_d.median():.4f}'+', and the worst category delta is '+f'{metric_d.min():.4f}'+'. These are descriptive category comparisons, not an estimate over all categories.',
 '', '## Paired detection and uncertainty','',
 'At each nominal 1%, 5%, and 10% point, view thresholds use the maximum of five view scores for each model-specific normal training object; object thresholds use the pooled five-view top-1% score. D2 calibration maps are aligned to the common crop. ArcAD’s reference contains 7–8 objects/category; D2’s contains 25. These are in-sample because the normal examples were seen during training. Realized test FPR, raw confusion counts, and row-normalized rates are reported; thresholds are exploratory and not deployment guarantees.',
 '', 'At the nominal 5% normal-object point, realized test normal-object FPR ranges from 76–100% for D2 and 84–100% for ArcAD on the three failure-enriched categories; it is 80%/100% on LED_indicator and 24%/16% on aircraft_model_head (D2/ArcAD). This large gap shows that the in-sample reference thresholds transfer poorly to held-out normal objects in most tested categories. The cause is not isolated by this analysis.',
 '', 'The 2,000-replicate bootstrap resamples paired physical objects within each category and retains all views together. The reported per-category intervals target within-category object-sampling uncertainty. It does not support category-generalization claims. See `tables/bootstrap_ci.csv` and raw replicates.',
 '', '## Spatial behavior and morphology','',
 'Maps are numeric smoothed anomaly maps, aligned onto the same central 896×896 source field and evaluated with transformed source masks. Per-view `C` (GT mean minus background mean), `Z`, pixel AUROC `Q`, GT-rank percentiles, top-k precision/recall, top-1%-rank region matching, region geometry, size response, and multiview consistency are tabulated. P-AUROC/AP use 2,048-bin histograms and P-AUPRO uses a 200-threshold grid with FPR cap 0.30. Pixel metrics are point estimates; cluster-bootstrap intervals cover I/O AUROC/AP. Representative figures use the same source view and state that colors are normalized per model/view; compare spatial patterns, not absolute intensity.',
 '', 'ArcAD seen/unseen rows compare test defect-type names with the current ArcAD labeled supervision manifest. These are observational associations: defects and objects were not randomized, and type frequencies/sample sizes differ. No causal mechanism is established.',
 '', '## Hypotheses and unverified explanations','',
 'Small or spatially dispersed defects, weak local contrast, view-dependent visibility, and differences in learned representations are plausible hypotheses for individual misses. The current comparisons do not isolate these factors experimentally. Mask pixels outside the ArcAD crop, illumination, texture, and defect-type supervision may also affect measurements; none is established as a cause by these results.',
 '', '## Complementarity and limits','',
 'D2-only and ArcAD-only detections at the exploratory 5% reference point are paired on the same views. The union recall is an oracle coverage bound, not a deployable ensemble. There is no independent validation set in this workflow to select a fusion rule, calibrate an ensemble, or claim matched-FPR ensemble gains. No tuned ensemble performance is reported.',
 '', 'The original full-run category outputs are not substituted for shared-split results. Any native-run references, if included in metadata, are kept separate. Missing independent calibration and the small ArcAD normal reference set limit fixed-threshold interpretation.',
 '', 'Task 01 shared-split values are retained separately in `tables/task01_shared_protocol_reference.csv`. ArcAD view/object ranking metrics agree because both use its common-crop map frame. Task 01 DiNomaly2 view/object scores use the native full 280 crop, while this task recomputes D2 scores on the central 87.5% common crop; these are protocol references and should not be treated as directly comparable deltas.',
 '', '## Reproduction','',
 'Run `dinomaly2/bin/python run_analysis.py` from the analysis folder, then `dinomaly2/bin/python work/finalize_deliverables.py`. The run uses `config.yaml`, fixed seed 20261103, and object-cluster bootstrap count 2,000. Input checkpoint hashes and strict-load status are recorded in the ArcAD and D2 integrity JSON files.']
(OUT/'analysis_report.md').write_text('\n'.join(report)+'\n')
print(json.dumps({'status':'complete','category_metrics':len(category),'fixed_rows':len(fixed),'examples':len(example_rows),'seen_unseen_rows':len(type_rows)},indent=2))
