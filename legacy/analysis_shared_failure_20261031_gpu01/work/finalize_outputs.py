#!/usr/bin/env python3
"""Add pooled/macro summaries, row-normalized confusion rates and env provenance."""
import csv,json,platform,sys,hashlib
from pathlib import Path
import numpy as np
from sklearn.metrics import roc_auc_score,average_precision_score
OUT=Path(__file__).resolve().parent.parent
def read(p):
    with open(p,encoding='utf-8') as f:return list(csv.DictReader(f))
def write(p,rows):
    with open(p,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),extrasaction='ignore');w.writeheader();w.writerows(rows)
def main():
    conf=read(OUT/'confusion_matrices.csv')
    for r in conf:
        tn,fp,fn,tp=(int(r[k]) for k in ('tn','fp','fn','tp'))
        r.update({'negative_support':tn+fp,'positive_support':fn+tp,'row_normalized_TN_rate':tn/max(1,tn+fp),'row_normalized_FP_rate':fp/max(1,tn+fp),'row_normalized_FN_rate':fn/max(1,fn+tp),'row_normalized_TP_rate':tp/max(1,fn+tp),'specificity':tn/max(1,tn+fp),'miss_rate':fn/max(1,fn+tp)})
    write(OUT/'confusion_matrices.csv',conf)
    # Paired GT-region detection on the shared physical field.
    regs=read(OUT/'region_diagnostics.csv'); paired={}
    for r in regs:paired.setdefault((r['category'],r['image_path'],r['region_id'],r['pixel_fpr_target']),{})[r['model']]=r
    prows=[]
    for key,p in paired.items():
        if set(p)!={'d2','arcad'}:continue
        d,a=p['d2'],p['arcad']; row={'category':key[0],'image_path':key[1],'region_id':key[2],'pixel_fpr_target':key[3],'object_key':d['object_key'],'view_id':d['view_id'],'defect_type':d['defect_type'],'camera_id':d.get('camera_id',''),'d2_best_iou':float(d['best_predicted_component_iou']),'arcad_best_iou':float(a['best_predicted_component_iou']),'d2_coverage':float(d['gt_region_coverage']),'arcad_coverage':float(a['gt_region_coverage'])}
        for cut in ('0.1','0.3','0.5'):
            hitd=d[f'detected_iou_{cut}']=='True'; hita=a[f'detected_iou_{cut}']=='True'
            row.update({f'd2_detect_iou_{cut}':hitd,f'arcad_detect_iou_{cut}':hita,f'both_detect_iou_{cut}':hitd and hita,f'd2_only_detect_iou_{cut}':hitd and not hita,f'arcad_only_detect_iou_{cut}':hita and not hitd,f'neither_detect_iou_{cut}':not hitd and not hita})
        prows.append(row)
    write(OUT/'paired_region_overlap.csv',prows)
    cats=read(OUT/'category_metrics.csv'); joined=read(OUT/'joined_predictions.csv'); out=[]
    for level,cols in [('I',('I-AUROC','I-AP')),('O',('O-AUROC','O-AP'))]:
        for metric in cols:
            for model in ('d2','arcad'):
                vals=[float(r[f'{model}_{metric}']) for r in cats];out.append({'aggregation':'macro_category_mean','metric_level':level,'metric':metric,'model':model,'value':float(np.mean(vals)),'categories':len(vals),'interpretation':'equal category weights; five selected categories only'})
            d= [r for r in joined if r['model']=='d2'];a=[r for r in joined if r['model']=='arcad']
            if level=='I':
                yd=np.array([int(r['gt_view_label']) for r in d]);sd=np.array([float(r['score_view']) for r in d]);ya=np.array([int(r['gt_view_label']) for r in a]);sa=np.array([float(r['score_view']) for r in a])
            else:
                def obj(rows):
                    groups={}
                    for r in rows:groups.setdefault((r['category'],r['object_key']),[]).append(r)
                    yy=[];ss=[]
                    for v in groups.values():yy.append(max(int(x['gt_view_label']) for x in v));ss.append(float(v[0]['score_object']))
                    return np.asarray(yy),np.asarray(ss)
                yd,sd=obj(d);ya,sa=obj(a)
            fn=roc_auc_score if metric.endswith('AUROC') else average_precision_score
            out.extend([{'aggregation':'pooled_selected_categories','metric_level':level,'metric':metric,'model':'d2','value':float(fn(yd,sd)),'observations':len(yd),'interpretation':'pooled metric on five selected categories; raw score scales may differ by category'},
                        {'aggregation':'pooled_selected_categories','metric_level':level,'metric':metric,'model':'arcad','value':float(fn(ya,sa)),'observations':len(ya),'interpretation':'pooled metric on five selected categories; raw score scales may differ by category'}])
    write(OUT/'macro_micro_summary.csv',out)
    meta={'run_id':'20261031_gpu01','analysis_python':sys.version,'platform':platform.platform(),'cuda_device':'NVIDIA GeForce RTX 3090; GPU inference confirmed','ArcAD':json.load(open(OUT/'arcad_export_metadata.json')),'D2_additional_categories':json.load(open(OUT/'d2_export_metadata.json')),
          'D2_low_three_maps_source':'Task 01 shared re-evaluation maps, CPU inference; Task 01 view scores verified within 1e-5','bootstrap_iterations':2000,'bootstrap_seed':20261031,'paired_categories':[r['category'] for r in cats]}
    (OUT/'environment_provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
    report=(OUT/'analysis_report.md').read_text()
    if '## Aggregate summaries and exploratory miss associations' not in report:
        report += '''\n\n## Aggregate summaries and exploratory miss associations\n\n`macro_micro_summary.csv` reports category macro means (equal category weights) and pooled metrics on the five selected categories. The pooled result is descriptive because raw score scales may differ by category. It is not a 160-category benchmark estimate. `mechanism_associations.csv` fits separate category/camera-adjusted logistic models for a miss among positive views with nonempty masks, using log area, GT/background contrast proxy, region count, and border distance; the intervals resample category/object clusters. Coefficients are exploratory associations and are not causal. `paired_region_overlap.csv` matches each GT component across both models and records fixed pixel-threshold hits and coverage. Camera IDs are parsed from source filenames and retained in joined, view, region, and morphology tables. Confusion matrices include raw counts and row-normalized class rates.\n'''
        (OUT/'analysis_report.md').write_text(report)
    state=json.load(open(OUT/'analysis_state.json'));state['status']='complete';state['deliverables']=['README.md','analysis_report.md','joined_predictions.csv','view_diagnostics.csv','region_diagnostics.csv','object_diagnostics.csv','paired_region_overlap.csv','category_metrics.csv','delta_by_category.csv','macro_micro_summary.csv','bootstrap_ci.csv','failure_overlap.csv','confusion_matrices.csv','seen_unseen_summary.csv','error_taxonomy_counts.csv','mechanism_associations.csv','selected_examples.csv','environment_provenance.json'];(OUT/'analysis_state.json').write_text(json.dumps(state,indent=2)+'\n')
if __name__=='__main__':main()
