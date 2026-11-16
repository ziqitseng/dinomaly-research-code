import numpy as np,pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score,average_precision_score
p=Path(__file__).resolve().parents[1]
v=pd.read_csv(p/'per_view_scores.csv');r=pd.read_csv(p/'per_view_region_metrics.csv')
v['source_mask_area_fraction']=v.source_mask_pixels/1024**2
bins=[0,.001,.01,.05,1.000001];labels=['<0.1%','0.1-1%','1-5%','>=5%']
v['size_bin']=pd.cut(v.source_mask_area_fraction,bins=bins,labels=labels,right=False,include_lowest=True)
out=[]
for (c,b,field),rr in r.groupby(['category','branch','field']):
 vv=v[(v.category==c)&(v.ArcAD_original_score_top1.notna())]
 scorecol={'ArcAD_original':'ArcAD_original_score_top1','ArcAD_fullframe':'ArcAD_fullframe_score_top1','D2_original':'D2_original_score_top1'}[b]
 # Region table only has positive GT rows. Report fully central masks separately from border/partial.
 for st in ['inside_original_crop','partly_outside_original_crop','wholly_outside_original_crop']:
  for size in labels:
   pos=vv[(vv.view_label==1)&(vv.source_gt_stratum==st)&(vv.size_bin==size)]
   if pos.empty:continue
   neg=vv[vv.view_label==0]
   yy=np.r_[np.ones(len(pos),np.uint8),np.zeros(len(neg),np.uint8)]
   ss=np.r_[pos[scorecol].to_numpy(float),neg[scorecol].to_numpy(float)]
   q=rr[(rr.source_gt_stratum==st)&(rr.image_path.isin(pos.image_path))]
   row={'category':c,'branch':b,'field':field,'source_mask_position_stratum':st,'source_mask_area_bin':size,
    'bin_edges_source_area_fraction':[0,.001,.01,.05,1.000001][labels.index(size):labels.index(size)+2],
    'positive_views':len(pos),'normal_comparison_views':len(neg),'positive_objects':pos.object_key.nunique(),
    'I_AUROC_positive_size_bin_vs_all_normal':float(roc_auc_score(yy,ss)) if len(pos) and len(neg) else np.nan,
    'I_AP_positive_size_bin_vs_all_normal':float(average_precision_score(yy,ss)) if len(pos) and len(neg) else np.nan,
    'mean_GT_region_recall_IoU_0_1_at_top1pct':float(q.recall_iou_0_1.mean()) if len(q) else np.nan,
    'mean_GT_region_recall_IoU_0_3_at_top1pct':float(q.recall_iou_0_3.mean()) if len(q) else np.nan,
    'mean_GT_region_recall_IoU_0_5_at_top1pct':float(q.recall_iou_0_5.mean()) if len(q) else np.nan,
    'mean_GT_coverage_at_top1pct':float(q.mean_coverage.mean()) if len(q) else np.nan,
    'mean_topk_precision_0_1pct':float(q['top0.001_precision'].mean()) if len(q) else np.nan,
    'mean_topk_recall_0_1pct':float(q['top0.001_recall'].mean()) if len(q) else np.nan,
    'mean_topk_precision_1pct':float(q['top0.01_precision'].mean()) if len(q) else np.nan,
    'mean_topk_recall_1pct':float(q['top0.01_recall'].mean()) if len(q) else np.nan,
    'mean_topk_precision_5pct':float(q['top0.05_precision'].mean()) if len(q) else np.nan,
    'mean_topk_recall_5pct':float(q['top0.05_recall'].mean()) if len(q) else np.nan,
    'strata_definition':'source mask area bins fixed geometrically; no score-based boundary selection'}
   out.append(row)
pd.DataFrame(out).to_csv(p/'size_strata_metrics.csv',index=False)
print('rows',len(out))

