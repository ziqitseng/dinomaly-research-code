#!/usr/bin/env python3
"""Summarize Task 03 outputs and audit completeness without changing source runs."""
import csv,json,platform,sys,hashlib
from pathlib import Path
import numpy as np
ROOT=Path('/media/avlab/1CE6451EE644FA12/2026_10_22_dinomaly');OUT=ROOT/'analysis_task03_20261102_01'
def rd(p):
 with open(p,encoding='utf8') as f:return list(csv.DictReader(f))
def write(p,rows):
 with open(p,'w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),extrasaction='ignore');w.writeheader();w.writerows(rows)
def main():
 # Normalize camera IDs to the source convention C1..C5.
 p=OUT/'object_camera_consistency.csv';rows=rd(p)
 for r in rows:
  if r.get('camera_id','').isdigit():r['camera_id']='C'+r['camera_id']
 write(p,rows)
 # Label sensitivity source mask accounting and exact key IDs.
 sm=json.load(open(OUT/'empty_mask_summary.json')); sens=rd(OUT/'label_sensitivity.csv')
 if sm['task02_empty_common_mask_count']!=41 or sm['status_counts_all_positive_views'].get('nonzero_source_mask_outside_common_crop',0)!=39 or sm['status_counts_all_positive_views'].get('genuinely_all_zero_source_mask',0)!=2:raise RuntimeError('empty-mask reconciliation failed')
 # Strict full precision 160 category screen.
 screen=rd(OUT/'category_160_audit.csv'); overlap=rd(OUT/'weak_category_overlap.csv')
 d2=next(r for r in overlap if r['D2_status']=='D2 weak' and r['ArcAD_status']=='ArcAD weak');donly=next(r for r in overlap if r['D2_status']=='D2 weak' and r['ArcAD_status']=='ArcAD not weak');aonly=next(r for r in overlap if r['D2_status']=='D2 not weak' and r['ArcAD_status']=='ArcAD weak');neither=next(r for r in overlap if r['D2_status']=='D2 not weak' and r['ArcAD_status']=='ArcAD not weak')
 # Fusion mean ranking summary on matched heldout rows.
 comp=rd(OUT/'fusion_comparison.csv');macro={(r['level'],r['system']):r for r in comp if r['category']=='MACRO_FIVE'};pool={(r['level'],r['system']):r for r in comp if r['category']=='ALL_FIVE'}
 op=rd(OUT/'operating_point_metrics.csv')
 operating={(r['category'],r['system'],r['level'],float(r['normal_object_fpr_target'])):r for r in op}
 summary={'screen_160':{'n_categories':160,'D2_strict_lt_0.90':sum(float(r['d2_I-AUROC'])<.9 for r in screen),'ArcAD_strict_lt_0.90':sum(float(r['arcad_I-AUROC'])<.9 for r in screen),'D2_macro':float(np.mean([float(r['d2_I-AUROC']) for r in screen])),'ArcAD_macro':float(np.mean([float(r['arcad_I-AUROC']) for r in screen])),'overlap':{'both':int(d2['category_count']),'D2_only':int(donly['category_count']),'ArcAD_only':int(aonly['category_count']),'neither':int(neither['category_count'])}},'mask_audit':sm,'task02_empty_mask_label_sensitivity':sens,'five_category_fusion_diagnostic':{'macro':{f'{level}_{system}':{'AUROC':float(r['AUROC']),'AP':float(r['AP'])} for (level,system),r in macro.items()},'pooled':{f'{level}_{system}':{'AUROC':float(r['AUROC']),'AP':float(r['AP'])} for (level,system),r in pool.items()},'operating_points':{f'{system}_{level}_{alpha:g}':{'recall':float(row['positive_recall']),'heldout_FP_normal_objects':int(row['heldout_normal_object_fp_count']),'heldout_normal_objects':int(row['heldout_normal_objects']),'heldout_FPR':int(row['heldout_normal_object_fp_count'])/max(1,int(row['heldout_normal_objects'])),'confusion':{k:int(row[f'confusion_{k}']) for k in ('TN','FP','FN','TP')}} for (cat,system,level,alpha),row in operating.items() if cat=='ALL_FIVE' and alpha in (.1,.2) and level=='object'},'threshold_free_eval_n_objects':1060,'threshold_free_eval_n_views':5300,'positive_prevalence_view':4825/5300,'positive_prevalence_object':985/1060,'fusion_decision':'fixed 50/50 normal-rank mean; no anomaly-based tuning; no deployability claim'}}
 (OUT/'task03_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
 # Update calibration note with exact finite-sample behavior.
 rel=rd(OUT/'calibration_reliability_counts.csv')
 note=(OUT/'calibration_reliability.md').read_text()
 if 'Actual held-out results' not in note:
  note+='\n## Actual held-out results\n\nSee `calibration_reliability_counts.csv` for each system, level and nominal 10%/20% point: exact false-positive object counts, denominators and Clopper-Pearson 95% intervals on both calibration and held-out normal objects. The pooled calibration has 50 normal objects; thresholds are based on higher empirical quantiles, so ties can make the realized calibration rate differ from the nominal target. The held-out pool has 75 normal objects across the five selected categories, giving wide binomial intervals.\n'
 (OUT/'calibration_reliability.md').write_text(note)
 # Required report sections with concise numeric evidence.
 m=summary['five_category_fusion_diagnostic']['macro'];p=summary['five_category_fusion_diagnostic']['pooled'];op10=summary['five_category_fusion_diagnostic']['operating_points']
 macro_i={sys:m[f'view_{sys}']['AUROC'] for sys in ('d2','arcad','fixed_50_50_mean')};macro_o={sys:m[f'object_{sys}']['AUROC'] for sys in ('d2','arcad','fixed_50_50_mean')}
 text=f'''# Task 03 — Verification, calibration and fusion follow-up

## Verified observations

The original native-split screen has 160 categories. From full-precision source files, D2 has **83/160** categories with I-AUROC `<0.90` (`<=0.90`: 83; none exactly 0.90), and ArcAD has **100/160** (`<=0.90`: 100; none exactly 0.90). The 2×2 screen is 78 weak on both, 5 D2-only, 22 ArcAD-only, and 55 neither. Macro means are D2 {summary['screen_160']['D2_macro']:.6f} and ArcAD {summary['screen_160']['ArcAD_macro']:.6f}; these match the reported 0.8859/0.8599. Native test lists and supervision differ, so this overlap is descriptive screening, not paired inference. See `category_160_audit.csv`, `weak_category_overlap.csv`, and `native_split_counts_160.csv`.

On the 41 laser_diode views Task 02 flagged as empty on the common field, source-mask recomputation found **39 nonzero masks entirely outside the shared crop** and **2 all-zero source masks** (OilStain/S0005 C1/C4). No mask vanished only at 224×224. All 41 sheets show the image and five sibling views with source/crop masks. These are unresolved visibility/annotation cases requiring manual review; no labels or masks were changed. `label_sensitivity.csv` reports common-grid pixel metrics before/after excluding the 41 ambiguous views from mask-dependent calculations, plus an explicitly exploratory view-label sensitivity. Object labels remain unchanged.

The coordinate audit independently matched source GT masks to both models' common-grid GT in 15 seeded sample views, and confirms 224×224 maps, finite nonconstant scores, and unique shared paths. The five-camera groups and D2/ArcAD view labels/mask paths match. One ArcAD-labeled object/category appears in each diagnostic category's D2 native test manifest; all those objects are excluded from the shared paired set. There is no ArcAD-labeled or D2-training object in the paired shared evaluation partition.

## Evidence for/against mechanisms

Paired per-view and per-object scores, raw normal/defect score distributions, predeclared top-score-percentage sensitivity, per-camera outcomes, GT area/component/width, RGB local luminance contrast, model-derived anomaly-score contrast, GT/background quantiles, Q, top-k precision/recall, and seen/unseen type counts are saved in their corresponding CSVs. Camera and defect-type tables retain physical object IDs. A separate exploratory category/camera-adjusted miss model is not fitted here; this follow-up uses stratified summaries to avoid over-interpreting sparse category/type effects.

Paired figures show the same source view, common-crop GT, both maps and actual top-1% map pixels. Raw top-1% image scores, map ranges, normal-CDF ranks, thresholds, Q and top-k measurements are printed separately from display-normalized heatmaps. `manual_review_sheet.md` separates visible observations from plausible but unverified explanations. No causal interpretation is assigned to contrast, view, or ArcAD seen/unseen association.

## Unverified hypotheses

The 39 crop-external masks could reflect legitimate defect location outside the common field, crop mismatch, or annotation placement; the 2 all-zero masks could be weak/missing annotation or image-label disagreement. Contact sheets alone do not resolve which explanation is correct. Lighting, texture, specularity, occlusion and supervision effects remain hypotheses requiring expert review or controlled data.

## Practical ensemble result

No independent anomaly-validation set was available, so no weighted fusion was tuned. The only ensemble baseline is a fixed 50/50 mean of model-wise global empirical normal-CDF ranks, with a calibrated OR rule also reported. This uses 50 normal calibration objects (10/category) and excludes them from held-out evaluation; calibration objects are from the same native test manifests, so this is a **diagnostic object-disjoint transfer experiment**, not deployment validation.

On held-out objects, category-macro AUROC was D2 {macro_o['d2']:.4f}, ArcAD {macro_o['arcad']:.4f}, fixed mean {macro_o['fixed_50_50_mean']:.4f}; pooled AUROC was D2 {p['object_d2']['AUROC']:.4f}, ArcAD {p['object_arcad']['AUROC']:.4f}, fixed mean {p['object_fixed_50_50_mean']['AUROC']:.4f}. The view-level macro AUROCs were D2 {macro_i['d2']:.4f}, ArcAD {macro_i['arcad']:.4f}, fixed mean {macro_i['fixed_50_50_mean']:.4f}. See `fusion_comparison.csv` for AP, per-category results and the corresponding prevalence baselines; the positive prevalence is high ({4825/5300:.1%} of views, {985/1060:.1%} of objects), so AP is correspondingly high.

At the pooled 10% operating point, held-out object results are in `operating_point_metrics.csv`; calibration/evaluation FP counts and exact binomial intervals are in `calibration_reliability_counts.csv`. The OR detector's false-positive union cost is reported in both operating-point and failure-overlap tables. Oracle union coverage is separately labeled and is not an implementable ensemble result. Pixel-level fusion was skipped; existing aligned per-model region diagnostics remain available, with their 10-object calibration caveat.

## Limitations

The 160-category results use different native test/supervision regimes. Five paired categories are selected diagnostics, not a representative benchmark sample; no new category inference was launched. Nominal 1/5/10% thresholds from Task 02's ten normals all collapse to one max-score threshold and were rejected as distinct operating points. The pooled 50-normal calibration improves resolution but is still native-test-derived and category-mixed. Fixed fusion gains may reflect this score-CDF transfer and the selected categories; no 160-category ensemble claim follows.

Bootstrap intervals resample physical objects and retain all five views (2,000 draws; see `bootstrap_intervals.csv` for valid and excluded single-class draws). Pixel/region bootstrap intervals and ROC/PR bootstrap bands were skipped. Task 02 numeric maps were reused; no new model inference or checkpoint changes were made. Task 01 low-three D2 maps were originally CPU-generated; existing ArcAD and added D2 maps were GPU-generated as recorded in provenance.

## Recommended next experiment

Have an annotator review the 41 laser_diode contact sheets against the original 1024 images and all five cameras, without editing the source labels in place. For a deployable fusion test, reserve a substantially larger, disjoint normal calibration set and an independent anomaly-validation set with object-level separation; freeze the CDF transform and any weight there, then evaluate on untouched objects. Before broadening to all 160 categories, evaluate the four frozen extension categories in `category_selection_plan.md` using matched objects, identical scoring definitions and a prospective holdout.
'''
 (OUT/'analysis_report.md').write_text(text)
 # Update summary README high-level findings and output inventory.
 readme=(OUT/'README.md').read_text();readme+='\n\n## Verified high-level findings\n\nThe original 160-category threshold counts reproduce: 83 weak D2 and 100 weak ArcAD, with overlap 78 both / 5 D2-only / 22 ArcAD-only / 55 neither. Forty-one laser_diode common-crop empty masks split into 39 source masks outside the crop and two source-all-zero cases. The fixed 50/50 normal-rank fusion is diagnostic only; see `analysis_report.md` for held-out metrics and calibration limits.\n'
 (OUT/'README.md').write_text(readme)
 # State and deliverable inventory; skipped items are explicit in issues.md.
 state={'status':'complete','run_id':'analysis_task03_20261102_01','no_training':True,'original_checkpoints_and_Task01_Task02_outputs_unchanged':True,'new_model_inference':'none; all analyses reuse cached numeric maps','deliverables':sorted(p.name for p in OUT.iterdir()),'screen_summary':summary['screen_160'],'calibration_summary':json.load(open(OUT/'calibration_split_manifest.json')),'known_skips':['tuned weighted fusion (no independent validation anomalies)','pixel fusion (optional; no independent score calibration)','additional category inference (not necessary for this diagnostic follow-up)','pixel/region bootstrap intervals and ROC/PR bootstrap bands']}
 (OUT/'analysis_state.json').write_text(json.dumps(state,indent=2)+'\n')
 print('Task03 finalization complete',summary['screen_160'])
if __name__=='__main__':main()
