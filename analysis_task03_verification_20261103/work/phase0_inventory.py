#!/usr/bin/env python3
import csv,json,re,hashlib,subprocess
from collections import defaultdict
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parents[1]
DATA=Path('/media/avlab/7458a474-6ea2-4b90-b232-373c3df8d329/real_iad_variety_1024')
ARCAD_CKPT=ROOT/'ArcAD/out/real_iad_variety out/arcad_realiad_variety_protoinit/model.pth'
D2_CKPT=ROOT/'Dinomaly2/out/real_iad_variety_1024 out/full run/variety_160cat/model.pth'
ARCAD_RESULTS=ROOT/'ArcAD/out/real_iad_variety out/arcad_realiad_variety_protoinit/results.csv'
D2_METRICS=ROOT/'Dinomaly2/out/real_iad_variety_1024 out/full run/variety_160cat/category_heatmap/category_heatmap_metrics.json'
TASK2=ROOT/'analysis_shared_failure_20261103';TASK1=ROOT/'analysis_task01_protoinit_shared_20261103'
CURRENT5=['lattice_block_plug','laser_diode','power_jack','LED_indicator','aircraft_model_head']
TASKFILE_EXPECTED5=['lattice_block_plug','laser_diode','power_jack','LED_indicator','thyristor']

def sha(path):
 h=hashlib.sha256()
 with open(path,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def object_key(p):
 z=p.replace('\\','/').split('/');return '/'.join(z[:2] if z[0]=='OK' else z[:3])
def camera(p):
 m=re.search(r'_C([1-5])_',Path(p).name);return int(m.group(1)) if m else None
def source_json(root,kind,cat):
 base=root/'realiadvariety_jsons'/('sup' if kind=='ArcAD' else 'Real-IAD_Variety_jsons')/f'{cat}.json'
 return json.loads(base.read_text())
def count_split(records):
 objs={};views=set();bad=[];view_labels={}
 for r in records:
  p=r.get('image_path',r.get('image','')).replace('\\','/');k=object_key(p);c=camera(p)
  views.add(p);objs.setdefault(k,[]).append(c);view_labels[p]=int(r.get('label',bool(r.get('mask_path'))))
 for k,cs in objs.items():
  if len(cs)!=5 or set(cs)!={1,2,3,4,5}:bad.append(k)
 positive_views=sum(view_labels.values())
 # Rebuild object labels from the same paths used above.
 obj_labels={k:[] for k in objs}
 for p in views:obj_labels[object_key(p)].append(view_labels[p])
 positive_objects=sum(any(q) for q in obj_labels.values())
 return {'views':len(views),'objects':len(objs),'positive_views':positive_views,'normal_views':len(views)-positive_views,'positive_objects':positive_objects,
  'normal_objects':len(objs)-positive_objects,'invalid_five_view_objects':bad}

# Screen original full-run summaries: D2 official full-split JSON and the current,
# prototype-initialized ArcAD run. The obsolete pre-initialization ArcAD run is excluded.
a=pd.read_csv(ARCAD_RESULTS);a=a[a['Category']!='Mean'].copy()
d=json.loads(D2_METRICS.read_text());dr=d['categories_sorted_by_ascending_I-AUROC']
arc={r['Category']:float(r['I-AUROC']) for _,r in a.iterrows()};d2={r['category']:float(r['I-AUROC']) for r in dr}
cats=sorted(set(arc)&set(d2));mismatch={'arcad_only':sorted(set(arc)-set(d2)),'d2_only':sorted(set(d2)-set(arc))}
if len(arc)!=160 or len(d2)!=160 or set(arc)!=set(d2):raise RuntimeError(f'Expected matching 160-category summaries, got {len(arc)}, {len(d2)}, mismatch={mismatch}')
rows=[]
for cat in cats:
 av=arc[cat];dv=d2[cat];aw=av<.9;dw=dv<.9
 rows.append({'category':cat,'ArcAD_I_AUROC_protoinit':av,'D2_I_AUROC':dv,'delta_D2_minus_ArcAD':dv-av,'weak_both':int(aw and dw),'weak_D2_only':int(dw and not aw),'weak_ArcAD_only':int(aw and not dw),'weak_neither':int(not aw and not dw),'ArcAD_lt_0_90':int(av<.9),'ArcAD_le_0_90':int(av<=.9),'D2_lt_0_90':int(dv<.9),'D2_le_0_90':int(dv<=.9)})
rows.sort(key=lambda r:r['D2_I_AUROC'])
for rank,r in enumerate(rows,1):r['D2_rank_ascending']=rank
with open(OUT/'category_160_audit.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
with open(OUT/'all160_category_screening.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
overlap=[{'screening_group':'both_weak','count':sum(r['weak_both'] for r in rows)},
 {'screening_group':'D2_only_weak','count':sum(r['weak_D2_only'] for r in rows)},
 {'screening_group':'ArcAD_only_weak','count':sum(r['weak_ArcAD_only'] for r in rows)},
 {'screening_group':'neither_weak','count':sum(r['weak_neither'] for r in rows)}]
with open(OUT/'weak_category_overlap.csv','w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(overlap[0]));w.writeheader();w.writerows(overlap)

# Check current Task 02 manifest, expected-category discrepancy, label/mask parity,
# object-level grouping, and ArcAD labeled-object exclusion.
shared=json.loads((TASK2/'shared_test_manifest.json').read_text());arcad_excluded={}
shared_checks={}
for cat in CURRENT5:
 rr=[r for r in shared if r['category']==cat];objs={r['object_key'] for r in rr}
 sup=source_json(DATA,'ArcAD',cat); lab={object_key(r['image_path']) for r in sup['labeled']}
 bad=[r['image_path'] for r in rr if int(r['label']) != int(bool(r.get('mask_path')))]
 shared_checks[cat]={'views':len(rr),'objects':len(objs),'positive_views':sum(int(r['label']) for r in rr),'normal_views':sum(not int(r['label']) for r in rr),
   'five_view_object_count':sum(sum(x['object_key']==o for x in rr)==5 for o in objs),'ArcAD_labeled_train_object_overlap':len(objs&lab),'label_mask_path_disagreement':bad}
 arcad_excluded[cat]=sorted(lab)

# Native split counts and five-view/object-disjoint checks over all 160 categories.
native={};global_bad=[]
for cat in cats:
 ap=source_json(DATA,'ArcAD',cat);dp=source_json(DATA,'D2',cat)
 at=count_split(ap.get('test',[]));dt=count_split(dp.get('test',[]))
 lobj={object_key(x['image_path']) for x in ap.get('labeled',[])};atest={object_key(x['image_path']) for x in ap.get('test',[])}
 dtrain={object_key(x['image_path']) for x in dp.get('train',[])};dtest={object_key(x['image_path']) for x in dp.get('test',[])}
 rec={'category':cat,'ArcAD_test_views':at['views'],'ArcAD_test_objects':at['objects'],'ArcAD_invalid_test_objects':len(at['invalid_five_view_objects']),
   'D2_test_views':dt['views'],'D2_test_objects':dt['objects'],'D2_invalid_test_objects':len(dt['invalid_five_view_objects']),
   'ArcAD_test_normal_views':at['normal_views'],'ArcAD_test_anomalous_views':at['positive_views'],'ArcAD_test_normal_objects':at['normal_objects'],'ArcAD_test_anomalous_objects':at['positive_objects'],
   'D2_test_normal_views':dt['normal_views'],'D2_test_anomalous_views':dt['positive_views'],'D2_test_normal_objects':dt['normal_objects'],'D2_test_anomalous_objects':dt['positive_objects'],
   'ArcAD_invalid_test_object_keys':at['invalid_five_view_objects'],'D2_invalid_test_object_keys':dt['invalid_five_view_objects'],
   'ArcAD_labeled_train_test_object_overlap':len(lobj&atest),'D2_train_test_object_overlap':len(dtrain&dtest),
   'ArcAD_labeled_train_test_object_overlap_keys':sorted(lobj&atest),'D2_train_test_object_overlap_keys':sorted(dtrain&dtest)}
 native[cat]=rec
 if any(rec[k] for k in ['ArcAD_invalid_test_objects','D2_invalid_test_objects','ArcAD_labeled_train_test_object_overlap','D2_train_test_object_overlap']):global_bad.append(cat)

# Precommitted diagnostic extensions, chosen only from full-run category rankings.
existing=set(CURRENT5)
eligible=[r for r in rows if r['category'] not in existing]
both=[r for r in eligible if r['weak_both']];donly=[r for r in eligible if r['weak_D2_only']];aonly=[r for r in eligible if r['weak_ArcAD_only']]
strong=[r for r in eligible if r['ArcAD_I_AUROC_protoinit']>=.95 and r['D2_I_AUROC']>=.95]
plan=[]
if both:plan.append({'role':'shared_weakness','category':min(both,key=lambda r:min(r['ArcAD_I_AUROC_protoinit'],r['D2_I_AUROC']))['category'],'selection_rule':'lowest minimum of both model I-AUROCs among categories where both are <0.90'})
if donly:plan.append({'role':'D2_only_weakness','category':min(donly,key=lambda r:r['D2_I_AUROC'])['category'],'selection_rule':'lowest D2 I-AUROC where D2<0.90 and ArcAD>=0.90'})
if aonly:plan.append({'role':'ArcAD_only_weakness','category':min(aonly,key=lambda r:r['ArcAD_I_AUROC_protoinit'])['category'],'selection_rule':'lowest ArcAD protoinit I-AUROC where ArcAD<0.90 and D2>=0.90'})
if strong:plan.append({'role':'strong_performance','category':max(strong,key=lambda r:min(r['ArcAD_I_AUROC_protoinit'],r['D2_I_AUROC']))['category'],'selection_rule':'highest minimum I-AUROC among categories where both are >=0.95'})
md=['# Precommitted Task 03 extension selection','',
 'The existing diagnostic set is the actual current Task 02 set: '+', '.join(CURRENT5)+'. The Task 03 brief’s prior-run list names `thyristor` instead of the current `aircraft_model_head`; this mismatch is recorded, and the current saved maps are the basis for this follow-up.',
 '', 'Before any new maps or per-object outcomes are examined, extensions are selected only from the 160-category native summary scores. Selection rules are fixed at I-AUROC <0.90 for weakness and >=0.95 for the strong comparison. Current Task 02 categories are excluded from extension candidates.', '']
for r in plan:md.append(f"- **{r['role']} — `{r['category']}`:** {r['selection_rule']}.")
md+=['','This is a diagnostic, score-ranked selection plan, not a representative sample. No extension inference is launched in Task 03 unless later evidence shows saved numeric maps are insufficient and a separate compute plan is approved.']
(OUT/'category_selection_plan.md').write_text('\n'.join(md)+'\n')

arc_task=json.loads((TASK2/'arcad_extra_integrity.json').read_text());d2_task=json.loads((TASK2/'d2_extra_integrity.json').read_text())
arc_ck={'path':str(ARCAD_CKPT),'sha256':sha(ARCAD_CKPT),'size_bytes':ARCAD_CKPT.stat().st_size,'inference_sha256_before_after':arc_task.get('sha256_before')==arc_task.get('sha256_after'),'device':arc_task.get('device'),'strict_load':arc_task.get('strict_load')}
d2_ck={'path':str(D2_CKPT),'sha256':sha(D2_CKPT),'size_bytes':D2_CKPT.stat().st_size,'inference_sha256_before_after':d2_task.get('sha256_before')==d2_task.get('sha256_after'),'device':d2_task.get('device'),'strict_load':d2_task.get('strict_load')}
try:
 import torch
 gpu={'cuda_available':torch.cuda.is_available(),'cuda_device_count':torch.cuda.device_count(),'torch_version':torch.__version__,'cuda_version':torch.version.cuda}
except Exception as e:gpu={'inspection_error':str(e)}
def commit(repo):
 try:return subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'],stderr=subprocess.DEVNULL,text=True).strip()
 except Exception:return None

protocol={'task':'03 verification, calibration, and fusion follow-up','created_utc':'2026-11-03',
 'data_root':str(DATA),'arcad_checkpoint':arc_ck,'d2_checkpoint':d2_ck,'source_commits':{'ArcAD':commit(ROOT/'ArcAD'),'DiNomaly2':commit(ROOT/'Dinomaly2')},
 'native_results':{'ArcAD_protoinit_results':str(ARCAD_RESULTS),'D2_full_run_metrics':str(D2_METRICS),'ArcAD_preinitialization_run':'excluded by standing user instruction; not read'},
 '160_category_summary':{'category_count':160,'matched_category_sets':not bool(mismatch),'mismatch':mismatch,
  'ArcAD_protoinit_macro_I_AUROC':float(a['I-AUROC'].mean()),'ArcAD_lt_0_90':int((a['I-AUROC']<.9).sum()),'ArcAD_le_0_90':int((a['I-AUROC']<=.9).sum()),
  'D2_macro_I_AUROC':float(sum(d2.values())/len(d2)),'D2_lt_0_90':int(sum(v<.9 for v in d2.values())),'D2_le_0_90':int(sum(v<=.9 for v in d2.values())),
  'strict_threshold_full_precision':True,'counts_are_based_on_current_ArcAD_protoinit_and_current_D2_full_run':True,
  'task_brief_expected_old_ArcAD_macro_0.8599_100_weak':'not audited because the pre-initialization ArcAD output is excluded; current protoinit summary is used'},
 'full_native_splits':{'ArcAD_test_views':sum(x['ArcAD_test_views'] for x in native.values()),'D2_test_views':sum(x['D2_test_views'] for x in native.values()),
  'categories_with_native_split_integrity_issues':global_bad,'category_counts':native},
 'current_task02':{'manifest':str(TASK2/'shared_test_manifest.json'),'validation':json.loads((TASK2/'manifest_validation.json').read_text()),
  'actual_categories':CURRENT5,'task03_brief_categories':TASKFILE_EXPECTED5,'category_list_mismatch':{'only_current':sorted(set(CURRENT5)-set(TASKFILE_EXPECTED5)),'only_task_brief':sorted(set(TASKFILE_EXPECTED5)-set(CURRENT5))},
  'category_checks':shared_checks},
 'preprocessing':{'D2':'Resize/crop 280x280, 256 map, Gaussian 5x5 sigma 4, top-1% mean; full run','ArcAD':'current protoinit run, Resize 448, CenterCrop 392, 256 map, Gaussian 5x5 sigma 4, top-1% mean',
  'common_physical_fov':'source [64:960) x [64:960), 224x224 grid independently generated in Phase 1/2'},
 'device':gpu,'fusion_precommit':'only fixed global 50/50 score-average using in-sample normal-reference z-scores; no anomaly-supervised weight selection; no deployable claim',
 'legacy_exclusion':'No ArcAD input or result under any legacy directory was read or used.'}
(OUT/'protocol_audit.json').write_text(json.dumps(protocol,indent=2)+'\n')
shared_object_labels=defaultdict(list)
for r in shared:shared_object_labels[(r['category'],object_key(r['image_path']))].append(int(r['label']))
heldout_normal_objects={c:sorted(k for (category,k),labels in shared_object_labels.items() if category==c and not any(labels)) for c in CURRENT5}
cal_manifest={'status':'INDEPENDENT_VALIDATION_NOT_AVAILABLE','current_task02_test_objects_per_category':{c:shared_checks[c]['objects'] for c in CURRENT5},
 'actual_normal_test_objects_per_category':{c:len(heldout_normal_objects[c]) for c in CURRENT5},
 'heldout_test_normal_object_keys':heldout_normal_objects,
 'ArcAD_labeled_supervision_objects_excluded_from_test':{c:len(arcad_excluded[c]) for c in CURRENT5},
 'normal_reference_sources':{'ArcAD':'7-9 native training-normal objects/category; seen during model training, in-sample','D2':'25 native training-normal objects/category; seen during model training, in-sample'},
 'independent_anomaly_validation_objects':0,'threshold_policy':'No calibrated 1/5/10% operating-point claims. Test-derived ROC/PR and measured-FPR sweeps are descriptive only.',
 'heldout_test_normal_objects_are_too_few_for_reliable_1_5_10_percent_calibration':True}
(OUT/'calibration_split_manifest.json').write_text(json.dumps(cal_manifest,indent=2)+'\n')
print(json.dumps({'ArcAD_protoinit_macro':protocol['160_category_summary']['ArcAD_protoinit_macro_I_AUROC'],'ArcAD_lt_le': [protocol['160_category_summary']['ArcAD_lt_0_90'],protocol['160_category_summary']['ArcAD_le_0_90']],
 'D2_macro':protocol['160_category_summary']['D2_macro_I_AUROC'],'D2_lt_le':[protocol['160_category_summary']['D2_lt_0_90'],protocol['160_category_summary']['D2_le_0_90']],
 'overlap':overlap,'source_split_integrity_issues':global_bad,'extensions':plan},indent=2))
