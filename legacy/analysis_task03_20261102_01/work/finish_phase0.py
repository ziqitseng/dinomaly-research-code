#!/usr/bin/env python3
"""Finish split inventory and final provenance after Phase 0/1 source audit."""
import importlib.util,json,hashlib,csv,subprocess,platform,sys
from pathlib import Path
from collections import defaultdict
import numpy as np
ROOT=Path('/media/avlab/1CE6451EE644FA12/2026_10_22_dinomaly');OUT=ROOT/'analysis_task03_20261102_01';DATA=Path('/media/avlab/7458a474-6ea2-4b90-b232-373c3df8d329/real_iad_variety_1024')
D2=ROOT/'Dinomaly2';ARC=ROOT/'ArcAD'; DS_D2=DATA/'realiadvariety_jsons/Real-IAD_Variety_jsons';DS_ARC=DATA/'realiadvariety_jsons/sup'
def obj(p):return '/'.join(p.split('/')[:-1])
def dlabel(r):return int(r.get('anomaly_class')!='OK' and r.get('mask_path') is not None)
def savecsv(p,rows):
 with open(p,'w',newline='',encoding='utf8') as f:
  w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def main():
 spec=importlib.util.spec_from_file_location('phase',OUT/'work/phase0_1_audit.py');mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
 # Rerun exact object-leakage checks and full positive laser mask accounting; contact sheets already exist.
 split_audit,empty,manifest=mod.manifest_audit()
 # Full native-split counts for every source category.
 rows=[]
 for fp in sorted(DS_D2.glob('*.json')):
  cat=fp.stem; ap=DS_ARC/f'{cat}.json'
  if not ap.exists():continue
  d=json.load(open(fp));a=json.load(open(ap));
  def stats(rs,label_fn):
   groups=defaultdict(list)
   for r in rs:groups[obj(r['image_path'])].append(label_fn(r))
   return {'views':len(rs),'objects':len(groups),'positive_views':sum(label_fn(r) for r in rs),'positive_objects':sum(max(x) for x in groups.values()),'normal_objects':sum(not max(x) for x in groups.values())}
  ds=stats(d['test'],dlabel);ats=stats(a['test'],lambda r:int(r['label']));tr=stats(d['train'],dlabel)
  labeled_objects={obj(r['image_path']) for r in a['labeled']}; arc_test_objects={obj(r['image_path']) for r in a['test']}; d2_test_objects={obj(r['image_path']) for r in d['test']};d2_train_objects={obj(r['image_path']) for r in d['train']}
  rows.append({'category':cat,'d2_test_views':ds['views'],'d2_test_objects':ds['objects'],'d2_test_positive_views':ds['positive_views'],'d2_test_positive_objects':ds['positive_objects'],'d2_train_views':tr['views'],'d2_train_objects':tr['objects'],'arcad_test_views':ats['views'],'arcad_test_objects':ats['objects'],'arcad_test_positive_views':ats['positive_views'],'arcad_test_positive_objects':ats['positive_objects'],'arcad_labeled_views':len(a['labeled']),'arcad_labeled_objects':len(labeled_objects),'arcad_labeled_object_overlap_arcad_test':len(labeled_objects&arc_test_objects),'arcad_labeled_object_overlap_d2_test':len(labeled_objects&d2_test_objects),'d2_train_test_object_overlap':len(d2_train_objects&d2_test_objects),'native_test_view_count_delta_d2_minus_arcad':ds['views']-ats['views']})
 if len(rows)!=160:raise RuntimeError(f'Expected 160 native split rows, got {len(rows)}')
 savecsv(OUT/'native_split_counts_160.csv',rows)
 p=OUT/'protocol_audit.json';audit=json.load(open(p));audit['source_native_splits']={'source_count_csv':str(OUT/'native_split_counts_160.csv'),'n_categories':160,'summary':{'D2_native_test_views':sum(int(r['d2_test_views']) for r in rows),'D2_native_test_objects':sum(int(r['d2_test_objects']) for r in rows),'ArcAD_native_test_views':sum(int(r['arcad_test_views']) for r in rows),'ArcAD_native_test_objects':sum(int(r['arcad_test_objects']) for r in rows),'categories_with_different_native_test_view_counts':sum(r['d2_test_views']!=r['arcad_test_views'] for r in rows)},'five_category_shared_manifest_audit':split_audit,'interpretation':'The original category scores use their respective native test manifests; do not treat the 160-category threshold-count overlap as paired inference.'}
 # Save stable source hashes, including the exact prior artifacts read.
 sources=[D2/'out/re_eval_shared_20261031_cpu01/per_view_scores.csv',D2/'out/re_eval_shared_20261031_cpu01/metrics.csv',ROOT/'analysis_shared_failure_20261031_gpu01/shared_manifest.json',ROOT/'analysis_shared_failure_20261031_gpu01/category_metrics.csv',ROOT/'analysis_shared_failure_20261031_gpu01/alignment_validation.json']
 hashes=[]
 for q in sources:
  if q.is_file():
   h=hashlib.sha256()
   with open(q,'rb') as f:
    for b in iter(lambda:f.read(1<<20),b''):h.update(b)
   hashes.append({'path':str(q),'sha256':h.hexdigest(),'size_bytes':q.stat().st_size})
 audit['prior_analysis_inputs_hashes']=hashes;audit['runtime']={'python':sys.version,'platform':platform.platform(),'torch_cuda':audit.get('device',{}),'new_inference':'none; no GPU inference required for saved-map phases'}
 p.write_text(json.dumps(audit,indent=2)+'\n')
 weak=json.load(open(OUT/'empty_mask_summary.json'));weak['task01_reconciliation']={'Task01_empty_views':2,'both_within_Task03_41':True,'identities':['NG/OilStain/S0005 C1','NG/OilStain/S0005 C4'],'additional_cases':'39 positive views have nonzero source masks entirely outside the joint [64:960] field.'};(OUT/'empty_mask_summary.json').write_text(json.dumps(weak,indent=2)+'\n')
 print('native split rows',len(rows),'ArcAD labeled overlap with D2 test in diagnostic cats',{c:v['arcad_labeled_object_overlap_with_d2_native_test'] for c,v in split_audit.items()})
if __name__=='__main__':main()
