#!/usr/bin/env python3
"""Regenerate leakage-free paired sample manifests for Task 02 categories."""
import csv, hashlib, json, re
from pathlib import Path

OUT=Path(__file__).resolve().parents[1]
DATA=Path('/media/avlab/7458a474-6ea2-4b90-b232-373c3df8d329/real_iad_variety_1024')
ARCAD=Path('/media/avlab/1CE6451EE644FA12/2026_10_22_dinomaly/ArcAD')
D2=Path('/media/avlab/1CE6451EE644FA12/2026_10_22_dinomaly/Dinomaly2')
TASK1= D2/'out/re_eval_shared_20261031_cpu01'
LOW3=('lattice_block_plug','laser_diode','power_jack')
ADDITIONAL=('LED_indicator','thyristor')
CATEGORIES=LOW3+ADDITIONAL

def okey(row): return '/'.join(row['image_path'].split('/')[:-1])
def label(row): return int(row.get('anomaly_class')!='OK' and row.get('mask_path') is not None)

def main():
    arcad_scores={r['Category']:float(r['I-AUROC']) for r in csv.DictReader(open(ARCAD/'out/real_iad_variety out/arcad_realiad_variety/results.csv')) if r['Category']!='Mean'}
    d2_scores={}
    for line in (D2/'out/real_iad_variety_1024 out/full run/variety_160cat/log.txt').read_text(errors='replace').splitlines():
        m=re.match(r'([^:]+): I-Auroc:([0-9.]+)',line)
        if m and m.group(1)!='Mean': d2_scores[m.group(1)]=float(m.group(2))
    candidates=sorted((min(arcad_scores[c],d2_scores[c]),(arcad_scores[c]+d2_scores[c])/2,c,arcad_scores[c],d2_scores[c]) for c in set(arcad_scores)&set(d2_scores))
    comparison=next((x for x in reversed(candidates) if x[2] not in LOW3 and x[2]!='LED_indicator'),None)
    if comparison is None: raise RuntimeError('No common preselection category available')
    if comparison[2] != 'thyristor': raise RuntimeError(f'Unexpected preselection winner: {comparison}')
    result={'dataset_root':str(DATA),'selection_rule':'For the favorable/control category, before Task 02 paired predictions, maximize the minimum of original native-test I-AUROC across ArcAD and D2 among common categories outside the low-three and LED_indicator.','preselected_favorable_category':{'category':comparison[2],'arcad_original_native_I-AUROC':comparison[3],'d2_original_native_I-AUROC':comparison[4]},'categories':{},'supervision':{},'calibration_plan':{'rule':'Within each category, deterministically select 10 normal shared-test objects by SHA-256(object_key) order as calibration-only objects; exclude those objects from all thresholded test metrics. Threshold-free baseline metrics continue to use the complete paired shared manifest.','note':'Calibration scores only are used for thresholds; no threshold is tuned to evaluation labels.'}}
    for cat in CATEGORIES:
        a=json.loads((DATA/'realiadvariety_jsons/sup'/f'{cat}.json').read_text())
        d=json.loads((DATA/'realiadvariety_jsons/Real-IAD_Variety_jsons'/f'{cat}.json').read_text())
        at={r['image_path']:r for r in a['test']}; al={r['image_path']:r for r in a['labeled']}; dt={r['image_path']:r for r in d['test']}
        train={r['image_path'] for r in d['train']}
        train_objects={okey(r) for r in d['train']}
        excluded_paths=sorted(set(at)&set(dt)&set(al))
        common=sorted((set(at)&set(dt))-set(al))
        records=[]
        for p in common:
            ar,dr=at[p],dt[p]
            if int(ar['label'])!=label(dr) or ar.get('mask_path')!=dr.get('mask_path'):
                raise ValueError(f'{cat}: source label/mask mismatch: {p}')
            if p in train: raise ValueError(f'{cat}: D2 train image in native test: {p}')
            rec={'category':cat,'image_path':p,'mask_path':ar.get('mask_path'),'label':int(ar['label']),'anomaly_class':ar.get('anomaly_class',''),'object_key':okey(ar)}
            vm=re.search(r'_C([1-5])_',Path(p).name)
            if not vm: raise ValueError('camera ID parse failed: '+p)
            rec['view_id']=int(vm.group(1)); records.append(rec)
        groups={}
        for r in records: groups.setdefault(r['object_key'],[]).append(r)
        objects=[]
        for key,rs in sorted(groups.items()):
            rs.sort(key=lambda x:x['view_id'])
            if len(rs)!=5 or len({r['view_id'] for r in rs})!=5: raise ValueError(f'{cat}: malformed five-view object {key}')
            obj_label=max(r['label'] for r in rs)
            objects.append({'category':cat,'object_key':key,'label':obj_label,'views':rs})
        labeled_objects={okey(r) for r in al.values()}
        train_obj_overlap=sorted(labeled_objects&train_objects)
        labeled_test_obj_overlap=sorted(labeled_objects&{okey(r) for r in d['test']})
        shared_obj_overlap=sorted({o['object_key'] for o in objects}&labeled_objects)
        if shared_obj_overlap: raise ValueError(f'{cat}: ArcAD labeled object leakage {shared_obj_overlap}')
        for r in records:
            if not (DATA/cat/r['image_path']).is_file(): raise FileNotFoundError(DATA/cat/r['image_path'])
            if r['label'] and (not r['mask_path'] or not (DATA/cat/r['mask_path']).is_file()): raise FileNotFoundError(f'{cat}: missing GT {r}')
            if not r['label'] and r['mask_path'] is not None: raise ValueError(f'{cat}: normal with mask path {r}')
        normals=[o for o in objects if o['label']==0]
        cal_keys=sorted((o['object_key'] for o in normals),key=lambda s:hashlib.sha256(s.encode()).hexdigest())[:10]
        if len(cal_keys)<5: raise ValueError(f'{cat}: too few normal objects for calibration')
        for o in objects: o['role']='calibration_normal' if o['object_key'] in cal_keys else 'evaluation'
        original_d2_objects=len({okey(r) for r in d['test']})
        original_arcad_objects=len({okey(r) for r in a['test']})
        result['categories'][cat]={'counts':{'arcad_native_test_objects':original_arcad_objects,'d2_native_test_objects':original_d2_objects,'shared_views':len(records),'shared_objects':len(objects),'positive_views':sum(r['label'] for r in records),'positive_objects':sum(o['label'] for o in objects),'normal_objects':sum(1-o['label'] for o in objects),'calibration_normal_objects':len(cal_keys),'thresholded_evaluation_objects':len(objects)-len(cal_keys)},'excluded_arcad_labeled_test_paths':excluded_paths,'arcad_labeled_object_keys':sorted(labeled_objects),'arcad_labeled_object_overlap_with_d2_train':train_obj_overlap,'arcad_labeled_object_overlap_with_d2_native_test':labeled_test_obj_overlap,'calibration_object_keys':cal_keys,'shared_test_images':records,'shared_test_objects':objects}
        by_type={}
        for r in a['labeled']:
            t=r.get('anomaly_class','')
            row=by_type.setdefault(t,{'distinct_objects':set(),'views':0})
            row['distinct_objects'].add(okey(r)); row['views']+=1
        result['supervision'][cat]={t:{'distinct_objects':len(v['distinct_objects']),'views':v['views']} for t,v in sorted(by_type.items())}
    (OUT/'shared_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print('Favorable category preselected:',comparison)
    for c,v in result['categories'].items(): print(c,v['counts'],'calibration object keys:',len(v['calibration_object_keys']))

if __name__=='__main__':main()
