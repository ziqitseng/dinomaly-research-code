#!/usr/bin/env python3
"""Read-only manifest, leakage and provenance checks for Task 04A."""
import csv,hashlib,json,platform,subprocess,sys,re
from pathlib import Path
from collections import Counter,defaultdict
ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'analysis_task04A_crop_ablation_20261102_01'
DATA=Path('/media/avlab/7458a474-6ea2-4b90-b232-373c3df8d329/real_iad_variety_1024')
SHARED=ROOT/'analysis_shared_failure_20261031_gpu01/shared_manifest.json'
TASK2=ROOT/'analysis_shared_failure_20261031_gpu01'
ARCAD_SPLIT=DATA/'realiadvariety_jsons/sup';D2_SPLIT=DATA/'realiadvariety_jsons/Real-IAD_Variety_jsons'
ARCAD=ROOT/'ArcAD'
CKPT=ARCAD/'out/real_iad_variety out/arcad_realiad_variety/model.pth'
D2CKPT=ROOT/'Dinomaly2/out/real_iad_variety_1024 out/full run/variety_160cat/model.pth'
CATS=('laser_diode','lattice_block_plug','power_jack','LED_indicator','thyristor')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
 return h.hexdigest()
def main():
 m=json.loads(SHARED.read_text()); checks={}; issues=[]
 counts={}; sample_shapes=Counter(); missing=[]; bad_groups=[]; arcad_train_overlap={}; d2_map_validation={};split_consistency={}
 for cat in CATS:
  c=m['categories'][cat]; rows=c['shared_test_images']; objects=c['shared_test_objects']
  a=json.loads((ARCAD_SPLIT/(cat+'.json')).read_text());d=json.loads((D2_SPLIT/(cat+'.json')).read_text())
  at={r['image_path']:r for r in a['test']};dt={r['image_path']:r for r in d['test']};al={r['image_path'] for r in a['labeled']}
  dtrain_objects={'/'.join(r['image_path'].split('/')[:-1]) for r in d['train']}
  arcad_labeled_objects={'/'.join(p.split('/')[:-1]) for p in al}
  shared_objects={r['object_key'] for r in rows}
  d2_train_object_overlap=sorted(shared_objects & dtrain_objects)
  arcad_labeled_object_overlap=sorted(shared_objects & arcad_labeled_objects)
  if d2_train_object_overlap:issues.append(f'D2 train object leakage in shared evaluation for {cat}: {d2_train_object_overlap}')
  if arcad_labeled_object_overlap:issues.append(f'ArcAD labeled object leakage in shared evaluation for {cat}: {arcad_labeled_object_overlap}')
  mismatches=[]
  for r in rows:
   aa=at.get(r['image_path']);dd=dt.get(r['image_path'])
   if aa is None or dd is None: mismatches.append({'image_path':r['image_path'],'reason':'not in both native test splits'});continue
   dlabel=int(dd.get('anomaly_class')!='OK' and dd.get('mask_path') is not None)
   if int(aa['label'])!=int(r['label']) or dlabel!=int(r['label']) or aa.get('mask_path')!=r.get('mask_path') or dd.get('mask_path')!=r.get('mask_path') or r['image_path'] in al:
    mismatches.append({'image_path':r['image_path'],'arcad_label':aa.get('label'),'d2_label':dlabel,'manifest_label':r['label'],'arcad_mask':aa.get('mask_path'),'d2_mask':dd.get('mask_path'),'manifest_mask':r.get('mask_path'),'is_arcad_labeled':r['image_path'] in al})
  split_consistency[cat]={'shared_rows':len(rows),'arcad_native_test_paths':len(at),'d2_native_test_paths':len(dt),'arcad_labeled_paths':len(al),'arcad_split_sha256':sha(ARCAD_SPLIT/(cat+'.json')),'d2_split_sha256':sha(D2_SPLIT/(cat+'.json')),'arcad_labeled_object_overlap':arcad_labeled_object_overlap,'d2_train_object_overlap':d2_train_object_overlap,'label_mask_arcad_labeled_mismatches':mismatches,'all_shared_rows_match_both_splits':not mismatches and not d2_train_object_overlap and not arcad_labeled_object_overlap}
  if mismatches:issues.append(f'Shared split label/mask/train mismatch for {cat}: {len(mismatches)} rows')
  keys=set(r['object_key'] for r in rows)
  leak=keys & set(c['arcad_labeled_object_keys']); arcad_train_overlap[cat]=sorted(leak)
  groups=defaultdict(list); paths=[]; cams=defaultdict(set)
  for r in rows:
   groups[r['object_key']].append(r); paths.append(r['image_path'])
   match=re.search(r'(C[1-5])_',Path(r['image_path']).stem)
   cams[r['object_key']].add(match.group(1) if match else 'UNKNOWN')
   ip=DATA/cat/r['image_path']; mp=DATA/cat/r['mask_path'] if r.get('mask_path') else None
   if not ip.exists(): ip=DATA/cat/cat/r['image_path']
   if mp is not None and not mp.exists(): mp=DATA/cat/cat/r['mask_path']
   if not ip.is_file() or (r['label'] and (mp is None or not mp.is_file())): missing.append({'category':cat,'image_path':r['image_path'],'resolved_image':str(ip),'resolved_mask':str(mp) if mp else None})
   if ip.is_file():
    from PIL import Image
    with Image.open(ip) as im: sample_shapes[str(im.size)]+=1
  dup=len(paths)-len(set(paths))
  bad=[k for k,v in groups.items() if len(v)!=5 or len(cams[k])!=5 or cams[k]!={'C1','C2','C3','C4','C5'}]
  if bad: bad_groups.extend({'category':cat,'object_key':k} for k in bad)
  if leak: issues.append(f'ArcAD labeled object overlap in shared set: {cat}: {sorted(leak)}')
  if dup: issues.append(f'Duplicate image paths in {cat}: {dup}')
  counts[cat]={'views':len(rows),'objects':len(groups),'normal_objects':sum(not o['label'] for o in objects),'positive_objects':sum(bool(o['label']) for o in objects),'positive_views':sum(int(r['label']) for r in rows),'duplicate_paths':dup,'five_camera_groups_ok':not bad,'arcad_labeled_objects_in_shared_test':sorted(leak),'declared_shared_counts':c['counts']}
  f=(ROOT/'Dinomaly2/out/re_eval_shared_20261031_cpu01/maps'/f'{cat}.npz') if cat in ('laser_diode','lattice_block_plug','power_jack') else TASK2/'maps'/f'd2_{cat}.npz'
  if f.exists():
   import numpy as np
   z=np.load(f); p=z['image_paths'].astype(str).tolist(); mk='postprocessed_maps' if 'postprocessed_maps' in z.files else 'smoothed_maps'; d2_map_validation[cat]={'file':str(f.relative_to(ROOT)),'sha256':sha(f),'map_array_key':mk,'shape':list(z[mk].shape),'finite':bool(np.isfinite(z[mk]).all()),'paths_exactly_match_manifest':p==paths}
  else: d2_map_validation[cat]={'file':str(f),'available':False}
  if d2_map_validation[cat].get('paths_exactly_match_manifest') is False: issues.append(f'D2 cached map path order differs for {cat}')
 if missing: issues.append(f'{len(missing)} missing image/mask paths')
 if bad_groups: issues.append(f'{len(bad_groups)} object groups do not have exactly C1-C5')
 gpu={'nvidia_smi':'not checked in sandbox; elevated check succeeded RTX 3090','torch_cuda':'requires elevated execution; sandbox reports unavailable'}
 details={'task':'04A cropping inference ablation','dataset_root':str(DATA),'dataset_root_exists':DATA.is_dir(),'shared_manifest':str(SHARED),'shared_manifest_sha256':sha(SHARED),'task02_maps_dir':str(TASK2/'maps'),'categories':counts,'source_image_size_counts':dict(sample_shapes),'missing_paths':missing,'bad_object_groups':bad_groups,'split_consistency':split_consistency,'d2_cached_map_validation':d2_map_validation,'arcad_train_overlap':arcad_train_overlap,'manifest_checks_pass':not issues,'issues':issues}
 (OUT/'manifest_validation.json').write_text(json.dumps(details,indent=2)+'\n')
 ch=[]
 for name,p in [('ArcAD',CKPT),('DiNomaly2',D2CKPT)]:
  if not p.is_file(): raise FileNotFoundError(p)
  st=p.stat(); ch.append({'model':name,'path':str(p),'sha256_before':sha(p),'mtime_ns_before':st.st_mtime_ns,'bytes':st.st_size})
 (OUT/'checkpoint_integrity.json').write_text(json.dumps({'checkpoints':ch,'before_inference':True},indent=2)+'\n')
 provenance={'workspace_git_commit':None,'workspace_git_status':None,'python':sys.version,'platform':platform.platform(),'task02_environment_provenance':json.loads((TASK2/'environment_provenance.json').read_text()),'arcad_source_files':{str(p.relative_to(ROOT)):sha(p) for p in [ARCAD/'dataset.py',ARCAD/'models/uad.py',ARCAD/'out/real_iad_variety out/low3_shared_split_comparison_20261031/evaluate_arcad_shared.py']},'task04_script_sha256':sha(OUT/'work/run_arcad_branches.py')}
 try: provenance['workspace_git_commit']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.DEVNULL).strip()
 except Exception: provenance['workspace_git_commit']='unavailable (workspace has no resolvable HEAD)'
 (OUT/'protocol_audit.json').write_text(json.dumps(provenance,indent=2)+'\n')
 print(json.dumps({'counts':counts,'manifest_checks_pass':not issues,'issues':issues,'map_validation':d2_map_validation},indent=2))
 if issues: raise SystemExit(2)
if __name__=='__main__':main()
