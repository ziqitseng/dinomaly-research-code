#!/usr/bin/env python3
"""Build the Task 04A path-verified evaluation manifest and preflight audits."""
import csv, hashlib, json, re, subprocess
from pathlib import Path
from collections import defaultdict
from PIL import Image

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parents[1]
DATA=Path('/media/avlab/7458a474-6ea2-4b90-b232-373c3df8d329/real_iad_variety_1024')
T2=ROOT/'analysis_shared_failure_20261103'
ARCAD=ROOT/'ArcAD'
D2=ROOT/'Dinomaly2'
ARCAD_CKPT=ARCAD/'out/real_iad_variety out/arcad_realiad_variety_protoinit/model.pth'
D2_CKPT=D2/'out/real_iad_variety_1024 out/full run/variety_160cat/model.pth'
CATS=['lattice_block_plug','laser_diode','power_jack','LED_indicator','thyristor']
TASK2_CATS=set(json.loads((T2/'manifest_validation.json').read_text())['categories'])

def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def rel(r):return r['image_path'].replace('\\','/')
def obj(path):
    p=path.split('/')
    return '/'.join(p[:2] if p[0]=='OK' else p[:3])
def resolve(cat,path):
    if not path:return ''
    p=DATA/cat/path
    if not p.is_file():p=DATA/cat/cat/path
    return str(p)
def load_native(cat):
    ap=DATA/'realiadvariety_jsons/sup'/f'{cat}.json'
    dp=DATA/'realiadvariety_jsons/Real-IAD_Variety_jsons'/f'{cat}.json'
    a=json.loads(ap.read_text());d=json.loads(dp.read_text())
    at={rel(x):x for x in a['test']};dt={rel(x):x for x in d['test']}
    labeled={obj(rel(x)) for x in a['labeled']}
    rows=[];excluded=[];errors=[]
    for p in sorted(set(at)&set(dt)):
        key=obj(p)
        if key in labeled:excluded.append(p);continue
        ar,dr=at[p],dt[p]
        if int(ar['label'])!=int(bool(dr.get('mask_path'))):errors.append(f'{cat}: label mismatch {p}')
        if ar.get('mask_path')!=dr.get('mask_path'):errors.append(f'{cat}: mask path mismatch {p}')
        m=re.search(r'_C([1-5])_',Path(p).name)
        if not m:errors.append(f'{cat}: camera parse failed {p}')
        cp=p.split('/')
        row={'category':cat,'object_key':key,'image_path':p,'mask_path':ar.get('mask_path') or '',
             'label':int(ar['label']),'defect_type':cp[1] if cp[0]=='NG' else 'OK',
             'anomaly_class':ar.get('source_anomaly_class',ar.get('anomaly_class','')),
             'camera_view_id':int(m.group(1)) if m else -1,
             'image_abs':resolve(cat,p),'mask_abs':resolve(cat,ar.get('mask_path')),
             'split_id':'task04a_native_test_intersection_excluding_arcad_labeled_objects'}
        if not Path(row['image_abs']).is_file():errors.append(f'{cat}: missing image {p}')
        if row['label'] and not Path(row['mask_abs']).is_file():errors.append(f'{cat}: missing mask {p}')
        rows.append(row)
    return rows,{'arcad_manifest':str(ap),'d2_manifest':str(dp),'arcad_labeled_objects':len(labeled),
        'arcad_labeled_object_keys':sorted(labeled),'native_test_intersection':len(set(at)&set(dt)),
        'excluded_arcad_labeled_test_paths':len(excluded),'excluded_arcad_labeled_paths':excluded,
        'errors':errors}

OUT.mkdir(parents=True,exist_ok=True)
(OUT/'maps').mkdir(exist_ok=True)
task2=json.loads((T2/'shared_test_manifest.json').read_text())
task2_validation=json.loads((T2/'manifest_validation.json').read_text())
if task2_validation.get('status')!='PASS':raise RuntimeError('Task 02 manifest validation is not PASS')
rows=[];cats_report={};errors=[]
for cat in CATS:
    native,native_meta=load_native(cat)
    if cat in TASK2_CATS:
        chosen=[r for r in task2 if r['category']==cat]
        a={r['image_path'] for r in chosen};b={r['image_path'] for r in native}
        if a!=b:errors.append(f'{cat}: Task 02 manifest differs from recomputed ArcAD/D2 intersection')
        source='exact Task 02 shared manifest; independently recomputed against both native test lists and ArcAD labeled objects'
    else:
        chosen=native;source='recomputed native ArcAD/D2 test intersection with all ArcAD labeled objects excluded'
    chosen=sorted(chosen,key=lambda r:r['image_path'])
    groups=defaultdict(list)
    for r in chosen:
        groups[r['object_key']].append(r)
        if r['category']!=cat:errors.append(f'{cat}: category field mismatch {r["image_path"]}')
        if r['label'] not in (0,1):errors.append(f'{cat}: invalid label {r["image_path"]}')
    bad={k:sorted(x['camera_view_id'] for x in v) for k,v in groups.items()
         if len(v)!=5 or {x['camera_view_id'] for x in v}!={1,2,3,4,5}}
    if bad:errors.append(f'{cat}: invalid five-camera groups {list(bad.items())[:5]}')
    leaked=set(groups)&set(native_meta['arcad_labeled_object_keys'])
    if leaked:errors.append(f'{cat}: ArcAD labeled-object leakage {sorted(leaked)}')
    label_conflicts=[]
    for r in chosen:
        im=Image.open(r['image_abs'])
        if im.size!=(1024,1024):errors.append(f'{cat}: image shape {im.size}, path {r["image_path"]}')
        if r['label'] and not r['mask_abs']:label_conflicts.append(r['image_path'])
    if label_conflicts:errors.extend([f'{cat}: positive view missing mask {p}' for p in label_conflicts])
    rows.extend(chosen)
    cats_report[cat]={'membership_source':source,'views':len(chosen),'objects':len(groups),
       'positive_views':sum(r['label'] for r in chosen),'positive_objects':sum(any(x['label'] for x in v) for v in groups.values()),
       'normal_objects':sum(not any(x['label'] for x in v) for v in groups.values()),
       'arcad_labeled_object_count':native_meta['arcad_labeled_objects'],
       'excluded_arcad_labeled_test_paths':native_meta['excluded_arcad_labeled_test_paths'],
       'checks':{'unique_image_paths':len({r['image_path'] for r in chosen})==len(chosen),
          'five_views_per_object':not bad,'no_arcad_labeled_object_overlap':not leaked,
          'source_image_shapes_1024':True,'image_label_mask_presence_consistent':not label_conflicts},
       'native_split_details':native_meta}
manifest_path=OUT/'shared_test_manifest.json'
rows.sort(key=lambda r:(CATS.index(r['category']),r['image_path']))
manifest_path.write_text(json.dumps(rows,indent=2)+'\n')
sha_manifest=sha(manifest_path)
validation={'status':'FAIL' if errors else 'PASS','manifest_sha256':sha_manifest,'dataset_root':str(DATA),
  'source_task02_manifest_sha256':sha(T2/'shared_test_manifest.json'),
  'task02_manifest_validation_status':task2_validation.get('status'),
  'categories':cats_report,'total_views':len(rows),'total_objects':sum(x['objects'] for x in cats_report.values()),'errors':errors,
  'arcad_legacy_used':False}
(OUT/'manifest_validation.json').write_text(json.dumps(validation,indent=2)+'\n')
if errors:raise RuntimeError('Manifest validation failed: '+'; '.join(errors[:10]))

ckpts={}
for model,path in [('ArcAD_protoinit',ARCAD_CKPT),('D2',D2_CKPT)]:
    ckpts[model]={'path':str(path),'exists':path.is_file(),'size_bytes':path.stat().st_size if path.is_file() else None,
       'sha256_before':sha(path),'mtime_ns_before':path.stat().st_mtime_ns}
(OUT/'checkpoint_integrity.json').write_text(json.dumps({'status':'PRE_INFERENCE','checkpoints':ckpts},indent=2)+'\n')

def source_coord(i,resize,crop):return ((i+crop)+.5)*(1024/resize)-.5
geom={'status':'PASS_STATIC_GEOMETRY','source_shape':[1024,1024],
 'branches':{
  'ArcAD_original':{'resize':[448,448],'center_crop':[392,392],'input_tensor':[392,392],
     'source_field_approx':[64,960],'input_center_pixel_source_xy':[source_coord(195.5,448,28),source_coord(195.5,448,28)],
     'map_shape':[256,256],'native_map_observation':'central field only'},
  'ArcAD_full_frame':{'resize':[392,392],'crop':None,'input_tensor':[392,392],
     'source_field_approx':[0,1024],'input_center_pixel_source_xy':[((195.5+.5)*(1024/392)-.5)]*2,
     'map_shape':[256,256],'native_map_observation':'full source field'},
  'D2_original':{'resize':[280,280],'crop':[280,280],'input_tensor':[280,280],
     'source_field_approx':[0,1024],'input_center_pixel_source_xy':[((139.5+.5)*(1024/280)-.5)]*2,
     'map_shape':[256,256],'native_map_observation':'full source field'}},
 'common_central_grid':{'shape':[224,224],'source_crop_xyxy':[64,64,960,960],
   'ArcAD_original_map':'bilinear resize native 256x256 crop map to 224x224',
   'ArcAD_full_frame_map':'crop native rows/cols [16:240]',
   'D2_original_map':'crop native rows/cols [16:240]',
   'GT':'raw source [64:960] crop then nearest resize to 224'},
 'full_source_grid':{'shape':[256,256],'models':['ArcAD_full_frame','D2_original'],
   'GT':'raw source mask nearest resize to 256','ArcAD_original':'outside original crop is unobserved, NaN/masked'},
 'patch_grid_check':{'ArcAD_input_392_divisible_by_14':392%14==0,'ArcAD_patch_grid':[392//14,392//14],
   'D2_input_280_divisible_by_14':280%14==0,'D2_patch_grid':[280//14,280//14],
   'full_frame_ArcAD_input_shape_matches_original':True,
   'runtime_synthetic_landmark_check':'pending GPU smoke pass before full inference'},
 'geometry_note':'Source-to-input values use pixel-center coordinates and square 1024x1024 source images. Runtime landmark/shape checks remain a mandatory pre-inference gate.'}
(OUT/'geometry_validation.json').write_text(json.dumps(geom,indent=2)+'\n')

protocol={'task':'04A controlled crop/FOV inference ablation','created':'2026-11-03',
 'selected_categories':CATS,'primary_categories':['lattice_block_plug','laser_diode','power_jack'],
 'optional_categories_included':['LED_indicator','thyristor'],
 'Task02_category_mismatch':'Task 02 has aircraft_model_head instead of thyristor; Task 04A optional thyristor is constructed from exact native split intersection. No substitution.',
 'dataset_root':str(DATA),'Task02_manifest_path':str(T2/'shared_test_manifest.json'),'Task02_manifest_sha256':validation['source_task02_manifest_sha256'],
 'Task04A_manifest_sha256':sha_manifest,'categories':cats_report,'checkpoints':ckpts,
 'input_transforms':{'ArcAD_original':'Resize 448x448; CenterCrop 392x392; RGB ToTensor; ImageNet mean/std normalization',
   'ArcAD_full_frame':'Resize 392x392; no crop; same RGB normalization; OOD inference relative to training',
   'D2_original':'Resize 280x280; CenterCrop 280x280; RGB ToTensor; ImageNet mean/std normalization'},
 'map_processing':'256x256 bilinear map resize; Gaussian smoothing kernel 5x5 sigma 4; per-view score mean of top 1% map pixels',
 'fusion_or_threshold_tuning':'none; threshold-free primary metrics only; any diagnostic thresholds will be explicitly test-derived',
 'arcad_legacy_used':False,'new_output_directory':str(OUT),
 'cuda_preflight':'GPU visible only when using elevated execution; nvidia-smi reported RTX 3090, 24576 MiB VRAM; PyTorch 2.7.1+cu118 reports CUDA available outside sandbox',
 'inference_status':'not_started; execute after all prepared-manifest and geometry checks pass'}
(OUT/'protocol_audit.json').write_text(json.dumps(protocol,indent=2)+'\n')
print(json.dumps({'manifest_status':validation['status'],'views':len(rows),'categories':{k:v['views'] for k,v in cats_report.items()},
 'arcad_legacy_used':False,'arcad_sha256':ckpts['ArcAD_protoinit']['sha256_before'],'d2_sha256':ckpts['D2']['sha256_before']},indent=2))
