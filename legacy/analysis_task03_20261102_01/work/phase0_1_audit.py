#!/usr/bin/env python3
"""Task 03 Phases 0/1: original 160-category audit, mask audit, alignment checks."""
import csv, json, re, hashlib, subprocess, platform, sys, math
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score

ROOT=Path('/media/avlab/1CE6451EE644FA12/2026_10_22_dinomaly')
OUT=ROOT/'analysis_task03_20261102_01'
DATA=Path('/media/avlab/7458a474-6ea2-4b90-b232-373c3df8d329/real_iad_variety_1024')
D2=ROOT/'Dinomaly2'; ARC=ROOT/'ArcAD'; T2=ROOT/'analysis_shared_failure_20261031_gpu01'
D2RUN=D2/'out/real_iad_variety_1024 out/full run/variety_160cat'
ARCRUN=ARC/'out/real_iad_variety out/arcad_realiad_variety'
DS_D2=DATA/'realiadvariety_jsons/Real-IAD_Variety_jsons'; DS_ARC=DATA/'realiadvariety_jsons/sup'
COMMON=(64,64,960,960); GRID=224

def pro_auc(masks,maps,cap=.3):
    masks=np.asarray(masks,bool);maps=np.asarray(maps,np.float32);regions=[];negative=[]
    for mask,amap in zip(masks,maps):
        n,l,stats,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
        for j in range(1,n):regions.append(np.sort(amap[l==j]))
        negative.append(amap[~mask].ravel())
    if not regions:return float('nan')
    neg=np.sort(np.concatenate(negative));ts=np.linspace(float(maps.min()),float(maps.max()),201)[:-1];pro=[];fpr=[]
    for t in ts:
        pro.append(float(np.mean([(len(x)-np.searchsorted(x,t,side='right'))/len(x) for x in regions])))
        fpr.append(float((len(neg)-np.searchsorted(neg,t,side='right'))/len(neg)))
    order=np.argsort(fpr);x=np.asarray(fpr)[order];y=np.asarray(pro)[order];ux,idx=np.unique(x,return_index=True);uy=y[idx]
    if ux[0]>0:ux=np.r_[0.,ux];uy=np.r_[uy[0],uy]
    xx=np.linspace(0,cap,301);return float(np.trapz(np.interp(xx,ux,uy),xx)/cap)

def readcsv(p):
    with open(p,encoding='utf-8') as f:return list(csv.DictReader(f))
def writecsv(p,rows):
    if not rows:return
    with open(p,'w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)),extrasaction='ignore');w.writeheader();w.writerows(rows)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def git_info(p):
    def g(args):
        try:return subprocess.check_output(['git','-C',str(p),*args],stderr=subprocess.DEVNULL,text=True).strip()
        except Exception:return 'unavailable'
    return {'root':g(['rev-parse','--show-toplevel']),'commit':g(['rev-parse','HEAD']),'status_porcelain':g(['status','--porcelain'])}
def objkey(p):return '/'.join(p.split('/')[:-1])
def label_d2(r):return int(r.get('anomaly_class')!='OK' and r.get('mask_path') is not None)
def categories_data():
    d=json.load(open(D2RUN/'category_heatmap/category_heatmap_metrics.json'))
    d2={x['category']:float(x['I-AUROC']) for x in d['categories_sorted_by_ascending_I-AUROC']}
    a={r['Category']:float(r['I-AUROC']) for r in readcsv(ARCRUN/'results.csv') if r['Category']!='Mean'}
    if len(d2)!=160 or len(a)!=160 or set(d2)!=set(a):raise RuntimeError(f'160 category mismatch: D2={len(d2)} ArcAD={len(a)} common={len(set(d2)&set(a))}')
    names=sorted(d2); rows=[]
    for cat in names:
        x,y=d2[cat],a[cat]
        weakd=x<.9; weaka=y<.9
        rows.append({'category':cat,'d2_I-AUROC':x,'arcad_I-AUROC':y,'delta_D2_minus_ArcAD':x-y,'weak_D2_strict_lt_0.90':weakd,'weak_ArcAD_strict_lt_0.90':weaka,'weak_both':weakd and weaka,'weak_D2_only':weakd and not weaka,'weak_ArcAD_only':weaka and not weakd,'neither_weak':not weakd and not weaka,'d2_rank_best1':1+sum(v>x for v in d2.values()),'arcad_rank_best1':1+sum(v>y for v in a.values()),'near_0.90_d2_rounded_log':False})
    # D2 source log is printed to 4 decimal places. The companion JSON is full precision;
    # use it for exact threshold counts, and report how the printed values classify.
    d2log={}
    pat=re.compile(r'^([^:]+): I-Auroc:([0-9.]+)')
    for ln in (D2RUN/'log.txt').read_text(errors='replace').splitlines():
        m=pat.match(ln)
        if m and m.group(1)!='Mean':d2log[m.group(1)]=float(m.group(2))
    if len(d2log)!=160:raise RuntimeError(f'D2 log has {len(d2log)} category I-AUROC rows')
    for r in rows:r['d2_displayed_4dp']=d2log[r['category']];r['near_0.90_d2_rounded_log']=abs(d2log[r['category']]-.9)<=.00005
    writecsv(OUT/'category_160_audit.csv',rows)
    bins=Counter()
    for r in rows:
        key=('D2 weak' if r['weak_D2_strict_lt_0.90'] else 'D2 not weak', 'ArcAD weak' if r['weak_ArcAD_strict_lt_0.90'] else 'ArcAD not weak');bins[key]+=1
    overlap=[{'D2_status':a0,'ArcAD_status':a1,'category_count':n,'categories':'|'.join(r['category'] for r in rows if ('D2 weak' if r['weak_D2_strict_lt_0.90'] else 'D2 not weak')==a0 and ('ArcAD weak' if r['weak_ArcAD_strict_lt_0.90'] else 'ArcAD not weak')==a1)} for (a0,a1),n in sorted(bins.items())]
    writecsv(OUT/'weak_category_overlap.csv',overlap)
    def choose(rows0,key,sortkey,reverse=False):return sorted([r for r in rows0 if key(r)],key=sortkey,reverse=reverse)[0]
    already={'lattice_block_plug','laser_diode','power_jack','LED_indicator','thyristor'}
    candidates=[r for r in rows if r['category'] not in already]
    selected={
      'shared_weakness':choose(candidates,lambda r:r['weak_both'],lambda r:(r['d2_I-AUROC']+r['arcad_I-AUROC'])/2),
      'D2_only_weakness':choose(candidates,lambda r:r['weak_D2_only'],lambda r:r['delta_D2_minus_ArcAD']),
      'ArcAD_only_weakness':choose(candidates,lambda r:r['weak_ArcAD_only'],lambda r:-r['delta_D2_minus_ArcAD']),
      'strong_both':choose(candidates,lambda r:not r['weak_D2_strict_lt_0.90'] and not r['weak_ArcAD_strict_lt_0.90'],lambda r:min(r['d2_I-AUROC'],r['arcad_I-AUROC']),reverse=True)}
    plan={'rule_frozen_before_any_new_per_object_outcome_or_inference':True,'basis':'Original native-split category I-AUROC only; weak means strict <0.90; category list is diagnostic, not a paired comparison because model native test lists differ.','selection_rules':{'shared_weakness':'Both below 0.90; select the lowest mean of the two I-AUROCs.','D2_only_weakness':'D2 below 0.90 and ArcAD at least 0.90; select the most negative D2-ArcAD delta.','ArcAD_only_weakness':'ArcAD below 0.90 and D2 at least 0.90; select the most positive D2-ArcAD delta.','strong_both':'Both at least 0.90; maximize the weaker model I-AUROC.'},'already_analyzed':['lattice_block_plug','laser_diode','power_jack','LED_indicator','thyristor'],'prioritized_additional_categories':{k:v['category'] for k,v in selected.items()},'inference_decision':'No new categories inferred in this run. Reuse Task 02 numeric maps first; any expansion requires a separate compute/storage review and prespecified held-out analysis.'}
    (OUT/'category_selection_plan.md').write_text('# Frozen extension category plan\n\n'+json.dumps(plan,indent=2)+'\n')
    # Exact-source counts and rounded-log sensitivity.
    counts={}
    for model,scores in [('D2',d2),('ArcAD',a)]:
        vals=list(scores.values());counts[model]={'n_categories':len(vals),'strict_lt_0.90':sum(v<.9 for v in vals),'less_or_equal_0.90':sum(v<=.9 for v in vals),'exactly_0.90_source_precision':sum(v==.9 for v in vals),'macro_mean':float(np.mean(vals)),'source_rounding':'D2 full precision category_heatmap_metrics.json; ArcAD full precision results.csv'}
    counts['D2_displayed_4dp']={'strict_lt_0.90':sum(v<.9 for v in d2log.values()),'less_or_equal_0.90':sum(v<=.9 for v in d2log.values()),'displayed_exact_0.9000':sum(v==.9 for v in d2log.values()),'rounding_ambiguous_categories':[k for k,v in d2log.items() if v==.9]}
    return d2,a,d,counts

def manifest_audit():
    m=json.load(open(T2/'shared_manifest.json')); report={}; empty=[]
    for cat,v in m['categories'].items():
        ds=json.load(open(DS_D2/f'{cat}.json')); arc=json.load(open(DS_ARC/f'{cat}.json'))
        dtest={r['image_path']:r for r in ds['test']}; atest={r['image_path']:r for r in arc['test']}; labeled={r['image_path']:r for r in arc['labeled']}
        labeled_objects={objkey(r['image_path']) for r in labeled.values()}
        shared=v['shared_test_images']; shared_objects={r['object_key'] for r in shared}
        d2_train_objects={objkey(r['image_path']) for r in ds['train']}
        d2_test_objects={objkey(r['image_path']) for r in ds['test']}
        arcad_test_objects={objkey(r['image_path']) for r in arc['test']}
        view_mismatch=[]; mask_mismatch=[]; duplicate=[p for p,c in Counter(r['image_path'] for r in shared).items() if c>1]
        byobj=defaultdict(list)
        for r in shared:
            byobj[r['object_key']].append(r)
            if r['image_path'] not in dtest or r['image_path'] not in atest:raise RuntimeError(f'{cat} shared path outside native test')
            if r['image_path'] in labeled:raise RuntimeError(f'ArcAD labeled path leak: {cat} {r["image_path"]}')
            if objkey(r['image_path']) in labeled_objects:raise RuntimeError(f'ArcAD labeled object leak: {cat} {r["object_key"]}')
            if r['label']!=int(atest[r['image_path']]['label']):view_mismatch.append(r['image_path'])
            if r['label']!=label_d2(dtest[r['image_path']]):view_mismatch.append(r['image_path'])
            if r['mask_path']!=dtest[r['image_path']].get('mask_path'):mask_mismatch.append(r['image_path'])
        for k,rs in byobj.items():
            cams=[int(r['view_id']) for r in rs]
            if len(rs)!=5 or sorted(cams)!=[1,2,3,4,5]:raise RuntimeError(f'{cat}: malformed five-camera group {k} {cams}')
        train_shared=sorted(shared_objects&d2_train_objects)
        if train_shared:raise RuntimeError(f'{cat}: D2 training object leaks into shared test: {train_shared[:5]}')
        if duplicate or view_mismatch or mask_mismatch:raise RuntimeError(f'{cat}: identity audit fail dup={duplicate} labels={view_mismatch} masks={mask_mismatch}')
        report[cat]={'arcad_test_views':len(atest),'d2_test_views':len(dtest),'shared_views':len(shared),'shared_objects':len(byobj),'views_per_object':'all exactly five cameras C1-C5','arcad_labeled_test_path_overlap':0,'arcad_labeled_test_object_overlap':0,'d2_train_shared_test_object_overlap':len(train_shared),'d2_native_train_test_object_overlap':len(d2_train_objects&d2_test_objects),'arcad_labeled_object_overlap_with_d2_native_test':len(labeled_objects&d2_test_objects),'arcad_labeled_object_overlap_with_arcad_native_test':len(labeled_objects&arcad_test_objects),'d2_arcad_view_label_match':True,'mask_path_match':True,'duplicate_shared_paths':0}
    # Recompute all positive laser_diode mask statuses from original source PNGs.
    cat='laser_diode'; recs=m['categories'][cat]['shared_test_images']; grouped=defaultdict(list)
    d2_test_lookup={x['image_path']:x for x in json.load(open(DS_D2/f'{cat}.json'))['test']}
    arc_test_lookup={x['image_path']:x for x in json.load(open(DS_ARC/f'{cat}.json'))['test']}
    for r in recs:grouped[r['object_key']].append(r)
    for r in recs:
        if not r['label']:continue
        p=DATA/cat/r['mask_path'] if r.get('mask_path') else None
        try:
            if not p or not p.is_file():raise FileNotFoundError(str(p))
            with Image.open(p) as im:
                arr=np.asarray(im.convert('L')); shape=list(arr.shape); src=int(np.count_nonzero(arr)); crop=arr[64:960,64:960]; cropn=int(np.count_nonzero(crop)); down=np.asarray(Image.fromarray(crop).resize((GRID,GRID),Image.Resampling.NEAREST)); downn=int(np.count_nonzero(down))
            if src==0:status='genuinely_all_zero_source_mask'
            elif cropn==0:status='nonzero_source_mask_outside_common_crop'
            elif downn==0:status='nonzero_crop_mask_vanishes_at_224'
            else:status='nonempty_common_crop_mask'
            error=''
        except Exception as e:shape=[];src=cropn=downn=None;status='missing_or_corrupt_mask';error=str(e)
        cams=re.search(r'_C([1-5])_',Path(r['image_path']).name); siblings=sorted(grouped[r['object_key']],key=lambda x:x['view_id'])
        object_label=max(x['label'] for x in siblings)
        arcrow=arc_test_lookup[r['image_path']];d2row=d2_test_lookup[r['image_path']]
        row={'category':cat,'object_key':r['object_key'],'image_path':r['image_path'],'camera_id':f'C{cams.group(1)}' if cams else '', 'view_id':r['view_id'],'defect_type':r['anomaly_class'],'image_label':r['label'],'arcad_view_label':int(arcrow['label']),'d2_view_label':label_d2(d2row),'view_label_match':int(arcrow['label'])==label_d2(d2row),'object_label':object_label,'object_label_matches_max_view_label':object_label==max(x['label'] for x in siblings),'source_mask_path':r.get('mask_path') or '', 'source_mask_shape':json.dumps(shape),'source_mask_pixels':src,'common_crop_mask_pixels':cropn,'resized_224_mask_pixels':downn,'status':status,'requires_manual_review':status in ('genuinely_all_zero_source_mask','nonzero_source_mask_outside_common_crop','nonzero_crop_mask_vanishes_at_224','missing_or_corrupt_mask'),'error':error,'sibling_cameras':'|'.join(f'C{x["view_id"]}' for x in siblings),'task02_previously_flagged_empty_common_mask':any(x['image_path']==r['image_path'] for x in json.load(open(T2/'alignment_validation.json'))['known_source_mask_exceptions'])}
        empty.append(row)
    writecsv(OUT/'empty_mask_audit.csv',empty)
    (OUT/'empty_mask_summary.json').write_text(json.dumps({'positive_laser_diode_views_audited':sum(r['label'] for r in recs),'audit_rows':len(empty),'status_counts_all_positive_views':dict(Counter(x['status'] for x in empty)),'task02_empty_common_mask_count':sum(x['task02_previously_flagged_empty_common_mask'] for x in empty),'manual_review_contact_sheet_count':sum(x['status']!='nonempty_common_crop_mask' for x in empty),'task01_two_cases':['NG/OilStain/S0005 C1','NG/OilStain/S0005 C4'],'original_label_mutation':False},indent=2)+'\n')
    return report,empty,m

def contact_sheets(empty,manifest):
    cat='laser_diode'; records=manifest['categories'][cat]['shared_test_images']; group=defaultdict(list)
    for r in records:group[r['object_key']].append(r)
    font=ImageFont.load_default()
    exceptions=[e for e in empty if e['status']!='nonempty_common_crop_mask']
    for n,e in enumerate(exceptions,1):
        name=f'{n:02d}_{e["object_key"].replace("/","_")}_{e["camera_id"]}.png'
        if (OUT/'empty_mask_contact_sheet'/name).is_file():continue
        sib=sorted(group[e['object_key']],key=lambda x:x['view_id']); fig,axes=plt.subplots(5,3,figsize=(10,15))
        for ax,r in zip(axes,sib):
            im_path=DATA/cat/r['image_path']
            try:
                with Image.open(im_path) as im: image=im.convert('RGB').resize((256,256),Image.Resampling.BILINEAR)
            except Exception:image=Image.new('RGB',(256,256),'gray')
            if r.get('mask_path'):
                try:
                    with Image.open(DATA/cat/r['mask_path']) as mi: source=mi.convert('L')
                    source_view=source.resize((256,256),Image.Resampling.NEAREST)
                    crop=source.crop(COMMON).resize((256,256),Image.Resampling.NEAREST)
                except Exception:source_view=Image.new('L',(256,256),0);crop=Image.new('L',(256,256),0)
            else:source_view=Image.new('L',(256,256),0);crop=Image.new('L',(256,256),0)
            arr=np.asarray(image); src=np.asarray(source_view); cp=np.asarray(crop); label=f'C{r["view_id"]} label={r["label"]} {r["anomaly_class"]}'
            axes[r['view_id']-1,0].imshow(arr);axes[r['view_id']-1,0].set_title(label,fontsize=8)
            axes[r['view_id']-1,1].imshow(src,cmap='gray',vmin=0,vmax=255);axes[r['view_id']-1,1].set_title('uncropped source GT (display resized)',fontsize=8)
            axes[r['view_id']-1,2].imshow(cp,cmap='gray',vmin=0,vmax=255);axes[r['view_id']-1,2].set_title('common [64:960] crop GT',fontsize=8)
        for ax in axes.ravel():ax.axis('off')
        fig.suptitle(f'Empty-mask case {n:02d} | {e["object_key"]} | flagged {e["camera_id"]} | {e["status"]}',fontsize=10);fig.tight_layout(rect=(0,0,1,.98))
        fig.savefig(OUT/'empty_mask_contact_sheet'/name,dpi=110);plt.close(fig)

def alignment_and_sensitivity(manifest):
    rng=np.random.default_rng(20261102); alignment={'status':'PASS','common_source_xyxy':[64,64,960,960],'common_grid':[224,224],'random_seed':20261102,'sampled_mask_rechecks':[],'maps':{},'pixel_checks':'Per Task02 geometry: D2 [16:240] crop; ArcAD 256->224 bilinear. GT nearest-neighbor from original 1024 masks.'}
    for cat in manifest['categories']:
        for model in ('d2','arcad'):
            p=T2/'maps'/f'common_{model}_{cat}.npz'
            with np.load(p) as z:
                maps=z['smoothed_maps']; masks=z['gt_masks']; paths=z['image_paths'].astype(str).tolist()
                if maps.shape!=(len(paths),224,224) or masks.shape!=maps.shape:raise RuntimeError(f'{cat} {model} map shape mismatch {maps.shape}/{masks.shape}')
                if len(paths)!=len(set(paths)):raise RuntimeError('duplicate paths in map archive')
                if not np.isfinite(maps).all() or np.all(maps==0):raise RuntimeError(f'nonfinite or empty maps {cat} {model}')
                alignment['maps'][f'{model}/{cat}']={'n_views':len(paths),'shape':list(maps.shape),'finite':True,'min':float(maps.min()),'max':float(maps.max()),'std':float(maps.std()),'unique_paths':True}
        recs=manifest['categories'][cat]['shared_test_images']; sample=sorted(rng.choice(len(recs),size=min(3,len(recs)),replace=False).tolist())
        for i in sample:
            r=recs[i]
            if r.get('mask_path'):
                with Image.open(DATA/cat/r['mask_path']) as im: actual=np.asarray(im.convert('L').crop(COMMON).resize((224,224),Image.Resampling.NEAREST))>0
            else:actual=np.zeros((224,224),dtype=bool)
            for model in ('d2','arcad'):
                with np.load(T2/'maps'/f'common_{model}_{cat}.npz') as z:
                    pp=z['image_paths'].astype(str).tolist();j=pp.index(r['image_path']); saved=z['gt_masks'][j].astype(bool)
                if not np.array_equal(actual,saved):raise RuntimeError(f'mask alignment mismatch {cat} {r["image_path"]} {model}')
            alignment['sampled_mask_rechecks'].append({'category':cat,'image_path':r['image_path'],'pixel_exact_for_both_models':True})
    (OUT/'alignment_reverification.json').write_text(json.dumps(alignment,indent=2)+'\n')
    # Mask-label sensitivity on laser diode using Task02 maps and exact shared paths.
    cat='laser_diode'; recs=manifest['categories'][cat]['shared_test_images']; audit={r['image_path']:r for r in readcsv(OUT/'empty_mask_audit.csv')}
    keep=[r['image_path'] not in audit or audit[r['image_path']]['status']=='nonempty_common_crop_mask' for r in recs]
    sens=[]
    for model in ('d2','arcad'):
        with np.load(T2/'maps'/f'common_{model}_{cat}.npz') as z:maps=z['smoothed_maps'].astype(np.float32); mask=z['gt_masks'].astype(bool); paths=z['image_paths'].astype(str).tolist()
        recidx=[paths.index(r['image_path']) for r in recs]; maps=maps[recidx];mask=mask[recidx];
        for scheme,ids in [('all_original_views',np.ones(len(recs),bool)),('exclude_ambiguous_empty_or_nonvisible_positive_views',np.asarray(keep,bool))]:
            mm=mask[ids];ss=maps[ids]; y=mm.ravel().astype(np.uint8); scores=ss.ravel()
            sens.append({'model':model,'analysis':'common-grid pixel metrics','label_scheme':scheme,'n_views':int(ids.sum()),'n_positive_image_labels':sum(r['label'] for r,k in zip(recs,ids) if k),'P-AUROC':float(roc_auc_score(y,scores)) if len(np.unique(y))==2 else None,'P-AP':float(average_precision_score(y,scores)) if len(np.unique(y))==2 else None,'P-AUPRO_cap_0.3':pro_auc(mm,ss,.3),'annotation_note':'Ambiguous views excluded only from mask-dependent pixel metrics; source labels unchanged.'})
    # Optional diagnostic view-label sensitivity: set ambiguous views to unlabeled (drop them),
    # retaining original object labels for all objects.
    for model in ('d2','arcad'):
        scorefile=readcsv(T2/'joined_predictions.csv'); lookup={r['image_path']:float(r['score_view']) for r in scorefile if r['category']==cat and r['model']==model}
        ids=np.asarray(keep,bool); y=np.asarray([r['label'] for r in recs]);sc=np.asarray([lookup[r['image_path']] for r in recs])
        yy=y[ids];ss=sc[ids]; sens.append({'model':model,'analysis':'view AUROC/AP sensitivity, exploratory','label_scheme':'drop ambiguous mask views from image-label metric only','n_views':int(ids.sum()),'n_positive_image_labels':int(yy.sum()),'I-AUROC':float(roc_auc_score(yy,ss)) if len(np.unique(yy))==2 else None,'I-AP':float(average_precision_score(yy,ss)) if len(np.unique(yy))==2 else None,'annotation_note':'Exploratory only; does not edit original image or object labels.'})
        met={r['category']:r for r in readcsv(T2/'category_metrics.csv')}[cat]
        sens.append({'model':model,'analysis':'object-level label sensitivity','label_scheme':'unchanged original object labels; ambiguous view labels retained','n_views':len(recs),'n_positive_image_labels':sum(r['label'] for r in recs),'O-AUROC':float(met[f'{model}_O-AUROC']),'O-AP':float(met[f'{model}_O-AP']),'annotation_note':'Object scores and labels remain as in the original shared split; no object label is changed.'})
    writecsv(OUT/'label_sensitivity.csv',sens)

def main():
    d2,arc,raw,counts=categories_data(); data_audit,empty,manifest=manifest_audit();contact_sheets(empty,manifest);alignment_and_sensitivity(manifest)
    d2ck=D2RUN/'model.pth';ack=ARCRUN/'model.pth'
    try:
        import torch; cuda={'available':torch.cuda.is_available(),'torch_version':torch.__version__,'device':torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    except Exception as e:cuda={'error':str(e)}
    protocol={'run_id':'analysis_task03_20261102_01','source_metrics':{'D2':str(D2RUN/'category_heatmap/category_heatmap_metrics.json'),'ArcAD':str(ARCRUN/'results.csv')},'reported_macro_claims_verified':counts,'source_native_splits':{'D2':{'split_manifest':str(DS_D2),'all_categories':data_audit},'ArcAD':{'split_manifest':str(DS_ARC),'all_categories':data_audit},'note':'Per-category native test lists differ; 160-category overlap is a screening comparison, not paired inference.'},'pipeline':{'D2':{'checkpoint':str(d2ck),'sha256':sha(d2ck),'transform':'resize/crop 280/280','map_size':256,'smooth':'5x5 sigma 4','score':'top 1% map mean'},'ArcAD':{'checkpoint':str(ack),'sha256':sha(ack),'transform':'resize448 center crop392','map_size':256,'smooth':'5x5 sigma 4','score':'source results.csv native metrics; Task02 common maps reused'}},'code_revisions':{'Dinomaly2':git_info(D2),'ArcAD':git_info(ARC)},'device':cuda,'new_model_inference':'None; existing Task02 numeric maps reused. No training or checkpoint mutation.','alignment_reverification':'alignment_reverification.json','known_empty_mask_cases':len(empty),'mask_sensitivity':'label_sensitivity.csv','seed':20261102}
    (OUT/'protocol_audit.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print('160 counts',counts);print('empty mask statuses',dict(Counter(x['status'] for x in empty)))
if __name__=='__main__':main()
