import json
from pathlib import Path

ROOT=Path('/media/avlab/7458a474-6ea2-4b90-b232-373c3df8d329/real_iad_variety_1024')
OUT=Path(__file__).resolve().parents[1]
CATS=['lattice_block_plug','laser_diode','power_jack','LED_indicator','aircraft_model_head']

def obj(p):
    q=p.split('/')
    return '/'.join(q[:2] if q[0]=='OK' else q[:3])

def resolve(c,p):
    x=ROOT/c/p
    return x if x.is_file() else ROOT/c/c/p

result={'status':'PASS','calibration_kind':'normal-only training reference; in-sample for both models; not independent validation','categories':{}}
for cat in CATS:
    arc=json.loads((ROOT/'realiadvariety_jsons/sup'/f'{cat}.json').read_text())
    d2=json.loads((ROOT/'realiadvariety_jsons/Real-IAD_Variety_jsons'/f'{cat}.json').read_text())
    arc_rows=[x for x in arc['labeled'] if int(x['label'])==0]
    d2_rows=[x for x in d2['train'] if x.get('anomaly_class')=='OK']
    for model, rows in [('ArcAD_train_normals',arc_rows),('D2_train_normals',d2_rows)]:
        grouped={}
        for x in rows: grouped.setdefault(obj(x['image_path']),[]).append(x)
        bad={k:len(v) for k,v in grouped.items() if len(v)!=5}
        if bad: result['status']='FAIL'
        recs=[]
        for key,views in sorted(grouped.items()):
            for x in sorted(views,key=lambda z:z['image_path']):
                ip=resolve(cat,x['image_path'])
                if not ip.is_file(): result['status']='FAIL'
                recs.append({'category':cat,'object_key':key,'image_path':x['image_path'],
                    'image_abs':str(ip),'label':0,'camera_view_id':int(x['image_path'].split('_C')[-1][0]),
                    'mask_path':None,'split':model})
        result['categories'].setdefault(cat,{})[model]={'objects':len(grouped),'views':len(recs),'records':recs,'invalid_groups':bad}
(OUT/'calibration_manifest.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({'status':result['status'],'counts':{c:{m:{'objects':v['objects'],'views':v['views']} for m,v in x.items()} for c,x in result['categories'].items()}},indent=2))
if result['status']!='PASS': raise SystemExit(1)
