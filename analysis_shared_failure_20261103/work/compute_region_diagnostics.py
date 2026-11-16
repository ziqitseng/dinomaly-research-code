import csv,json
from pathlib import Path
import numpy as np
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]));import run_analysis as ra
OUT=Path(__file__).resolve().parents[1];manifest=json.loads((OUT/'shared_test_manifest.json').read_text());rows=[]
for cat in ra.CATS:
    rs=sorted([r for r in manifest if r['category']==cat],key=lambda r:r['image_path'])
    z=np.load(OUT/'maps/common'/f'{cat}.npz',allow_pickle=False);masks=z['gt_masks'].astype(bool)
    for model,key in [('D2','d2_maps'),('ArcAD','arcad_maps')]:
        maps=z[key]
        for i,r in enumerate(rs):
            if not masks[i].any():continue
            flat=maps[i].ravel();k=max(1,int(.01*flat.size));pred=np.zeros(flat.size,bool);pred[np.argpartition(flat,-k)[-k:]]=True;pred=pred.reshape(256,256)
            matches,fp=ra.region_eval(masks[i],pred)
            for m in matches:rows.append({'category':cat,'image_path':r['image_path'],'object_key':r['object_key'],'view_id':r['camera_view_id'],'model':model,'prediction_rule':'top 1% common-crop pixels','false_pred_regions':fp,**m})
for p in [OUT/'region_diagnostics.csv',OUT/'tables/region_diagnostics.csv']:
    with open(p,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print(json.dumps({'rows':len(rows),'gt_regions':len({(r['category'],r['image_path'],r['model'],r['gt_region_id']) for r in rows})},indent=2))
