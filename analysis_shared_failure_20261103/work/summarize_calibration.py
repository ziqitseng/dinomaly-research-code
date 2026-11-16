import json
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

OUT=Path(__file__).resolve().parents[1]
CATS=['lattice_block_plug','laser_diode','power_jack','LED_indicator','aircraft_model_head']
result={}
for cat in CATS:
    result[cat]={}
    for model,folder in [('ArcAD','arcad'),('D2','d2')]:
        p=OUT/'calibration_maps'/folder/f'{cat}.npz'
        with np.load(p,allow_pickle=False) as z:
            maps=z['maps'].astype(np.float32);objects=z['object_keys'].astype(str)
        if model=='D2':
            y,x=torch.meshgrid(torch.arange(256,dtype=torch.float32),torch.arange(256,dtype=torch.float32),indexing='ij')
            sx=.875*(x+.5)+15.5;sy=.875*(y+.5)+15.5
            grid=torch.stack([2*(sx+.5)/256-1,2*(sy+.5)/256-1],-1).unsqueeze(0).expand(len(maps),-1,-1,-1)
            with torch.no_grad():maps=F.grid_sample(torch.from_numpy(maps[:,None]),grid,mode='bilinear',padding_mode='border',align_corners=False)[:,0].numpy()
        view_scores=[];max_view_scores=[];obj_scores=[];unique=sorted(set(objects))
        for obj in unique:
            ix=np.where(objects==obj)[0];group=maps[ix];flatviews=group.reshape(len(ix),-1);kv=max(1,int(flatviews.shape[1]*.01))
            view_scores.extend(np.partition(flatviews,-kv,axis=1)[:,-kv:].mean(axis=1).tolist());max_view_scores.append(float(np.max(np.partition(flatviews,-kv,axis=1)[:,-kv:].mean(axis=1))))
            flat=group.reshape(-1);k=max(1,int(flat.size*.01));obj_scores.append(float(np.partition(flat,-k)[-k:].mean()))
        result[cat][model]={'view_scores':view_scores,'max_view_scores':max_view_scores,'object_scores':obj_scores,'objects':len(unique),'views':len(view_scores),
            'source':'model native training-normal split mapped to common crop; max-per-normal-object for view operating points; pooled five-view score for object points; in-sample'}
(OUT/'calibration_scores.json').write_text(json.dumps(result,indent=2)+'\n')
print('wrote in-sample normal reference score distributions')
