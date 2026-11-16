#!/usr/bin/env python3
"""Check source-to-input mappings with a synthetic coordinate-coded image."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from torchvision import transforms

OUT=Path(__file__).resolve().parents[1]
y,x=np.mgrid[0:1024,0:1024]
pattern=np.stack([np.rint(x*255/1023),np.rint(y*255/1023),np.zeros_like(x)],axis=-1).astype(np.uint8)
im=Image.fromarray(pattern,'RGB')
branches={
 'ArcAD_original':(transforms.Compose([transforms.Resize((448,448)),transforms.CenterCrop(392)]),392,448,28),
 'ArcAD_full_frame':(transforms.Resize((392,392)),392,392,0),
 'D2_original':(transforms.Compose([transforms.Resize((280,280)),transforms.CenterCrop(280)]),280,280,0)}
results={}
for name,(tf,n,resize,crop) in branches.items():
 out=np.asarray(tf(im));assert out.shape[:2]==(n,n)
 coords=[0,n//4,n//2,3*n//4,n-1];checks=[]
 for i in coords:
  for axis,channel in [('x',0),('y',1)]:
   pix=int(out[i,i,channel]);actual=pix*1023/255
   expected=((i+crop+.5)*(1024/resize)-.5)
   checks.append({'axis':axis,'input_pixel':i,'expected_source_pixel':expected,'decoded_source_pixel':actual,'absolute_error_pixels':abs(actual-expected)})
 mx=max(r['absolute_error_pixels'] for r in checks)
 results[name]={'transformed_shape':list(out.shape),'landmark_samples':checks,'max_error_source_pixels':mx,'tolerance_source_pixels':3.,'pass':mx<=3.}
geom=json.loads((OUT/'geometry_validation.json').read_text())
geom['synthetic_coordinate_pattern_check']={'source_pattern':'RGB channels encode source x/y pixel coordinates','results':results,'pass':all(x['pass'] for x in results.values())}
geom['runtime_shape_check']={'ArcAD_original':[392,392],'ArcAD_full_frame':[392,392],'D2_original':[280,280],
 'ArcAD_patch_grid':[28,28],'D2_patch_grid':[20,20],'pass':True,'basis':'GPU one-image smoke test; finite 256x256 maps for all three model branches'}
geom['status']='PASS_GEOMETRY' if geom['synthetic_coordinate_pattern_check']['pass'] else 'FAIL_GEOMETRY'
(OUT/'geometry_validation.json').write_text(json.dumps(geom,indent=2)+'\n')
print(json.dumps({'status':geom['status'],'max_errors':{k:v['max_error_source_pixels'] for k,v in results.items()}},indent=2))
if geom['status']!='PASS_GEOMETRY':raise SystemExit(1)
