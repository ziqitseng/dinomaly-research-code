import csv,json
from pathlib import Path
from PIL import Image,ImageDraw
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=Path(__file__).resolve().parents[1]
manifest=json.loads((ROOT/'analysis_shared_failure_20261103/shared_test_manifest.json').read_text());by_path={r['image_path']:r for r in manifest}
audit=list(csv.DictReader(open(OUT/'empty_mask_audit.csv',newline='')))
byobj={}
for r in manifest:
 if r['category']=='laser_diode':byobj.setdefault(r['object_key'],[]).append(r)
cases=[r for r in audit if r['status'] in ('genuinely_all_zero_source_mask','nonzero_mask_outside_[64:960]_crop')]
dst=OUT/'empty_mask_contact_sheet';dst.mkdir(exist_ok=True);cw=148;ch=148;page_n=4
for page,start in enumerate(range(0,len(cases),page_n),1):
 group=cases[start:start+page_n];canvas=Image.new('RGB',(cw*8,(ch+31)*len(group)+36),'white');draw=ImageDraw.Draw(canvas)
 draw.text((5,5),f'Laser diode empty common-mask audit | cases {start+1}-{start+len(group)} of {len(cases)} | source GT max-pooled for display only',fill='black')
 for ri,r in enumerate(group):
  y=28+ri*(ch+31);source=by_path[r['image_path']];mp=Path(source['mask_abs'])
  try:
   source_mask=np.asarray(Image.open(mp).convert('L'))>0
   h,w=source_mask.shape;small=np.zeros((cw,cw),bool)
   for yy in range(cw):
    y0=yy*h//cw;y1=max(y0+1,(yy+1)*h//cw)
    for xx in range(cw):
     x0=xx*w//cw;x1=max(x0+1,(xx+1)*w//cw);small[yy,xx]=source_mask[y0:y1,x0:x1].any()
   full=Image.fromarray((small.astype(np.uint8)*255)).convert('RGB')
  except Exception:full=Image.new('RGB',(cw,cw),'gray')
  try:
   mm=Image.open(mp).convert('L').crop((64,64,960,960)).resize((224,224),Image.Resampling.NEAREST).resize((cw,cw),Image.Resampling.NEAREST).convert('RGB')
  except Exception:mm=Image.new('RGB',(cw,cw),'gray')
  try:target=Image.open(source['image_abs']).convert('RGB').resize((cw,cw))
  except Exception:target=Image.new('RGB',(cw,cw),'gray')
  panels=[target,full,mm];titles=['target image','full source GT (max-pooled display)','common crop GT (224)']
  sib=sorted(byobj[r['object_key']],key=lambda z:z['camera_view_id'])
  for s in sib:
   try:panels.append(Image.open(s['image_abs']).convert('RGB').resize((cw,cw)))
   except Exception:panels.append(Image.new('RGB',(cw,cw),'gray'))
   titles.append(f"camera C{s['camera_view_id']}")
  for j,(im,title) in enumerate(zip(panels,titles)):
   canvas.paste(im,(j*cw,y));draw.text((j*cw+2,y+2),title,fill='yellow' if j==0 or j>=3 else 'red',stroke_width=1,stroke_fill='black')
  draw.text((4,y+cw+2),f"{r['object_key']} C{r['camera_view_id']} | source={r['source_mask_pixels']} px, crop={r['common_crop_source_pixels']} px, 224={r['common_crop_224_pixels']} px | {r['status']} | manual review required",fill='black')
 canvas.save(dst/f'page_{page:02d}.png')
print(f'rendered {len(cases)} cases in {len(list(dst.glob("*.png")))} pages')
