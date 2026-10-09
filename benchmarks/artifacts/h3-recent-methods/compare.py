from pathlib import Path
import sys,json,torch,math
from PIL import Image,ImageDraw
R=Path(__file__).resolve().parent
ref=sys.argv[1];names=sys.argv[2:]
def load(name,kind):return torch.load(R/(name+'-'+kind+'.pt'),map_location='cpu',weights_only=True)
report={}
for name in names:
 report[name]={}
 for kind in ['latents','frames','audio_decode']:
  a=load(ref,kind);b=load(name,kind);out={}
  for key,x in a.items():
   y=b[key];xf=x.float();yf=y.float();delta=xf-yf
   row=dict(shape=list(x.shape),equal=torch.equal(x,y),relative_l2=float(delta.norm()/xf.norm().clamp_min(1e-20)),rmse=float(delta.square().mean().sqrt()),max_abs=float(delta.abs().max()),finite=bool(torch.isfinite(y).all()))
   if kind=='frames':
    row['psnr']=-10*math.log10(max(float(delta.square().mean()),1e-30));row['mean_rgb_delta']=(yf.mean((0,1,2))-xf.mean((0,1,2))).tolist()
    if x.shape[0]>1:row['temporal_delta_rmse']=float((delta[1:]-delta[:-1]).square().mean().sqrt())
   out[key]=row
  report[name][kind]=out
 images=[]
 for label,run in [('exact',ref),(name,name)]:
  f=next(iter(load(run,'frames').values()));indices=[0,len(f)//2,len(f)-1]
  row=[]
  for i in indices:
   im=Image.fromarray((f[i].float().clamp(0,1).numpy()*255).round().astype('uint8')).resize((432,240))
   tile=Image.new('RGB',(432,265));tile.paste(im,(0,25));ImageDraw.Draw(tile).text((6,5),label+' frame '+str(i),fill='white');row.append(tile)
  images.append(row)
 sheet=Image.new('RGB',(1296,530))
 for y,row in enumerate(images):
  for x,im in enumerate(row):sheet.paste(im,(x*432,y*265))
 sheet.save(R/(name+'-compare.jpg'),quality=95)
(R/(names[0]+'-quality.json')).write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
