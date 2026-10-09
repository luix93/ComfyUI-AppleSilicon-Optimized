from pathlib import Path
import sys,av
from PIL import Image,ImageDraw
D=Path(__file__).resolve().parent
names=sys.argv[1:]
for name in names:
    video=next((D/'output').glob(name+'_*.mp4'))
    with av.open(str(video)) as container:
        frames=list(container.decode(video=0))
    idx=[round(i*(len(frames)-1)/7) for i in range(8)]
    sheet=Image.new('RGB',(432*4,267*2))
    for n,i in enumerate(idx):
        im=frames[i].to_image().resize((432,240))
        x=(n%4)*432;y=(n//4)*267
        sheet.paste(im,(x,y+27));ImageDraw.Draw(sheet).text((x+4,y+6),name+f' frame {i}',fill='white')
    sheet.save(D/(name+'-contact.jpg'),quality=94)
    print(D/(name+'-contact.jpg'))
