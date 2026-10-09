import av,numpy as np,sys
from PIL import Image,ImageDraw
from pathlib import Path
D=Path(__file__).resolve().parent
name=sys.argv[1];runs=sys.argv[2:];videos=[]
for run in runs:
    path=next((D/'output').glob(run+'_*.mp4'))
    with av.open(str(path)) as container:
        videos.append([f.to_image().resize((432,240)) for f in container.decode(video=0)])
assert len({len(v) for v in videos})==1
path=D/(name+'.mp4')
with av.open(str(path),'w') as out:
    stream=out.add_stream('libx264',rate=24);stream.width=432*len(runs);stream.height=266;stream.pix_fmt='yuv420p';stream.options={'crf':'18','preset':'fast','threads':'2'}
    for i in range(len(videos[0])):
        sheet=Image.new('RGB',(stream.width,stream.height));draw=ImageDraw.Draw(sheet)
        for col,(run,video) in enumerate(zip(runs,videos)):
            sheet.paste(video[i],(432*col,26));draw.text((432*col+5,6),run,fill='white')
        frame=av.VideoFrame.from_ndarray(np.asarray(sheet),format='rgb24')
        for packet in stream.encode(frame):out.mux(packet)
    for packet in stream.encode():out.mux(packet)
print(path)
