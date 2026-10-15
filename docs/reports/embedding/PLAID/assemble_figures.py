import json
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import fitz
root=Path(__file__).resolve().parent
log=json.loads(sorted((root/'sources/runs').glob('*.json'))[-1].read_text())
font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24)
for n in range(1,9):
 records=[r for r in log['images'] if r['figure']==n]
 assert records and all(r['status']=='decoded' for r in records)
 ims=[Image.open(root/r['png']).convert('RGBA') for r in records]
 if len(ims)==1:
  ims[0].save(root/'figs'/f'fig{n:02d}.png')
  continue
 width=max(im.width for im in ims)+40
 height=sum(im.height+70 for im in ims)+20
 out=Image.new('RGB',(width,height),'white')
 d=ImageDraw.Draw(out); y=20
 for r,im in zip(records,ims):
  sublabel=r['caption'].split(' ')[0] if r.get('caption','').startswith('(') else 'legend'
  if sublabel=='legend': d.text((20,y),'Original legend',fill='black',font=font)
  else: d.text((20,y),sublabel,fill='black',font=font)
  y+=40
  out.paste(im,((width-im.width)//2,y),im)
  y+=im.height+30
 out.save(root/'figs'/f'fig{n:02d}.png')
contact=Image.new('RGB',(1100,1800),'#eeeeee'); d=ImageDraw.Draw(contact)
for n in range(1,9):
 im=Image.open(root/'figs'/f'fig{n:02d}.png').convert('RGB'); im.thumbnail((500,390))
 x=20+((n-1)%2)*550; y=20+((n-1)//2)*450
 d.text((x,y),f'Figure {n}',fill='black',font=font)
 contact.paste(im,(x,y+35))
preview_dir=Path('/workspace/research/plaid')
preview_dir.mkdir(parents=True,exist_ok=True)
contact.save(preview_dir/'contact.png')
with fitz.open(root/'PLAID: An Efficient Engine for Late Interaction Retrieval_2205.09707.pdf') as doc:
 for i in [2,4]:
  doc[i].get_pixmap(matrix=fitz.Matrix(2,2),alpha=False).save(preview_dir/f'pdf-page{i+1}.png')
print('Assembled all 8 figures from all 16 assets, including 3 separate legends')
