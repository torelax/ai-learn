"""Check source coverage, tables, equations, links and figure integrity."""
import hashlib
import importlib.util
import json
import re
from collections import Counter
from decimal import Decimal
from pathlib import Path
from urllib.parse import unquote

import fitz
from bs4 import BeautifulSoup
from PIL import Image

ROOT = Path(__file__).resolve().parent
html = BeautifulSoup((ROOT/'sources/paper.html').read_text(), 'html.parser')
texts = {name:(ROOT/name).read_text() for name in ['en.md','zh.md','PLAID详解.md']}
results = []

def check(name, passed, detail):
    results.append({'check':name,'passed':bool(passed),'detail':detail})

source_ids = {p['id'] for p in html.select('p.ltx_p[id]')}
for name in ['en.md','zh.md']:
    ids = re.findall(r'<!-- ((?:abstract\d+|S\d+)\.[^>]+) -->',texts[name])
    check(name+' paragraph coverage',set(ids)==source_ids and len(ids)==len(source_ids),
          {'source':len(source_ids),'actual':len(ids),'missing':sorted(source_ids-set(ids))})
    check(name+' five original equations',len(re.findall(r'<!-- equation \d+ -->',texts[name]))==5,
          'Original LaTeX retained; dimensional caveat annotated in Chinese')
    for e in html.select('.ltx_equation[id]'):
        check(name+' '+e['id']+' LaTeX',e.find('math')['alttext'] in texts[name],e.find('math')['alttext'])
    check(name+' six tables',len(re.findall(r'<!-- table \d+ -->',texts[name]))==6,'Table 1–6')
    refs=re.findall(r'^\[(\d+)\] ',texts[name],re.M)
    check(name+' references',refs==[str(i) for i in range(1,55)],len(refs))

spec=importlib.util.spec_from_file_location('tables_formatter',ROOT/'format_english.py')
formatter=importlib.util.module_from_spec(spec);spec.loader.exec_module(formatter)
numeric_re=re.compile(r'^(\d+(?:\.\d+)?)([MB]?)(\*?)$')
def normalized(text):
    text=text.strip().replace('$','')
    m=numeric_re.fullmatch(text)
    if m:return str(Decimal(m[1]).normalize())+m[2]+m[3]
    if text in ['-','OOM']:return text
    return None
for f in html.select('figure.ltx_table'):
    n=int(f['id'].split('.T')[-1])
    tmp=BeautifulSoup(str(f),'html.parser')
    for m in tmp.find_all('math'):m.replace_with(m.get('alttext',''))
    source=Counter(x for cell in tmp.select('td,th') if (x:=normalized(cell.get_text(' ',strip=True))) is not None)
    formatted=Counter(x for row in formatter.table_md(n).splitlines()[2:] for cell in row.split('|')[1:-1]
                      if (x:=normalized(cell)) is not None)
    check(f'Table {n} numeric cells',source==formatted,
          {'source_count':sum(source.values()),'formatted_count':sum(formatted.values()),
           'source_only':dict(source-formatted),'formatted_only':dict(formatted-source)})

missing=[]
for name in ['en.md','zh.md','PLAID详解.md']:
    for target in re.findall(r'\]\(([^)]+)\)',texts[name]):
        if target.startswith(('http://','https://','#')):continue
        target=unquote(target.split('#')[0])
        if not (ROOT/target).exists():missing.append({'file':name,'target':target})
    check(name+' all eight figures',set(re.findall(r'figs/fig(\d{2})\.png',texts[name]))=={f'{i:02d}' for i in range(1,9)},'Figures 1–8')
    check(name+' semantic headings',not re.search(r'^#{1,6}\s+\d+\.',texts[name],re.M),'No numbered section headings')
check('Markdown relative links',not missing,missing)

run=json.loads(sorted((ROOT/'sources/runs').glob('*.json'))[-1].read_text())
check('Latest acquisition succeeded',run['outcome']=='assets_retrieved_manual_review_pending',run['outcome'])
check('All HTML components retrieved',len(run['images'])==16 and all(i['status']=='decoded' for i in run['images']),len(run['images']))
check('All three standalone legends retained',sum(i.get('legend',False) for i in run['images'])==3,'Figures 2, 3, 7')
check('Fifteen SVG sources retained',len(list((ROOT/'figs').glob('*.svg')))==15,15)
images=[]
for p in sorted((ROOT/'figs').glob('*.png')):
    try:
        with Image.open(p) as im:im.verify()
        with Image.open(p) as im:
            assert im.format=='PNG' and min(im.size)>20
            images.append({'file':p.name,'size':list(im.size),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()})
    except Exception as exc:check('PNG '+p.name,False,str(exc))
check('PNG decode and signatures',len(images)==24,{'verified':len(images)})

spec=importlib.util.spec_from_file_location('existing_pdf_figures',ROOT.parent/'_embed_figures.py')
pdf_helpers=importlib.util.module_from_spec(spec);spec.loader.exec_module(pdf_helpers)
with fitz.open(ROOT/'PLAID: An Efficient Engine for Late Interaction Retrieval_2205.09707.pdf') as doc:
    captions=pdf_helpers.collect_captions(doc)
    page_map={num:page+1 for num,page,_ in captions}
    check('PDF title/version/page count',len(doc)==10 and '2205.09707v1' in doc[0].get_text(),len(doc))
check('PDF Figure inventory matches HTML',set(page_map)==set(range(1,9))==set(run['html']['caption_numbers']),page_map)

source_manifest=json.loads((ROOT/'sources/manifest.json').read_text())
for record in source_manifest:
    path=ROOT/'sources'/record['path']
    if record['path']=='LICENSE':path=ROOT/'sources/LICENSE.ColBERT'
    check('Source hash '+record['path'],hashlib.sha256(path.read_bytes()).hexdigest()==record['sha256'],record['commit'])
formula=json.loads((ROOT/'sources/formula-checks.json').read_text())
check('Formula numerical checks',formula['passed']==8 and formula['failed']==0,formula['scope'])
audit={'passed':sum(r['passed'] for r in results),'failed':sum(not r['passed'] for r in results),
       'checks':results,'pdf_figure_pages':page_map,'images':images,
       'limits':['Structural coverage does not establish perfect translation semantics.',
                 'Human review of wording and paper inconsistencies remains pending.',
                 'No billion-scale benchmark, compiled PLAID engine or model training reproduced.']}
(ROOT/'sources/audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in audit.items() if k not in ['images','checks']},ensure_ascii=False,indent=2))
for result in results:
    if not result['passed']:print('FAIL',result)
raise SystemExit(1 if audit['failed'] else 0)
