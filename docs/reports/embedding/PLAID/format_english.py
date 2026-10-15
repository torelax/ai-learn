"""Convert the licensed arXiv HTML into readable English Markdown."""
import re
from pathlib import Path
from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parent
soup = BeautifulSoup((ROOT / 'sources/paper.html').read_text(), 'html.parser')
equations = {e['id']: e.find('math')['alttext'] for e in soup.select('.ltx_equation[id]')}
for m in soup.find_all('math'):
    m.replace_with('$' + m.get('alttext', '') + '$')
for note in soup.select('.ltx_note'):
    note.decompose()

TABLES = {
1: ('Dataset | Passages | Tokens | Queries | Vanilla index GiB | PLAID index GiB', [
 'MS MARCO v1 | 8.8M | 597.9M | 6980 | 24.6 | 21.6',
 'Wikipedia | 21.0M | 2.6B | 8757 | 105.2 | 92.0',
 'LoTTE pooled | 2.4M | 339.4M | 2931 | 14.0 | 12.3',
 'MS MARCO v2 | 138.4M | 9.4B | 3903 | 246.0 | 202.2']),
2: ('k | nprobe | t_cs | ndocs', ['10 | 1 | 0.5 | 256', '100 | 2 | 0.45 | 1024', '1000 | 4 | 0.4 | 4096']),
3: ('System | MRR@10 | R@100 | R@1k | 1-CPU ms | 8-CPU ms | GPU ms', [
 'BM25 (PISA; k=1000) | 18.7* | - | - | 8.3* | - | -',
 'SPLADEv2 (PISA; k=1000) | 36.8* | - | 97.9* | 220.3* | - | -',
 'ColBERTv1 | 36.1 | 87.3 | 95.2 | - | - | 54.3',
 'Vanilla ColBERTv2 (p=2,c=2^13) | 39.7 | 90.4 | 96.6 | 3485.1 | 921.8 | 53.4',
 'Vanilla ColBERTv2 (p=4,c=2^16) | 39.7 | 91.4 | 98.3 | - | 4568.5 | 259.6',
 'PLAID ColBERTv2 (k=10) | 39.4 | - | - | 185.5 | 31.5 | 11.5',
 'PLAID ColBERTv2 (k=100) | 39.8 | 90.6 | - | 222.3 | 52.9 | 20.2',
 'PLAID ColBERTv2 (k=1000) | 39.8 | 91.3 | 97.5 | 352.3 | 101.3 | 38.4']),
4: ('System | Success@5 | Success@100 | 8-CPU ms | GPU ms', [
 'DPR | 66.8 | 85.0 | - | -',
 'ColBERT-QA uncompressed | 75.3 | 89.2 | - | -',
 'Vanilla ColBERT-QA (p=4,c=2^15; residual compression) | 74.3 | 89.0 | 5077.9 | 204.1',
 'PLAID ColBERT-QA (k=10; residual compression) | 73.3 | - | 67.1 | 13.6',
 'PLAID ColBERT-QA (k=100; residual compression) | 74.1 | 88.0 | 120.1 | 26.9',
 'PLAID ColBERT-QA (k=1000; residual compression) | 74.4 | 88.9 | 228.4 | 55.3']),
5: ('System | Success@5 | Success@100 | 8-CPU ms | GPU ms', [
 'BM25 | 47.8* | 77.6* | - | -', 'SPLADEv2 | 67.0* | 89.0* | - | -',
 'Vanilla ColBERTv2 (p=2,c=2^13) | 69.3 | 90.3 | 1508.4 | 66.9',
 'PLAID ColBERTv2 (k=10) | 69.1 | - | 35.5 | 9.2',
 'PLAID ColBERTv2 (k=100) | 69.4 | 89.9 | 64.8 | 17.4',
 'PLAID ColBERTv2 (k=1000) | 69.6 | 90.5 | 163.1 | 27.3']),
6: ('System | MRR@100 | R@100 | R@1k | 8-CPU ms | GPU ms', [
 'BM25 (Anserini; augmented) | 8.7 | 40.3 | 69.3 | - | -',
 'Vanilla ColBERTv2 (p=4,c=2^16) | 18.0 | 68.2 | 88.1 | 5228.5 | OOM',
 'PLAID ColBERTv2 (k=10) | - | - | - | 136.4 | 47.1',
 'PLAID ColBERTv2 (k=100) | 17.9 | 67.0 | - | 181.9 | 96.1',
 'PLAID ColBERTv2 (k=1000) | 18.0 | 68.4 | 85.7 | 251.3 | OOM'])}

def table_md(n):
    head, rows = TABLES[n]
    return '\n'.join(['| ' + head + ' |', '| ' + ' | '.join(['---'] * len(head.split('|'))) + ' |']
                     + ['| ' + row + ' |' for row in rows])

def main():
    out = ['# PLAID: An Efficient Engine for Late Interaction Retrieval',
           '> Keshav Santhanam, Omar Khattab, Christopher Potts, Matei Zaharia. arXiv:2205.09707v1 (2022-05-19).',
           '> Source: https://arxiv.org/html/2205.09707v1 · License: https://creativecommons.org/licenses/by/4.0/',
           '> HTML-to-Markdown conversion: semantic headings, normalized tables and composed figure panels. Original equations and source inconsistencies are preserved; see the Chinese study notes for dimensional corrections.']
    for node in soup.find_all(True):
        if node.name in ['h2', 'h3', 'h4', 'h6'] and 'ltx_title' in node.get('class', []):
            title = re.sub(r'^\d+(?:\.\d+)*\.?\s*', '', node.get_text(' ', strip=True))
            if title == 'References' or 'reporting errors' in title: continue
            out.append(('## ' if node.name in ['h2','h6'] else '### ') + title)
        elif node.name == 'p' and 'ltx_p' in node.get('class', []):
            if node.find_parent(class_='ltx_bibitem') or node.find_parent('figure'): continue
            text = re.sub(r'\s+', ' ', node.get_text(' ', strip=True))
            out.append(f'<!-- {node.get("id")} -->\n{text}')
        elif node.get('id') in equations:
            number = node['id'].split('.E')[-1]
            out.append(f'<!-- equation {number} -->\n$$\n{equations[node["id"]]}\n$$\n\nEquation ({number}).')
        elif node.name == 'figure' and re.fullmatch(r'.*\.F\d+', node.get('id','')):
            n = int(node['id'].split('.F')[-1]); cap=node.find('figcaption',recursive=False)
            out.append(f'![Figure {n}](figs/fig{n:02d}.png)\n\n' + cap.get_text(' ',strip=True))
        elif node.name == 'figure' and 'ltx_table' in node.get('class', []):
            n = int(node['id'].split('.T')[-1])
            out.append(f'<!-- table {n} -->\n'+table_md(n)+'\n\n'+node.find('figcaption').get_text(' ',strip=True))
    refs = ['## References (original bibliographic entries)']
    for n, item in enumerate(soup.select('li.ltx_bibitem'), 1):
        refs.append(f'[{n}] ' + re.sub(r'\s+',' ',item.get_text(' ',strip=True)))
    out += ['## Original footnotes',
            '* Santhanam and Khattab contributed equally.',
            '1. Code maintained at https://github.com/stanford-futuredata/ColBERT. As of May 2022, PLAID lies under the branch fast_search but will soon be merged upstream.',
            '2. The 32-bit passage-ID representation assumes no more than 2^32 (4 billion) passages, about 30 times the MS MARCO v2 scale.'] + refs
    (ROOT/'en.md').write_text('\n\n'.join(out)+'\n')
    (ROOT/'sources/references.md').write_text('\n\n'.join(refs)+'\n')
    (ROOT/'sources/tables.md').write_text('\n\n'.join(f'## Table {n}\n\n'+table_md(n) for n in TABLES)+'\n')
    print('Converted 60 paragraphs, 5 equations, 6 tables, 8 figures, 54 references')

if __name__ == '__main__':
    main()
