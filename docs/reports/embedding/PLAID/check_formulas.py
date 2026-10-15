"""Audit formulas and examples; this is not a PLAID benchmark reproduction."""
import ast
import json
from pathlib import Path
from types import SimpleNamespace
import torch

ROOT = Path(__file__).resolve().parent
torch.set_num_threads(2)
source = ROOT / 'sources/colbert/modeling/colbert.py'
tree = ast.parse(source.read_text())
functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in {'colbert_score', 'colbert_score_reduce'}]
assert len(functions) == 2

class Config:
    total_visible_gpus = 0
    interaction = 'colbert'

namespace = {'torch': torch, 'ColBERTConfig': Config}
exec(compile(ast.Module(body=functions, type_ignores=[]), str(source), 'exec'), namespace)
official_score = namespace['colbert_score']
outcomes = []

def check(name, condition, evidence):
    assert condition, (name, evidence)
    outcomes.append({'check': name, 'passed': True, 'evidence': evidence})

Q = torch.tensor([[1., 0.], [0., 1.]])
D = torch.tensor([[1., 0.], [-1., 0.], [0., 1.]])
score = official_score(Q[None], D[None], torch.ones(1, 3, dtype=torch.bool), Config()).item()
check('Equation 1: official MaxSim reduction', score == 2., {'score': score, 'all_pairs_sum': (Q@D.T).sum().item()})

masked = official_score(torch.tensor([[[1., 0.]]]), torch.tensor([[[-1.,0.],[0.,0.]]]),
                        torch.tensor([[True, False]]), Config()).item()
check('Padding excluded before max, including negative valid scores', masked == -1., {'score': masked})

one = torch.tensor([[2**-.5, 2**-.5]])
a = official_score(Q[None], one[None], torch.ones(1, 1, dtype=torch.bool), Config()).item()
b = official_score(one[None], Q[None], torch.ones(1, 2, dtype=torch.bool), Config()).item()
check('MaxSim generally asymmetric', abs(a-b) > .7, {'forward': a, 'reverse': b})

C = torch.tensor([[1.,0.], [0.,1.], [-1.,0.]])
codes = torch.tensor([0,2,1])
A = C @ Q.T
B = A[codes]
from_gather = B.max(dim=0).values.sum()
direct = (Q @ C[codes].T).max(dim=1).values.sum()
check('Equations 2–4: row gather and correct max axis', A.shape == (3,2) and B.shape == (3,2) and from_gather == direct,
      {'A_shape': list(A.shape), 'B_shape': list(B.shape), 'score': from_gather.item()})

duplicates = A[torch.tensor([0,0,2,1,1])].max(dim=0).values.sum()
check('Repeated centroid IDs do not change unpruned MaxSim', duplicates == from_gather, {'score': duplicates.item()})

keep = A.max(dim=1).values >= 1.
check('Equation 5: max over query axis, inclusive threshold', keep.tolist() == [True,True,False], {'keep': keep.tolist()})

centroid = torch.tensor([1.,0.]); original = torch.tensor([.8,.6])
under = ((Q[1]@centroid).item(), (Q[1]@original).item())
over = ((Q[0]@centroid).item(), (Q[0]@original).item())
check('Centroid approximation is neither guaranteed upper nor lower bound', under[0]<under[1] and over[0]>over[1],
      {'underestimate_approx_exact': under, 'overestimate_approx_exact': over})

gpu = 259.6/38.4; cpu = 4568.5/101.3
check('Table 3 speedup arithmetic and recall-point differences', round(gpu,1)==6.8 and round(cpu)==45,
      {'GPU_speedup': gpu, '8CPU_speedup': cpu, 'v1_R1000_drop_pp': round(98.3-97.5,1), 'v2_R1000_drop_pp': round(88.1-85.7,1)})

result = {'scope':'Pinned official pure-Python scoring functions plus educational arithmetic; no compiled engine, trained model or retrieval benchmark run',
          'checks':outcomes,'passed':len(outcomes),'failed':0}
(ROOT/'sources/formula-checks.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False,indent=2))
