from pathlib import Path
import csv, json, math, statistics
import numpy as np
p=Path(__file__).resolve().parent
with (p/'participant_seed_pairs.csv').open(encoding='utf-8-sig',newline='') as f:
    rows=list(csv.DictReader(f))
expected=json.loads((p/'STATISTICAL_RESULTS.json').read_text(encoding='utf-8'))
draws={}
out=[]
for item in expected['intervals']:
    scope,obj=item['study'],item['objective']
    selected=[r for r in rows if r['study']==scope and r['objective']==obj]
    people=sorted({int(r['subject']) for r in selected})
    effects=[]
    for subject in people:
        person=[r for r in selected if int(r['subject'])==subject]
        assert sorted(int(r['seed']) for r in person)==[42,43,44]
        effects.append(statistics.mean(float(r['D_minus_G_bpm']) for r in person))
    n=len(people)
    if scope not in draws:
        draws[scope]=np.random.Generator(np.random.PCG64(item['rng_seed'])).integers(0,n,size=(200000,n))
    lo,hi=np.quantile(np.asarray(effects)[draws[scope]].mean(axis=1),[.025,.975],method='linear')
    point=statistics.mean(effects)
    for actual,key in [(point,'mean_D_minus_G_bpm'),(lo,'bootstrap95_lower_bpm'),(hi,'bootstrap95_upper_bpm')]:
        assert math.isclose(actual,item[key],rel_tol=1e-12,abs_tol=1e-10),(scope,obj,key,actual,item[key])
    out.append(dict(study=scope,objective=obj,people=n,mean=point,lower=float(lo),upper=float(hi)))
print(json.dumps(out,indent=2))
print('All six conditional descriptive intervals match the archived results.')
