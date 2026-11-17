"""Reanalyse bundled, already-fixed development predictions; stdlib only."""
from pathlib import Path
import argparse,csv,json,collections,math,statistics
def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1]);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 root=a.root;cfg=json.loads((root/'experiments/candidate_versions/supplemental_cv_loss_v1_1/config.json').read_text())
 rows=[];cells={}
 for run in cfg['runs']:
  groups=collections.defaultdict(list)
  with (root/'evidence/development_cv'/run['id']/'outer_predictions.csv').open(newline='') as f:
   for r in csv.DictReader(f):
    assert r['run_id']==run['id'] and r['eligible']=='1' and r['status']=='ok'
    assert int(r['subject']) in cfg['folds'][run['fold']-1]['outer']
    error=abs(float(r['prediction_bpm'])-float(r['reference_bpm']));assert math.isfinite(error)
    groups[int(r['subject'])].append(error)
  for subject,errors in groups.items():
   value=statistics.mean(errors);cells[run['objective'],run['method'],run['seed'],subject]=value
   rows.append({'run_id':run['id'],'fold':run['fold'],'objective':run['objective'],'method':run['method'],'seed':run['seed'],'subject':subject,'n':len(errors),'mae_bpm':value})
 original=json.loads((root/'evidence/development_cv/aggregate.json').read_text());summaries={};maxdiff=0.
 for key,expected in original['summaries'].items():
  objective,method=key.split('/');values=[statistics.mean(cells[objective,method,seed,s] for s in range(1,13)) for seed in (42,43,44)]
  diff=max(abs(x-y) for x,y in zip(values,expected['per_seed_equal12_subject_mae']));maxdiff=max(maxdiff,diff)
  assert diff<1e-5,(key,diff)
  summaries[key]={'per_seed_equal12_subject_mae':values,'mean':statistics.mean(values),'seed_sample_sd':statistics.stdev(values)}
 a.output.mkdir(parents=True,exist_ok=True)
 with (a.output/'subject_seed_mae.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 result={'scope':'already-fixed posthoc development CV; no final-test access or inference','subjects':12,'arms':len(cfg['runs']),'maximum_absolute_difference_from_original_seed_mae':maxdiff,'summaries':summaries}
 (a.output/'summary.json').write_text(json.dumps(result,indent=2));print(json.dumps({'arms':90,'max_difference':maxdiff}))
if __name__=='__main__':main()
