"""Read committed development histories only; no torch or inference imports."""
from pathlib import Path
import csv,hashlib,json,statistics,collections
ROOT=Path(__file__).resolve().parents[1];OUT=Path(__file__).resolve().parent/'stopping';OUT.mkdir(exist_ok=True)
BUNDLE=ROOT/'results/supplemental_cv_loss_v1_1_20261108T033819Z'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def csvout(name,rows):
 with (OUT/name).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
cfg=read(ROOT/'candidate_versions/supplemental_cv_loss_v1_1/config.json');arms=[];curves=[];sources=[]
for run in cfg['runs']:
 arm=BUNDLE/run['id'];commit=read(arm/'train_COMMIT.json')
 assert commit['identity']['run_id']==run['id']
 for name in ('history.json','summary.json'):
  expected=next(r['sha256'] for r in commit['files'] if r['name']==name)
  assert sha(arm/name)==expected
  sources.append({'run_id':run['id'],'source':str((arm/name).relative_to(ROOT)).replace('\\','/'),'sha256':expected,'train_commit_sha256':sha(arm/'train_COMMIT.json')})
 hist=read(arm/'history.json');s=read(arm/'summary.json')
 assert s['status']=='completed' and len(hist)==s['epochs']
 assert [x['epoch'] for x in hist]==list(range(1,len(hist)+1))
 best=min(hist,key=lambda x:x['inner_equal_person_mae']) # first strict minimum
 assert best['updates']==s['best_update'] and best['inner_equal_person_mae']==s['best_inner_mae']
 ident={k:run[k] for k in ('id','fold','method','objective','seed')}
 arms.append(dict(ident,actual_stopping_epoch=s['epochs'],actual_updates=s['actual_updates'],selected_best_epoch=best['epoch'],selected_best_updates=s['best_update'],best_inner_mae_bpm=s['best_inner_mae'],stop_reason=s['stop_reason'],stale_checks=s['stale'],epoch_cap=200,patience=20,history_sha256=sha(arm/'history.json')))
 for h in hist:
  curves.append(dict(ident,epoch=h['epoch'],updates=h['updates'],train_objective_raw=h['train_objective'],inner_equal_person_mae_bpm=h['inner_equal_person_mae'],inner_pooled_mae_bpm=h['scores']['pooled_mae'],inner_logcosh_macro_raw=h['scores']['validation_macro_logcosh'],selected_best=int(h['epoch']==best['epoch']),smoothed=0))
groups=collections.defaultdict(list)
for a in arms:groups[a['fold'],a['method'],a['objective']].append(a)
summary=[]
for (fold,method,objective),rows in groups.items():
 d={'fold':fold,'method':method,'objective':objective,'n_seed_runs':len(rows),'descriptive_scope':'within_fold_only'}
 for metric in ('actual_stopping_epoch','selected_best_epoch','actual_updates','selected_best_updates'):
  values=[r[metric] for r in rows];d.update({metric+'_min':min(values),metric+'_median':statistics.median(values),metric+'_mean':statistics.mean(values),metric+'_max':max(values)})
 d['stop_reason_counts']=json.dumps(dict(collections.Counter(r['stop_reason'] for r in rows)),sort_keys=True);summary.append(d)
assert len(arms)==90
csvout('stopping_by_run.csv',arms);csvout('stopping_by_fold_method_objective.csv',summary);csvout('inner_validation_curves_unsmoothed.csv',curves);csvout('source_hashes.csv',sources)
stats={'arms':len(arms),'curve_rows':len(curves),'stop_reasons':dict(collections.Counter(a['stop_reason'] for a in arms)),'stopping_epoch_min':min(a['actual_stopping_epoch'] for a in arms),'stopping_epoch_max':max(a['actual_stopping_epoch'] for a in arms),'selected_epoch_min':min(a['selected_best_epoch'] for a in arms),'selected_epoch_max':max(a['selected_best_epoch'] for a in arms),'all_history_summary_hashes_match_commits':True}
(OUT/'CHECKS.json').write_text(json.dumps(stats,indent=2),encoding='utf-8')
(OUT/'README.md').write_text('''# Committed development stopping histories

All90 registered development arms; source histories and summaries checked against each train_COMMIT SHA256. No model loading, new training/inference, final-test access or smoothing.

stopping_by_run.csv gives actual stopping epoch/update and selected best epoch/update. The selected epoch is the first strictly minimal inner subject-equal MAE, checked against recorded best_update and best_inner_mae. stopping_by_fold_method_objective.csv describes three seeds within each fold; means/medians are descriptive, not independent-subject evidence. Raw per-run curves are in inner_validation_curves_unsmoothed.csv. Compare MAE columns in bpm; raw objective columns have different scales/units and must not be compared across loss objectives. Curves have no interpolation, smoothing, or averaging. source_hashes.csv binds every source history/summary to its commit.

Shared optimizer settings, patience20 and epoch cap200 are execution controls. They do not isolate a pure effect of loss formulation or prove all runs sufficiently converged. A recorded early stop indicates20 validation checks without a strictly lower inner MAE under this recipe; it does not establish a global optimum. The retrospective development folds are not a new untouched final test.
''',encoding='utf-8')
print(json.dumps(stats))
