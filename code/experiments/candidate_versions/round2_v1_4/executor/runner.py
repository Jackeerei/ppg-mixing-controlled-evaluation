"""Round2 review-gated public entry. Default plan only. Never resume/overwrite."""
from pathlib import Path
import argparse,csv,hashlib,json,math,shutil,subprocess,sys,time,traceback
import numpy as np
import torch,psutil
import engine
from cache import get_or_create
HERE=Path(__file__).resolve().parent;V=HERE.parent;ROOT=engine.ROOT;PROJECT=ROOT.parent
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def write(p,d):Path(p).write_text(json.dumps(d,indent=2,allow_nan=False),encoding='utf-8')
def verify():
    assert not sys.flags.optimize
    cfg=read(V/'protocol.json');catalog=read(V/'data_catalog.json')
    assert cfg['train_subjects']==list(range(1,11)) and cfg['validation_subjects']==[11,12] and cfg['seeds']==[42,43,44] and cfg['methods']==list('ABCGD')
    assert cfg['selection_metric']=='validation_macro_logcosh'
    for row in cfg['model_source_bindings']:assert sha(row['path'])==row['sha256']
    assert [r['subject'] for r in catalog['subjects']]==list(range(1,13))
    assert sum(r['windows'] for r in catalog['subjects'] if r['subject']<=10)==cfg['train_windows']
    assert cfg['updates_per_full_training_epoch']==math.ceil(cfg['train_windows']/128)
    for row in read(V/'EXECUTION_FREEZE.json')['files']:
        p=(V/row['path']).resolve();assert p.is_relative_to(V.resolve()) and sha(p)==row['sha256']
    for row in catalog['subjects']:
        assert row['subject'] in range(1,13)
        for f in row['files']:assert sha(f['path'])==f['sha256']
    return cfg,catalog
def receipt(p):
    p=Path(p).resolve();assert p.is_relative_to((PROJECT/'review').resolve())
    assert sha(V/'shutdown_evidence_binding.json')==read(p)['shutdown_evidence_binding_sha256']
    for item in read(V/'shutdown_evidence_binding.json')['files']:assert sha(item['path'])==item['sha256']
    r=read(p);assert r['status']=='passed' and r['execution_freeze_sha256']==sha(V/'EXECUTION_FREEZE.json')
    report=(PROJECT/r['review_report']).resolve();assert report.is_relative_to((PROJECT/'review').resolve()) and sha(report)==r['review_report_sha256']
    return r
class Lock:
    def __enter__(self):
        import msvcrt
        self.f=(ROOT/'round2_v1_4.lock').open('a+b');self.f.write(b'0');self.f.flush();self.f.seek(0);msvcrt.locking(self.f.fileno(),msvcrt.LK_NBLCK,1);return self
    def __exit__(self,*a):self.f.close()
def resource(cfg):
    assert psutil.virtual_memory().available>=cfg['operations']['minimum_available_ram_bytes']
    assert shutil.disk_usage(ROOT).free>=cfg['operations']['minimum_free_disk_bytes']
def prepare(bundle,cfg,catalog):
    resource(cfg);began=time.monotonic();parts={};bindings=[]
    for row in catalog['subjects']:
        resource(cfg);assert time.monotonic()-began<cfg['operations']['prepare_timeout_seconds']
        subject=row['subject']
        with np.load(ROOT/f'data/processed/S{subject}.npz',allow_pickle=False) as z:
            x=z['x'];y=z['y'];activity=z['activity'];start=z['start_sample_32hz']
        p=get_or_create(bundle/'frontend_cache',x,cfg['proposal_sha256'])
        with np.load(p,allow_pickle=False) as z:d=z['filtered_ppg'];f=z['features']
        parts[subject]={'x':x,'y':y,'activity':activity,'start':start,'subject':np.full(len(y),subject,np.int16),'d':d,'f':f}
        engine.validate_data(parts[subject]);assert len(y)==row['windows'] and np.array_equal(start,np.arange(len(y))*64)
        bindings.append({'subject':subject,'frontend_sha256':sha(p),'binding':read(p.with_suffix('.json'))})
    for role,subjects in [('train',cfg['train_subjects']),('val',cfg['validation_subjects'])]:
        arrays={k:np.concatenate([parts[s][k] for s in subjects]) for k in parts[subjects[0]]}
        with (bundle/(role+'.npz')).open('xb') as out:np.savez(out,**arrays)
    write(bundle/'input_binding.json',{'sources':catalog,'frontend':bindings,'prepared_sha256':{r:sha(bundle/(r+'.npz')) for r in ['train','val']},'prepare_seconds':time.monotonic()-began})
def load(bundle,cfg):
    binding=read(bundle/'input_binding.json');result=[]
    for role in ['train','val']:
        p=bundle/(role+'.npz');assert sha(p)==binding['prepared_sha256'][role]
        with np.load(p,allow_pickle=False) as z:data={k:z[k] for k in z.files}
        data['required_subjects']=cfg['validation_subjects'];engine.validate_data(data);result.append(data)
    return result
def paired(metrics):
    assert set(metrics)=={(s,m,k) for s in [11,12] for m in 'ABCGD' for k in [42,43,44]}
    contrasts={}
    for c in 'CGAB':
        values={str(s):[metrics[s,'D',k]-metrics[s,c,k] for k in [42,43,44]] for s in [11,12]}
        per={s:float(np.mean(v)) for s,v in values.items()};byseed=[float(np.mean([values[str(s)][i] for s in [11,12]])) for i in range(3)]
        contrasts['D-'+c]={'six_subject_seed_deltas':values,'subject_seed_mean':per,'equal_subject_mean':float(np.mean(list(per.values()))),'per_seed_subject_macro_delta':byseed,'per_seed_macro_sample_sd':float(np.std(byseed,ddof=1))}
    return contrasts
def aggregate(bundle,cfg,val,reason=None):
    rows=[];metrics={}
    for seed in cfg['seeds']:
        for method in cfg['methods']:
            arm=bundle/f'seed{seed}_{method}';row={'seed':seed,'method':method,'status':'not_run','reason':reason}
            if (arm/'failure.json').exists():row.update(status='failed',reason=read(arm/'failure.json'))
            elif (arm/'summary.json').exists():
                try:
                    assert val is not None
                    with (arm/'predictions.csv').open(newline='') as f:r=list(csv.DictReader(f))
                    assert len(r)==len(val['y'])
                    for i,v in enumerate(r):
                        assert int(v['subject'])==val['subject'][i] and int(v['window_id'])*64==val['start'][i] and int(v['activity'])==val['activity'][i]
                        assert float(v['reference_bpm'])==float(val['y'][i]) and float(v['start_seconds'])==val['start'][i]/32 and float(v['end_seconds'])==val['start'][i]/32+8
                        a=float(v['filtered_weight_alpha']);assert math.isfinite(a) and 0<=a<=1
                        if method in 'ABC':assert a=={'A':0,'B':1,'C':.5}[method]
                        assert v['method']==method and int(v['seed'])==seed and v['status']=='ok' and v['eligible']=='1'
                    scores=engine.score_predictions(np.array([float(v['prediction_bpm']) for v in r]),val['y'],val['subject'],[11,12])
                    stored=read(arm/'summary.json');assert abs(scores['validation_macro_logcosh']-stored['scores']['validation_macro_logcosh'])<1e-9
                    assert sha(arm/'best.pt')==stored['checkpoint_sha256']
                    for s in [11,12]:metrics[s,method,seed]=scores['per_subject'][str(s)]['mae']
                    row.update(status='completed',scores=scores,actual_updates=stored['actual_updates'],best_update=stored['best_update'],coverage=1.,eligible_windows=len(r))
                except Exception as exc:row.update(status='output_integrity_failed',reason=repr(exc))
            rows.append(row)
    complete=all(r['status']=='completed' for r in rows)
    write(bundle/'aggregate.json',{'status':'completed' if complete else 'incomplete','arms':rows,'contrasts':paired(metrics) if complete else {},'subject_count':2,'seed_count':3,'not_six_independent_subjects':True})
    return complete
def verify_snapshot(bundle):
    source=bundle/'source';freeze=source/'EXECUTION_FREEZE.json';saved=read(bundle/'snapshot_manifest.json')
    assert sha(freeze)==saved['execution_freeze_sha256']==sha(V/'EXECUTION_FREEZE.json')
    assert saved['files']==read(freeze)['files']
    for row in saved['files']:
        p=(source/row['path']).resolve();assert p.is_relative_to(source.resolve())
        assert p.stat().st_size==row['bytes'] and sha(p)==row['sha256']

def create_snapshot(bundle):
    shutil.copytree(V,bundle/'source',ignore=shutil.ignore_patterns('__pycache__'))
    freeze=bundle/'source/EXECUTION_FREEZE.json'
    write(bundle/'snapshot_manifest.json',{'execution_freeze_sha256':sha(freeze),'files':read(freeze)['files']})
    verify_snapshot(bundle)

def worker(bundle,method,seed):
    cfg,_=verify();receipt(read(bundle/'provenance.json')['review_receipt_source']);verify_snapshot(bundle);resource(cfg)
    train,val=load(bundle,cfg);initial=torch.load(V/f'initial_backbone_seed{seed}.pt',weights_only=True,map_location='cpu')
    engine.train_arm(bundle/f'seed{seed}_{method}',method,seed,cfg,train,val,initial,scope='ROUND2_DEVELOPMENT_ONLY')
def execute(bundle,review):
    cfg,catalog=verify();receipt(review);resource(cfg);bundle=bundle.resolve();assert bundle.is_relative_to((ROOT/'results').resolve())
    with Lock():
        assert not bundle.exists(),'Output must be a fresh directory'
        ledger_path=ROOT/'logs/round2_v1_4_attempts.json';ledger=read(ledger_path) if ledger_path.exists() else []
        assert not ledger,'Version already attempted; any running/failed/completed record blocks all new output paths; new reviewed version required'
        ledger.append({'bundle':str(bundle),'status':'running'});write(ledger_path,ledger)
        bundle.mkdir(exist_ok=False);val=None
        try:
            create_snapshot(bundle)
            shutil.copy2(review,bundle/'review_receipt.json');write(bundle/'provenance.json',{'review_receipt_source':str(Path(review).resolve()),'execution_freeze_sha256':sha(V/'EXECUTION_FREEZE.json'),'resume_supported':False})
            with (bundle/'prepare.log').open('x') as f:
                subprocess.run([sys.executable,'-B',str(HERE/'runner.py'),'--prepare-worker',str(bundle)],check=True,stdout=f,stderr=subprocess.STDOUT,timeout=cfg['operations']['prepare_timeout_seconds'])
            train,val=load(bundle,cfg);del train
            for seed in cfg['seeds']:
                for method in cfg['run_order_by_seed'][str(seed)]:
                    with (bundle/f'{seed}_{method}.log').open('x') as f:
                        try:subprocess.run([sys.executable,'-B',str(HERE/'runner.py'),'--worker',str(bundle),'--method',method,'--seed',str(seed)],check=True,stdout=f,stderr=subprocess.STDOUT,timeout=cfg['operations']['per_arm_timeout_seconds'])
                        except BaseException as exc:
                            arm=bundle/f'seed{seed}_{method}';arm.mkdir(exist_ok=True)
                            if not (arm/'failure.json').exists():write(arm/'failure.json',{'exception':repr(exc),'resume_supported':False})
                            raise
                    aggregate(bundle,cfg,val)
                    assert not any(r['status']=='output_integrity_failed' for r in read(bundle/'aggregate.json')['arms'])
            complete=aggregate(bundle,cfg,val);ledger[-1]['status']='completed' if complete else 'failed';write(ledger_path,ledger);return complete
        except BaseException as exc:
            write(bundle/'failure.json',{'exception':repr(exc),'traceback':traceback.format_exc(),'resume_supported':False});aggregate(bundle,cfg,val,repr(exc));ledger[-1]['status']='failed';write(ledger_path,ledger);return False
def main():
    p=argparse.ArgumentParser();p.add_argument('--execute');p.add_argument('--review-receipt');p.add_argument('--worker');p.add_argument('--prepare-worker');p.add_argument('--method',choices=list('ABCGD'));p.add_argument('--seed',type=int,choices=[42,43,44]);a=p.parse_args()
    if a.prepare_worker:
        cfg,catalog=verify();bundle=Path(a.prepare_worker);receipt(read(bundle/'provenance.json')['review_receipt_source']);verify_snapshot(bundle);prepare(bundle,cfg,catalog);return
    if a.worker:worker(Path(a.worker),a.method,a.seed);return
    cfg,_=verify()
    if not a.execute:print(json.dumps({'mode':'PLAN_ONLY_NO_REAL_FRONTEND_OR_TRAINING','protocol':cfg}));return
    assert a.review_receipt
    if not execute(Path(a.execute),a.review_receipt):raise SystemExit(1)
if __name__=='__main__':main()
