"""Post-hoc development CV; default verifies only. No resume or automatic retries."""
from pathlib import Path
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,csv,datetime,hashlib,json,shutil,subprocess,sys,time,traceback
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PROJECT=ROOT.parent
LEDGER=ROOT/'logs/supplemental_cv_loss_v1_attempt.json'
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def put(p,v,mode='x'):
    with Path(p).open(mode,encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False)
def verify():
    assert not sys.flags.optimize
    for r in read(HERE/'FREEZE.json')['files']:assert sha(PROJECT/r['path'])==r['sha256']
    cfg=read(HERE/'config.json');assert len(cfg['runs'])==90 and len({r['id'] for r in cfg['runs']})==90
    outer=[]
    for f in cfg['folds']:
        sets=[set(f[k]) for k in ['train','inner','outer']]
        assert all(not a&b for i,a in enumerate(sets) for b in sets[i+1:]) and set.union(*sets)==set(range(1,13))
        outer+=f['outer']
    assert sorted(outer)==list(range(1,13));return cfg
def runtime():
    import numpy as np
    import torch,psutil
    import engine
    sys.path.insert(0,str(ROOT/'candidate_versions/c_ablation_cuda_v1'))
    from gate import GateAblation
    original=engine.Variant
    engine.Variant=lambda backbone,method:GateAblation(backbone,method) if method=='D_without_c' else original(backbone,method)
    assert torch.cuda.is_available() and torch.version.cuda
    assert psutil.virtual_memory().available>=2.5*1024**3 and shutil.disk_usage(ROOT).free>=10*1024**3
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    return np,torch,engine
def prepare(out,cfg):
    np,torch,engine=runtime();parts={};bindings={}
    for source in cfg['prepared_sources']:
        p=Path(source['path']);assert sha(p)==source['sha256']
        with np.load(p,allow_pickle=False) as z:
            assert set(map(int,z['subject']))<=set(range(1,13))
            for s in set(map(int,z['subject'])):parts[s]={k:z[k][z['subject']==s] for k in ['x','y','d','f','subject','activity','start']}
    assert set(parts)==set(range(1,13))
    for fold in cfg['folds']:
        d=out/f"fold{fold['fold']}";d.mkdir(exist_ok=False)
        for role in ['train','inner','outer']:
            arrays={k:np.concatenate([parts[s][k] for s in fold[role]]) for k in parts[1]}
            engine.validate_data(arrays);p=d/(role+'.npz');np.savez(p,**arrays);bindings[str(p.relative_to(out))]=sha(p)
    put(out/'prepared_bindings.json',bindings)
def load(out,fold,role,engine,np):
    p=out/f'fold{fold}'/(role+'.npz');assert sha(p)==read(out/'prepared_bindings.json')[str(p.relative_to(out))]
    with np.load(p,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    engine.validate_data(data);data['required_subjects']=list(map(int,np.unique(data['subject'])));return data
def worker(out,index,outer):
    cfg=verify();assert read(LEDGER)['status']=='running' and read(LEDGER)['bundle']==str(out)
    np,torch,engine=runtime();r=cfg['runs'][index];f=cfg['folds'][r['fold']-1];arm=out/r['id']
    if not outer:
        assert not (out/'BEST_LOCK.json').exists()
        train=load(out,r['fold'],'train',engine,np);inner=load(out,r['fold'],'inner',engine,np)
        options={**cfg['training'],'objective':r['objective'],'train_subjects':f['train'],'validation_subjects':f['inner'],'updates_per_full_training_epoch':r['batches'],'validation_interval_updates':r['batches'],'max_updates_per_method_seed':200*r['batches']}
        initial=torch.load(ROOT/f"candidate_versions/round2_v1_4/initial_backbone_seed{r['seed']}.pt",map_location='cpu',weights_only=True)
        engine.train_arm(arm,r['method'],r['seed'],options,train,inner,initial,device='cuda',scope='POSTHOC_SUBJECT_CV_INNER_SELECTION')
    else:
        locked=read(out/'BEST_LOCK.json');assert len(locked['models'])==90 and locked['all_training_completed'] is True
        assert sha(arm/'best.pt')==locked['models'][r['id']]
        assert not (arm/'outer_predictions.csv').exists() and not (arm/'outer_summary.json').exists()
        data=load(out,r['fold'],'outer',engine,np)
        model=engine.Variant(engine.build_model(),r['method']).to('cuda');model.load_state_dict(torch.load(arm/'best.pt',weights_only=True,map_location='cuda'),strict=True)
        scores,pred,alpha=engine.evaluate(model,data,'cuda',128)
        with (arm/'outer_predictions.csv').open('x',newline='') as f:
            w=csv.writer(f);w.writerow(['run_id','fold','method','objective','seed','subject','activity','start_sample_32hz','reference_bpm','prediction_bpm','alpha','eligible','status'])
            for j in range(len(pred)):w.writerow([r['id'],r['fold'],r['method'],r['objective'],r['seed'],int(data['subject'][j]),int(data['activity'][j]),int(data['start'][j]),float(data['y'][j]),float(pred[j]),float(alpha[j]),1,'ok'])
        put(arm/'outer_summary.json',{'identity':r,'scores':scores,'candidate_windows':len(pred),'eligible_windows':len(pred),'successful_predictions':len(pred),'coverage':1.,'best_sha256':sha(arm/'best.pt'),'best_lock_sha256':sha(out/'BEST_LOCK.json'),'scope':'POSTHOC_DEVELOPMENT_ROBUSTNESS','test_subjects_S13_S15_used':False})
def child(command,log,timeout):
    # subprocess timeout enforces child termination; sequential workers spawn no children.
    with log.open('x') as f:subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=timeout)
def execute(review):
    import msvcrt
    cfg=verify();review=review.resolve();receipt=read(review)
    assert review.is_relative_to(PROJECT/'review') and receipt['status']=='passed' and receipt['execution_freeze_sha256']==sha(HERE/'FREEZE.json')
    runtime() # Import libraries and check resources BEFORE ledger creation.
    with (ROOT/'supplemental_cv_loss_v1.lock').open('a+b') as lock:
        lock.write(b'0');lock.flush();lock.seek(0);msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        assert not LEDGER.exists(),'Any attempted stage blocks restart/resume'
        out=ROOT/'results'/('supplemental_cv_loss_v1_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));out.mkdir(exist_ok=False)
        started=time.monotonic();state={'status':'running','bundle':str(out),'freeze_sha256':sha(HERE/'FREEZE.json'),'review':str(review),'review_sha256':sha(review),'started_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'training_completed':[],'outer_completed':[]}
        put(LEDGER,state);put(out/'intent.json',state);shutil.copytree(HERE,out/'source',ignore=shutil.ignore_patterns('__pycache__'));shutil.copy2(review,out/'review_receipt.json')
        def budget(cap):
            remaining=72*3600-(time.monotonic()-started);assert remaining>0,'72h stage exhausted';return min(cap,remaining)
        try:
            child([sys.executable,'-B','-X','utf8',str(HERE/'run.py'),'--prepare',str(out)],out/'prepare.log',budget(1800))
            for i,r in enumerate(cfg['runs']):
                child([sys.executable,'-B','-X','utf8',str(HERE/'run.py'),'--worker',str(out),'--index',str(i)],out/(r['id']+'.train.log'),budget(7200))
                assert read(out/r['id']/'summary.json')['status']=='completed'
                state['training_completed'].append(r['id']);put(LEDGER,state,'w');print(json.dumps({'trained':r['id'],'completed':len(state['training_completed']),'outer_scored':0}),flush=True)
            locked={r['id']:sha(out/r['id']/'best.pt') for r in cfg['runs']}
            put(out/'BEST_LOCK.json',{'all_training_completed':True,'models':locked,'locked_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
            for i,r in enumerate(cfg['runs']):
                child([sys.executable,'-B','-X','utf8',str(HERE/'run.py'),'--worker',str(out),'--index',str(i),'--outer'],out/(r['id']+'.outer.log'),budget(7200))
                state['outer_completed'].append(r['id']);put(LEDGER,state,'w')
            collect(out,cfg);state['status']='completed'
        except BaseException as exc:
            state.update(status='failed',exception=repr(exc),traceback=traceback.format_exc());put(out/'failure.json',state);raise
        finally:
            state.update(finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),elapsed_seconds=time.monotonic()-started);put(out/'terminal.json',state);put(LEDGER,state,'w')
            put(out/'MANIFEST.json',{'files':[{'path':str(p.relative_to(out)),'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file()]})
def collect(out,cfg):
    import numpy as np
    cells={};arms=[]
    for r in cfg['runs']:
        v=read(out/r['id']/'outer_summary.json');arms.append(v)
        for s,m in v['scores']['per_subject'].items():cells[r['objective'],r['method'],r['seed'],int(s)]=m['mae']
    summaries={};contrasts={}
    for objective,methods in [('logcosh',['C','G','D','D_without_c'])]+[(o,['G','D']) for o in ['MSE','MAE','Huber5']]:
        for m in methods:
            byseed=[float(np.mean([cells[objective,m,k,s] for s in range(1,13)])) for k in [42,43,44]]
            rr=[a for a in arms if a['identity']['objective']==objective and a['identity']['method']==m]
            pooled=[sum(a['scores']['pooled_mae']*a['eligible_windows'] for a in rr if a['identity']['seed']==k)/sum(a['eligible_windows'] for a in rr if a['identity']['seed']==k) for k in [42,43,44]]
            summaries[objective+'/'+m]={'per_seed_equal12_subject_mae':byseed,'mean':float(np.mean(byseed)),'seed_sample_sd':float(np.std(byseed,ddof=1)),'per_seed_pooled_mae':pooled,'pooled_mae_mean':float(np.mean(pooled))}
        for a,b in ([('D','G'),('D','C'),('D','D_without_c')] if objective=='logcosh' else [('D','G')]):
            v={str(s):[cells[objective,a,k,s]-cells[objective,b,k,s] for k in [42,43,44]] for s in range(1,13)}
            contrasts[objective+'/'+a+'-'+b]={'subject_seed_cells':v,'subject_seed_means':{s:float(np.mean(x)) for s,x in v.items()},'equal12_subject_mean':float(np.mean([np.mean(x) for x in v.values()]))}
    effects={}
    for m in ['G','D']:
        for o in ['MSE','MAE','Huber5']:effects[o+'-logcosh/'+m]={str(s):[cells[o,m,k,s]-cells['logcosh',m,k,s] for k in [42,43,44]] for s in range(1,13)}
    put(out/'aggregate.json',{'scope':'POSTHOC_DEVELOPMENT_ROBUSTNESS_NOT_NEW_FINAL_TEST','status':'completed','arms':arms,'summaries':summaries,'contrasts':contrasts,'objective_effects_subject_seed_cells':effects,'independent_people':12,'no_seed_or_fold_pseudoreplication':True})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--execute',action='store_true');p.add_argument('--review');p.add_argument('--prepare');p.add_argument('--worker');p.add_argument('--index',type=int);p.add_argument('--outer',action='store_true');a=p.parse_args()
    if a.prepare:
        cfg=verify();out=Path(a.prepare).resolve();assert read(LEDGER)['bundle']==str(out) and read(LEDGER)['status']=='running';prepare(out,cfg)
    elif a.worker:worker(Path(a.worker).resolve(),a.index,a.outer)
    elif a.execute:assert a.review;execute(Path(a.review))
    else:cfg=verify();print(json.dumps({'status':'verified_candidate_only','unique_training_runs':len(cfg['runs']),'training_started':False}))
