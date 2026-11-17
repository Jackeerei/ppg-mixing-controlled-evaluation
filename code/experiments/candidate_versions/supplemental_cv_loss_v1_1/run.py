"""One armed, resumable local study. Never launches new work merely on login."""
from pathlib import Path
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import argparse,copy,datetime,importlib.util,json,shutil,subprocess,sys,threading,time,traceback,uuid
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PROJECT=ROOT.parent;OLD=HERE.parent/'supplemental_cv_loss_v1'
CONTROL=ROOT/'logs/supplemental_cv_loss_v1_1_control.json'
if __name__=='__main__' and '--resume-armed' in sys.argv:
    if not CONTROL.exists():raise SystemExit(0)
    _control=json.loads(CONTROL.read_text(encoding='utf-8'))
    if not _control.get('armed') or _control.get('version')!='supplemental_cv_loss_v1_1':raise SystemExit(0)
    _bundle=Path(_control['bundle']);assert _bundle==ROOT/'results'/_control['run_id']
    sys.stdout=(_bundle/'scheduler.stdout.log').open('a',encoding='utf-8',buffering=1)
    sys.stderr=(_bundle/'scheduler.stderr.log').open('a',encoding='utf-8',buffering=1)
sys.path.insert(0,str(HERE))
from checkpoints import atomic_json,sha
from resumable import Trainer,core,clone_state,fingerprint
spec=importlib.util.spec_from_file_location('v1_lifecycle',OLD/'run.py');v1=importlib.util.module_from_spec(spec);spec.loader.exec_module(v1)
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def verify():
    assert not sys.flags.optimize
    freeze=read(HERE/'FREEZE.json')
    for row in freeze['files']:assert sha(PROJECT/row['path'])==row['sha256']
    cfg=read(HERE/'config.json');assert len(cfg['runs'])==90 and cfg['folds']==read(OLD/'config.json')['folds'];return cfg
def runtime():
    import torch,psutil
    assert torch.cuda.is_available() and psutil.virtual_memory().available>=2.5*1024**3 and shutil.disk_usage(ROOT).free>=10*1024**3
    torch.set_num_threads(2);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True;torch.use_deterministic_algorithms(True)
    return torch
class ConcurrentStudyError(RuntimeError):pass
class StudyLock:
    def __init__(self,path=None):self.path=path
    def __enter__(self):
        import msvcrt
        self.f=(self.path or ROOT/'supplemental_cv_loss_v1_1.lock').open('a+b');self.f.write(b'0');self.f.flush();self.f.seek(0)
        try:msvcrt.locking(self.f.fileno(),msvcrt.LK_NBLCK,1)
        except OSError as exc:self.f.close();raise ConcurrentStudyError('Another process owns this study/arm lock') from exc
        return self
    def __exit__(self,*args):self.f.close()
class Clock:
    """Durable active wall time, offline excluded; abrupt loss uncertainty <=1s tick."""
    def __init__(self,path,run_id,allow_initialize=False):
        assert path.exists() or allow_initialize,'Missing cumulative active accounting; refuse budget reset'
        self.path=path;self.run_id=run_id;self.base=read(path) if path.exists() else {'run_id':run_id,'stage_active_seconds':0.,'arm_active_seconds':{}}
        assert self.base['run_id']==run_id
        assert self.base['stage_active_seconds']>=0 and all(v>=0 for v in self.base['arm_active_seconds'].values())
        if not path.exists():atomic_json(path,self.base)
        self.started=time.monotonic();self.actor=None;self.actor_started=None;self.guard=threading.Lock();self.stop=threading.Event();self.thread=threading.Thread(target=self.loop,daemon=True);self.thread.start()
    def snapshot(self):
        with self.guard:
            now=time.monotonic();v=copy.deepcopy(self.base);v['stage_active_seconds']+=now-self.started
            if self.actor:v['arm_active_seconds'][self.actor]=v['arm_active_seconds'].get(self.actor,0.)+now-self.actor_started
            return v
    def loop(self):
        while not self.stop.wait(1):atomic_json(self.path,self.snapshot())
    def set_actor(self,actor):
        with self.guard:
            now=time.monotonic()
            if self.actor:self.base['arm_active_seconds'][self.actor]=self.base['arm_active_seconds'].get(self.actor,0.)+now-self.actor_started
            self.actor=actor;self.actor_started=now
        atomic_json(self.path,self.snapshot())
    def close(self):self.stop.set();self.thread.join();atomic_json(self.path,self.snapshot())
def arm(review):
    cfg=verify();r=read(review);assert r['status']=='passed' and r['execution_freeze_sha256']==sha(HERE/'FREEZE.json')
    assert r['approved_run_count']==90,'Pending scope decision must be resolved in the receipt'
    with StudyLock():
        assert not CONTROL.exists(),'Existing study control must not be reset'
        run_id='supplemental_cv_loss_v1_1_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        out=ROOT/'results'/run_id;out.mkdir(exist_ok=False);shutil.copytree(HERE,out/'source',ignore=shutil.ignore_patterns('__pycache__'));shutil.copy2(review,out/'review_receipt.json')
        assert r['study_active_budget_seconds']==345600
        control={'armed':True,'run_id':run_id,'bundle':str(out),'version':'supplemental_cv_loss_v1_1','freeze_sha256':sha(HERE/'FREEZE.json'),'review':str(review),'review_sha256':sha(review),'status':'armed_not_started','study_budget_seconds':345600,'per_arm_budget_seconds':7200,'environment':None,'automatic_new_run':False}
        atomic_json(out/'state.json',{'run_id':run_id,'status':'armed','training_completed':[],'outer_completed':[]});atomic_json(CONTROL,control);print(json.dumps(control))
def prepare(out,cfg):
    import numpy as np
    if (out/'prepared_bindings.json').exists():
        for p,h in read(out/'prepared_bindings.json').items():assert sha(out/p)==h
        return
    parts={};bindings=read(out/'prepare_progress.json') if (out/'prepare_progress.json').exists() else {}
    for source in cfg['prepared_sources']:
        p=Path(source['path']);assert sha(p)==source['sha256']
        with np.load(p,allow_pickle=False) as z:
            assert set(map(int,z['subject']))<=set(range(1,13))
            for s in set(map(int,z['subject'])):parts[s]={k:z[k][z['subject']==s] for k in ['x','d','f','y','subject','activity','start']}
    for fold in cfg['folds']:
        d=out/f"fold{fold['fold']}";d.mkdir(exist_ok=True)
        for role in ['train','inner','outer']:
            p=d/(role+'.npz');key=str(p.relative_to(out))
            if key in bindings:assert sha(p)==bindings[key];continue
            arrays={k:np.concatenate([parts[s][k] for s in fold[role]]) for k in parts[1]};core.validate_data(arrays)
            temp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
            with temp.open('xb') as f:np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
            os.replace(temp,p);bindings[key]=sha(p);atomic_json(out/'prepare_progress.json',bindings)
    atomic_json(out/'prepared_bindings.json',bindings)
def load(out,fold,role):
    import numpy as np
    p=out/f'fold{fold}'/(role+'.npz');assert sha(p)==read(out/'prepared_bindings.json')[str(p.relative_to(out))]
    with np.load(p,allow_pickle=False) as z:data={k:z[k] for k in z.files}
    core.validate_data(data);data['required_subjects']=list(map(int,np.unique(data['subject'])));return data
def identity(control,r):
    return {'run_id':r['id'],'study_run_id':control['run_id'],'fold':r['fold'],'objective':r['objective'],'seed':r['seed'],'method':r['method'],'freeze_sha256':control['freeze_sha256'],'initial_sha256':sha(HERE.parent/f"round2_v1_4/initial_backbone_seed{r['seed']}.pt")}
def committed(arm,name,expected):
    p=arm/(name+'_COMMIT.json')
    if not p.exists():return False
    c=read(p);assert c['identity']==expected
    for row in c['files']:assert sha(arm/row['name'])==row['sha256']
    return True
def save_torch(torch,p,state):
    tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('xb') as f:torch.save(state,f);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
def worker_body(out,index,outer=False):
    cfg=verify();control=read(CONTROL);assert control['bundle']==str(out) and control['armed'];torch=runtime();r=cfg['runs'][index];arm_path=out/r['id'];arm_path.mkdir(exist_ok=True);ident=identity(control,r)
    assert control['environment']==fingerprint('cuda:0')
    if committed(arm_path,'outer' if outer else 'train',ident):return
    if r['method']=='D_without_c':
        sys.path.insert(0,str(HERE.parent/'c_ablation_cuda_v1'));from gate import GateAblation
        factory=GateAblation
    else:factory=core.Variant
    if not outer:
        train=load(out,r['fold'],'train');inner=load(out,r['fold'],'inner')
        import random,numpy as np
        random.seed(r['seed']);np.random.seed(r['seed']);torch.manual_seed(r['seed']);torch.cuda.manual_seed_all(r['seed'])
        initial=torch.load(HERE.parent/f"round2_v1_4/initial_backbone_seed{r['seed']}.pt",weights_only=True,map_location='cpu')
        backbone=core.build_model();backbone.load_state_dict(initial,strict=True);model=factory(backbone,r['method']).to('cuda:0');torch.manual_seed(r['seed'])
        options={**cfg['training'],'objective':r['objective'],'max_epochs':200,'patience':20,'active_budget_seconds':7200}
        opt,_=core.optimizer_for(model,options);trainer=Trainer(arm_path,model,opt,train,inner,options,ident,'cuda:0');trainer.advance();s=trainer.state
        save_torch(torch,arm_path/'best.pt',s['best_model']);save_torch(torch,arm_path/'last.pt',s['model'])
        atomic_json(arm_path/'history.json',s['history']);atomic_json(arm_path/'summary.json',{'status':'completed','identity':ident,'actual_updates':s['updates'],'best_update':s['best_update'],'best_inner_mae':s['best_inner_mae'],'stale':s['stale'],'epochs':s['epoch'],'stop_reason':s['stop_reason'],'checkpoint_sha256':sha(arm_path/'best.pt'),'cumulative_active_seconds':trainer.active(),'resume_info':trainer.resume_info})
        names=['best.pt','last.pt','history.json','summary.json']
    else:
        assert len(read(out/'BEST_LOCK.json')['models'])==90
        assert committed(arm_path,'train',ident) and sha(arm_path/'best.pt')==read(out/'BEST_LOCK.json')['models'][r['id']]
        data=load(out,r['fold'],'outer');model=factory(core.build_model(),r['method']).to('cuda:0');model.load_state_dict(torch.load(arm_path/'best.pt',weights_only=True,map_location='cuda:0'))
        scores,pred,alpha=core.evaluate(model,data,'cuda:0',128)
        import csv
        tmp=arm_path/'outer_predictions.csv.partial'
        with tmp.open('w',newline='') as f:
            w=csv.writer(f);w.writerow(['run_id','fold','method','objective','seed','subject','activity','start_sample_32hz','reference_bpm','prediction_bpm','alpha','eligible','status'])
            for j in range(len(pred)):w.writerow([r['id'],r['fold'],r['method'],r['objective'],r['seed'],int(data['subject'][j]),int(data['activity'][j]),int(data['start'][j]),float(data['y'][j]),float(pred[j]),float(alpha[j]),1,'ok'])
            f.flush();os.fsync(f.fileno())
        os.replace(tmp,arm_path/'outer_predictions.csv')
        atomic_json(arm_path/'outer_summary.json',{'identity':r,'scores':scores,'candidate_windows':len(pred),'eligible_windows':len(pred),'successful_predictions':len(pred),'coverage':1.,'best_sha256':sha(arm_path/'best.pt'),'scope':'POSTHOC_DEVELOPMENT_ROBUSTNESS'})
        names=['outer_predictions.csv','outer_summary.json']
    atomic_json(arm_path/(('outer' if outer else 'train')+'_COMMIT.json'),{'identity':ident,'files':[{'name':n,'sha256':sha(arm_path/n)} for n in names]})
def worker(out,index,outer=False):
    cfg=verify();arm_path=out/cfg['runs'][index]['id'];arm_path.mkdir(exist_ok=True)
    with StudyLock(arm_path/'worker.lock'):worker_body(out,index,outer)
def resume():
    if not CONTROL.exists():return
    with StudyLock():
        control=read(CONTROL)
        if not control['armed'] or control['version']!='supplemental_cv_loss_v1_1':return
        out=Path(control['bundle']);assert out==ROOT/'results'/control['run_id'];cfg=verify();assert sha(HERE/'FREEZE.json')==control['freeze_sha256']
        state=read(out/'state.json')
        first_start=(state['status']=='armed' and control['environment'] is None and control['status']=='armed_not_started' and not state['training_completed'] and not state['outer_completed'] and not (out/'prepare_progress.json').exists() and not (out/'prepared_bindings.json').exists() and not any((out/r['id']).exists() for r in cfg['runs']))
        clock=Clock(out/'active_accounting.json',control['run_id'],allow_initialize=first_start)
        try:
            runtime();env=fingerprint('cuda:0')
            if control['environment'] is None:control['environment']=env;atomic_json(CONTROL,control)
            assert control['environment']==env,'Environment/device identity mismatch; continuation blocked'
            state.update(status='running',runner_pid=os.getpid(),last_resume_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());atomic_json(out/'state.json',state)
            def child(args,label,cap):
                account=clock.snapshot();remaining=control['study_budget_seconds']-account['stage_active_seconds'];arm_remaining=cap-account['arm_active_seconds'].get(label,0.)
                assert min(remaining,arm_remaining)>0,'Cumulative active budget exhausted'
                clock.set_actor(label)
                try:
                    log=out/(label+'.'+uuid.uuid4().hex+'.log')
                    with log.open('x') as f:
                        process=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-B','-X','utf8',str(HERE/'run.py')]+args,stdout=f,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                        atomic_json(out/'worker_current.json',{'label':label,'pid':process.pid,'runner_pid':os.getpid(),'log':str(log),'state':'running'})
                        try:code=process.wait(timeout=min(remaining,arm_remaining))
                        except subprocess.TimeoutExpired:process.kill();process.wait();raise
                        atomic_json(out/'worker_current.json',{'label':label,'pid':process.pid,'exit_code':code,'state':'exited'})
                        if code:raise subprocess.CalledProcessError(code,process.args)
                finally:clock.set_actor(None)
            if not (out/'prepared_bindings.json').exists():child(['--prepare',str(out)],'prepare',1800)
            for i,r in enumerate(cfg['runs']):
                if not committed(out/r['id'],'train',identity(control,r)):child(['--worker',str(out),'--index',str(i)],r['id'],7200)
                if r['id'] not in state['training_completed']:state['training_completed'].append(r['id'])
                atomic_json(out/'state.json',state)
            locks={r['id']:sha(out/r['id']/'best.pt') for r in cfg['runs']}
            if (out/'BEST_LOCK.json').exists():assert read(out/'BEST_LOCK.json')['models']==locks
            else:atomic_json(out/'BEST_LOCK.json',{'all_training_completed':True,'models':locks})
            for i,r in enumerate(cfg['runs']):
                if not committed(out/r['id'],'outer',identity(control,r)):child(['--worker',str(out),'--index',str(i),'--outer'],'outer_'+r['id'],7200)
                if r['id'] not in state['outer_completed']:state['outer_completed'].append(r['id'])
                atomic_json(out/'state.json',state)
            assert clock.snapshot()['stage_active_seconds']<control['study_budget_seconds'],'Cumulative stage active budget exhausted before collect'
            original_put=v1.put
            try:
                v1.put=lambda p,v,mode='x':atomic_json(Path(p),v)
                v1.collect(out,cfg) # Reconstruct only from committed outer outputs, never rerun predictions.
            finally:v1.put=original_put
            assert clock.snapshot()['stage_active_seconds']<control['study_budget_seconds'],'Cumulative stage active budget exhausted after collect'
            state['status']='completed';control.update(status='completed')
        except BaseException as exc:
            # Reboot/OS termination bypasses this handler and leaves armed run resumable.
            # Explicit numerical/resource/identity/timeout failures block blind continuation.
            state.update(status='blocked_failure',exception=repr(exc),traceback=traceback.format_exc());control.update(status='blocked_failure')
            atomic_json(out/'failure.json',state)
        finally:
            clock.close();state['active_accounting']=read(out/'active_accounting.json');atomic_json(out/'state.json',state)
        if control['status'] in ['completed','blocked_failure']:
            state.update(run_id=control['run_id'],ledger_path=str(CONTROL),bundle=str(out),exit_code=0 if state['status']=='completed' else 1,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
            atomic_json(out/'terminal.json',state)
            atomic_json(out/'MANIFEST.json',{'files':[{'path':str(p.relative_to(out)),'sha256':sha(p)} for p in sorted(out.rglob('*')) if p.is_file() and p!=out/'MANIFEST.json']})
            control['armed']=False;atomic_json(CONTROL,control)
            # Callback cannot change a finalized training ledger; all exceptions isolated.
            try:
                hook_spec=importlib.util.spec_from_file_location('notify_hook',ROOT/'scripts/launch_supplemental_background_v1.py');hook=importlib.util.module_from_spec(hook_spec);hook_spec.loader.exec_module(hook);hook.queue_once(out,state)
            except BaseException as exc:atomic_json(out/'callback_pending.json',{'exception':repr(exc),'training_status_unchanged':state['status']})
            try:
                from completion import notify_once
                notify_once(out,state['status'])
            except BaseException as exc:atomic_json(out/'notification_pending.json',{'exception':repr(exc),'training_status_unchanged':state['status']})
def resume_safe():
    try:resume()
    except ConcurrentStudyError:return # Already running, never invalidate or duplicate its ledger.
    except BaseException as exc:
        # Includes source/data/environment admission errors occurring before the stage clock.
        with StudyLock():
            control=read(CONTROL)
            if not control['armed']:return
            out=Path(control['bundle']);assert out==ROOT/'results'/control['run_id']
            failure={'status':'blocked_entry_failure','exception':repr(exc),'traceback':traceback.format_exc(),'run_id':control['run_id'],'bundle':str(out),'ledger_path':str(CONTROL),'exit_code':1,'no_blind_restart':True}
            atomic_json(out/'entry_failure.json',failure);atomic_json(out/'terminal.json',failure);control.update(armed=False,status='blocked_entry_failure');atomic_json(CONTROL,control)
        try:
            spec=importlib.util.spec_from_file_location('entry_failure_hook',ROOT/'scripts/launch_supplemental_background_v1.py');hook=importlib.util.module_from_spec(spec);spec.loader.exec_module(hook);hook.queue_once(out,failure)
        except BaseException as callback_error:atomic_json(out/'callback_pending.json',{'exception':repr(callback_error),'training_status_unchanged':failure['status']})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arm');p.add_argument('--resume-armed',action='store_true');p.add_argument('--prepare');p.add_argument('--worker');p.add_argument('--index',type=int);p.add_argument('--outer',action='store_true');a=p.parse_args()
    if a.arm:arm(Path(a.arm).resolve())
    elif a.prepare:
        cfg=verify();control=read(CONTROL);assert control['armed'] and Path(a.prepare).resolve()==Path(control['bundle']);runtime();prepare(Path(a.prepare).resolve(),cfg)
    elif a.worker:worker(Path(a.worker).resolve(),a.index,a.outer)
    elif a.resume_armed:resume_safe()
    else:print(json.dumps({'candidate_only':True,'training_started':False,'armed':CONTROL.exists() and read(CONTROL)['armed']}))
