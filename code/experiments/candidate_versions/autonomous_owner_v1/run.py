"""Same-study recovery owner; bounded resource admission and one authorized retry."""
from pathlib import Path
import argparse,datetime,hashlib,json,os,subprocess,sys,threading,time,traceback,uuid
import psutil
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PROJECT=ROOT.parent
SOURCE=ROOT/'candidate_versions/supplemental_cv_loss_v1_1';BUNDLE=ROOT/'results/supplemental_cv_loss_v1_1_20261108T033819Z'
OWNER=BUNDLE/'parallel_recovery_v1';CONTROL=OWNER/'control.json';SOURCE_CONTROL=ROOT/'logs/supplemental_cv_loss_v1_1_control.json'
from io_guard import atomic
from active_clock import seconds,KIND
from resources import snapshot,admissible,parallel_ready
time.monotonic=seconds
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
class Lock:
    def __init__(self,p):self.p=p
    def __enter__(self):
        import msvcrt
        self.f=self.p.open('a+b');self.f.write(b'0');self.f.flush();self.f.seek(0)
        try:msvcrt.locking(self.f.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:self.f.close();raise
        return self
    def __exit__(self,*a):self.f.close()
def verify():
    for row in read(HERE/'FREEZE.json')['files']:assert sha(PROJECT/row['path'])==row['sha256']
    assert sha(SOURCE/'FREEZE.json')=='e655f4d594264a46a76befc06ecfe4a3fac1e745f62b8ec9d1854674d7d1fb8d'
    return read(SOURCE/'config.json')
def commit(arm,r,outer=False):
    p=arm/('outer_COMMIT.json' if outer else 'train_COMMIT.json')
    if not p.exists():return False
    c=read(p);assert c['identity']['run_id']==r['id'] and c['identity']['study_run_id']==BUNDLE.name
    assert c['identity']['freeze_sha256']=='e655f4d594264a46a76befc06ecfe4a3fac1e745f62b8ec9d1854674d7d1fb8d'
    for row in c['files']:assert sha(arm/row['name'])==row['sha256']
    return True
class Accounting:
    def __init__(self,state):
        self.state=state;self.active={};self.last=time.monotonic();self.guard=threading.RLock();self.stop=threading.Event();self.error=None
    def tick(self):
        with self.guard:
            now=time.monotonic();delta=now-self.last;self.last=now;n=len(self.active)
            self.state['global_active_seconds']+=delta*max(n,1)
            for name in self.active:self.state['per_arm_active_seconds'][name]=self.state['per_arm_active_seconds'].get(name,0.)+delta
            self.state['shared_only_seconds']+=delta if n==0 else 0
    def save(self):
        with self.guard:
            self.tick();self.state['workers']={name:dict(v) for name,v in self.active.items()};self.state['heartbeat_utc']=datetime.datetime.now(datetime.timezone.utc).isoformat();atomic(CONTROL,self.state)
    def loop(self):
        while not self.stop.wait(1):
            try:self.save()
            except BaseException as exc:self.error=exc;return
def stop_owned_process(p):
    # Venv launcher and actual runtime are separate Windows processes.
    try:
        descendants=psutil.Process(p.pid).children(recursive=True)
    except psutil.NoSuchProcess:descendants=[]
    for child in reversed(descendants):
        try:child.kill()
        except psutil.NoSuchProcess:pass
    if p.poll() is None:p.kill()
    p.wait()

def run():
    if not CONTROL.exists():return
    with Lock(ROOT/'supplemental_cv_loss_v1_1.lock'):
        cfg=verify();state=read(CONTROL)
        if not state['armed']:return
        assert state['owner_freeze_sha256']==sha(HERE/'FREEZE.json') and state['study_run_id']==BUNDLE.name
        anchor=state.pop('handoff_awake_anchor',None)
        boot=state.pop('handoff_boot_time',None)
        if anchor is not None:
            gap=max(0.,seconds()-anchor) if boot==psutil.boot_time() else 0.
            state['handoff_dispatch_gap_seconds']=gap
            state['global_active_seconds']+=gap;state['shared_only_seconds']+=gap
        state.update(status='running',owner_pid=os.getpid(),resource_admission=None)
        clock=Accounting(state);processes={};logs={};resource_wait_started=None;next_resource_check=0.
        # A process restart continues the already-authorized attempt, not a second retry.
        continuing={n for n,s in state['arm_status'].items() if s=='running' and state['retry_attempts_by_arm'].get(n)==1}
        # After reboot, committed arms skip; uncommitted active identities continue same e655 checkpoint.
        for r in cfg['runs']:
            name=r['id']
            if state['arm_status'][name] in ['running','queued']:
                if commit(BUNDLE/name,r):state['arm_status'][name]='completed'
                else:state['arm_status'][name]='queued'
            if state.get('outer_status',{}).get(name)=='running':state['outer_status'][name]='completed' if commit(BUNDLE/name,r,True) else 'queued'
        clock.save();thread=threading.Thread(target=clock.loop,daemon=True);thread.start()
        try:
            while True:
                if clock.error:raise clock.error
                outer=state['phase']=='outer';statusmap=state['outer_status'] if outer else state['arm_status']
                with clock.guard:clock.tick();global_left=state['global_limit_seconds']-state['global_active_seconds']
                if 0<global_left<=15 and not (OWNER/'resource_pause.request').exists():atomic(OWNER/'resource_pause.request',{'reason':'global_budget_near_limit_checkpoint'})
                if global_left<=0:
                    for name,p in list(processes.items()):
                        stop_owned_process(p);statusmap[name]='global_budget_exhausted'
                    break
                for name,p in list(processes.items()):
                    code=p.poll()
                    if code is None:continue
                    r=next(r for r in cfg['runs'] if r['id']==name)
                    with clock.guard:
                        clock.tick();clock.active.pop(('outer_' if outer else '')+name,None)
                        if commit(BUNDLE/name,r,outer):statusmap[name]='completed'
                        else:
                            statusmap[name]='failed'
                            if (OWNER/'resource_pause.request').exists():
                                statusmap[name]='running'
                                state['dispatch_paused_reason']='resource_pressure_checkpointed'
                            # Isolated failures remain failed; dispatch other queued identities.
                        atomic(OWNER/(('outer_' if outer else '')+name+'.terminal.json'),{'id':name,'phase':state['phase'],'exit_code':code,'status':statusmap[name],'cumulative_active_seconds':state['per_arm_active_seconds'][('outer_' if outer else '')+name]})
                        processes.pop(name);logs.pop(name).close();clock.save()
                while len(processes)<2 and not state.get('dispatch_paused_reason'):
                    r=next((r for r in cfg['runs'] if statusmap[r['id']]=='queued'),None)
                    if r is None:break
                    if any(p.poll() is not None for p in processes.values()):break
                    now=time.monotonic()
                    if now<next_resource_check:break
                    observation=snapshot(ROOT)
                    if any(p.poll() is not None for p in processes.values()):break
                    with clock.guard:state['resource_admission']=observation
                    with clock.guard:state['max_workers']=2 if parallel_ready(observation) else 1
                    if len(processes)>=state['max_workers']:
                        next_resource_check=now+15;break # Keep current worker(s); native bounded re-admission later.
                    if not admissible(observation):
                        if resource_wait_started is None:resource_wait_started=now
                        if now-resource_wait_started>=300:
                            state['dispatch_paused_reason']='resources_unavailable_after_bounded_300_seconds'
                        next_resource_check=now+10;clock.save();break
                    resource_wait_started=None;next_resource_check=0.
                    name=r['id']
                    if name in state['retry_authorization']['eligible_ids'] and not outer:
                        used=state['retry_attempts_by_arm'].get(name,0)
                        assert used==0 or name in continuing,'One authorized retry already consumed'
                        if used==0:
                            with clock.guard:state['retry_attempts_by_arm'][name]=1
                        continuing.discard(name)
                    log=(OWNER/(name+'.'+uuid.uuid4().hex+'.log')).open('x');logs[name]=log
                    actor=('outer_' if outer else '')+name
                    with clock.guard:
                        clock.tick();statusmap[name]='running';clock.active[actor]={'pid':None,'id':name,'phase':state['phase'],'index':cfg['runs'].index(r),'start_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()};clock.save()
                    p=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-B','-X','utf8',str(HERE/'worker.py'),'--worker',str(BUNDLE),'--index',str(cfg['runs'].index(r)),'--owner',str(CONTROL)]+(['--outer'] if outer else []),stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                    processes[name]=p;clock.active[actor]['pid']=p.pid;clock.save()
                if not processes:
                    if resource_wait_started is not None and not state.get('dispatch_paused_reason'):
                        time.sleep(.5);continue
                    if not outer and all(v=='completed' for v in state['arm_status'].values()):
                        locks={r['id']:sha(BUNDLE/r['id']/'best.pt') for r in cfg['runs']}
                        if (BUNDLE/'BEST_LOCK.json').exists():assert read(BUNDLE/'BEST_LOCK.json')['models']==locks
                        else:atomic(BUNDLE/'BEST_LOCK.json',{'all_training_completed':True,'models':locks,'owner_freeze':state['owner_freeze_sha256']})
                        state['phase']='outer'
                        for r in cfg['runs']:
                            if commit(BUNDLE/r['id'],r,True):state['outer_status'][r['id']]='completed'
                            elif state['outer_status'][r['id']]=='running':state['outer_status'][r['id']]='queued'
                        clock.save();continue
                    break
                time.sleep(.5)
            statuses=list(state['arm_status'].values());state['outer_started']=state['phase']=='outer'
            if statuses.count('completed')==90 and all(v=='completed' for v in state['outer_status'].values()):
                state['phase']='collect';clock.active['collect']={'pid':None,'phase':'collect'};clock.save();remaining=state['global_limit_seconds']-state['global_active_seconds'];assert remaining>0
                with (OWNER/'collect.log').open('a') as log:
                    p=subprocess.Popen([str(ROOT/'.venv/Scripts/python.exe'),'-B','-X','utf8',str(HERE/'worker.py'),'--worker',str(BUNDLE),'--index','-1','--owner',str(CONTROL),'--collect'],stdout=log,stderr=subprocess.STDOUT,creationflags=subprocess.CREATE_NO_WINDOW)
                    try:code=p.wait(timeout=remaining)
                    except subprocess.TimeoutExpired:stop_owned_process(p);raise
                    assert code==0
                clock.tick();clock.active.pop('collect',None);state['status']='completed'
            else:state['status']='paused_with_preserved_queue' if state.get('dispatch_paused_reason') else 'finished_incomplete_with_preserved_failures'
        except BaseException as exc:
            state.update(status='owner_blocked_failure',exception=repr(exc),traceback=traceback.format_exc())
            for p in processes.values():
                if p.poll() is None:stop_owned_process(p)
        finally:
            clock.stop.set();thread.join();clock.tick();state['armed']=False;clock.save()
            for f in logs.values():f.close()
            source_control=read(SOURCE_CONTROL);source_control.update(armed=False,status='parallel_owner_finished',external_owner=str(CONTROL));atomic(SOURCE_CONTROL,source_control)
            atomic(OWNER/'terminal.json',state)
            atomic(OWNER/'MANIFEST.json',{'files':[{'path':p.name,'sha256':sha(p)} for p in OWNER.iterdir() if p.is_file() and p.name!='MANIFEST.json']})
        # One NEW terminal event, separate from the prior failed daemon-resume claim; never per-arm.
        try:
            import importlib.util
            event=OWNER/state['terminal_event_id'];event.mkdir(exist_ok=True);atomic(event/'terminal.json',state)
            spec=importlib.util.spec_from_file_location('terminal_hook',HERE/'hook.py');hook=importlib.util.module_from_spec(spec);spec.loader.exec_module(hook)
            hook.queue_once(event,{'status':state['status'],'exit_code':0 if state['status']=='completed' else 1,'bundle':str(BUNDLE),'ledger_path':str(CONTROL)})
            sys.path.insert(0,str(SOURCE));from completion import notify_once
            notify_once(event,state['status'])
        except BaseException as exc:atomic(OWNER/'callback_pending.json',{'exception':repr(exc),'training_status_unchanged':state['status']})
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--resume-armed',action='store_true');a=p.parse_args()
    if a.resume_armed:
        if CONTROL.exists() and read(CONTROL)['armed']:
            OWNER.mkdir(exist_ok=True);sys.stdout=(OWNER/'owner.stdout.log').open('a',encoding='utf-8',buffering=1);sys.stderr=(OWNER/'owner.stderr.log').open('a',encoding='utf-8',buffering=1);run()
    else:print(json.dumps({'prepared_only':True,'starts_training':False}))
