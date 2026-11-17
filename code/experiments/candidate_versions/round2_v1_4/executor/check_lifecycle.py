"""Actual launch.main isolated fault matrix; no real Popen/prepare/training."""
from pathlib import Path
import contextlib,hashlib,io,json,time
from unittest.mock import patch
import launch,runner,host_observer

class Child:
    pid=910001
    def __init__(self,code=0,fast=False):self.returncode=None;self.calls=0;self.code=code;self.fast=fast;self.terminated=False
    def poll(self):
        self.calls+=1
        if self.returncode is None and (self.fast or self.calls>=3):self.returncode=self.code
        return self.returncode
    def terminate(self):self.terminated=True;self.returncode=-15
    def kill(self):self.terminated=True;self.returncode=-9
    def wait(self,timeout=None):assert timeout is not None;return self.returncode

class Process:
    def __init__(self,pid,created=200):self.pid=pid;self.created=created;self.terminated=False
    def create_time(self):return self.created
    def is_running(self):return not self.terminated
    def children(self,recursive=False):return []
    def terminate(self):self.terminated=True
    def kill(self):self.terminated=True

def main():
    output=runner.ROOT/'reports'/('round2_v1_3_lifecycle_SYNTHETIC_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()));output.mkdir(exist_ok=False)
    original_process=launch.psutil.Process;original_open=Path.open;original_append=launch.append_event
    cases=['normal0','nonzero3','popen_error','fast_exit_identity_query','identity_query_error','identity_persist_error','observe_error','observations_open_error','observations_write_error','terminal_persist_error','ceiling_owned_tree','ceiling_pid_reused']
    report={}
    for name in cases:
        root=output/name;(root/'logs').mkdir(parents=True);(root/'results').mkdir();fake=root/'SYNTHETIC_NOT_AN_APPROVAL.json';fake.write_text('{"status":"SYNTHETIC_NOT_AN_APPROVAL"}')
        child=Child(3 if name=='nonzero3' else 0,name=='fast_exit_identity_query');queried=[0];sentinel=Process(910002,300);sentinel_stopped=[]
        if name.startswith('ceiling'):
            def persistent_poll():return child.returncode
            child.poll=persistent_poll
        cfg={'operations':{'launcher_observation_ceiling_seconds':0 if name.startswith('ceiling') else 100}}
        def process(pid=None):
            if pid!=child.pid:return original_process(pid)
            queried[0]+=1
            if name in ['fast_exit_identity_query','identity_query_error']:raise launch.psutil.NoSuchProcess(pid)
            result=Process(pid,201 if name=='ceiling_pid_reused' and queried[0]>1 else 200)
            if name=='ceiling_owned_tree':result.children=lambda recursive=False:[sentinel]
            if name=='ceiling_pid_reused':result.children=lambda recursive=False:(_ for _ in ()).throw(AssertionError('Reused PID descendants inspected'))
            return result
        def popen(*args,**kwargs):
            if name=='popen_error':raise OSError('INJECTED Popen failure')
            return child
        def append(path,kind,value):
            if name=='identity_persist_error' and kind=='identity':raise OSError('INJECTED identity persistence')
            if name=='terminal_persist_error' and kind=='terminal':raise OSError('INJECTED terminal persistence')
            return original_append(path,kind,value)
        class BrokenWriter:
            def __init__(self,f):self.f=f
            def write(self,*a):raise OSError('INJECTED observation write')
            def close(self):self.f.close()
        def opened(p,*args,**kwargs):
            if str(p).endswith('.observations.jsonl'):
                if name=='observations_open_error':raise OSError('INJECTED observation open')
                if name=='observations_write_error':return BrokenWriter(original_open(p,*args,**kwargs))
            return original_open(p,*args,**kwargs)
        def observation(record):
            if name=='observe_error':raise OSError('INJECTED observe failure')
            return {'classification':'running_identity_verified','scope':'SYNTHETIC'}
        stderr=io.StringIO();stdout=io.StringIO()
        with contextlib.ExitStack() as stack:
            for target,value in [(runner,'ROOT')]:stack.enter_context(patch.object(target,value,root))
            stack.enter_context(patch.object(runner,'receipt',lambda p:None));stack.enter_context(patch.object(runner,'resource',lambda c:None));stack.enter_context(patch.object(runner,'read',lambda p:cfg))
            stack.enter_context(patch.object(launch.subprocess,'Popen',popen));stack.enter_context(patch.object(launch.psutil,'Process',process));stack.enter_context(patch.object(launch.psutil,'process_iter',lambda *a:[]))
            stack.enter_context(patch.object(launch.psutil,'wait_procs',lambda *a,**k:([],[])));stack.enter_context(patch.object(launch,'append_event',append));stack.enter_context(patch.object(launch,'observe',observation));stack.enter_context(patch.object(launch.time,'sleep',lambda n:None));stack.enter_context(patch.object(Path,'open',opened))
            stack.enter_context(contextlib.redirect_stderr(stderr));stack.enter_context(contextlib.redirect_stdout(stdout))
            code=launch.main(['--review-receipt',str(fake),'--availability-note','SYNTHETIC ONLY NO EXECUTION APPROVAL'])
        intents=list((root/'logs').glob('round2_v13_launch_*.json'));intents=[p for p in intents if '_identity_' not in p.name and '_terminal_' not in p.name];assert len(intents)==1
        intent=intents[0];original=json.loads(intent.read_text());h=hashlib.sha256(intent.read_bytes()).hexdigest();assert original['status']=='starting' and 'runner_pid' not in original
        identities=list(intent.parent.glob(intent.stem+'_identity_*.json'));terminals=list(intent.parent.glob(intent.stem+'_terminal_*.json'))
        for p in identities+terminals:assert json.loads(p.read_text())['intent_sha256']==h
        if name=='terminal_persist_error':assert not terminals and 'persistence_failed' in stderr.getvalue()
        else:
            assert len(terminals)==1;terminal=json.loads(terminals[0].read_text())['value']
            if name in ['normal0','nonzero3','fast_exit_identity_query','identity_query_error','identity_persist_error','observe_error','observations_open_error','observations_write_error']:assert terminal['exit_code']==child.code
        if name!='popen_error':assert child.poll() is not None
        if name=='identity_persist_error':assert not identities and 'identity' in stderr.getvalue()
        if name=='ceiling_owned_tree':assert sentinel.terminated and child.terminated
        if name=='ceiling_pid_reused':assert not sentinel.terminated and child.terminated
        starting=host_observer.observe(original);assert starting['classification']=='starting_runner_identity_unknown'
        (root/'stderr.txt').write_text(stderr.getvalue());(root/'stdout.txt').write_text(stdout.getvalue())
        report[name]={'passed':True,'main_return':code,'intent_sha256':h,'identity_events':len(identities),'terminal_events':len(terminals),'owned_child_reaped_or_never_started':name=='popen_error' or child.returncode is not None}
    value={'status':'passed','scope':'ISOLATED_ACTUAL_MAIN_LIFECYCLE_FAULTS_NO_REAL_SUBPROCESSES','execution_freeze_sha256':runner.sha(runner.V/'EXECUTION_FREEZE.json'),'cases':report,'subprocesses_started':0,
           'substitutions':'Review/resource preconditions, Popen child handles, psutil identity, observation faults and sleep only; actual main, intent/sidecar/log I/O, locks, exception/finally flow and safe_stop are production code.'}
    (output/'checks.json').write_text(json.dumps(value,indent=2));print(json.dumps({'output':str(output),**value},indent=2))
if __name__=='__main__':main()
