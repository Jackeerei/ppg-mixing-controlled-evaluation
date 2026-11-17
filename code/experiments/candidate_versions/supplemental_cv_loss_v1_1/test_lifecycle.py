"""Isolated commits/locks/time and mocked callback failures, never real queue/train."""
from pathlib import Path
import datetime,importlib.util,json,subprocess,sys,time
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
import run
from checkpoints import atomic_json,sha
def main():
    stage=ROOT/'reports'/('resume_lifecycle_mock_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ'));stage.mkdir(exist_ok=False)
    arm=stage/'arm';arm.mkdir();(arm/'best.pt').write_bytes(b'SYNTHETIC_COMMITTED_RESULT');ident={'run_id':'MOCK_ONLY'}
    atomic_json(arm/'train_COMMIT.json',{'identity':ident,'files':[{'name':'best.pt','sha256':sha(arm/'best.pt')}]})
    assert run.committed(arm,'train',ident)
    invoked=[]
    if not run.committed(arm,'train',ident):invoked.append('train')
    assert not invoked
    run.ROOT=stage
    lock_code="import sys;from pathlib import Path;sys.path.insert(0,sys.argv[1]);import run;run.ROOT=Path(sys.argv[2]);\ntry:\n with run.StudyLock(): pass\nexcept run.ConcurrentStudyError:\n print('double_start_rejected');sys.exit(0)\nsys.exit(9)"
    with run.StudyLock():
        result=subprocess.run([sys.executable,'-B','-c',lock_code,str(HERE),str(stage)],capture_output=True,text=True,timeout=60)
        assert result.returncode==0 and 'double_start_rejected' in result.stdout,(result.stdout,result.stderr)
    clock=run.Clock(stage/'active.json','MOCK_RUN',allow_initialize=True);time.sleep(.03);clock.close();first=json.loads((stage/'active.json').read_text())['stage_active_seconds'];time.sleep(.03)
    clock=run.Clock(stage/'active.json','MOCK_RUN');time.sleep(.03);clock.close();second=json.loads((stage/'active.json').read_text())['stage_active_seconds'];assert first<second<first+.15
    spec=importlib.util.spec_from_file_location('callback',ROOT/'scripts/launch_supplemental_background_v1.py');hook=importlib.util.module_from_spec(spec);spec.loader.exec_module(hook)
    for code,stderr in [(429,'quota exhausted'),(1,'daemon unavailable'),(0,'warning: daemon resume failed')]:
        case=stage/f'callback_{code}_{len(stderr)}';case.mkdir();calls=[]
        def fake(argv,**kwargs):calls.append(argv);return subprocess.CompletedProcess(argv,code,'',stderr)
        terminal={'status':'completed','exit_code':0,'bundle':'MOCK_RESULT','ledger_path':'MOCK_LEDGER'};atomic_json(case/'terminal.json',terminal)
        result=hook.queue_once(case,terminal,invoke=fake);assert result['status']=='callback_pending_attempted' and not result['daemon_resume_confirmed']
        hook.queue_once(case,terminal,invoke=fake);assert len(calls)==1 and json.loads((case/'terminal.json').read_text())==terminal
    from completion import notify_once
    def notification_failure(*args,**kwargs):raise RuntimeError('MOCK_NOTIFICATION_FAILURE')
    notify_once(case,'completed',invoke=notification_failure);notify_once(case,'completed',invoke=notification_failure)
    assert json.loads((case/'terminal.json').read_text())==terminal and json.loads((case/'notification.result.json').read_text())['attempts']==1
    result={'status':'passed','scope':'isolated synthetic lifecycle/mock callback, no actual training/task install/queue','completed_arm_skipped':True,'OS_double_start_rejected':True,'cumulative_active_preserved_offline_excluded':True,'callback_429_nonzero_exit0_warning_isolated_and_once':True,'TaskScheduler_parent_exit_test':'PENDING after reviewed installer is authorized; no breakaway guarantee claimed'}
    atomic_json(stage/'CHECKS.json',result);print(json.dumps({'stage':str(stage),**result}))
if __name__=='__main__':main()
