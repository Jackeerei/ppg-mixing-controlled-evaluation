"""Native lifecycle supervisor, no LLM polling; stopped-ledger access only."""
from pathlib import Path
import datetime,hashlib,json,os,psutil,subprocess,sys,time,uuid,shutil
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];PROJECT=ROOT.parent
sys.path.insert(0,str(HERE))
import run as owner
from resources import snapshot,admissible,parallel_ready
from io_guard import atomic
from active_clock import seconds
OUT=ROOT/'reports/autonomous_guardian_v1';STATE=OUT/'state.json'
TASK='Codex_PPG_Recovery_Resume_v1'

def utc():return datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def producers():
    rows=[]
    suffixes=tuple(v+'/'+f for v in ('parallel_recovery_v1','parallel_local_v1','notification_resolution_v1','autonomous_owner_v1') for f in ('run.py','worker.py','wrapper.py','bridge.py'))
    for p in psutil.process_iter(['pid','cmdline']):
        try:
            if any(a.replace('\\','/').endswith(suffixes) for a in p.info['cmdline'] or []):rows.append(p.info)
        except(psutil.NoSuchProcess,psutil.AccessDenied):pass
    return rows
def task_command(code):
    return subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-WindowStyle','Hidden','-Command',"$ErrorActionPreference='Stop'; "+code],capture_output=True,text=True,check=True,timeout=30,creationflags=subprocess.CREATE_NO_WINDOW)
def validate_checkpoint(arm):
    p=arm/'checkpoints/CURRENT.json'
    if not p.exists():
        assert not (arm/'arm_intent.json').exists(),'Started identity missing full checkpoint'
        return
    value=json.loads(p.read_text());assert value['identity']['run_id']==arm.name
    assert value['identity']['study_run_id']==owner.BUNDLE.name
    assert value['identity']['freeze_sha256']=='e655f4d594264a46a76befc06ecfe4a3fac1e745f62b8ec9d1854674d7d1fb8d'
    valid=False
    for g in value['generations']:
        path=p.parent/g['name'];assert path.parent==p.parent and path.name.startswith('checkpoint_')
        valid=valid or (path.is_file() and sha(path)==g['sha256'])
    assert valid,'No valid full checkpoint; never restart blindly'
def retry_failed(state,attempts):
    # Queue work first; one bounded extra continuation per failed identity afterward.
    if any(v in ('queued','running') for v in state['arm_status'].values()):return []
    chosen=[]
    for name,status in state['arm_status'].items():
        if status=='failed' and attempts.get(name,0)<1:
            state['arm_status'][name]='running' # Resume same state, retain legacy attempt counters.
            attempts[name]=attempts.get(name,0)+1;chosen.append(name)
    return chosen
def verify_files():
    for row in json.loads((HERE/'FREEZE.json').read_text())['files']:assert sha(PROJECT/row['path'])==row['sha256']
def notify_stop(state,reason):
    from hook import queue_once
    event=OUT/('unrecoverable-'+uuid.uuid4().hex);event.mkdir()
    terminal={'status':'guardian_stopped:'+reason,'bundle':str(owner.BUNDLE),'exit_code':1}
    atomic(event/'terminal.json',terminal);queue_once(event,terminal)
    state.update(stopped=True,reason=reason);atomic(STATE,state)
def main():
    OUT.mkdir(exist_ok=True);verify_files()
    with owner.Lock(ROOT/'autonomous_guardian_v1.lock'):
        record=json.loads(STATE.read_text()) if STATE.exists() else {'attempts_by_identity':{},'launches_by_fingerprint':{},'handled_events':[],'stopped':False}
        if record.get('stopped'):return
        atomic(OUT/('STARTED-'+uuid.uuid4().hex+'.json'),{'pid':os.getpid(),'utc':utc(),'freeze':sha(HERE/'FREEZE.json'),'no_model_polling':True})
        pressure_samples=0
        while True:
            try:
                active=producers()
                if active:
                    resources=snapshot(ROOT)
                    severe=resources['available_ram_bytes']<1.5*1024**3 or resources['commit_headroom_bytes']<2.5*1024**3
                    pressure_samples=pressure_samples+1 if severe else 0
                    native=any(any(a.replace('\\','/').endswith('autonomous_owner_v1/run.py') for a in p['cmdline']) for p in active)
                    if native and pressure_samples>=2 and not (owner.OWNER/'resource_pause.request').exists():
                        atomic(owner.OWNER/'resource_pause.request',{'utc':utc(),'resources':resources,'action':'full checkpoint and drain; no worker kill'})
                    record.update(phase='observing_active_native_owner' if native else 'waiting_for_legacy_natural_exit',utc=utc(),producers=active)
                    atomic(STATE,record);time.sleep(15);continue
                # Task can still be completing its one-shot callback: do not race it.
                if task_command("(Get-ScheduledTask -TaskName '"+TASK+"').State.ToString()").stdout.strip()=='Running':time.sleep(15);continue
                resources=snapshot(ROOT)
                # Additional reserve: original gates remain lower bounds. Single worker unless ample space.
                ready=admissible(resources) and resources['available_ram_bytes']>=4*1024**3 and resources['commit_headroom_bytes']>=4*1024**3
                if not ready:
                    record.update(phase='waiting_for_resources',utc=utc(),resource_observation=resources);atomic(STATE,record);time.sleep(30);continue
                with owner.Lock(ROOT/'supplemental_cv_loss_v1_1.lock'):
                    if producers():continue
                    state=owner.read(owner.CONTROL);cfg=owner.verify()
                    if state['global_active_seconds']>=state['global_limit_seconds']:
                        notify_stop(record,'global_budget_exhausted');return
                    if state['status']=='completed':record.update(stopped=True,reason='completed');atomic(STATE,record);return
                    if state.get('user_stop') or state['status'] in ('user_stopped','cancelled'):
                        notify_stop(record,'explicit_user_stop');return
                    event=state['terminal_event_id']
                    pending=next((n for n,v in state['arm_status'].items() if v in ('queued','running')),next((n for n,v in state['arm_status'].items() if v=='failed'),'none'))
                    cp=owner.BUNDLE/pending/'checkpoints/CURRENT.json'
                    progress=owner.read(cp)['generations'][0]['updates'] if cp.exists() else 0
                    fingerprint='|'.join(map(str,(state['status'],state.get('dispatch_paused_reason'),state.get('exception'),pending,progress)))
                    count=record['launches_by_fingerprint'].get(fingerprint,0)
                    if count>=2:notify_stop(record,'same_event_start_failed_twice');return
                    archive=OUT/('handoff-'+uuid.uuid4().hex);archive.mkdir()
                    for n in ('control.json','terminal.json','MANIFEST.json'):
                        p=owner.OWNER/n
                        if p.exists():shutil.copy2(p,archive/n)
                    for r in cfg['runs']:
                        arm=owner.BUNDLE/r['id']
                        if owner.commit(arm,r):state['arm_status'][r['id']]='completed'
                        elif state['arm_status'][r['id']] in ('queued','running','failed'):validate_checkpoint(arm)
                        else:raise RuntimeError('Unknown arm status')
                    retry=retry_failed(state,record['attempts_by_identity'])
                    all_training=all(v=='completed' for v in state['arm_status'].values())
                    if not all_training and not any(v in ('queued','running') for v in state['arm_status'].values()):
                        notify_stop(record,'all_remaining_identities_reached_retry_limit');return
                    # Protect counters on abrupt owner exit; conservative extra debit is explicit, never a refund.
                    safety=5.*max(len(state.get('workers',{})),1) if state.get('armed') else 0.
                    state['global_active_seconds']+=safety
                    state['shared_only_seconds']+=safety
                    state.update(armed=True,status='ready',dispatch_paused_reason=None,resource_admission=None,
                                 owner_freeze_sha256=sha(HERE/'FREEZE.json'),terminal_event_id='autonomous-terminal-'+uuid.uuid4().hex,
                                 max_workers=2 if parallel_ready(resources) else 1)
                    marker=owner.OWNER/'resource_pause.request'
                    if marker.exists():shutil.move(str(marker),str(archive/'resource_pause.request'))
                    atomic(archive/'MIGRATION.json',{'utc':utc(),'retry':retry,'crash_safety_debit_seconds':safety,'resources':resources,'max_workers':state['max_workers'],'new_event':state['terminal_event_id'],'preserved_global_seconds_after_debit':state['global_active_seconds'],'old_failure_files_untouched':True})
                    record['launches_by_fingerprint'][fingerprint]=count+1
                    record.update(phase='launch_committed',utc=utc(),last_migration=str(archive/'MIGRATION.json'));atomic(STATE,record)
                    source=owner.read(owner.SOURCE_CONTROL);source.update(armed=True,status='external_autonomous_owner',external_owner=str(owner.CONTROL));atomic(owner.SOURCE_CONTROL,source);atomic(owner.CONTROL,state)
                    executable=str(ROOT/'.venv/Scripts/pythonw.exe');entry=str(HERE/'run.py')
                    code="$a=New-ScheduledTaskAction -Execute '"+executable+"' -Argument '-B -X utf8 \""+entry+"\" --resume-armed' -WorkingDirectory '"+str(PROJECT)+"'; Set-ScheduledTask -TaskName '"+TASK+"' -Action $a | Out-Null"
                    task_command(code)
                task_command("Start-ScheduledTask -TaskName '"+TASK+"'");time.sleep(15)
            except Exception as exc:
                # No silent loops on identity/counter corruption or unexpected lifecycle faults.
                atomic(OUT/'FAILURE.json',{'utc':utc(),'exception':repr(exc),'training_killed':False})
                notify_stop(record,repr(exc));return

if __name__=='__main__':main()
