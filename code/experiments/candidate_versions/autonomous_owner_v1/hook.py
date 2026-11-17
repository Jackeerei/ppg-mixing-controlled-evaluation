"""Resolve the current desktop CLI at event time; never resend a claimed event."""
from pathlib import Path
import datetime,json,os,shutil,subprocess,uuid

TARGET_THREAD='01a12dd2-482d-7470-a256-572b41c74f30'

def resolve_codex(invoke=subprocess.run):
    candidates=[]
    for name in ('codex.exe','codex'):
        p=shutil.which(name)
        if p and Path(p).suffix.lower()=='.exe':candidates.append(Path(p))
    base=Path(os.environ.get('LOCALAPPDATA',str(Path.home()/'AppData/Local')))/'OpenAI/Codex/bin'
    if base.is_dir():candidates.extend(sorted(base.glob('*/codex.exe'),key=lambda p:p.stat().st_mtime,reverse=True))
    seen=set();errors=[]
    for path in candidates:
        key=str(path.resolve()).lower()
        if key in seen or not path.is_file():continue
        seen.add(key)
        try:
            q=invoke([str(path),'queue','--help'],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=10,creationflags=subprocess.CREATE_NO_WINDOW)
            if q.returncode==0 and '--thread' in q.stdout and '--message' in q.stdout and '--resume' in q.stdout:return path
            errors.append({'path':str(path),'returncode':q.returncode})
        except (OSError,subprocess.TimeoutExpired) as exc:errors.append({'path':str(path),'error':repr(exc)})
    raise FileNotFoundError('No installed CLI with queue capability: '+json.dumps(errors))

def write(path,value):
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    with temporary.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(temporary,path)

def _deliver_once(stage,state,invoke=subprocess.run,resolver=resolve_codex):
    stage=Path(stage)
    try:
        with (stage/'queue.claim.json').open('x',encoding='utf-8') as f:
            json.dump({'claimed_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'target_thread':TARGET_THREAD},f);f.flush();os.fsync(f.fileno())
    except FileExistsError:return {'status':'already_claimed_no_resend'}
    result={'attempts':0,'daemon_resume_confirmed':False}
    try:
        executable=resolver()
        message=('本地实验出现一次新终态：'+state['status']+'；bundle='+str(state.get('bundle'))+'；终态='+str(stage/'terminal.json')+'。只核新事件；失败不自动重训，不重开预算，不重复最终测试。')
        argv=[str(executable),'queue','--thread',TARGET_THREAD,'--message',message,'--resume']
        result.update(argv=argv,attempts=1,resolved_executable=str(executable))
        q=invoke(argv,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=30,creationflags=subprocess.CREATE_NO_WINDOW)
        (stage/'queue.stdout.log').write_text(q.stdout,encoding='utf-8');(stage/'queue.stderr.log').write_text(q.stderr,encoding='utf-8')
        warning='warn' in (q.stdout+q.stderr).lower()
        result.update(returncode=q.returncode,warning_present=warning,status='callback_pending_attempted' if q.returncode!=0 or warning else 'attempted_resume_unverified')
    except Exception as exc:result.update(status='queue_failed_preserved',exception=repr(exc))
    write(stage/'queue.result.json',result);return result


def queue_once(stage,state,invoke=subprocess.run,resolver=resolve_codex):
    if state['status']!='completed' and not state['status'].startswith('guardian_stopped:'):
        path=Path(stage)/'notification.deferred.json'
        if not path.exists():write(path,{'status':'native_guardian_handles_recoverable_stop','model_message_sent':False})
        return {'status':'deferred_to_native_guardian'}
    return _deliver_once(stage,state,invoke,resolver)
