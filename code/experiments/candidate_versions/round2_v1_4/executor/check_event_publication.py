"""Narrow L03: actual persist->append partial writes and resolver fault tolerance."""
from pathlib import Path
import contextlib,hashlib,io,json,time
from unittest.mock import patch
import psutil
import runner,launch,host_observer as observer
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    output=runner.ROOT/'reports'/('round2_v1_4_L03_SYNTHETIC_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()));output.mkdir(exist_ok=False)
    now=psutil.Process();identity={'status':'running','boot_time':psutil.boot_time(),'launcher_pid':now.pid,'launcher_created':now.create_time(),'runner_pid':now.pid,'runner_created':now.create_time()}
    terminal={'status':'completed','exit_code':0,'child_created':True,'finish_utc':'SYNTHETIC_NO_TRAINING'}
    def intent(case):
        root=output/case;root.mkdir();p=root/'launch.json';p.write_text(json.dumps({k:v for k,v in identity.items() if not k.startswith('runner_')}));return p
    def torn(value,file,**kwargs):file.write('{"kind":');file.flush();raise OSError('INJECTED_AFTER_ACTUAL_WRITE_AND_FLUSH')
    results={}
    p=intent('identity_torn_valid_terminal');before=digest(p);error=io.StringIO()
    with contextlib.redirect_stderr(error),patch.object(observer.json,'dump',torn):assert launch.persist(p,'identity',identity,None) is None
    pending=list(p.parent.glob('*.pending'));assert len(pending)==1 and pending[0].read_text()=='{"kind":'
    assert not list(p.parent.glob(p.stem+'_*.json')) and 'persistence_failed' in error.getvalue()
    t=observer.append_event(p,'terminal',terminal);old_t=digest(t);resolved=observer.resolve(p)
    assert resolved['classification']=='recorded_terminal' and resolved['terminal']['status']=='completed' and resolved['invalid_events']
    assert digest(p)==before and digest(t)==old_t and pending[0].exists()
    (p.parent/'diagnostics.txt').write_text(error.getvalue());results['actual_identity_half_write_is_unpublished_legal_terminal_readable']=True
    p=intent('terminal_torn_valid_identity');before=digest(p);i=observer.append_event(p,'identity',identity);old_i=digest(i);error=io.StringIO()
    with contextlib.redirect_stderr(error),patch.object(observer.json,'dump',torn):assert launch.persist(p,'terminal',terminal,None) is None
    resolved=observer.resolve(p);assert resolved['terminal_status']=='unknown' and resolved['classification']=='running_identity_verified' and resolved['invalid_events']
    assert digest(p)==before and digest(i)==old_i and len(list(p.parent.glob('*.pending')))==1
    (p.parent/'diagnostics.txt').write_text(error.getvalue());results['actual_terminal_half_write_keeps_identity_and_terminal_unknown']=True
    p=intent('bad_published_events');before=digest(p)
    bad={p.stem+'_01.json':'{"kind":',p.stem+'_02.json':'[]',p.stem+'_03.json':'{}',p.stem+'_04.json':json.dumps({'intent_sha256':'WRONG','kind':'terminal','value':terminal})}
    for name,text in bad.items():(p.parent/name).write_text(text)
    incomplete=observer.append_event(p,'terminal',{'status':'completed','exit_code':0});snapshot={q:digest(q) for q in p.parent.glob('*.json')}
    resolved=observer.resolve(p);assert resolved['terminal_status']=='unknown' and resolved['classification']=='starting_runner_identity_unknown' and len(resolved['invalid_events'])==5
    legal=observer.append_event(p,'terminal',terminal);legal_sha=digest(legal)
    unreadable=p.parent/(p.stem+'_05.json');unreadable.write_text('{}');original_read=Path.read_text
    def guarded(path,*a,**k):
        if path==unreadable:raise PermissionError('INJECTED_UNREADABLE_EVENT')
        return original_read(path,*a,**k)
    with patch.object(Path,'read_text',guarded):resolved=observer.resolve(p)
    assert resolved['classification']=='recorded_terminal' and resolved['terminal']==terminal and len(resolved['invalid_events'])==6
    assert all(q.exists() and digest(q)==h for q,h in snapshot.items()) and digest(p)==before and digest(legal)==legal_sha
    results['truncated_nonobject_incomplete_wrong_binding_unreadable_events_reported_not_deleted']=True
    # Failure during atomic publication preserves complete pending evidence and does not alter existing event.
    before_events={q:digest(q) for q in p.parent.glob('*.json')}
    with contextlib.redirect_stderr(io.StringIO()),patch.object(observer.os,'replace',side_effect=OSError('INJECTED_PUBLISH_FAILURE')):assert launch.persist(p,'identity',identity,None) is None
    assert len(list(p.parent.glob('*.pending')))==1 and all(digest(q)==h for q,h in before_events.items())
    results['publication_failure_keeps_pending_and_all_old_events_immutable']=True
    report={'status':'passed','scope':'NARROW_L03_ACTUAL_PERSIST_APPEND_AND_RESOLVE_NO_SUBPROCESSES','execution_freeze_sha256':runner.sha(runner.V/'EXECUTION_FREEZE.json'),'checks':results,'real_subprocesses_started':0,'real_cache_or_training_started':False,'test_script_sha256':digest(__file__)}
    (output/'checks.json').write_text(json.dumps(report,indent=2));print(json.dumps({'output':str(output),**report},indent=2))
if __name__=='__main__':main()
