"""Immutable intent + append-only identity/terminal; owned bounded lifecycle."""
from pathlib import Path
import argparse,datetime,json,subprocess,sys,time
import psutil
import runner
from host_observer import observe,append_event

def diagnostic(output,event):
    text=json.dumps(event)
    print(text,file=sys.stderr,flush=True)
    if output is not None:
        try:output.write(text+'\n');output.flush()
        except Exception as exc:print('Diagnostic file unavailable: '+repr(exc),file=sys.stderr,flush=True)

def persist(path,kind,value,output):
    try:return append_event(path,kind,value)
    except Exception as exc:
        diagnostic(output,{'event':'persistence_failed','kind':kind,'exception':repr(exc),'not_persisted':value})
        return None

def safe_stop(child,identity):
    # Never use a numeric PID alone to terminate. Popen.terminate uses the held OS handle.
    result={'policy':'owned Popen handle; descendants only with verified boot/create_time','descendants_verified':False,'errors':[]}
    owned=[]
    try:
        if identity.get('runner_created') is not None and abs(psutil.boot_time()-identity['boot_time'])<=1:
            root=psutil.Process(child.pid)
            if abs(root.create_time()-identity['runner_created'])<=.01 and root.is_running():
                owned=[(q,q.create_time()) for q in root.children(recursive=True)]
                result['descendants_verified']=True
        for q,created in reversed(owned):
            try:
                if q.is_running() and abs(q.create_time()-created)<=.01:q.terminate()
            except (psutil.NoSuchProcess,psutil.AccessDenied) as exc:result['errors'].append(repr(exc))
    except (psutil.NoSuchProcess,psutil.AccessDenied) as exc:result['errors'].append(repr(exc))
    if child.poll() is None:
        try:child.terminate();child.wait(timeout=5)
        except subprocess.TimeoutExpired:child.kill();child.wait(timeout=5)
        except Exception as exc:result['errors'].append(repr(exc))
    if owned:
        _,alive=psutil.wait_procs([q for q,_ in owned],timeout=5)
        for q in alive:
            try:
                created=next(t for p,t in owned if p.pid==q.pid)
                if q.is_running() and abs(q.create_time()-created)<=.01:q.kill()
            except (psutil.NoSuchProcess,psutil.AccessDenied) as exc:result['errors'].append(repr(exc))
        _,remaining=psutil.wait_procs(alive,timeout=5)
        result['remaining_verified_descendant_pids']=[q.pid for q in remaining]
    result['owned_handle_exit_code']=child.poll()
    if not result['descendants_verified']:result['descendant_status']='unknown; never claim all descendants stopped'
    return result

def main(argv=None):
    p=argparse.ArgumentParser();p.add_argument('--review-receipt',required=True);p.add_argument('--availability-note',required=True);a=p.parse_args(argv)
    assert a.availability_note.strip()
    receipt=Path(a.review_receipt).resolve();runner.receipt(receipt);cfg=runner.read(runner.V/'protocol.json');runner.resource(cfg)
    assert not (runner.ROOT/'logs/round2_v1_4_attempts.json').exists()
    active=[]
    for process in psutil.process_iter(['pid','name','cmdline']):
        try:
            if 'python' in (process.info['name'] or '').lower() and any('runner.py' in s for s in process.info['cmdline'] or []):active.append(process.info)
        except (psutil.NoSuchProcess,psutil.AccessDenied):pass
    assert not active,active
    with runner.Lock():pass
    stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');bundle=runner.ROOT/'results'/('round2_v14_'+stamp)
    path=runner.ROOT/'logs'/('round2_v14_launch_'+stamp+'.json');assert not bundle.exists()
    command=[sys.executable,'-B','-X','utf8',str(runner.HERE/'runner.py'),'--execute',str(bundle),'--review-receipt',str(receipt)]
    intent={'status':'starting','bundle':str(bundle),'boot_time':psutil.boot_time(),'launcher_pid':psutil.Process().pid,'launcher_created':psutil.Process().create_time(),
            'launch_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'command':command,'availability_note':a.availability_note,
            'execution_freeze_sha256':runner.sha(runner.V/'EXECUTION_FREEZE.json'),'review_receipt_sha256':runner.sha(receipt),'automatic_retry':False}
    # Original intent is written exactly once, never updated even on normal exit.
    try:
        with path.open('x',encoding='utf-8') as f:json.dump(intent,f,indent=2)
    except Exception as exc:
        diagnostic(None,{'event':'intent_write_failed','exception':repr(exc),'child_started':False});return 1
    output=None;observations=None;child=None;identity={**intent};terminal={'status':'launch_failed','child_created':False};started=time.monotonic();degraded=[]
    try:
        output=path.with_suffix('.log').open('x',encoding='utf-8')
        child=subprocess.Popen(command,stdout=output,stderr=subprocess.STDOUT,cwd=runner.PROJECT)
        terminal['child_created']=True;identity.update(runner_pid=child.pid,runner_created=None)
        try:identity['runner_created']=psutil.Process(child.pid).create_time()
        except (psutil.NoSuchProcess,psutil.AccessDenied) as exc:
            degraded.append('identity_query');diagnostic(output,{'event':'identity_query_failed','pid':child.pid,'exception':repr(exc),'policy':'retain owned Popen and continue bounded wait'})
        persisted=persist(path,'identity',identity,output)
        identity_sha=runner.sha(persisted) if persisted is not None else None
        terminal.update(runner_pid=child.pid,runner_created=identity['runner_created'],identity_sha256=identity_sha)
        if persisted is None:degraded.append('identity_persistence')
        try:observations=path.with_suffix('.observations.jsonl').open('x',encoding='utf-8')
        except Exception as exc:degraded.append('observations_open');diagnostic(output,{'event':'observations_open_failed','exception':repr(exc)})
        while child.poll() is None:
            if time.monotonic()-started>=cfg['operations']['launcher_observation_ceiling_seconds']:raise TimeoutError('Launcher observation ceiling; preserve runner hard limits, stop verified owned processes')
            try:
                value=observe(identity)
                if observations is not None:observations.write(json.dumps(value)+'\n');observations.flush()
            except Exception as exc:
                degraded.append('observation');diagnostic(output,{'event':'observation_degraded','exception':repr(exc),'policy':'continue owned Popen bounded wait; no retry'})
                if observations is not None:
                    try:observations.close()
                    except Exception:pass
                    observations=None
            time.sleep(10)
        terminal.update(status='completed' if child.returncode==0 else 'failed',exit_code=child.returncode)
    except BaseException as exc:
        terminal.update(status='launcher_error',exception=repr(exc));diagnostic(output,{'event':'launcher_error','exception':repr(exc)})
    finally:
        if child is not None and child.poll() is None:
            try:terminal['owned_process_disposition']=safe_stop(child,identity)
            except BaseException as exc:
                terminal['disposition_error']=repr(exc);diagnostic(output,{'event':'disposition_failed','pid':child.pid,'exception':repr(exc),'all_descendants_stopped':False})
                try:
                    if child.poll() is None:child.terminate();child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    try:child.kill();child.wait(timeout=5)
                    except Exception as nested:terminal['owned_handle_fallback_error']=repr(nested)
                except Exception as nested:terminal['owned_handle_fallback_error']=repr(nested)
        terminal.update(finish_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),monitoring_degradation=sorted(set(degraded)),intent_never_overwritten=True)
        persist(path,'terminal',terminal,output)
        for handle in [observations,output]:
            if handle is not None:
                try:handle.close()
                except Exception as exc:print('Close failed: '+repr(exc),file=sys.stderr,flush=True)
    print(json.dumps({'record':str(path),'terminal':terminal}),flush=True)
    return 0 if terminal['status']=='completed' else 1
if __name__=='__main__':raise SystemExit(main())
