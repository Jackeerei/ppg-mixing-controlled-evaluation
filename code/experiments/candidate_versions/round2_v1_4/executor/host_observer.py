"""Append-only events bound to immutable launch intent; tolerate incomplete identity."""
from pathlib import Path
import argparse,datetime,json,hashlib,os,uuid,math,sys
import psutil

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def classify(record,boot,current_pid_created):
    if record.get('status') in ('completed','failed'):return 'recorded_'+record['status']
    if abs(boot-record['boot_time'])>1:return 'interrupted_host_boot_changed'
    if record.get('runner_pid') is None or record.get('runner_created') is None:return 'starting_runner_identity_unknown'
    if current_pid_created is None:return 'interrupted_process_missing'
    if abs(current_pid_created-record['runner_created'])>.01:return 'interrupted_pid_reused'
    return 'running_identity_verified'

def observe(record):
    created=None;processes=[];errors=[];launcher_state='not_checked'
    if record.get('runner_pid') is not None:
        try:
            parent=psutil.Process(record['runner_pid']);created=parent.create_time()
            if record.get('runner_created') is not None and abs(created-record['runner_created'])<=.01:
                for p in [parent]+parent.children(recursive=True):
                    try:processes.append({'pid':p.pid,'created':p.create_time(),'ppid':p.ppid()})
                    except (psutil.NoSuchProcess,psutil.AccessDenied) as exc:errors.append(repr(exc))
        except (psutil.NoSuchProcess,psutil.AccessDenied) as exc:errors.append(repr(exc))
    if record.get('runner_created') is None:
        try:
            q=psutil.Process(record['launcher_pid']);launcher_state='same_identity' if abs(q.create_time()-record['launcher_created'])<=.01 else 'pid_reused'
        except psutil.NoSuchProcess:launcher_state='missing'
        except psutil.AccessDenied:launcher_state='unavailable'
    return {'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'classification':classify(record,psutil.boot_time(),created),
            'boot_time':psutil.boot_time(),'runner_created_now':created,'launcher_state':launcher_state,'processes':processes,'identity_errors':errors,
            'scope':'Missing starting identity is unknown, not proof of an active or absent runner'}

def append_event(intent_path,kind,value):
    p=Path(intent_path);stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ');name=p.stem+'_'+kind+'_'+stamp+'_'+uuid.uuid4().hex
    destination=p.with_name(name+'.json');temporary=p.with_name(name+'.pending')
    row={'kind':kind,'intent_path':str(p),'intent_sha256':digest(p),'value':value}
    # Failed/partial writes remain unpublished evidence, never a final *.json event.
    with temporary.open('x',encoding='utf-8') as f:
        json.dump(row,f,indent=2);f.flush();os.fsync(f.fileno())
    assert not destination.exists()
    os.replace(temporary,destination)
    return destination

def sidecar(path,result):return append_event(path,'terminal_observation',result)
def resolve(path):
    p=Path(path);intent=json.loads(p.read_text());expected=digest(p)
    identities=[];terminals=[];invalid=[]
    for pending in sorted(p.parent.glob(p.stem+'_*.pending')):
        invalid.append({'path':str(pending),'error':'unpublished_pending_event; preserved, not trusted as terminal'})
    for f in sorted(p.parent.glob(p.stem+'_*.json')):
        try:
            row=json.loads(f.read_text());assert isinstance(row,dict),'event is not an object'
            assert row.get('intent_sha256')==expected,'intent SHA mismatch'
            assert Path(row['intent_path']).resolve()==p.resolve(),'intent path mismatch'
            kind=row['kind'];value=row['value'];assert isinstance(value,dict),'value is not an object'
            if kind=='identity':
                assert value.get('status') in ['starting','running'],'missing identity status'
                assert all(k in value for k in ['boot_time','launcher_pid','launcher_created','runner_pid','runner_created']),'incomplete identity'
                assert all(isinstance(value[k],(int,float)) and math.isfinite(value[k]) for k in ['boot_time','launcher_pid','launcher_created','runner_pid']),'invalid identity numbers'
                assert value['runner_created'] is None or isinstance(value['runner_created'],(int,float)) and math.isfinite(value['runner_created']),'invalid runner creation time'
                identities.append(row)
            elif kind=='terminal':
                assert value.get('status') in ['completed','failed','launch_failed','launcher_error'],'invalid terminal status'
                assert isinstance(value.get('finish_utc'),str) and value['finish_utc'],'missing terminal time'
                assert isinstance(value.get('child_created'),bool),'missing child-created state'
                if value['status']=='completed':assert value.get('exit_code')==0 and value['child_created'],'invalid completion'
                if value['status']=='failed':assert isinstance(value.get('exit_code'),int) and value['exit_code']!=0,'invalid failure code'
                terminals.append(row)
            elif kind=='terminal_observation':
                assert isinstance(value.get('classification'),str),'incomplete terminal observation'
            else:raise ValueError('Unknown event kind')
        except Exception as exc:invalid.append({'path':str(f),'error':repr(exc)})
    if terminals:return {'classification':'recorded_terminal','terminal':terminals[-1]['value'],'intent_sha256':expected,'invalid_events':invalid}
    record=identities[-1]['value'] if identities else intent
    result=observe(record);result.update(terminal_status='unknown',invalid_events=invalid)
    return result
def main():
    p=argparse.ArgumentParser();p.add_argument('--record',required=True);a=p.parse_args();value=resolve(a.record)
    if value['classification']!='running_identity_verified':
        try:value['observation_sidecar']=str(sidecar(a.record,value))
        except Exception as exc:
            value['observation_sidecar_error']=repr(exc);print('Observation not persisted: '+repr(exc),file=sys.stderr,flush=True)
    print(json.dumps(value,indent=2))
if __name__=='__main__':main()
