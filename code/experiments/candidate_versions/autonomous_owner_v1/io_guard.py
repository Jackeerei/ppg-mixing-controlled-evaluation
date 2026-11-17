"""Bound only Windows access/sharing conflicts; original exception survives exhaustion."""
from pathlib import Path
import json,os,time,uuid
ORIGINAL_REPLACE=os.replace
def replace(source,target):
    delays=[.02,.04,.08,.12,.16,.20,.25,.30]
    for attempt in range(len(delays)+1):
        try:return ORIGINAL_REPLACE(source,target)
        except OSError as exc:
            if getattr(exc,'winerror',None) not in [5,32,33] or attempt==len(delays):raise
            print(json.dumps({'event':'bounded_replace_sharing_retry','attempt':attempt+1,'winerror':exc.winerror,'target':str(target)}),flush=True);time.sleep(delays[attempt])
def atomic(p,value):
    p=Path(p);tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    replace(tmp,p)
def install():os.replace=replace
