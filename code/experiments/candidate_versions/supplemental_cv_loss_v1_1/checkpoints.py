"""Project-owned two-generation atomic checkpoints; trusted local tensor payloads."""
from pathlib import Path
import hashlib,json,os,uuid
import torch
def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def atomic_json(p,value):
    p=Path(p);tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    with tmp.open('x',encoding='utf-8') as f:json.dump(value,f,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    os.replace(tmp,p)
class Store:
    def __init__(self,path,identity):
        self.path=Path(path);self.path.mkdir(parents=True,exist_ok=True);self.identity=identity;self.index=self.path/'CURRENT.json'
    def entries(self):
        if not self.index.exists():return []
        try:value=json.loads(self.index.read_text(encoding='utf-8'))
        except (ValueError,OSError):value=json.loads((self.path/'PREVIOUS.json').read_text(encoding='utf-8'))
        assert value['identity']==self.identity,'Checkpoint identity mismatch'
        return value['generations']
    def valid(self,row):
        p=self.path/row['name'];return p.parent==self.path and p.is_file() and sha(p)==row['sha256']
    def save(self,state):
        old=self.entries();valid=[r for r in old if self.valid(r)]
        name='checkpoint_'+uuid.uuid4().hex+'.pt';p=self.path/name;tmp=p.with_suffix('.tmp')
        payload={'format_version':1,'complete':True,'identity':self.identity,'state':state}
        with tmp.open('xb') as f:torch.save(payload,f);f.flush();os.fsync(f.fileno())
        os.replace(tmp,p);row={'name':name,'sha256':sha(p),'updates':state['updates']}
        keep=[row]+valid[:1]
        if valid:atomic_json(self.path/'PREVIOUS.json',{'identity':self.identity,'generations':valid})
        atomic_json(self.index,{'identity':self.identity,'generations':keep})
        # Delete only obsolete, verified project-owned checkpoint generations after commit.
        for stale in valid[1:]:
            q=self.path/stale['name'];assert q.parent==self.path and q.name.startswith('checkpoint_') and q.suffix=='.pt';q.unlink()
        return row
    def load(self):
        rows=self.entries()
        for i,row in enumerate(rows):
            if not self.valid(row):continue
            payload=torch.load(self.path/row['name'],map_location='cpu',weights_only=False)
            assert payload['format_version']==1 and payload['complete'] is True and payload['identity']==self.identity
            return payload['state'],{'generation':row,'fallback_to_previous':i>0}
        raise RuntimeError('No complete compatible checkpoint; refuse blind restart')
