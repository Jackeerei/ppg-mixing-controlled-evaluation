"""One immutable label-free frontend artifact shared by B/C/G/D."""
from pathlib import Path
from dataclasses import asdict
import hashlib,json
import numpy as np
from frontend import process_windows,FrontendConfig

def key_for(windows,config,proposal_sha):
    x=np.ascontiguousarray(windows,dtype=np.float32)
    payload={'input_shape':list(x.shape),'input_sha256':hashlib.sha256(x.tobytes()).hexdigest(),
             'config':asdict(config),'proposal_sha256':proposal_sha,
             'frontend_code_sha256':hashlib.sha256(Path(__file__).with_name('frontend.py').read_bytes()).hexdigest()}
    return hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest(),payload

def get_or_create(directory,windows,proposal_sha,config=FrontendConfig()):
    key,meta=key_for(windows,config,proposal_sha);directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    p=directory/(key+'.npz');m=directory/(key+'.json')
    if p.exists():
        report=json.loads(m.read_text())
        assert report['binding']==meta
        assert hashlib.sha256(p.read_bytes()).hexdigest()==report['artifact_sha256']
    else:
        result=process_windows(windows,config)
        with p.open('xb') as f:np.savez(f,**{k:v for k,v in result.items() if isinstance(v,np.ndarray)})
        with m.open('x') as f:json.dump({'binding':meta,'artifact_sha256':hashlib.sha256(p.read_bytes()).hexdigest()},f,indent=2)
    return p
