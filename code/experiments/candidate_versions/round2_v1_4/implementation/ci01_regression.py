"""CI01 synthetic regression. No real signals, labels or model training."""
from pathlib import Path
from dataclasses import asdict
import hashlib,json
import numpy as np
from frontend import process_windows,FrontendConfig
from checks import oracle_one
from cache import key_for

V=Path(__file__).resolve().parents[1]

if __name__=='__main__':
    t=np.arange(256)*2*np.pi/16;cfg=FrontendConfig();records=[]
    for ppg_scale in (1e-5,1.,1e5,1e10):
        for acc_scale in (1e-5,1.,1e5,1e10):
            for phase in (0.,np.pi/4,np.pi/2,np.pi):
                x=np.zeros((1,4,256),dtype=np.float32)
                x[0,0]=ppg_scale*np.sin(t);x[0,1]=acc_scale*np.sin(t+phase)
                r=process_windows(x);d,f=oracle_one(x[0],cfg)
                pearson=abs(np.corrcoef(x[0,0].astype('float64'),x[0,1].astype('float64'))[0,1])
                assert abs(float(r['features'][0,0])-pearson)<2e-6
                np.testing.assert_allclose(r['features'][0],f,atol=2e-5,rtol=2e-5)
                normalized_error=float(np.max(np.abs(r['filtered_ppg'][0]-d))/ppg_scale)
                assert normalized_error<2e-5
                records.append({'ppg_scale':ppg_scale,'acc_scale':acc_scale,'phase':phase,'c':float(r['features'][0,0]),
                    'fp64_c':float(pearson),'filtered_error_divided_by_ppg_scale':normalized_error})
    rejected=[]
    for kind in ('variance_overflow','mean_overflow'):
        huge=np.zeros((1,4,256),dtype=np.float32)
        huge[0,0]=1e20*np.sin(t) if kind=='variance_overflow' else np.float32(1e38)
        assert np.isfinite(huge).all()
        try:process_windows(huge)
        except FloatingPointError as e:rejected.append({'kind':kind,'exception':str(e)})
        else:raise AssertionError('Unsupported intermediate range was silently accepted')
    # The same ordered input/config/proposal cannot hit the old frontend cache.
    probe=np.zeros((1,4,256),dtype=np.float32);probe[0,0]=np.sin(t);probe[0,1]=np.cos(t)
    new_key,binding=key_for(probe,cfg,'fixed-proposal-test')
    old_source=V.parent/'v0.1_2ee6e442e02d/implementation/frontend.py'
    old_binding={**binding,'frontend_code_sha256':hashlib.sha256(old_source.read_bytes()).hexdigest()}
    old_key=hashlib.sha256(json.dumps(old_binding,sort_keys=True).encode()).hexdigest()
    assert new_key!=old_key
    report={'status':'passed','scope':'Synthetic CI01 regression only, no HR result; not coverage of arbitrary finite FP32 range',
        'amplitude_phase_cases':len(records),'records':records,'unsupported_finite_input_cases_rejected':rejected,
        'old_cache_key':old_key,'new_cache_key':new_key,'cache_key_changed':True,
        'frontend_sha256':hashlib.sha256(Path(__file__).with_name('frontend.py').read_bytes()).hexdigest(),
        'required_1e10_same_signal_case':[r for r in records if r['ppg_scale']==1e10 and r['acc_scale']==1e10 and r['phase']==0.][0]}
    with (V/'ci01_regression.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
