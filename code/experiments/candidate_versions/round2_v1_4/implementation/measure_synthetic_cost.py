"""Small single-window CPU timings of implemented components, not deployment."""
from pathlib import Path
import sys,json,time,csv,platform,statistics
import numpy as np
import torch
from frontend import process_windows
from gating import Variant
V=Path(__file__).resolve().parents[1];ROOT=V.parents[1]
sys.path.insert(0,str(ROOT))
from models.qppg_fixed import build_model

def measure(fn):
    for _ in range(2):fn()
    times=[]
    for _ in range(10):
        t=time.perf_counter_ns();fn();times.append((time.perf_counter_ns()-t)/1e6)
    return statistics.median(times)

if __name__=='__main__':
    torch.set_num_threads(2);rng=np.random.default_rng(908)
    x=rng.normal(size=(1,4,256)).astype('float32');xt=torch.from_numpy(x)
    front=process_windows(x);dt=torch.from_numpy(front['filtered_ppg']);ft=torch.from_numpy(front['features'])
    state=torch.load(V/'initial_backbone_seed42.pt',map_location='cpu',weights_only=True)
    rows=[]
    with torch.inference_mode():
        for method in ('A','B','C','G','D'):
            b=build_model();b.load_state_dict(state);model=Variant(b,method).eval()
            mixed,_=model.mix(xt,dt,ft)
            def end_to_end():
                if method=='A':return model.backbone(xt)
                r=process_windows(x)
                return model(xt,torch.from_numpy(r['filtered_ppg']),torch.from_numpy(r['features']))
            rows.append({'method':method,'device':'cpu','dtype':'float32','batch_size':1,'warmups':2,'iterations':10,
                'frontend_ms_median':0.0 if method=='A' else measure(lambda:process_windows(x)),
                'gate_mix_ms_median':measure(lambda:model.mix(xt,dt,ft)),
                'backbone_ms_median':measure(lambda:model.backbone(mixed)),
                'uncached_end_to_end_ms_median':measure(end_to_end),'parameter_count':sum(p.numel() for p in model.parameters())})
    with (V/'synthetic_cost_records.csv').open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    report={'scope':'Small synthetic single-window CPU implementation timing, no accuracy/MCU/GPU/energy claim; no stress test',
        'platform':platform.platform(),'processor':platform.processor(),'threads':2,'numpy':np.__version__,'torch':torch.__version__,
        'timing_scope':'In-memory; frontend includes standardization/NLMS/features, no disk/PKL/cache-file load; end-to-end includes frontend and numpy-to-torch views for B/C/G/D',
        'component_timings_not_additive':'Measured separately; do not add medians to claim total latency',
        'phase_timings':{'standardization_only':None,'nlms_only':None,'features_only':None},
        'memory_peak_bytes':None,'memory_note':'Not measured. 192B filter state is not full-process peak.',
        'nlms_leading_operation_estimate':{'multiplications':18432,'scalar_divisions':256,'additional_gain_multiplications':256,'kind':'arithmetic estimate, not MAC count, excludes normalization/features/mix/backbone'},'records':rows}
    with (V/'synthetic_cost_context.json').open('x') as f:json.dump(report,f,indent=2)
    print(json.dumps(report,indent=2))
