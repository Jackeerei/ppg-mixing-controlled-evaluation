"""Original e655 worker, unchanged math/checkpoint identity, explicit I/O adapter."""
from pathlib import Path
import argparse,hashlib,json,os,runpy,sys
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def main():
    p=argparse.ArgumentParser();p.add_argument('--bundle','--worker',dest='bundle',required=True);p.add_argument('--index',required=True,type=int);p.add_argument('--owner',required=True);p.add_argument('--outer',action='store_true');p.add_argument('--collect',action='store_true');a=p.parse_args()
    manifest=json.loads((HERE/'FREEZE.json').read_text())
    for r in manifest['files']:assert hashlib.sha256((ROOT.parent/r['path']).read_bytes()).hexdigest()==r['sha256']
    owner=json.loads(Path(a.owner).read_text());assert owner['armed'] and owner['source_bundle']==a.bundle
    source=ROOT/'candidate_versions/supplemental_cv_loss_v1_1';assert hashlib.sha256((source/'FREEZE.json').read_bytes()).hexdigest()==owner['worker_math_freeze_sha256']
    cfg=json.loads((source/'config.json').read_text())
    from io_guard import install
    install() # Adapter records bounded OS retry only, no numeric/env/RNG changes.
    import time
    from active_clock import seconds
    time.monotonic=seconds # Awake ACTIVE budget; original past counters remain unchanged.
    sys.path.insert(0,str(source))
    if a.collect:
        assert owner['phase']=='collect' and all(v=='completed' for v in owner['outer_status'].values())
        import importlib.util
        spec=importlib.util.spec_from_file_location('source_collector',source/'run.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        from io_guard import atomic
        module.v1.put=lambda p,v,mode='x':atomic(Path(p),v);module.v1.collect(Path(a.bundle),cfg);return
    identity=cfg['runs'][a.index]['id'];assert owner['outer_status' if a.outer else 'arm_status'][identity]=='running'
    import resumable,copy
    OriginalTrainer=resumable.Trainer
    class RuntimeBudgetTrainer(OriginalTrainer):
        def __init__(self,*args,**kwargs):
            super().__init__(*args,**kwargs) # Original identity/options restore model/Adam/RNG/BN/cursor first.
            original_options=copy.deepcopy(self.options);self.options=copy.deepcopy(self.options)
            current=json.loads(Path(a.owner).read_text());remaining=current['global_limit_seconds']-current['global_active_seconds'];assert remaining>0
            self.options['active_budget_seconds']=self.baseline_active+remaining
            from io_guard import atomic
            atomic(self.arm/'runtime_budget_policy_override.json',{'source_identity_options_preserved':original_options,'runtime_options':self.options,'owner_freeze_sha256':current['owner_freeze_sha256'],'policy':'User explicitly cancels per-arm2h; original epoch/patience/optimizer unchanged; authorized cumulative global cap read from owner ledger','restored_updates':self.state['updates'],'restored_epoch':self.state['epoch'],'restored_next_batch':self.state['next_batch'],'restored_active_seconds':self.baseline_active,'resume_info':self.resume_info})
            self._recovery_global_limit=current['global_limit_seconds'];self._recovery_restored_updates=self.state['updates'];self._recovery_progress_published=False
        def account_active(self):
            super().account_active()
            if not getattr(self,'_pressure_saving',False):
                import ctypes
                from ctypes import wintypes
                api=ctypes.WinDLL('kernel32').GetFileAttributesW
                api.argtypes=[wintypes.LPCWSTR];api.restype=wintypes.DWORD
                if api(str(Path(a.owner).parent/'resource_pause.request'))!=0xffffffff:
                    self._pressure_saving=True
                    self.save() # Full model/Adam/BN/RNG/cursor at the optimizer boundary.
                    raise RuntimeError('RESOURCE_PAUSE_CHECKPOINTED')

            if hasattr(self,'_recovery_restored_updates') and not self._recovery_progress_published and self.state['updates']>self._recovery_restored_updates:
                from io_guard import atomic
                proof=Path(a.owner).parent/(identity+'.ACTUAL_UPDATE.json')
                update_proof={'pid':os.getpid(),'id':identity,'runtime_global_limit_seconds':self._recovery_global_limit,
                       'restored_updates':self._recovery_restored_updates,'actual_updates':self.state['updates'],
                       'epoch':self.state['epoch'],'next_batch':self.state['next_batch'],'resume_info':self.resume_info,
                       'original_math_freeze':'e655f4d594264a46a76befc06ecfe4a3fac1e745f62b8ec9d1854674d7d1fb8d'}
                if not proof.exists():atomic(proof,update_proof)
                import uuid
                atomic(proof.with_name(identity+'.ACTUAL_UPDATE.'+str(os.getpid())+'.'+uuid.uuid4().hex+'.json'),update_proof)
                self._recovery_progress_published=True
    resumable.Trainer=RuntimeBudgetTrainer
    sys.argv=[str(source/'run.py'),'--worker',a.bundle,'--index',str(a.index)]+(['--outer'] if a.outer else [])
    # Load the same frozen source without its CLI, then wrap only startup diagnostics.
    namespace=runpy.run_path(str(source/'run.py'),run_name='delegated_frozen_source')
    original_runtime=namespace['runtime']
    from resources import snapshot
    from io_guard import atomic
    report=Path(a.owner).parent/(identity+'.startup_resources.json')
    evidence={'before_torch_runtime':snapshot(ROOT),'trainer_entered':False}
    atomic(report,evidence)
    original_init=RuntimeBudgetTrainer.__init__
    def diagnosed_init(self,*args,**kwargs):
        evidence['trainer_entered']=True;atomic(report,evidence)
        original_init(self,*args,**kwargs)
        evidence['restored_updates']=self.state['updates'];atomic(report,evidence)
    RuntimeBudgetTrainer.__init__=diagnosed_init
    def diagnosed_runtime():
        try:
            torch=original_runtime()
            evidence['after_torch_runtime']=snapshot(ROOT)
            evidence['cuda_available']=bool(torch.cuda.is_available())
            atomic(report,evidence);return torch
        except BaseException as exc:
            evidence['after_runtime_failure']=snapshot(ROOT)
            evidence['exception']=repr(exc)
            try:
                import torch
                evidence['cuda_available']=bool(torch.cuda.is_available())
            except BaseException as second:evidence['cuda_probe_exception']=repr(second)
            atomic(report,evidence);raise
    namespace['worker'].__globals__['runtime']=diagnosed_runtime
    namespace['worker'](Path(a.bundle).resolve(),a.index,a.outer)
if __name__=='__main__':main()
