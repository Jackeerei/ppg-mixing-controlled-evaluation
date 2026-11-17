"""Optimizer-boundary continuation with persisted RNG/cursor/stopping and active time."""
from pathlib import Path
import copy,importlib.util,json,math,os,platform,random,subprocess,sys,time
import numpy as np
import torch
from checkpoints import Store,atomic_json,sha
SOURCE=Path(__file__).resolve().parent.parent/'supplemental_cv_loss_v1/engine.py'
spec=importlib.util.spec_from_file_location('supplemental_original_core',SOURCE);core=importlib.util.module_from_spec(spec);spec.loader.exec_module(core)
def rng_state():
    return {'python':random.getstate(),'numpy':np.random.get_state(),'torch_cpu':torch.get_rng_state(),'torch_cuda':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}
def restore_rng(r):
    random.setstate(r['python']);np.random.set_state(r['numpy']);torch.set_rng_state(r['torch_cpu'].cpu())
    if r['torch_cuda']:torch.cuda.set_rng_state_all([s.cpu() for s in r['torch_cuda']])
def fingerprint(device):
    value={'python':sys.version,'numpy':np.__version__,'torch':torch.__version__,'cuda_build':torch.version.cuda,'device':str(device),'gpu':torch.cuda.get_device_name(device) if str(device).startswith('cuda') else None,'deterministic':torch.are_deterministic_algorithms_enabled(),'cudnn_deterministic':torch.backends.cudnn.deterministic,'cudnn_benchmark':torch.backends.cudnn.benchmark,'matmul_tf32':torch.backends.cuda.matmul.allow_tf32,'cudnn_tf32':torch.backends.cudnn.allow_tf32,'cpu_threads':torch.get_num_threads(),'CUBLAS_WORKSPACE_CONFIG':os.environ.get('CUBLAS_WORKSPACE_CONFIG'),'CUDA_VISIBLE_DEVICES':os.environ.get('CUDA_VISIBLE_DEVICES'),'platform':platform.platform()}
    if str(device).startswith('cuda'):
        value['driver']=subprocess.run(['nvidia-smi','--query-gpu=driver_version','--format=csv,noheader'],capture_output=True,text=True,check=True,timeout=20).stdout.strip()
    return value
def clone_state(model):return {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
class Trainer:
    def __init__(self,arm,model,optimizer,train,inner,options,identity,device='cpu',checkpoint_updates=100,checkpoint_seconds=60):
        self.arm=Path(arm);self.model=model;self.optimizer=optimizer;self.train=train;self.inner=inner;self.options=options;self.device=device
        self.identity={**identity,'options':options,'environment':fingerprint(device),'prepared_tensor_sha256':{role:{k:core.array_sha(d[k]) for k in ['x','d','f','y','subject','activity','start']} for role,d in [('train',train),('inner',inner)]}}
        self.store=Store(self.arm/'checkpoints',self.identity);self.account=self.arm/'active_accounting.json';self.every=checkpoint_updates;self.seconds=checkpoint_seconds;self.resume_info=None
        if self.store.index.exists():
            self.state,self.resume_info=self.store.load();self.model.load_state_dict(self.state['model'],strict=True);self.optimizer.load_state_dict(self.state['optimizer']);self.model.train(self.state.get('model_training',True));restore_rng(self.state['rng'])
        else:
            assert not (self.arm/'arm_intent.json').exists(),'No complete checkpoint for already-started arm'
            atomic_json(self.arm/'arm_intent.json',{'identity':self.identity,'status':'starting'})
            self.state={'phase':'train','epoch':0,'next_batch':0,'updates':0,'loss_sum':0.,'seen':0,'best_inner_mae':math.inf,'best_update':None,'best_model':None,'stale':0,'history':[],'batch_order_hashes':[],'active_seconds':0.,'model':clone_state(model),'optimizer':copy.deepcopy(optimizer.state_dict()),'rng':rng_state()}
            self.store.save(self.state)
        if self.account.exists():
            a=json.loads(self.account.read_text(encoding='utf-8'));assert a['identity']==self.identity
            self.state['active_seconds']=max(self.state['active_seconds'],a['active_seconds'])
        self.baseline_active=self.state['active_seconds'];self.started=time.monotonic();self.last_save=time.monotonic();self.last_save_update=self.state['updates']
    def active(self):return self.baseline_active+(time.monotonic()-self.started)
    def account_active(self):
        atomic_json(self.account,{'identity':self.identity,'active_seconds':self.active(),'updates_observed':self.state['updates']})
    def save(self):
        self.account_active();self.state.update(model=clone_state(self.model),model_training=self.model.training,optimizer=copy.deepcopy(self.optimizer.state_dict()),rng=rng_state(),active_seconds=self.active())
        self.store.save(self.state);self.last_save=time.monotonic();self.last_save_update=self.state['updates']
    def advance(self,stop_after_updates=None):
        s=self.state;n=len(self.train['y']);batch=self.options['batch_size'];seed=self.identity['seed']
        while s['phase']!='complete':
            if self.active()>=self.options['active_budget_seconds']:self.save();raise RuntimeError('Cumulative active per-arm budget exhausted')
            if s['phase']=='train':
                order=core.epoch_order(n,seed,s['epoch']);parts=core.batch_indices(order,batch)
                if len(s['batch_order_hashes'])==s['epoch']:s['batch_order_hashes'].append(core.array_sha(order))
                assert s['batch_order_hashes'][s['epoch']]==core.array_sha(order)
                while s['next_batch']<len(parts):
                    ids=parts[s['next_batch']];self.model.train();self.optimizer.zero_grad(set_to_none=True)
                    args=[torch.from_numpy(self.train[k][ids]).to(self.device) for k in ['x','d','f']];target=torch.from_numpy(self.train['y'][ids,None]).to(self.device)
                    prediction,_=self.model(*args);assert prediction.shape==target.shape
                    loss=core.objective(prediction-target,self.options['objective']);assert torch.isfinite(loss)
                    loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in self.model.parameters())
                    assert all(p.grad is not None for name,p in self.model.named_parameters() if not name.startswith('backbone.'))
                    self.optimizer.step();assert all(torch.isfinite(p).all() for p in self.model.parameters())
                    s['updates']+=1;s['next_batch']+=1;s['seen']+=len(ids);s['loss_sum']+=loss.item()*len(ids)
                    self.account_active() # Persist active time even for replayed work after last full checkpoint.
                    if s['updates']-self.last_save_update>=self.every or time.monotonic()-self.last_save>=self.seconds:self.save()
                    if self.active()>=self.options['active_budget_seconds']:self.save();raise RuntimeError('Cumulative active per-arm budget exhausted')
                    if stop_after_updates is not None and s['updates']>=stop_after_updates:self.save();return 'paused_for_test'
                assert s['seen']==n;s['phase']='inner_validation';self.save()
            if s['phase']=='inner_validation':
                scores,_,_=core.evaluate(self.model,self.inner,self.device,batch);metric=scores['subject_macro_mae'];assert math.isfinite(metric)
                s['history'].append({'epoch':s['epoch']+1,'updates':s['updates'],'train_objective':s['loss_sum']/s['seen'],'objective':self.options['objective'],'inner_equal_person_mae':metric,'scores':scores})
                if metric<s['best_inner_mae']:s.update(best_inner_mae=metric,best_update=s['updates'],best_model=clone_state(self.model),stale=0)
                else:s['stale']+=1
                s['epoch']+=1;s.update(next_batch=0,seen=0,loss_sum=0.)
                if s['epoch']>=self.options['max_epochs'] or s['stale']>=self.options['patience']:
                    s.update(phase='complete',stop_reason='inner_early_stop' if s['stale']>=self.options['patience'] else 'max_epochs')
                else:s['phase']='train'
                self.save()
        self.save();return 'complete'
