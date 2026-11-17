"""Isolated ledger/snapshot regression. No real approval, preparation or training."""
from pathlib import Path
import hashlib,json,time
import runner
from host_observer import classify,sidecar
def main():
    real_root=runner.ROOT;cfg=runner.read(runner.V/'protocol.json')
    output=real_root/'reports'/('round2_v1_3_SYNTHETIC_'+time.strftime('%Y%m%dT%H%M%SZ',time.gmtime()));output.mkdir(exist_ok=False)
    original=(runner.ROOT,runner.verify,runner.receipt,runner.resource,runner.subprocess.run)
    reached=[]
    def intercept(command,**kwargs):
        assert '--prepare-worker' in command
        reached.append(command);raise RuntimeError('SYNTHETIC_PREPARE_BOUNDARY_NO_CHILD_LAUNCHED')
    results={}
    identity={'status':'running','boot_time':100.,'runner_created':200.,'runner_pid':910001}
    assert classify(identity,100.,200.)=='running_identity_verified'
    assert classify(identity,101.5,200.)=='interrupted_host_boot_changed'
    assert classify(identity,100.,None)=='interrupted_process_missing'
    assert classify(identity,100.,201.)=='interrupted_pid_reused'
    stale=output/'synthetic_stale_launch.json';stale.write_text(json.dumps(identity));original_bytes=stale.read_bytes()
    evidence=sidecar(stale,{'classification':'interrupted_host_boot_changed'})
    assert stale.read_bytes()==original_bytes and evidence.exists()
    results['boot_pid_identity_and_external_terminal_without_original_overwrite']=True
    try:runner.receipt(real_root.parent/'review/round2_v1_1_review_receipt_20261106.json')
    except (AssertionError,KeyError):pass
    else:raise AssertionError('Old receipt accepted')
    results['old_version_review_receipt_rejected']=True
    try:
        runner.verify=lambda:(cfg,{})
        runner.receipt=lambda p:None
        runner.resource=lambda cfg:None
        runner.subprocess.run=intercept
        for status in [None,'running','failed','completed']:
            case=output/('empty' if status is None else status);(case/'logs').mkdir(parents=True);(case/'results').mkdir()
            runner.ROOT=case
            fake=case/'SYNTHETIC_NOT_APPROVAL.json';fake.write_text('{"status":"SYNTHETIC_NOT_AN_APPROVAL"}')
            ledger=case/'logs/round2_v1_3_attempts.json';original_bundle=case/'results/original';(original_bundle/'source').mkdir(parents=True)
            sentinel=original_bundle/'source/KEEP.txt';sentinel.write_text('retained synthetic original')
            if status is not None:runner.write(ledger,[{'bundle':str(original_bundle),'status':status}])
            before=ledger.read_bytes() if ledger.exists() else None;before_source=sentinel.read_bytes();count=len(reached)
            trial=case/'results/new_bundle'
            if status is None:
                assert runner.execute(trial,fake) is False
                assert len(reached)==count+1 and runner.read(ledger)[0]['status']=='failed'
                runner.verify_snapshot(trial)
                saved=runner.read(trial/'snapshot_manifest.json');assert len(saved['files'])==len(runner.read(runner.V/'EXECUTION_FREEZE.json')['files'])
                assert any('synthetic_' in row['path'] for row in saved['files'])
                results['empty_ledger_reaches_prepare_once_full_frozen_snapshot_matches']=True
                # A changed bundle path cannot reuse the same first-run approval after the injected failure.
                before=ledger.read_bytes();alternate=case/'results/changed_path'
                try:runner.execute(alternate,fake)
                except AssertionError:pass
                else:raise AssertionError('Second path bypassed one-attempt ledger')
                assert not alternate.exists() and ledger.read_bytes()==before and len(reached)==count+1
                results['failed_first_attempt_changed_path_rejected_ledger_preserved']=True
            else:
                for name in ['new_bundle','changed_path']:
                    candidate=case/'results'/name
                    try:runner.execute(candidate,fake)
                    except AssertionError:pass
                    else:raise AssertionError('Nonempty ledger accepted')
                    assert not candidate.exists() and ledger.read_bytes()==before and len(reached)==count
                results[status+'_ledger_rejected_all_paths_without_overwrite']=True
            assert sentinel.read_bytes()==before_source
    finally:
        runner.ROOT,runner.verify,runner.receipt,runner.resource,runner.subprocess.run=original
    report={'status':'passed','scope':'ISOLATED_SYNTHETIC_CONTROL_FLOW_NOT_AN_APPROVAL','execution_freeze_sha256':runner.sha(runner.V/'EXECUTION_FREEZE.json'),
            'checks':results,'prepare_subprocesses_started':0,'real_frontend_caches_created':0,'real_training_started':False,
            'test_substitutions':'Only verify/receipt/resource preconditions and subprocess boundary replaced. Actual execute, Windows Lock, ledger checks/writes, copytree and snapshot hash verification exercised.',
            'output':str(output),'test_script_sha256':runner.sha(__file__)}
    runner.write(output/'checks.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
