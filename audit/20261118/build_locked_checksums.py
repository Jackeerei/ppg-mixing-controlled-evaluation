"""Extract hashes and deidentified archival metadata only. No torch/NumPy imports."""
from pathlib import Path
import json
from verify_locked_archives import sha,metadata,verify
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def read(p):return json.loads(p.read_text(encoding='utf-8'))
specs=[('locked_final','results/final_evaluation_v1_20261107T132631Z','reports/final_preparation_20261107T111259Z/locked_evaluation','PREPARATION_MANIFEST.json',
 ['subject','window_id','start_sample_32hz','stop_sample_32hz','reference_label_index','activity']),
 ('external_wrist','results/wrist_external_v1_20261107T074907Z','reports/wrist_external_prepare_20261107T025937Z','MANIFEST.json',
 ['record','subject','activity','window_id','start_sample_256hz','stop_sample_256hz'])]
scopes=[];binding_checks=[]
for name,bundle_rel,prep_rel,prep_manifest_name,fields in specs:
 bundle=ROOT/bundle_rel;prep=ROOT/prep_rel
 run_manifest=read(bundle/'MANIFEST.json');prep_manifest=read(prep/prep_manifest_name)
 run_index={r['path']:r for r in run_manifest['files']};prep_index={r['path']:r for r in prep_manifest['files']}
 artifacts=[]
 def add(p,role,expected=None):
  actual=sha(p)
  if expected:assert actual==expected,p.name
  row={'relative_path':p.relative_to(ROOT).as_posix(),'role':role,'bytes':p.stat().st_size,'sha256':actual}
  if expected:row['matches_archived_manifest_sha256']=True
  artifacts.append(row);return actual
 run_sha=add(bundle/'MANIFEST.json','archived_fixed_output_manifest')
 prep_sha=add(prep/prep_manifest_name,'archived_candidate_source_manifest')
 candidate_path=prep/'candidate_windows.csv';add(candidate_path,'candidate_manifest',prep_index['candidate_windows.csv']['sha256'])
 candidate=metadata(candidate_path,fields)
 intent=read(bundle/'intent.json');bound=intent.get('preparation_manifest_sha256',intent.get('prepared_manifest_sha256'));assert bound==prep_sha
 add(bundle/'intent.json','archived_execution_intent',run_index.get('intent.json',{}).get('sha256'))
 for p in (bundle/'models.json',prep/'models.json',prep/'config.json'):
  parent_index=run_index if p.parent==bundle else prep_index
  add(p,'archived_model_or_protocol_binding',parent_index.get(p.name,{}).get('sha256'))
 arms=[]
 for arm in sorted(bundle.iterdir()):
  if not arm.is_dir() or not (arm/'predictions.csv').is_file():continue
  rel=(arm/'predictions.csv').relative_to(bundle).as_posix();entry=run_index.get(rel,run_index.get(rel.replace('/','\\')));assert entry
  add(arm/'predictions.csv','fixed_predictions_bytes_only',entry['sha256'])
  sr=(arm/'summary.json').relative_to(bundle).as_posix();entry=run_index.get(sr,run_index.get(sr.replace('/','\\')));assert entry
  add(arm/'summary.json','fixed_output_summary_bytes_only',entry['sha256'])
  output=metadata(arm/'predictions.csv',fields,True);summary=read(arm/'summary.json')
  for field,target in [('candidate_windows','candidate_rows'),('eligible_windows','eligible_rows'),('prediction_success','success_rows')]:assert summary[field]==output[target],(arm.name,field)
  info={'arm_id':arm.name,'method':summary['method'],'seed':summary['seed'],'prediction_relative_path':(arm/'predictions.csv').relative_to(ROOT).as_posix(),'metadata':output}
  for k in ('set','training_device','inference_device'):
   if k in summary:info[k]=summary[k]
  arms.append(info)
 assert len(arms)==(21 if name=='locked_final' else 15)
 scopes.append({'scope_id':name,'run_manifest_sha256':run_sha,'preparation_manifest_sha256':prep_sha,'preparation_hash_matches_execution_intent':True,
  'candidate_relative_path':candidate_path.relative_to(ROOT).as_posix(),'identity_fields':fields,'candidate_metadata':candidate,'arms':arms,'artifacts':artifacts})
manifest={'version':1,'release_scope':'hashes and deidentified aggregate identity/coverage metadata only; not publicly uploaded',
 'canonicalization':'identity tuples retain exact CSV text; availability flags normalize 1/true and0/false; list of tuples sorted lexicographically; ensure_ascii=True compact JSON UTF8 plus single LF; SHA256',
 'explicitly_excluded_columns':['reference_bpm','prediction_bpm','filtered_weight_alpha','r_peak_count','first_r_sample','last_r_sample'],
 'scopes':scopes}
path=HERE/'LOCKED_ARCHIVE_CHECKSUMS.json';path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
result=verify(ROOT,manifest);result['manifest_sha256']=sha(path)
(HERE/'CHECKS.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
print(json.dumps(result))
