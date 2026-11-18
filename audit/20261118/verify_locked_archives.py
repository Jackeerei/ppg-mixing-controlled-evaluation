"""Independent stdlib-only byte/identity/coverage verifier, never model evaluation."""
from pathlib import Path
import argparse,collections,csv,hashlib,json

def sha(path):
 with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
def truth(value):
 if value.lower() in ('1','true'):return True
 if value.lower() in ('0','false'):return False
 raise ValueError('Noncanonical availability flag')
def digest(rows):
 value=json.dumps(sorted(rows),ensure_ascii=True,separators=(',',':'))+'\n'
 return hashlib.sha256(value.encode('utf-8')).hexdigest()
def metadata(path,identity_fields,predictions=False):
 """Only named identity/flags/status columns are used; no numerical output metrics."""
 identities=[];eligible=[];successful=[];availability=[];statuses=collections.Counter();subjects=set();records=set()
 with path.open(newline='',encoding='utf-8-sig') as f:
  for row in csv.DictReader(f):
   key=[row[field] for field in identity_fields]
   assert all(v!='' for v in key),'Missing window identity'
   identities.append(key);subjects.add(row['subject'])
   if 'record' in row:records.add(row['record'])
   flags=[truth(row[k]) for k in ('input_available','reference_available','eligible')]
   assert flags[2]==(flags[0] and flags[1]),'Eligibility differs from availability intersection'
   availability.append(key+flags)
   if flags[2]:eligible.append(key)
   if predictions:
    status=row['prediction_status'];statuses[status]+=1
    if status=='ok':
     assert flags[2],'Prediction marked ok for excluded window'
     successful.append(key)
    else:assert not flags[2] and status=='excluded','Unexpected eligible inference failure/status'
 assert len(set(map(tuple,identities)))==len(identities),'Duplicate window identity'
 result={'candidate_rows':len(identities),'eligible_rows':len(eligible),'excluded_rows':len(identities)-len(eligible),
  'identity_sha256':digest(identities),'eligible_identity_sha256':digest(eligible),'availability_sha256':digest(availability),
  'subject_ids':sorted(subjects),'record_count':len(records)}
 if predictions:result.update(success_rows=len(successful),successful_identity_sha256=digest(successful),status_counts=dict(statuses))
 return result
def safe(root,rel):
 p=(root/rel).resolve();assert p.is_relative_to(root.resolve()),'Archive path escapes supplied root';return p
def verify(root,manifest):
 checked=0
 for scope in manifest['scopes']:
  by_role={a['role']:a for a in scope['artifacts']}
  run_path=safe(root,by_role['archived_fixed_output_manifest']['relative_path'])
  prep_path=safe(root,by_role['archived_candidate_source_manifest']['relative_path'])
  run_index={r['path'].replace('\\','/'):r['sha256'] for r in json.loads(run_path.read_text())['files']}
  prep_index={r['path'].replace('\\','/'):r['sha256'] for r in json.loads(prep_path.read_text())['files']}
  for a in scope['artifacts']:
   p=safe(root,a['relative_path']);assert p.stat().st_size==a['bytes'] and sha(p)==a['sha256'],a['relative_path'];checked+=1
   if a.get('matches_archived_manifest_sha256'):
    index=run_index if p.is_relative_to(run_path.parent) else prep_index
    parent=run_path.parent if index is run_index else prep_path.parent
    assert index[p.relative_to(parent).as_posix()]==a['sha256'],'Historical archive binding mismatch'
  intent=json.loads(safe(root,by_role['archived_execution_intent']['relative_path']).read_text())
  assert intent.get('preparation_manifest_sha256',intent.get('prepared_manifest_sha256'))==sha(prep_path)
  fields=scope['identity_fields'];candidate=metadata(safe(root,scope['candidate_relative_path']),fields)
  assert candidate==scope['candidate_metadata'],'Candidate manifest metadata changed'
  for arm in scope['arms']:
   actual=metadata(safe(root,arm['prediction_relative_path']),fields,True)
   assert actual==arm['metadata'],arm['arm_id']
   for k in ('candidate_rows','eligible_rows','excluded_rows','identity_sha256','eligible_identity_sha256','availability_sha256'):
    assert actual[k]==candidate[k],(arm['arm_id'],k)
   assert actual['success_rows']==candidate['eligible_rows'] and actual['successful_identity_sha256']==candidate['eligible_identity_sha256']
 return {'scopes':len(manifest['scopes']),'artifacts_verified':checked,'arms_verified':sum(len(s['arms']) for s in manifest['scopes']),
  'scope':'archived bytes and identity/coverage only; no inference, signal decoding or outcome metric recalculation'}
def main():
 p=argparse.ArgumentParser();p.add_argument('--archive-root',type=Path,required=True);p.add_argument('--manifest',type=Path,default=Path(__file__).with_name('LOCKED_ARCHIVE_CHECKSUMS.json'));p.add_argument('--report',type=Path);a=p.parse_args()
 result=verify(a.archive_root,json.loads(a.manifest.read_text(encoding='utf-8')))
 if a.report:a.report.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(result))
if __name__=='__main__':main()
