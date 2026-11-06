"""Reviewed C-route programs using the existing SDK acceptance and v2 Runner.

This is an artifact/review workflow for trusted code, not an execution sandbox.
Model text, optional edits and executed source retain independent identities.
No test results are exported into development feedback.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time

from .asset_runner import execute_asset_experiment,_plugin_reference
from .models import canonical_json,stable_id
from .run_lifecycle import OutputLease
from .process_environment import reviewed_environment


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def task_ids(spec):
    # Conservative task exposure: budget/seed changes do not make a previously
    # revealed Shape/implementation a fresh blind target. Unspecified hardware
    # stays conservatively unspecified, never guessed from a host path.
    return sorted({stable_id([spec.get('hardware'),spec['operator'],spec.get('dtype','float32'),
        spec.get('parameters',{}),kernel if isinstance(kernel,str) else kernel['asset'],shape])
        for kernel in spec['kernels'] for shape in spec['shapes']})
def write(path,value):
    temporary=path.with_suffix('.tmp');temporary.write_text(canonical_json(value)+'\n');temporary.replace(path)


def submit(directory,*,response,program,algorithm_ref,provenance,edited_program=None,parent=None):
    directory=Path(directory).resolve()
    if directory.exists():raise ValueError('immutable submission already exists; submit a new version')
    if not algorithm_ref or '@' not in algorithm_ref:raise ValueError('versioned SDK algorithm ref required')
    required={'route','provider','model','evidence_level'}
    if not required<=set(provenance) or provenance['route']!='C':raise ValueError('C model/provider/evidence provenance required')
    if edited_program and not provenance.get('edit_description'):raise ValueError('distinguish model original from static/human source edits')
    blobs={name:Path(path).read_bytes() for name,path in
        [('model-response.txt',response),('model-program.py',program),('program.py',edited_program or program)]}
    if any(len(b)>1024**2 for b in blobs.values()):raise ValueError('program artifacts limited to1MiB each')
    compile(blobs['program.py'],'reviewed-program.py','exec') # syntax only; never imports/executes generated code
    inherited=[]
    if parent:
        previous=verify(parent)
        if previous.get('frozen') is None:raise ValueError('parent version must be frozen before iteration')
        inherited=sorted(set(previous.get('inherited_exposed_tasks',[]))|set(previous.get('exposed_tasks',[])))
        parent={'artifact_id':previous['artifact_id'],'source_sha256':previous['hashes']['program.py'],
            'test_exposed':previous['test_exposed']}
    directory.mkdir(parents=True,mode=0o700)
    for name,blob in blobs.items():(directory/name).write_bytes(blob)
    manifest={'schema_version':1,'algorithm_ref':algorithm_ref,'provenance':provenance,'parent':parent,
        'hashes':{name:sha(directory/name) for name in blobs},'created_unix_s':time.time(),
        'review':None,'checks':[],'development':[],'frozen':None,'tests':[],'test_exposed':False,
        'inherited_exposed_tasks':inherited,'exposed_tasks':[]}
    manifest['artifact_id']=stable_id({k:manifest[k] for k in ('algorithm_ref','provenance','parent','hashes')})
    write(directory/'artifact.json',manifest);return manifest


def verify(directory):
    directory=Path(directory);value=json.loads((directory/'artifact.json').read_text())
    if any(sha(directory/name)!=digest for name,digest in value['hashes'].items()):raise ValueError('submitted program/response changed; submit a new version')
    if value.get('frozen') and value['frozen']['source_sha256']!=value['hashes']['program.py']:
        raise ValueError('frozen source identity changed')
    for entry in (*value['checks'],*value['development']):
        reuse=entry.get('reused_evidence')
        if reuse and any(not Path(path).is_file() or sha(path)!=digest for path,digest in reuse['files'].items()):
            raise ValueError('reused evaluation evidence changed or is unavailable')
    return value


def review(directory,reviewed_by):
    directory=Path(directory)
    with OutputLease(directory/'workflow-lock'):
        value=verify(directory)
        if value['frozen']:raise ValueError('frozen review cannot be changed')
        if not reviewed_by:raise ValueError('explicit trusted-code review identity required')
        value['review']={'reviewed_by':reviewed_by,'source_sha256':value['hashes']['program.py'],
            'trust':'reviewed local Python; no sandbox claim','unix_s':time.time()}
        write(directory/'artifact.json',value);return value['review']


def evaluate(directory,spec_path,*,stage):
    if stage not in ('check','development','test'):raise ValueError('stage is check/development/test')
    directory=Path(directory).resolve();spec_path=Path(spec_path).resolve()
    with OutputLease(directory/'workflow-lock'):
        value=verify(directory)
        if not value['review']:raise ValueError('review generated code before executing its SDK plugin')
        from .inference_resources import COORDINATOR
        COORDINATOR.assert_worker_handoff()
        if stage=='test' and not value['frozen']:raise ValueError('freeze the exact program version before test')
        if stage!='test' and (value['frozen'] or value['test_exposed']):raise ValueError('cannot develop an exposed/frozen version; submit a new version with explicit exposure lineage')
        spec=json.loads(spec_path.read_text())
        if spec.get('schema_version')!=2:raise ValueError('use the existing schema_version=2 experiment')
        if stage=='development' and spec.get('split','train') not in ('train','validation'):raise ValueError('development accepts split=train or validation only, never held-out test')
        if stage=='test' and spec.get('split')!='test':raise ValueError('test stage requires an explicitly labelled test spec')
        tasks=task_ids(spec);inherited=set(value.get('inherited_exposed_tasks',[]))
        if stage=='development' and inherited&set(tasks):
            raise ValueError('parent exposed this target; do not turn its test labels into blind-workflow development feedback')
        if stage=='check' and spec.get('split','train')=='test':raise ValueError('do not inspect held-out task in development contract checks')
        if len(spec.get('algorithms',[]))!=1 or spec['algorithms'][0]['asset']!=value['algorithm_ref']:
            raise ValueError('experiment must select exactly the submitted algorithm ref')
        spec['plugins']=[_plugin_reference(p,spec_path.parent) for p in spec.get('plugins',[])]+[str(directory/'program.py')]
        key='checks' if stage=='check' else 'tests' if stage=='test' else 'development'
        attempt=directory/'evaluations'/f'{stage}-{len(value[key]):03d}'
        attempt.mkdir(parents=True);write(attempt/'spec.json',spec)
        # Persist exposure BEFORE a worker can read a target, including failures.
        if stage=='test':
            value['test_exposed']=True;value['exposed_tasks']=sorted(set(value.get('exposed_tasks',[]))|set(tasks))
        developed={task for run in (*value['checks'],*value['development']) for task in run.get('task_ids',[])}
        entry={'spec_sha256':sha(attempt/'spec.json'),'stage':stage,'output':str(attempt/'run'),'status':'running','task_ids':tasks,
            'target_evidence_scope':'previously_exposed_or_developed_task_retest' if stage=='test' and (inherited|developed)&set(tasks)
                else 'frozen_target_evaluation; external split/provenance audit still required' if stage=='test' else 'development_only'}
        value[key].append(entry);write(directory/'artifact.json',value)
        result=execute_asset_experiment(attempt/'spec.json',attempt/'run',check_only=stage=='check',worker_environment=reviewed_environment())
        entry.update(status=result['status'],exit_code=result.get('exit_code'),summary_sha256=sha(attempt/'run/run_summary.json'))
        write(directory/'artifact.json',value)
        return result


def reuse_evaluation(directory,source,spec_path,*,stage):
    """Attach the frozen parent's identical historical evaluation without execution.

    This attests prior evidence, never the current machine or environment. Both
    exact executed source and complete experiment specification must match.
    """
    if stage not in ('check','development'):raise ValueError('only check/development evidence can be reused')
    directory=Path(directory).resolve();source=Path(source).resolve();spec_path=Path(spec_path).resolve()
    if directory==source:raise ValueError('reuse requires a separate child artifact')
    with OutputLease(directory/'workflow-lock'):
        value=verify(directory);previous=verify(source)
        if not value['review'] or value['frozen'] or value['test_exposed']:
            raise ValueError('reuse requires a reviewed, unexposed, unfrozen child')
        if not previous['frozen'] or not value.get('parent') or value['parent']['artifact_id']!=previous['artifact_id']:
            raise ValueError('reuse requires the explicitly frozen parent artifact')
        if value['hashes']['program.py']!=previous['hashes']['program.py'] or value['algorithm_ref']!=previous['algorithm_ref']:
            raise ValueError('reuse requires identical executed program and algorithm ref')
        spec=json.loads(spec_path.read_text());tasks=task_ids(spec)
        if spec.get('split','train') not in ('train','validation') or set(tasks)&set(value.get('inherited_exposed_tasks',[])):
            raise ValueError('test or previously exposed tasks cannot be reused as development')
        def normalized(spec,artifact,*,submitted=False):
            spec=dict(spec)
            plugins=[_plugin_reference(p,spec_path.parent if submitted else artifact) for p in spec.get('plugins',[])]
            own=str(artifact/'program.py')
            if submitted:plugins.append(own)
            spec['plugins']=['<reviewed-identical-program>' if p==own else p for p in plugins]
            return spec
        key='checks' if stage=='check' else 'development'
        if value[key]:raise ValueError('reuse must not replace or duplicate an existing evaluation')
        wanted=normalized(spec,directory,submitted=True);entry=None;attempt_spec=None
        for candidate in reversed(previous[key]):
            if candidate['status']!='passed':continue
            run=Path(candidate['output']);file=run.parent/'spec.json'
            if not file.is_file() or sha(file)!=candidate.get('spec_sha256'):
                raise ValueError('parent evaluation specification identity mismatch')
            if file.is_file() and normalized(json.loads(file.read_text()),source)==wanted:
                entry=candidate;attempt_spec=file;break
        if entry is None:raise ValueError('no passed parent evaluation with identical full specification')
        run=Path(entry['output']);summary=run/'run_summary.json'
        if sha(summary)!=entry['summary_sha256'] or json.loads(summary.read_text())['status']!='passed':
            raise ValueError('parent evaluation summary identity/status mismatch')
        files=[source/'artifact.json',attempt_spec,summary]
        files.extend(file for file in (run/'measurements.sqlite',run/'trials.json') if file.is_file())
        reused={'source_artifact_id':previous['artifact_id'],'source_program_sha256':previous['hashes']['program.py'],
            'files':{str(file):sha(file) for file in files},'new_physical_measurements':0,
            'scope':'historical parent evaluation only; no current hardware/environment verification',
            'unix_s':time.time()}
        value[key].append({**entry,'reused_evidence':reused})
        write(directory/'artifact.json',value)
        return {'status':'passed','stage':stage,'evidence_kind':'historical_evaluation_reuse',
            'new_physical_measurements':0,'source_artifact_id':previous['artifact_id']}


def feedback(directory):
    directory=Path(directory);value=verify(directory)
    rows=[]
    # Deliberately no test paths, outputs, latencies or success labels here.
    for item in value['development']:
        path=Path(item['output']);trials=json.loads((path/'trials.json').read_text())
        for trial in trials:
            rows.append({key:trial.get(key) for key in ('shape','kernel_ref','best_latency_ms','evaluations','cache_hits','stop_reason','status')})
    return {'route':'C','development_feedback':{'artifact_id':value['artifact_id'],
        'test_labels_read':False,'development_only':True,'observations':rows,
        'notice':'these are development summaries, not an oracle for new test tasks'},
        'previous_program':(directory/'program.py').read_text()}


def freeze(directory):
    directory=Path(directory)
    with OutputLease(directory/'workflow-lock'):
        value=verify(directory)
        if value['frozen']:return value['frozen']
        if not value['review'] or not any(c['status']=='passed' for c in value['checks']) or not any(c['status']=='passed' for c in value['development']):
            raise ValueError('freeze requires reviewed source, passed SDK check and development evaluation')
        value['frozen']={'source_sha256':value['hashes']['program.py'],'artifact_id':value['artifact_id'],
            'unix_s':time.time(),'development_evaluations':len(value['development']),
            'notice':'frozen version; future target exposure cannot be called unseen development'}
        write(directory/'artifact.json',value);return value['frozen']


def generate(client,task,*,max_output_tokens=2048):
    """API or an injected protocol fixture; authorization remains in client."""
    schema={'type':'object','properties':{name:{'type':'string'} for name in ('program','algorithm_ref','rationale')},
        'required':['program','algorithm_ref','rationale'],'additionalProperties':False}
    return client.invoke(task,route='C',max_output_tokens=max_output_tokens,schema=schema,
        system='Generate a complete SDK Searcher program. Use only the public search interface, space queries and revealed/development feedback supplied. Return JSON containing program, algorithm_ref, and rationale. Do not read files, credentials, external labels or invoke a measurement engine. Source will be reviewed before execution.')
