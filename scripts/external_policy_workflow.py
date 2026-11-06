"""Review/freeze external programs with the existing SDK and PolicyRepository.

This is a manual external coding-agent workflow, not a code sandbox or training
service. Feedback refuses test jobs before opening any measurement journal.
"""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

from analyze_phase1 import current_records
from kernel_tuning_lab.models import ShapeSpec, canonical_json
from kernel_tuning_lab.policies import PolicyRepository
from kernel_tuning_lab.research import read_json, snapshot_directory
from kernel_tuning_lab.storage import ExperimentStore


def export_feedback(study, protocol, destination):
    study=Path(study);protocol=read_json(protocol);manifest=read_json(study/'research_manifest.json')
    allowed={'train':protocol['train_shapes'],'validation':protocol['validation_shapes']}
    # Check the public job roster BEFORE reading even one record/latency.
    for job in manifest['jobs']:
        split=job['experiment']['split']
        if split not in allowed or any(s not in allowed[split] for s in job['shapes']):
            raise ValueError('feedback may contain only predeclared train/validation jobs; test journal was not opened')
    if manifest.get('status')!='finished':
        raise ValueError('feedback requires a finished immutable attempt snapshot')
    # Validate indexed row metadata before deserializing any latency payload.
    # This also catches a stale/tampered trial pointing to an unrelated test ID.
    for job in manifest['jobs']:
        snapshot=snapshot_directory(study,job)
        if snapshot is None or not (snapshot/'trials.json').exists():continue
        allowed_ids={ShapeSpec(s).id for s in job['shapes']}
        with ExperimentStore(study/job['output']/'measurements.sqlite',read_only=True) as store:
            for trial in read_json(snapshot/'trials.json'):
                for rid in trial['record_ids']:
                    row=store.connection.execute('SELECT shape_id,split FROM measurements WHERE id=?',(rid,)).fetchone()
                    if row is None or row[0] not in allowed_ids or row[1]!=job['experiment']['split']:
                        raise ValueError('record metadata outside the declared development split; latency payload not opened')
    rows=current_records(study)
    result={'schema_version':1,'allowed_splits':['train','validation'],
        'test_labels_read':False,'record_ids':[r.id for r in rows],
        'observations':[{'id':r.id,'shape':dict(r.shape.dims),'split':r.split,'kernel':r.operator.implementation,
            'config':dict(r.config.params),'seed':r.seed,'status':r.status.value,
            'latency_ms':r.median_ms if r.correctness_verified else None,
            'reference_verified':r.correctness_verified,'verified_on_gpu':r.metadata.get('correctness',{}).get('verified_on_gpu'),
            'failure_reason':r.failure_reason} for r in rows],
        'notice':'Only actually revealed development observations. No oracle, test table/path, or missing-label predictions.'}
    blob=(json.dumps(result,indent=2,allow_nan=False)+'\n').encode();Path(destination).write_bytes(blob)
    return {'status':'development_feedback_exported','records':len(rows),'sha256':hashlib.sha256(blob).hexdigest()}


def freeze_program(program, repository, tag, feedback=None):
    path=Path(program).resolve();module_name='reviewed_policy_'+hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    spec=importlib.util.spec_from_file_location(module_name,path);module=importlib.util.module_from_spec(spec)
    sys.modules[module_name]=module;spec.loader.exec_module(module)
    metadata={'frozen_before_test':True,'review_required':'trusted reviewed Python, no malicious-code sandbox',
              'development_splits':[],'feedback_sha256':None}
    if feedback is not None:
        content=read_json(feedback)
        if content.get('test_labels_read') is not False or set(content['allowed_splits'])!={'train','validation'}:
            raise ValueError('feedback is not a train/validation-only contract')
        if any(r['split'] not in ('train','validation') for r in content['observations']):
            raise ValueError('test observations cannot enter development feedback')
        metadata.update(development_splits=['train','validation'],
            feedback_sha256=hashlib.sha256(Path(feedback).read_bytes()).hexdigest(),
            development_record_ids=content['record_ids'])
    manifest=PolicyRepository(repository).register(module.create_policy(None,seed=19),tag,metadata=metadata)
    return {'status':'program_frozen','source_sha256':manifest['source_sha256'],
            'artifact_id':manifest['artifact_id'],'name':manifest['name'],'tag':tag}


def make_test_spec(protocol, template, manifests, destination):
    protocol=read_json(protocol);value=read_json(template);sources=[];assets=[]
    for item in manifests:
        path=Path(item);manifest=read_json(path);source=path.with_name('source.py')
        if not manifest['metadata'].get('frozen_before_test'):
            raise ValueError('test requires an explicitly frozen policy version')
        if hashlib.sha256(source.read_bytes()).hexdigest()!=manifest['source_sha256']:
            raise ValueError('frozen program source changed')
        sources.append(str(source.resolve()))
        # Load only reviewed frozen code; no data table is passed to its factory.
        module_name='frozen_'+manifest['source_sha256'][:12];spec=importlib.util.spec_from_file_location(module_name,source)
        module=importlib.util.module_from_spec(spec);sys.modules[module_name]=module;spec.loader.exec_module(module)
        assets.extend(a.ref for a in module.ASSETS)
    if len(set(assets))!=len(assets):raise ValueError('frozen versions need distinct SDK asset identities')
    value.update(name='external-reviewed-programs-frozen-test-v1',plugins=['kernel_tuning_lab.production.marlin',*sources],
        algorithms=assets,seeds=[protocol['input_seed']],
        budgets=[{'max_evaluations':protocol['budget'],'max_wall_time_s':180}],
        cases=[{'name':'frozen-test','operator':protocol['operator'],'kernels':[protocol['kernel']],
                'shapes':protocol['test_shapes'],'split':'test',
                'validation':{'max_configs':len(protocol['test_shapes'])}}])
    Path(destination).write_text(json.dumps(value,indent=2)+'\n')
    return {'status':'test_spec_exported_not_executed','frozen_versions':len(sources),
            'test_labels_read':False,'spec_sha256':hashlib.sha256(canonical_json(value).encode()).hexdigest()}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);commands=p.add_subparsers(dest='command',required=True)
    a=commands.add_parser('feedback');a.add_argument('--study',required=True);a.add_argument('--protocol',required=True);a.add_argument('--output',required=True)
    a=commands.add_parser('freeze');a.add_argument('--program',required=True);a.add_argument('--repository',required=True);a.add_argument('--tag',required=True);a.add_argument('--feedback')
    a=commands.add_parser('test-spec');a.add_argument('--protocol',required=True);a.add_argument('--template',required=True);a.add_argument('--manifest',action='append',required=True);a.add_argument('--output',required=True)
    args=p.parse_args()
    if args.command=='feedback':result=export_feedback(args.study,args.protocol,args.output)
    elif args.command=='freeze':result=freeze_program(args.program,args.repository,args.tag,args.feedback)
    else:result=make_test_spec(args.protocol,args.template,args.manifest,args.output)
    print(json.dumps(result,indent=2))
