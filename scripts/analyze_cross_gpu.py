"""Analyze verified Anchor journals without merging GPU/software contexts.

Uses the existing Store, PerformanceMatrix, hardware splits and reveal-only
Replay. Missing target data remains unavailable; synthetic data is never emitted
as GPU evidence. Source/target paths and host identities are not public output.
"""
import argparse
from collections import Counter,defaultdict
from copy import deepcopy
from dataclasses import replace
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics as st
import tempfile

import analyze_phase1 as core
from kernel_tuning_lab.datasets import DatasetSplits,training_record_manifest
from kernel_tuning_lab.search import _transfer_operator_identity

DESIGN=core.read_json(core.ROOT/'examples/crossgpu_anchor/frozen-design.json')
RUNTIME=core.read_json(core.ROOT/'examples/crossgpu_anchor/runtime-sources.json')
SOURCE_HASHES={Path(name).name:value for name,value in RUNTIME['sha256'].items()}
CONFIGS=tuple(core.KernelConfig(c) for c in DESIGN['configs'])
SHAPES=tuple(core.ShapeSpec(s['dims']) for s in DESIGN['shapes'])
LEGACY_GUARD_SHA256='87cdf18731d113d0ad3573d1a75df0bf99829f45ab16cb5fa6b56abd3f92e249'


def resource_admission(path,rows,required=False):
    """Bind the external admission receipt to this physical GPU and command.

    Never publish the receipt itself: it includes private paths and snapshots of
    other GPUs. Local desktop measurements predate the shared-server wrapper.
    """
    path=Path(path);proof_path=path.with_name(path.name+'-resource-guard.json')
    if not proof_path.is_file():
        if required:raise ValueError('shared-server dataset lacks resource admission receipt')
        return {'status':'not_required_local_desktop'}
    proof=core.read_json(proof_path)
    if not (proof.get('status')=='passed' and proof.get('eligible') is True
            and proof.get('exit_code')==0 and proof.get('external_processes_observed')==0
            and proof.get('conflict_checks',0)>0 and len(proof.get('guard_source_sha256',''))==64):
        raise ValueError('resource admission did not pass for the complete measurement command')
    normalize=lambda value:str(value or '').removeprefix('GPU-').lower()
    expected=normalize(proof.get('gpu',{}).get('uuid'))
    if not expected or not rows or any(normalize(r.hardware.metadata.get('device_uuid'))!=expected for r in rows):
        raise ValueError('resource receipt and measured physical GPU identities differ')
    command=proof.get('command',[]);binding='recorded_command'
    if command:
        if not isinstance(command,list) or '--output' not in command:
            raise ValueError('resource receipt lacks measurement output identity')
        output=command[command.index('--output')+1:command.index('--output')+2]
        if not output or Path(output[0]).name!=path.name:
            raise ValueError('resource receipt belongs to another measurement command')
    else:
        # The already-used cbccd9d wrapper predates command serialization.
        # Do not alter old receipts or invent fields as if captured originally.
        # Bind that exact legacy version through retained driver/console data.
        control_path=path.with_name(path.name+'-resource-control.json')
        if proof['guard_source_sha256']!=LEGACY_GUARD_SHA256 or not control_path.is_file():
            raise ValueError('resource receipt lacks measurement output identity')
        control=core.read_json(control_path);manifest=core.read_json(path/'research_manifest.json')
        if not (control.get('derived_after_collection') is True and control.get('stage')==path.name
                and control.get('manifest_run_id')==manifest['run_id']
                and control.get('raw_guard_sha256')==hashlib.sha256(proof_path.read_bytes()).hexdigest()):
            raise ValueError('legacy resource control binding differs from dataset')
        for key in ('raw_guard','raw_console'):
            filename=control.get(key+'_basename','')
            if not filename or Path(filename).name!=filename:raise ValueError('invalid control evidence filename')
            blob=(path.parent/'ops-logs'/filename).read_bytes()
            if hashlib.sha256(blob).hexdigest()!=control.get(key+'_sha256'):
                raise ValueError('legacy control evidence content differs')
            if key=='raw_console':
                summary=json.JSONDecoder().raw_decode(blob.decode())[0]
                if summary.get('run_id')!=manifest['run_id'] or summary.get('status')!='passed':
                    raise ValueError('legacy console belongs to another dataset run')
        binding='legacy_driver_manifest_console_attestation'
    return {'status':'passed','guard_source_sha256':proof['guard_source_sha256'],
        'conflict_checks':proof['conflict_checks'],'external_processes_observed':0,
        'wall_cap_s':proof.get('wall_cap_s'),'wall_s':proof.get('wall_s'),
        'retired_owned_nvml_entries':proof.get('retired_owned_nvml_entries',0),
        'physical_identity_matched':True,'command_binding':binding}


def finite_space(configs):
    if not configs:raise ValueError('a nonempty observed common cohort is required')
    ids=frozenset(c.id for c in configs)
    def member(config,hardware,operator,shape):
        return None if config.id in ids else 'outside the frozen, physically observed common cohort'
    return core.ConfigSpace({key:list(dict.fromkeys(c.params[key] for c in configs)) for key in DESIGN['configs'][0]},
        constraints=(member,),name='crossgpu physically common finite cohort',identity={'config_ids':sorted(ids)})


def _text_sources(asset):
    callbacks={key:{'entry':asset[key].get('entry'),'source':asset[key].get('source')} for key in ('build','supports','generate_inputs','reference','validate') if isinstance(asset.get(key),dict)}
    return {'ref':asset['ref'],'callbacks':callbacks,'source_files':asset.get('source_files',[]),
        'tolerances':asset.get('tolerances'),'parameters':asset.get('parameters')}


def pairing_protocol(record):
    protocol=deepcopy(record.metadata['protocol'])
    for key in ('config_space','kernel_fingerprint','operator_fingerprint'):protocol.pop(key,None)
    return {'mode':record.measurement_mode,'warmup':record.warmup,'repetitions':record.repetitions,
        'dtype':record.operator.dtype,'operator_kind':record.operator.kind,'parameters':record.operator.parameters,
        'protocol':protocol,'runtime_source_sha256':record.environment.get('runtime_source_sha256'),
        'torch_context_source_sha256':record.environment.get('torch_execution_context',{}).get('source_sha256')}


def load_dataset(label,path):
    path=Path(path)
    if not (path/'research_manifest.json').is_file():return {'label':label,'status':'not_executed','records':()}
    manifest=core.read_json(path/'research_manifest.json');rows=core.current_records(path)
    if manifest.get('git_commit')!=RUNTIME['frozen_experiment_commit']:
        raise ValueError('dataset was not collected at the frozen Anchor commit: '+label)
    cells=defaultdict(list);allowed={c.id for c in CONFIGS};source_signatures=set()
    for job in manifest['jobs']:
        snapshot=core.snapshot_directory(path,job)
        if snapshot is None:continue
        lock=snapshot/'assets.lock.json'
        if not lock.is_file():continue
        assets=[_text_sources(a) for a in core.read_json(lock)['assets'] if a['kind'] in ('operator','kernel')]
        if {asset['ref'] for asset in assets}!={'gemm@1.0.0','triton/t1_gemm@1.0.0'}:
            raise ValueError('dataset needs the frozen operator and implementation source locks: '+label)
        for asset in assets:
            sources=[c['source'] for c in asset['callbacks'].values()]+asset['source_files']
            if any(not src or SOURCE_HASHES.get(src['name'])!=src['sha256'] for src in sources):
                raise ValueError('asset source bytes differ from the frozen runtime: '+label)
        source_signatures.add(core.canonical_json(assets))
    for r in rows:
        if r.shape.id not in {s.id for s in SHAPES} or r.config.id not in allowed or r.seed not in DESIGN['seeds']:
            raise ValueError('record outside frozen Anchor cohort: '+label)
        if r.successful:
            if not (r.correctness_verified and r.provenance=='measured' and r.metadata['correctness'].get('verified_on_gpu')
                    and r.hardware.metadata.get('source')=='torch.cuda.get_device_properties' and not r.hardware.metadata.get('specification_only')
                    and len(r.environment.get('runtime_source_sha256',''))==64
                    and len(r.environment.get('torch_execution_context',{}).get('source_sha256',''))==64
                    and r.environment['runtime_source_sha256']==SOURCE_HASHES['kernel_runtime.py']
                    and r.environment['torch_execution_context']['source_sha256']==SOURCE_HASHES['torch_context.py']
                    and r.metadata['protocol'].get('measurement_driver_source_sha256')==SOURCE_HASHES['runner.py']
                    and len(r.runtime_samples_ms)==31 and r.repetitions==31 and r.warmup==10
                    and r.measurement_mode=='cuda_graph_events_full_flow_amortized'
                    and r.metadata['protocol']['claim_policy']['force_correctness']
                    and not r.metadata['protocol']['claim_policy']['accept_claims']
                    and r.operator.implementation=='triton/t1_gemm@1.0.0'
                    and r.operator.dtype=='float16' and r.metadata['protocol']['cuda_graph']['calls_per_replay']==512):
                raise ValueError('Anchor label lacks the required physical GPU/protocol evidence')
        cells[r.shape.id,r.config.id,r.seed].append(r)
    if len({r.hardware.id for r in rows})>1:raise ValueError('one dataset contains several physical GPU identities')
    if rows and len(source_signatures)!=1:raise ValueError('dataset needs one complete implementation/reference text source signature')
    protocols={core.canonical_json(pairing_protocol(r)) for r in rows}
    if len(protocols)>1:raise ValueError('one dataset contains incompatible timing/correctness protocols')
    coverage=[]
    for shape,config,seed in itertools.product(SHAPES,CONFIGS,DESIGN['seeds']):
        found=cells[shape.id,config.id,seed]
        status=found[0].status.value if len(found)==1 else 'not_measured' if not found else 'duplicate'
        verified=len(found)==1 and found[0].correctness_verified
        coverage.append({'shape_id':shape.id,'config_id':config.id,'seed':seed,'status':status,'verified':verified})
    admission=resource_admission(path,rows,required=label in ('4090','H800'))
    return {'label':label,'status':'passed' if all(c['verified'] for c in coverage) else 'partial',
        'records':rows,'coverage':coverage,'manifest':manifest,
        'text_sources':next(iter(source_signatures),None),'pairing_protocol':next(iter(protocols),None),
        'resource_admission':admission}


def pooled_table(rows,configs):
    return core.PerformanceMatrix.from_records(core.WorkloadDistribution(SHAPES,tuple([1]*len(SHAPES)),name='crossgpu finite Anchor'),configs,rows)


def paired_metrics(source,target,configs):
    """Target-relative scores only; the Top-K minimum is evaluator coverage."""
    if len(source)!=len(target) or any(len(row)!=len(configs) for row in (*source,*target)):
        raise ValueError('inconsistent paired finite matrices')
    result=[]
    for i,(left,right) in enumerate(zip(source,target)):
        order=sorted(range(len(configs)),key=lambda j:(left[j],configs[j].id));best=min(right)
        overlaps={}
        for fraction in (.05,.1,.2):
            n=max(1,math.ceil(len(configs)*fraction));a=set(order[:n]);b=set(sorted(range(len(configs)),key=lambda j:(right[j],configs[j].id))[:n])
            overlaps[str(fraction)]={'count':n,'intersection':len(a&b),'jaccard':len(a&b)/len(a|b)}
        result.append({'shape_id':SHAPES[i].id,'spearman':core.correlation(left,right),'top_overlap':overlaps,
            'source_top1_config':dict(configs[order[0]].params),'source_top1_target_regret':right[order[0]]/best-1,
            'source_topk_target_coverage':[{'k':k,'regret':min(right[j] for j in order[:k])/best-1}
                for k in (1,2,4,8) if k<=len(configs)],
            'notice':'Top-K target minimum is oracle evaluation coverage, not a zero-target-measurement selector.'})
    return result


def hardware_fold(source,target,fold_id):
    """Explicit derived labels; physical records and original identities stay intact."""
    def relabel(record,role):
        return replace(record,split=role,metadata={**record.metadata,'hardware_holdout_fold':{
            'id':fold_id,'source_record_id':record.id,'original_split':record.split,'role':role}})
    result=DatasetSplits(tuple(relabel(r,'train') for r in source),(),tuple(relabel(r,'test') for r in target),holdout='hardware')
    return result


class StaticSourcePriority:
    """Fixed source-Shape rankings; history only excludes attempted configs."""
    name='source_static_priority'
    def __init__(self,records,configs):
        if any(r.split!='train' for r in records):raise ValueError('source priority needs train-only labels')
        self.training_records=tuple(records);groups=defaultdict(lambda:defaultdict(list));self.orders={}
        for r in records:groups[r.shape.id][r.config.id].extend(r.runtime_samples_ms)
        for sid,values in groups.items():
            self.orders[sid]=sorted(configs,key=lambda c:(st.median(values[c.id]),c.id))
    def suggest(self,hardware,operator,shape,space,history,budget,batch_size=1):
        if budget.exhausted(0,0):return []
        seen={r.config.id for r in history}|set(history.attempted_config_ids)
        return [c for c in self.orders[shape.id] if c.id not in seen and space.validate(c,hardware,operator,shape).valid][:min(batch_size,budget.max_evaluations)]


def replay_case(rows,searcher,configs,budget,rng):
    result=core.run_replay(rows,searcher,finite_space(configs),core.SearchBudget(budget),seed=rng).to_dict()
    originals={r.config.id:r.metadata.get('hardware_holdout_fold',{}).get('source_record_id',r.id) for r in rows}
    for step in result['trace']:step['original_source_record_id']=originals[core.KernelConfig(step['config']).id]
    return result


def public_device(dataset):
    if not dataset['records']:
        return {'label':dataset['label'],'status':dataset['status'],
            'job_statuses':dict(Counter(j['status'] for j in dataset.get('manifest',{}).get('jobs',[])))}
    rows=dataset['records'];r=rows[0];hw=r.hardware;env=r.environment
    costs={'new_evaluations':sum(j.get('result',{}).get('evaluations',0) for j in dataset['manifest']['jobs']),
        'cache_hits':sum(j.get('result',{}).get('cache_hits',0) for j in dataset['manifest']['jobs']),
        'acceptance_evaluations':sum(j.get('result',{}).get('validation_evaluations',0) for j in dataset['manifest']['jobs']),
        'job_wall_s':sum(j.get('wall_time_s',0) for j in dataset['manifest']['jobs']),
        'unique_source_measurement_wall_s':sum(r.wall_time_s for r in rows),'compile_time_ms':sum(r.compile_time_ms for r in rows)}
    phases=Counter()
    for record in rows:phases.update(record.metadata.get('phase_wall_time_ms',{}))
    costs['phase_wall_time_ms']=dict(phases)
    executions=[record.metadata.get('correctness',{}).get('execution',{}) for record in rows]
    costs['compile_subphases_ms']={key:sum(e.get(key,0) or 0 for e in executions)
        for key in ('build_time_ms','setup_time_ms','cuda_graph_setup_time_ms')}
    costs['scope_notice']='Compile includes build/setup/Graph capture; subphases are nested, not additive. Existing compiler cache is uncontrolled; source costs do not estimate independent cold starts. Job wall additionally includes acceptance and orchestration.'
    execution=deepcopy(env.get('torch_execution_context',{}));overrides=execution.pop('compiler_overrides',{})
    execution['compiler_override_names']=sorted(overrides)
    execution['compiler_overrides_sha256']=hashlib.sha256(core.canonical_json(overrides).encode()).hexdigest()
    cells=defaultdict(list)
    for record in rows:cells[record.shape.id,record.config.id].append(record)
    repeat_quality=[]
    for (sid,cid),group in cells.items():
        values=[r.median_ms for r in group]
        cv=st.stdev(values)/st.mean(values) if len(values)>1 else None
        repeat_quality.append({'shape_id':sid,'config_id':cid,'physical_input_runs':len(group),
            'sample_run_cv':cv,'run_medians_ms':values,
            'within_run_p10_p90_relative_width_max':max((r.quantile_ms(.9)-r.quantile_ms(.1))/r.median_ms for r in group),
            'run_cv_above_5pct':cv is not None and cv>.05})
    cvs=[r['sample_run_cv'] for r in repeat_quality if r['sample_run_cv'] is not None]
    quality={'cell_count':len(cells),'sample_run_cv_median':st.median(cvs) if cvs else None,
        'sample_run_cv_p90':core.quantile(cvs,.9),'sample_run_cv_max':max(cvs,default=None),
        'cells_above_5pct':sum(r['run_cv_above_5pct'] for r in repeat_quality),'cells':repeat_quality,
        'notice':'Two physical input runs per cell; internal Event samples and algorithm RNG seeds are not independent physical repetitions. CV flags uncertainty, not a significance test; no records excluded for being slow.'}
    return {'label':dataset['label'],'status':dataset['status'],'hardware':{'name':hw.name,'architecture':hw.architecture,
        'sm_count':hw.sm_count,'memory_bytes':hw.memory_bytes,'l2_bytes':hw.l2_bytes,'smem_per_block':hw.smem_per_block},
        'software':{k:env.get(k) for k in ('python','torch','triton','cuda_runtime','nvidia_driver','runtime_source_sha256')},
        'execution_controls':execution,
        'experiment_commit':dataset['manifest'].get('git_commit'),'records':len(rows),'costs':costs,
        'resource_admission':dataset.get('resource_admission',{'status':'not_provided'}),
        'repeat_quality':quality,
        'record_statuses':dict(Counter(r.status.value for r in rows)),
        'missing_or_failed':sum(not c['verified'] for c in dataset['coverage']),
        'source_record_ids_sha256':hashlib.sha256(core.canonical_json(sorted(r.id for r in rows)).encode()).hexdigest()}


def pairing_issue(left,right):
    if not left['manifest'].get('git_commit') or left['manifest'].get('git_commit')!=right['manifest'].get('git_commit'):
        return 'different or unknown frozen experiment commit'
    if left['records'][0].hardware.id==right['records'][0].hardware.id:
        return 'same or indistinguishable physical GPU identity'
    if (not left['text_sources'] or left['text_sources']!=right['text_sources']
            or left['pairing_protocol']!=right['pairing_protocol']):
        return 'different or missing implementation/reference text source or timing/correctness scope'
    return None


def public_matrix(matrix):
    result=matrix.to_dict();metadata=result.get('metadata',{})
    result['metadata']={'publication':'sanitized finite measured matrix; full context remains in the private standard journal',
        'original_metadata_sha256':hashlib.sha256(core.canonical_json(metadata).encode()).hexdigest(),
        'measurement_context_id':metadata.get('measurement_context_id'),'hardware_id':metadata.get('hardware_id'),
        'operator_id':metadata.get('operator_id'),'protocol':DESIGN['protocol']}
    return result


def export_dataset(dataset,output):
    """Standard Store export and full-record round trip; preserve original IDs."""
    rows=dataset['records'];identity=hashlib.sha256(core.canonical_json(sorted(r.id for r in rows)).encode()).hexdigest()
    folder=output/'datasets'/identity;folder.mkdir(parents=True,exist_ok=True)
    with core.ExperimentStore(folder/'measurements.sqlite') as store:
        store.add_many(rows);store.export_jsonl(folder/'measurements.jsonl')
    originals={r.id:core.canonical_json(r.to_dict()) for r in rows}
    with tempfile.TemporaryDirectory(prefix='store-audit-',dir=output) as temporary:
        with core.ExperimentStore(Path(temporary)/'restored.sqlite') as restored:
            restored.import_jsonl(folder/'measurements.jsonl')
            if {r.id:core.canonical_json(r.to_dict()) for r in restored.records()}!=originals:
                raise ValueError('standard Store round trip changed full record semantics')
    report={'dataset_id':identity,'records':len(rows),'full_record_roundtrip':True,
        'jsonl_sha256':hashlib.sha256((folder/'measurements.jsonl').read_bytes()).hexdigest(),
        'directory_relative_to_analysis':str(folder.relative_to(output)),
        'finite_cohort':'Use the explicit frozen/common cohort in this analyzer; the broad SDK Cartesian domain is not the 16-point matrix.'}
    core.write(output/(dataset['label']+'-dataset.json'),report)
    return report


def analyze(inputs,output,skip_replay=False):
    output=Path(output);figures=output/'figures';figures.mkdir(parents=True,exist_ok=True)
    datasets=[load_dataset(label,path) for label,path in inputs];available=[d for d in datasets if d['records']]
    success_sets=[]
    for d in available:
        good={(c['shape_id'],c['config_id'],c['seed']) for c in d['coverage'] if c['verified']}
        success_sets.append({c.id for c in CONFIGS if all((s.id,c.id,seed) in good for s,seed in itertools.product(SHAPES,DESIGN['seeds']))})
    common=set.intersection(*success_sets) if success_sets else set();configs=tuple(c for c in CONFIGS if c.id in common)
    tables={};public=[];dataset_exports={};matrix_exports={}
    for d in datasets:
        public.append(public_device(d))
        if d['records']:
            dataset_exports[d['label']]=export_dataset(d,output)
            core.write(output/(d['label']+'-coverage.json'),d['coverage'])
            # A per-GPU matrix keeps full source context; no cross-GPU pooling.
            own=[c for c in CONFIGS if all(cov['verified'] for cov in d['coverage'] if cov['config_id']==c.id)]
            if own:
                matrix=pooled_table([r for r in d['records'] if r.config.id in {c.id for c in own}],tuple(own))
                core.write(output/(d['label']+'-matrix.json'),matrix.to_dict())
                path=output/(d['label']+'-matrix-public.json')
                core.write(path,public_matrix(matrix))
                matrix_exports[d['label']]={'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
        if d['records'] and configs:tables[d['label']]=pooled_table([r for r in d['records'] if r.config.id in common],configs)
    pairs=[];replays=[];not_applicable=[];training_costs=[]
    for left,right in itertools.permutations(available,2):
        issue=pairing_issue(left,right)
        if issue:
            not_applicable.append({'source':left['label'],'target':right['label'],'method':'paired_comparison','reason':issue});continue
        if not configs:continue
        ls,rs=left['label'],right['label'];metrics=paired_metrics(tables[ls].latencies_ms,tables[rs].latencies_ms,configs)
        source=[r for r in left['records'] if r.config.id in common];target=[r for r in right['records'] if r.config.id in common]
        input_hashes=[]
        for s,seed in itertools.product(SHAPES,DESIGN['seeds']):
            a=next(r for r in source if r.shape.id==s.id and r.seed==seed);b=next(r for r in target if r.shape.id==s.id and r.seed==seed)
            ha=a.metadata['correctness'].get('input_integrity',{}).get('expected');hb=b.metadata['correctness'].get('input_integrity',{}).get('expected')
            input_hashes.append({'shape_id':s.id,'input_seed':seed,'bitwise_inputs_equal':ha==hb if ha and hb else None})
        pairs.append({'source':ls,'target':rs,'metrics':metrics,'input_hashes':input_hashes,
            'software_confounds':(public_device(left)['software'],public_device(left)['execution_controls'])!=(public_device(right)['software'],public_device(right)['execution_controls']),
            'notice':'Source-text/protocol matched; software contexts remain separate. Architecture-only causality is not inferred.'})
        fold=hardware_fold(source,target,ls+'-to-'+rs);manifest=training_record_manifest(fold.train)
        training_costs.append({'source':ls,'target':rs,'records':manifest['records'],'source_acquisition_wall_s':manifest['unique_source_wall_s'],
            'training_hash':manifest['record_ids_sha256'],'new_gpu_measurements':0})
        compatible=_transfer_operator_identity(fold.train[0].operator)==_transfer_operator_identity(fold.test[0].operator)
        if not compatible:not_applicable.append({'source':ls,'target':rs,'method':'transfer','reason':'full asset/operator fingerprints are incompatible; no relabelling to bypass this guard'})
        if skip_replay:continue
        for s,seed,rng,budget,method in itertools.product(SHAPES,DESIGN['seeds'],DESIGN['search_rng_seeds'],(1,2,4,8),('source_static_priority','transfer')):
            if budget>len(configs) or (method=='transfer' and not compatible):continue
            target_rows=[r for r in fold.test if r.shape.id==s.id and r.seed==seed]
            searcher=StaticSourcePriority(fold.train,configs) if method=='source_static_priority' else core.make_searcher('transfer',seed=rng,training_records=fold.train)
            result=replay_case(target_rows,searcher,configs,budget,rng)
            replays.append({**result,'source':ls,'target':rs,'method':method,'input_seed':seed,'new_gpu_measurements':0})
    if not skip_replay and configs:
        for dataset,s,seed,rng,budget,method in itertools.product(available,SHAPES,DESIGN['seeds'],DESIGN['search_rng_seeds'],(1,2,4,8),('random','heuristic','surrogate')):
            if budget>len(configs):continue
            rows=[r for r in dataset['records'] if r.shape.id==s.id and r.seed==seed and r.config.id in common]
            result=replay_case(rows,core.make_searcher(method,seed=rng),configs,budget,rng)
            replays.append({**result,'source':None,'target':dataset['label'],'method':method,'input_seed':seed,'new_gpu_measurements':0})
    groups=defaultdict(list)
    for r in replays:groups[r['source'],r['target'],r['method'],r['budget']].append(r)
    aggregates=[]
    for (source,target,method,budget),group in groups.items():
        valid=[r['regret'] for r in group if r['regret'] is not None]
        aggregates.append({'source':source,'target':target,'method':method,'budget':budget,'budget_fraction':budget/len(configs),
            'trials':len(group),'mean_regret':st.mean(valid) if valid else None,'p90_regret':core.quantile(valid,.9),
            'within_5pct':st.mean(r['regret'] is not None and r['regret']<=.05 for r in group),
            'within_10pct':st.mean(r['regret'] is not None and r['regret']<=.1 for r in group),
            'failed_or_missing':sum(r['failed_evaluations']+r['missing_evaluations'] for r in group),
            'mean_known_target_source_cost_s':st.mean(r['known_measurement_cost_s'] for r in group)})
    core.write(output/'replay.json',{'results':replays,'training_costs':training_costs,'aggregates':aggregates,'not_applicable':not_applicable})
    result={'schema_version':1,'cohort_id':DESIGN['cohort_id'],'status':'paired' if pairs else 'blocked_no_paired_gpu',
        'devices':public,'datasets':dataset_exports,'matrix_exports':matrix_exports,
        'common_config_count':len(configs) if pairs else None,
        'common_configs':[dict(c.params) for c in configs] if pairs else [],
        'measured_device_intersection_config_count':len(configs),
        'requested_all_devices_measured':len(available)==len(datasets) and all(d['status']=='passed' for d in datasets),
        'common_cohort_scope':'Only available, verified datasets; absent GPUs are never assumed to support these configs.',
        'requested_config_count':len(CONFIGS),'shapes':DESIGN['shapes'],'pairs':pairs,'not_applicable':not_applicable,
        'replay_trials':len(replays),'replay_aggregates':aggregates,'training_costs':training_costs,
        'notice':'Only verified physical records; Replay creates no GPU measurements. B8 of 16 is 50%, not a 1%-cost result. RNG repeats are not independent physical runs.'}
    core.write(output/'summary.json',result)
    if pairs:
        zero={p['source']+' to '+p['target']:[(i+1,r['source_top1_target_regret']*100) for i,r in enumerate(p['metrics'])] for p in pairs}
        ranks={p['source']+' / '+p['target']:[(i+1,r['spearman']) for i,r in enumerate(p['metrics']) if r['spearman'] is not None]
            for p in pairs if p['source']<p['target']}
        coverage={p['source']+' to '+p['target']:[(k,st.mean(next(c['regret'] for c in m['source_topk_target_coverage'] if c['k']==k) for m in p['metrics'])*100)
            for k in (1,2,4,8) if k<=len(configs)] for p in pairs}
        for name,series,ylabel,title,context in (
            ('zero-target-transfer',zero,'Target finite-set regret (%)','Source Top1, zero target measurements','Source ranking only; target finite optimum used for scoring, not candidate selection'),
            ('paired-ranks',ranks,'Spearman rho','Paired GPU configuration ranks','Per-Shape ranks over the actually verified common cohort; no architectural causality inferred'),
            ('topk-oracle-coverage',coverage,'Target finite-set regret (%)','Source TopK oracle coverage','Post-hoc target best within source TopK; not a zero-observation selector')):
            core._line_chart(figures/(name+'.svg'),title,context,series,
                'Source TopK' if name=='topk-oracle-coverage' else 'Shape index / frozen order (1-6)',ylabel,render_png=False)
    for target in {r['target'] for r in aggregates}:
        rows=[r for r in aggregates if r['target']==target]
        series={str(source or 'cold')+'/'+method:[(r['budget'],r['mean_regret']*100) for r in rows if r['source']==source and r['method']==method and r['mean_regret'] is not None]
            for source,method in {(r['source'],r['method']) for r in rows}}
        subtitle='Physical labels; train hardware excluded from target; software differences annotated' if pairs else 'Single-GPU cold Replay only; cross-hardware transfer has not executed'
        core._line_chart(figures/('l2-'+target+'.svg'),'Anchor finite-cohort Replay',subtitle,series,'Target observations / '+str(len(configs)),'Mean target finite-set regret (%)',render_png=False)
    print(core.canonical_json({'status':result['status'],'devices':len(available),'paired_directions':len(pairs),'replay_trials':len(replays)}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--dataset',action='append',required=True,help='label=existing research root; no SSH or GPU execution')
    parser.add_argument('--output',required=True);parser.add_argument('--skip-replay',action='store_true')
    args=parser.parse_args();inputs=[item.split('=',1) for item in args.dataset]
    if any(len(item)!=2 for item in inputs) or len({item[0] for item in inputs})!=len(inputs):raise ValueError('distinct label=path inputs required')
    if any(not label.replace('-','').isalnum() for label,_ in inputs):raise ValueError('labels must be alphanumeric with optional hyphens')
    analyze(inputs,args.output,args.skip_replay)
