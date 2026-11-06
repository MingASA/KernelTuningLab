"""Independent consistency audit of the frozen study; no CUDA or predictions.

Run after analyze_phase1.py and analyze_phase1_followups.py. This checks exact
observations, standard storage round trips, reveal costs and deployment counts.
It never repairs, deletes or overwrites a source measurement.
"""
import argparse
from collections import Counter
import hashlib
import math
from pathlib import Path
import statistics as st
import tempfile
import xml.etree.ElementTree as ET

import analyze_phase1 as core
from kernel_tuning_lab.asset_space import build_effective_space
from kernel_tuning_lab.catalog import get_catalog


def require(condition,message):
    if not condition: raise ValueError(message)


def file_sha(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): digest.update(block)
    return digest.hexdigest()


def audit(root):
    root=Path(root);analysis=root/'analysis'
    source=core.current_records(root/'matrix');by_id={r.id:r for r in source}
    require(len(source)==len(by_id)==1920,'expected 1920 distinct primary observations')
    require(all(r.provenance=='measured' and r.correctness_verified and
        r.metadata['correctness'].get('verified_on_gpu') is True for r in source),'unverified/non-GPU primary label')
    require(all(len(r.runtime_samples_ms)==31 and r.warmup==10 and r.repetitions==31 and
        r.metadata['protocol']['cuda_graph']['calls_per_replay']==512 for r in source),'primary timing protocol changed')
    require(len({r.hardware.id for r in source})==1,'physical hardware contexts mixed')
    cells=Counter((r.shape.id,r.config.id,r.seed) for r in source)
    require(len(cells)==1920 and set(cells.values())=={1},'repeated or absent cell/seed')
    require({r.seed for r in source}=={7,19,31},'input seed mismatch')
    kernel=get_catalog().resolve('kernel','triton/t1_gemm@1.0.0')
    legal_spaces={}
    for record in source:
        key=(record.hardware.id,record.operator.id,record.shape.id)
        if key not in legal_spaces:
            legal_spaces[key]=build_effective_space(kernel,record.hardware,record.operator,record.shape,core.design.SPACE)
        space=legal_spaces[key]
        require(space.validate(record.config,record.hardware,record.operator,record.shape).valid,'illegal primary config')
    matrix=core.PerformanceMatrix.from_records(
        core.WorkloadDistribution(tuple(core.ShapeSpec(dict(zip(('M','N','K'),s[2]))) for s in core.design.SHAPES),tuple([1]*10),name='T1 finite Shape distribution'),
        tuple(core.KernelConfig(c) for c in core.design.configs()),source)
    require(matrix.complete and matrix.to_dict()==core.read_json(analysis/'matrix.json'),'exported matrix differs from source labels')
    manifest=core.read_json(analysis/'dataset-manifest.json')
    require(manifest['dataset_id']==hashlib.sha256(core.canonical_json(sorted(by_id)).encode()).hexdigest(),'dataset identity mismatch')
    original={r.id:core.canonical_json(r.to_dict()) for r in source}
    with core.ExperimentStore(analysis/manifest['sqlite'],read_only=True) as store:
        require(store.connection.execute('PRAGMA quick_check').fetchall()==[('ok',)],'export SQLite damaged')
        require({r.id:core.canonical_json(r.to_dict()) for r in store.iter_records()}==original,'SQLite export lost semantics')
    with tempfile.TemporaryDirectory(prefix='ktlab-phase1-roundtrip-') as temporary:
        with core.ExperimentStore(Path(temporary)/'roundtrip.sqlite') as store:
            require(store.import_jsonl(analysis/manifest['jsonl'])=={'read':1920,'added':1920,'duplicates':0},'JSONL import lost observations')
            require(store.import_jsonl(analysis/manifest['jsonl'])=={'read':1920,'added':0,'duplicates':1920},'duplicate import changed identities')
            require({r.id:core.canonical_json(r.to_dict()) for r in store.iter_records()}==original,'JSONL round trip lost measurement semantics')
    replay=core.read_json(analysis/'replay.json')
    require(len(replay['results'])==3600 and replay['new_gpu_measurements']==0,'Replay coverage or physical cost misreported')
    for row in replay['results']:
        trace=row['trace'];require(len(trace)==row['evaluations']==row['budget'],'Replay budget mismatch')
        observed=[]
        for step in trace:
            record=by_id[step['source_record_id']]
            require(record.seed==row['measurement_seed'] and record.shape.id==row['shape_id'] and
                record.split==row['split'] and record.split!='train','Replay target/training label mismatch')
            require(core.KernelConfig(step['config']).id==record.config.id and step['latency_ms']==record.median_ms,'Replay label mismatch')
            observed.append(record)
            require(step['best_latency_ms']==min(r.median_ms for r in observed),'Replay history uses unobserved best')
        require(len({r.id for r in observed})==len(observed),'Replay charged duplicated observations')
        require(math.isclose(row['known_measurement_cost_s'],sum(r.wall_time_s for r in observed)),'Replay source cost mismatch')
        require(row['failed_evaluations']==row['missing_evaluations']==row['new_physical_measurements']==0,'Replay failure/new GPU count mismatch')
    by_cell={(tuple(r.shape.dims[k] for k in ('M','N','K')),r.config.id,r.seed):r for r in source}
    shapes={s[0]:s[2] for s in core.design.SHAPES}
    joint=core.read_json(analysis/'joint.json')
    require(len(joint['results'])==540,'joint simulation coverage mismatch')
    for row in joint['results']:
        acquired={tuple(pair) for pair in row['acquired_cell_ids']}
        require(len(acquired)==row['revealed_cells']<=row['budget'],'joint reveal budget mismatch')
        require(all(name in joint['target_shapes'] for name,cid in acquired),'joint charged non-target cell')
        require(len(row['selected_config_ids'])<=row['k'],'joint config budget exceeded')
        require(all((name,cid) in acquired for name in joint['target_shapes'] for cid in row['selected_config_ids']),
            'joint selected config has uncharged target measurements')
        cost=sum(by_cell[shapes[name],cid,row['measurement_seed']].wall_time_s for name,cid in acquired)
        require(math.isclose(cost,row['known_source_wall_s']) and row['new_gpu_measurements']==0,'joint cost mismatch')
    l3=core.read_json(analysis/'l3.json')
    require(l3['budget_kind']=='config_count' and len(l3['results'])==24,'L3 budget/scope mislabeled')
    for row in l3['results']:
        require(row['status']=='passed' and row['config_count']<=row['k'],'L3 config budget exceeded')
        require(row['compiled_variant_count'] is None and row['binary_size_bytes'] is None,'solver invented a physical budget solution')
        require(len(row['assignment_config_ids'])==10 and row['dispatch_only_artifacts']['compiler_specializations']==10,
            'shape-specialized dispatch identity count mismatch')
    compiled=core.read_json(root/'compiled-artifacts/index.json')
    require(compiled['status']=='complete' and not compiled['missing_cubin_sha256'],'compiler artifact archive incomplete')
    for identity,item in compiled['artifacts'].items():
        require(file_sha(root/'compiled-artifacts'/item['file'])==identity,'compiler CUBIN hash mismatch')
    for path in (analysis/'figures').glob('*.svg'): ET.parse(path)
    output={'status':'passed','source_record_ids_sha256':manifest['dataset_id'],
        'primary_records':len(source),'raw_latency_samples':sum(len(r.runtime_samples_ms) for r in source),
        'timed_full_flow_calls':len(source)*31*512,'standard_sqlite_jsonl_roundtrip':'all record payloads identical; duplicate import idempotent',
        'replay_trials_checked':3600,'joint_trials_checked':540,'l3_results_checked':24,
        'archived_cubin_sha256_checked':len(compiled['artifacts']),
        'scope':'No new GPU execution; source records remain unchanged. Hidden-label selection isolation is additionally covered by unit tests.'}
    core.write(root/'audit.json',output)
    print(core.canonical_json(output))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    audit(parser.parse_args().root)
