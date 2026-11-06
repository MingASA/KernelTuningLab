"""Read-only source audit plus standard-store round trip of phase2 evidence."""
import argparse
from datetime import datetime
import hashlib
from pathlib import Path

import analyze_phase2 as phase
core=phase.core


def audit(root,history_root=None):
    root=Path(root);rows=phase.physical_records(root);coverage=phase.check_coverage(rows,phase.design.SHAPES)
    if coverage['missing_successes']:raise ValueError('incomplete primary matrix')
    by_id={r.id:r for r in rows};assert len(by_id)==2304
    replay=core.read_json(root/'analysis/replay.json')
    expected=sum(14*5*4*len({r['algorithm'] for r in replay['results'] if r['measurement_seed']==seed}) for seed in phase.design.SEEDS)
    assert len(replay['results'])==expected
    checked=0
    for trial in replay['results']:
        trace=trial['trace'];assert len(trace)==trial['budget']==trial['evaluations']
        assert len({core.KernelConfig(step['config']).id for step in trace})==len(trace)
        best=None
        for step in trace:
            r=by_id[step['source_record_id']]
            assert r.shape.id==trial['shape_id'] and r.seed==trial['measurement_seed']
            assert r.split==trial['split'] and r.correctness_verified
            assert r.config.params==step['config'] and r.median_ms==step['latency_ms']
            best=min(best,r.median_ms) if best is not None else r.median_ms
            assert step['best_latency_ms']==best
            checked+=1
        assert trial['best_latency_ms']==best and trial['failed_evaluations']==trial['missing_evaluations']==0
        observed_best=min(by_id[s['source_record_id']].median_ms for s in trace)
        assert observed_best==trial['best_latency_ms']
    primary=core.read_json(root/'analysis/dataset-manifest.json')
    original={r.id:core.canonical_json(r.to_dict()) for r in rows}
    restored=root/'audit/roundtrip.sqlite';exported=root/'audit/roundtrip.jsonl'
    with core.ExperimentStore(restored) as store:
        store.import_jsonl(root/'analysis'/primary['jsonl']);store.export_jsonl(exported)
        assert {r.id:core.canonical_json(r.to_dict()) for r in store.records()}==original
    prior=core.read_json(root/'pretest/prior.json')
    assert prior['heldout_not_started']
    for label in ('validation','test'):
        start=core.read_json(root/label/'run_metadata.json')['started_at']
        assert datetime.fromisoformat(prior['created_utc'])<datetime.fromisoformat(start)
    history=phase.optional_history(history_root)
    train={r.id:r for r in (*rows,*history) if r.split=='train'}
    for manifest in replay['training_manifests'].values():
        source=[train[identity] for identity in manifest['record_ids']]
        assert phase.training_record_manifest(source)==manifest
        assert not {r.shape.id for r in source}&{r.shape.id for r in rows if r.split!='train'}
    index=core.read_json(root/'compiled/index.json');assert index['status']=='complete'
    for identity,entry in index['artifacts'].items():
        blob=(root/'compiled'/entry['file']).read_bytes()
        assert hashlib.sha256(blob).hexdigest()==identity and len(blob)==entry['bytes']
    archives={}
    for name in ('5060ti-evidence-20261025.tar.gz','research-workflow-evidence-20261025.tar.gz','research-phase1-evidence-20261025.tar.gz'):
        path=core.ROOT/'artifacts'/name
        if not path.exists():
            archives[name]={'status':'not_provided'};continue
        expected=Path(str(path)+'.sha256').read_text().split()[0]
        digest=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(1024*1024),b''):digest.update(block)
        assert digest.hexdigest()==expected
        archives[name]=expected
    report={'status':'passed','primary_records':len(rows),'replay_trials':len(replay['results']),
        'replay_reveals_checked':checked,'full_record_roundtrip':True,'prior_precedes_heldout_acquisition':True,
        'training_ids_and_costs_verified':True,'old_archives_sha256':archives,
        'compiler_artifacts_verified':len(index['artifacts']),
        'scope':'Exact source IDs, revealed-only best, train/target isolation, complete schema round trip and SHA. Trusted plugins are not an OS security sandbox; GPU numeric evidence is retained in original records.'}
    core.write(root/'audit.json',report);print(core.canonical_json(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    parser.add_argument('--history-root',help='same optional historical source supplied to the analyzer')
    args=parser.parse_args();audit(args.root,args.history_root)
