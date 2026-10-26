"""Read-only audit of exact Anchor labels, Replay reveals and compiler artifacts.

No CUDA execution or modification of source journals. Run after the analyzer;
missing GPUs remain missing, and the audit never fills performance cells.
"""
import argparse
import hashlib
import math
from pathlib import Path
import xml.etree.ElementTree as ET

import analyze_cross_gpu as study
core=study.core


def audit(root):
    root=Path(root);analysis=root/'analysis';summary=core.read_json(analysis/'summary.json')
    datasets={};all_rows={}
    for entry in summary['devices']:
        if not entry.get('records'):continue
        label=entry['label'];dataset=study.load_dataset(label,root/label/'anchor')
        rows=dataset['records'];by_id={r.id:r for r in rows};all_rows[label]=by_id
        assert len(rows)==len(by_id)==entry['records']
        exported=summary['datasets'][label]
        assert exported['dataset_id']==hashlib.sha256(core.canonical_json(sorted(by_id)).encode()).hexdigest()
        folder=analysis/exported['directory_relative_to_analysis']
        assert hashlib.sha256((folder/'measurements.jsonl').read_bytes()).hexdigest()==exported['jsonl_sha256']
        with core.ExperimentStore(folder/'measurements.sqlite',read_only=True) as store:
            assert store.connection.execute('PRAGMA quick_check').fetchall()==[('ok',)]
            assert {r.id:core.canonical_json(r.to_dict()) for r in store.records()}=={r.id:core.canonical_json(r.to_dict()) for r in rows}
        compiled=root/label/'compiled/index.json';artifacts=0
        if compiled.is_file():
            index=core.read_json(compiled);assert index['status']=='complete'
            for identity,item in index['artifacts'].items():
                blob=(compiled.parent/item['file']).read_bytes()
                assert hashlib.sha256(blob).hexdigest()==identity and len(blob)==item['bytes']
                artifacts+=1
        datasets[label]={'records':len(rows),'verified_cells':sum(c['verified'] for c in dataset['coverage']),
            'dataset_id':exported['dataset_id'],'sqlite_quick_check':'ok','compiler_cubin_sha256_checked':artifacts}
    replay=core.read_json(analysis/'replay.json');reveals=0
    for trial in replay['results']:
        trace=trial['trace'];rows=all_rows[trial['target']]
        assert len(trace)==trial['budget']==trial['evaluations']
        assert trial['new_gpu_measurements']==trial['failed_evaluations']==trial['missing_evaluations']==0
        observed=[]
        for step in trace:
            record=rows[step['original_source_record_id']]
            assert record.shape.id==trial['shape_id'] and record.seed==trial['input_seed'] and record.correctness_verified
            assert step['config']==record.config.params and step['latency_ms']==record.median_ms
            observed.append(record)
            assert step['best_latency_ms']==min(r.median_ms for r in observed)
            reveals+=1
        assert len({r.config.id for r in observed})==len(observed)
        assert trial['best_latency_ms']==min(r.median_ms for r in observed)
        assert math.isclose(trial['known_measurement_cost_s'],sum(r.wall_time_s for r in observed))
    for item in summary['matrix_exports'].values():
        assert hashlib.sha256((analysis/item['file']).read_bytes()).hexdigest()==item['sha256']
    for path in (analysis/'figures').glob('*.svg'):ET.parse(path)
    report={'status':'passed','datasets':datasets,'replay_trials_checked':len(replay['results']),
        'replay_reveals_checked':reveals,'new_gpu_measurements_in_audit':0,
        'paired_directions':len(summary['pairs']),
        'scope':'Original record IDs, observed-only best and costs, SQLite payloads, declared matrix and compiler SHA; no missing hardware inferred.'}
    core.write(root/'audit.json',report);print(core.canonical_json(report))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True)
    audit(parser.parse_args().root)
