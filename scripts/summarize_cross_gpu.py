"""Publish sanitized cross-GPU evidence; full journals remain private."""
import argparse
import hashlib
from pathlib import Path
import shutil

import analyze_cross_gpu as study
core=study.core


def publication_matrix(analysis,device,exports):
    """Ignore stale derivatives; publish only this run's declared, hashed file."""
    declared=exports.get(device['label'])
    if not device.get('records') or not declared:return None
    if Path(declared['file']).name!=declared['file']:
        raise ValueError('matrix export must be a local analysis filename')
    path=analysis/declared['file']
    if hashlib.sha256(path.read_bytes()).hexdigest()!=declared['sha256']:
        raise ValueError('declared matrix changed after analysis: '+device['label'])
    return path


def summarize(root,docs):
    root,docs=Path(root),Path(docs);analysis=root/'analysis';data=docs/'data';figures=docs/'figures'
    result=core.read_json(analysis/'summary.json')
    result['remote_resource_status']=core.read_json(root/'baseline/public-resource-status.json')
    result['prior_archive_checks']=core.read_json(root/'baseline/archive-check.json')
    result['prior_file_integrity']=core.read_json(root/'baseline/prior-file-integrity.json')
    result['local_pilot']=core.read_json(root/'baseline/pilot-review.json')
    result['local_quality']=core.read_json(root/'baseline/anchor-quality.json')
    result['frozen_source']=core.read_json(root/'baseline/frozen-source.json')
    result['deployment_source']=core.read_json(root/'deploy/manifest.json')
    result['audit']=core.read_json(root/'audit.json')
    result['compiler_archive']={}
    for device in result['devices']:
        path=root/device['label']/'compiled/index.json'
        if device.get('records') and path.is_file():
            index=core.read_json(path)
            result['compiler_archive'][device['label']]={key:index[key] for key in
                ('status','wanted_cubin_artifacts','copied_cubin_artifacts','missing_cubin_sha256',
                 'total_cubin_container_bytes','known_text_section_bytes','scope')}
    result['analysis_source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in
        (core.ROOT/'scripts').glob('*cross_gpu*.py')}
    best=[]
    for device in result['devices']:
        label=device['label'];path=publication_matrix(analysis,device,result.get('matrix_exports',{}))
        if path is None:continue
        matrix=core.read_json(path)
        configs=[core.KernelConfig(c) for c in matrix['configs']]
        for i,(shape,row) in enumerate(zip(matrix['workload']['shapes'],matrix['latencies_ms'])):
            column=min(range(len(row)),key=lambda j:row[j])
            best.append({'gpu_label':label,'shape':shape,'latency_us':row[column]*1000,
                'config':dict(configs[column].params),'scope':'best of the verified finite '+str(len(configs))+'-config Anchor; two input runs'})
        data.mkdir(parents=True,exist_ok=True);figures.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(path,data/('crossgpu-anchor-'+label+'-matrix.json'))
        core._heatmap(figures/('crossgpu-anchor-'+label+'-relative-loss.svg'),
            label+' finite Anchor relative latency','Verified physical labels; missing target GPUs are not inferred',
            [s['name'] for s in study.DESIGN['shapes']],['C'+str(i+1) for i in range(len(configs))],
            [[v/min(row) for v in row] for row in matrix['latencies_ms']])
        if (analysis/'figures'/('l2-'+label+'.svg')).is_file():
            shutil.copyfile(analysis/'figures'/('l2-'+label+'.svg'),figures/('crossgpu-anchor-'+label+'-l2.svg'))
    result['best_by_shape']=best
    if result['pairs']:
        for name in ('zero-target-transfer','paired-ranks','topk-oracle-coverage'):
            shutil.copyfile(analysis/'figures'/(name+'.svg'),figures/('crossgpu-anchor-'+name+'.svg'))
    result['scope_notice']='One current GPU is a new local source Anchor, not cross-hardware transfer evidence. Original sample/cost/protocol records are in the standard private journal.' if not result['pairs'] else result['notice']
    core.write(data/'crossgpu-anchor-summary.json',result)
    print(core.canonical_json({'status':result['status'],'published_devices':[d['label'] for d in result['devices'] if d.get('records')],
        'paired_directions':len(result['pairs']),'public_summary':str(data/'crossgpu-anchor-summary.json')}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--root',required=True);parser.add_argument('--docs',default='docs')
    args=parser.parse_args();summarize(args.root,args.docs)
