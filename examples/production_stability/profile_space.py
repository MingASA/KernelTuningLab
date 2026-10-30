"""CPU-only full production space profiling from an existing measurement context."""
import argparse,cProfile,hashlib,json,pstats,sys,time
from pathlib import Path

def main():
    p=argparse.ArgumentParser();p.add_argument('--record',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    from kernel_tuning_lab.catalog import AssetCatalog
    from kernel_tuning_lab.plugins import load_plugin
    from kernel_tuning_lab.models import MeasurementRecord,stable_id
    from kernel_tuning_lab.assets import SpaceContext
    from kernel_tuning_lab.asset_space import build_effective_space
    from kernel_tuning_lab.space_tools import SpaceBuildBudget
    record=MeasurementRecord.from_dict(json.loads(Path(a.record).read_text()))
    catalog=AssetCatalog();load_plugin('kernel_tuning_lab.production.vllm_moe_compat',catalog)
    kernel=catalog.resolve('kernel','production/vllm085_moe_explicit@1.0.0')
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False);results=[];identities=[]
    for profiled in [False,True]:
        space=build_effective_space(kernel,record.hardware,record.operator,record.shape,
            context=SpaceContext(record.hardware,record.operator,record.shape,record.environment,512*1024**2))
        profiler=cProfile.Profile();began=time.perf_counter()
        if profiled:profiler.enable()
        built=space.build_feasible(record.hardware,record.operator,record.shape,
            budget=SpaceBuildBudget(max_nodes=500000,max_wall_time_s=150,max_configs=50000))
        if profiled:profiler.disable();profiler.dump_stats(str(out/'build.pstats'))
        wall=time.perf_counter()-began
        if not built.complete:raise RuntimeError('full construction incomplete: '+built.status)
        identities.append(stable_id(sorted(c.id for c in built.configs)))
        results.append({'profiled':profiled,'wall_s':wall,'configs':len(built.configs),'config_ids_sha':identities[-1],'diagnostics':dict(built.diagnostics)})
    assert len(set(identities))==1
    stats=pstats.Stats(str(out/'build.pstats'));functions=[]
    for (file,line,name),(primitive,calls,self_s,cumulative,callers) in sorted(stats.stats.items(),key=lambda item:item[1][3],reverse=True)[:35]:
        functions.append({'file':Path(file).name,'line':line,'function':name,'calls':calls,'self_s':self_s,'cumulative_s':cumulative})
    torch=sys.modules.get('torch');assert torch is None or not torch.cuda.is_initialized()
    result={'scope':'CPU-only exact production context; no CUDA initialization; same complete config IDs',
        'results':results,'profile_top_cumulative':functions,'torch_imported':torch is not None}
    (out/'summary.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
