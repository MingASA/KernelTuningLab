"""Bounded failure diagnosis/repair acceptance using the existing strict Runner.

Run only through the normal idle-GPU guard. The plan lists exact cases; no
optimizer, model, performance ranking or tolerance override is involved.
"""
import argparse,gc,json,subprocess
from pathlib import Path

def write(path,value):path.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    plan=json.loads(Path(a.plan).read_text());out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    assert subprocess.check_output(['git','-C',plan['code'],'rev-parse','HEAD'],text=True).strip()==plan['source_commit']
    from kernel_tuning_lab.catalog import AssetCatalog
    from kernel_tuning_lab.plugins import load_plugin
    from kernel_tuning_lab.sdk import AssetKernelAdapter,ExperimentStore,KernelConfig,OperatorSpec,ShapeSpec,SpaceContext
    from kernel_tuning_lab.asset_space import build_effective_space
    from kernel_tuning_lab.runner import BenchmarkRunner
    from kernel_tuning_lab.inference_resources import COORDINATOR
    from kernel_tuning_lab.production import gemlite
    class InspectAdapter(AssetKernelAdapter):
        def compile(self,*args):
            self.compiled=super().compile(*args);return self.compiled
    state={'status':'running','source_commit':plan['source_commit'],'cases':[]};write(out/'state.json',state)
    original_padding=gemlite.pad_split_metadata
    try:
        with ExperimentStore(out/'measurements.sqlite') as store:
            for row in plan['cases']:
                catalog=AssetCatalog();load_plugin(row['plugin'],catalog)
                op=catalog.resolve('operator','production/uniform_lowbit_gemm@1.0.0');kernel=catalog.resolve('kernel',row['kernel'])
                adapter=InspectAdapter(kernel,op,force_correctness=True,accept_claims=False,timing_mode='cuda_graph',cuda_graph_batch_size=64,workspace_memory_limit_bytes=512*1024**2)
                hardware=adapter.probe_hardware();assert hardware.metadata['device_uuid'].lower().removeprefix('gpu-')==plan['gpu_uuid'].lower().removeprefix('gpu-')
                operator=OperatorSpec(op.id,kind=op.id,version=op.version,implementation=kernel.ref,dtype=row['dtype'],parameters=row['parameters'],
                    correctness_baseline=op.ref,metadata={'operator_asset_fingerprint':op.fingerprint,'kernel_asset_fingerprint':kernel.fingerprint})
                shape=ShapeSpec(row['shape']);space=build_effective_space(kernel,hardware,operator,shape,context=SpaceContext(hardware,operator,shape,adapter.environment(),512*1024**2))
                if row.get('diagnostic_unpadded'):
                    gemlite.pad_split_metadata=lambda torch,scales,**kwargs:(scales,{'diagnostic_unpadded':True})
                else:gemlite.pad_split_metadata=original_padding
                runner=BenchmarkRunner(adapter,store=store,warmup=2,repetitions=3,seed=row['seed'],experiment_id='production-stability-'+row['id'],failure_log_path=out/'failures.jsonl')
                record=runner.measure(hardware,operator,shape,KernelConfig(row['config']),space,split='validation',resume=False)
                result={'id':row['id'],'record_id':record.id,'status':record.status.value,'verified':record.correctness_verified,'diagnostic_unpadded':row.get('diagnostic_unpadded',False),'invocations':[]}
                compiled=getattr(adapter,'compiled',None)
                if compiled is not None:
                    ctx=compiled.context;torch=ctx.torch;atol,rtol=op.tolerance(operator.dtype);reference=compiled.prepared.reference['out']
                    modes=['reuse','reset'] if 'flute_streamk' in ctx.workspace else ['reuse']
                    with COORDINATOR.measurement_window(hardware):
                        for mode in modes:
                            for index in range(4):
                                if mode=='reset':ctx.workspace['flute_streamk'].zero_()
                                adapter._invoke(compiled);adapter._synchronize()
                                actual=ctx.outputs['out'];delta=(actual.float()-reference.float()).abs();finite=torch.isfinite(actual)
                                bad=(~finite)|(delta>atol+rtol*reference.float().abs());coords=bad.nonzero()[:5]
                                samples=[]
                                for coord in coords.tolist():
                                    key=tuple(coord);av=float(actual[key]);rv=float(reference[key]);samples.append({'index':coord,'actual':av if torch.isfinite(actual[key]).item() else None,'reference':rv})
                                result['invocations'].append({'workspace':mode,'iteration':index,'bad_elements':int(bad.sum()),'nonfinite_elements':int((~finite).sum()),
                                    'max_absolute_error':float(delta.max()) if finite.all().item() else None,'samples':samples})
                    result['input_integrity']=adapter._input_integrity(compiled)[0]
                    result['execution']=ctx.metadata.get('experiment_annotations',{})
                state['cases'].append(result);write(out/'state.json',state)
                del compiled,adapter;gc.collect()
            store.export_jsonl(out/'records.jsonl')
        state['status']='completed';state['registered_producers_after_close']=len(COORDINATOR.participants)
    except Exception as error:state.update(status='failed',error=type(error).__name__+': '+str(error));raise
    finally:gemlite.pad_split_metadata=original_padding;write(out/'state.json',state)

if __name__=='__main__':main()
