"""Synthetic training labels test provenance; they are never GPU evidence."""
from dataclasses import replace

from kernel_tuning_lab.asset_context import AssetCase
from kernel_tuning_lab.asset_search import search_case
from kernel_tuning_lab.asset_space import build_effective_space
from kernel_tuning_lab.catalog import get_catalog
from kernel_tuning_lab.kernel_runtime import AssetKernelAdapter
from kernel_tuning_lab.models import KernelConfig, MeasurementRecord, OperatorSpec, RecordStatus, SearchBudget, ShapeSpec
from kernel_tuning_lab.storage import ExperimentStore


def test_mutating_training_database_changes_online_trial_identity_and_preserves_source_ids(tmp_path):
    catalog=get_catalog();kernel=catalog.resolve('kernel','python/gemm@1.0.0');definition=catalog.resolve('operator','gemm@1.0.0')
    adapter=AssetKernelAdapter(kernel,definition,accept_claims=False);hardware=adapter.probe_hardware()
    operator=OperatorSpec('gemm',kind='gemm',implementation=kernel.ref,dtype='float32',version='1.0.0')
    shape=ShapeSpec({'M':2,'N':3,'K':4})
    space=build_effective_space(kernel,hardware,operator,shape)
    case=AssetCase(kernel,definition,hardware,operator,shape,space,adapter)
    source=MeasurementRecord(hardware,operator,ShapeSpec({'M':3,'N':4,'K':5}),KernelConfig({'tile_size':1}),
        RecordStatus.SUCCESS,runtime_samples_ms=(1.,),wall_time_s=.25,provenance='mock',split='train',
        metadata={'correctness_verified':True})
    database=tmp_path/'training.sqlite'
    observations=tmp_path/'observations.sqlite'
    def run(directory):
        with ExperimentStore(observations) as store:
            return search_case(case,catalog.resolve('algorithm','transfer@1.0.0'),{'training_database':str(database)},
                SearchBudget(1,10),store=store,seed=7,warmup=0,repetitions=1,split='test',
                experiment_id='same-experiment',output_dir=tmp_path/directory)
    with ExperimentStore(database) as store:store.add(source)
    first=run('first')
    second_source=replace(source,config=KernelConfig({'tile_size':4}),runtime_samples_ms=(.1,))
    with ExperimentStore(database) as store:store.add(second_source)
    second=run('second')
    assert first['status']==second['status']=='passed'
    assert first['trial_id']!=second['trial_id'],'changed historical labels must not keep the same trial identity'
    assert first['training_data']['record_ids']==[source.id]
    assert set(second['training_data']['record_ids'])=={source.id,second_source.id}
    assert second['training_data']['unique_source_wall_s']==.5
    from kernel_tuning_lab.analysis.research import _trial
    job={'id':'unit','algorithm':'transfer@1.0.0','case':'unit','operator':'gemm@1.0.0','kernel':kernel.ref,
         'seed':7,'dtype':'float32','budget':{'max_evaluations':1,'max_wall_time_s':10},
         'experiment':{'parameters':{},'algorithms':[{'asset':'transfer@1.0.0','options':first['options']}]}}
    with ExperimentStore(observations,read_only=True) as store:
        before=_trial(tmp_path,job,first,store,tmp_path/'first')
        after=_trial(tmp_path,job,second,store,tmp_path/'second')
    assert before['series_id']!=after['series_id'],'reports must not pool different historical datasets'
