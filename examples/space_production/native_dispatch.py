"""Finite native-option deployment, using the real ShapeDispatcher before launch.

The artifact contains selected options and bindings, no latency/oracle table.
Unknown inputs are rejected. This is a deployment rule, not a search algorithm.
"""
import hashlib
import json
from pathlib import Path

from kernel_tuning_lab.sdk import KernelConfig,SearchAlgorithmDefinition,SearchCapabilities
from kernel_tuning_lab.models import ShapeSpec
from kernel_tuning_lab.l3 import ShapeDispatcher


class NativeDispatch:
    def __init__(self,seed=0,*,table_artifact,table_sha256):
        path=Path(table_artifact)
        if not path.is_absolute():path=Path(__file__).resolve().parent/path
        blob=path.read_bytes()
        if hashlib.sha256(blob).hexdigest()!=table_sha256:raise ValueError('deployment table SHA mismatch')
        self.binding=json.loads(blob);table=self.binding['dispatch'];self.table_sha=table_sha256
        selected=tuple(KernelConfig(c) for c in table['selected_configs'])
        if len(selected)>self.binding['config_budget']:raise ValueError('deployment exceeds Config Budget')
        assignments={}
        for entry in table['assignments']:
            shape=ShapeSpec(entry['shape']);key=tuple(sorted(shape.dims.items()));config=KernelConfig(entry['config'])
            if key in assignments or config.id!=entry['config_id']:raise ValueError('ambiguous deployment assignment')
            assignments[key]=config
        self.dispatcher=ShapeDispatcher(assignments,selected_configs=selected,provenance=table['provenance'])
        self.diagnostics={'actual_table_lookups':0,'config_budget':self.binding['config_budget'],
            'selected_config_count':len(selected),'unseen_input':'reject without native fallback or budget expansion'}
    @property
    def runtime_identity(self):
        return {'deployment_table_sha256':self.table_sha,
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'selection_cost_scope':'host lookup outside GPU Graph; charged to proposal wall'}
    def select(self,hardware,operator,shape,space):
        if hardware.id!=self.binding['hardware_id'] or operator.id!=self.binding['operator_id']:
            raise ValueError('deployment hardware/operator binding mismatch; rebuild from compatible measurements')
        if space.asset_identity.get('kernel_fingerprint')!=self.binding['kernel_fingerprint']:
            raise ValueError('deployment native binary/source binding mismatch')
        config=self.dispatcher.select_config(shape)
        if not space.validate(config,hardware,operator,shape).valid:raise ValueError('deployment is outside current legal native candidates')
        self.diagnostics['actual_table_lookups']+=1
        return config
    def suggest(self,hardware,operator,shape,space,history,budget,batch_size=1):
        if budget.exhausted(0,0) or history:return []
        return [self.select(hardware,operator,shape,space)]


ASSETS=(SearchAlgorithmDefinition('production/finite_dispatch','1.0.0',NativeDispatch,
    SearchCapabilities(parameter_kinds=('boolean',),required_parameters=('USE_ATOMIC_ADD','USE_FP32_REDUCE')),
    metadata={'scope':'finite observed workload; actual ShapeDispatcher host lookup; rejects unseen shapes',
        'budget_kind':'config_count, not CUBIN or Graph count'}),)
