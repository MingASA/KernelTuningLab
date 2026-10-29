"""Finite production configuration tables, sharing the L3 ShapeDispatcher.

Table assignments never scan measured performance during deployment. They
select one already observed config in one implementation/semantic context.
Unseen shapes reject; native fallback is a separate expert execution and can
exceed this budget. Config count never stands in for compiled/code/Graph size.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path

from ..sdk import KernelConfig,SearchAlgorithmDefinition,SearchCapabilities
from ..models import ShapeSpec,canonical_json
from ..l3 import ShapeDispatcher
from ..l3.dispatcher import UnknownShapeError


class ProductionTable:
    def __init__(self,seed=0,*,table_artifact,table_sha256):
        path=Path(table_artifact).resolve();blob=path.read_bytes()
        if hashlib.sha256(blob).hexdigest()!=table_sha256:raise ValueError('deployment table SHA mismatch')
        self.binding=json.loads(blob);self.sha=table_sha256;table=self.binding['dispatch']
        if self.binding.get('unknown_input','reject')!='reject':raise ValueError('unsupported implicit native fallback; use a separately recorded expert call')
        configs=tuple(KernelConfig(c) for c in table['selected_configs'])
        if len(configs)!=len({c.id for c in configs}) or len(configs)>self.binding['config_budget']:raise ValueError('invalid Config Budget')
        assignments={}
        for entry in table['assignments']:
            shape=ShapeSpec(entry['shape']);key=tuple(sorted(shape.dims.items()));config=KernelConfig(entry['config'])
            if key in assignments or config.id!=entry['config_id']:raise ValueError('ambiguous deployment assignment')
            assignments[key]=config
        self.dispatcher=ShapeDispatcher(assignments,selected_configs=configs,provenance=table['provenance'])
        self.diagnostics={'table_lookups':0,'config_budget':self.binding['config_budget'],'selected_configs':len(configs),
            'physical_compiled_variants':self.binding.get('physical_compiled_variants'),
            'code_bytes':self.binding.get('code_bytes'),'graphs':self.binding.get('graphs'),
            'packing_artifacts':self.binding.get('packing_artifacts'),'unknown_input':'reject'}
    @property
    def runtime_identity(self):return {'deployment_table_sha256':self.sha,'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    def select(self,hardware,operator,shape,space):
        if hardware.id!=self.binding['hardware_id'] or operator.id!=self.binding['operator_id']:
            raise ValueError('deployment hardware/operator binding mismatch')
        if space.asset_identity.get('kernel_fingerprint')!=self.binding['kernel_fingerprint']:
            raise ValueError('deployment implementation binding mismatch')
        config=self.dispatcher(shape)
        validation=space.validate(config,hardware,operator,shape)
        if not validation.valid:raise ValueError('stored assignment no longer valid: '+';'.join(validation.reasons))
        self.diagnostics['table_lookups']+=1;return config
    def suggest(self,hardware,operator,shape,space,history,budget,batch_size=1):
        if budget.exhausted(0,0) or history or history.attempted_config_ids:return []
        # SDK synthetic acceptance has no production identity and is explicitly
        # not a deployment run. Actual bind_trial never takes this branch.
        if getattr(self,'output',None) is None:
            return list(space.complete({},hardware,operator,shape,count=1).configs)
        return [self.select(hardware,operator,shape,space)]
    def bind_trial(self,*,output_dir,identity):self.output=Path(output_dir)


def export_table(path,matrix,result,*,hardware,operator,space,kernel_fingerprint):
    dispatcher=ShapeDispatcher.from_result(matrix,result,hardware=hardware,operator=operator,space=space)
    value={'schema_version':1,'hardware_id':hardware.id,'operator_id':operator.id,
        'kernel_fingerprint':kernel_fingerprint,'config_budget':result.k,'unknown_input':'reject',
        'dispatch':dispatcher.to_dict(),'physical_compiled_variants':None,'code_bytes':None,'graphs':None,
        'packing_artifacts':None,'scope':'finite observed workload, single implementation/protocol; unknown metrics not inferred from K'}
    path=Path(path);path.write_text(canonical_json(value)+'\n');return hashlib.sha256(path.read_bytes()).hexdigest()


ASSETS=(SearchAlgorithmDefinition('production/config_table','1.0.0',ProductionTable,SearchCapabilities(),
    metadata={'scope':'exact finite-workload dispatch through existing Runner; unknown inputs reject','budget_kind':'config count only'}),)
