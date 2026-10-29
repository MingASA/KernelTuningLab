"""SDK strategies driven by recorded real-model output, not a live GPU service.

Prediction portfolios never use target feedback to reorder candidates. Generated
heuristics may use only the revealed history. Their expression language has no
file access, imports, attributes or arbitrary Python statements. Scores are
preferences, never measured/predicted latency records.
"""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path

from .sdk import SearchAlgorithmDefinition,SearchCapabilities,KernelConfig,SpaceBuildBudget
from .models import canonical_json


TILE_KEYS=('BLOCK_M','BLOCK_N','BLOCK_K','num_warps','num_stages','GROUP_M')
PARAMETERS=(*TILE_KEYS,'SPLIT_K','DEQUANT_BLOCK_SIZE')
FEATURES=('M','N','K','SM','L2_BYTES','L2_KNOWN','SMEM_BYTES','SMEM_KNOWN',
    'IS_QUANTIZED','IS_PIPELINE','OBSERVATIONS','BEST_MS','NEAREST_MS','NEAREST_DISTANCE',
    'INCUMBENT_DISTANCE','UNCERTAINTY','RANDOM')
FUNCTION_ARITY={'min':2,'max':2,'where':3,'abs':1,'log2':1,'sqrt':1,'ceil':1,'floor':1,'exp':1,'clip':3}


def validate_expression(text):
    if not isinstance(text,str) or not text or len(text)>4096:raise ValueError('bounded score_expression required')
    tree=ast.parse(text,mode='eval');nodes=list(ast.walk(tree))
    if len(nodes)>256:raise ValueError('generated expression too complex')
    allowed=(ast.Expression,ast.BinOp,ast.UnaryOp,ast.Compare,ast.Call,ast.Name,ast.Load,ast.Constant,
        ast.Add,ast.Sub,ast.Mult,ast.Div,ast.FloorDiv,ast.Mod,ast.Pow,ast.USub,ast.UAdd,
        ast.Lt,ast.LtE,ast.Gt,ast.GtE,ast.Eq,ast.NotEq)
    names=set()
    for node in nodes:
        if not isinstance(node,allowed):raise ValueError('unsupported generated syntax: '+type(node).__name__)
        if isinstance(node,ast.Name):
            if node.id not in (*PARAMETERS,*FEATURES,*FUNCTION_ARITY):raise ValueError('unknown generated feature: '+node.id)
            names.add(node.id)
        if isinstance(node,ast.Constant) and (type(node.value) not in (int,float) or abs(node.value)>1e12):
            raise ValueError('only bounded numeric constants are allowed')
        if isinstance(node,ast.Call):
            if not isinstance(node.func,ast.Name) or node.func.id not in FUNCTION_ARITY or node.keywords or len(node.args)!=FUNCTION_ARITY[node.func.id]:
                raise ValueError('generated function/arity not allowed')
        if isinstance(node,ast.Compare) and len(node.ops)!=1:raise ValueError('use where() with individual comparisons')
        if isinstance(node,ast.BinOp) and isinstance(node.op,ast.Pow):
            exponent=node.right
            if isinstance(exponent,ast.UnaryOp) and isinstance(exponent.op,ast.USub):exponent=exponent.operand
            if not isinstance(exponent,ast.Constant) or type(exponent.value) not in (int,float) or abs(exponent.value)>4:
                raise ValueError('power exponent must be a constant within[-4,4]')
    return compile(tree,'<model_generated_preference>','eval'),names


def _artifact(path,sha,kind):
    file=Path(path);blob=file.read_bytes()
    if hashlib.sha256(blob).hexdigest()!=sha:raise ValueError('LLM output artifact differs from frozen SHA256')
    value=json.loads(blob)
    if value.get('kind')!=kind or value.get('generation_evidence',{}).get('evidence_level')!='real_local_model_inference':
        raise ValueError('a recorded real-local-model artifact is required; protocol fixtures are not accepted')
    return value


def task_key(hardware,operator,shape,space):
    return hashlib.sha256(canonical_json([hardware.id,operator.id,shape.id,space.fingerprint]).encode()).hexdigest()


class FrozenPrediction:
    def __init__(self,seed=0,*,artifact,artifact_sha256):
        self.value=_artifact(artifact,artifact_sha256,'llm_static_predictions');self.sha=artifact_sha256
        self.diagnostics={'mode':'static real-LLM portfolio; no target latency feedback used for selection',
            'precomputed_model_cost':self.value.get('costs',{}),'returned':0}
    @property
    def runtime_identity(self):return {'artifact_sha256':self.sha,**self.value['generation_evidence']}
    def suggest(self,hardware,operator,shape,space,history,budget,batch_size=1):
        if budget.exhausted(0,0):return []
        key=task_key(hardware,operator,shape,space)
        if key not in self.value['tasks']:raise ValueError('real-LLM predictions do not cover this exact task/protocol')
        configs=[space.normalize(c) for c in self.value['tasks'][key]['configs']]
        seen={r.config.id for r in history}|set(history.attempted_config_ids)
        result=[]
        for c in configs:
            if not space.validate(c,hardware,operator,shape).valid:raise ValueError('frozen model prediction is not statically legal')
            if c.id not in seen:result.append(c);seen.add(c.id)
            if len(result)>=min(batch_size,budget.max_evaluations):break
        self.diagnostics['returned']+=len(result)
        if not result:self.last_proposal_status='frozen_prediction_portfolio_exhausted'
        return result


class GeneratedRule:
    def __init__(self,seed=0,*,artifact,artifact_sha256):
        self.value=_artifact(artifact,artifact_sha256,'llm_generated_rule');self.sha=artifact_sha256;self.seed=seed
        self.program,self.names=validate_expression(self.value['score_expression']);self.cache={}
        self.diagnostics={'source':'actual model-generated expression; no agent rewrite',
            'score_units':'preference only, not latency','precomputed_model_cost':self.value.get('costs',{}),
            'suggestions':0,'nonfinite_scores':0}
    @property
    def runtime_identity(self):return {'artifact_sha256':self.sha,**self.value['generation_evidence']}
    def suggest(self,hardware,operator,shape,space,history,budget,batch_size=1):
        if budget.exhausted(0,0):return []
        import numpy as np
        if set(space.domains)-set(PARAMETERS):raise ValueError('generated-rule executor supports declared numeric GEMM parameters only')
        if hardware.architecture not in self.value['scope']['architectures'] or operator.implementation not in self.value['scope']['kernel_refs']:
            raise ValueError('generated rule outside its declared hardware/kernel scope')
        key=task_key(hardware,operator,shape,space)
        if key not in self.cache:
            built=space.build_feasible(hardware,operator,shape,budget=SpaceBuildBudget(max_nodes=500000,max_wall_time_s=15,max_configs=50000))
            if not built.complete:raise ValueError('this generated-rule executor requires a complete bounded static set: '+built.status)
            configs=built.configs
            if not configs:self.last_proposal_status=built.status;return []
            features={name:np.asarray([c.get(name,1 if name=='SPLIT_K' else 0) for c in configs],dtype=np.float64) for name in PARAMETERS}
            matrix=np.stack([features[name] for name in space.domains],axis=1)
            span=np.maximum(matrix.max(0)-matrix.min(0),1)
            self.cache[key]=(configs,features,matrix,span)
            self.diagnostics['space_build']=dict(built.diagnostics)
        configs,features,matrix,span=self.cache[key]
        values={**features,'M':shape.get('M'),'N':shape.get('N'),'K':shape.get('K'),
            'SM':hardware.sm_count,'L2_BYTES':hardware.l2_bytes or 0,'L2_KNOWN':int(bool(hardware.l2_bytes)),
            'SMEM_BYTES':hardware.smem_per_block,'SMEM_KNOWN':int(bool(hardware.smem_per_block)),
            'IS_QUANTIZED':int('a16_gemm' in operator.kind),'IS_PIPELINE':int('pipeline' in operator.implementation),
            'OBSERVATIONS':len(history),'BEST_MS':min((r.median_ms for r in history if r.correctness_verified),default=0.)}
        needed=self.names & {'NEAREST_MS','NEAREST_DISTANCE','INCUMBENT_DISTANCE','UNCERTAINTY'}
        if needed:
            distance=np.ones(len(configs));nearest=np.zeros(len(configs));incumbent=np.ones(len(configs));best=None
            for r in history:
                if not r.correctness_verified:continue
                point=np.asarray([r.config.get(k) for k in space.domains],dtype=np.float64)
                d=np.mean(((matrix-point)/span)**2,axis=1)
                if best is None:distance[:]=np.inf
                mask=d<distance;nearest[mask]=r.median_ms;distance=np.minimum(distance,d)
                if best is None or r.median_ms<best.median_ms:best=r;incumbent=d
            values.update(NEAREST_MS=nearest,NEAREST_DISTANCE=distance,INCUMBENT_DISTANCE=incumbent,UNCERTAINTY=distance)
        if 'RANDOM' in self.names:values['RANDOM']=np.random.default_rng(self.seed+len(history)*1000003).random(len(configs))
        functions={'min':np.minimum,'max':np.maximum,'where':np.where,'abs':np.abs,'log2':np.log2,
            'sqrt':np.sqrt,'ceil':np.ceil,'floor':np.floor,'exp':np.exp,'clip':np.clip}
        with np.errstate(all='ignore'):
            scores=np.broadcast_to(np.asarray(eval(self.program,{'__builtins__':{},**functions},values),dtype=float),(len(configs),)).copy()
        seen={r.config.id for r in history}|set(history.attempted_config_ids)
        finite=np.isfinite(scores);self.diagnostics['nonfinite_scores']+=int((~finite).sum())
        if not finite.any():raise ValueError('generated rule produced no finite ranking; not a proof of an empty space')
        scores[~finite]=np.inf
        order=np.argsort(scores,kind='stable');result=[]
        for index in order:
            c=configs[int(index)]
            if c.id in seen:continue
            result.append(c);seen.add(c.id)
            if len(result)>=min(batch_size,budget.max_evaluations):break
        self.diagnostics['suggestions']+=len(result)
        self.diagnostics['target_feedback_records']=len(history)
        return result


CAPABILITIES=SearchCapabilities(required_parameters=TILE_KEYS,parameter_kinds=('integer','real','boolean'))
ASSETS=(SearchAlgorithmDefinition('llm/frozen_prediction','1.0.0',FrozenPrediction,CAPABILITIES),
        SearchAlgorithmDefinition('llm/generated_rule','1.0.0',GeneratedRule,CAPABILITIES))
