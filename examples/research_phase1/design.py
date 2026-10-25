"""Frozen, outcome-independent experiment design; regenerates checked-in JSON."""
import itertools
import json
from pathlib import Path

SPACE = {"BLOCK_M": [32, 128], "BLOCK_N": [32, 128], "BLOCK_K": [32, 64],
         "num_warps": [4, 8], "num_stages": [2, 3], "GROUP_M": [1, 8]}
# Whole Shape cases, never configuration-level train/test splits.
SHAPES = [
    ("decode-l2", "train", (1, 2048, 2048), "8 MiB B within 32 MiB L2; very low arithmetic intensity"),
    ("small-tokens", "train", (16, 4096, 2048), "16 MiB B; more reuse with still sparse M tiles"),
    ("balanced-medium", "train", (512, 1024, 1024), "Tensor Core throughput and CTA occupancy"),
    ("tall", "train", (2048, 512, 1024), "many M tiles, grouped ordering and rectangular reuse"),
    ("decode-beyond-l2", "validation", (1, 8192, 4096), "64 MiB B exceeds L2; memory-sensitive decode"),
    ("irregular", "validation", (257, 769, 513), "tails in all axes, last partial program group"),
    ("few-tokens-wide", "test", (8, 8192, 2048), "32 MiB B at L2 boundary, wide output and small M"),
    ("prefill", "test", (128, 2048, 2048), "transition from memory traffic to reuse"),
    ("compute-large", "test", (1024, 2048, 2048), "compute-heavy multi-wave workload"),
    ("wide-k-small", "test", (512, 4096, 512), "wide N / small K; loop overhead and group ordering"),
]


def configs():
    return [dict(zip(SPACE, values)) for values in itertools.product(*SPACE.values())]


def spec(name, cases, *, count, seeds=(7, 19, 31), algorithms=None, dtypes=("float16",)):
    return {"schema_version": 1, "kind": "research", "name": name, "plugins": ["cohort.py"],
        "algorithms": algorithms or ["phase1/cohort@1.0.0"], "dtypes": list(dtypes), "seeds": list(seeds),
        "budgets": [{"max_evaluations": count, "max_wall_time_s": 180}],
        "defaults": {"warmup": 10, "repetitions": 31, "timing_mode": "cuda_graph", "cuda_graph_batch_size": 512,
            "preflight_seconds": 60, "validation": {"max_configs": 4, "max_cases": 0, "max_wall_time_s": 90, "trust_claims": False}},
        "cases": cases}


def case(entry, kernel="triton/t1_gemm@1.0.0", space=SPACE):
    name, split, dims, _ = entry
    return {"name": name, "split": split, "operator": "gemm@1.0.0",
            "kernels": [{"asset": kernel, "space": space}], "shapes": [dict(zip(("M", "N", "K"), dims))]}


def generate(directory):
    directory = Path(directory)
    # Factor-spanning, predetermined pilot: include minimum/maximum resources.
    chosen = [configs()[i] for i in (0, 3, 5, 10, 17, 22, 29, 34, 41, 47, 58, 63)]
    selected = [SHAPES[i] for i in (0, 2, 4, 5)]
    outputs = {
        "pilot.json": spec("t1-pilot-v1", [case(s) for s in selected], count=12, seeds=(7, 19),
                           algorithms=[{"asset": "phase1/cohort@1.0.0", "options": {"configs": chosen}}]),
        "matrix.json": spec("t1-matrix-v1", [case(s) for s in SHAPES], count=64),
        "bf16.json": spec("t1-bf16-check-v1", [case(SHAPES[i]) for i in (0, 2, 5, 8)], count=12,
                          seeds=(7, 19), dtypes=("bfloat16",),
                          algorithms=[{"asset": "phase1/cohort@1.0.0", "options": {"configs": chosen}}]),
        "library.json": spec("t1-library-v1", [case(s, "torch/mm@1.0.0", {}) for s in SHAPES], count=1),
        "simple.json": spec("t1-simple-control-v1", [case(s, "triton/gemm@1.0.0",
                {k: v for k, v in SPACE.items() if k != "GROUP_M"}) for s in SHAPES], count=8,
                algorithms=[{"asset": "phase1/cohort@1.0.0", "options": {"configs": [
                    {k:v for k,v in c.items() if k!='GROUP_M'} for c in chosen[::2]]}}]),
        "correctness.json": spec("t1-boundaries-v1", [{"name": name, "operator": "gemm@1.0.0",
            "kernels": [{"asset": "triton/t1_gemm@1.0.0", "space": SPACE}],
            "shapes": [dict(zip(("M", "N", "K"), dims)) for dims in [(1,1,1),(15,17,31),(33,129,65),(513,71,129)]]}
            for name in ["tails"]], count=12, seeds=(7, 19), dtypes=("float16","bfloat16"),
            algorithms=[{"asset":"phase1/cohort@1.0.0","options":{"configs":chosen}}])}
    # Pilot: beyond-L2 decode averaged 5.3 s/candidate (64 * 5.3 > 180 s).
    # Preserve sample quality; only increase the explicit admission deadline.
    outputs["matrix.json"]["budgets"][0]["max_wall_time_s"]=600
    outputs['train-only.json']=json.loads(json.dumps(outputs['matrix.json']))
    outputs['train-only.json']['cases']=[c for c in outputs['train-only.json']['cases'] if c['split']=='train']
    for filename, value in outputs.items():
        (directory / filename).write_text(json.dumps(value,indent=2)+'\n')
    anchors=[configs()[0], configs()[-1]]
    for phase in ("before","after"):
        value=spec("t1-anchor-"+phase+"-v1",[case(SHAPES[i]) for i in (0,4,8)],count=2,
            algorithms=[{"asset":"phase1/cohort@1.0.0","options":{"configs":anchors}}])
        (directory/('anchor-'+phase+'.json')).write_text(json.dumps(value,indent=2)+'\n')
    # Match T1's M/N/group coordinates where possible; Split-K is an additional
    # real quantized-GEMM parameter, never merge the two implementation spaces.
    quant={"BLOCK_M":[32,128],"BLOCK_N":[128],"BLOCK_K":[32],"num_warps":[4],"num_stages":[2],"GROUP_M":[1,8],"SPLIT_K":[1,2,4]}
    value=spec("w4-phase1-v1",[{"name":"w4","operator":"w4a16_gemm@1.0.0","parameters":{"group_size":32},
        "kernels":[{"asset":"triton/w4a16_gemm@1.0.0","space":quant},
            {"asset":"triton_pipeline/w4a16_gemm@1.0.0","space":{**quant,"DEQUANT_BLOCK_SIZE":[512]}}],
        "shapes":[{"M":m,"N":n,"K":2048} for m,n in [(1,2048),(128,2048),(512,1024)]]}],count=12,seeds=(7,19))
    (directory/'w4.json').write_text(json.dumps(value,indent=2)+'\n')
    value=spec('t1-online-check-v1',[case(SHAPES[i]) for i in (6,8)],count=8,seeds=(7,),
        algorithms=['random@1.0.0','heuristic@1.0.0','surrogate@1.0.0'])
    (directory/'online.json').write_text(json.dumps(value,indent=2)+'\n')
    # Independent middle-tile quality check. Keep it separate from the complete
    # 64-config factorial so missing cells are never silently filled or pooled.
    expert_space={**SPACE,'BLOCK_M':[16,32,64,128],'BLOCK_N':[32,64,128]}
    experts=[{'BLOCK_M':16,'BLOCK_N':32,'BLOCK_K':32,'num_warps':4,'num_stages':2,'GROUP_M':1}]+[
        {'BLOCK_M':bm,'BLOCK_N':bn,'BLOCK_K':bk,'num_warps':warps,'num_stages':stages,'GROUP_M':group}
        for bm,bn,bk,warps,stages,group in [(64,64,32,4,3,8),(64,128,64,4,3,8),(128,64,64,4,3,8),(64,64,64,8,2,1),(16,64,64,4,3,1)]]
    value=spec('t1-middle-tile-check-v1',[case(SHAPES[i],space=expert_space) for i in (0,2,5,8)],count=6,
        algorithms=[{'asset':'phase1/cohort@1.0.0','options':{'configs':experts}}])
    (directory/'expert-probe.json').write_text(json.dumps(value,indent=2)+'\n')
    (directory / "shape-design.json").write_text(json.dumps({"schema_version":1,"candidate_space":SPACE,
        "cases":[{"name":name,"split":split,"shape":dict(zip(("M","N","K"),dims)),"hypothesis":reason,
            "fp16_inputs_output_bytes":2*(dims[0]*dims[2]+dims[2]*dims[1]+dims[0]*dims[1])} for name,split,dims,reason in SHAPES]},indent=2)+'\n')
    cross=directory/'cross-hardware';cross.mkdir(exist_ok=True)
    for label,profile in [('h20',None),('h200','h200'),('h800','h800'),('4090','4090')]:
        value=json.loads(json.dumps(outputs['matrix.json']))
        value['name']='t1-matrix-v1-'+label
        value['plugins']=['../cohort.py']
        value['defaults']['hardware']=profile
        (cross/(label+'.json')).write_text(json.dumps(value,indent=2)+'\n')


if __name__ == "__main__":
    generate(Path(__file__).parent)
