"""Reproducible A/B/C analysis of exact v2 online observations, using stdlib.

No measured cells are imputed. Replay uses the original Runner records and the
existing private-oracle evaluator. Compiled artifacts are counted separately
from config dictionaries. Analysis never executes CUDA or imports GPU SDKs.
"""
from collections import Counter, defaultdict
from dataclasses import replace
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random
import statistics as st
from html import escape

from kernel_tuning_lab.analysis.report import _bar_chart, _heatmap, _line_chart, _table, _svg, _text
from kernel_tuning_lab.l3 import PerformanceMatrix, OracleLimitError, greedy, greedy_local_swap, oracle
from kernel_tuning_lab.models import KernelConfig, SearchBudget, ShapeSpec, WorkloadDistribution, canonical_json
from kernel_tuning_lab.replay import run_replay
from kernel_tuning_lab.research import read_json, snapshot_directory
from kernel_tuning_lab.search import make_searcher
from kernel_tuning_lab.space import ConfigSpace
from kernel_tuning_lab.storage import ExperimentStore

ROOT = Path(__file__).resolve().parents[1]
module_spec = importlib.util.spec_from_file_location("phase1_design", ROOT/"examples/research_phase1/design.py")
design = importlib.util.module_from_spec(module_spec)
module_spec.loader.exec_module(design)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+"\n")


def quantile(values, q):
    values=sorted(values)
    if not values: return None
    offset=(len(values)-1)*q; low=math.floor(offset); high=math.ceil(offset)
    return values[low]+(values[high]-values[low])*(offset-low)


def current_records(directory):
    """Only this suite's actually revealed IDs; no journal-wide best query."""
    directory=Path(directory); manifest=read_json(directory/"research_manifest.json")
    records={}
    for job in manifest["jobs"]:
        snapshot=snapshot_directory(directory,job)
        if snapshot is None or not (snapshot/"trials.json").is_file(): continue
        with ExperimentStore(directory/job["output"]/"measurements.sqlite",read_only=True) as store:
            for trial in read_json(snapshot/"trials.json"):
                for identity in trial["record_ids"]:
                    record=store.get_record(identity)
                    if record is None or record.id!=identity: raise ValueError("missing/corrupt exact measurement ID")
                    if record.operator.implementation!=trial["kernel_ref"]: raise ValueError("Kernel identity mismatch")
                    records[identity]=record
    return tuple(records.values())


def ranks(values):
    ordered=sorted(range(len(values)),key=lambda i: values[i]); result=[0.]*len(values); pos=0
    while pos<len(ordered):
        end=pos+1
        while end<len(ordered) and values[ordered[end]]==values[ordered[pos]]: end+=1
        for i in ordered[pos:end]: result[i]=(pos+end-1)/2+1
        pos=end
    return result


def correlation(left,right):
    a,b=ranks(left),ranks(right); am,bm=st.mean(a),st.mean(b)
    numerator=sum((x-am)*(y-bm) for x,y in zip(a,b))
    denominator=math.sqrt(sum((x-am)**2 for x in a)*sum((y-bm)**2 for y in b))
    return numerator/denominator if denominator else None


def correlation_plot(path,labels,values):
    parts=[];left,top,cell=180,160,60
    for i,name in enumerate(labels):
        parts.append(_text(left-10,top+i*cell+32,name,size=11,anchor="end"))
        parts.append(_text(left+i*cell+15,top-15,name,size=11,extra=f'transform="rotate(-50 {left+i*cell+15} {top-15})"'))
        for j,value in enumerate(values[i]):
            v=value or 0; mix=abs(v)
            color=f'rgb({int(245-180*max(v,0))},{int(245-120*mix)},{int(245-180*max(-v,0))})'
            parts += [f'<rect x="{left+j*cell}" y="{top+i*cell}" width="{cell-2}" height="{cell-2}" fill="{color}"><title>{escape(name)} / {escape(labels[j])}: {value}</title></rect>',
                _text(left+j*cell+cell/2,top+i*cell+32,f"{value:.2f}" if value is not None else "unknown",size=11,anchor="middle")]
    _svg(path,"T1 cross-Shape rank correlation","Spearman rho from -1 (opposite) to +1 (same); 64 finite measured configs",
        900,top+len(labels)*cell+40,parts,render_png=False)


def require_baseline_compatible(target, baseline):
    if (target.hardware.id,target.operator.dtype,target.operator.kind,target.operator.parameters,target.measurement_mode,target.warmup,target.repetitions) != (
        baseline.hardware.id,baseline.operator.dtype,baseline.operator.kind,baseline.operator.parameters,baseline.measurement_mode,baseline.warmup,baseline.repetitions):
        raise ValueError("baseline hardware/semantics/dtype/timing/repetition mismatch")
    for key in ("python","torch","cuda_runtime","nvidia_driver","runtime_source_sha256","operator_fingerprint"):
        if target.environment.get(key)!=baseline.environment.get(key): raise ValueError("baseline software/reference mismatch: "+key)
    for key in ("cuda_graph","tolerances","input_contract","measurement_scope","measurement_driver_source_sha256"):
        if target.metadata["protocol"].get(key)!=baseline.metadata["protocol"].get(key): raise ValueError("baseline protocol mismatch: "+key)


def describe(matrix, records, labels):
    if not matrix.complete: raise ValueError("structural ranking needs complete observed finite matrix; inspect coverage.json")
    rows=matrix.latencies_ms; configs=matrix.configs; baselines=[min(row) for row in rows]
    by_cell=defaultdict(list)
    for record in records:
        if record.correctness_verified: by_cell[record.shape.id,record.config.id].append(record.median_ms)
    cv={}; sensitivity=[]; overlaps={}
    for i,shape in enumerate(matrix.workload.shapes):
        row=rows[i]
        cv[labels[i]]=[st.pstdev(values)/st.mean(values) if len(values)>1 else None
                      for config in configs for values in [by_cell[shape.id,config.id]]]
        for key,values in design.SPACE.items():
            lookup={tuple((k,c.params[k]) for k in design.SPACE if k!=key):{} for c in configs}
            for j,c in enumerate(configs): lookup[tuple((k,c.params[k]) for k in design.SPACE if k!=key)][c.params[key]]=row[j]
            ratios=[pair[values[1]]/pair[values[0]] for pair in lookup.values()]
            sensitivity.append({"shape":labels[i],"parameter":key,"low":values[0],"high":values[1],
                "paired_geomean_high_over_low":math.exp(st.mean(math.log(v) for v in ratios)),
                "paired_median_high_over_low":st.median(ratios),"p10":quantile(ratios,.1),"p90":quantile(ratios,.9),
                "pair_count":len(ratios),"high_faster_by_5pct_fraction":st.mean(v<1/1.05 for v in ratios)})
    for fraction in (.05,.1,.2):
        count=math.ceil(len(configs)*fraction)
        top=[set(sorted(range(len(configs)),key=lambda j:(row[j],configs[j].id))[:count]) for row in rows]
        overlaps[str(fraction)]={"top_count":count,"all_shape_intersection":sorted(set.intersection(*top)),
            "sets":{name:sorted(indices) for name,indices in zip(labels,top)},
            "jaccard":[[len(a&b)/len(a|b) for b in top] for a in top]}
    coverage=[{"config_id":config.id,"config":dict(config.params),
        "mean_relative_loss":st.mean(rows[i][j]/baselines[i]-1 for i in range(len(rows))),
        "max_relative_loss":max(rows[i][j]/baselines[i]-1 for i in range(len(rows))),
        "shapes_within_5pct":sum(rows[i][j]<=baselines[i]*1.05 for i in range(len(rows))),
        "shapes_within_10pct":sum(rows[i][j]<=baselines[i]*1.1 for i in range(len(rows)))} for j,config in enumerate(configs)]
    rho=[[correlation(a,b) for b in rows] for a in rows]
    clusters=[(i,) for i in range(len(rows))];merges=[]
    while len(clusters)>1:
        _,a,b=min((st.mean(1-(rho[i][j] or 0) for i in left for j in right),a,b)
            for a,left in enumerate(clusters) for b,right in enumerate(clusters) if a<b)
        left,right=clusters[a],clusters[b]
        distance=st.mean(1-(rho[i][j] or 0) for i in left for j in right)
        merged=tuple(sorted(left+right));merges.append({"shapes":[labels[i] for i in merged],"average_rank_distance":distance})
        clusters=[c for n,c in enumerate(clusters) if n not in (a,b)]+[merged]
    return {"aggregation":matrix.metadata["aggregation"],"sensitivity":sensitivity,"top_overlap":overlaps,
        "spearman":rho,"descriptive_average_linkage":merges,"config_coverage":coverage,
        "relative_loss":[[value/baselines[i]-1 for value in row] for i,row in enumerate(rows)],
        "repeat_cv":cv,"best_by_shape":[{"name":name,"shape":dict(shape.dims),"latency_ms":min(row),
            "config":dict(configs[min(range(len(row)),key=lambda j:row[j])].params),
            "best_config_repeat_medians_ms":by_cell[shape.id,configs[min(range(len(row)),key=lambda j:row[j])].id]}
            for name,shape,row in zip(labels,matrix.workload.shapes,rows)],
        "noise_notice":"CV uses independent run medians; 3 runs are insufficient for precise confidence intervals. Top sets are strict observed ranks; tiny differences are not asserted significant."}


def artifact_counts(records, selected_ids=None, assignments=None):
    identities=set(); binaries={}; unknown=0
    for record in records:
        if not record.correctness_verified: continue
        if selected_ids is not None and record.config.id not in selected_ids: continue
        if assignments is not None and assignments.get(record.shape.id)!=record.config.id: continue
        resources=record.metadata.get("correctness",{}).get("execution",{}).get("compiled_resources",{})
        if not resources: unknown+=1
        for resource in resources.values():
            identity=resource.get("compiler_cache_identity")
            if identity: identities.add(identity)
            else: unknown+=1
            binary=resource.get("artifacts",{}).get("cubin")
            if binary: binaries[binary["sha256"]]=binary["bytes"]
    return {"compiler_specializations":len(identities) if not unknown else None,"known_compiler_specializations":len(identities),
        "distinct_cubin_artifacts":len(binaries),"deduplicated_cubin_container_bytes":sum(binaries.values()) if not unknown else None,
        "unknown_resource_records":unknown,"size_scope":"compiler-produced cubin ELF containers; not instruction-only size, driver residency or packaged deployment size"}


def pareto_indices(rows):
    n=len(rows[0]); keep=[]
    for j in range(n):
        if not any(all(row[k]<=row[j] for row in rows) and
                   (any(row[k]<row[j] for row in rows) or k<j) for k in range(n) if k!=j):
            keep.append(j)
    return keep


def l3_study(matrix,records):
    results=[]; weights={"uniform":[1]*len(matrix.workload.shapes),
        "synthetic_decode_skew":[.25,.15,.05,.03,.20,.02,.15,.07,.05,.03]}
    keep=pareto_indices(matrix.latencies_ms)
    for label,weight in weights.items():
        workload=WorkloadDistribution(matrix.workload.shapes,tuple(weight),name=label)
        table=replace(matrix,workload=workload)
        reduced=PerformanceMatrix(workload,tuple(matrix.configs[j] for j in keep),
            tuple(tuple(row[j] for j in keep) for row in matrix.latencies_ms),matrix.provenance,dict(matrix.metadata))
        for k in (1,2,4,8):
            for name,solver in [("greedy",greedy),("local_swap",greedy_local_swap),("exact",oracle)]:
                try:
                    # Exact Pareto deletion preserves every possible row-wise
                    # minimum; this is an offline oracle, never training input.
                    result=solver(reduced if name=="exact" else table,k)
                except OracleLimitError as error:
                    results.append({"weights":label,"method":name,"k":k,"status":"not_computed","reason":str(error)});continue
                selected={c.id for c in result.configs}
                source=reduced if name=="exact" else table
                assigned={shape.id:source.configs[j].id for shape,j in zip(workload.shapes,result.assignment_indices) if j is not None}
                losses=[min(row[j] for j,c in enumerate(matrix.configs) if c.id in selected)/min(row)-1 for row in matrix.latencies_ms]
                results.append({"weights":label,"weight_values":list(workload.normalized_weights),"method":name,"status":"passed",
                    **result.to_dict(),"pareto_candidates":len(keep) if name=="exact" else None,
                    "mean_shape_relative_loss":st.mean(losses),"max_shape_relative_loss":max(losses),"shape_losses":losses,
                    "assignment_config_ids":assigned,
                    "retained_all_shape_config_pairs":artifact_counts(records,selected),
                    "dispatch_only_artifacts":artifact_counts(records,assignments=assigned)})
    # A small predetermined subset also gives an exact, explicitly limited
    # oracle if full-set K=4/8 exceeds the enumeration guard.
    subset=list(range(0,len(matrix.configs),4))
    small=PerformanceMatrix(matrix.workload,tuple(matrix.configs[j] for j in subset),
        tuple(tuple(row[j] for j in subset) for row in matrix.latencies_ms),matrix.provenance)
    return {"budget_kind":"config_count","shape_count":len(matrix.workload.shapes),"candidate_count":len(matrix.configs),
        "pareto_indices":keep,"results":results,"small_exact_scope":"predetermined every-fourth config; 16 candidates only",
        "small_exact":[oracle(small,k).to_dict() for k in (1,2,4,8)],"all_compiled_artifacts":artifact_counts(records)}


def replay_study(records,space):
    results=[]
    for measurement_seed in (7,19,31):
        rows=[r for r in records if r.seed==measurement_seed]
        training=[r for r in rows if r.split=="train"]
        targets=defaultdict(list)
        for record in rows:
            if record.split in {"validation","test"}: targets[record.case_id].append(record)
        for case in sorted(targets):
            for name in ("random","grid","heuristic","surrogate","transfer"):
                for seed in range(10):
                    for budget in (4,8,16,32):
                        searcher=make_searcher(name,seed=seed,training_records=training)
                        result=run_replay(targets[case],searcher,space,SearchBudget(budget),seed=seed,target_ratio=1.05)
                        row=result.to_dict();row["measurement_seed"]=measurement_seed
                        row["target_10pct_evaluations"]=next((step.evaluation for step in result.trace
                            if step.best_latency_ms is not None and step.best_latency_ms<=result.oracle_best_latency_ms*1.1),None)
                        row.update(new_physical_measurements=0,source_reveals=result.evaluations,
                                   cost_scope="offline reveals of recorded experiments; source cost, not new GPU tuning")
                        results.append(row)
    aggregates=[]
    groups=defaultdict(list)
    for row in results: groups[row["split"],row["algorithm"],row["budget"]].append(row)
    for (split,algorithm,budget),rows in groups.items():
        values=[r["regret"] for r in rows if r["regret"] is not None]
        aggregates.append({"split":split,"algorithm":algorithm,"budget":budget,"trials":len(rows),
            "mean_regret":st.mean(values),"median_regret":st.median(values),"p10":quantile(values,.1),"p90":quantile(values,.9),
            "within_5pct_fraction":st.mean(r["regret"] is not None and r["regret"]<=.05 for r in rows),
            "within_10pct_fraction":st.mean(r["regret"] is not None and r["regret"]<=.1 for r in rows),
            "failed_trials":sum(r["failed_evaluations"]>0 or r["missing_evaluations"]>0 or r["evaluations"]!=budget for r in rows),
            "mean_source_cost_s":st.mean(r["known_measurement_cost_s"] for r in rows)})
    return {"protocol":"separate original measurement replicate per replay; whole-Shape train/validation/test split fixed before collection",
        "training_shapes":[name for name,split,_,_ in design.SHAPES if split=="train"],
        "new_gpu_measurements":0,"results":results,"aggregates":aggregates}


def joint_study(matrix,records):
    """Controlled offline acquisition, all scoring labels private until reveal."""
    labels=[name for name,_,_,_ in design.SHAPES]; train=[i for i,s in enumerate(design.SHAPES) if s[1]=="train"]
    target=[i for i,s in enumerate(design.SHAPES) if s[1]=="test"]
    source=matrix.latencies_ms; n=len(matrix.configs); count=len(target)
    # Frozen before looking at target labels: normalize source Shape costs.
    prior=sorted(range(n),key=lambda j:st.mean(source[i][j]/min(source[i]) for i in train))
    results=[]
    for seed in range(20):
        for budget in (32,64,128):
            for method in ("independent_l2_then_l3","random_common_panel","training_prior_common_panel"):
                seen={}
                def reveal(i,j):
                    if (i,j) not in seen:
                        if len(seen)>=budget: raise ValueError("joint acquisition exceeded observation allowance")
                        seen[i,j]=source[target[i]][j]
                    return seen[i,j]
                rng=random.Random(seed)
                if method=="independent_l2_then_l3":
                    depth=(budget-count*count)//count; candidates=set()
                    for i in range(count):
                        sample=rng.sample(range(n),depth)
                        for j in sample: reveal(i,j)
                        candidates.add(min(sample,key=lambda j:seen[i,j]))
                    candidates=sorted(candidates)
                    # Explicit, charged completion before deployment selection.
                    for i in range(count):
                        for j in candidates: reveal(i,j)
                else:
                    order=prior if method=="training_prior_common_panel" else rng.sample(range(n),n)
                    candidates=order[:budget//count]
                    for i in range(count):
                        for j in candidates: reveal(i,j)
                observed=PerformanceMatrix(WorkloadDistribution(tuple(matrix.workload.shapes[i] for i in target),tuple([1]*count)),
                    tuple(matrix.configs[j] for j in candidates),tuple(tuple(seen[i,j] for j in candidates) for i in range(count)),"measured")
                result=greedy_local_swap(observed,2); selected=[candidates[j] for j in result.selected_indices]
                # Full finite oracle enters only final scoring, never selection.
                losses=[min(source[i][j] for j in selected)/min(source[i])-1 for i in target]
                results.append({"seed":seed,"budget":budget,"method":method,"revealed_cells":len(seen),"k":2,
                    "selected_config_ids":[matrix.configs[j].id for j in selected],"mean_shape_regret":st.mean(losses),
                    "max_shape_regret":max(losses),"acquired_cell_ids":[[labels[target[i]],matrix.configs[j].id] for i,j in sorted(seen)]})
    return {"kind":"offline simulated acquisition over real measured finite matrix; no new GPU execution",
        "target_shapes":[labels[i] for i in target],"training_shapes":[labels[i] for i in train],"results":results,
        "notice":"Equal upper allowances, actual reveals reported. Independent L2 reserves up to 16 completion cells. Common panels can spend more cells exploring configs shared by all target Shapes. Prior uses training Shapes only; all target measurements are charged."}


def analyze(directory,output,library=None,simple=None,skip_replay=False):
    output=Path(output);output.mkdir(parents=True,exist_ok=True);figures=output/"figures";figures.mkdir(exist_ok=True)
    records=current_records(directory);configs=tuple(KernelConfig(c) for c in design.configs())
    shapes=tuple(ShapeSpec(dict(zip(("M","N","K"),dims))) for _,_,dims,_ in design.SHAPES)
    labels=[name for name,_,_,_ in design.SHAPES]
    matrix=PerformanceMatrix.from_records(WorkloadDistribution(shapes,tuple([1]*len(shapes)),name="T1 finite Shape distribution"),configs,records)
    counts=Counter((r.shape.id,r.config.id,r.seed) for r in records if r.correctness_verified)
    missing=[{"shape":dict(shape.dims),"config":dict(config.params),"seed":seed} for shape in shapes for config in configs for seed in (7,19,31)
             if not counts[shape.id,config.id,seed]]
    coverage={"records":len(records),"expected_research_records":len(shapes)*len(configs)*3,"missing_successes":missing,
        "record_statuses":dict(Counter(r.status.value for r in records)),"matrix_cell_coverage":matrix.coverage,
        "hardware":records[0].hardware.name,"kernel":records[0].operator.implementation,"dtype":records[0].operator.dtype,
        "kernel_fingerprint":records[0].environment.get("kernel_fingerprint"),"protocol":records[0].metadata.get("protocol"),
        "environment":{k:v for k,v in records[0].environment.items() if k not in {"platform"}},
        "sources":[r.id for r in records]}
    write(output/"coverage.json",coverage)
    dataset_id=hashlib.sha256(canonical_json(sorted(r.id for r in records)).encode()).hexdigest()
    dataset=output/"datasets"/dataset_id
    with ExperimentStore(dataset/"measurements.sqlite") as store:
        store.add_many(records)
        if {r.id for r in store.iter_records()}!={r.id for r in records}:
            raise ValueError("dataset version contains unrelated records")
        store.export_jsonl(dataset/"measurements.jsonl")
    write(output/"dataset-manifest.json",{"schema_version":1,"dataset_id":dataset_id,"records":len(records),
        "collection_git_commit":read_json(Path(directory)/"research_manifest.json").get("git_commit"),
        "collection_run_id":read_json(Path(directory)/"research_manifest.json").get("run_id"),
        "sqlite":(dataset/"measurements.sqlite").relative_to(output).as_posix(),
        "jsonl":(dataset/"measurements.jsonl").relative_to(output).as_posix(),
        "selection":"only current v2 observed IDs; validation excluded; original records unmodified"})
    write(output/"matrix.json",matrix.to_dict())
    if missing or not matrix.complete: raise ValueError("incomplete physical repetitions; preserved coverage, raw export and partial matrix; no complete-matrix analysis asserted")
    structure=describe(matrix,records,labels);write(output/"structure.json",structure)
    _heatmap(figures/"relative-loss.svg","T1 finite config matrix","Ratio to each Shape's observed best; physical RTX 5060 Ti; FP16 Graph 512",labels,
        [str(i) for i in range(len(configs))],[[v+1 for v in row] for row in structure["relative_loss"]])
    correlation_plot(figures/"rank-correlation.svg",labels,structure["spearman"])
    sensitivity_series={key:[(i,100*(next(s for s in structure["sensitivity"] if s["shape"]==name and s["parameter"]==key)["paired_geomean_high_over_low"]-1))
        for i,name in enumerate(labels)] for key in design.SPACE}
    _line_chart(figures/"sensitivity.svg","Matched parameter sensitivity","Change high vs low, geometric mean across 32 matched pairs; indices in shape-design.json",
        sensitivity_series,"Shape index","Latency change (%)",render_png=False)
    l3=l3_study(matrix,records);write(output/"l3.json",l3)
    for weight in ("uniform","synthetic_decode_skew"):
        series={name:[(r["k"],r["relative_loss"]*100) for r in l3["results"] if r["weights"]==weight and r["method"]==name and r["status"]=="passed"]
                for name in ("greedy","local_swap","exact")}
        _line_chart(figures/("l3-"+weight+".svg"),"Config-budget K vs loss",weight+"; 10 Shapes / 64 configs, finite observed reference",series,
            "Maximum distinct config dictionaries","Weighted latency loss (%)",render_png=False)
    joint=joint_study(matrix,records);write(output/"joint.json",joint)
    joint_series={name:[(budget,100*st.mean(r["mean_shape_regret"] for r in joint["results"] if r["method"]==name and r["budget"]==budget))
        for budget in (32,64,128)] for name in ("independent_l2_then_l3","random_common_panel","training_prior_common_panel")}
    _line_chart(figures/"joint.svg","Offline L2-L3 acquisition experiment","Real measured labels, private target reveal budget; 4 test Shapes, K=2",joint_series,
        "Maximum revealed Shape-config cells","Mean per-Shape regret (%)",render_png=False)
    replay=None
    if not skip_replay:
        replay=replay_study(records,ConfigSpace(design.SPACE,name="T1 controlled 64-config finite space"));write(output/"replay.json",replay)
        for split in ("validation","test"):
            series={name:[(r["budget"],r["mean_regret"]*100) for r in replay["aggregates"] if r["split"]==split and r["algorithm"]==name]
                    for name in ("random","grid","heuristic","surrogate","transfer")}
            _line_chart(figures/("l2-"+split+".svg"),"Offline Replay: finite-set regret",split+"; 3 physical replicates x 10 search seeds; no fresh GPU measurements",series,
                "Revealed observations","Mean relative regret (%)",render_png=False)
    baselines=[]
    for tag,source in (("torch/mm",library),("simple",simple)):
        if not source: continue
        baseline_records=current_records(source)
        for record in baseline_records: require_baseline_compatible(records[0],record)
        for i,shape in enumerate(shapes):
            cells=defaultdict(list)
            for r in baseline_records:
                if r.shape.id==shape.id and r.correctness_verified: cells[r.config.id].extend(r.runtime_samples_ms)
            if cells:
                value=min(st.median(v) for v in cells.values())
                baselines.append({"shape":labels[i],"baseline":tag,"latency_ms":value,"t1_best_ms":min(matrix.latencies_ms[i]),
                    "t1_over_baseline_latency":min(matrix.latencies_ms[i])/value,"baseline_configs":len(cells),
                    "notice":"same contiguous dtype/semantic, preallocated output, Graph protocol; independently collected inputs share generator seeds. Library internal dispatch is opaque."})
    write(output/"baselines.json",baselines)
    summary={"status":"passed","records":len(records),"shapes":len(shapes),"configs":len(configs),"repetitions_per_cell":3,
        "matrix_complete":matrix.complete,"replay_trials":len(replay["results"]) if replay else 0,"joint_trials":len(joint["results"]),
        "compiled_artifacts":l3["all_compiled_artifacts"],"source_record_ids_sha256":hashlib.sha256(canonical_json(sorted(r.id for r in records)).encode()).hexdigest()}
    write(output/"summary.json",summary)
    lines=["# T1 real-matrix analysis","",canonical_json(summary),"",
        "All references below are the best in this finite measured config set. 3 independent input seeds; 31 samples/run. No global optimum claim.","",
        _table(("Shape","Best us","Best config","Independent medians us"),((r["name"],r["latency_ms"]*1000,r["config"],[v*1000 for v in r["best_config_repeat_medians_ms"]]) for r in structure["best_by_shape"])),"",
        _table(("Shape","Baseline","Baseline us","T1 us","T1/baseline"),((r["shape"],r["baseline"],r["latency_ms"]*1000,r["t1_best_ms"]*1000,r["t1_over_baseline_latency"]) for r in baselines)),"",
        "Exact config budgets do not constrain compiled specialization counts or deployment size. See l3.json for both retained-all-pairs and dispatch-only artifact counts.",""]
    for path in sorted(figures.glob("*.svg")): lines += [f"![Analysis](figures/{path.name})",""]
    (output/"report.md").write_text("\n".join(lines))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matrix",required=True);parser.add_argument("--output",required=True)
    parser.add_argument("--library");parser.add_argument("--simple");parser.add_argument("--skip-replay",action="store_true")
    args=parser.parse_args();analyze(args.matrix,args.output,args.library,args.simple,args.skip_replay)
