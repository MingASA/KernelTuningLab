"""Build one ordinary research spec using an existing model and frozen rule.

Run on the experiment host after fingerprinting its existing local checkpoint.
This only prepares JSON and previews the existing Runner's bounded workload.
"""
import argparse
import hashlib
import json
from pathlib import Path

from examples.h800_llm_study.design import research_spec
from kernel_tuning_lab.research import expand_research


def prepare(model_manifest,rule,*,budget=8,seed=7):
    model_manifest=Path(model_manifest).resolve();rule=Path(rule).resolve()
    model=json.loads(model_manifest.read_text())
    if not model.get('full_weight_hashes'):raise ValueError('fingerprint all existing model weights first')
    program=json.loads(rule.read_text())
    if program.get('generation_evidence',{}).get('evidence_level')!='real_local_model_inference':
        raise ValueError('the rule must have recorded real model generation evidence')
    algorithms=['random@1.0.0','heuristic@1.0.0',
        {'asset':'smac_rf@1.0.0','options':{'initial_design_size':4}},
        {'asset':'kernel_tuner_de@1.0.0','options':{'population_size':8}},
        {'asset':'llm/live_local_decision@1.0.0','options':{'model_manifest':str(model_manifest),
            'manifest_sha256':hashlib.sha256(model_manifest.read_bytes()).hexdigest(),
            'decoding':'sampled','structured_output':True,'max_new_tokens':192,'max_retries':2}},
        {'asset':'llm/generated_rule@1.0.0','options':{'artifact':str(rule),
            'artifact_sha256':hashlib.sha256(rule.read_bytes()).hexdigest()}}]
    spec=research_spec(algorithms,splits=('test',),budget=budget,seeds=[seed],name='local-model-research-example')
    # One low-M Shape over all three implementations; expand deliberately after
    # inspecting costs. Different methods keep the same authoritative spaces.
    spec['cases']=[case for case in spec['cases'] if case['shapes'][0]['M']==1]
    return spec


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-manifest',required=True);parser.add_argument('--rule',required=True)
    parser.add_argument('--output',required=True);parser.add_argument('--budget',type=int,default=8);parser.add_argument('--seed',type=int,default=7)
    args=parser.parse_args();spec=prepare(args.model_manifest,args.rule,budget=args.budget,seed=args.seed)
    plan=expand_research(spec)
    Path(args.output).write_text(json.dumps(spec,indent=2)+'\n')
    print(json.dumps({'planned':plan['planned'],'note':'B8 truncates DE at initialization; use the separately reported B32 trajectories to assess actual evolution. Full allowance is a bound, not a runtime prediction.'},indent=2))


if __name__=='__main__':main()
