"""Retry only zero-observation CPU-construction failures; keep frozen GPU code.

The only changed search allowance is inner static construction15s->120s,
within the unchanged240s trial bound. Existing measurements are copied as
exact-context cache, and acceptance/native cache misses refuse preparation.
"""
import argparse,hashlib,importlib.util,json,pathlib,sqlite3,subprocess

def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    plan=json.loads(pathlib.Path(a.plan).read_text());out=pathlib.Path(a.output);original=pathlib.Path(plan['original_worker'])
    assert hashlib.sha256(pathlib.Path(__file__).read_bytes()).hexdigest()==plan['recovery_helper_sha256']
    assert subprocess.check_output(['git','-C',plan['code'],'rev-parse','HEAD'],text=True).strip()==plan['source_commit']
    old=json.loads((original/'state.json').read_text());selected=('B_full_static_grammar','kernel_tuner_de')
    for method in selected:
        trial=next(t for t in old['trials'] if t['method']==method)
        if trial['observations']!=0 or not any('budget' in f.get('error','') or 'allowance' in f.get('error','') for f in trial['failures']):
            raise ValueError('only original zero-observation construction failures may be retried')
    if (out/'state.json').exists():raise ValueError('immutable retry already exists; recover receipt')
    out.mkdir(parents=True,exist_ok=True)
    import kernel_tuning_lab.sdk as sdk
    import kernel_tuning_lab.runner as runners
    import kernel_tuning_lab.live_llm as live
    import kernel_tuning_lab.optimizers as optimizers
    real_store=sdk.ExperimentStore;real_budget=sdk.SpaceBuildBudget;real_runner=runners.BenchmarkRunner
    def store_with_original_cache(path,*args,**kwargs):
        source=original/pathlib.Path(path).parent.name/'measurements.sqlite'
        if pathlib.Path(path).exists():raise ValueError('fresh retry database required')
        with sqlite3.connect('file:'+str(source)+'?mode=ro',uri=True) as src,sqlite3.connect(path) as dst:src.backup(dst)
        return real_store(path,*args,**kwargs)
    budget_calls=[]
    def expanded_cpu_budget(*args,**kwargs):
        if kwargs.get('max_wall_time_s')==15:kwargs['max_wall_time_s']=120;budget_calls.append(dict(kwargs))
        return real_budget(*args,**kwargs)
    class CacheOnlyIndependentChecks(real_runner):
        def measure(self,*args,**kwargs):
            cache_only=self.experiment_id.endswith('-acceptance') or self.experiment_id=='production-study-native-expert'
            if not cache_only:return super().measure(*args,**kwargs)
            prepare=self.adapter.prepare
            def refuse(*a,**k):raise RuntimeError('refuse repeat of completed independent acceptance/native measurement')
            self.adapter.prepare=refuse
            try:record=super().measure(*args,**kwargs)
            finally:self.adapter.prepare=prepare
            if not self.last_cache_hit:raise RuntimeError('original independent check cache mismatch; no kernel preparation executed')
            return record
    sdk.ExperimentStore=store_with_original_cache;runners.BenchmarkRunner=CacheOnlyIndependentChecks
    live.SpaceBuildBudget=expanded_cpu_budget;optimizers.SpaceBuildBudget=expanded_cpu_budget
    file=pathlib.Path(plan['code'])/'examples/production_ai_study/work.py'
    spec=importlib.util.spec_from_file_location('frozen_study_work',file);work=importlib.util.module_from_spec(spec);spec.loader.exec_module(work)
    protocol=json.loads(pathlib.Path(plan['protocol']).read_text());assert work.digest(plan['protocol'])==plan['protocol_sha256']
    protocol['methods']=list(selected)
    try:work.run_trials(plan,protocol,out,'test')
    finally:work.write(out/'construction-recovery.json',{'inner_static_wall_s':120,'original_inner_wall_s':15,
        'original_worker':str(original),'trial_wall_s':protocol['trial_wall_s'],'budget_calls':budget_calls,
        'policy_prompt_decoder_space_seed_tolerances_unchanged':True,'old_successful_methods_retested':False})

if __name__=='__main__':main()
