import random
import math
from kernel_tuning_lab.sdk import KernelConfig, SearchAlgorithmDefinition, SpaceBuildBudget

class ProductionSearcher:
    def __init__(self, seed=0):
        self.rng = random.Random(seed)
        self.state = {}

    def suggest(self, hardware, operator, shape, space, history, budget, batch_size=1):
        key = (str(operator), str(shape))
        if key not in self.state:
            self.state[key] = {'seen': set(), 'best': None, 'best_ms': float('inf')}
        st = self.state[key]
        
        for rec in history:
            if rec.config and rec.config.id not in st['seen']:
                st['seen'].add(rec.config.id)
                if rec.correctness_verified and rec.median_ms is not None:
                    if rec.median_ms < st['best_ms']:
                        st['best_ms'] = rec.median_ms
                        st['best'] = rec.config

        if budget.exhausted(0, 0):
            return []

        candidates = []
        if st['best']:
            try:
                neighbors = space.neighbors(st['best'], max_changed=2, max_neighbors=32)
                for n in neighbors:
                    if n.id not in st['seen'] and space.validate(n, hardware, operator, shape).valid:
                        candidates.append(n)
            except Exception:
                pass

        if len(candidates) < batch_size:
            try:
                result = space.complete({}, hardware, operator, shape, count=max(1, batch_size * 2), rng=self.rng, exclude_ids=st['seen'], budget=SpaceBuildBudget(max_nodes=20000, max_wall_time_s=1))
                for c in result.configs:
                    if c.id not in st['seen'] and space.validate(c, hardware, operator, shape).valid:
                        candidates.append(c)
            except Exception:
                pass

        if len(candidates) < batch_size:
            try:
                result = space.complete({}, hardware, operator, shape, count=batch_size, rng=self.rng, exclude_ids=st['seen'], budget=SpaceBuildBudget(max_nodes=20000, max_wall_time_s=1))
                for c in result.configs:
                    if c.id not in st['seen'] and space.validate(c, hardware, operator, shape).valid:
                        candidates.append(c)
            except Exception:
                pass

        if not candidates:
            return []

        self.rng.shuffle(candidates)
        return candidates[:batch_size]

ASSETS = (SearchAlgorithmDefinition("generated/production_policy", "1.0.0", ProductionSearcher),)
algorithm_ref = "generated/production_policy@1.0.0"