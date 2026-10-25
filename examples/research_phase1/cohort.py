"""SDK-only finite cohort traversal. No latency table or oracle is read here."""
import random
from kernel_tuning_lab.sdk import KernelConfig, SearchAlgorithmDefinition


class ShuffledCohort:
    def __init__(self, seed=0, configs=None):
        self.seed, self.configs = seed, configs

    def suggest(self, hardware, operator, shape, space, history, budget, batch_size=1):
        if budget.exhausted(0, 0):
            return []
        candidates = ([KernelConfig(dict(c)) for c in self.configs] if self.configs is not None else
                      [space.config_at(i) for i in range(space.cardinality)])
        if len(candidates) > 4096:
            raise ValueError("cohort traversal is for explicitly bounded experiments, at most 4096 configs")
        random.Random(self.seed + int(shape.id[:8], 16)).shuffle(candidates)
        seen = set(history.attempted_config_ids) | {r.config.id for r in history}
        return [c for c in candidates if c.id not in seen and space.validate(c, hardware, operator, shape).valid][:min(batch_size, budget.max_evaluations)]


def create(seed=0, configs=None):
    return ShuffledCohort(seed, configs)


ASSETS = (SearchAlgorithmDefinition("phase1/cohort", "1.0.0", create),)
