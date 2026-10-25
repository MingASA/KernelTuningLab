"""A standalone, bounded search plugin using only the public SDK.

Seeded coprime traversal works on finite mixed/conditional spaces. No operator,
hardware, performance table, compiler or platform-core customization is needed.
"""

import hashlib
import math
import random

from kernel_tuning_lab.sdk import SearchAlgorithmDefinition, SearchCapabilities


class StrideSearch:
    def __init__(self, seed=0, probe_limit=512):
        if type(probe_limit) is not int or probe_limit < 1:
            raise ValueError("probe_limit must be a positive integer")
        self.seed, self.probe_limit = seed, probe_limit
        self.positions = {}

    def suggest(self, hardware, operator, shape, space, history, budget, batch_size=1):
        if budget.exhausted(0, 0) or space.cardinality == 0:
            return []
        key = (hardware.id, operator.id, shape.id, space.fingerprint)
        if key not in self.positions:
            rng = random.Random(int.from_bytes(hashlib.sha256(repr((self.seed, key)).encode()).digest(), "big"))
            n = space.cardinality
            step = rng.randrange(1, n + 1)
            # Avoid unbounded work even for a very large or adversarial domain.
            for _ in range(64):
                if math.gcd(step, n) == 1:
                    break
                step = step % n + 1
            else:
                step = 1
            self.positions[key] = [rng.randrange(n), step, 0]
        offset, step, visited = self.positions[key]
        seen = {r.config.id for r in history} | set(history.attempted_config_ids)
        result = []
        for _ in range(min(self.probe_limit, space.cardinality - visited)):
            candidate = space.config_at((offset + step * visited) % space.cardinality)
            visited += 1
            if candidate.id not in seen and space.validate(candidate, hardware, operator, shape).valid:
                result.append(candidate)
                seen.add(candidate.id)
                if len(result) >= min(batch_size, budget.max_evaluations):
                    break
        self.positions[key][2] = visited
        return result


def create_search(seed=0, probe_limit=512):
    return StrideSearch(seed=seed, probe_limit=probe_limit)


ASSETS = (SearchAlgorithmDefinition("example/stride", "1.0.0", create_search, SearchCapabilities()),)
