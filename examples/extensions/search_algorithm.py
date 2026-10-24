"""Independent search algorithm using only the public contributor SDK.

Load this file as a plugin and select ``demo/reverse@1.0.0``.  The algorithm
walks Cartesian indices backwards with bounded validation work per suggestion;
it never constructs a candidate list proportional to the search space.
"""

from kernel_tuning_lab.sdk import SearchAlgorithmDefinition, SearchCapabilities


class ReverseSearch:
    name = "demo/reverse"

    def __init__(self, seed=0, probe_limit=256):
        if isinstance(probe_limit, bool) or not isinstance(probe_limit, int) or probe_limit < 1:
            raise ValueError("probe_limit must be a positive integer")
        self.seed = seed
        self.probe_limit = probe_limit
        self._cursors = {}

    def suggest(self, hardware, operator, shape, space, history, budget, batch_size=1):
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
            raise ValueError("batch_size must be a positive integer")
        if budget.exhausted(0, 0):
            return []
        count = min(batch_size, budget.max_evaluations)
        seen = {record.config.id for record in history}
        seen.update(getattr(history, "attempted_config_ids", ()))
        key = (hardware.id, operator.id, shape.id, space.fingerprint)
        cursor = self._cursors.get(key, space.cardinality - 1)
        result = []
        checked = 0
        while cursor >= 0 and len(result) < count and checked < self.probe_limit:
            candidate = space.config_at(cursor)
            cursor -= 1
            checked += 1
            if candidate.id not in seen and space.validate(candidate, hardware, operator, shape).valid:
                result.append(candidate)
                seen.add(candidate.id)
        self._cursors[key] = cursor
        return result


def create_reverse(seed=0, probe_limit=256):
    return ReverseSearch(seed=seed, probe_limit=probe_limit)


ASSETS = (
    SearchAlgorithmDefinition(
        id="demo/reverse", version="1.0.0", factory=create_reverse,
        capabilities=SearchCapabilities(),
        metadata={"description": "Bounded reverse traversal of any finite mixed parameter space"}),
)
