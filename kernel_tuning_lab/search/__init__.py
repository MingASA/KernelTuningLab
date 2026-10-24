"""Reproducible search algorithms with an observation-only suggestion contract."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
import math
import random
from typing import Protocol

from ..datasets import case_id, require_training_records
from ..models import HardwareSpec, KernelConfig, MeasurementRecord, OperatorSpec, SearchBudget, ShapeSpec, canonical_json
from ..space import ConfigSpace


@dataclass(frozen=True)
class SearchHistory(Sequence[MeasurementRecord]):
    """Only revealed records, plus identities of attempted missing configurations."""

    records: tuple[MeasurementRecord, ...] = ()
    attempted_config_ids: frozenset[str] = frozenset()

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]

    def __iter__(self) -> Iterator[MeasurementRecord]:
        return iter(self.records)


class Searcher(Protocol):
    name: str

    def suggest(self, hardware: HardwareSpec, operator: OperatorSpec, shape: ShapeSpec,
                space: ConfigSpace, history: Sequence[MeasurementRecord], budget: SearchBudget,
                batch_size: int = 1) -> list[KernelConfig]: ...


def _seen(history: Sequence[MeasurementRecord]) -> set[str]:
    return {r.config.id for r in history} | set(getattr(history, "attempted_config_ids", ()))


def _best(history: Sequence[MeasurementRecord]) -> MeasurementRecord | None:
    successes = [r for r in history if r.successful]
    return min(successes, key=lambda r: (r.median_ms, r.config.id), default=None)


def _cardinality(space) -> int:
    value = getattr(space, "cardinality", None)
    return value if value is not None else len(space)


def _config_at(space, index):
    method = getattr(space, "config_at", None)
    if method is not None:
        return method(index)
    # Compatibility with user-owned finite spaces implementing only domains.
    values = {}
    for key, domain in reversed(tuple(space.domains.items())):
        index, position = divmod(index, len(domain))
        values[key] = domain[position]
    return KernelConfig({key: values[key] for key in space.domains})


def _sample_configs(space, rng, count, exclude_ids):
    method = getattr(space, "sample_configs", None)
    if method is not None:
        return method(rng, count, exclude_ids=exclude_ids)
    result, seen = [], set(exclude_ids)
    for _ in range(max(64, count * 16)):
        candidate = _config_at(space, rng.randrange(_cardinality(space)))
        if candidate.id not in seen:
            result.append(candidate)
            seen.add(candidate.id)
            if len(result) >= count:
                break
    return result


def _pool_key(hardware, operator, shape, space):
    return id(space), hardware.id, operator.id, shape.id


def _candidates(searcher, hardware, operator, shape, space, history, budget, batch_size):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if budget.exhausted(0, 0):
        return [], 0
    key = _pool_key(hardware, operator, shape, space)
    seen = _seen(history)
    count = min(batch_size, budget.max_evaluations)
    if _cardinality(space) <= searcher.exhaustive_threshold:
        if key not in searcher._candidate_cache:
            searcher._candidate_cache[key] = tuple((c.id, c) for c in space.legal_configs(hardware, operator, shape))
        return [c for identity, c in searcher._candidate_cache[key] if identity not in seen], count
    # Retain a bounded active pool. Refill after consumption, not by materializing C.
    active = {identity: c for identity, c in searcher._candidate_cache.get(key, ()) if identity not in seen}
    target = max(searcher.candidate_pool_size, count)
    if len(active) < max(count, target // 4):
        excluded = seen | active.keys()
        for _ in range(8):
            requested = target - len(active)
            if requested <= 0:
                break
            sampled = _sample_configs(space, searcher.rng, requested, excluded)
            if not sampled:
                break
            for config in sampled:
                identity = config.id
                excluded.add(identity)
                if space.validate(config, hardware, operator, shape).valid:
                    active[identity] = config
    searcher._candidate_cache[key] = tuple(active.items())
    return list(active.values()), count


def heuristic_score(config: KernelConfig, hardware: HardwareSpec, shape: ShapeSpec) -> float:
    """Analytic ordering, using shape/resources rather than measured target labels."""
    p = config.params
    if {"BLOCK_M", "BLOCK_N", "BLOCK_K"} <= p.keys():
        bm, bn, bk = (float(p[key]) for key in ("BLOCK_M", "BLOCK_N", "BLOCK_K"))
        m, n, k = (shape.get(key) for key in ("M", "N", "K"))
        padding = (math.ceil(m / bm) * bm * math.ceil(n / bn) * bn * math.ceil(k / bk) * bk) / (m * n * k)
        tiles = math.ceil(m / bm) * math.ceil(n / bn) * shape.get("Batch", shape.get("batch", 1))
        occupancy = min(1.0, tiles / max(1, hardware.sm_count * 2))
        tile_work = math.sqrt(bm * bn) / 64
        stages = float(config.get("num_stages", 2))
        warps = float(config.get("num_warps", 4))
        resource = 1 + 0.06 * (stages - 2) + 0.04 * abs(warps - 4)
        return padding * resource / max(0.05, occupancy * tile_work)
    if "BLOCK_SIZE" in p:
        block = float(config.get("BLOCK_SIZE"))
        padding = block / shape.get("N")
        preferred_warps = 4 if block <= 2048 else 8
        warp_penalty = 1 + 0.08 * abs(float(config.get("num_warps", 4)) - preferred_warps) / 4
        return padding * warp_penalty
    # This also works for categorical/non-GEMM spaces, without assuming new knobs.
    return float(sum(len(canonical_json(v)) for v in p.values()))


class RandomSearch:
    name = "random"

    def __init__(self, seed: int = 0, candidate_pool_size: int = 512, exhaustive_threshold: int = 4096):
        if not isinstance(candidate_pool_size, int) or isinstance(candidate_pool_size, bool) or candidate_pool_size < 1:
            raise ValueError("candidate_pool_size must be a positive integer")
        if not isinstance(exhaustive_threshold, int) or isinstance(exhaustive_threshold, bool) or exhaustive_threshold < 0:
            raise ValueError("exhaustive_threshold must be a nonnegative integer")
        self.seed = seed
        self.rng = random.Random(seed)
        self.candidate_pool_size = candidate_pool_size
        self.exhaustive_threshold = exhaustive_threshold
        self._candidate_cache = {}
        self._grid_cursors = {}

    def suggest(self, hardware, operator, shape, space, history, budget, batch_size=1):
        candidates, count = _candidates(self, hardware, operator, shape, space, history, budget, batch_size)
        self.rng.shuffle(candidates)
        return candidates[:count]


class GridSearch(RandomSearch):
    name = "grid"

    def suggest(self, hardware, operator, shape, space, history, budget, batch_size=1):
        if _cardinality(space) > self.exhaustive_threshold:
            if batch_size < 1:
                raise ValueError("batch_size must be positive")
            if budget.exhausted(0, 0):
                return []
            count = min(batch_size, budget.max_evaluations)
            key = _pool_key(hardware, operator, shape, space)
            cursor = self._grid_cursors.get(key, 0)
            seen = _seen(history)
            result = []
            # Bound validation work when almost every point is illegal.
            stop = min(_cardinality(space), cursor + max(count, self.candidate_pool_size) * 8)
            while cursor < stop and len(result) < count:
                candidate = _config_at(space, cursor)
                cursor += 1
                if candidate.id not in seen and space.validate(candidate, hardware, operator, shape).valid:
                    result.append(candidate)
                    seen.add(candidate.id)
            self._grid_cursors[key] = cursor
            return result
        candidates, count = _candidates(self, hardware, operator, shape, space, history, budget, batch_size)
        return candidates[:count]


class HeuristicSearch(RandomSearch):
    name = "heuristic"

    def suggest(self, hardware, operator, shape, space, history, budget, batch_size=1):
        candidates, count = _candidates(self, hardware, operator, shape, space, history, budget, batch_size)
        return sorted(candidates, key=lambda c: (heuristic_score(c, hardware, shape), c.id))[:count]


class LocalSearch(RandomSearch):
    name = "local"

    def suggest(self, hardware, operator, shape, space, history, budget, batch_size=1):
        candidates, count = _candidates(self, hardware, operator, shape, space, history, budget, batch_size)
        incumbent = _best(history)
        if incumbent is None:
            self.rng.shuffle(candidates)
            return candidates[:count]
        if _cardinality(space) > self.exhaustive_threshold:
            seen = _seen(history)
            nearby = []
            dimensions = [(key, values) for key, values in space.domains.items() if len(values) > 1]
            per_dimension = max(1, math.ceil(self.candidate_pool_size / max(1, len(dimensions))))
            for key, values in dimensions:
                for index in self.rng.sample(range(len(values)), min(len(values), per_dimension)):
                    candidate = KernelConfig({**incumbent.config.params, key: values[index]})
                    if candidate.id not in seen and space.validate(candidate, hardware, operator, shape).valid:
                        nearby.append(candidate)
                        seen.add(candidate.id)
            self.rng.shuffle(nearby)
            self.rng.shuffle(candidates)
            return (nearby + [c for c in candidates if c.id not in seen])[:count]
        neighborhood = {c.id for c in space.neighbors(incumbent.config)}
        nearby = [c for c in candidates if c.id in neighborhood]
        distant = [c for c in candidates if c.id not in neighborhood]
        self.rng.shuffle(nearby)
        self.rng.shuffle(distant)
        return (nearby + distant)[:count]


def _features(config: KernelConfig, space: ConfigSpace, specifications=None) -> tuple:
    values = []
    if specifications is not None:
        for key, numeric, minimum, width, categorical in specifications:
            item = config.get(key)
            if numeric == "linear":
                values.append((float(item) - minimum) / width if width else 0.0)
            elif numeric:
                values.append((math.log(float(item)) - minimum) / width if width else 0.0)
            elif categorical is None:
                values.append(canonical_json(item))
            else:
                values.extend(float(canonical_json(item) == value) for value in categorical)
        return tuple(values)
    for key, domain in space.domains.items():
        item = config.get(key)
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 for v in domain):
            logs = [math.log(float(v)) for v in domain]
            width = max(logs) - min(logs)
            values.append((math.log(float(item)) - min(logs)) / width if width else 0.0)
        else:
            values.extend(float(canonical_json(item) == canonical_json(value)) for value in domain)
    return tuple(values)


def _distance(left: tuple, right: tuple) -> float:
    return math.sqrt(sum((a - b) ** 2 if isinstance(a, (float, int)) and isinstance(b, (float, int))
                         else float(a != b) for a, b in zip(left, right)))


class SurrogateSearch(RandomSearch):
    """Distance-weighted log-latency surrogate with an exploration bonus."""

    name = "surrogate"

    def __init__(self, seed: int = 0, warmup: int = 4, exploration: float = 0.35,
                 candidate_pool_size: int = 512, exhaustive_threshold: int = 4096):
        super().__init__(seed, candidate_pool_size=candidate_pool_size, exhaustive_threshold=exhaustive_threshold)
        if warmup < 1 or exploration < 0:
            raise ValueError("warmup must be positive and exploration nonnegative")
        self.warmup = warmup
        self.exploration = exploration
        self._feature_specs = {}
        self._feature_cache = {}

    def _encode(self, config, space):
        identity = id(space), config.id
        if identity in self._feature_cache:
            return self._feature_cache[identity]
        if id(space) not in self._feature_specs:
            specs = []
            for key, domain in space.domains.items():
                numeric = all(isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0 for v in domain)
                if numeric:
                    minimum = math.log(float(min(domain)))
                    width = math.log(float(max(domain))) - minimum
                    specs.append((key, True, minimum, width, None))
                elif (_cardinality(space) > self.exhaustive_threshold
                      and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in domain)):
                    minimum, maximum = float(min(domain)), float(max(domain))
                    specs.append((key, "linear", minimum, maximum - minimum, None))
                else:
                    categories = tuple(canonical_json(v) for v in domain) if len(domain) <= 32 else None
                    specs.append((key, False, 0.0, 0.0, categories))
            self._feature_specs[id(space)] = specs
        encoded = _features(config, space, self._feature_specs[id(space)])
        self._feature_cache[identity] = encoded
        return encoded

    def suggest(self, hardware, operator, shape, space, history, budget, batch_size=1):
        candidates, count = _candidates(self, hardware, operator, shape, space, history, budget, batch_size)
        known = [r for r in history if r.successful]
        if len(known) < self.warmup:
            self.rng.shuffle(candidates)
            return candidates[:count]
        observations = [(self._encode(r.config, space), math.log(r.median_ms)) for r in known]
        def acquisition(config):
            point = self._encode(config, space)
            distances = sorted((_distance(point, x), y) for x, y in observations)[:5]
            weights = [1 / (distance + 0.05) ** 2 for distance, _ in distances]
            mean = sum(weight * row[1] for weight, row in zip(weights, distances)) / sum(weights)
            uncertainty = min(1.0, distances[0][0])
            return mean - self.exploration * uncertainty, config.id
        return sorted(candidates, key=acquisition)[:count]


class TransferSearch(RandomSearch):
    """Rank source-case winners; all source observations must be training data."""

    name = "transfer"

    def __init__(self, seed: int = 0, training_records=(), candidate_pool_size: int = 512,
                 exhaustive_threshold: int = 4096):
        super().__init__(seed, candidate_pool_size=candidate_pool_size, exhaustive_threshold=exhaustive_threshold)
        self.training_records = require_training_records(training_records)
        self._training_case_ids = {r.case_id for r in self.training_records}
        grouped: dict[str, list[MeasurementRecord]] = {}
        for record in self.training_records:
            if record.successful:
                grouped.setdefault(record.case_id, []).append(record)
        self._source_cases = []
        for rows in grouped.values():
            source = rows[0]
            baseline = min(r.median_ms for r in rows)
            observations = tuple((r.config.id, r.median_ms / baseline) for r in rows)
            self._source_cases.append((source.hardware, source.operator, source.shape, observations))
        self._rank_cache = {}
        self._source_configs = {record.config.id: record.config for record in self.training_records if record.successful}

    def suggest(self, hardware, operator, shape, space, history, budget, batch_size=1):
        target = case_id(hardware, operator, shape)
        if target in self._training_case_ids:
            raise ValueError("training data overlaps the target evaluation case")
        candidates, count = _candidates(self, hardware, operator, shape, space, history, budget, batch_size)
        if target not in self._rank_cache:
            scores: dict[str, list[tuple[float, float]]] = {}
            for source_hw, source_op, source_shape, observations in self._source_cases:
                if (source_op.kind, source_op.dtype, source_op.implementation) != (operator.kind, operator.dtype, operator.implementation):
                    continue
                shape_distance = sum(abs(math.log(source_shape.get(key) / shape.get(key))) for key in shape.dims)
                sm_distance = abs(math.log(max(1, source_hw.sm_count) / max(1, hardware.sm_count)))
                similarity = 1 / (1 + shape_distance + 0.25 * sm_distance)
                for identity, normalized_latency in observations:
                    scores.setdefault(identity, []).append((similarity, normalized_latency))
            self._rank_cache[target] = {identity: sum(w * value for w, value in values) / sum(w for w, _ in values)
                                        for identity, values in scores.items()}
        scores = self._rank_cache[target]
        if _cardinality(space) > self.exhaustive_threshold and scores:
            seen = _seen(history)
            source_candidates = []
            for identity in sorted(scores, key=lambda identity: (scores[identity], identity)):
                if identity in seen:
                    continue
                config = self._source_configs[identity]
                if space.validate(config, hardware, operator, shape).valid:
                    source_candidates.append(config)
                    seen.add(identity)
                    if len(source_candidates) >= self.candidate_pool_size:
                        break
            candidates = (source_candidates + [config for config in candidates if config.id not in seen])[:self.candidate_pool_size]
        def rank(config):
            value = scores.get(config.id)
            if value is not None:
                return (0, value, config.id)
            return (1, heuristic_score(config, hardware, shape), config.id)
        return sorted(candidates, key=rank)[:count]


SEARCHER_NAMES = ("random", "grid", "heuristic", "local", "surrogate", "transfer")


def make_searcher(name: str, seed: int = 0, training_records=(), candidate_pool_size: int = 512,
                  exhaustive_threshold: int = 4096) -> Searcher:
    factories = {"random": RandomSearch, "grid": GridSearch, "heuristic": HeuristicSearch,
                 "local": LocalSearch, "surrogate": SurrogateSearch}
    if name == "transfer":
        return TransferSearch(seed=seed, training_records=training_records, candidate_pool_size=candidate_pool_size,
                              exhaustive_threshold=exhaustive_threshold)
    if name not in factories:
        raise ValueError(f"unknown searcher {name!r}; choose from {SEARCHER_NAMES}")
    return factories[name](seed=seed, candidate_pool_size=candidate_pool_size,
                           exhaustive_threshold=exhaustive_threshold)


__all__ = ["Searcher", "SearchHistory", "RandomSearch", "GridSearch", "HeuristicSearch", "LocalSearch",
           "SurrogateSearch", "TransferSearch", "make_searcher", "SEARCHER_NAMES", "heuristic_score"]
