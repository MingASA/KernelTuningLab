"""Versioned extension registry with deterministic, content-checked resolution."""

from __future__ import annotations

import functools
from pathlib import Path
from threading import RLock
from typing import Iterable

from .assets import KernelDefinition, OperatorDefinition, SearchAlgorithmDefinition, SearchCapabilities


_KINDS = {OperatorDefinition: "operator", KernelDefinition: "kernel", SearchAlgorithmDefinition: "algorithm"}
_KIND_ALIASES = {"operator": "operator", "operators": "operator", "kernel": "kernel", "kernels": "kernel",
                 "algorithm": "algorithm", "algorithms": "algorithm", "search": "algorithm", "searcher": "algorithm"}


def _kind(value: str) -> str:
    try:
        return _KIND_ALIASES[value]
    except (KeyError, TypeError):
        raise ValueError(f"unknown asset kind {value!r}; choose operator, kernel, or algorithm") from None


class AssetCatalog:
    """An isolated registry; constructing it does not import plugins or runtimes."""

    def __init__(self):
        self._assets: dict[str, dict[str, object]] = {kind: {} for kind in ("operator", "kernel", "algorithm")}
        self._fingerprints: dict[tuple[str, str], str] = {}

    def register(self, asset):
        kind = next((value for definition, value in _KINDS.items() if isinstance(asset, definition)), None)
        if kind is None:
            raise TypeError("register expects an OperatorDefinition, KernelDefinition, or SearchAlgorithmDefinition")
        fingerprint = asset.fingerprint
        key = (kind, asset.ref)
        if key in self._fingerprints and self._fingerprints[key] != fingerprint:
            raise ValueError(f"asset {kind} {asset.ref!r} was already registered with different content; bump its version")
        # Same implementation identity can receive a refreshed correctness claim.
        self._assets[kind][asset.ref] = asset
        self._fingerprints[key] = fingerprint
        return asset

    def register_many(self, assets: Iterable) -> tuple:
        """Register a collection atomically, including collision checks."""
        staged = AssetCatalog()
        staged._assets = {kind: dict(values) for kind, values in self._assets.items()}
        staged._fingerprints = dict(self._fingerprints)
        result = tuple(staged.register(asset) for asset in assets)
        self._assets, self._fingerprints = staged._assets, staged._fingerprints
        return result

    def resolve(self, kind: str, ref: str):
        kind = _kind(kind)
        if not isinstance(ref, str) or not ref:
            raise ValueError("asset reference must be a nonempty string")
        if "@" in ref:
            if ref not in self._assets[kind]:
                raise ValueError(f"unknown {kind} asset {ref!r}")
            return self._assets[kind][ref]
        matches = [asset for asset in self._assets[kind].values() if asset.id == ref]
        if not matches:
            raise ValueError(f"unknown {kind} asset {ref!r}")
        if len(matches) != 1:
            choices = ", ".join(sorted(asset.ref for asset in matches))
            raise ValueError(f"ambiguous {kind} reference {ref!r}; specify one of: {choices}")
        return matches[0]

    def list(self, kind: str | None = None) -> tuple:
        kinds = (_kind(kind),) if kind is not None else ("operator", "kernel", "algorithm")
        return tuple(asset for entry_kind in kinds for _, asset in sorted(self._assets[entry_kind].items()))


def _builtin_search_factory(name: str, seed: int = 0, options=None, **kwargs):
    """New options mapping and legacy factory keyword arguments share one path."""
    from .search import make_searcher
    selected = dict(options or {})
    duplicates = selected.keys() & kwargs.keys()
    if duplicates:
        raise ValueError("algorithm options supplied twice: " + ", ".join(sorted(duplicates)))
    selected.update(kwargs)
    if "seed" in selected:
        raise ValueError("algorithm seed is supplied separately from options")
    training_database = selected.pop("training_database", None)
    if training_database is not None:
        if name != "transfer" or "training_records" in selected:
            raise ValueError("training_database is exclusive to transfer and cannot be combined with training_records")
        from .storage import ExperimentStore
        with ExperimentStore(training_database, read_only=True) as source:
            selected["training_records"] = tuple(record for record in source.iter_records(split="train")
                                                 if record.correctness_verified)
    return make_searcher(name, seed=seed, **selected)


def register_builtin_search_algorithms(catalog: AssetCatalog) -> None:
    from .search import SEARCHER_NAMES
    source_file = str(Path(__file__).with_name("search") / "__init__.py")
    for name in SEARCHER_NAMES:
        catalog.register(SearchAlgorithmDefinition(
            name, "1.0.0", functools.partial(_builtin_search_factory, name), SearchCapabilities(),
            source_files=(source_file,), metadata={"builtin": True, "legacy_name": name}))


_DEFAULT_CATALOG: AssetCatalog | None = None
_CATALOG_LOCK = RLock()


def get_catalog() -> AssetCatalog:
    global _DEFAULT_CATALOG
    with _CATALOG_LOCK:
        if _DEFAULT_CATALOG is None:
            catalog = AssetCatalog()
            register_builtin_search_algorithms(catalog)
            from .operator_library import register_builtin_assets
            register_builtin_assets(catalog)
            _DEFAULT_CATALOG = catalog
        return _DEFAULT_CATALOG


__all__ = ["AssetCatalog", "get_catalog", "register_builtin_search_algorithms"]
