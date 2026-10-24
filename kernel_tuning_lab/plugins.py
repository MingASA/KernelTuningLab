"""Load explicitly requested trusted local Python extensions."""

from __future__ import annotations

import hashlib
import importlib
import importlib.util
from pathlib import Path
import sys
from types import ModuleType

from .assets import KernelDefinition, OperatorDefinition, SearchAlgorithmDefinition


def _register_loaded(value, catalog):
    """Accept declarative modules, registration hooks, or entry-point assets."""
    if isinstance(value, ModuleType):
        hook = getattr(value, "register_assets", None)
        if hook is not None:
            if not callable(hook):
                raise TypeError("plugin register_assets must be callable")
            hook(catalog)
        elif hasattr(value, "ASSETS"):
            _register_loaded(value.ASSETS, catalog)
        else:
            hook = getattr(value, "register_plugin", None)
            if hook is not None:
                if not callable(hook):
                    raise TypeError("plugin register_plugin must be callable")
                hook()
        return
    if isinstance(value, (OperatorDefinition, KernelDefinition, SearchAlgorithmDefinition)):
        catalog.register(value)
    elif callable(value):
        value(catalog)
    elif isinstance(value, dict):
        catalog.register_many(value.values())
    else:
        try:
            catalog.register_many(value)
        except TypeError as exc:
            raise TypeError("asset plugin must expose ASSETS or register_assets(catalog)") from exc


def load_plugin(module_or_path: str, catalog=None) -> ModuleType:
    if catalog is None:
        from .catalog import get_catalog
        catalog = get_catalog()
    path = Path(module_or_path)
    if path.suffix == ".py" or path.is_file():
        if not path.is_file():
            raise FileNotFoundError(f"plugin does not exist: {path}")
        name = "ktlab_plugin_" + hashlib.sha256(str(path.resolve()).encode()).hexdigest()[:16]
        specification = importlib.util.spec_from_file_location(name, path.resolve())
        if specification is None or specification.loader is None:
            raise ValueError(f"cannot load plugin: {path}")
        module = importlib.util.module_from_spec(specification)
        sys.modules[name] = module
        # Reading source directly avoids stale timestamp-based .pyc reuse when
        # an agent edits a same-size source file twice within one second.
        exec(compile(path.read_bytes(), str(path.resolve()), "exec"), module.__dict__)
    else:
        module = importlib.import_module(module_or_path)
    _register_loaded(module, catalog)
    return module


def discover_entry_points(catalog=None) -> tuple[str, ...]:
    """Load installed, explicitly declared ``kernel_tuning_lab.assets`` entries."""
    from importlib.metadata import entry_points
    if catalog is None:
        from .catalog import get_catalog
        catalog = get_catalog()
    available = entry_points()
    entries = available.select(group="kernel_tuning_lab.assets") if hasattr(available, "select") else available.get("kernel_tuning_lab.assets", ())
    loaded = []
    for entry in sorted(entries, key=lambda item: item.name):
        _register_loaded(entry.load(), catalog)
        loaded.append(entry.name)
    return tuple(loaded)
