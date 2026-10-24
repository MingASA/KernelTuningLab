"""Discover bundled asset providers through the same registration contract."""

from importlib import import_module
from pkgutil import iter_modules


def register_builtin_extensions(catalog) -> None:
    for provider in sorted(iter_modules(__path__), key=lambda item: item.name):
        if provider.name.startswith("_"):
            continue
        module = import_module(f"{__name__}.{provider.name}")
        hook = getattr(module, "register_assets", None)
        if not callable(hook):
            raise TypeError(f"bundled asset provider {provider.name!r} must expose register_assets(catalog)")
        hook(catalog)


__all__ = ["register_builtin_extensions"]
