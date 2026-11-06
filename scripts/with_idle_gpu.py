"""Compatibility entrypoint for the packaged, unchanged GPU admission guard."""
from pathlib import Path

_source = Path(__file__).resolve().parents[1] / "kernel_tuning_lab/gpu_admission.py"
__file__ = str(_source)
# Keep standalone callers and import-based fixtures on the canonical implementation.
exec(compile(_source.read_bytes(), str(_source), "exec"), globals())
