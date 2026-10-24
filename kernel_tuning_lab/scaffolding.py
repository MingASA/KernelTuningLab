"""Generate working, SDK-only contributor starting points and experiment files."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from .asset_experiments import AssetExperiment, AlgorithmSelection, KernelSelection, ValidationBudget
from .models import SearchBudget


_ALGORITHM_SOURCE = '''"""A bounded, history-aware search algorithm using only the public SDK."""
from kernel_tuning_lab.sdk import SearchAlgorithmDefinition, SearchCapabilities

class Search:
    def __init__(self, seed=0):
        self.seed, self.cursors = seed, {}

    def suggest(self, hardware, operator, shape, space, history, budget, batch_size=1):
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if budget.exhausted(0, 0):
            return []
        seen = {row.config.id for row in history} | set(getattr(history, "attempted_config_ids", ()))
        key = (hardware.id, operator.id, shape.id, space.fingerprint)
        cursor = self.cursors.get(key, 0)
        stop = min(space.cardinality, cursor + 512)
        result = []
        while cursor < stop and len(result) < min(batch_size, budget.max_evaluations):
            config = space.config_at(cursor)
            cursor += 1
            if config.id not in seen and space.validate(config, hardware, operator, shape).valid:
                result.append(config)
                seen.add(config.id)
        self.cursors[key] = cursor
        return result

def create_search(seed=0):
    return Search(seed)

ASSETS = (SearchAlgorithmDefinition("local/search", "0.1.0", create_search, SearchCapabilities()),)
'''


def scaffold(kind: str, output_dir: str | Path, *, operator: str = "gemm@1.0.0",
             runtime: str = "python", catalog=None) -> dict:
    from .catalog import get_catalog
    if kind not in {"kernel", "algorithm"} or runtime not in {"python", "cuda"}:
        raise ValueError("scaffold requires kernel/algorithm and python/cuda")
    catalog = catalog or get_catalog()
    op = catalog.resolve("operator", operator)
    prefix = "python" if runtime == "python" else "triton"
    base = catalog.resolve("kernel", f"{prefix}/{op.id}@{op.version}")
    directory = Path(output_dir).resolve()
    files = [directory / name for name in ("assets.py", "experiment.json", "README.md")]
    existing = [path.name for path in files if path.exists()]
    if existing:
        raise ValueError("scaffold refuses to overwrite existing files: " + ", ".join(existing))
    if not op.validation_shapes:
        raise ValueError("operator has no validation shapes; provide an explicit experiment instead")
    if kind == "kernel":
        # The template is already executable. Replace only build to supply a new
        # implementation; the shared operator/reference/space services stay intact.
        source = f'''"""Working Kernel template. Replace build with your implementation."""
from dataclasses import replace
from kernel_tuning_lab.sdk import get_catalog

BASE = get_catalog().resolve("kernel", {base.ref!r})

def build(ctx, **parameters):
    # Inputs: ctx.inputs; named outputs: ctx.outputs; shape/dtype: ctx.shape/operator.
    # Return one callable, or ExecutionPlan with ordered steps and initialization.
    return BASE.build(ctx, **parameters)

ASSETS = (replace(BASE, id={f"local/{op.id}"!r}, version="0.1.0", build=build, claims=(),
                  source_files=(), metadata={{"template_base_fingerprint": BASE.fingerprint}}),)
'''
        kernels = (KernelSelection(f"local/{op.id}@0.1.0"),)
        algorithms = (AlgorithmSelection("random@1.0.0"),)
    else:
        source = _ALGORITHM_SOURCE
        kernels = (KernelSelection(base.ref),)
        algorithms = (AlgorithmSelection("local/search@0.1.0"),)
    dtype = "float32" if runtime == "python" or op.id in {"softmax", "layernorm"} else "float16"
    spec = AssetExperiment(name=f"local-{kind}-experiment", operator=op.ref, kernels=kernels,
                           algorithms=algorithms, shapes=op.validation_shapes[:2], plugins=("assets.py",),
                           dtype=dtype, parameters={}, budget=SearchBudget(4, 30),
                           validation=ValidationBudget(max_configs=4, max_cases=0), warmup=1, repetitions=3)
    directory.mkdir(parents=True, exist_ok=True)
    files[0].write_text(source, encoding="utf-8")
    spec.save(files[1])
    files[2].write_text("# Local extension\n\n"
                        "Edit `assets.py` to implement the Kernel or search algorithm; "
                        "edit `experiment.json` to select research variables.\n\n"
                        f"```bash\nktlab check --spec {files[1]} --output {directory / 'check'}\n"
                        f"ktlab run --spec {files[1]} --output {directory / 'results'}\n```\n\n"
                        "CUDA checks remain deferred on a CPU-only host. "
                        "See the repository's docs/EXTENDING.md for the public contract.\n", encoding="utf-8")
    return {"kind": kind, "files": [str(path) for path in files], "operator": op.ref,
            "runtime": runtime, "spec": str(files[1]), "notice": "Working baseline template; edit the SDK implementation entry point."}
