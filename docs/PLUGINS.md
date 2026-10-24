# Extending the framework

New contributors should start with the [public asset SDK](EXTENDING.md) and working `ktlab scaffold` templates. The complete Adapter instructions below remain compatible for advanced backends.

The core has no required Torch, Triton, NumPy, AI SDK or paid API dependency. New implementations can be developed and tested against Mock observations before connecting a compiler or device.

## Backend factories and the CLI

```python
from kernel_tuning_lab.backends import register_adapter

def register_plugin():
    register_adapter("my_backend", MyKernelAdapter)
```

The factory is an ordinary Python callable accepting your adapter constructor arguments. Built-in names are reserved; duplicate plugin names require explicit `replace=True`. Import your trusted module explicitly:

```bash
ktlab --plugin path/to/my_backend.py benchmark --spec my_experiment.json
```

The JSON `backend` value selects the registered factory. The CLI imports the requested module and invokes `register_plugin()` when provided. Plugins can also register factories directly during module import. `examples/plugins/deterministic_mock.py` and `examples/experiments/plugin_mock.json` provide a runnable example.

Plugins execute trusted Python code in the current process. The loader does not claim operating-system isolation, dependency installation, remote discovery or credential management.

## A new kernel implementation

Implement the `KernelAdapter` protocol in `backends/base.py`:

| Callback | Contract |
| --- | --- |
| `environment()` | JSON metadata, runtime/compiler/device versions and implementation source identity; no credentials |
| `protocol_metadata()` | JSON measurement settings and numerical tolerances affecting cache compatibility |
| `validate(hardware, operator, shape, config)` | `ValidationResult`; reject unsupported layouts, dtypes and features explicitly |
| `prepare(hardware, operator, shape, seed)` | Seeded inputs and independent correctness reference; kept outside JIT timing |
| `compile(prepared, config)` | Compiled/launchable object; exceptions become compile failures |
| `check_correctness(compiled)` | `(bool, JSON details)`; no latency ranking or oracle labels |
| `warmup(compiled, iterations)` | Perform and synchronize the declared warmup mode |
| `measure(compiled, repetitions)` | Exactly that many finite positive runtime samples in milliseconds |

Set `name`, `provenance` and `measurement_mode` on the adapter. Use `mock` only for synthetic samples, `measured` for physical runs, and the public-data adapters for imported observations. Never write predicted latency as a successful physical measurement.

Create an `OperatorSpec` with a versioned implementation, precise dtype and correctness baseline. Choose its own `ConfigSpace(domains, constraints, name)`; constraints return a rejection reason or `None`. Hardware feature availability does not make a feature part of your kernel. For example, the portable GEMM does not expose TMA/cluster/split-K just because H200 supports them.

Programmatic workflows can pass custom spaces directly to `BenchmarkRunner`, `tune` or `run_replay`. For portable CLI JSON, built-in operator kinds use their declared validators; a new operator kind delegates implementation-specific legality to the adapter. Save any custom schema/version in operator parameters and protocol metadata. A concrete TileLang adapter is future work; this contract already provides the integration boundary.

## Searchers and policies

Implement `Searcher.suggest(hardware, operator, shape, space, history, budget, batch_size)` and pass your object directly to `tune` or `run_replay`. `history` contains revealed records plus identities of requested missing configurations. The scorer's private table and reference optimum never belong in your algorithm. Keep hyperparameters frozen before held-out evaluation.

For reusable policies, implement `select_config` or `search_config`; trusted generated modules export `create_policy(space, seed)`. Use `SafePolicy`, `PolicyRepository` and `evaluate_policy` for legality fallback, source snapshots and independent scoring. Select mode scores ranked Top-1 without deployment measurements; testing a set of predictions with an oracle does not establish a deployable zero-shot choice. See [SEARCH_POLICY.md](SEARCH_POLICY.md).

## Larger datasets and spaces

Prefer `space.config_at` or seeded `space.sample_configs` for large Cartesian spaces. The built-in searchers switch to bounded pools above 4096 combinations; `--candidate-pool-size` controls the pool for online/replay commands. This is an explicit approximation, and global regret cannot be reported without a known full reference.

Use `store.add_many`, `store.iter_records`, transactional `import_jsonl`, and atomic `export_jsonl` for batches and exchange. Existing database analysis uses read-only connections. Hub files have fixed upstream identities and verified bytes; correctness evidence is tracked separately from download integrity.

`ktlab scale-demo` exercises the larger workflow. CPU scale, budget and fault regressions live in `tests/test_integration_scale.py` and `tests/test_scale_contracts.py`; physical GPU validation is deferred to the target hosts.
## Recommended contributor workflow

For new Kernel implementations, operators and search algorithms, use the public asset SDK described in [EXTENDING.md](EXTENDING.md). It provides shared operator services, versioned discovery, effective spaces, automatic acceptance, complete-flow execution and supervised configuration-driven runs. `ktlab scaffold` generates working contributor templates. The complete Adapter protocol below remains available for advanced backends and existing v1 integrations.

