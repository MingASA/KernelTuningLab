# Automatic integration plan

## Goal

Add a complete local, configuration-driven extension workflow. A contributor supplies a versioned Kernel or search algorithm and its declarations; an experiment selects those assets. Operator input generation, references, tolerances, acceptance, budgets, execution and archiving belong to the platform. No Web UI is included.

## Decisions

- Operator semantics, Kernel implementation and search algorithms are separate versioned assets. References are `id@version`; ambiguous unversioned references fail. Local Python files/modules and installed entry points can register assets without editing core lists.
- This contract supports finite discrete spaces, including integer, finite real, boolean, categorical and conditional parameters. Infinite/continuous domains are rejected explicitly. Effective spaces intersect Kernel declarations, input/hardware constraints and experiment restrictions; experiments cannot expand Kernel capabilities.
- An operator owns shape/dtype/parameter semantics, seeded inputs, an independent reference, tolerances and validation cases. A Kernel normally supplies a build entry point and parameter declarations, with an optional argument map. Existing adapters and v1 experiments remain usable.
- A single callable is the simple execution plan. Complex implementations declare ordered steps, workspace and initialization. Per-invocation initialization/reset is inside both correctness execution and complete-flow timing. Allocation, compilation and one-time setup are separately recorded. CUDA full-flow timing synchronizes around the complete invocation.
- Submitter correctness claims are version/content/scope-bound assertions. Accepted claims may skip reference comparisons within scope, but produce `submitter_declared` evidence, never platform-verified correctness. Direct failures override claims. Sampled acceptance proves only the sampled cases/configs. Declared-only observations are excluded from default verified replay/L3 oracles.
- Outside claim scope, acceptance uses deterministic configuration sampling with a separate hard budget. Actual research measurements continue to check each unclaimed configuration. Invalid/unsupported input, failures and insufficient evidence are explicit records, never silently dropped.
- Algorithm acceptance combines declared capability checks with bounded synthetic interactions on the resolved space. The platform holds the authoritative space and budgets, exposes a read-only space view, bounds candidate consumption, and rejects illegal, repeated or oversized proposals. It does not silently replace an incompatible algorithm.
- V2 research budgets count revealed candidate observations, including cached outcomes and failures. New physical evaluations and cache reuse remain separate counters, so combinations can reuse measurements without granting later algorithms extra free observations. Direct v1 Runner admission semantics are retained.
- A parent process enforces phase wall deadlines for plugin loading/preflight, correctness acceptance and research execution. Timeout archives survive worker termination. Terminating a process is not a guarantee of resetting an unresponsive GPU driver.
- CUDA absence is `deferred` for physical checks. CPU examples run real Python implementations and are labelled CPU measurements; they do not manufacture GPU validation evidence.

## Implementation order

1. **Contracts and discovery:** public SDK, versioned catalog, manifests/fingerprints, plugin and entry-point loading, declared parameter spaces and immutable views.
2. **Reusable operator/runtime services:** four current operators, seeded Python/CUDA inputs and references, common verification, single/multiple-step execution, workspace/reset rules, content-bound claims.
3. **Automatic acceptance:** schema/signature/space/capability checks, synthetic algorithm tests, scoped claims and reproducible bounded reference checks with structured evidence.
4. **Experiment orchestration:** v2 asset configuration, automatic Kernel × algorithm × shape matching, phase deadlines, budgeted tuning, caching and SQLite/JSONL archives, dynamic CLI discovery/run/check commands.
5. **Contributor experience:** independent success/failure plugins, custom operator and algorithm examples, composed-flow example, CPU-only and CUDA configurations, a concise onboarding guide.
6. **Acceptance evidence:** tests written against public SDK/configuration, existing regressions, CPU end-to-end runs, missing-CUDA/timeout/failure paths and package checks. Delegate execution and long logs to Luna.

## Required acceptance cases

- New same-operator Kernel, custom operator, composed workflow and custom algorithm load from external files, using only public documentation and interfaces.
- Kernel and algorithms combine repeatedly without bespoke glue; algorithm versions affect trial identity, while compatible Kernel measurements remain reusable.
- Wrong outputs/NaNs, missing parameter mappings, unknown knobs, widened/empty spaces, incompatible conditional/categorical spaces, expired/out-of-scope claims, incorrect proposals, repeated candidates and timeouts are archived with non-success status.
- Source/reference/constraint changes invalidate incompatible evidence and caches. Old successful run markers cannot approve an interrupted retry.
- Validation and search attempts/time are reported separately; rejected and failed attempts are accounted for. Zero search budget performs no research evaluation.
- No CUDA dependency is imported by ordinary asset discovery or CPU workflows. GPU acceptance stays deferred until the corresponding physical machine is available.

## Deliverables

Public SDK and schema, implementation, runnable plugin/configuration examples, onboarding and reference documents, structured acceptance/lock/result artifacts, and a reproducible CPU validation summary. H200/H800/4090/5060 Ti physical runs remain on their respective machines.
