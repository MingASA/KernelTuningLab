# Interpreting generated reports

`generate_report(records, output_dir, replay_results=(), l3_results=())` creates
Markdown, seven base SVG figures plus individual search/L3 cohort figures,
and JSON audit artifacts using the standard library. Matplotlib is optional
and loads lazily to render matching PNGs.
Empty inputs produce explicit insufficient-data results.

## Cohorts and correctness

Every configuration comparison retains hardware, operator, and provenance.
Heatmap cases additionally retain shape, split, and experiment identity. Search
curves group by hardware ID, operator ID, context, split when supplied, method,
provenance, source protocol/context ID when supplied, wall-time budget, target
ratio, and requested evaluation budget. L3 curves retain the supplied hardware,
operator, and workload context. Different GPU or operator results never become
one averaged curve. Result callers should provide `hardware_id`,
`hardware_name`, `operator_id`, and optionally `operator_name` for clear labels.
Native result objects lacking context are labelled unspecified; the report
does not invent a hardware identity.

The main report presents **one figure per hardware/operator/source cohort**.
Its legend contains only the comparison methods, typically six L2 methods or
four L3 methods. Hardware and readable operator names appear in the title.
The chart index provides direct SVG/PNG links, cohort plots appear in collapsible
details, and large score tables are collapsed separately. Combined
`regret_budget.svg` and `k_loss.svg` overlays are retained as compatibility
artifacts; they are linked rather than embedded as the main reading path.
Reports fill missing display names from the supplied measurement records when
the corresponding hardware/operator IDs are present.

Only successful records whose metadata does not explicitly contradict
verification enter latency comparisons. Public imports default to
`correctness_unknown`; they remain in the inventory and outcome counts, while
their timings do not enter verified heatmaps, transfer diagnostics, dimension
comparisons, replay optima, or L3 matrices. Unknown correctness is distinct from
invalid configuration, compilation failure, incorrect output, and runtime
failure. Slow correct configurations remain successful.

MOCK figures and report headers explicitly identify synthetic data. Synthetic
hardware characteristics and measurements demonstrate the infrastructure;
they cannot support claims about H200, H800, RTX 4090, or any physical GPU.

## Metric definitions

| Output | Definition and interpretation |
| --- | --- |
| Configuration heatmap | A record's runtime median, aggregated by median across repeated records for the same case/config, divided by that case's best observed verified latency. Missing/failed/unknown cells stay gray. |
| Regret versus budget | `best_found / best_observed - 1`, shown as percent. At one requested budget, mean and population standard deviation summarize supplied seeds/cases within the same hardware/operator cohort. Failed runs remain in the denominator of run counts. |
| Replay evaluation count | Actual requested attempts, which may be smaller than the requested budget when a search exhausts candidates or a time limit. Failures and missing measurements still consume attempts. |
| Measurement cost | Historical source cost replayed by the evaluator, separate from actual local CPU replay duration. Incomplete source cost information remains unavailable. |
| Budget to target | Evaluations needed to reach each run's stored latency target ratio. Mean is conditional on reaching the target; reached/run counts expose non-attainment. |
| GPU transfer matrix | For a matched operator/shape/provenance/split/experiment, choose the source's observed best config and divide its target measured latency by the target's observed best latency. Average only matched measured pairs; no interpolation or hardware prediction. |
| Dimension sensitivity | Within a fully matched case, keep every other parameter fixed and normalize each dimension value to that matched slice's best. Geometric means combine these ratios only within one hardware/operator/provenance/split/experiment cohort. |
| K-loss | `expected_latency / observed_unlimited_variants_reference - 1`. The reference uses each shape's observed best. K means a maximum compiled variant budget, so selected count may be smaller. |
| Failure rate | Known failure counts divided by all attempts and, separately, by known outcomes. Unknown correctness is shown separately. |
| Observed coverage | Attempted/verified shape-config cells divided by the rectangle of observed shapes × the observed config union in a cohort. This is explicitly not complete legal search-space coverage. |

The report's geometric mean does not average unrelated absolute latencies.
For example, vector widths 1 and 4 are comparable only when they were both
verified on the same input shape and all other parameters were fixed. A change
in vector width alongside a block-size change is confounded and creates no
matched sensitivity observation. If a vector width changes padding, masking,
alignment, or supported inputs, the runner's correctness and legality checks
must establish applicability before a performance comparison is meaningful.

Shapes are generic mappings. GEMM dimensions, batch dimensions, and row-wise
softmax/layernorm dimensions are not interchangeable. The report uses exact
operator and shape identities and does not reinterpret `ROWS`, `COLS`, or a
public import's `X`/`Y` dimensions as GEMM `M`/`N`/`K`.

## Incomplete data and evidence limits

An observed optimum is the best verified entry in the supplied table. It is not
a guarantee over missing configurations. An incomplete L3 matrix can still
cover every positive-weight shape using selected observed cells, in which case
an observed expected latency is meaningful; the unknown complete-matrix
optimum remains uncertified. If any positive-weight shape is uncovered, the
full expectation and K-loss are unavailable; conditional latency and covered
weight are reported separately.

Report normalization is descriptive; it does not grant searchers access to
the hidden replay oracle. Reports contain held-out scores and must not be fed
back into policy generation or global parameter fitting. Hardware and shape
distribution weights belong to experiment specifications; the report does not
infer deployment prevalence from the number of recorded runs.

The heatmap displays at most 30 cases and 48 configuration columns, transfer
matrices at most 16 hardware/provenance identities, and sensitivity tables at
most 80 value/cohort groups. Truncation is explicitly noted. Full inventory and
solver/replay JSON remain available. Figure tooltips retain full group names
when chart labels are shortened.

For public-source licenses, original cache fields, sample selection, pinned
source hashes, and caller-supplied verification evidence, see [PUBLIC_DATA.md](PUBLIC_DATA.md).
