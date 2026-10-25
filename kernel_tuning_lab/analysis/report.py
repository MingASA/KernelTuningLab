"""Markdown and SVG analysis with explicit provenance and missing-data semantics."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, is_dataclass
from html import escape
import json
import math
from pathlib import Path
import re
import statistics
import textwrap
from typing import Any, Iterable, Mapping, Sequence
import xml.etree.ElementTree as ET

from ..models import MeasurementRecord, RecordStatus, canonical_json, stable_id


_COLORS = ("#2563eb", "#d97706", "#059669", "#dc2626", "#7c3aed", "#0891b2", "#be185d", "#475569")
_UNKNOWN = "correctness_unknown"
_PLOTTING: Any = None


def _plain(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "to_dict"):
        return dict(value.to_dict())
    if is_dataclass(value):
        return asdict(value)
    return dict(vars(value))


def _number(value: Any) -> bool:
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def _fmt(value: Any, digits: int = 4) -> str:
    return f"{value:.{digits}g}" if _number(value) else "N/A"


def _md(value: Any) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ").replace("\r", " ")


def _table(headers: Sequence[str], rows: Iterable[Sequence[Any]]) -> str:
    lines = ["| " + " | ".join(map(_md, headers)) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    lines.extend("| " + " | ".join(map(_md, row)) + " |" for row in rows)
    return "\n".join(lines)


def _label(record: MeasurementRecord) -> str:
    dims = ",".join(f"{key}={value}" for key, value in sorted(record.shape.dims.items()))
    return f"{record.hardware.name} / {record.operator.name} / {dims} [{record.provenance}; {record.split}; {record.experiment_id}]"


def _case(record: MeasurementRecord) -> tuple[str, ...]:
    return (record.hardware.id, record.operator.id, record.shape.id, record.provenance, record.split, record.experiment_id)


def _cohort(record: MeasurementRecord) -> tuple[str, ...]:
    return (record.hardware.id, record.operator.id, record.provenance, record.split, record.experiment_id)


def _verified(record: MeasurementRecord) -> bool:
    return record.correctness_verified and record.median_ms is not None


def _context(provenances: Iterable[str]) -> str:
    values = sorted(set(provenances))
    if not values:
        return "Insufficient data"
    if values == ["mock"]:
        return "MOCK / SYNTHETIC DATA — not measured GPU performance"
    if "mock" in values or "mixed" in values:
        return "Includes MOCK / SYNTHETIC data; provenance groups are labelled"
    return "Data provenance: " + ", ".join(values)


def _result_cohort(row: Mapping[str, Any]) -> tuple[str, str]:
    """Preserve hardware/operator/workload identity before aggregating seeds."""
    metadata = row.get("metadata") or {}
    name = str(row.get("hardware_name") or row.get("hardware_id") or row.get("context") or metadata.get("workload") or "unspecified")
    context = str(row.get("context") or metadata.get("workload") or "")
    identity = canonical_json([row.get("hardware_id", name), row.get("operator_id", row.get("operator_name", "")), context,
                               row.get("split", ""), row.get("source_context_id", ""), row.get("measurement_mode", ""),
                               row.get("max_wall_time_s"), row.get("target_ratio")])
    label = name + (f" / {context}" if context and context != name and not context.startswith(name) else "")
    if row.get("hardware_id"):
        label += f" ({str(row['hardware_id'])[:4]})"
    if row.get("operator_name") or row.get("operator_id"):
        operator_label = str(row.get("operator_name") or "op")
        if row.get("operator_id"):
            operator_label += f" ({str(row['operator_id'])[:6]})"
        label += " / " + operator_label
    if row.get("split"):
        label += f" / {row['split']}"
    if row.get("measurement_mode"):
        label += f" / {row['measurement_mode']}"
    if row.get("source_context_id"):
        label += f" / source={str(row['source_context_id'])[:8]}"
    return identity, label


def _text(x: float, y: float, text: Any, *, size: int = 12, color: str = "#334155", anchor: str = "start", extra: str = "") -> str:
    return f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" fill="{color}" text-anchor="{anchor}" {extra}>{escape(str(text))}</text>'


def _svg(path: Path, title: str, subtitle: str, width: int, height: int, content: Sequence[str], *, render_png: bool = True) -> None:
    body = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
            f"<title id=\"title\">{escape(title)}</title>", f"<desc id=\"desc\">{escape(subtitle)}</desc>",
            '<rect width="100%" height="100%" fill="#ffffff"/>',
            '<g font-family="Arial, Helvetica, sans-serif">', _text(24, 30, title, size=20, color="#0f172a"),
            _text(24, 54, subtitle[:160], size=12, color="#9f1239" if "MOCK" in subtitle else "#64748b"), *content, "</g></svg>"]
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
    if render_png:
        _optional_png(path)


def _optional_png(path: Path) -> None:
    """Render this module's small SVG subset with optional lazy matplotlib.

    SVG remains the canonical, dependency-free artifact. Using the same draw
    primitives for PNG keeps both formats aligned without a Cairo dependency.
    """
    global _PLOTTING
    if _PLOTTING is None:
        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg
            from matplotlib.figure import Figure
            from matplotlib.patches import Circle, Rectangle
            _PLOTTING = (FigureCanvasAgg, Figure, Circle, Rectangle)
        except ImportError:
            _PLOTTING = False
    if not _PLOTTING:
        return
    FigureCanvasAgg, Figure, Circle, Rectangle = _PLOTTING
    root = ET.parse(path).getroot()
    width, height = float(root.attrib["width"]), float(root.attrib["height"])
    figure = Figure(figsize=(width / 100, height / 100), dpi=100, facecolor="white")
    FigureCanvasAgg(figure)
    axes = figure.add_axes((0, 0, 1, 1))
    axes.set_xlim(0, width)
    axes.set_ylim(height, 0)
    axes.set_axis_off()
    anchors = {"start": "left", "middle": "center", "end": "right"}

    def color(value: str) -> Any:
        if value == "none":
            return "none"
        if value.startswith("rgb("):
            return tuple(int(part) / 255 for part in value[4:-1].split(","))
        return value

    for element in root.iter():
        tag = element.tag.rsplit("}", 1)[-1]
        a = element.attrib
        if tag == "rect":
            if a.get("width") == "100%":
                continue
            axes.add_patch(Rectangle((float(a.get("x", 0)), float(a.get("y", 0))), float(a["width"]), float(a["height"]), facecolor=color(a.get("fill", "none")), edgecolor="none"))
        elif tag == "circle":
            axes.add_patch(Circle((float(a["cx"]), float(a["cy"])), float(a["r"]), facecolor=color(a.get("fill", "none")), edgecolor="none"))
        elif tag == "line":
            axes.plot([float(a["x1"]), float(a["x2"])], [float(a["y1"]), float(a["y2"])], color=color(a.get("stroke", "#334155")), linewidth=float(a.get("stroke-width", 1)) * 0.72)
        elif tag == "polyline":
            points = [[float(value) for value in pair.split(",")] for pair in a["points"].split()]
            axes.plot([point[0] for point in points], [point[1] for point in points], color=color(a.get("stroke", "#334155")), linewidth=float(a.get("stroke-width", 1)) * 0.72)
        elif tag == "text":
            rotation = re.match(r"rotate\(([-\d.]+)", a.get("transform", ""))
            axes.text(float(a["x"]), float(a["y"]), element.text or "", fontsize=float(a.get("font-size", 12)) * 0.72, color=color(a.get("fill", "#334155")), ha=anchors.get(a.get("text-anchor", "start"), "left"), va="baseline", rotation=-float(rotation.group(1)) if rotation else 0, rotation_mode="anchor")
    figure.savefig(path.with_suffix(".png"), dpi=100, facecolor="white")
    figure.clear()


def _empty_chart(path: Path, title: str, context: str, reason: str, *, render_png: bool = True) -> None:
    _svg(path, title, context, 960, 180, [_text(28, 105, "Insufficient data", size=18), _text(28, 134, reason[:135])], render_png=render_png)


def _line_chart(path: Path, title: str, context: str, series: Mapping[str, Sequence[tuple[float, float]]], xlabel: str, ylabel: str, *, series_labels: Mapping[str, str] | None = None, render_png: bool = True) -> None:
    filtered = {key: sorted((float(x), float(y)) for x, y in values if _number(x) and _number(y)) for key, values in series.items()}
    filtered = {key: value for key, value in filtered.items() if value}
    if not filtered:
        _empty_chart(path, title, context, "No finite scored points were supplied.", render_png=render_png)
        return
    width, height, left, right, top, bottom = 1100, 540, 100, 810, 95, 440
    xs = [x for values in filtered.values() for x, _ in values]
    ys = [y for values in filtered.values() for _, y in values]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(0.0, min(ys)), max(ys)
    if xmax == xmin:
        xmin, xmax = max(0, xmin - 1), xmax + 1
    if ymax == ymin:
        ymax = ymin + 1
    ymax += (ymax - ymin) * 0.08
    px = lambda x: left + (x - xmin) / (xmax - xmin) * (right - left)
    py = lambda y: bottom - (y - ymin) / (ymax - ymin) * (bottom - top)
    parts = []
    for i in range(6):
        y = ymin + (ymax - ymin) * i / 5
        yy = py(y)
        parts.extend([f'<line x1="{left}" x2="{right}" y1="{yy}" y2="{yy}" stroke="#e2e8f0"/>', _text(left - 12, yy + 4, _fmt(y, 3), anchor="end")])
    ticks = sorted(set(xs))
    if len(ticks) > 10:
        ticks = [xmin + (xmax - xmin) * i / 8 for i in range(9)]
    for x in ticks:
        parts.append(_text(px(x), bottom + 25, _fmt(x, 3), anchor="middle"))
    parts.extend([f'<line x1="{left}" x2="{right}" y1="{bottom}" y2="{bottom}" stroke="#94a3b8"/>',
                  _text((left + right) / 2, height - 35, xlabel, size=14, anchor="middle"),
                  _text(25, (top + bottom) / 2, ylabel, size=13, anchor="middle", extra=f'transform="rotate(-90 25 {(top + bottom) / 2})"')])
    legend_cursor = top
    for i, (series_key, values) in enumerate(sorted(filtered.items())):
        name = (series_labels or {}).get(series_key, series_key)
        color = _COLORS[i % len(_COLORS)]
        points = " ".join(f"{px(x):.2f},{py(y):.2f}" for x, y in values)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        for x, y in values:
            parts.append(f'<circle cx="{px(x):.2f}" cy="{py(y):.2f}" r="3" fill="{color}"><title>{escape(name)}: {_fmt(x)}, {_fmt(y)}</title></circle>')
        legend_lines = textwrap.wrap(name, width=35) or [name]
        parts.append(f'<g><title>{escape(name)}</title><line x1="840" x2="858" y1="{legend_cursor}" y2="{legend_cursor}" stroke="{color}" stroke-width="3"/>')
        parts.extend(_text(865, legend_cursor + 4 + line_index * 14, line, size=11) for line_index, line in enumerate(legend_lines))
        parts.append("</g>")
        legend_cursor += max(22, len(legend_lines) * 14 + 8)
    _svg(path, title, context, width, max(height, legend_cursor + 30), parts, render_png=render_png)


def _heatmap(path: Path, title: str, context: str, rows: Sequence[str], columns: Sequence[str], values: Sequence[Sequence[float | None]], *, cell_titles: Sequence[Sequence[str]] | None = None) -> None:
    if not rows or not columns:
        _empty_chart(path, title, context, "No comparable verified measurements are available.")
        return
    cell_width, cell_height, left, top = 38, 33, 350, 128
    width, height = max(960, left + len(columns) * cell_width + 35), max(240, top + len(rows) * cell_height + 65)
    available = [v for row in values for v in row if v is not None and _number(v)]
    maximum = max(available, default=1.0)
    span = max(math.log(max(maximum, 1.001)), 0.01)
    parts = []
    for j, column in enumerate(columns):
        x = left + j * cell_width + cell_width / 2
        parts.append(_text(x, top - 13, column[:11], size=10, anchor="start", extra=f'transform="rotate(-55 {x} {top - 13})"'))
    for i, row in enumerate(rows):
        y = top + i * cell_height
        short = row if len(row) <= 49 else row[:46] + "..."
        parts.append(_text(left - 10, y + 21, short, size=10, anchor="end"))
        for j, value in enumerate(values[i]):
            title_text = cell_titles[i][j] if cell_titles is not None else f"{row} / {columns[j]}: {_fmt(value)}"
            if value is None:
                fill = "#e2e8f0"
                label = "—"
            else:
                level = min(1.0, max(0.0, math.log(max(1.0, value)) / span))
                fill = f"rgb({int(45 + level * 200)},{int(185 - level * 90)},{int(140 - level * 75)})"
                label = f"{value:.2g}"
            x = left + j * cell_width
            parts.extend([f'<rect x="{x}" y="{y}" width="{cell_width - 2}" height="{cell_height - 2}" rx="2" fill="{fill}"><title>{escape(title_text)}</title></rect>', _text(x + (cell_width - 2) / 2, y + 20, label, size=10, color="#0f172a", anchor="middle")])
    parts.append(_text(24, height - 20, "Ratio to observed best; 1 is best. Gray = missing, failed, or correctness unknown.", size=11))
    _svg(path, title, context, width, height, parts)


def _bar_chart(path: Path, title: str, context: str, labels: Sequence[str], values: Sequence[float], *, xlabel: str, render_png: bool = True) -> None:
    if not labels:
        _empty_chart(path, title, context, "No numeric groups are available for this chart.", render_png=render_png)
        return
    width, left, right, top = 1080, 335, 970, 90
    wrapped = [textwrap.wrap(label, width=46) or [label] for label in labels]
    row_heights = [max(32, len(lines) * 14 + 12) for lines in wrapped]
    height = max(220, top + sum(row_heights) + 60)
    maximum = max(max(values, default=0), 0.01)
    parts = []
    cursor = top
    for i, (label, value, lines, row_height) in enumerate(zip(labels, values, wrapped, row_heights)):
        y = cursor + (row_height - 32) / 2
        length = (right - left) * value / maximum
        parts.append(f'<g><title>{escape(label)}</title>')
        parts.extend(_text(left - 10, cursor + (row_height - len(lines) * 14) / 2 + 12 + n * 14,
                           line, size=11, anchor="end") for n, line in enumerate(lines))
        parts.extend([
                      f'<rect x="{left}" y="{y + 3}" width="{length:.3f}" height="22" rx="3" fill="{_COLORS[i % len(_COLORS)]}"/>',
                      _text(left + length + 8, y + 20, _fmt(value, 3), size=11), "</g>"])
        cursor += row_height
    parts.append(_text((left + right) / 2, height - 15, xlabel, size=13, anchor="middle"))
    _svg(path, title, context, width, height, parts, render_png=render_png)


def _grouped(records: Sequence[MeasurementRecord]) -> tuple[dict[tuple[str, ...], list[MeasurementRecord]], dict[tuple[str, ...], dict[str, float]]]:
    cases: dict[tuple[str, ...], list[MeasurementRecord]] = defaultdict(list)
    samples: dict[tuple[tuple[str, ...], str], list[float]] = defaultdict(list)
    for record in records:
        case = _case(record)
        cases[case].append(record)
        if _verified(record):
            samples[case, record.config.id].append(float(record.median_ms))
    latency: dict[tuple[str, ...], dict[str, float]] = defaultdict(dict)
    for (case, config), values in samples.items():
        latency[case][config] = statistics.median(values)
    return dict(cases), dict(latency)


def _config_heatmap(records: Sequence[MeasurementRecord], cases: Mapping[tuple[str, ...], list[MeasurementRecord]], latencies: Mapping[tuple[str, ...], dict[str, float]], output: Path, context: str) -> list[str]:
    configs = {record.config.id: record.config for record in records}
    config_ids = sorted(configs)[:48]
    case_keys = sorted(cases)[:30]
    values = []
    for case in case_keys:
        observed = latencies.get(case, {})
        best = min(observed.values(), default=None)
        values.append([observed[config] / best if config in observed and best is not None else None for config in config_ids])
    _heatmap(output / "config_heatmap.svg", "Configuration latency heatmap", context, [_label(cases[key][0]) for key in case_keys], [key[:6] for key in config_ids], values)
    lines = ["## Configuration performance", "", "Each cell is median latency / best observed verified latency in the same hardware, operator, shape, provenance, split, and experiment. Repeated records are combined using the median of record medians. Gray cells include missing, failed, and correctness unknown configurations.", "", "![Configuration performance](config_heatmap.svg)", ""]
    if not latencies:
        lines += ["**Insufficient data:** no correctness-verified successful measurements. Imported timings with unknown correctness are retained in the data inventory and excluded from performance scoring.", ""]
    if len(configs) > 48 or len(cases) > 30:
        lines += [f"Displayed first {len(config_ids)} of {len(configs)} configurations and {len(case_keys)} of {len(cases)} cases in stable ID order. The inventory and coverage statistics include every supplied record.", ""]
    lines += [_table(["Config ID", "Parameters"], [(key[:6], "`" + canonical_json(configs[key].params) + "`") for key in config_ids]), ""]
    return lines


def _cohort_curve_figures(
    output: Path,
    prefix: str,
    title: str,
    curves: Mapping[tuple[str, str], Mapping[str, Sequence[tuple[float, float]]]],
    examples: Mapping[tuple[str, str], dict[str, Any]],
    xlabel: str,
    ylabel: str,
) -> list[str]:
    """Keep method comparisons readable as the number of operators/GPUs grows."""
    charts = []
    for (provenance, cohort), methods in sorted(curves.items()):
        example = examples[provenance, cohort]
        hardware = str(example.get("hardware_name") or example.get("hardware_id") or "unspecified hardware")
        operator = str(example.get("operator_name") or example.get("operator_id") or "unspecified operator")
        chart_title = f"{title}: {hardware} / {operator} [{provenance}]"
        stem = prefix + "_" + stable_id([provenance, cohort])[:12]
        svg_path = output / (stem + ".svg")
        context = _context([provenance])
        if example.get("measurement_mode"):
            context += f" | mode={example['measurement_mode']}"
        if example.get("source_context_id"):
            context += f" | source={str(example['source_context_id'])[:8]}"
        _line_chart(svg_path, chart_title, context, methods, xlabel, ylabel)
        label = _result_cohort(example)[1]
        links = f"[SVG]({svg_path.name})"
        png_path = svg_path.with_suffix(".png")
        if png_path.exists():
            links += f" · [PNG]({png_path.name})"
        charts.append((label, provenance, len(methods), links, chart_title, svg_path.name))
    if not charts:
        return []
    lines = ["Each figure compares methods within one hardware/operator/source cohort. Open a cohort below for its figure; legends contain only the search or selection methods.", "",
             _table(["Hardware / operator / workload", "Source", "Methods", "Figure"], [row[:4] for row in charts]), ""]
    for index, (label, provenance, _, links, chart_title, filename) in enumerate(charts):
        lines += [f'<details{" open" if index == 0 else ""}>', f"<summary>{escape(label)} [{escape(provenance)}]</summary>", "",
                  f"![{_md(chart_title)}]({filename})", "", links, "", "</details>", ""]
    return lines


def _replay_report(results: Sequence[dict[str, Any]], output: Path) -> list[str]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    cohort_labels: dict[str, str] = {}
    for row in results:
        budget = row.get("budget")
        if isinstance(budget, int) and not isinstance(budget, bool):
            cohort, label = _result_cohort(row)
            cohort_labels[cohort] = label
            grouped[str(row.get("algorithm", "unknown")), str(row.get("provenance", "unspecified")), cohort, budget].append(row)
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    series_labels: dict[str, str] = {}
    cohort_curves: dict[tuple[str, str], dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    cohort_examples: dict[tuple[str, str], dict[str, Any]] = {}
    rows = []
    for (algorithm, provenance, cohort, budget), runs in sorted(grouped.items()):
        cohort_examples[provenance, cohort] = runs[0]
        cohort_curves.setdefault((provenance, cohort), defaultdict(list))
        regrets = [float(row["regret"]) for row in runs if _number(row.get("regret"))]
        evaluations = [float(row["evaluations"]) for row in runs if _number(row.get("evaluations"))]
        costs = [float(row["source_measurement_cost_s"]) for row in runs if _number(row.get("source_measurement_cost_s"))]
        cpus = [float(row["cpu_wall_time_s"]) for row in runs if _number(row.get("cpu_wall_time_s"))]
        targets = [float(row["target_evaluations"]) for row in runs if _number(row.get("target_evaluations"))]
        if regrets:
            series_key = canonical_json([algorithm, provenance, cohort])
            series[series_key].append((budget, statistics.mean(regrets) * 100))
            series_labels[series_key] = f"{algorithm} / {cohort_labels[cohort]} [{provenance}]"
            cohort_curves[provenance, cohort][algorithm].append((budget, statistics.mean(regrets) * 100))
        rows.append([algorithm, provenance, cohort_labels[cohort], budget, len(runs), len(regrets), _fmt(statistics.mean(regrets) * 100) if regrets else "N/A", _fmt(statistics.pstdev(regrets) * 100) if regrets else "N/A", _fmt(statistics.mean(evaluations)) if evaluations else "N/A", _fmt(statistics.mean(costs)) if costs else "N/A", f"{len(costs)}/{len(runs)}", _fmt(statistics.mean(cpus)) if cpus else "N/A", _fmt(statistics.mean(targets)) if targets else "N/A", f"{len(targets)}/{len(runs)}"])
    _line_chart(output / "regret_budget.svg", "Search regret versus evaluation budget", _context(row.get("provenance", "unspecified") for row in results), series, "Requested evaluation budget", "Mean observed regret (%)", series_labels=series_labels)
    lines = ["## Search replay", "", "Regret = best latency found / best known verified latency − 1. Curves preserve hardware, operator, workload context, and provenance; points average scored runs at each requested budget within that cohort. The table reports population standard deviation across the supplied seeds/cases. Unknown or failed runs remain in the run count. References are observed optima, and partial source tables do not establish a global optimum.", ""]
    lines += _cohort_curve_figures(output, "regret_budget", "Search regret", cohort_curves, cohort_examples, "Requested evaluation budget", "Mean observed regret (%)")
    lines += ["[Combined overview SVG](regret_budget.svg) is retained for compatibility. Use the individual cohort figures for method comparisons.", ""]
    if rows:
        lines += ["<details>", "<summary>All search replay scores</summary>", "", _table(["Algorithm", "Provenance", "Hardware / workload", "Budget", "Runs", "Scored", "Mean regret %", "SD %", "Mean evals", "Source cost s", "Known costs", "CPU replay s", "Mean target evals", "Target reached"], rows), "", "</details>", "", "Source cost is replayed historical measurement cost; CPU replay duration is actual local execution time. Target evaluation averages are conditional on reaching each run's target ratio; target ratios are stored in replay_results.json. Missing source costs remain N/A.", ""]
    else:
        lines += ["**Insufficient data:** no replay results with a requested evaluation budget were supplied.", ""]
    return lines


def _transfer_report(cases: Mapping[tuple[str, ...], list[MeasurementRecord]], latencies: Mapping[tuple[str, ...], dict[str, float]], output: Path, context: str) -> list[str]:
    hardware = {(key[0], key[3]): records[0].hardware for key, records in cases.items() if latencies.get(key)}
    hw_ids = sorted(hardware)[:16]
    pairs: dict[tuple[tuple[str, str], tuple[str, str]], list[float]] = defaultdict(list)
    by_match: dict[tuple[str, ...], dict[tuple[str, str], dict[str, float]]] = defaultdict(dict)
    for case, values in latencies.items():
        if values:
            by_match[case[1:]][case[0], case[3]] = values
    for matched in by_match.values():
        for source, source_values in matched.items():
            source_best = min(source_values, key=lambda config: (source_values[config], config))
            for target, target_values in matched.items():
                if source_best in target_values:
                    pairs[source, target].append(target_values[source_best] / min(target_values.values()))
    values = [[statistics.mean(pairs[source, target]) if pairs.get((source, target)) else None for target in hw_ids] for source in hw_ids]
    titles = [[f"Source {hardware[source].name} -> target {hardware[target].name}: {_fmt(values[i][j])}; matched cases={len(pairs.get((source, target), []))}" for j, target in enumerate(hw_ids)] for i, source in enumerate(hw_ids)]
    labels = [f"{hardware[key].name} [{key[1]}] ({key[0][:4]})" for key in hw_ids]
    has_transfer = any(source[0] != target[0] for source, target in pairs)
    if not has_transfer:
        _empty_chart(output / "gpu_transfer.svg", "GPU configuration transfer", context, "At least two hardware identities and measured transferred configurations are required.")
    else:
        _heatmap(output / "gpu_transfer.svg", "GPU configuration transfer: source rows / target columns", context, labels, [str(i + 1) for i in range(len(labels))], values, cell_titles=titles)
    lines = ["## GPU configuration transfer", "", "For each matched operator/shape/provenance/split/experiment, select the best observed source configuration, then divide its measured target latency by the target's best observed latency. Cells average these ratios over matched cases. This diagnostic uses source observed optima, and does not estimate a budgeted transfer searcher's score. Missing target measurements are omitted rather than predicted.", "", "![GPU transfer matrix](gpu_transfer.svg)", ""]
    if not has_transfer:
        lines += ["**Insufficient data:** no pair of different hardware identities has a measured transferred configuration for a matched case and provenance.", ""]
    else:
        lines += [_table(["Column", "Hardware"], [(i + 1, name) for i, name in enumerate(labels)]), "", _table(["Source", "Target", "Provenance", "Matched cases", "Mean latency ratio"], [(hardware[source].name, hardware[target].name, source[1], len(ratios), _fmt(statistics.mean(ratios))) for (source, target), ratios in sorted(pairs.items()) if source in hw_ids and target in hw_ids]), ""]
    if len(hardware) > len(hw_ids):
        lines += [f"Displayed first {len(hw_ids)} of {len(hardware)} hardware/provenance identities in stable ID order.", ""]
    return lines


def _sensitivity_report(records: Sequence[MeasurementRecord], latencies: Mapping[tuple[str, ...], dict[str, float]], output: Path, context: str) -> list[str]:
    configs = {record.config.id: record.config for record in records if _verified(record)}
    cohort_labels = {(record.hardware.id, record.operator.id): f"{record.hardware.name} ({record.hardware.id[:4]}) / {record.operator.name} ({record.operator.id[:4]})" for record in records}
    dimensions = sorted({key for config in configs.values() for key in config.params})
    aggregate: dict[tuple[str, str, str, str, str, str, str], list[float]] = defaultdict(list)
    for case, values in latencies.items():
        for dimension in dimensions:
            slices: dict[str, list[tuple[str, float]]] = defaultdict(list)
            for config_id, latency in values.items():
                params = configs[config_id].params
                if dimension in params:
                    controls = canonical_json({key: value for key, value in params.items() if key != dimension})
                    slices[controls].append((canonical_json(params[dimension]), latency))
            for matched in slices.values():
                if len({value for value, _ in matched}) < 2:
                    continue
                best = min(latency for _, latency in matched)
                for value, latency in matched:
                    aggregate[case[0], case[1], case[3], case[4], case[5], dimension, value].append(latency / best)
    rows = [(f"{cohort_labels[hardware, operator]} / {split} / {experiment}", provenance, dimension, value, len(ratios), math.exp(statistics.mean(math.log(ratio) for ratio in ratios))) for (hardware, operator, provenance, split, experiment, dimension, value), ratios in sorted(aggregate.items())]
    shown = rows[:80]
    _bar_chart(output / "dimension_sensitivity.svg", "Configuration dimension sensitivity", context, [f"{dimension}={value} [{provenance}] / {cohort}" for cohort, provenance, dimension, value, _, _ in shown], [ratio for _, _, _, _, _, ratio in shown], xlabel="Geometric mean matched-slice latency / slice best")
    lines = ["## Configuration dimension sensitivity", "", "Only matched slices are compared: hardware, operator, shape, provenance, split, experiment, and every other configuration parameter are held fixed. Each value's latency is normalized to its slice's best latency, then ratios are combined with a geometric mean within the same hardware/operator/provenance/split/experiment cohort. This is descriptive evidence, not a causal model. A dimension with no matched variation is insufficient data.", "", "![Dimension sensitivity](dimension_sensitivity.svg)", ""]
    if rows:
        lines += [_table(["Hardware / operator / split / experiment", "Provenance", "Dimension", "Value", "Matched observations", "Geometric mean ratio"], [(cohort, p, d, v, n, _fmt(ratio)) for cohort, p, d, v, n, ratio in shown]), ""]
        if len(rows) > len(shown):
            lines += [f"Displayed {len(shown)} of {len(rows)} dimension/value groups in stable order.", ""]
    else:
        lines += ["**Insufficient data:** no verified measurements differ in exactly one configuration dimension within a case.", ""]
    return lines


def _l3_report(results: Sequence[dict[str, Any]], output: Path) -> list[str]:
    points: dict[tuple[str, str, str, int], list[float]] = defaultdict(list)
    cohort_labels: dict[str, str] = {}
    cohort_examples: dict[tuple[str, str], dict[str, Any]] = {}
    rows = []
    for row in results:
        method, provenance, k = str(row.get("method", "unknown")), str(row.get("provenance", "unspecified")), row.get("k")
        cohort, cohort_label = _result_cohort(row)
        cohort_labels[cohort] = cohort_label
        cohort_examples[provenance, cohort] = row
        loss = row.get("relative_loss")
        if _number(k) and _number(loss) and _number(row.get("expected_latency_ms")) and _number(row.get("coverage")) and row["coverage"] >= 1:
            points[method, provenance, cohort, int(k)].append(float(loss) * 100)
        metadata = row.get("metadata") or {}
        rows.append([method, provenance, cohort_label, k, len(row.get("selected_indices", row.get("configs", ()))), _fmt(row.get("expected_latency_ms")), _fmt(row.get("conditional_latency_ms")), _fmt(loss * 100) if _number(loss) else "N/A", _fmt(row.get("coverage")), str(bool(row.get("exact", False))), str(metadata.get("matrix_complete", "unknown"))])
    series: dict[str, list[tuple[float, float]]] = defaultdict(list)
    series_labels: dict[str, str] = {}
    cohort_curves: dict[tuple[str, str], dict[str, list[tuple[float, float]]]] = {key: defaultdict(list) for key in cohort_examples}
    for (method, provenance, cohort, k), losses in sorted(points.items()):
        series_key = canonical_json([method, provenance, cohort])
        series[series_key].append((k, statistics.mean(losses)))
        series_labels[series_key] = f"{method} / {cohort_labels[cohort]} [{provenance}]"
        cohort_curves[provenance, cohort][method].append((k, statistics.mean(losses)))
    _line_chart(output / "k_loss.svg", "L3 variant budget versus latency loss", _context(row.get("provenance", "unspecified") for row in results), series, "Maximum compiled variants K", "Mean loss versus observed per-shape best (%)", series_labels=series_labels)
    lines = ["## L3 finite variant selection", "", "Loss = expected latency / observed per-shape best reference latency − 1. This reference can use one variant per shape. Curves preserve hardware, operator, workload context, and provenance; repeated results within a cohort are averaged. An incomplete performance matrix provides an observed reference only; it does not certify the unknown global optimum. Uncovered workloads have no full expected latency and no full-workload K-loss point; conditional latency and observed weight coverage are shown separately.", ""]
    lines += _cohort_curve_figures(output, "k_loss", "L3 K-loss", cohort_curves, cohort_examples, "Maximum compiled variants K", "Mean loss versus observed per-shape best (%)")
    lines += ["[Combined overview SVG](k_loss.svg) is retained for compatibility. Use the individual cohort figures for method comparisons.", ""]
    if rows:
        lines += ["<details>", "<summary>All L3 selection scores</summary>", "", _table(["Method", "Provenance", "Hardware / workload", "K", "Selected", "Expected ms", "Conditional ms", "Loss %", "Weight coverage", "Exact solver", "Matrix complete"], rows), "", "</details>", ""]
    else:
        lines += ["**Insufficient data:** no L3 selection results were supplied.", ""]
    return lines


def _inventory_report(records: Sequence[MeasurementRecord], cases: Mapping[tuple[str, ...], list[MeasurementRecord]], output: Path, context: str) -> list[str]:
    counts = Counter((record.provenance, record.status.value) for record in records)
    labels = [f"{provenance}: {status}" for provenance, status in sorted(counts)]
    _bar_chart(output / "failure_rates.svg", "Experiment outcome counts", context, labels, [counts[key] for key in sorted(counts)], xlabel="Recorded attempts (correctness unknown is a separate outcome)")
    lines = ["## Failure rates and measurement coverage", "", "Outcome rates use all supplied unique records, including invalid configurations and failed compilation, correctness, or runtime attempts. Correctness unknown is reported separately and is excluded from verified performance summaries. Slow, correct measurements remain successes.", "", "![Experiment outcomes](failure_rates.svg)", ""]
    if records:
        totals = Counter(record.provenance for record in records)
        lines += [_table(["Provenance", "Outcome", "Attempts", "Rate within provenance"], [(p, s, n, f"{n / totals[p]:.2%}") for (p, s), n in sorted(counts.items())]), ""]
        failure_rows = []
        for provenance, total in sorted(totals.items()):
            failures = sum(n for (p, status), n in counts.items() if p == provenance and status not in {"success", _UNKNOWN})
            unknown = counts.get((provenance, _UNKNOWN), 0)
            known = total - unknown
            failure_rows.append((provenance, failures, f"{failures / total:.2%}", f"{failures / known:.2%}" if known else "N/A", unknown))
        lines += [_table(["Provenance", "Failures", "Failure / all attempts", "Failure / known outcomes", "Correctness unknown"], failure_rows), ""]
    else:
        lines += ["**Insufficient data:** no measurement records were supplied.", ""]
    groups: dict[tuple[str, ...], list[MeasurementRecord]] = defaultdict(list)
    for record in records:
        groups[_cohort(record)].append(record)
    rows, bar_labels, bar_values = [], [], []
    for key, group in sorted(groups.items()):
        shapes = {record.shape.id for record in group}
        configs = {record.config.id for record in group}
        attempts = {(record.shape.id, record.config.id) for record in group}
        successes = {(record.shape.id, record.config.id) for record in group if _verified(record)}
        denominator = len(shapes) * len(configs)
        label = f"{group[0].hardware.name} / {group[0].operator.name} [{key[2]}; {key[3]}; {key[4]}]"
        rows.append([label, len(shapes), len(configs), len(attempts), len(successes), denominator, f"{len(attempts) / denominator:.2%}", f"{len(successes) / denominator:.2%}"])
        bar_labels.append(label)
        bar_values.append(len(successes) / denominator)
    _bar_chart(output / "coverage.svg", "Observed rectangle: verified measurement coverage", context, bar_labels, bar_values, xlabel="Verified cells / (observed shapes × observed configuration union)")
    lines += ["![Observed coverage](coverage.svg)", "", "The denominator below is the rectangle formed by observed shapes and the observed configuration union within each hardware/operator/provenance/split/experiment. It may contain illegal pairs and omit unattempted configurations. **It is not full legal search-space coverage.** Full legal coverage is unavailable without a declared, validated candidate universe.", ""]
    if rows:
        lines += [_table(["Cohort", "Shapes", "Config union", "Attempted cells", "Verified cells", "Rectangle cells", "Attempt coverage", "Verified coverage"], rows), ""]
    manifests = {canonical_json(record.metadata.get("source_manifest", {})): record.metadata.get("source_manifest", {}) for record in records if record.metadata.get("source_manifest")}
    source_rows = []
    for manifest in manifests.values():
        total, sample = manifest.get("source_record_count"), manifest.get("sample_record_count")
        source_rows.append([manifest.get("repository", "unknown"), manifest.get("commit", "unknown"), sample, total, f"{sample / total:.2%}" if _number(sample) and _number(total) and total > 0 else "N/A", manifest.get("source_license", "unspecified")])
    if source_rows:
        lines += ["### Public source file inventory", "", "These are declared source-artifact row counts, not verified feasible-space counts or a substitute for the report's supplied-record count. Full-file retention does not establish correctness; a small excerpt cannot support full-search replay claims.", "", _table(["Repository", "Commit", "Retained artifact rows", "Full source rows", "Artifact retention", "License"], source_rows), ""]
    return lines


def generate_report(
    records: Iterable[MeasurementRecord],
    output_dir: str | Path,
    replay_results: Iterable[Any] = (),
    l3_results: Iterable[Any] = (),
) -> Path:
    """Write Markdown, seven base figures, and readable per-cohort SVG curves.

    Input iterables are consumed once. Exact duplicate records are deduplicated
    by content ID. Replay/L3 inputs accept their native result objects or JSON
    mappings. Missing metrics are explicitly marked instead of fabricated.
    Standard-library only; no plotting, GPU, or network dependency is required.
    When matplotlib is available each SVG also receives a matching PNG.
    """
    unique: dict[str, MeasurementRecord] = {}
    input_count = 0
    for record in records:
        if not isinstance(record, MeasurementRecord):
            raise TypeError("records must contain MeasurementRecord objects")
        unique[record.id] = record
        input_count += 1
    data = sorted(unique.values(), key=lambda record: (record.provenance, record.case_id, record.config.id, record.id))
    replay = [_plain(result) for result in replay_results]
    l3 = [_plain(result) for result in l3_results]
    hardware_names = {record.hardware.id: record.hardware.name for record in data}
    operator_names = {record.operator.id: record.operator.name for record in data}
    for row in [*replay, *l3]:
        if not row.get("hardware_name") and row.get("hardware_id") in hardware_names:
            row["hardware_name"] = hardware_names[row["hardware_id"]]
        if not row.get("operator_name") and row.get("operator_id") in operator_names:
            row["operator_name"] = operator_names[row["operator_id"]]
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    provenance = {record.provenance for record in data} | {str(row.get("provenance", "unspecified")) for row in [*replay, *l3]}
    context = _context(provenance)
    cases, latencies = _grouped(data)
    lines = ["# Kernel Tuning Research Lab experiment report", "", f"**{context}**", ""]
    if "mock" in provenance or "mixed" in provenance:
        lines += ["> MOCK measurements are generated from a synthetic CPU model. They demonstrate framework behavior and must not be cited as real GPU performance, hardware comparisons, or transfer evidence.", ""]
    lines += [f"Unique measurement records: **{len(data)}**; exact duplicate inputs removed: **{input_count - len(data)}**; observed cases: **{len(cases)}**; correctness-verified successful records: **{sum(_verified(record) for record in data)}**.", "", "Provenance, data splits, and experiment identities are preserved. All timing units in this report are milliseconds unless explicitly marked otherwise. Missing values remain N/A. Reports are descriptive analysis and must not be used to train a policy on held-out test results.", ""]
    lines += _config_heatmap(data, cases, latencies, output, context)
    lines += _replay_report(replay, output)
    lines += _transfer_report(cases, latencies, output, context)
    lines += _sensitivity_report(data, latencies, output, context)
    lines += _l3_report(l3, output)
    lines += _inventory_report(data, cases, output, context)
    png_count = len(list(output.glob("*.png")))
    svg_count = len(list(output.glob("*.svg")))
    lines += ["## Audit artifacts", "", f"SVG charts present: {svg_count}, including individual cohort figures and seven base/compatibility figures. Optional matplotlib PNG charts present: {png_count}. `measurement_inventory.json` records IDs, statuses, provenance, source URLs, and source verification claims. `replay_results.json` and `l3_results.json` retain the supplied result objects, including traces, coverage, costs, and solver metadata, with readable identity names filled from measurement records when available. Original raw measurement exports remain the source of truth.", ""]
    inventory = [{"id": record.id, "case_id": record.case_id, "hardware": record.hardware.name, "operator": record.operator.name, "shape": record.shape.dims, "config_id": record.config.id, "status": record.status.value, "provenance": record.provenance, "split": record.split, "experiment_id": record.experiment_id, "measurement_mode": record.measurement_mode, "correctness_verified": record.correctness_verified, "source_url": record.metadata.get("source_url"), "failure_reason": record.failure_reason} for record in data]
    for filename, payload in [("measurement_inventory.json", inventory), ("replay_results.json", replay), ("l3_results.json", l3)]:
        (output / filename).write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    report_path = output / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
