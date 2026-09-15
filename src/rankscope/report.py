"""Text and Markdown rendering of reports, calibrations, verdicts and comparisons."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from .metrics import ALL
from .stats import negatives_for


def fmt_num(value: float, digits: int = 4) -> str:
    if isinstance(value, float) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.{digits}f}"


def fmt_p(prop: Mapping) -> str:
    return f"{prop['k']}/{prop['n']} = {prop['p']:.0%} (95% CI {prop['lo']:.0%}-{prop['hi']:.0%})"


LATEX_ESCAPES = {"&": "\\&", "%": "\\%", "#": "\\#", "_": "\\_", "{": "\\{", "}": "\\}"}


def latex_escape(text: str) -> str:
    return "".join(LATEX_ESCAPES.get(c, c) for c in str(text))


def table(headers: Sequence[str], rows: Sequence[Sequence[str]], fmt: bool | str = "text") -> list[str]:
    """Rows as aligned text, a Markdown table, or a LaTeX tabular. `fmt` may be True for Markdown."""
    if fmt is True or fmt == "markdown":
        return ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|",
                *("| " + " | ".join(row) + " |" for row in rows)]
    if fmt == "latex":
        spec = "l" + " r" * (len(headers) - 1)
        body = [" & ".join(latex_escape(cell) for cell in row) + " \\\\" for row in rows]
        return [f"\\begin{{tabular}}{{{spec}}}", "\\hline",
                " & ".join(latex_escape(h) for h in headers) + " \\\\", "\\hline", *body, "\\hline", "\\end{tabular}"]
    widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *rows)]
    lines = ["  " + "  ".join(f"{h:<{w}}" for h, w in zip(headers, widths))]
    for row in rows:
        lines.append("  " + "  ".join(f"{cell:<{w}}" for cell, w in zip(row, widths)))
    return lines


def _heading(text: str, fmt: bool | str) -> list[str]:
    if fmt is True or fmt == "markdown":
        return ["", f"### {text}", ""]
    return ["", text]


def with_interval(row: Mapping, key: str, digits: int) -> str:
    """`0.301 [0.21-0.39]` when a bootstrap interval was recorded, else the value alone."""
    text = f"{row[key]:.{digits}f}"
    ci = row.get(f"{key}_ci")
    return f"{text} [{ci[0]:.{digits}f}-{ci[1]:.{digits}f}]" if ci else text


def metrics_table(summary: Mapping[str, Mapping], ks: Sequence[int], level: str, ci_k: int, fmt: bool | str) -> list[str]:
    headers = ["stratum", "n"] + [f"hit@{k}" for k in ks] + ["MRR", f"hit@{ci_k} 95% CI"]
    if level == "doc":
        headers += [f"recall@{max(ks)}", f"ndcg@{10 if 10 in ks else max(ks)}"]
    rows = []
    for stratum in sorted(summary, key=lambda s: (s != ALL, s)):
        row = summary[stratum]
        lo, hi = row.get(f"hit@{ci_k}_ci", (0.0, 0.0))
        cells = [stratum, str(row["n"])] + [f"{row[f'hit@{k}']:.2f}" for k in ks]
        cells += [with_interval(row, "mrr", 3), f"{lo:.2f}-{hi:.2f}"]
        if level == "doc":
            cells += [with_interval(row, f"recall@{max(ks)}", 2), with_interval(row, f"ndcg@{10 if 10 in ks else max(ks)}", 3)]
        rows.append(cells)
    return table(headers, rows, fmt)


def query_lines(report: Mapping) -> list[str]:
    names = report["meta"]["rankings"]
    lines = []
    for record in report["queries"]:
        if record["negative"]:
            counts = " · ".join(f"{name} {record['rankings'][name]['candidates']} cands" for name in names)
            line = f"{record['query']:<28} negative    {counts}"
        else:
            ranks = " · ".join(f"{name} {record['rankings'][name]['rank'] or 'MISS'}" for name in names)
            line = f"{record['query']:<28} {record['stratum']:<11} {ranks} · {record['attribution']}"
        for name, gate in report.get("gates", {}).items():
            line += f" · {name}={gate['queries'][record['query']]['outcome']}"
        lines.append(line)
    return lines


def render(report: Mapping, *, ci_k: int | None = None, markdown: bool = False, fmt: str | None = None) -> str:
    fmt = fmt or ("markdown" if markdown else "text")
    markdown = fmt == "markdown"
    meta = report["meta"]
    n = meta["n"]
    ks = meta["ks"]
    ci_k = ci_k or max(ks)
    lines = [
        f"queries: {n['positives']} positives ({n['indexed']} indexed) · {n['negatives']} negatives"
        + (f" · {n['unjudged_in_lanes']} unjudged in lanes (ignored)" if n["unjudged_in_lanes"] else "")
        + f" · lanes: {', '.join(meta['lanes'])} · rankings: {', '.join(meta['rankings'])} · window={meta['window']}"
    ]
    depths = meta.get("depths", {})
    if depths:
        line = "depth: " + " · ".join(f"{name} {depth}" for name, depth in depths.items())
        if len(set(depths.values())) > 1:
            line += "   (lanes differ in depth; fusion treats a document below a lane's depth as absent from it)"
        lines.append(line)
    missing = {name: count for name, count in meta.get("missing_from_lane", {}).items() if count}
    if missing:
        lines.append("judged queries missing from a lane, scored as misses: "
                     + " · ".join(f"{name} {count}" for name, count in missing.items()))
    if meta.get("weights"):
        lines.append("convex weights: " + ", ".join(f"{k}={v:.2f}" for k, v in meta["weights"].items()))
    for name in meta["rankings"]:
        if name not in report["metrics"]:
            continue
        for level in meta["levels"]:
            lines += _heading(f"{name} · {level} level", fmt)
            lines += metrics_table(report["metrics"][name][level], ks, level, ci_k, fmt)

    attribution = report["attribution"]
    if attribution["counts"]:
        lines += _heading(f"miss attribution (doc level, window={meta['window']})", fmt)
        rows = [[key, str(count)] for key, count in sorted(attribution["counts"].items(), key=lambda kv: -kv[1])]
        lines += table(["cause", "queries"], rows, fmt)
        lines.append("")
        lines.append(f"{'- ' if markdown else '  '}indexed but not surfaced by any lane: {fmt_p(attribution['not_surfaced_share'])}")
        for name, prop in attribution["not_surfaced_by"].items():
            lines.append(f"{'- ' if markdown else '  '}not surfaced by {name}: {fmt_p(prop)}")
        for name, prop in attribution["blind_share_of_misses"].items():
            lines.append(f"{'- ' if markdown else '  '}{meta['blind']} share of {name} misses: {fmt_p(prop)}")

    if n["negatives"]:
        lines += _heading(f"negatives ({n['negatives']}): candidates handed to the gate", fmt)
        lines += table(["ranking", "handed"], [[name, fmt_p(prop)] for name, prop in report["negatives"].items()], fmt)

    if report.get("gates"):
        lines += _heading(f"gates on {meta['gate_on']} ({meta['gate_level']} level; accept the top hit if score >= floor and top1 - top2 >= margin)", fmt)
        rows = []
        for name, gate in report["gates"].items():
            counts = gate["counts"]
            rows.append([name, fmt_num(gate["floor"]), fmt_num(gate["margin"], 3), str(counts["accept_correct"]),
                         str(counts["accept_wrong"]), str(counts["abstain"]), fmt_p(gate["far"]), fmt_p(gate["fnir"])])
        lines += table(["gate", "floor", "margin", "correct", "wrong", "abstain", "FAR (negatives accepted)", "FNIR (positives missed)"], rows, fmt)
    if report.get("sweep"):
        rows_all = report["sweep"]
        step = max(1, len(rows_all) // 24)
        shown = rows_all[::step] if rows_all[-1] in rows_all[::step] else rows_all[::step] + [rows_all[-1]]
        lines += _heading(f"sweep, {len(shown)} of {len(rows_all)} floors", fmt)
        lines += table(["floor", "FPIR", "FNIR"], [[fmt_num(r["floor"]), f"{r['fpir']['p']:.2f}", f"{r['fnir']['p']:.2f}"] for r in shown], fmt)
    if fmt == "latex":
        lines = [line if line.startswith(("\\begin", "\\end", "\\hline")) or line.endswith("\\\\") or not line.strip()
                 else "% " + line for line in lines]
    return "\n".join(lines)


def render_calibration(cal: Mapping) -> str:
    operating = cal["operating"]
    rule = " on the Wilson upper bound" if operating.get("rule") == "wilson-upper" else ""
    lines = [
        f"calibration on {cal['n']['positives']} positives · {cal['n']['negatives']} negatives · gate on {cal['gate_on']} ({cal['gate_level']} level)",
        "fusion weights (" + cal["fit"]["objective"] + "): " + ", ".join(f"{k}={v:.2f}" for k, v in cal["weights"].items())
        + f" -> {cal['fit']['convex']:.4f}   (rrf baseline {cal['fit']['rrf']:.4f})",
        f"null floor: n={cal['null']['n']} mean={fmt_num(cal['null']['mean'])} std={fmt_num(cal['null']['std'])} -> floor {fmt_num(cal['null']['floor'])}",
        f"conformal floor (alpha={cal['conformal']['alpha']:g}): n={cal['conformal']['n']} (min n for a finite bound: {cal['conformal']['min_n']}) -> floor {fmt_num(cal['conformal']['floor'])}",
        f"operating floor at FPIR <= {operating['fpir_max']:g}{rule}: {fmt_num(operating['floor'])} -> FPIR {operating['fpir']['p']:.2f} "
        f"(95% up to {operating['fpir']['hi']:.2f}), FNIR {operating['fnir']['p']:.2f} (on the calibration queries)",
    ]
    if math.isinf(operating["floor"]) and operating.get("rule") == "wilson-upper":
        lines.append(f"no finite floor bounds FPIR at {operating['fpir_max']:g} on its upper limit with {cal['n']['negatives']} negatives; "
                     f"zero false accepts in at least {negatives_for(operating['fpir_max'])} would")
    missing = {k: v for k, v in cal["n"].get("missing_from_lane", {}).items() if v}
    if missing:
        lines.append("judged queries missing from a lane, scored as misses: " + " · ".join(f"{k} {v}" for k, v in missing.items()))
    return "\n".join(lines)


def render_verdict(rows: Sequence[Mapping], summary: str, markdown: bool = False) -> str:
    body = []
    for row in rows:
        ci = f"{row['lo']:.2f}-{row['hi']:.2f}" if row["lo"] is not None else "-"
        body.append([row["name"], f"{row['op']} {row['threshold']:.2f}", f"{row['value']:.2f}", ci,
                     "pass" if row["point"] else "FAIL", "pass" if row["interval"] else "open"])
    lines = table(["gate", "threshold", "measured", "95% CI", "point", "interval"], body, markdown)
    return "\n".join(lines + ["", f"verdict: {summary}"])


def render_diff(rows: Sequence[Mapping], counts: Mapping[str, int], markdown: bool = False) -> str:
    body = []
    for row in rows:
        arrow = "lower is better" if row["lower_is_better"] else ""
        body.append([row["where"], row["metric"], f"{row['before']:.3f}", f"{row['after']:.3f}",
                     f"{row['delta']:+.3f}", row["verdict"] + (f" ({arrow})" if arrow and row["verdict"] != "no interval" else "")])
    lines = table(["where", "metric", "before", "after", "delta", "verdict"], body, markdown)
    lines.append("")
    lines.append("summary: " + " · ".join(f"{count} {name}" for name, count in counts.items()))
    return "\n".join(lines)


def render_compare(result: Mapping, metric: str, baseline: str, candidate: str, min_delta: float) -> str:
    better = result["p_better"] >= 0.95 and result["delta"] >= min_delta
    return "\n".join([
        f"{metric}: {baseline} {result['mean_baseline']:.4f} -> {candidate} {result['mean_candidate']:.4f} "
        f"over {result['n']} shared positives",
        f"delta {result['delta']:+.4f}  95% CI [{result['ci_low']:+.4f}, {result['ci_high']:+.4f}]  "
        f"P(candidate > baseline) = {result['p_better']:.3f}  ({result['resamples']} paired resamples)",
        "verdict: " + ("BETTER (P >= 0.95 and delta >= min effect)" if better else
                       f"NOT SHOWN BETTER (needs P >= 0.95 and delta >= {min_delta:+.4f})"),
    ])


PER_QUERY_FIELDS = ("query", "stratum", "negative", "indexed", "attribution")


def write_per_query_csv(report: Mapping, path) -> int:
    """One row per judged query: strata, attribution, each ranking's ranks and each gate's outcome."""
    import csv
    from pathlib import Path

    names = report["meta"]["rankings"]
    has_group = "group" in report["meta"]["levels"]
    gates = list(report.get("gates", {}))
    header = list(PER_QUERY_FIELDS)
    for name in names:
        header += [f"{name}_rank"] + ([f"{name}_group_rank"] if has_group else []) + [f"{name}_candidates"]
    header += [f"gate_{gate}" for gate in gates]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for record in report["queries"]:
            row = [record["query"], record["stratum"], int(record["negative"]),
                   "" if record["negative"] else int(record.get("indexed", True)), record.get("attribution", "")]
            for name in names:
                entry = record["rankings"][name]
                row += [entry.get("rank", "")] + ([entry.get("group_rank", "")] if has_group else []) + [entry["candidates"]]
            row += [report["gates"][gate]["queries"][record["query"]]["outcome"] for gate in gates]
            writer.writerow(row)
    return len(report["queries"])
