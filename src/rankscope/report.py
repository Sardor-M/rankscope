"""Text and Markdown rendering of reports, calibrations, verdicts and comparisons."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from .metrics import ALL


def fmt(value: float, digits: int = 4) -> str:
    if isinstance(value, float) and math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.{digits}f}"


def fmt_p(prop: Mapping) -> str:
    return f"{prop['k']}/{prop['n']} = {prop['p']:.0%} (95% CI {prop['lo']:.0%}-{prop['hi']:.0%})"


def table(headers: Sequence[str], rows: Sequence[Sequence[str]], markdown: bool) -> list[str]:
    if markdown:
        return ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|",
                *("| " + " | ".join(row) + " |" for row in rows)]
    widths = [max(len(str(cell)) for cell in column) for column in zip(headers, *rows)]
    lines = ["  " + "  ".join(f"{h:<{w}}" for h, w in zip(headers, widths))]
    for row in rows:
        lines.append("  " + "  ".join(f"{cell:<{w}}" for cell, w in zip(row, widths)))
    return lines


def _heading(text: str, markdown: bool) -> list[str]:
    return ["", f"### {text}", ""] if markdown else ["", text]


def metrics_table(summary: Mapping[str, Mapping], ks: Sequence[int], level: str, ci_k: int, markdown: bool) -> list[str]:
    headers = ["stratum", "n"] + [f"hit@{k}" for k in ks] + ["MRR", f"hit@{ci_k} 95% CI"]
    if level == "doc":
        headers += [f"recall@{max(ks)}", f"ndcg@{10 if 10 in ks else max(ks)}"]
    rows = []
    for stratum in sorted(summary, key=lambda s: (s != ALL, s)):
        row = summary[stratum]
        lo, hi = row.get(f"hit@{ci_k}_ci", (0.0, 0.0))
        cells = [stratum, str(row["n"])] + [f"{row[f'hit@{k}']:.2f}" for k in ks]
        cells += [f"{row['mrr']:.3f}", f"{lo:.2f}-{hi:.2f}"]
        if level == "doc":
            cells += [f"{row[f'recall@{max(ks)}']:.2f}", f"{row[f'ndcg@{10 if 10 in ks else max(ks)}']:.3f}"]
        rows.append(cells)
    return table(headers, rows, markdown)


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


def render(report: Mapping, *, ci_k: int | None = None, markdown: bool = False) -> str:
    meta = report["meta"]
    n = meta["n"]
    ks = meta["ks"]
    ci_k = ci_k or max(ks)
    lines = [
        f"queries: {n['positives']} positives ({n['indexed']} indexed) · {n['negatives']} negatives"
        + (f" · {n['unjudged_in_lanes']} unjudged in lanes (ignored)" if n["unjudged_in_lanes"] else "")
        + f" · lanes: {', '.join(meta['lanes'])} · rankings: {', '.join(meta['rankings'])} · window={meta['window']}"
    ]
    if meta.get("weights"):
        lines.append("convex weights: " + ", ".join(f"{k}={v:.2f}" for k, v in meta["weights"].items()))
    for name in meta["rankings"]:
        if name not in report["metrics"]:
            continue
        for level in meta["levels"]:
            lines += _heading(f"{name} · {level} level", markdown)
            lines += metrics_table(report["metrics"][name][level], ks, level, ci_k, markdown)

    attribution = report["attribution"]
    if attribution["counts"]:
        lines += _heading(f"miss attribution (doc level, window={meta['window']})", markdown)
        rows = [[key, str(count)] for key, count in sorted(attribution["counts"].items(), key=lambda kv: -kv[1])]
        lines += table(["cause", "queries"], rows, markdown)
        lines.append("")
        lines.append(f"{'- ' if markdown else '  '}indexed but not surfaced by any lane: {fmt_p(attribution['not_surfaced_share'])}")
        for name, prop in attribution["not_surfaced_by"].items():
            lines.append(f"{'- ' if markdown else '  '}not surfaced by {name}: {fmt_p(prop)}")
        for name, prop in attribution["blind_share_of_misses"].items():
            lines.append(f"{'- ' if markdown else '  '}{meta['blind']} share of {name} misses: {fmt_p(prop)}")

    if n["negatives"]:
        lines += _heading(f"negatives ({n['negatives']}): candidates handed to the gate", markdown)
        lines += table(["ranking", "handed"], [[name, fmt_p(prop)] for name, prop in report["negatives"].items()], markdown)

    if report.get("gates"):
        lines += _heading(f"gates on {meta['gate_on']} ({meta['gate_level']} level; accept the top hit if score >= floor and top1 - top2 >= margin)", markdown)
        rows = []
        for name, gate in report["gates"].items():
            counts = gate["counts"]
            rows.append([name, fmt(gate["floor"]), fmt(gate["margin"], 3), str(counts["accept_correct"]),
                         str(counts["accept_wrong"]), str(counts["abstain"]), fmt_p(gate["far"]), fmt_p(gate["fnir"])])
        lines += table(["gate", "floor", "margin", "correct", "wrong", "abstain", "FAR (negatives accepted)", "FNIR (positives missed)"], rows, markdown)
    if report.get("sweep"):
        rows_all = report["sweep"]
        step = max(1, len(rows_all) // 24)
        shown = rows_all[::step] if rows_all[-1] in rows_all[::step] else rows_all[::step] + [rows_all[-1]]
        lines += _heading(f"sweep, {len(shown)} of {len(rows_all)} floors", markdown)
        lines += table(["floor", "FPIR", "FNIR"], [[fmt(r["floor"]), f"{r['fpir']['p']:.2f}", f"{r['fnir']['p']:.2f}"] for r in shown], markdown)
    return "\n".join(lines)


def render_calibration(cal: Mapping) -> str:
    return "\n".join([
        f"calibration on {cal['n']['positives']} positives · {cal['n']['negatives']} negatives · gate on {cal['gate_on']} ({cal['gate_level']} level)",
        "fusion weights (" + cal["fit"]["objective"] + "): " + ", ".join(f"{k}={v:.2f}" for k, v in cal["weights"].items())
        + f" -> {cal['fit']['convex']:.4f}   (rrf baseline {cal['fit']['rrf']:.4f})",
        f"null floor: n={cal['null']['n']} mean={fmt(cal['null']['mean'])} std={fmt(cal['null']['std'])} -> floor {fmt(cal['null']['floor'])}",
        f"conformal floor (alpha={cal['conformal']['alpha']:g}): n={cal['conformal']['n']} (min n for a finite bound: {cal['conformal']['min_n']}) -> floor {fmt(cal['conformal']['floor'])}",
        f"operating floor at FPIR <= {cal['operating']['fpir_max']:g}: {fmt(cal['operating']['floor'])} -> FPIR {cal['operating']['fpir']['p']:.2f}, FNIR {cal['operating']['fnir']['p']:.2f} (on the calibration queries)",
    ])


def render_verdict(rows: Sequence[Mapping], summary: str, markdown: bool = False) -> str:
    body = []
    for row in rows:
        ci = f"{row['lo']:.2f}-{row['hi']:.2f}" if row["lo"] is not None else "-"
        body.append([row["name"], f"{row['op']} {row['threshold']:.2f}", f"{row['value']:.2f}", ci,
                     "pass" if row["point"] else "FAIL", "pass" if row["interval"] else "open"])
    lines = table(["gate", "threshold", "measured", "95% CI", "point", "interval"], body, markdown)
    return "\n".join(lines + ["", f"verdict: {summary}"])


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
