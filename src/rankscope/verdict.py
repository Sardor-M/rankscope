"""Gate rules written before the run, judged on the point estimate and on the interval."""

from __future__ import annotations

from typing import Mapping, Sequence

OPS = (">=", "<=")


def resolve(report: Mapping, path: str) -> tuple[float, float | None, float | None]:
    """`(value, lo, hi)` at a dotted path; keys may themselves contain dots (`fpir-0.05`)."""
    parts = path.split(".")
    node: object = report
    parent: Mapping = report
    key = ""
    index = 0
    while index < len(parts):
        if not isinstance(node, Mapping):
            raise KeyError(f"no {path!r} in the report (stopped at {key!r})")
        stop = next((j for j in range(len(parts), index, -1) if ".".join(parts[index:j]) in node), None)
        if stop is None:
            raise KeyError(f"no {path!r} in the report (stopped at {parts[index]!r})")
        key, index = ".".join(parts[index:stop]), stop
        parent, node = node, node[key]
    if isinstance(node, Mapping) and "p" in node:
        return float(node["p"]), float(node["lo"]), float(node["hi"])
    if isinstance(node, (int, float)) and not isinstance(node, bool):
        ci = parent.get(f"{key}_ci") if isinstance(parent, Mapping) else None
        if ci:
            return float(node), float(ci[0]), float(ci[1])
        return float(node), None, None
    raise TypeError(f"{path!r} is not a number: {node!r}")


def judge(report: Mapping, rules: Sequence[Mapping]) -> list[dict]:
    rows = []
    for rule in rules:
        op = rule["op"]
        if op not in OPS:
            raise ValueError(f"op must be one of {OPS}, not {op!r}")
        threshold = float(rule["threshold"])
        value, lo, hi = resolve(report, rule["path"])
        if op == ">=":
            point, interval = value >= threshold, lo is not None and lo >= threshold
        else:
            point, interval = value <= threshold, hi is not None and hi <= threshold
        rows.append({"name": rule["name"], "path": rule["path"], "op": op, "threshold": threshold,
                     "value": value, "lo": lo, "hi": hi, "point": point, "interval": interval})
    return rows


def overall(rows: Sequence[Mapping]) -> str:
    if all(row["interval"] for row in rows):
        return "PASS (every gate clears on its interval)"
    if all(row["point"] for row in rows):
        return "PASS ON POINT ESTIMATES ONLY (some intervals straddle a threshold; more queries first)"
    return "FAIL (" + ", ".join(row["name"] for row in rows if not row["point"]) + ")"
