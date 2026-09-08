"""Reference vectors so another implementation of the arithmetic can prove parity with this one."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable

from . import fusion, metrics, stats
from .lanes import Hit
from .ranks import rank_of

HEADER = "rankscope-goldens 1"
TOLERANCE = 1e-9


def _num(value: float) -> str:
    if math.isinf(value):
        return "inf" if value > 0 else "-inf"
    return f"{value:.12g}"


def _list(values: Iterable[float]) -> str:
    text = ",".join(_num(float(v)) for v in values)
    return text or "-"


def _parse_num(text: str) -> float:
    return math.inf if text == "inf" else -math.inf if text == "-inf" else float(text)


def _parse_list(text: str) -> list[float]:
    return [] if text == "-" else [_parse_num(part) for part in text.split(",")]


WILSON = [(45, 50), (0, 30), (0, 3), (8, 12), (3, 12), (12, 12), (0, 0), (1, 1000)]
NULL = [[0.5, 0.6, 0.7], [0.85, 0.79, 0.83], [], [0.33], [0.1, 0.9, 0.5, 0.5]]
CONFORMAL = [(0.1, [i / 30 for i in range(1, 31)]), (0.5, [i / 30 for i in range(1, 31)]), (0.1, [0.2] * 8),
             (0.1, [0.2] * 9), (0.1, []), (0.05, [0.4, 0.1, 0.3, 0.2] * 5)]
RANKS = [[1, 3, 0, 6], [], [0, 0, 0], [1, 1, 2, 20, 21]]
RECALL = [(5, ["a", "b", "c"], ["a", "x", "b", "y", "z", "c"]), (2, ["a"], ["x", "y", "a"]), (3, [], ["x"])]
NDCG = [(10, {"a": 2, "b": 1}, ["x", "a", "b"]), (3, {"a": 3}, ["a", "x", "y"]), (5, {"a": 1}, ["x", "y"]),
        (5, {"a": 1, "b": 2, "c": 3}, ["c", "b", "a"])]
RRF = [(60, [["a", "b", "c"], ["c", "d"]]), (60, [["b"], ["a"]]), (1, [["x", "y"], ["y", "x"], ["z"]])]
HITS = [("G1", "a"), ("G2", "b"), ("G1", "c"), ("G3", "d")]
RANK = [("doc", ["c"]), ("group", ["c"]), ("doc", ["zz"]), ("group", ["zz"]), ("doc", ["d", "b"])]


def _group_of(doc: str) -> str:
    return dict((identifier, group) for group, identifier in HITS).get(doc, doc)


def compute(kind: str, args: list[str]) -> str:
    """The reference answer for one line, rendered exactly as `build` writes it."""
    if kind == "wilson":
        lo, hi = stats.wilson(int(args[0]), int(args[1]))
        return f"{_num(lo)} {_num(hi)}"
    if kind == "rule_of_three":
        return _num(stats.rule_of_three(int(args[0])))
    if kind == "negatives_for":
        return str(stats.negatives_for(float(args[0])))
    if kind == "null_floor":
        out = stats.null_floor(_parse_list(args[1]), float(args[0]))
        return f"{out['n']} {_num(out['mean'])} {_num(out['std'])} {_num(out['floor'])}"
    if kind == "conformal":
        return _num(stats.conformal_quantile(_parse_list(args[1]), float(args[0])))
    if kind == "hit":
        return _num(metrics.hit_rate([int(r) for r in _parse_list(args[1])], int(args[0])))
    if kind == "mrr":
        return _num(metrics.mrr([int(r) for r in _parse_list(args[0])]))
    if kind == "recall":
        relevant, ranking = args[1].split("|")
        return _num(metrics.recall_at(ranking.split(","), [] if relevant == "-" else relevant.split(","), int(args[0])))
    if kind == "ndcg":
        grades_text, ranking = args[1].split("|")
        grades = {doc: int(grade) for doc, grade in (pair.split(":") for pair in grades_text.split(","))}
        return _num(metrics.ndcg_at(ranking.split(","), grades, int(args[0])))
    if kind == "rrf":
        lanes = {f"L{i}": [Hit(doc) for doc in lane.split(",")] for i, lane in enumerate(args[1].split(";"))}
        return ",".join(f"{hit.doc}:{_num(hit.score)}" for hit in fusion.rrf(lanes, int(args[0])))
    if kind == "rank":
        hits = [Hit(pair.split(":")[1]) for pair in args[1].split(",")]
        return str(rank_of(hits, args[2].split("+"), level=args[0], group_of=_group_of))
    raise ValueError(f"unknown golden kind {kind!r}")


def build() -> str:
    lines = [HEADER]
    lines += [f"wilson {s} {n} -> {compute('wilson', [str(s), str(n)])}" for s, n in WILSON]
    lines += [f"rule_of_three {n} -> {compute('rule_of_three', [str(n)])}" for n in (3, 30, 0)]
    lines += [f"negatives_for {b} -> {compute('negatives_for', [str(b)])}" for b in (0.1, 0.05, 0.01)]
    lines += [f"null_floor 3 {_list(v)} -> {compute('null_floor', ['3', _list(v)])}" for v in NULL]
    lines += [f"conformal {a:g} {_list(v)} -> {compute('conformal', [str(a), _list(v)])}" for a, v in CONFORMAL]
    for ranks in RANKS:
        lines += [f"hit {k} {_list(ranks)} -> {compute('hit', [str(k), _list(ranks)])}" for k in (1, 5, 20)]
        lines.append(f"mrr {_list(ranks)} -> {compute('mrr', [_list(ranks)])}")
    for k, relevant, ranking in RECALL:
        spec = f"{','.join(relevant) or '-'}|{','.join(ranking)}"
        lines.append(f"recall {k} {spec} -> {compute('recall', [str(k), spec])}")
    for k, grades, ranking in NDCG:
        spec = f"{','.join(f'{d}:{g}' for d, g in grades.items())}|{','.join(ranking)}"
        lines.append(f"ndcg {k} {spec} -> {compute('ndcg', [str(k), spec])}")
    for k, lanes in RRF:
        spec = ";".join(",".join(lane) for lane in lanes)
        lines.append(f"rrf {k} {spec} -> {compute('rrf', [str(k), spec])}")
    hits_text = ",".join(f"{group}:{doc}" for group, doc in HITS)
    for level, relevant in RANK:
        lines.append(f"rank {level} {hits_text} {'+'.join(relevant)} -> {compute('rank', [level, hits_text, '+'.join(relevant)])}")
    return "\n".join(lines) + "\n"


def _close(expected: str, actual: str) -> bool:
    if expected == actual:
        return True
    try:
        left = [_parse_num(part.split(":")[-1]) for part in expected.replace(" ", ",").split(",")]
        right = [_parse_num(part.split(":")[-1]) for part in actual.replace(" ", ",").split(",")]
    except ValueError:
        return False
    if len(left) != len(right):
        return False
    labels_left = [part.split(":")[0] for part in expected.replace(" ", ",").split(",") if ":" in part]
    labels_right = [part.split(":")[0] for part in actual.replace(" ", ",").split(",") if ":" in part]
    if labels_left != labels_right:
        return False
    return all((a == b) if (math.isinf(a) or math.isinf(b)) else abs(a - b) <= TOLERANCE for a, b in zip(left, right))


def check(text: str) -> list[str]:
    """Recompute every line with this implementation; return the lines whose answer disagrees."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != HEADER:
        return [f"expected the header {HEADER!r}"]
    problems = []
    for line in lines[1:]:
        if not line.strip() or " -> " not in line:
            continue
        left, expected = line.split(" -> ", 1)
        kind, *args = left.split()
        try:
            actual = compute(kind, args)
        except Exception as exc:
            problems.append(f"{line}: {exc}")
            continue
        if not _close(expected.strip(), actual):
            problems.append(f"{line}\n    this implementation: {actual}")
    return problems


def write(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(build(), encoding="utf-8")
    return path
