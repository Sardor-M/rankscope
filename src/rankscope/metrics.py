"""Hit rate, recall, MRR, nDCG, per-stratum summaries with intervals, and miss attribution."""

from __future__ import annotations

from math import log2
from typing import Mapping, Sequence

from .stats import wilson

ALL = "all"
DEFAULT_KS = (1, 5, 10, 20)


def hits_at(ranks: Sequence[int], k: int) -> int:
    return sum(1 for r in ranks if 0 < r <= k)


def hit_rate(ranks: Sequence[int], k: int) -> float:
    return hits_at(ranks, k) / len(ranks) if ranks else 0.0


def mrr(ranks: Sequence[int]) -> float:
    return sum(1.0 / r for r in ranks if r) / len(ranks) if ranks else 0.0


def recall_at(docs: Sequence[str], relevant: Sequence[str] | frozenset[str], k: int) -> float:
    """Fraction of the relevant documents that appear in the top k."""
    wanted = set(relevant)
    if not wanted:
        return 0.0
    return len(wanted.intersection(docs[:k])) / len(wanted)


def ndcg_at(docs: Sequence[str], grades: Mapping[str, int], k: int) -> float:
    """Graded nDCG with gain 2^grade - 1 and discount 1 / log2(rank + 1)."""
    gains = [2 ** grades.get(doc, 0) - 1 for doc in docs[:k]]
    dcg = sum(gain / log2(rank + 1) for rank, gain in enumerate(gains, 1) if gain)
    ideal = sorted((g for g in grades.values() if g > 0), reverse=True)[:k]
    idcg = sum((2 ** grade - 1) / log2(rank + 1) for rank, grade in enumerate(ideal, 1))
    return dcg / idcg if idcg else 0.0


def proportion(successes: int, n: int) -> dict:
    lo, hi = wilson(successes, n)
    return {"p": successes / n if n else 0.0, "k": successes, "n": n, "lo": lo, "hi": hi}


def summarise(rows: Sequence[tuple[str, Mapping[str, float]]], ks: Sequence[int] = DEFAULT_KS) -> dict:
    """Per-stratum hit@k with Wilson intervals, MRR, and the mean of every other metric; plus `all`."""
    by_stratum: dict[str, list[Mapping[str, float]]] = {ALL: []}
    for stratum, metrics in rows:
        by_stratum.setdefault(stratum, []).append(metrics)
        by_stratum[ALL].append(metrics)
    out: dict[str, dict] = {}
    for stratum, items in by_stratum.items():
        ranks = [int(m["rank"]) for m in items]
        entry: dict = {"n": len(ranks), "mrr": mrr(ranks)}
        for k in ks:
            entry[f"hit@{k}"] = hit_rate(ranks, k)
            entry[f"hit@{k}_ci"] = list(wilson(hits_at(ranks, k), len(ranks)))
        extra = sorted({key for m in items for key in m if key != "rank"})
        for key in extra:
            values = [float(m[key]) for m in items if key in m]
            entry[key] = sum(values) / len(values) if values else 0.0
        out[stratum] = entry
    return out


def miss_attribution(*, indexed: bool, ranks: Mapping[str, int], window: int) -> str:
    """`not_indexed` is coverage, `not_surfaced` is retrieval, otherwise the lanes that carried it."""
    if not indexed:
        return "not_indexed"
    carried = sorted(lane for lane, rank in ranks.items() if 0 < rank <= window)
    if not carried:
        return "not_surfaced"
    return "surfaced_by_" + "+".join(carried)
