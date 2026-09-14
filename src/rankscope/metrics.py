"""Hit rate, recall, MRR, nDCG, per-stratum summaries with intervals, and miss attribution."""

from __future__ import annotations

from math import log2
from typing import Mapping, Sequence

from .stats import bootstrap_mean, wilson

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


def headline_metrics(ks: Sequence[int]) -> list[str]:
    """The mean metrics that carry a bootstrap interval by default: MRR, recall at the deepest k, nDCG at 10."""
    return ["mrr", f"recall@{max(ks)}", f"ndcg@{10 if 10 in ks else max(ks)}"]


def summarise(
    rows: Sequence[tuple[str, Mapping[str, float]]],
    ks: Sequence[int] = DEFAULT_KS,
    *,
    resamples: int = 0,
    seed: int = 0,
    ci_metrics: Sequence[str] | None = None,
) -> dict:
    """Per-stratum hit@k with Wilson intervals, MRR and the other means (bootstrap intervals when asked); plus `all`."""
    by_stratum: dict[str, list[Mapping[str, float]]] = {ALL: []}
    for stratum, metrics in rows:
        by_stratum.setdefault(stratum, []).append(metrics)
        by_stratum[ALL].append(metrics)
    wanted = list(ci_metrics) if ci_metrics is not None else headline_metrics(ks)
    out: dict[str, dict] = {}
    for stratum, items in by_stratum.items():
        ranks = [int(m["rank"]) for m in items]
        entry: dict = {"n": len(ranks), "mrr": mrr(ranks)}
        for k in ks:
            entry[f"hit@{k}"] = hit_rate(ranks, k)
            entry[f"hit@{k}_ci"] = list(wilson(hits_at(ranks, k), len(ranks)))
        extra = sorted({key for m in items for key in m if key != "rank"})
        per_query: dict[str, list[float]] = {"mrr": [1.0 / r if r else 0.0 for r in ranks]}
        for key in extra:
            values = [float(m[key]) for m in items if key in m]
            per_query[key] = values
            entry[key] = sum(values) / len(values) if values else 0.0
        if resamples > 0 and len(items) > 1:
            for key in wanted:
                if key in per_query:
                    entry[f"{key}_ci"] = list(bootstrap_mean(per_query[key], resamples=resamples, seed=seed))
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
