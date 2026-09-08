"""One evaluation: metrics per lane and per fusion, attribution, negatives, gates."""

from __future__ import annotations

from collections import Counter
from typing import Mapping, Sequence

from .fusion import RRF_K, convex, rrf
from .gate import Gate, evaluate as evaluate_gate, sweep
from .lanes import Hit, Lane, slice_lanes
from .metrics import DEFAULT_KS, miss_attribution, ndcg_at, proportion, recall_at, summarise
from .ranks import GroupOf, Judgment, Qrels, indexed, rank_of

RRF = "rrf"
CONVEX = "convex"
FUSIONS = (RRF, CONVEX)


def rankings_for(
    lanes: Mapping[str, Lane],
    query: str,
    *,
    fuse: Sequence[str] = (RRF,),
    rrf_k: int = RRF_K,
    weights: Mapping[str, float] | None = None,
) -> dict[str, list[Hit]]:
    """Every ranking a query has: each lane, then the requested fusions."""
    per_lane = slice_lanes(lanes, query)
    every: dict[str, list[Hit]] = dict(per_lane)
    for method in fuse:
        if method == RRF:
            every[RRF] = rrf(per_lane, rrf_k)
        elif method == CONVEX:
            if not weights:
                raise ValueError("convex fusion needs weights; run `calibrate` or pass --weights")
            every[CONVEX] = convex(per_lane, weights)
        elif method:
            raise ValueError(f"unknown fusion {method!r}; known: {FUSIONS}")
    return every


def per_query_metrics(hits: Sequence[Hit], judgment: Judgment, ks: Sequence[int], group_of: GroupOf | None) -> dict:
    docs = [hit.doc for hit in hits]
    doc_level: dict[str, float] = {"rank": rank_of(hits, judgment.relevant)}
    for k in ks:
        doc_level[f"recall@{k}"] = recall_at(docs, judgment.relevant, k)
        doc_level[f"ndcg@{k}"] = ndcg_at(docs, judgment.grades, k)
    out = {"doc": doc_level}
    if group_of is not None:
        out["group"] = {"rank": rank_of(hits, judgment.relevant, level="group", group_of=group_of)}
    return out


def evaluate(
    lanes: Mapping[str, Lane],
    qrels: Qrels,
    *,
    ks: Sequence[int] = DEFAULT_KS,
    window: int = 5,
    group_of: GroupOf | None = None,
    fuse: Sequence[str] = (RRF,),
    rrf_k: int = RRF_K,
    weights: Mapping[str, float] | None = None,
    corpus: set[str] | None = None,
    blind: str | None = None,
    gate_on: str | None = None,
    gates: Sequence[Gate] = (),
    gate_level: str = "doc",
    margin: float = 0.0,
    with_sweep: bool = False,
) -> dict:
    """The report: a JSON-able dict. `report.py` renders it; `verdict.py` judges it."""
    lane_names = list(lanes)
    levels = ["doc"] + (["group"] if group_of is not None else [])
    rows: dict[str, dict[str, list]] = {}
    attribution: Counter = Counter()
    missed_by: Counter = Counter()
    blind_missed_by: Counter = Counter()
    handed: Counter = Counter()
    gate_rankings: dict[str, list[Hit]] = {}
    per_query: list[dict] = []
    ranking_names: list[str] = []
    n_indexed = 0

    for judgment in qrels:
        every = rankings_for(lanes, judgment.query, fuse=fuse, rrf_k=rrf_k, weights=weights)
        ranking_names = list(every)
        record: dict = {
            "query": judgment.query, "stratum": judgment.stratum, "negative": judgment.negative, "rankings": {},
        }
        if judgment.negative:
            for name, hits in every.items():
                record["rankings"][name] = {"candidates": len(hits)}
                if hits:
                    handed[name] += 1
        else:
            is_indexed = indexed(corpus, judgment.relevant)
            n_indexed += int(is_indexed)
            lane_ranks: dict[str, int] = {}
            for name, hits in every.items():
                metrics = per_query_metrics(hits, judgment, ks, group_of)
                for level in levels:
                    rows.setdefault(name, {lvl: [] for lvl in levels})[level].append((judgment.stratum, metrics[level]))
                doc_rank = int(metrics["doc"]["rank"])
                record["rankings"][name] = {
                    "rank": doc_rank,
                    "group_rank": int(metrics["group"]["rank"]) if "group" in metrics else None,
                    "candidates": len(hits),
                }
                if name in lane_names:
                    lane_ranks[name] = doc_rank
                if is_indexed and not 0 < doc_rank <= window:
                    missed_by[name] += 1
                    if judgment.stratum == blind:
                        blind_missed_by[name] += 1
            record["indexed"] = is_indexed
            record["attribution"] = miss_attribution(indexed=is_indexed, ranks=lane_ranks, window=window)
            attribution[record["attribution"]] += 1
        if gate_on:
            gate_rankings[judgment.query] = every.get(gate_on, [])
        per_query.append(record)

    positives = len(qrels.positives())
    negatives = len(qrels.negatives())
    judged = set(qrels.by_query)
    unjudged = {query for lane in lanes.values() for query in lane.hits} - judged
    report: dict = {
        "meta": {
            "lanes": lane_names,
            "rankings": ranking_names,
            "levels": levels,
            "ks": list(ks),
            "window": window,
            "rrf_k": rrf_k,
            "weights": dict(weights or {}),
            "blind": blind,
            "n": {
                "queries": len(qrels), "positives": positives, "negatives": negatives,
                "indexed": n_indexed, "unjudged_in_lanes": len(unjudged),
            },
        },
        "metrics": {name: {level: summarise(rows[name][level], ks) for level in levels} for name in rows},
        "attribution": {
            "counts": dict(attribution),
            "not_surfaced_share": proportion(attribution["not_surfaced"], n_indexed),
            "not_surfaced_by": {name: proportion(missed_by[name], n_indexed) for name in rows},
            "blind_share_of_misses": (
                {name: proportion(blind_missed_by[name], missed_by[name]) for name in rows} if blind else {}
            ),
        },
        "negatives": {name: proportion(handed[name], negatives) for name in ranking_names},
        "gates": {},
        "queries": per_query,
    }
    if gate_on:
        if gate_on not in ranking_names:
            raise ValueError(f"cannot gate on {gate_on!r}; rankings are {ranking_names}")
        report["meta"]["gate_on"] = gate_on
        report["meta"]["gate_level"] = gate_level
        for gate in gates:
            report["gates"][gate.name] = evaluate_gate(qrels, gate_rankings, gate, level=gate_level, group_of=group_of)
        if with_sweep:
            report["sweep"] = sweep(qrels, gate_rankings, margin=margin, level=gate_level, group_of=group_of)
    return report
