"""Fit fusion weights and floors on calibration queries, never on the queries being reported."""

from __future__ import annotations

from typing import Mapping, Sequence

from .fusion import RRF_K, fit_weights, rrf
from .gate import operating_point, sweep
from .lanes import Hit, Lane, slice_lanes
from .measure import CONVEX, RRF, rankings_for
from .ranks import GroupOf, Qrels, rank_of
from .stats import conformal_floor, conformal_min_n, null_floor


def objective_for(relevant, kind: str, window: int):
    def score(fused: list[Hit]) -> float:
        rank = rank_of(fused, relevant)
        if kind == "mrr":
            return 1.0 / rank if rank else 0.0
        return 1.0 if 0 < rank <= window else 0.0

    return score


def calibrate(
    lanes: Mapping[str, Lane],
    qrels: Qrels,
    *,
    gate_on: str,
    alpha: float = 0.1,
    fpir: float = 0.05,
    sigmas: float = 3.0,
    rrf_k: int = RRF_K,
    objective: str = "mrr",
    window: int = 5,
    step: float = 0.1,
    margin: float = 0.0,
    group_of: GroupOf | None = None,
    gate_level: str = "doc",
) -> dict:
    names = list(lanes)
    positives = qrels.positives()
    negatives = qrels.negatives()
    if not positives or not negatives:
        raise ValueError("calibration needs both positives and out-of-corpus negatives (judged, nothing relevant)")

    per_query = [
        (slice_lanes(lanes, judgment.query), objective_for(judgment.relevant, objective, window))
        for judgment in positives
    ]
    if len(names) > 1:
        weights, fitted, table = fit_weights(per_query, names, step)
    else:
        weights, fitted, table = {names[0]: 1.0}, 0.0, []
    rrf_value = sum(score(rrf(per_lane, rrf_k)) for per_lane, score in per_query) / len(per_query)

    rankings = {
        judgment.query: rankings_for(lanes, judgment.query, fuse=(RRF, CONVEX), rrf_k=rrf_k, weights=weights).get(gate_on)
        for judgment in qrels
    }
    if any(ranking is None for ranking in rankings.values()):
        raise ValueError(f"cannot gate on {gate_on!r}; rankings are {names + [RRF, CONVEX]}")
    if any(hits and hits[0].score is None for hits in rankings.values()):
        raise ValueError(f"ranking {gate_on!r} carries no scores; floors need scores")

    null_scores = [rankings[j.query][0].score for j in negatives if rankings[j.query]]
    true_scores = []
    for judgment in positives:
        hits = rankings[judgment.query]
        rank = rank_of(hits, judgment.relevant, level=gate_level, group_of=group_of)
        if rank:
            true_scores.append(hits[rank - 1].score)

    null = null_floor(null_scores, sigmas)
    conformal = {"alpha": alpha, "n": len(true_scores), "min_n": conformal_min_n(alpha),
                 "floor": conformal_floor(true_scores, alpha)}
    rows = sweep(qrels, rankings, margin=margin, level=gate_level, group_of=group_of)
    operating = operating_point(rows, fpir)
    return {
        "gate_on": gate_on,
        "gate_level": gate_level,
        "lanes": names,
        "rrf_k": rrf_k,
        "margin": margin,
        "weights": weights,
        "fit": {"objective": objective, "window": window, "convex": fitted, "rrf": rrf_value,
                "grid": [{"weights": w, "value": v} for w, v in table]},
        "null": null,
        "conformal": conformal,
        "operating": {"fpir_max": fpir, **operating},
        "gates": [
            {"name": "null", "floor": null["floor"]},
            {"name": f"conformal-{alpha:g}", "floor": conformal["floor"]},
            {"name": f"fpir-{fpir:g}", "floor": operating["floor"]},
        ],
        "n": {
            "positives": len(positives), "negatives": len(negatives), "null_scores": len(null_scores),
            "missing_from_lane": {
                name: sum(1 for judgment in qrels if judgment.query not in lane.hits) for name, lane in lanes.items()
            },
        },
    }
