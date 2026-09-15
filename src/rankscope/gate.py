"""The abstain gate: a calibrated floor and a margin decide accept or none; FNIR at fixed FPIR reports it."""

from __future__ import annotations

import math
from bisect import bisect_left
from dataclasses import dataclass
from typing import Mapping, Sequence

from .lanes import Hit
from .metrics import proportion
from .ranks import GroupOf, Judgment, Qrels, rank_of

ACCEPT = "accept"
ABSTAIN = "abstain"
OUTCOMES = ("accept_correct", "accept_wrong", "abstain", "false_accept", "reject")


@dataclass(frozen=True)
class Gate:
    """Accept the top hit when its score clears `floor`, leads the runner-up by `margin`, and, with `mutual` > 0,
    when the query sits within that rank among the queries that retrieved the same document."""

    name: str
    floor: float = -math.inf
    margin: float = 0.0
    mutual: int = 0


@dataclass(frozen=True)
class Decision:
    status: str
    reason: str
    hit: Hit | None = None


def reverse_index(rankings: Mapping[str, Sequence[Hit]]) -> dict[str, list[tuple[float, str]]]:
    """For every document, the queries that retrieved it, best score first: the other direction of the ranking."""
    index: dict[str, list[tuple[float, str]]] = {}
    for query, hits in rankings.items():
        for hit in hits:
            if hit.score is not None:
                index.setdefault(hit.doc, []).append((hit.score, query))
    for scored in index.values():
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
    return index


def reverse_rank(index: Mapping[str, Sequence[tuple[float, str]]], doc: str, query: str) -> int:
    """1-based position of `query` among the queries that retrieved `doc`; 0 when it is not among them."""
    for position, (_, other) in enumerate(index.get(doc, ()), start=1):
        if other == query:
            return position
    return 0


def decide(hits: Sequence[Hit], gate: Gate, reverse: int | None = None) -> Decision:
    if not hits:
        return Decision(ABSTAIN, "no_candidates")
    top = hits[0]
    score = top.score if top.score is not None else -math.inf
    if score < gate.floor:
        return Decision(ABSTAIN, "below_floor")
    if len(hits) > 1:
        second = hits[1].score if hits[1].score is not None else -math.inf
        if score - second < gate.margin:
            return Decision(ABSTAIN, "margin")
    if gate.mutual and not (reverse and reverse <= gate.mutual):
        return Decision(ABSTAIN, "not_mutual")
    return Decision(ACCEPT, "top", top)


def outcome(judgment: Judgment, decision: Decision, level: str = "doc", group_of: GroupOf | None = None) -> str:
    if judgment.negative:
        return "false_accept" if decision.status == ACCEPT else "reject"
    if decision.status != ACCEPT:
        return "abstain"
    correct = rank_of([decision.hit], judgment.relevant, level=level, group_of=group_of) == 1
    return "accept_correct" if correct else "accept_wrong"


def evaluate(
    qrels: Qrels,
    rankings: Mapping[str, Sequence[Hit]],
    gate: Gate,
    *,
    level: str = "doc",
    group_of: GroupOf | None = None,
) -> dict:
    """Confusion counts plus FAR (negatives accepted) and FNIR (positives not correctly accepted), overall and per stratum."""
    counts = {name: 0 for name in OUTCOMES}
    by_stratum: dict[str, dict[str, int]] = {}
    per_query: dict[str, dict] = {}
    index = reverse_index(rankings) if gate.mutual else {}
    for judgment in qrels:
        hits = rankings.get(judgment.query, [])
        reverse = reverse_rank(index, hits[0].doc, judgment.query) if gate.mutual and hits else None
        decision = decide(hits, gate, reverse)
        result = outcome(judgment, decision, level, group_of)
        counts[result] += 1
        by_stratum.setdefault(judgment.stratum, {name: 0 for name in OUTCOMES})[result] += 1
        per_query[judgment.query] = {
            "status": decision.status,
            "reason": decision.reason,
            "outcome": result,
            "doc": decision.hit.doc if decision.hit else "",
        }
    def rates(c: Mapping[str, int]) -> dict:
        positives = c["accept_correct"] + c["accept_wrong"] + c["abstain"]
        negatives = c["false_accept"] + c["reject"]
        return {
            "counts": dict(c),
            "far": proportion(c["false_accept"], negatives),
            "fnir": proportion(c["accept_wrong"] + c["abstain"], positives),
            "accept_correct": proportion(c["accept_correct"], positives),
            "accept_wrong": proportion(c["accept_wrong"], positives),
        }

    return {
        "name": gate.name,
        "floor": gate.floor,
        "margin": gate.margin,
        "mutual": gate.mutual,
        **rates(counts),
        "by_stratum": {stratum: rates(c) for stratum, c in by_stratum.items()},
        "queries": per_query,
    }


def _eligible(
    qrels: Qrels,
    rankings: Mapping[str, Sequence[Hit]],
    *,
    floor: float,
    margin: float,
    mutual: int,
    level: str,
    group_of: GroupOf | None,
    key: str,
) -> tuple[list[float], list[float], int, int]:
    """Sorted `key` (score or gap) of negatives and of correct positives that the swept condition alone would decide."""
    negatives: list[float] = []
    positives: list[float] = []
    n_pos = n_neg = 0
    index = reverse_index(rankings) if mutual else {}
    for judgment in qrels:
        if judgment.negative:
            n_neg += 1
        else:
            n_pos += 1
        hits = rankings.get(judgment.query, [])
        if not hits:
            continue
        score = hits[0].score if hits[0].score is not None else -math.inf
        second = (hits[1].score if hits[1].score is not None else -math.inf) if len(hits) > 1 else -math.inf
        gap = score - second
        if key == "score" and gap < margin:
            continue
        if key == "gap" and score < floor:
            continue
        if mutual and not (0 < reverse_rank(index, hits[0].doc, judgment.query) <= mutual):
            continue
        value = score if key == "score" else gap
        if judgment.negative:
            negatives.append(value)
        elif rank_of([hits[0]], judgment.relevant, level=level, group_of=group_of) == 1:
            positives.append(value)
    return sorted(negatives), sorted(positives), n_pos, n_neg


def _rows(candidates: Sequence[float], negatives: list[float], positives: list[float], n_pos: int, n_neg: int, key: str) -> list[dict]:
    rows = []
    for threshold in candidates:
        accepted_negatives = len(negatives) - bisect_left(negatives, threshold)
        accepted_positives = len(positives) - bisect_left(positives, threshold)
        rows.append({key: threshold, "fpir": proportion(accepted_negatives, n_neg),
                     "fnir": proportion(n_pos - accepted_positives, n_pos)})
    return rows


def sweep(
    qrels: Qrels,
    rankings: Mapping[str, Sequence[Hit]],
    *,
    margin: float = 0.0,
    mutual: int = 0,
    level: str = "doc",
    group_of: GroupOf | None = None,
) -> list[dict]:
    """FPIR and FNIR at every floor the observed top scores suggest, lowest floor first; linear time."""
    tops = sorted({hits[0].score for hits in rankings.values() if hits and hits[0].score is not None})
    negatives, positives, n_pos, n_neg = _eligible(qrels, rankings, floor=-math.inf, margin=margin, mutual=mutual,
                                                   level=level, group_of=group_of, key="score")
    return _rows([-math.inf, *tops, math.inf], negatives, positives, n_pos, n_neg, "floor")


def margin_sweep(
    qrels: Qrels,
    rankings: Mapping[str, Sequence[Hit]],
    *,
    floor: float = -math.inf,
    mutual: int = 0,
    level: str = "doc",
    group_of: GroupOf | None = None,
) -> list[dict]:
    """FPIR and FNIR at every margin the observed top1 - top2 gaps suggest, at a fixed floor; smallest margin first."""
    gaps = set()
    for hits in rankings.values():
        if hits and hits[0].score is not None:
            second = (hits[1].score if hits[1].score is not None else -math.inf) if len(hits) > 1 else -math.inf
            gaps.add(hits[0].score - second)
    negatives, positives, n_pos, n_neg = _eligible(qrels, rankings, floor=floor, margin=0.0, mutual=mutual,
                                                   level=level, group_of=group_of, key="gap")
    return _rows([0.0, *sorted(g for g in gaps if g > 0), math.inf], negatives, positives, n_pos, n_neg, "margin")


def operating_point(rows: Sequence[dict], fpir_max: float, *, conservative: bool = False) -> dict:
    """The first row (lowest floor or margin) whose FPIR is within `fpir_max`: its point estimate, or its Wilson upper bound."""
    key = "hi" if conservative else "p"
    for row in rows:
        if row["fpir"][key] <= fpir_max:
            return row
    return rows[-1]
