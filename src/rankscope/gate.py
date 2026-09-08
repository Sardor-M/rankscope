"""The abstain gate: a calibrated floor and a margin decide accept or none; FNIR at fixed FPIR reports it."""

from __future__ import annotations

import math
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
    name: str
    floor: float = -math.inf
    margin: float = 0.0


@dataclass(frozen=True)
class Decision:
    status: str
    reason: str
    hit: Hit | None = None


def decide(hits: Sequence[Hit], gate: Gate) -> Decision:
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
    """Confusion counts plus FAR (negatives accepted) and FNIR (positives not correctly accepted)."""
    counts = {name: 0 for name in OUTCOMES}
    per_query: dict[str, dict] = {}
    for judgment in qrels:
        decision = decide(rankings.get(judgment.query, []), gate)
        result = outcome(judgment, decision, level, group_of)
        counts[result] += 1
        per_query[judgment.query] = {
            "status": decision.status,
            "reason": decision.reason,
            "outcome": result,
            "doc": decision.hit.doc if decision.hit else "",
        }
    positives = counts["accept_correct"] + counts["accept_wrong"] + counts["abstain"]
    negatives = counts["false_accept"] + counts["reject"]
    return {
        "name": gate.name,
        "floor": gate.floor,
        "margin": gate.margin,
        "counts": counts,
        "far": proportion(counts["false_accept"], negatives),
        "fnir": proportion(counts["accept_wrong"] + counts["abstain"], positives),
        "accept_correct": proportion(counts["accept_correct"], positives),
        "accept_wrong": proportion(counts["accept_wrong"], positives),
        "queries": per_query,
    }


def sweep(
    qrels: Qrels,
    rankings: Mapping[str, Sequence[Hit]],
    *,
    margin: float = 0.0,
    level: str = "doc",
    group_of: GroupOf | None = None,
) -> list[dict]:
    """FPIR and FNIR at every floor the observed top scores suggest, lowest floor first."""
    tops = sorted({hits[0].score for hits in rankings.values() if hits and hits[0].score is not None})
    rows = []
    for floor in [-math.inf, *tops, math.inf]:
        result = evaluate(qrels, rankings, Gate("sweep", floor, margin), level=level, group_of=group_of)
        rows.append({"floor": floor, "fpir": result["far"], "fnir": result["fnir"]})
    return rows


def operating_point(rows: Sequence[dict], fpir_max: float) -> dict:
    """The lowest floor whose FPIR point estimate is within `fpir_max`."""
    for row in rows:
        if row["fpir"]["p"] <= fpir_max:
            return row
    return rows[-1]
