import math

from rankscope import gate as g
from rankscope.lanes import Hit
from rankscope.ranks import Judgment, Qrels

POS = Judgment("p", {"G1#t": 1})
NEG = Judgment("n", {})


def hits(*pairs):
    return [Hit(d, s) for d, s in pairs]


def test_decide_reasons():
    assert g.decide([], g.Gate("x")).reason == "no_candidates"
    assert g.decide(hits(("G1#t", 0.5)), g.Gate("x", floor=0.6)).reason == "below_floor"
    assert g.decide(hits(("G1#t", 0.7), ("G2#a", 0.69)), g.Gate("x", margin=0.05)).reason == "margin"
    decision = g.decide(hits(("G1#t", 0.7), ("G2#a", 0.5)), g.Gate("x", floor=0.6, margin=0.1))
    assert decision.status == g.ACCEPT and decision.hit.doc == "G1#t"
    assert g.decide([Hit("G1#t")], g.Gate("x", floor=0.0)).reason == "below_floor"


def test_outcomes():
    accept = g.decide(hits(("G1#t", 0.9)), g.Gate("x"))
    wrong = g.decide(hits(("G2#a", 0.9)), g.Gate("x"))
    none = g.decide([], g.Gate("x"))
    assert g.outcome(POS, accept) == "accept_correct"
    assert g.outcome(POS, wrong) == "accept_wrong"
    assert g.outcome(POS, none) == "abstain"
    assert g.outcome(NEG, accept) == "false_accept"
    assert g.outcome(NEG, none) == "reject"
    sibling = g.decide(hits(("G1#other", 0.9)), g.Gate("x"))
    assert g.outcome(POS, sibling, level="group", group_of=lambda d: d.split("#")[0]) == "accept_correct"


def test_evaluate_sweep_and_operating_point():
    qrels = Qrels([POS, Judgment("p2", {"G1#t": 1}), NEG, Judgment("n2", {})])
    rankings = {"p": hits(("G1#t", 0.8)), "p2": hits(("G1#t", 0.55)), "n": hits(("G3#z", 0.7)), "n2": hits(("G3#z", 0.4))}
    out = g.evaluate(qrels, rankings, g.Gate("floor", floor=0.6))
    assert out["counts"] == {"accept_correct": 1, "accept_wrong": 0, "abstain": 1, "false_accept": 1, "reject": 1}
    assert out["far"]["p"] == 0.5 and out["fnir"]["p"] == 0.5
    rows = g.sweep(qrels, rankings)
    floors = [row["floor"] for row in rows]
    assert floors[0] == -math.inf and floors[-1] == math.inf and floors == sorted(floors)
    assert [row["fpir"]["p"] for row in rows] == sorted((row["fpir"]["p"] for row in rows), reverse=True)
    assert g.operating_point(rows, 0.0)["floor"] > 0.7
    assert g.operating_point(rows, 1.0)["floor"] == -math.inf


def _reference_sweep(qrels, rankings, margin=0.0):
    tops = sorted({hits[0].score for hits in rankings.values() if hits and hits[0].score is not None})
    return [g.evaluate(qrels, rankings, g.Gate("ref", floor, margin)) for floor in [-math.inf, *tops, math.inf]]


def test_sweep_matches_the_brute_force_definition():
    import random

    rng = random.Random(4)
    judgments, rankings = [], {}
    for i in range(120):
        query = f"q{i}"
        negative = i % 3 == 0
        judgments.append(Judgment(query, {} if negative else {"G1#t": 1}))
        if i % 11 == 0:
            rankings[query] = []
        elif i % 13 == 0:
            rankings[query] = [Hit("G1#t"), Hit("G2#a")]
        else:
            top = rng.choice(["G1#t", "G2#a"])
            first = round(rng.random(), 3)
            rankings[query] = [Hit(top, first), Hit("G3#z", round(first - rng.random() * 0.2, 3))]
    for margin in (0.0, 0.05):
        fast = g.sweep(Qrels(judgments), rankings, margin=margin)
        slow = _reference_sweep(Qrels(judgments), rankings, margin)
        assert len(fast) == len(slow)
        for row, ref in zip(fast, slow):
            assert row["floor"] == ref["floor"]
            assert row["fpir"] == ref["far"] and row["fnir"] == ref["fnir"]


def test_operating_point_conservative_uses_the_upper_bound():
    rows = [{"floor": 0.5, "fpir": {"p": 0.0, "k": 0, "n": 30, "lo": 0.0, "hi": 0.114}, "fnir": {"p": 0.2}},
            {"floor": math.inf, "fpir": {"p": 0.0, "k": 0, "n": 30, "lo": 0.0, "hi": 0.114}, "fnir": {"p": 1.0}}]
    assert g.operating_point(rows, 0.05)["floor"] == 0.5
    assert g.operating_point(rows, 0.05, conservative=True)["floor"] == math.inf
    assert g.operating_point(rows, 0.12, conservative=True)["floor"] == 0.5
