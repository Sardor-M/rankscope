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


def test_reverse_index_and_mutual_condition():
    rankings = {"q1": hits(("G1#t", 0.9), ("G2#a", 0.5)), "q2": hits(("G1#t", 0.95), ("G3#z", 0.4)), "q3": hits(("G2#a", 0.7))}
    index = g.reverse_index(rankings)
    assert [q for _, q in index["G1#t"]] == ["q2", "q1"] and g.reverse_rank(index, "G1#t", "q1") == 2
    assert g.reverse_rank(index, "G1#t", "q9") == 0
    assert g.decide(rankings["q1"], g.Gate("m", mutual=1), reverse=2).reason == "not_mutual"
    assert g.decide(rankings["q1"], g.Gate("m", mutual=2), reverse=2).status == g.ACCEPT
    assert g.decide(rankings["q1"], g.Gate("m", mutual=1), reverse=None).reason == "not_mutual"
    qrels = Qrels([Judgment("q1", {"G1#t": 1}), Judgment("q2", {"G1#t": 1}), Judgment("q3", {})])
    out = g.evaluate(qrels, rankings, g.Gate("m", mutual=1))
    assert out["queries"]["q1"]["outcome"] == "abstain" and out["queries"]["q2"]["outcome"] == "accept_correct"
    assert out["queries"]["q3"]["outcome"] == "false_accept" and out["mutual"] == 1
    assert out["by_stratum"]["unlabelled"]["counts"]["accept_correct"] == 1


def test_per_stratum_gate_rates():
    qrels = Qrels([Judgment("p1", {"G1#t": 1}, "easy"), Judgment("p2", {"G1#t": 1}, "hard"), Judgment("n1", {}, "negative")])
    rankings = {"p1": hits(("G1#t", 0.9)), "p2": hits(("G2#a", 0.9)), "n1": hits(("G3#z", 0.9))}
    out = g.evaluate(qrels, rankings, g.Gate("open"))
    assert out["by_stratum"]["easy"]["fnir"]["p"] == 0.0 and out["by_stratum"]["hard"]["fnir"]["p"] == 1.0
    assert out["by_stratum"]["negative"]["far"]["p"] == 1.0 and out["by_stratum"]["negative"]["fnir"]["n"] == 0


def _reference_margin_sweep(qrels, rankings, floor=-math.inf):
    gaps = set()
    for h in rankings.values():
        if h and h[0].score is not None:
            gaps.add(h[0].score - ((h[1].score if h[1].score is not None else -math.inf) if len(h) > 1 else -math.inf))
    return [g.evaluate(qrels, rankings, g.Gate("ref", floor, m)) for m in [0.0, *sorted(x for x in gaps if x > 0), math.inf]]


def test_margin_sweep_matches_the_brute_force_definition():
    import random

    rng = random.Random(9)
    judgments, rankings = [], {}
    for i in range(90):
        query = f"q{i}"
        judgments.append(Judgment(query, {} if i % 3 == 0 else {"G1#t": 1}))
        if i % 10 == 0:
            rankings[query] = [Hit("G1#t", 0.8)]
        else:
            top = rng.choice(["G1#t", "G2#a"])
            first = round(rng.random(), 3)
            rankings[query] = [Hit(top, first), Hit("G3#z", round(first - rng.random() * 0.3, 3))]
    for floor in (-math.inf, 0.4):
        fast = g.margin_sweep(Qrels(judgments), rankings, floor=floor)
        slow = _reference_margin_sweep(Qrels(judgments), rankings, floor)
        assert len(fast) == len(slow)
        for row, ref in zip(fast, slow):
            assert row["margin"] == ref["margin"] and row["fpir"] == ref["far"] and row["fnir"] == ref["fnir"]
    rows = g.margin_sweep(Qrels(judgments), rankings)
    assert rows[-1]["margin"] == math.inf and rows[-1]["fpir"]["p"] == 0.1   # lone candidates pass every margin
    point = g.operating_point(rows, 0.1)
    assert "margin" in point and point["fpir"]["p"] <= 0.1
