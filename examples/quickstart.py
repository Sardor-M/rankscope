"""Evaluate two lanes from Python objects, no files needed. Run: python examples/quickstart.py"""

from rankscope import Hit, Judgment, Lane, Qrels, evaluate, group_by_separator
from rankscope.report import render

bm25 = Lane("bm25", {
    "q1": [Hit("acme#p12", 12.1), Hit("acme#p02", 9.4)],
    "q2": [Hit("bolt#p3", 15.0), Hit("bolt#p4", 14.1)],
    "q3": [Hit("press#p2", 5.5)],
    "n1": [Hit("acme#p02", 7.7)],
})
dense = Lane("dense", {
    "q1": [Hit("acme#p13", 0.81), Hit("acme#p12", 0.79)],
    "q2": [Hit("bolt#p4", 0.66), Hit("bolt#p9", 0.60)],
    "q3": [Hit("acme#p40", 0.74), Hit("acme#p41", 0.70)],
    "n1": [Hit("acme#p41", 0.71)],
})
qrels = Qrels([
    Judgment("q1", {"acme#p12": 2, "acme#p13": 1}, stratum="easy"),
    Judgment("q2", {"bolt#p3": 2}, stratum="easy"),
    Judgment("q3", {"acme#p40": 1}, stratum="hard"),
    Judgment("n1", {}, stratum="negative"),          # judged: nothing relevant exists
])

report = evaluate({"bm25": bm25, "dense": dense}, qrels, ks=(1, 5), window=5,
                  group_of=group_by_separator("#"), blind="hard")
print(render(report, ci_k=5))
print()
print("per-query attribution:", {q["query"]: q.get("attribution", "negative") for q in report["queries"]})
