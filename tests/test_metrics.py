import pytest

from rankscope import metrics as m


def test_hit_rate_and_mrr():
    ranks = [1, 3, 0, 6]
    assert m.hit_rate(ranks, 1) == 0.25 and m.hit_rate(ranks, 5) == 0.5 and m.hit_rate(ranks, 20) == 0.75
    assert m.mrr(ranks) == pytest.approx((1 + 1 / 3 + 1 / 6) / 4)
    assert m.hit_rate([], 5) == 0.0 and m.mrr([]) == 0.0


def test_recall_fraction_and_ndcg():
    assert m.recall_at(["a", "x", "b", "y"], ["a", "b", "c"], 3) == pytest.approx(2 / 3)
    assert m.recall_at(["a"], [], 3) == 0.0
    assert m.ndcg_at(["a", "b"], {"a": 2, "b": 1}, 10) == 1.0
    assert m.ndcg_at(["b", "a"], {"a": 2, "b": 1}, 10) < 1.0
    assert m.ndcg_at(["x", "y"], {"a": 1}, 10) == 0.0
    assert m.ndcg_at(["x"], {}, 10) == 0.0


def test_summarise_by_stratum():
    rows = [("hard", {"rank": 0, "ndcg@5": 0.0}), ("hard", {"rank": 2, "ndcg@5": 0.6}), ("easy", {"rank": 1, "ndcg@5": 1.0})]
    table = m.summarise(rows, ks=(1, 5))
    assert table["hard"]["n"] == 2 and table["hard"]["hit@5"] == 0.5 and table["hard"]["mrr"] == 0.25
    assert table["hard"]["ndcg@5"] == pytest.approx(0.3)
    assert table[m.ALL]["hit@1"] == pytest.approx(1 / 3)
    lo, hi = table[m.ALL]["hit@5_ci"]
    assert 0 < lo < 2 / 3 < hi <= 1


def test_miss_attribution():
    assert m.miss_attribution(indexed=False, ranks={"a": 1}, window=5) == "not_indexed"
    assert m.miss_attribution(indexed=True, ranks={"a": 0, "b": 30}, window=5) == "not_surfaced"
    assert m.miss_attribution(indexed=True, ranks={"a": 0, "b": 3}, window=5) == "surfaced_by_b"
    assert m.miss_attribution(indexed=True, ranks={"b": 3, "a": 1}, window=5) == "surfaced_by_a+b"
