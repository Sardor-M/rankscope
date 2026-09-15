import pytest

from rankscope import fusion as f
from rankscope.lanes import Hit
from rankscope.ranks import rank_of


def hits(*docs, scores=None):
    scores = scores or [None] * len(docs)
    return [Hit(d, s) for d, s in zip(docs, scores)]


def test_rrf_prefers_docs_in_both_lanes_and_is_deterministic():
    fused = f.rrf({"a": hits("a", "b", "c"), "b": hits("c", "d")})
    assert fused[0].doc == "c" and fused[0].score == pytest.approx(1 / 63 + 1 / 61)
    assert {h.doc for h in fused} == {"a", "b", "c", "d"}
    assert [h.doc for h in f.rrf({"a": hits("b"), "b": hits("a")})] == ["a", "b"]


def test_rrf_dilutes_a_confident_lane():
    dense = hits("truth", "x", "y", scores=[0.9, 0.5, 0.4])
    fused = f.rrf({"lexical": hits("p", "x", "q", "r"), "dense": dense})
    assert rank_of(dense, ["truth"]) == 1 and fused[0].doc == "x"


def test_minmax_and_convex():
    assert f.minmax([2.0, 4.0, 6.0]) == [0.0, 0.5, 1.0]
    assert f.minmax([3.0, 3.0]) == [1.0, 1.0] and f.minmax([]) == []
    fused = f.convex({"a": hits("a", "b", scores=[10.0, 0.0]), "b": hits("b", "a", scores=[0.9, 0.1])}, {"a": 0.3, "b": 0.7})
    assert [h.doc for h in fused] == ["b", "a"] and fused[0].score == pytest.approx(0.7)
    with pytest.raises(ValueError):
        f.convex({"a": hits("a", "b")}, {"a": 1.0})


def test_weight_grid_and_fit():
    grid = f.weight_grid(["a", "b"], 0.1)
    assert len(grid) == 11 and len(f.weight_grid(["a", "b", "c"], 0.1)) == 66
    assert all(abs(sum(w.values()) - 1) < 1e-9 for w in grid)

    def score(fused):
        rank = rank_of(fused, ["t"])
        return 1.0 / rank if rank else 0.0

    per_query = [
        ({"good": hits("t", "a", scores=[0.9, 0.2]), "bad": hits("a", "b", "t", scores=[9.0, 8.0, 1.0])}, score),
        ({"good": hits("t", "a", scores=[0.8, 0.3]), "bad": hits("a", "t", scores=[5.0, 4.0])}, score),
    ]
    weights, value, table = f.fit_weights(per_query, ["good", "bad"], 0.5)
    assert weights == {"good": 1.0, "bad": 0.0} and value == 1.0 and len(table) == 3


def test_parse_weights():
    assert f.parse_weights("lexical=0.3, dense=0.7") == {"lexical": 0.3, "dense": 0.7}
    with pytest.raises(ValueError):
        f.parse_weights("lexical")


def test_cap_per_group_and_document_first():
    group_of = lambda doc: doc.split("#")[0]
    flood = hits("C#1", "C#2", "C#3", "C#4", "C#5", "C#6", "T#t", "C#7", "T#u", scores=[9, 8, 7, 6, 5, 4, 3, 2, 1])
    capped = f.cap_per_group(flood, group_of, 2)
    assert [h.doc for h in capped] == ["C#1", "C#2", "T#t", "T#u"]
    with pytest.raises(ValueError):
        f.cap_per_group(flood, group_of, 0)
    right_doc_floods = hits("M#p1", "M#p2", "M#p3", "M#p4", "M#p5", "M#p6", "M#truth", "X#a", scores=[8, 7, 6, 5, 4, 3, 2, 1])
    assert rank_of(f.cap_per_group(right_doc_floods, group_of, 5), ["M#truth"]) == 0
    first = f.document_first(right_doc_floods, group_of)
    assert rank_of(first, ["M#truth"]) == 7 and [h.doc for h in first][-1] == "X#a"
    mixed = hits("M#p1", "X#a", "M#truth", "X#b", "X#c", scores=[5, 4, 3, 2, 1])
    assert [h.doc for h in f.document_first(mixed, group_of, cap_others=1)] == ["M#p1", "M#truth", "X#a"]
    assert f.document_first([], group_of) == []
