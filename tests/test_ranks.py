import pytest

from rankscope.lanes import Hit
from rankscope.ranks import Judgment, Qrels, group_by_mapping, group_by_separator, indexed, rank_of

HITS = [Hit("G2#a", 0.9), Hit("G1#b", 0.8), Hit("G1#c", 0.7)]
GROUP = group_by_separator("#")


def test_judgment_relevant_and_negative():
    j = Judgment("q", {"a": 2, "b": 0, "c": 1})
    assert j.relevant == frozenset({"a", "c"}) and not j.negative
    assert Judgment("n", {"a": 0}).negative and Judgment("n2", {}).negative


def test_qrels_helpers():
    qrels = Qrels([Judgment("q", {"a": 1}), Judgment("n", {"a": 0})])
    assert [j.query for j in qrels.positives()] == ["q"] and [j.query for j in qrels.negatives()] == ["n"]
    with_strata = qrels.with_strata({"q": "hard"})
    assert with_strata.by_query["q"].stratum == "hard" and with_strata.by_query["n"].stratum == "unlabelled"
    extended = qrels.with_negatives(["n", "n3"])
    assert len(extended) == 3 and extended.by_query["n3"].negative and extended.by_query["n3"].stratum == "negative"
    with pytest.raises(ValueError):
        Qrels([Judgment("q", {}), Judgment("q", {})])


def test_doc_and_group_levels():
    assert rank_of(HITS, ["G1#c"]) == 3
    assert rank_of(HITS, ["G1#c"], level="group", group_of=GROUP) == 2
    assert rank_of(HITS, ["G9#x"]) == 0
    assert rank_of(HITS, ["G9#x", "G2#a"]) == 1
    with pytest.raises(ValueError):
        rank_of(HITS, ["G1#c"], level="page")
    with pytest.raises(ValueError):
        rank_of(HITS, ["G1#c"], level="group")


def test_group_by_mapping():
    group_of = group_by_mapping({"G1#b": "manual-1", "G1#c": "manual-1"})
    assert rank_of(HITS, ["G1#c"], level="group", group_of=group_of) == 2
    assert group_of("unknown") == "unknown"


def test_indexed_is_coverage_not_ranking():
    assert indexed(None, ["anything"])
    assert indexed({"a", "b"}, ["b", "z"])
    assert not indexed({"a", "b"}, ["z"])
