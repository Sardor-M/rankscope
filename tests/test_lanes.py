import pytest

from rankscope.lanes import Hit, Lane, load_lanes, read_ids, read_lane, read_mapping, read_qrels, write_lane


def test_trec_run_is_ordered_by_score(tmp_path):
    path = tmp_path / "bm25.run"
    path.write_text("q1 Q0 d2 1 3.5 tag\nq1 Q0 d1 2 9.0 tag\nq2 Q0 d3 1 1.0 tag\n# comment\n")
    lane = read_lane(path)
    assert lane.name == "bm25" and [h.doc for h in lane.hits["q1"]] == ["d1", "d2"]
    assert lane.hits["q1"][0].score == 9.0 and lane.depth() == 2 and lane.has_scores()
    (tmp_path / "bad.run").write_text("q1 Q0 d1 1 2.0 t\nq1 Q0 d1 2 1.0 t\n")
    with pytest.raises(ValueError):
        read_lane(tmp_path / "bad.run")
    (tmp_path / "short.run").write_text("q1 d1 1\n")
    with pytest.raises(ValueError):
        read_lane(tmp_path / "short.run")


def test_jsonl_lane_keeps_file_order(tmp_path):
    path = tmp_path / "rerank.jsonl"
    path.write_text('{"query": "q1", "hits": [{"doc": "d2"}, {"doc": "d1", "score": 0.9}]}\n')
    lane = read_lane(path, "rerank")
    assert [h.doc for h in lane.hits["q1"]] == ["d2", "d1"] and not lane.has_scores()


def test_write_then_read_round_trip(tmp_path):
    lane = Lane("x", {"q1": [Hit("a", 2.0), Hit("b")]})
    write_lane(tmp_path / "x.run", lane)
    back = read_lane(tmp_path / "x.run")
    assert [h.doc for h in back.hits["q1"]] == ["a", "b"] and back.hits["q1"][1].score == 0.5


def test_load_lanes_specs(tmp_path):
    (tmp_path / "a.run").write_text("q1 Q0 d1 1 1.0 t\n")
    (tmp_path / "b.run").write_text("q1 Q0 d1 1 1.0 t\n")
    lanes = load_lanes([f"lexical={tmp_path / 'a.run'}", str(tmp_path / "b.run")])
    assert list(lanes) == ["lexical", "b"]
    with pytest.raises(ValueError):
        load_lanes([str(tmp_path / "a.run"), f"a={tmp_path / 'b.run'}"])


def test_qrels_trec_and_jsonl(tmp_path):
    trec = tmp_path / "qrels.txt"
    trec.write_text("q1 0 d1 2\nq1 0 d2 0\nn1 0 d9 0\n")
    qrels = read_qrels(trec)
    assert qrels.by_query["q1"].relevant == frozenset({"d1"}) and qrels.by_query["n1"].negative
    jsonl = tmp_path / "qrels.jsonl"
    jsonl.write_text('{"query": "q1", "relevant": {"d1": 1}, "stratum": "hard"}\n{"query": "n1", "relevant": {}}\n')
    qrels = read_qrels(jsonl)
    assert qrels.by_query["q1"].stratum == "hard" and qrels.by_query["n1"].negative


def test_mapping_and_ids(tmp_path):
    (tmp_path / "strata.tsv").write_text("q1\thard\nq2\teasy\n# c\n")
    assert read_mapping(tmp_path / "strata.tsv") == {"q1": "hard", "q2": "easy"}
    (tmp_path / "m.json").write_text('{"d1": "G1"}')
    assert read_mapping(tmp_path / "m.json") == {"d1": "G1"}
    (tmp_path / "ids.txt").write_text("a\n\nb\n# x\n")
    assert read_ids(tmp_path / "ids.txt") == {"a", "b"}
