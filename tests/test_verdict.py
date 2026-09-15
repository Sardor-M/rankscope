import pytest

from rankscope import verdict as v

REPORT = {
    "metrics": {"dense": {"doc": {"hard": {"hit@20": 0.8, "hit@20_ci": [0.6, 0.92]}}}},
    "gates": {"fpir-0.05": {"far": {"p": 0.0, "k": 0, "n": 30, "lo": 0.0, "hi": 0.114}}},
}


def test_resolve():
    assert v.resolve(REPORT, "gates.fpir-0.05.far") == (0.0, 0.0, 0.114)
    assert v.resolve(REPORT, "metrics.dense.doc.hard.hit@20") == (0.8, 0.6, 0.92)
    with pytest.raises(KeyError):
        v.resolve(REPORT, "metrics.dense.doc.easy.hit@20")


def test_judge_and_overall():
    rows = v.judge(REPORT, [
        {"name": "recall", "path": "metrics.dense.doc.hard.hit@20", "op": ">=", "threshold": 0.7},
        {"name": "far", "path": "gates.fpir-0.05.far", "op": "<=", "threshold": 0.05},
    ])
    assert rows[0]["point"] and not rows[0]["interval"] and rows[1]["point"] and not rows[1]["interval"]
    assert v.overall(rows).startswith("PASS ON POINT")
    assert v.overall(v.judge(REPORT, [{"name": "r", "path": "metrics.dense.doc.hard.hit@20", "op": ">=", "threshold": 0.5}])).startswith("PASS (")
    assert v.overall(v.judge(REPORT, [{"name": "r", "path": "metrics.dense.doc.hard.hit@20", "op": ">=", "threshold": 0.9}])).startswith("FAIL")
    with pytest.raises(ValueError):
        v.judge(REPORT, [{"name": "x", "path": "gates.fpir-0.05.far", "op": "==", "threshold": 1}])


def test_relative_rules_compare_two_paths():
    report = {"metrics": {
        "rrf": {"doc": {"all": {"hit@5": 0.63, "hit@5_ci": [0.51, 0.74]}}},
        "dense": {"doc": {"all": {"hit@5": 0.85, "hit@5_ci": [0.74, 0.92]}}},
        "lexical": {"doc": {"all": {"hit@5": 0.35, "hit@5_ci": [0.24, 0.47]}}},
    }}
    rows = v.judge(report, [
        {"name": "fusion beats lexical", "path": "metrics.rrf.doc.all.hit@5", "op": ">=",
         "against": "metrics.lexical.doc.all.hit@5", "threshold": 0.03},
        {"name": "fusion keeps dense", "path": "metrics.rrf.doc.all.hit@5", "op": ">=",
         "against": "metrics.dense.doc.all.hit@5"},
        {"name": "fusion within 0.3 of dense", "path": "metrics.rrf.doc.all.hit@5", "op": "<=",
         "against": "metrics.lexical.doc.all.hit@5", "threshold": 0.3},
    ])
    assert rows[0]["point"] and rows[0]["interval"] and rows[0]["other"] == 0.35
    assert not rows[1]["point"] and not rows[1]["interval"]
    assert rows[2]["point"] and not rows[2]["interval"]
    assert v.overall(rows).startswith("FAIL")
    from rankscope.report import render_verdict

    text = render_verdict(rows, v.overall(rows))
    assert "0.63 vs 0.35" in text and ">= other +0.03" in text
