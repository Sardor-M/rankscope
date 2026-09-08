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
