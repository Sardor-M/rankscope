"""synth -> calibrate -> evaluate -> verdict -> compare -> fuse, through the CLI, and the API."""

import json
import math
import subprocess
import sys

import pytest

from rankscope import cli, goldens
from rankscope.jsonio import load


@pytest.fixture(scope="module")
def synth_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("synth")
    assert cli.main(["synth", "--out", str(out), "--seed", "3"]) == 0
    return out


def test_full_pipeline(synth_dir, tmp_path, capsys):
    lanes = [f"--lane={synth_dir / 'lanes' / 'lexical.run'}", f"--lane={synth_dir / 'lanes' / 'dense.run'}"]
    common = [*lanes, "--group-sep", "#", "--strata", str(synth_dir / "strata.tsv")]
    thresholds = tmp_path / "thresholds.json"
    assert cli.main(["calibrate", *common, "--qrels", str(synth_dir / "qrels-calibration.txt"),
                     "--gate-on", "dense", "--out", str(thresholds)]) == 0
    cal = load(thresholds)
    assert cal["gate_on"] == "dense" and abs(sum(cal["weights"].values()) - 1) < 1e-9
    assert cal["conformal"]["n"] >= cal["conformal"]["min_n"] and cal["fit"]["convex"] >= cal["fit"]["rrf"] - 1e-9

    report_path = tmp_path / "report.json"
    assert cli.main(["evaluate", *common, "--qrels", str(synth_dir / "qrels.txt"), "--thresholds", str(thresholds),
                     "--corpus", str(synth_dir / "corpus.txt"), "--blind", "hard", "--sweep", "--quiet",
                     "--out", str(report_path)]) == 0
    report = load(report_path)
    metrics = report["metrics"]
    assert metrics["lexical"]["doc"]["hard"]["hit@20"] <= 0.25
    assert metrics["dense"]["doc"]["hard"]["hit@20"] >= 0.6
    assert metrics["rrf"]["doc"]["all"]["hit@20"] >= metrics["lexical"]["doc"]["all"]["hit@20"]
    assert metrics["dense"]["group"]["all"]["hit@1"] >= metrics["dense"]["doc"]["all"]["hit@1"]
    assert set(report["gates"]) == {"null", "conformal-0.1", "fpir-0.05"}
    assert "not_indexed" in report["attribution"]["counts"]
    assert report["sweep"][0]["floor"] == -math.inf
    assert report["meta"]["levels"] == ["doc", "group"]

    code = cli.main(["verdict", "--report", str(report_path), "--gates", str(synth_dir / "gates.json")])
    assert "verdict:" in capsys.readouterr().out and code in (0, 1)

    assert cli.main(["evaluate", *common, "--qrels", str(synth_dir / "qrels.txt"), "--format", "markdown", "--quiet"]) == 0
    assert "| stratum |" in capsys.readouterr().out
    assert cli.main(["evaluate", *common, "--qrels", str(synth_dir / "qrels.txt"), "--format", "json"]) == 0
    assert json.loads(capsys.readouterr().out)["meta"]["lanes"] == ["lexical", "dense"]


def test_compare_and_fuse(synth_dir, tmp_path, capsys):
    lanes = [f"--lane={synth_dir / 'lanes' / 'lexical.run'}", f"--lane={synth_dir / 'lanes' / 'dense.run'}"]
    code = cli.main(["compare", *lanes, "--qrels", str(synth_dir / "qrels.txt"), "--metric", "hit@5",
                     "--resamples", "500"])
    out = capsys.readouterr().out
    assert "P(candidate > baseline)" in out and code in (0, 1)
    assert cli.main(["compare", *lanes, "--qrels", str(synth_dir / "qrels.txt"), "--baseline", "lexical",
                     "--candidate", "rrf", "--metric", "ndcg@10", "--resamples", "500"]) in (0, 1)
    fused = tmp_path / "fused.run"
    assert cli.main(["fuse", *lanes, "--out", str(fused)]) == 0
    lines = fused.read_text().splitlines()
    assert lines and lines[0].split()[1] == "Q0" and lines[0].endswith("rrf")
    assert cli.main(["fuse", *lanes, "--weights", "lexical=0.3,dense=0.7", "--out", str(tmp_path / "c.run")]) == 0


def test_evaluate_without_groups_and_with_extra_negatives(synth_dir, tmp_path):
    negatives = tmp_path / "neg.txt"
    negatives.write_text("extra-1\nextra-2\n")
    report_path = tmp_path / "r.json"
    assert cli.main(["evaluate", f"--lane={synth_dir / 'lanes' / 'dense.run'}", "--qrels", str(synth_dir / "qrels.txt"),
                     "--negatives", str(negatives), "--fuse", "none", "--gate", "manual=0.7", "--quiet",
                     "--out", str(report_path)]) == 0
    report = load(report_path)
    assert report["meta"]["levels"] == ["doc"] and report["meta"]["rankings"] == ["dense"]
    assert report["meta"]["n"]["negatives"] == 32 and report["gates"]["manual"]["floor"] == 0.7


def test_goldens_self_check_and_js_reference(tmp_path):
    text = goldens.build()
    assert text.startswith("rankscope-goldens 1\n") and goldens.check(text) == []
    broken = text.replace("rule_of_three 30 -> 0.1", "rule_of_three 30 -> 0.2")
    assert len(goldens.check(broken)) == 1
    path = tmp_path / "goldens.txt"
    path.write_text(text)
    node = subprocess.run(["node", "reference/js/check.mjs", str(path)], capture_output=True, text=True)
    if node.returncode == 2 or "not found" in node.stderr:
        pytest.skip("node not available")
    assert node.returncode == 0, node.stdout + node.stderr
    emitted = subprocess.run(["node", "reference/js/check.mjs", str(path), "--emit"], capture_output=True, text=True).stdout
    assert goldens.check(emitted) == []


def test_depth_missing_lanes_and_conservative_floor(synth_dir, tmp_path, capsys):
    lanes = [f"--lane={synth_dir / 'lanes' / 'lexical.run'}", f"--lane={synth_dir / 'lanes' / 'dense.run'}"]
    assert cli.main(["evaluate", *lanes, "--qrels", str(synth_dir / "qrels.txt"), "--depth", "5", "--format", "json"]) == 0
    meta = json.loads(capsys.readouterr().out)["meta"]
    assert meta["depths"] == {"lexical": 5, "dense": 5} and meta["missing_from_lane"] == {"lexical": 0, "dense": 0}

    partial = tmp_path / "partial.jsonl"
    partial.write_text('{"query": "q000", "hits": [{"doc": "D01#001", "score": 0.9}]}\n')
    assert cli.main(["evaluate", f"--lane={synth_dir / 'lanes' / 'dense.run'}", f"--lane=partial={partial}",
                     "--qrels", str(synth_dir / "qrels.txt"), "--quiet"]) == 0
    out = capsys.readouterr().out
    assert "missing from a lane" in out and "partial 89" in out and "lanes differ in depth" in out
