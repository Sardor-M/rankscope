from rankscope import cli
from rankscope import diff as d
from rankscope.jsonio import load


def test_judge_by_interval_overlap():
    assert d.judge(0.5, 0.7, (0.4, 0.6), (0.65, 0.8)) == d.IMPROVED
    assert d.judge(0.5, 0.3, (0.4, 0.6), (0.2, 0.35)) == d.REGRESSED
    assert d.judge(0.5, 0.55, (0.4, 0.6), (0.45, 0.65)) == d.NOISE
    assert d.judge(0.5, 0.9, None, (0.8, 1.0)) == d.NO_INTERVAL
    assert d.judge(0.1, 0.3, (0.05, 0.15), (0.25, 0.4), higher_is_better=False) == d.REGRESSED


def test_diff_between_full_and_truncated_lanes(tmp_path, capsys):
    out = tmp_path / "synth"
    cli.main(["synth", "--out", str(out), "--seed", "5"])
    lanes = [f"--lane={out / 'lanes' / 'lexical.run'}", f"--lane={out / 'lanes' / 'dense.run'}"]
    before, after = tmp_path / "before.json", tmp_path / "after.json"
    assert cli.main(["evaluate", *lanes, "--qrels", str(out / "qrels.txt"), "--quiet", "--out", str(before)]) == 0
    assert cli.main(["evaluate", *lanes, "--qrels", str(out / "qrels.txt"), "--depth", "2", "--quiet", "--out", str(after)]) == 0
    capsys.readouterr()
    rows = d.diff(load(before), load(after))
    hit20 = next(r for r in rows if r["where"] == "dense · doc · all" and r["metric"] == "hit@20")
    assert hit20["delta"] < 0 and hit20["verdict"] == d.REGRESSED
    assert any(r["metric"] == "not_surfaced_share" for r in rows)
    assert d.summary(rows)[d.REGRESSED] >= 1 and len(d.notable(rows)) < len(rows)
    assert cli.main(["diff", "--before", str(before), "--after", str(after)]) == 0
    text = capsys.readouterr().out
    assert "regressed" in text and "summary:" in text
    assert cli.main(["diff", "--before", str(before), "--after", str(after), "--fail-on-regression", "--all",
                     "--format", "markdown", "--out", str(tmp_path / "d.json")]) == 1
    assert "| where |" in capsys.readouterr().out and load(tmp_path / "d.json")["summary"]["regressed"] >= 1
    assert cli.main(["diff", "--before", str(before), "--after", str(before), "--fail-on-regression"]) == 0
