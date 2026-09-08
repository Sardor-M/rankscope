"""Calibrate an abstain gate on one query set, evaluate it on another. Run: python examples/rag_gate.py"""

import tempfile
from pathlib import Path

from rankscope import Gate, calibrate, evaluate, group_by_separator, read_lane, read_qrels
from rankscope.lanes import read_mapping
from rankscope.report import render, render_calibration
from rankscope.synth import SynthConfig, write

out = Path(tempfile.mkdtemp()) / "synth"
write(SynthConfig(seed=1), out)                       # two lanes, strata, negatives, a calibration set

lanes = {name: read_lane(out / "lanes" / f"{name}.run") for name in ("lexical", "dense")}
strata = read_mapping(out / "strata.tsv")
calibration = read_qrels(out / "qrels-calibration.txt").with_strata(strata)
held_out = read_qrels(out / "qrels.txt").with_strata(strata)
group_of = group_by_separator("#")

thresholds = calibrate(lanes, calibration, gate_on="dense", alpha=0.1, fpir=0.05, group_of=group_of)
print(render_calibration(thresholds))
print()

gates = [Gate(g["name"], g["floor"]) for g in thresholds["gates"]]
report = evaluate(lanes, held_out, group_of=group_of, weights=thresholds["weights"], fuse=("rrf", "convex"),
                  gate_on="dense", gates=gates, blind="hard")
print(render(report, ci_k=5))
