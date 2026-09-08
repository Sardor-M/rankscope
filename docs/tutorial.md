# Tutorial: the internals, one module at a time

Thirteen lessons. Each names the module to read, a command to run, a question the output
answers, and an exercise that changes something. Commands assume `pip install -e ".[dev]"`
and `rankscope synth --out data/synth` from the repository root. Numbers quoted are for the
default seed; yours will match until you change the generator.

Two rules for the whole path. Write the number down before you explain it. And never move
a threshold after seeing what it would have decided.

## 1. Lanes and qrels (`lanes.py`)

Read `read_lane` and `read_qrels`, then `examples/trec/`.

```sh
rankscope evaluate --lane examples/trec/lanes/bm25.run --lane examples/trec/lanes/dense.run \
    --qrels examples/trec/qrels.txt --k 1,5
```

Question: the TREC run for `q4` lists `press-guide#p7` at rank 6 in the file. Where does it
land after loading, and why does the rank column not matter?

Exercise: write a JSONL lane for the same five queries with no scores, load it, and confirm
that file order is preserved. Then give the same document twice for one query and read the
error.

## 2. A rank has a level (`ranks.py`)

Read `rank_of`, `group_by_separator`, `Judgment`.

```sh
rankscope evaluate --lane examples/trec/lanes/bm25.run --lane examples/trec/lanes/dense.run \
    --qrels examples/trec/qrels.txt --group-sep '#' --k 1,5 --quiet
```

Question: bm25 scores hit@5 0.50 at doc level and 0.75 at group level. Which query is the
difference, and what kind of fix does it call for (better retrieval, or better ordering
inside a document the lane already found)?

Exercise: add a second relevant page for `q2` in a *different* group and observe both
levels. Then judge a group id itself as relevant (`q5 0 press-guide 1`) and explain what
each level reports.

## 3. Proportions deserve intervals (`metrics.py`, `stats.wilson`)

```sh
rankscope evaluate --lane data/synth/lanes/lexical.run --lane data/synth/lanes/dense.run \
    --qrels data/synth/qrels.txt --strata data/synth/strata.tsv --quiet
```

Question: the `easy` stratum has 24 queries and the `hard` stratum 14. Compare the width
of their hit@20 intervals. What would you need to make the hard interval as narrow as the
easy one?

Exercise: run `rankscope synth --queries 12 --negatives 4 --out data/tiny` and evaluate it.
Then compute `wilson(0, 3)` by hand from the formula in `docs/concepts.md` and check it
with `python -c "from rankscope import wilson; print(wilson(0, 3))"`.

## 4. Strata and the blind class (`--strata`, `--blind`)

Question: the lexical lane's hit@20 is 0.57 overall and 0.00 on `hard`. If you only had the
overall number, what decision would you have made about the lexical lane?

Exercise: on your own data, define a stratum from the labels (for example, queries sharing
no term with their relevant document) rather than from the query alone, and explain why a
label-derived stratum is a measurement while a query-derived one is a proxy that has to be
validated first.

## 5. Whose fault is a miss (`metrics.miss_attribution`)

```sh
rankscope evaluate ... --corpus data/synth/corpus.txt --blind hard --quiet | sed -n '/miss attribution/,/^$/p'
```

Question: six positives are `not_indexed`. No lane can help them and they are excluded
from the not-surfaced share. Why would counting them as retrieval misses be wrong?

Question: "not surfaced by lexical: 34/54" and "not surfaced by rrf: 16/54". Which number
tells you the dense lane earns its place, and which tells you fusion is not free?

Exercise: change `--window` from 5 to 20 and watch the attribution move. The window is the
number of candidates the next stage sees; say what that stage is in your own system.

## 6. Reciprocal rank fusion, and what it dilutes (`fusion.rrf`)

Question: dense hit@1 is 0.70 alone and 0.37 under RRF, while hit@20 is 0.87 in both.
Compute the RRF score of a document that is rank 1 in dense and absent from lexical, and
of one that is rank 3 in lexical and rank 5 in dense. Which wins?

Exercise: rerun with `--rrf-k 5` and `--rrf-k 600`; predict the direction of the hit@1
change before you run. Then look at `tests/test_fusion.py::test_rrf_dilutes_a_confident_lane`.

## 7. Fitting weights without lying (`fusion.fit_weights`, `calibrate.py`)

```sh
rankscope calibrate --lane data/synth/lanes/lexical.run --lane data/synth/lanes/dense.run \
    --qrels data/synth/qrels-calibration.txt --strata data/synth/strata.tsv --gate-on dense --out data/synth/thresholds.json
```

Question: the grid picks lexical 0.1 / dense 0.9 with MRR 0.82 on the calibration set
against RRF's 0.48. `evaluate --thresholds` then shows `convex` at hit@1 0.70 on the
held-out queries. Why is a held-out number that differs from the fitted one acceptable, and
what would be worrying?

Exercise: do the forbidden thing once, on purpose: calibrate on `qrels.txt` and evaluate
on `qrels.txt`. Compare with the honest run. Then read `docs/design.md` on why the two
commands take separate files.

## 8. The null floor (`stats.null_floor`)

Question: the calibration output says `null floor: n=60 mean=0.665 std=0.048 -> 0.81`.
On the held-out set that floor gives FAR 0/30 and FNIR 72 percent. What does it mean that
the best score a negative reaches (0.665 on average) is close to what true matches score?

Exercise: in `synth.py`, lower the hard-negative group bump from 0.08 to 0.0 and regenerate.
Watch the floor drop and FNIR improve. That is what an easy negative set does to a
measurement; calibrate on negatives that resemble your positives.

## 9. The conformal floor, and how few is too few (`stats.conformal_quantile`)

Question: with alpha 0.1 the floor is the ceil((n+1)(1−alpha))-th smallest nonconformity.
For n from 9 to 18 that index equals n. What is the floor then, in words? What does
rankscope return for n = 8, and why is that more honest than the sample minimum?

Question: the conformal gate has FNIR 32 percent and FAR 53 percent on the synthetic set.
What does a conformal floor bound, and what does it not?

Exercise: read `tests/test_stats.py::test_conformal_floor_holds_its_miss_rate` and change
the calibration size from 40 to 8; explain the failure.

## 10. FNIR at a fixed FPIR (`gate.sweep`, `gate.operating_point`)

```sh
rankscope evaluate ... --thresholds data/synth/thresholds.json --sweep --quiet | sed -n '/sweep/,$p'
```

Question: find the floor where FPIR first reaches 0.05 on the held-out negatives and
compare it with the floor `calibrate` chose on the calibration negatives. Which is the
measurement and which is the tuning?

Exercise: with 30 negatives the smallest non-zero FPIR you can observe is 1/30. How many
negatives do you need to *claim* FPIR at or below 0.02 after observing zero false accepts?
`rankscope.negatives_for(0.02)` has the Wilson answer; the rule of three has another.

## 11. Pre-registered verdicts (`verdict.py`)

```sh
rankscope verdict --report data/synth/report.json --gates data/synth/gates.json
```

Question: every rule passes on its point estimate and one is "open" on its interval. When
is deciding on point estimates acceptable, and what about the cost of being wrong would
make you require the interval?

Exercise: write `gates.json` for a system you have not built yet, in rankscope's paths. Put
it in version control before the first run.

## 12. Paired comparisons (`stats.paired_bootstrap`, `compare`)

```sh
rankscope compare --lane data/synth/lanes/lexical.run --lane data/synth/lanes/dense.run \
    --qrels data/synth/qrels.txt --candidate rrf --metric mrr --min-delta 0.02
```

Question: the delta is +0.31 with interval [+0.24, +0.38] and P = 1.000. Rerun with
`--metric hit@1` and `--candidate convex --thresholds data/synth/thresholds.json`. Which
comparison would you trust to survive another sample of 60 queries?

Exercise: compare `dense` against `convex`. The weights were fitted on the calibration
set; the comparison runs on the held-out set. Explain why that ordering matters.

## 13. Parity in another language (`goldens.py`, `reference/js/`)

```sh
rankscope goldens --out tests/goldens.txt
node reference/js/check.mjs tests/goldens.txt
node reference/js/check.mjs tests/goldens.txt --emit > /tmp/js.txt && rankscope goldens --check /tmp/js.txt
```

Exercise: change `Z_95` in `stats.py` to 1.960001, regenerate the goldens, and watch the
JavaScript check fail on the Wilson lines. Revert. Then port `minmax` and `convex` to
`rankscope.mjs`, add golden lines for them in `goldens.py`, and make both checks pass. The
ordering rule (score descending, doc id ascending) is the part that bites.

## Where to go next

Put your own runs through `evaluate` with strata that mean something in your domain, add
judged negatives, and calibrate a floor on negatives that look like your positives. The
first number to read is not a metric. It is the share of positives your new lane surfaces
that no existing lane does.
