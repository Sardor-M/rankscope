# rankscope

Look inside a retrieval ranking. rankscope reads standard TREC run and qrels files and
tells you what a search or RAG pipeline actually does: metrics per query class with
confidence intervals, which component found each answer, honest rank fusion, a calibrated
"not found" gate, A/B comparisons that survive small samples, and pass/fail verdicts you
wrote down before the run.

Pure Python, no dependencies, one command.

## The questions it answers

A single recall number hides four things. rankscope makes each one visible.

- **Is the number stable?** Every hit rate carries a Wilson confidence interval sized to
  the queries you actually have, and system comparisons use a paired bootstrap.
- **Which queries fail?** Every table is broken down by stratum, so a component that is
  blind on one class of query cannot hide behind the average.
- **Which component found what?** Each miss is attributed: never indexed, never surfaced
  by any lane, or surfaced by exactly these lanes. That is the number that says whether a
  new retriever earns its place.
- **Can the system say "not found"?** Judged negatives, three ways to calibrate an
  abstain threshold, and the false-accept and miss rates each one produces.

## Install

```sh
pip install rankscope
```

Python 3.11 or newer. Nothing else.

## Quick start

The repository ships a five-query example that can be traced by hand.

```sh
rankscope evaluate --lane examples/trec/lanes/bm25.run --lane examples/trec/lanes/dense.run \
    --qrels examples/trec/qrels.txt --strata examples/trec/strata.tsv \
    --corpus examples/trec/corpus.txt --group-sep '#' --k 1,5 --ci-k 5 --blind hard
```

```
q1                           easy        bm25 1 · dense 1 · rrf 1 · surfaced_by_bm25+dense
q2                           easy        bm25 1 · dense MISS · rrf 3 · surfaced_by_bm25
q3                           hard        bm25 MISS · dense 1 · rrf 2 · surfaced_by_dense
q4                           hard        bm25 6 · dense MISS · rrf 6 · not_surfaced
n1                           negative    bm25 2 cands · dense 2 cands · rrf 4 cands

queries: 4 positives (4 indexed) · 1 negatives · lanes: bm25, dense · rankings: bm25, dense, rrf · window=5
depth: bm25 6 · dense 3   (lanes differ in depth; fusion treats a document below a lane's depth as absent from it)

bm25 · doc level
  stratum  n  hit@1  hit@5  MRR                  hit@5 95% CI  recall@5          ndcg@5
  all      4  0.50   0.50   0.542 [0.083-1.000]  0.15-0.85     0.38 [0.00-0.75]  0.457 [0.000-0.913]
  easy     2  1.00   1.00   1.000 [1.000-1.000]  0.34-1.00     0.75 [0.50-1.00]  0.913 [0.826-1.000]
  hard     2  0.00   0.00   0.083 [0.000-0.167]  0.00-0.66     0.00 [0.00-0.00]  0.000 [0.000-0.000]

bm25 · group level
  stratum  n  hit@1  hit@5  MRR                  hit@5 95% CI
  all      4  0.75   0.75   0.750 [0.250-1.000]  0.30-0.95
  easy     2  1.00   1.00   1.000 [1.000-1.000]  0.34-1.00
  hard     2  0.50   0.50   0.500 [0.000-1.000]  0.09-0.91

...

miss attribution (doc level, window=5)
  cause                   queries
  surfaced_by_bm25+dense  1
  surfaced_by_bm25        1
  surfaced_by_dense       1
  not_surfaced            1

  indexed but not surfaced by any lane: 1/4 = 25% (95% CI 5%-70%)
  ...
  hard share of bm25 misses: 2/2 = 100% (95% CI 34%-100%)
```

How to read it: `q4` is found at **group** level (the right manual) but not at **doc**
level within the window of five (the wrong page). `q3` is surfaced by the dense lane
only. The intervals are as wide as four queries deserve. And the negative `n1` still
receives candidates from every lane, which is why a gate exists.

## Commands

| Command | What it does |
| :--- | :--- |
| `evaluate` | Metrics per lane and per fusion, miss attribution, negatives, gates. Optional variants with a per-group cap or document-first ordering, and mutual-nearest-neighbour gates. Text, Markdown, LaTeX or JSON output, plus a per-query CSV. |
| `calibrate` | Fits fusion weights (MRR, hit or nDCG objective), three abstain floors, an operating margin and a mutual gate on a separate calibration set and writes `thresholds.json`. `--conservative` picks the FPIR floor on the Wilson upper bound. |
| `compare` | Paired bootstrap of one metric between two rankings, with a minimum effect size. |
| `verdict` | Judges a report against rules written before the run. Exits 1 on failure, so it fits in CI. |
| `diff` | What moved between two reports, per metric, judged on whether the intervals overlap. `--fail-on-regression` for CI. |
| `fuse` | Writes a fused TREC run (reciprocal rank fusion, or weighted) from several lanes. |
| `synth` | Generates lanes, judgments, strata and negatives with known truth, for learning and testing. |
| `goldens` | Writes or checks the reference vectors that other implementations use to prove parity. |

`rankscope <command> --help` lists every option.

## A typical workflow

```sh
# 1. fit weights and floors on a development set
rankscope calibrate --lane bm25.run --lane dense.run --qrels qrels-dev.txt --gate-on dense --out thresholds.json

# 2. report on the test set, with strata and the blind class named
rankscope evaluate --lane bm25.run --lane dense.run --qrels qrels-test.txt --thresholds thresholds.json \
    --strata strata.tsv --blind no-overlap --format markdown --out report.json

# 3. judge the report against rules you committed before running
rankscope verdict --report report.json --gates gates.json

# 4. is fusion really better than the lexical lane alone?
rankscope compare --lane bm25.run --lane dense.run --qrels qrels-test.txt --candidate rrf --metric ndcg@10 --min-delta 0.02
```

Calibration and evaluation deliberately take separate judgment files. A threshold fitted
and reported on the same queries is optimism, not measurement.

## Inputs

- **Lanes**: TREC run files (`qid Q0 docid rank score tag`), JSONL, or `.json` in the
  `{query: {doc: score}}` convention pytrec_eval and BEIR use. One file per retrieval
  system; pass several with repeated `--lane`.
- **Judgments**: TREC qrels (`qid 0 docid grade`), JSONL, or `.json` as `{query: {doc:
  grade}}`. A query whose grades are all zero counts as a **negative**: the corpus holds no answer for it. Extra negatives can be
  listed with `--negatives`.
- **Strata** (optional): a `query<TAB>label` file. Tables are reported per label.
- **Groups** (optional): `--group-sep '#'` reads `acme-manual#p12` as document
  `acme-manual`, enabling "right document, wrong passage" reporting. `--groups` takes an
  explicit mapping instead.
- **Corpus** (optional): one indexed id per line, so a relevant document that was never
  indexed is reported as a coverage miss rather than a retrieval miss.
- **Depth** (optional): `--depth N` cuts every lane to the same depth before fusion and
  floors. The report prints each lane's depth, warns when they differ, and counts judged
  queries missing from any lane.

The full specification, including the report JSON, is in [`docs/formats.md`](docs/formats.md).

## Python

```python
from rankscope import Hit, Judgment, Lane, Qrels, evaluate, group_by_separator
from rankscope.report import render

lanes = {
    "bm25":  Lane("bm25",  {"q1": [Hit("acme#p12", 12.1), Hit("acme#p02", 9.4)]}),
    "dense": Lane("dense", {"q1": [Hit("acme#p13", 0.81), Hit("acme#p12", 0.79)]}),
}
qrels = Qrels([Judgment("q1", {"acme#p12": 2, "acme#p13": 1}, stratum="easy")])

report = evaluate(lanes, qrels, ks=(1, 5), group_of=group_by_separator("#"))
print(render(report, ci_k=5))
```

`evaluate` returns a plain dictionary, the same JSON the CLI writes. `Lane.from_dict` and
`Qrels.from_dict` take the `{query: {doc: value}}` shape directly. `calibrate`,
`paired_bootstrap`, `rrf`, `convex`, `wilson`, `null_floor`, `conformal_quantile` and
`rank_of` are importable on their own. Two runnable scripts live in `examples/`.

## The ideas behind it

- **Intervals on everything reported.** Zero false accepts in three negatives bounds the
  rate at 56 percent, not at zero. Wilson intervals say so on every hit rate; MRR, recall
  and nDCG carry bootstrap intervals; `diff` calls a change real only when intervals part.
- **Two rank levels.** Right document and right passage are different failures with
  different fixes.
- **Miss attribution in a window.** The window is what the next stage sees (a reranker's
  input, a reader's context), not the depth of the run.
- **Fusion with a baseline.** Weighted fusion is fitted on calibration queries; reciprocal
  rank fusion is always reported beside it as the zero-tuning reference.
- **Three floors for "not found".** A null floor from what negatives score, a conformal
  floor with a finite-sample miss bound, and an operating floor at a chosen false-accept
  rate. Each is reported with its false-accept and miss rates so the trade-off is visible.
- **Verdicts before numbers.** Rules live in a file you commit before the run and are
  judged on the point estimate and on the interval.

[`docs/concepts.md`](docs/concepts.md) has every formula and the small-sample tables.

## Learning the internals

The package is small enough to read in a sitting, and each source file holds one idea.
[`docs/tutorial.md`](docs/tutorial.md) walks through them in sixteen lessons, each with a
command to run, a question the output answers, and an exercise. [`docs/workbook.html`](docs/workbook.html)
is an interactive version: open it in a browser and every idea becomes a slider, from a
rank with a level to a pre-registered verdict, computed by the same kernel the package
ships. [`docs/beir.md`](docs/beir.md) runs the whole loop on a public BEIR dataset. The
design decisions are recorded in [`docs/design.md`](docs/design.md).

## Parity in other languages

`rankscope goldens` writes reference vectors for every function in the kernel. Another
implementation prints the same lines with its own answers, and `rankscope goldens --check`
diffs them numerically. A JavaScript port ships in `reference/js` as the first example, so
a service in another language can gate its answers with the same floors the evaluation
calibrated.

## Development

```sh
pip install -e ".[dev]"
pytest -q
rankscope goldens --out tests/goldens.txt && node reference/js/check.mjs tests/goldens.txt
python -m build
```

## Status

Version 0.4.0. The arithmetic is tested and cross-checked against the JavaScript port; the
command-line surface may still change before 1.0. Not included yet: significance tests
other than the paired bootstrap, and per-stratum fusion weights.

## Licence

MIT.
