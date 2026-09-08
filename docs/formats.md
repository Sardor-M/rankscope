# File formats

## Lanes (runs)

**TREC run**, any extension other than `.jsonl`:

```
qid Q0 docid rank score tag
```

Whitespace separated, one hit per line, `#` lines ignored. Hits are ordered by score
descending (the rank column is not trusted, as in trec_eval); ties break on doc id
ascending so a run evaluates the same way everywhere. A document listed twice for one
query is an error.

**JSONL lane**, extension `.jsonl`:

```json
{"query": "q1", "hits": [{"doc": "d7", "score": 0.83}, {"doc": "d2"}]}
```

Hits are kept in file order, so a reranker's output without scores is a valid lane.
Scores are needed on the lane you gate on and on every lane you fuse convexly.

`--lane NAME=PATH` names a lane; `--lane PATH` uses the file stem. Order is kept and is the
order tables appear in.

## Qrels

**TREC qrels**:

```
qid 0 docid grade
```

A grade above 0 is relevant; higher is better (nDCG uses it). A query whose grades are all
0 is a **negative**: judged, nothing relevant. To add negatives that have no qrels row at
all, list their ids one per line and pass `--negatives FILE`.

**JSONL qrels**, extension `.jsonl`:

```json
{"query": "q1", "relevant": {"d7": 2, "d9": 1}, "stratum": "hard"}
{"query": "n1", "relevant": {}}
```

## Strata

`--strata FILE`: `qid<TAB>stratum` lines, or a JSON object `{"q1": "hard"}`. Unlabelled
queries get the stratum `unlabelled`.

## Groups

`--group-sep SEP`: the group of a doc id is everything before the first separator
(`acme-manual#p12` -> `acme-manual`). `--groups FILE`: an explicit `docid<TAB>group`
mapping or JSON object. Either enables group-level tables and `--gate-level group`.

## Corpus

`--corpus FILE`: one indexed doc id per line. Enables `not_indexed` attribution: a relevant
document absent from this list is a coverage miss, not a retrieval miss.

## Thresholds (written by `calibrate`)

```json
{
  "gate_on": "dense", "gate_level": "doc", "lanes": ["lexical", "dense"],
  "weights": {"lexical": 0.1, "dense": 0.9},
  "fit": {"objective": "mrr", "window": 5, "convex": 0.819, "rrf": 0.485, "grid": [...]},
  "null": {"n": 60, "mean": 0.665, "std": 0.048, "floor": 0.810},
  "conformal": {"alpha": 0.1, "n": 107, "min_n": 9, "floor": 0.663},
  "operating": {"fpir_max": 0.05, "floor": 0.760, "fpir": {...}, "fnir": {...}},
  "gates": [{"name": "null", "floor": 0.810}, {"name": "conformal-0.1", "floor": 0.663}, {"name": "fpir-0.05", "floor": 0.760}]
}
```

`evaluate --thresholds FILE` applies the weights (adding `convex` to the rankings), the
gate target and the three floors. Infinite floors are written as the strings `"inf"` and
`"-inf"`.

## Gates (rules for `verdict`)

```json
{"rules": [
  {"name": "dense finds the hard stratum", "path": "metrics.dense.doc.hard.hit@20", "op": ">=", "threshold": 0.6},
  {"name": "no false accepts at the operating floor", "path": "gates.fpir-0.05.far", "op": "<=", "threshold": 0.05}
]}
```

`path` is a dotted path into the report JSON; keys may contain dots. A path that lands on
a proportion object (`{"p", "k", "n", "lo", "hi"}`) is judged with its own interval; a path
that lands on a float uses a `<name>_ci` sibling when one exists.

## Report JSON (`evaluate --out`)

```
meta          lanes, rankings, levels, ks, window, rrf_k, weights, blind, gate_on, n {queries, positives, negatives, indexed, unjudged_in_lanes}
metrics       ranking -> level -> stratum -> {n, mrr, hit@k, hit@k_ci, recall@k, ndcg@k}   (recall and ndcg at doc level only)
attribution   counts, not_surfaced_share, not_surfaced_by[ranking], blind_share_of_misses[ranking]
negatives     ranking -> proportion of negatives that received any candidate
gates         name -> {floor, margin, counts, far, fnir, accept_correct, accept_wrong, queries}
sweep         [{floor, fpir, fnir}]   (with --sweep)
queries       one record per judged query: ranks per ranking, attribution, indexed
```

Proportions are objects `{"p", "k", "n", "lo", "hi"}` (Wilson, 95 percent).
