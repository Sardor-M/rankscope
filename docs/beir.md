# A walkthrough on a public dataset

rankscope reads the `{query: {doc: score}}` dictionaries that the BEIR toolkit and
pytrec_eval already use, so a BEIR run needs no conversion. This page shows the whole loop
on one dataset. It assumes `pip install beir` and network access to download the data;
rankscope itself needs neither.

## 1. Retrieve with two systems and save the dictionaries

```python
import json
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from beir.retrieval import models
from beir.retrieval.evaluation import EvaluateRetrieval
from beir.retrieval.search.dense import DenseRetrievalExactSearch
from beir.retrieval.search.lexical import BM25Search

path = util.download_and_unzip(util.get_dataset_url("scifact"), "datasets")   # small, graded qrels
corpus, queries, qrels = GenericDataLoader(path).load(split="test")

bm25 = EvaluateRetrieval(BM25Search(index_name="scifact", hostname="localhost"))     # needs Elasticsearch
dense = EvaluateRetrieval(DenseRetrievalExactSearch(models.SentenceBERT("BAAI/bge-small-en-v1.5")))
json.dump(bm25.retrieve(corpus, queries), open("bm25.json", "w"))
json.dump(dense.retrieve(corpus, queries), open("dense.json", "w"))
json.dump(qrels, open("qrels.json", "w"))
```

Any two retrievers will do; the point is two lanes over the same queries and qrels.

## 2. Add what a benchmark does not have: strata and negatives

Strata come from the queries themselves. Query length is the cheapest honest one:

```python
strata = {qid: "short" if len(q.split()) <= 8 else "long" for qid, q in queries.items()}
open("strata.tsv", "w").write("".join(f"{q}\t{s}\n" for q, s in strata.items()))
```

Negatives, queries whose answer is not in the corpus, do not exist in a benchmark. Make
them the honest way: hold out the relevant documents of a slice of queries from the index,
retrieve those queries again, and judge them with a zero grade so rankscope reads them as
negatives:

```python
held = list(qrels)[:50]
removed = {doc for qid in held for doc in qrels[qid]}
reduced = {doc: text for doc, text in corpus.items() if doc not in removed}
negatives = dense.retrieve(reduced, {qid: queries[qid] for qid in held})
json.dump({f"neg-{qid}": hits for qid, hits in negatives.items()}, open("dense-neg.json", "w"))
json.dump({f"neg-{qid}": {next(iter(reduced)): 0} for qid in held}, open("qrels-neg.json", "w"))
```

Merge the negative dictionaries into the lane and qrels files, or keep them as a second
pair and evaluate them separately; same-domain negatives are the ones a floor has to be
calibrated on.

## 3. Calibrate on one split, report on another

```sh
rankscope calibrate --lane bm25=bm25-dev.json --lane dense=dense-dev.json --qrels qrels-dev.json \
    --gate-on dense --conservative --out thresholds.json
rankscope evaluate --lane bm25=bm25.json --lane dense=dense.json --qrels qrels.json --strata strata.tsv \
    --thresholds thresholds.json --blind short --format markdown --out report.json --per-query rows.csv
```

The first numbers to read: the per-stratum rows for each lane, the not-surfaced share of
each lane, and the count of positives the dense lane surfaces that BM25 does not.

## 4. Decide with rules written first, and guard the repository

```json
{"rules": [
  {"name": "fusion keeps the best lane", "path": "metrics.rrf.doc.all.ndcg@10", "op": ">=",
   "against": "metrics.dense.doc.all.ndcg@10", "threshold": -0.01},
  {"name": "short queries are not left behind", "path": "metrics.rrf.doc.short.hit@10", "op": ">=", "threshold": 0.6},
  {"name": "no false accepts at the floor", "path": "gates.fpir-0.05.far", "op": "<=", "threshold": 0.05}
]}
```

```sh
rankscope verdict --report report.json --gates gates.json
rankscope diff --before baseline.json --after report.json --fail-on-regression
```

Commit `gates.json` and `baseline.json` before the change you are evaluating, and let CI
run the last two commands on every pull request.
