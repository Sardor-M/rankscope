# Examples

| Path | What it shows |
| :--- | :--- |
| `trec/` | A five-query TREC run pair with graded qrels, strata, a corpus list, one judged negative and a gates file. Small enough to trace every number by hand. |
| `quickstart.py` | The Python API on in-memory lanes. |
| `rag_gate.py` | Calibrating fusion weights and abstain floors on one query set and evaluating them on another. |

```sh
rankscope evaluate --lane examples/trec/lanes/bm25.run --lane examples/trec/lanes/dense.run \
    --qrels examples/trec/qrels.txt --strata examples/trec/strata.tsv --corpus examples/trec/corpus.txt \
    --group-sep '#' --k 1,5 --ci-k 5
python examples/quickstart.py
python examples/rag_gate.py
```
