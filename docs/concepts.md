# Concepts

Every quantity rankscope computes, what it assumes, and what it does not tell you. Function
names point into `src/rankscope/`.

## Lanes, qrels, cases

A **lane** is one system's ranked output: a TREC run file (`qid Q0 docid rank score tag`)
or a JSONL file. Several lanes over the same queries are what fusion and attribution
compare. **Qrels** are graded relevance judgments (`qid 0 docid grade`). A query with at
least one positive grade is a **positive**; a query that is judged but has no positive
grade is a **negative**: the corpus does not contain its answer, and the right response is
to say so. A query that appears in a lane but not in the qrels is unjudged and ignored.

## Ranks at two levels (`ranks.py`)

`rank_of(hits, relevant)` is the 1-based position of the first relevant document, and
**0 for a miss**. Zero is a sentinel, not a rank; every consumer tests `0 < r <= k`.

With a group function (`--group-sep '#'` turns `acme-manual#p12` into `acme-manual`),
the same ranking has a second rank: the first hit whose *group* contains a relevant
document. Group level answers "right document?"; doc level answers "right passage, page or
figure inside it?". A group hit with a doc miss is a ranking problem inside something the
lane already found, and it wants a different fix (re-rank within the group) than a group
miss (retrieval or coverage).

## Hit rate, recall, MRR, nDCG (`metrics.py`)

For n positives with first-relevant ranks r_1..r_n:

```
hit@k    = |{ i : 0 < r_i <= k }| / n          did any relevant document make the top k
MRR      = (1/n) * sum over r_i > 0 of 1/r_i
recall@k = |relevant ∩ top k| / |relevant|      the fraction of relevant documents retrieved
nDCG@k   = DCG@k / ideal DCG@k, gain 2^grade - 1, discount 1/log2(rank + 1)
```

`hit@k` is a proportion, so it gets a Wilson interval. `recall@k` and `nDCG@k` are means
of per-query values; compare them across systems with the paired bootstrap (`compare`).
The two recalls coincide when every query has one relevant document.

## Strata

A **stratum** is a label per query (`--strata`). Every table is reported per stratum and
overall, because a mean hides classes: a lexical lane can score 0.57 hit@20 overall and
0.00 on the stratum where the query shares no term with its answer. The stratum you
name with `--blind` is the one the attribution block singles out.

## Wilson interval, rule of three (`stats.wilson`, `rule_of_three`, `negatives_for`)

For s successes in n trials, p = s/n, z = 1.96:

```
centre = (p + z^2/2n) / (1 + z^2/n)
half   = z * sqrt( p(1-p)/n + z^2/4n^2 ) / (1 + z^2/n)
```

Unlike the normal approximation it never collapses to zero width at s = 0 or s = n.

| s / n | 95 % interval | reading |
| :--- | :--- | :--- |
| 0 / 3 | 0 to 0.56 | zero false accepts in three negatives says almost nothing |
| 0 / 30 | 0 to 0.11 | still cannot claim a rate below ten percent |
| 8 / 12 | 0.39 to 0.86 | eight of twelve is not "67 percent", it is "somewhere in 39 to 86" |
| 45 / 50 | 0.79 to 0.96 | |

After zero events the upper limit is z²/(n + z²). The negatives needed to claim a rate at
or below a bound b, having seen zero: n >= z²(1 − b)/b.

| bound | Wilson n | rule of three (3/n) |
| :--- | ---: | ---: |
| 0.10 | 35 | 30 |
| 0.05 | 73 | 60 |
| 0.02 | 189 | 150 |
| 0.01 | 381 | 300 |

## Miss attribution (`metrics.miss_attribution`)

For each positive, given every lane's doc-level rank and a **window** (the number of
candidates the next stage will look at: a reranker's input, a reader's context, the crops
shown to a verifier):

```
relevant document absent from --corpus    -> not_indexed      coverage; no lane can help
no lane has 0 < rank <= window            -> not_surfaced     retrieval
otherwise                                 -> surfaced_by_<lanes with 0 < rank <= window>
```

This converts a recall number into a cause. The share of positives a new lane surfaces
that no existing lane does is the number that says whether it earns its keep.

## Reciprocal rank fusion (`fusion.rrf`)

```
score(doc) = sum over lanes where doc appears at rank r of 1 / (k + r),   k = 60 by default
```

Presence in two lanes beats being first in one: rank 1 in a single lane scores 1/61 ≈ 0.0164,
ranks 3 and 5 in two lanes score 1/63 + 1/65 ≈ 0.0313. That is the point of RRF and also
its cost: a lane's confident-wrong top hits dilute another lane's correct top hit. On the
synthetic set the dense lane's hit@1 drops from 0.70 alone to 0.37 fused while hit@20 is
unchanged. Small k trusts the top of each lane; large k trusts presence. A lane's 21st
item scores nothing, so the depth you fuse at matters, and the fused list is the union,
longer than any lane.

## Convex fusion and the weight fit (`fusion.convex`, `fit_weights`)

Each lane's scores are min-max normalised over its own list (absent document = 0, constant
list = all ones), then summed with weights on the simplex. `calibrate` walks a grid
(11 vectors for two lanes at step 0.1) and keeps the best mean objective (MRR or hit
within the window) over the **calibration** queries. RRF is always reported beside it as
the zero-tuning baseline. Weights fitted and reported on the same queries are optimism,
not measurement, which is why `calibrate` and `evaluate` read different qrels.

## The null floor (`stats.null_floor`)

Take the best score each **negative** query reaches against the index, and over those n
maxima compute `mean + 3·std` (sample std). A top score below the floor is relative
evidence: it orders candidates but says nothing about whether the answer exists. Above the
floor it counts as absolute evidence. The floor is a property of the negatives you fed it:
negatives from another domain give a low floor that looks safe and is not. Calibrate on
negatives that resemble the positives (a document deliberately left out of the index and
queried, a sibling product whose manual is missing). With n = 0 the floor is +inf; with
n = 3 it is arithmetic, not evidence, so n is always printed beside it.

## The conformal floor (`stats.conformal_quantile`, `conformal_floor`)

Given the scores of n calibration true matches and a miss rate alpha, take nonconformity
= −score and the threshold

```
q = the ceil((n + 1)(1 − alpha))-th smallest nonconformity,   floor = −q
```

For a new true match exchangeable with the calibration set, its score is at or above the
floor with probability at least 1 − alpha. The guarantee is marginal, over the randomness
of the calibration set; `tests/test_stats.py` simulates it. When the index exceeds n there
is no finite threshold with that guarantee, and rankscope returns +inf (floor −inf: accept
everything). At alpha = 0.1:

| n | index | the floor is |
| ---: | ---: | :--- |
| 1 to 8 | > n | undefined (`conformal_min_n(0.1)` = 9) |
| 9 to 18 | n | the smallest calibration score |
| 19 to 28 | n − 1 | the second smallest |
| 30 | 28 | a quantile at last |

A conformal floor bounds **misses** and says nothing about false accepts. On the synthetic
set it keeps FNIR at 32 percent and lets 53 percent of negatives through. Use it to know
how low you may set a floor before losing true matches, not as the floor itself.

## The gate, FAR and FNIR (`gate.py`)

```
no candidates                         -> abstain
top score < floor                     -> abstain (below_floor)
top score − second score < margin     -> abstain (margin)
otherwise                             -> accept the top hit
```

A positive ends as `accept_correct`, `accept_wrong` or `abstain`; a negative as
`false_accept` or `reject`.

```
FAR  (FPIR) = false_accept / negatives
FNIR        = (accept_wrong + abstain) / positives
```

A wrong accept counts as a miss, the 1:N identification convention. `sweep` evaluates
every floor the observed top scores suggest; `operating_point` picks the lowest floor whose
FPIR is within a target. Done on calibration negatives that is a threshold; done on the
negatives you report it is tuning. Both are printed so the difference stays visible.

## Paired bootstrap (`stats.paired_bootstrap`)

Two systems scored on the same queries give one delta per query. Resample the queries with
replacement 10 000 times, average the deltas each time, and read off the 95 percent
interval and P(delta > 0). `compare` calls it better only when P >= 0.95 and the mean delta
clears a minimum effect size you choose, because with enough queries a delta of 0.001
becomes "significant" and still means nothing.

## Verdicts (`verdict.py`)

A rule is a dotted path into the report, `>=` or `<=`, and a threshold. It passes on the
**point** when the value clears the threshold and on the **interval** when the conservative
end of the Wilson interval does. Write the rules file before the run. The tool cannot stop
you editing it afterwards; the discipline can.
