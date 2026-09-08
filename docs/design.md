# Design notes

Decisions that would be expensive to reverse or confusing to rediscover.

## Two commands, two files

`calibrate` fits weights and floors; `evaluate` reports. They take separate qrels on
purpose and there is no flag to use one file for both. A threshold fitted and reported on
the same queries is optimism. Users who want the forbidden number can pass the same path
twice, and the tutorial asks them to do it once so they see the size of the lie.

## Negatives are judged queries with no positive grade

TREC qrels have no notion of "the answer is not in the corpus". rankscope reads a query
whose grades are all 0 as exactly that, and `--negatives` adds ids that have no qrels row.
This keeps existing qrels files valid: a query with no rows stays unjudged and ignored,
which is the TREC convention. Negatives are what make floors and the gate measurable; the
honest ones look like positives (a document left out of the index, a sibling product with
no manual), and `docs/concepts.md` says so.

## Hit rate and recall are both reported, under their own names

`hit@k` is "did any relevant document make the top k", a proportion with a Wilson
interval. `recall@k` is the TREC fraction of relevant documents retrieved. They coincide
for single-answer queries and differ otherwise; reporting one under the other's name is a
common source of numbers that do not reproduce across tools.

## Two levels, generic names

`doc` and `group`. A group is whatever the caller says it is: a document for passages, a
manual for figures, a product for pages. `--group-sep` covers the common `parent#child`
id convention; `--groups` covers everything else.

## The window is not the depth

Lanes may be 100 deep; the next stage usually sees 5 or 10. Attribution is computed in
the window and says so in its header, because "not surfaced" at 5 next to "hit@20" is the
kind of pair that gets misread.

## The conformal quantile returns infinity when n is too small

The ceil((n+1)(1−alpha))-th smallest value does not exist for n below
`conformal_min_n(alpha)`. Clamping to the maximum would print a number whose coverage is
n/(n+1), below the promised 1−alpha. rankscope returns +inf and prints `min_n` beside every
conformal result.

## The null floor is not capped

`mean + 3·std` is used on whatever score a lane emits: cosine, BM25, RRF, a convex sum.
Capping it to 1.0 would be a cosine-only convention and would turn "too few negatives"
into a silent "never accept". With no negatives the floor is +inf and says so.

## Fusion at evaluation fuses what it is given

RRF over lanes of depth 20 is a different function from RRF over depth 100 truncated to
20. rankscope fuses the lanes as loaded and keeps the whole union, and the report's depth
is the caller's responsibility. `fuse` writes the result as a TREC run so it can go back
through any other tool.

## Verdicts on point and on interval

A rules file judged only on point estimates rewards small samples; judged only on
intervals it rewards nothing until the sample is large. Both are printed, the summary
names the regime, and the exit code follows the point verdict so CI stays usable while the
interval column keeps everyone honest.

## Goldens, and a JavaScript reference

The arithmetic is small enough to port in an afternoon, and a service that gates its own
answers should compute the floor the same way the evaluation did. `goldens` fixes the
contract as text; `reference/js/` is the first port and the proof that the contract is
sufficient. Any language can join by emitting the same lines.

## What rankscope is not

Not a retriever, not a fusion library for production traffic, not a replacement for
`trec_eval` on the metrics it already computes, and not a statistics package. It is the
layer of questions that come after the metric.
