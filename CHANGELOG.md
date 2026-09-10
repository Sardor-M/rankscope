# Changelog

## 0.2.0 (2026-09-10)

- `sweep` is linear in the number of queries: scores are sorted once and each floor is a
  bisection. On 4,500 queries an `evaluate --sweep` that took 20 seconds takes well under
  one.
- The paired bootstrap and `bootstrap_mean` resample with `random.choices`, about three
  times faster (4.2 s to 1.4 s for 10,000 resamples over 3,000 queries). The random
  stream changed, so seeds are not comparable with 0.1.0.
- `evaluate` and `calibrate` count judged queries missing from each lane and say so in
  the header; a query absent from a run file is still scored as a miss, now visibly.
- `--depth N` on `evaluate`, `calibrate` and `compare` cuts every lane to the same depth;
  the header prints each lane's depth and warns when they differ.
- `calibrate --conservative` chooses the FPIR-bounded floor on the Wilson upper bound of
  FPIR rather than its point estimate, and says how many negatives a finite floor would
  need when none qualifies. Thresholds record the rule.
- `compare` no longer recomputes fusion twice per query.

## 0.1.0 (2026-09-08)

First release.

- `evaluate`: hit@k with Wilson intervals, recall@k, MRR, nDCG@k per stratum at doc and
  group level; miss attribution in a window; judged negatives; RRF and convex fusion; gates
  with FAR and FNIR; text, Markdown and JSON output.
- `calibrate`: convex weights by grid search, null floor, conformal floor, FPIR-bounded
  operating floor, on a separate qrels file.
- `compare`: paired bootstrap with a minimum effect size.
- `verdict`: pre-registered rules judged on point and on interval.
- `fuse`, `synth`, `goldens`, and a JavaScript reference implementation checked against
  the goldens.
