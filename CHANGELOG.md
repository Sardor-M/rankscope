# Changelog

## 0.4.0 (2026-09-15)

- Gates can require a mutual nearest neighbour (the query within rank r among the queries
  that retrieved its top document, built from the lanes); the sweep honours it, and
  `--mutual` applies it to manual gates.
- `margin_sweep`, an operating margin in the thresholds, and a `margin-<fpir>` gate.
- Gate results per stratum; a per-stratum table when there is more than one.
- `calibrate --objective ndcg`; thresholds gates carry margin and mutual.
- `evaluate --cap N` and `--document-first` add re-ranked variants beside the base ranking.
- Verdict rules with `against` compare two paths.
- The JavaScript kernel and the goldens cover `minmax` and `convex`.
- `docs/workbook.html`, the interactive workbook, ships with the repository; `docs/beir.md`
  runs the loop on a public dataset.

## 0.3.0 (2026-09-15)

- `.json` lanes and qrels in the `{query: {doc: score}}` and `{query: {doc: grade}}`
  convention pytrec_eval and BEIR use; `Lane.from_dict`, `Lane.to_dict`, `Qrels.from_dict`.
- Bootstrap intervals on MRR, recall at the deepest k and nDCG at 10, per stratum, shown
  inline and stored as `<metric>_ci`; `--resamples` (default 1000, 0 to skip) and `--seed`.
- `diff`: what moved between two reports, judged on interval overlap; `--fail-on-regression`
  for CI.
- `evaluate --format latex` and `--per-query FILE.csv`.

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
