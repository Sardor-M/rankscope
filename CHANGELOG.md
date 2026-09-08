# Changelog

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
