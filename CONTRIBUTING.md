# Contributing

- The package has no runtime dependencies. Keep it that way; adding one needs a reason in
  `docs/design.md`.
- Every function in `stats.py`, `metrics.py`, `fusion.py` and `ranks.py` that changes
  behaviour needs a golden line in `goldens.py` and the matching change in
  `reference/js/rankscope.mjs`. Run `rankscope goldens --out tests/goldens.txt` and
  `node reference/js/check.mjs tests/goldens.txt` before opening a pull request.
- Tests: `pytest -q`. New behaviour gets a test with synthetic ids only.
- Docstrings are one line. Reasoning goes in `docs/`, not in comments.
