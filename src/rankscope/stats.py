"""Small-sample statistics: Wilson intervals, null floors, conformal quantiles, paired bootstrap."""

from __future__ import annotations

import math
import random
from typing import Sequence

Z_95 = 1.96


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def sample_std(values: Sequence[float]) -> float:
    n = len(values)
    if n < 2:
        return 0.0
    centre = mean(values)
    return math.sqrt(sum((v - centre) ** 2 for v in values) / (n - 1))


def wilson(successes: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """Wilson score interval for a proportion; (0, 0) when n is 0."""
    if n <= 0:
        return (0.0, 0.0)
    p = successes / n
    denominator = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denominator
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return (max(0.0, centre - half), min(1.0, centre + half))


def rule_of_three(n: int) -> float:
    """Upper 95 percent bound on a rate observed zero times in n trials."""
    return 3.0 / n if n > 0 else 1.0


def negatives_for(bound: float, z: float = Z_95) -> int:
    """Smallest n whose Wilson upper limit after zero events is at or below `bound`."""
    if not 0 < bound < 1:
        raise ValueError("bound must be in (0, 1)")
    n = 1
    while wilson(0, n, z)[1] > bound:
        n += 1
    return n


def null_floor(scores: Sequence[float], sigmas: float = 3.0) -> dict:
    """mean + sigmas * std over the best score each out-of-corpus query reached; +inf when unmeasured."""
    n = len(scores)
    if n == 0:
        return {"n": 0, "mean": 0.0, "std": 0.0, "floor": math.inf}
    centre, spread = mean(scores), sample_std(scores)
    return {"n": n, "mean": centre, "std": spread, "floor": centre + sigmas * spread}


def conformal_index(n: int, alpha: float) -> int:
    return math.ceil((n + 1) * (1 - alpha))


def conformal_min_n(alpha: float) -> int:
    """Smallest calibration size at which the ceil((n+1)(1-alpha))-th value exists."""
    n = 1
    while conformal_index(n, alpha) > n:
        n += 1
    return n


def conformal_quantile(nonconformity: Sequence[float], alpha: float) -> float:
    """The ceil((n+1)(1-alpha))-th smallest value; +inf when n is too small for any finite bound."""
    if not 0 < alpha < 1:
        raise ValueError("alpha must be in (0, 1)")
    values = sorted(nonconformity)
    n = len(values)
    index = conformal_index(n, alpha)
    if n == 0 or index > n:
        return math.inf
    return values[index - 1]


def conformal_floor(true_scores: Sequence[float], alpha: float) -> float:
    """Score floor that misses at most an alpha share of true matches; -inf when uncalibratable."""
    return -conformal_quantile([-s for s in true_scores], alpha)


def paired_bootstrap(
    baseline: Sequence[float], candidate: Sequence[float], *, resamples: int = 10_000, seed: int = 0
) -> dict:
    """Bootstrap the mean per-query delta (candidate - baseline) over the same queries."""
    if len(baseline) != len(candidate):
        raise ValueError("paired bootstrap needs one value per query from each system")
    n = len(baseline)
    if n == 0:
        raise ValueError("no queries to compare")
    deltas = [c - b for b, c in zip(baseline, candidate)]
    rng = random.Random(seed)
    means = []
    wins = 0
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += deltas[rng.randrange(n)]
        resampled = total / n
        means.append(resampled)
        wins += resampled > 0.0
    means.sort()
    return {
        "n": n,
        "resamples": resamples,
        "mean_baseline": mean(baseline),
        "mean_candidate": mean(candidate),
        "delta": mean(deltas),
        "ci_low": means[int(0.025 * resamples)],
        "ci_high": means[min(int(0.975 * resamples), resamples - 1)],
        "p_better": wins / resamples,
    }


def bootstrap_mean(values: Sequence[float], *, resamples: int = 10_000, seed: int = 0) -> tuple[float, float]:
    """Percentile interval for a mean; (0, 0) when there are no values."""
    n = len(values)
    if n == 0:
        return (0.0, 0.0)
    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        total = 0.0
        for _ in range(n):
            total += values[rng.randrange(n)]
        means.append(total / n)
    means.sort()
    return (means[int(0.025 * resamples)], means[min(int(0.975 * resamples), resamples - 1)])
