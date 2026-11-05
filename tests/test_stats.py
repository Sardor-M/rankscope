import math
import random

import pytest

from rankscope import stats


def test_wilson_known_values():
    lo, hi = stats.wilson(45, 50)
    assert 0.78 < lo < 0.80 and 0.95 < hi < 0.97
    lo, hi = stats.wilson(0, 30)
    assert lo == 0.0 and 0.11 < hi < 0.12
    assert stats.wilson(0, 3)[1] > 0.55
    assert stats.wilson(0, 0) == (0.0, 0.0)
    assert stats.wilson(12, 12)[1] == 1.0


def test_rule_of_three_and_negatives_needed():
    assert stats.rule_of_three(30) == pytest.approx(0.1)
    assert stats.negatives_for(0.05) == 73
    assert stats.wilson(0, 72)[1] > 0.05 >= stats.wilson(0, 73)[1]
    with pytest.raises(ValueError):
        stats.negatives_for(0)


def test_null_floor():
    out = stats.null_floor([0.5, 0.6, 0.7])
    assert out["n"] == 3 and out["mean"] == pytest.approx(0.6) and out["std"] == pytest.approx(0.1)
    assert out["floor"] == pytest.approx(0.9)
    assert stats.null_floor([0.4])["floor"] == pytest.approx(0.4)
    assert stats.null_floor([])["floor"] == math.inf


def test_conformal_quantile_and_small_n():
    values = [i / 30 for i in range(1, 31)]
    assert stats.conformal_quantile(values, 0.1) == pytest.approx(28 / 30)
    assert stats.conformal_quantile(values, 0.5) == pytest.approx(16 / 30)
    assert stats.conformal_min_n(0.1) == 9
    assert stats.conformal_quantile([0.2] * 8, 0.1) == math.inf
    assert stats.conformal_quantile(list(range(9)), 0.1) == 8
    assert stats.conformal_quantile(list(range(18)), 0.1) == 17
    assert stats.conformal_quantile(list(range(19)), 0.1) == 17
    assert stats.conformal_quantile([], 0.1) == math.inf
    assert stats.conformal_floor([0.9, 0.8], 0.1) == -math.inf
    with pytest.raises(ValueError):
        stats.conformal_quantile([0.1], 1.5)


def test_conformal_floor_holds_its_miss_rate():
    rng = random.Random(7)
    misses = 0
    trials = 400
    for _ in range(trials):
        calibration = [rng.gauss(0.7, 0.1) for _ in range(40)]
        floor = stats.conformal_floor(calibration, alpha=0.1)
        misses += rng.gauss(0.7, 0.1) < floor
    assert 0.05 < misses / trials < 0.15


def test_paired_bootstrap():
    baseline = [0.1, 0.2, 0.3, 0.4] * 10
    candidate = [b + 0.1 for b in baseline]
    out = stats.paired_bootstrap(baseline, candidate, resamples=2000, seed=1)
    assert out["delta"] == pytest.approx(0.1) and out["p_better"] == 1.0 and out["ci_low"] > 0.09
    same = stats.paired_bootstrap(baseline, baseline, resamples=500, seed=1)
    assert same["delta"] == 0.0 and same["p_better"] == 0.0
    with pytest.raises(ValueError):
        stats.paired_bootstrap([1.0], [1.0, 2.0])
    lo, hi = stats.bootstrap_mean([0.0, 1.0] * 20, resamples=500, seed=2)
    assert lo < 0.5 < hi


def test_bootstrap_means_shares_draws_without_changing_any_interval():
    def alone(values, resamples, seed):
        # bootstrap_mean as it was before draws were shared
        rng = random.Random(seed)
        n = len(values)
        means = sorted(sum(rng.choices(values, k=n)) / n for _ in range(resamples))
        return (means[int(0.025 * resamples)], means[min(int(0.975 * resamples), resamples - 1)])

    rng = random.Random(5)
    columns = [[rng.random() for _ in range(n)] for n in (30, 30, 7, 30, 1, 7)]
    assert stats.bootstrap_means(columns + [[]], resamples=300, seed=4) == \
        [alone(values, 300, 4) for values in columns] + [(0.0, 0.0)]
    assert stats.bootstrap_mean(columns[2], resamples=300, seed=4) == alone(columns[2], 300, 4)
