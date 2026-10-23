import math

import numpy as np
import pytest
import statsmodels.api as sm
from scipy import stats

from evalstats import (
    Interval,
    bca_interval,
    cluster_robust_mean,
    interval_for_mean,
    percentile_interval,
    t_interval,
)
from evalstats.intervals import bootstrap_means, mean_se


def test_t_interval_matches_scipy() -> None:
    x = np.array([0.2, 0.5, 0.9, 0.4, 0.7, 0.1])
    got = t_interval(x, 0.9)
    lo, hi = stats.t.interval(0.9, df=x.size - 1, loc=x.mean(), scale=stats.sem(x))
    assert got.lo == pytest.approx(lo) and got.hi == pytest.approx(hi)
    assert got.method == "cluster_t" and got.contains(x.mean())
    assert got.width == pytest.approx(hi - lo)


def test_t_interval_single_value_is_unbounded() -> None:
    got = t_interval([0.5])
    assert math.isinf(got.lo) and math.isinf(got.hi) and mean_se([0.5]) == 0.0


def test_cluster_robust_matches_statsmodels() -> None:
    rng = np.random.default_rng(1)
    clusters = np.repeat(np.arange(25), rng.integers(1, 6, size=25))
    y = (rng.random(clusters.size) < 0.4).astype(float)
    interval, se = cluster_robust_mean(y, clusters, 0.95)
    fit = sm.OLS(y, np.ones_like(y)).fit(cov_type="cluster", cov_kwds={"groups": clusters})
    assert interval.estimate == pytest.approx(float(fit.params[0]))
    assert se == pytest.approx(float(fit.bse[0]), rel=1e-10)


def test_cluster_robust_singletons_equal_classic_se() -> None:
    y = np.array([0.0, 1.0, 1.0, 0.0, 1.0])
    _, se = cluster_robust_mean(y, np.arange(5))
    assert se == pytest.approx(mean_se(y))


def test_cluster_robust_errors_and_single_cluster() -> None:
    with pytest.raises(ValueError, match="same length"):
        cluster_robust_mean([1.0, 0.0], [1])
    interval, se = cluster_robust_mean([1.0, 0.0], [7, 7])
    assert math.isinf(se) and math.isinf(interval.hi)


def test_bootstrap_reproducible_and_reasonable() -> None:
    x = np.random.default_rng(2).random(40)
    a = percentile_interval(x, n_boot=4000, seed=5)
    b = percentile_interval(x, n_boot=4000, seed=5)
    assert a == b and a.method == "bootstrap" and a.lo < x.mean() < a.hi
    t = t_interval(x)
    assert a.width == pytest.approx(t.width, rel=0.15)


def test_bootstrap_means_chunking_and_validation() -> None:
    reps = bootstrap_means([1.0, 2.0, 3.0], 4500, np.random.default_rng(0))
    assert reps.shape == (4500,) and reps.min() >= 1 and reps.max() <= 3
    with pytest.raises(ValueError, match="positive"):
        bootstrap_means([1.0], 0, np.random.default_rng(0))


def test_bca_degenerate_and_skewed() -> None:
    flat = bca_interval([0.3, 0.3, 0.3])
    assert (flat.lo, flat.hi) == (0.3, 0.3)
    x = np.random.default_rng(3).exponential(size=60)
    got = bca_interval(x, n_boot=4000, seed=1)
    assert got.method == "bca" and got.lo < x.mean() < got.hi
    # BCa shifts right relative to the t interval for right-skewed data.
    assert got.hi - x.mean() > x.mean() - got.lo


@pytest.mark.parametrize("bad", [[], [[1.0, 2.0]], [1.0, float("nan")]])
def test_vector_validation(bad: list[object]) -> None:
    with pytest.raises(ValueError):
        t_interval(bad)


def test_level_and_method_validation() -> None:
    with pytest.raises(ValueError, match="level"):
        t_interval([0.1, 0.2], level=1.0)
    with pytest.raises(ValueError, match="unknown method"):
        interval_for_mean([0.1, 0.2], method="wald")
    got = interval_for_mean([0.1, 0.2, 0.4], method="bootstrap", n_boot=500, seed=0)
    assert isinstance(got, Interval) and got.level == 0.95
