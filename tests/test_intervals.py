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


def test_cluster_wilson_interval() -> None:
    from evalstats.intervals import cluster_wilson_interval

    ones = cluster_wilson_interval(np.ones(30), np.full(30, 2.0))
    z = float(stats.t.ppf(0.975, 29))
    assert ones.estimate == 1.0 and ones.hi == 1.0
    assert ones.lo == pytest.approx(60 / (60 + z * z))  # Wilson at the run count
    means = np.array([0.0, 0.5, 1.0, 1.0, 0.5, 1.0, 0.0, 1.0])
    runs = np.full(8, 2.0)
    iv = cluster_wilson_interval(means, runs)
    p, t_tasks = means.mean(), 8
    var = max(means.var(ddof=1) / t_tasks, p * (1 - p) * 0.5 / t_tasks)
    n = p * (1 - p) / var
    q = float(stats.t.ppf(0.975, 7))
    center = (p + q * q / (2 * n)) / (1 + q * q / n)
    half = q * math.sqrt(p * (1 - p) / n + q * q / (4 * n * n)) / (1 + q * q / n)
    assert (iv.lo, iv.hi) == pytest.approx((center - half, center + half))
    assert iv.method == "cluster_wilson" and 0.0 <= iv.lo < p < iv.hi <= 1.0
    # A design effect below 1 is floored: never narrower than the binomial Wilson interval.
    even = cluster_wilson_interval(np.full(10, 0.5), np.full(10, 4.0))
    n40 = 40.0
    q10 = float(stats.t.ppf(0.975, 9))
    half40 = q10 * math.sqrt(0.25 / n40 + q10 * q10 / (4 * n40 * n40)) / (1 + q10 * q10 / n40)
    assert even.hi - even.lo == pytest.approx(2 * half40)
    assert cluster_wilson_interval(np.array([0.3]), np.array([5.0])).lo == 0.0
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        cluster_wilson_interval(np.array([0.5, 1.5]), np.array([2.0, 2.0]))
    with pytest.raises(ValueError, match="run count"):
        cluster_wilson_interval(np.array([0.5, 1.0]), np.array([2.0]))


def test_mover_wilson_interval_and_p_value() -> None:
    from evalstats.intervals import (
        cluster_wilson_interval,
        mover_wilson_interval,
        mover_wilson_p_value,
    )

    a = np.array([1.0, 1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0, 1.0])
    b = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0])
    ra, rb = np.ones(10), np.ones(10)
    iv = mover_wilson_interval(a, ra, b, rb)
    wa, wb = cluster_wilson_interval(a, ra), cluster_wilson_interval(b, rb)
    r = float(np.corrcoef(a, b)[0, 1])
    d = a.mean() - b.mean()
    lo = d - math.sqrt(
        (wa.estimate - wa.lo) ** 2
        + (wb.hi - wb.estimate) ** 2
        - 2 * r * (wa.estimate - wa.lo) * (wb.hi - wb.estimate)
    )
    hi = d + math.sqrt(
        (wa.hi - wa.estimate) ** 2
        + (wb.estimate - wb.lo) ** 2
        - 2 * r * (wa.hi - wa.estimate) * (wb.estimate - wb.lo)
    )
    assert iv.estimate == pytest.approx(d) and (iv.lo, iv.hi) == pytest.approx((lo, hi))
    assert iv.method == "mover_wilson" and -1.0 <= iv.lo < d < iv.hi <= 1.0
    p = mover_wilson_p_value(a, ra, b, rb)
    assert 0.0 < p < 1.0
    edge = mover_wilson_interval(a, ra, b, rb, level=1 - p)
    assert min(abs(edge.lo), abs(edge.hi)) == pytest.approx(0.0, abs=1e-6)
    assert mover_wilson_p_value(a, ra, a, ra) == 1.0
    assert mover_wilson_p_value(np.ones(30), np.ones(30), np.zeros(30), np.ones(30)) < 1e-6
    with pytest.raises(ValueError, match="shared task"):
        mover_wilson_interval(a, ra, b[:5], rb[:5])
