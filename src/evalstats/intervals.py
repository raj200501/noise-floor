"""Confidence intervals: t intervals, cluster-robust means, and task bootstraps.

All resampling is over tasks (the cluster), never over individual runs, so the
intervals account for the correlation of runs within a task.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats

FloatArray = npt.NDArray[np.float64]

BOOTSTRAP_CHUNK = 2000


@dataclass(frozen=True)
class Interval:
    """A point estimate with a two-sided confidence interval."""

    estimate: float
    lo: float
    hi: float
    level: float
    method: str

    @property
    def width(self) -> float:
        """Interval width ``hi - lo``."""
        return self.hi - self.lo

    def contains(self, value: float) -> bool:
        """True if ``lo <= value <= hi``."""
        return self.lo <= value <= self.hi


def _check_level(level: float) -> None:
    if not 0 < level < 1:
        raise ValueError(f"level must be in (0, 1), got {level}")


def _as_vector(values: npt.ArrayLike, name: str = "values") -> FloatArray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} has non-finite entries")
    return arr


def mean_se(values: npt.ArrayLike) -> float:
    """Standard error of the mean, ``sd / sqrt(n)`` with ``ddof=1`` (0 if n == 1)."""
    arr = _as_vector(values)
    if arr.size < 2:
        return 0.0
    return float(arr.std(ddof=1) / math.sqrt(arr.size))


def t_interval(values: npt.ArrayLike, level: float = 0.95) -> Interval:
    """Student t interval for the mean of independent values (n - 1 degrees of freedom).

    Applied to per-task scores or per-task paired differences this is the
    task-clustered interval of Miller (2024): runs are averaged within a task first.
    """
    _check_level(level)
    arr = _as_vector(values)
    est = float(arr.mean())
    if arr.size < 2:
        return Interval(est, -math.inf, math.inf, level, "cluster_t")
    se = mean_se(arr)
    q = float(stats.t.ppf(0.5 + level / 2, df=arr.size - 1))
    return Interval(est, est - q * se, est + q * se, level, "cluster_t")


def cluster_wilson_interval(
    task_means: npt.ArrayLike, runs: npt.ArrayLike, level: float = 0.95
) -> Interval:
    """Wilson interval for a success rate at its design-effect sample size.

    ``task_means`` are per-task success rates in [0, 1] and ``runs`` the runs behind each;
    the estimate is their mean (every task weighs the same). Its variance is the larger of
    the task-clustered variance ``sd**2 / T`` and the binomial variance of the same
    estimator, ``p (1 - p) mean(1 / runs) / T``, so the effective sample size
    ``p (1 - p) / variance`` is at most the run count. The Wilson interval at that sample
    size uses the t quantile with ``T - 1`` degrees of freedom. Unlike the t interval it
    stays inside [0, 1] and keeps its width when most tasks are all-success or all-failure.
    """
    _check_level(level)
    means = _as_vector(task_means, "task_means")
    counts = _as_vector(runs, "runs")
    if counts.shape != means.shape or np.any(counts < 1):
        raise ValueError("runs must give a positive run count for every task")
    if np.any(means < 0) or np.any(means > 1):
        raise ValueError("cluster_wilson needs success rates in [0, 1]")
    n_tasks = means.size
    est = float(means.mean())
    if n_tasks < 2:
        return Interval(est, 0.0, 1.0, level, "cluster_wilson")
    binomial_unit = float(np.mean(1.0 / counts)) / n_tasks  # times p (1 - p)
    clustered = float(means.var(ddof=1)) / n_tasks
    pq = est * (1 - est)
    n_eff = pq / max(clustered, pq * binomial_unit) if pq > 0 else 1.0 / binomial_unit
    z = float(stats.t.ppf(0.5 + level / 2, df=n_tasks - 1))
    denom = 1 + z * z / n_eff
    center = (est + z * z / (2 * n_eff)) / denom
    half = z * math.sqrt(pq / n_eff + z * z / (4 * n_eff * n_eff)) / denom
    return Interval(est, max(0.0, center - half), min(1.0, center + half), level, "cluster_wilson")


def mover_wilson_interval(
    a_means: npt.ArrayLike,
    a_runs: npt.ArrayLike,
    b_means: npt.ArrayLike,
    b_runs: npt.ArrayLike,
    level: float = 0.95,
) -> Interval:
    """Interval for the paired difference of two success rates, ``a - b``, on shared tasks.

    MOVER (method of variance estimates recovery): each rate gets its
    :func:`cluster_wilson_interval`, and the two are combined with ``r``, the correlation of
    the per-task rates (0 when either is constant), as
    ``d - sqrt((p_a - l_a)**2 + (u_b - p_b)**2 - 2 r (p_a - l_a)(u_b - p_b))`` for the lower
    end and symmetrically for the upper end. It keeps nominal coverage at success rates near
    0 or 1, where the paired t interval on per-task differences under-covers.
    """
    am, bm = _as_vector(a_means, "a_means"), _as_vector(b_means, "b_means")
    if am.shape != bm.shape:
        raise ValueError("a and b need one rate per shared task")
    ia = cluster_wilson_interval(am, a_runs, level)
    ib = cluster_wilson_interval(bm, b_runs, level)
    r = 0.0
    if am.size > 1 and float(am.std()) > 0 and float(bm.std()) > 0:
        r = float(np.corrcoef(am, bm)[0, 1])
    pa, pb = ia.estimate, ib.estimate
    d = pa - pb
    down_a, up_b = pa - ia.lo, ib.hi - pb
    up_a, down_b = ia.hi - pa, pb - ib.lo
    lo = d - math.sqrt(max(0.0, down_a**2 + up_b**2 - 2 * r * down_a * up_b))
    hi = d + math.sqrt(max(0.0, up_a**2 + down_b**2 - 2 * r * up_a * down_b))
    return Interval(d, max(-1.0, lo), min(1.0, hi), level, "mover_wilson")


def mover_wilson_p_value(
    a_means: npt.ArrayLike,
    a_runs: npt.ArrayLike,
    b_means: npt.ArrayLike,
    b_runs: npt.ArrayLike,
) -> float:
    """Two-sided p-value of "no difference" that matches :func:`mover_wilson_interval`.

    It is ``1 - L``, where ``L`` is the largest level whose interval excludes 0 (found by
    bisection), so the interval at level ``1 - alpha`` excludes 0 exactly when the p-value
    is below ``alpha``.
    """

    def excludes_zero(level: float) -> bool:
        iv = mover_wilson_interval(a_means, a_runs, b_means, b_runs, level)
        return iv.lo > 0 or iv.hi < 0

    lo, hi = 1e-9, 1 - 1e-9
    if not excludes_zero(lo):
        return 1.0
    if excludes_zero(hi):
        return 0.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if excludes_zero(mid):
            lo = mid
        else:
            hi = mid
    return float(1 - lo)


def cluster_robust_mean(
    values: npt.ArrayLike, clusters: npt.ArrayLike, level: float = 0.95
) -> tuple[Interval, float]:
    """Mean of observations with a CR1 cluster-robust standard error.

    Returns the interval (t with G - 1 degrees of freedom, G clusters) and the
    standard error. Every observation weighs the same, so tasks with more runs
    weigh more; use :func:`t_interval` on task means for equal task weights.
    """
    _check_level(level)
    y = _as_vector(values)
    groups = np.asarray(clusters)
    if groups.shape != y.shape:
        raise ValueError("values and clusters must have the same length")
    uniques, codes = np.unique(groups, return_inverse=True)
    n_groups = len(uniques)
    est = float(y.mean())
    if n_groups < 2:
        return Interval(est, -math.inf, math.inf, level, "cluster_robust_cr1"), math.inf
    scores = np.bincount(codes, weights=y - est, minlength=n_groups)
    var = n_groups / (n_groups - 1) * float(np.sum(scores**2)) / y.size**2
    se = math.sqrt(var)
    q = float(stats.t.ppf(0.5 + level / 2, df=n_groups - 1))
    return Interval(est, est - q * se, est + q * se, level, "cluster_robust_cr1"), se


def bootstrap_means(values: npt.ArrayLike, n_boot: int, rng: np.random.Generator) -> FloatArray:
    """Means of ``n_boot`` resamples (with replacement) of ``values``."""
    arr = _as_vector(values)
    if n_boot < 1:
        raise ValueError("n_boot must be positive")
    out = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, BOOTSTRAP_CHUNK):
        stop = min(start + BOOTSTRAP_CHUNK, n_boot)
        idx = rng.integers(0, arr.size, size=(stop - start, arr.size))
        out[start:stop] = arr[idx].mean(axis=1)
    return out


def percentile_interval(
    values: npt.ArrayLike, level: float = 0.95, n_boot: int = 10_000, seed: int | None = None
) -> Interval:
    """Percentile bootstrap interval for the mean, resampling the given values."""
    _check_level(level)
    arr = _as_vector(values)
    est = float(arr.mean())
    reps = bootstrap_means(arr, n_boot, np.random.default_rng(seed))
    lo, hi = np.quantile(reps, [0.5 - level / 2, 0.5 + level / 2])
    return Interval(est, float(lo), float(hi), level, "bootstrap")


def _jackknife_acceleration(arr: FloatArray, stat: Callable[[FloatArray], float]) -> float:
    n = arr.size
    loo = np.array([stat(np.delete(arr, i)) for i in range(n)], dtype=np.float64)
    dev = loo.mean() - loo
    denom = 6.0 * float(np.sum(dev**2)) ** 1.5
    return float(np.sum(dev**3)) / denom if denom > 0 else 0.0


def bca_interval(
    values: npt.ArrayLike, level: float = 0.95, n_boot: int = 10_000, seed: int | None = None
) -> Interval:
    """Bias-corrected and accelerated (BCa) bootstrap interval for the mean.

    Acceleration comes from the jackknife over the resampled units (tasks).
    """
    _check_level(level)
    arr = _as_vector(values)
    est = float(arr.mean())
    if arr.size < 2 or float(np.ptp(arr)) == 0.0:
        return Interval(est, est, est, level, "bca")
    reps = bootstrap_means(arr, n_boot, np.random.default_rng(seed))
    below = (np.sum(reps < est) + 0.5 * np.sum(reps == est)) / reps.size
    below = min(max(float(below), 1.0 / (2 * reps.size)), 1 - 1.0 / (2 * reps.size))
    z0 = float(stats.norm.ppf(below))
    accel = _jackknife_acceleration(arr, lambda x: float(x.mean()))
    alpha = (1 - level) / 2
    quantiles = []
    for z_alpha in (float(stats.norm.ppf(alpha)), float(stats.norm.ppf(1 - alpha))):
        adj = z0 + (z0 + z_alpha) / (1 - accel * (z0 + z_alpha))
        quantiles.append(float(stats.norm.cdf(adj)))
    lo, hi = np.quantile(reps, quantiles)
    return Interval(est, float(lo), float(hi), level, "bca")


INTERVAL_METHODS: dict[str, Callable[..., Interval]] = {
    "cluster_t": lambda v, level, n_boot, seed: t_interval(v, level),
    "bootstrap": percentile_interval,
    "bca": bca_interval,
}


def interval_for_mean(
    values: npt.ArrayLike,
    method: str = "cluster_t",
    level: float = 0.95,
    n_boot: int = 10_000,
    seed: int | None = None,
) -> Interval:
    """Interval for the mean of per-task values by the named method.

    Methods: ``cluster_t`` (t interval on task-level values), ``bootstrap``
    (percentile, resampling tasks), ``bca`` (bias-corrected and accelerated). For a
    success rate, :func:`cluster_wilson_interval` (``score_interval(method=
    "cluster_wilson")``) also needs each task's run count, so it is not listed here.
    """
    try:
        fn = INTERVAL_METHODS[method]
    except KeyError:
        raise ValueError(
            f"unknown method {method!r}; choose from {', '.join(INTERVAL_METHODS)}"
        ) from None
    return fn(values, level, n_boot, seed)
