"""Paired comparison of two models, and intervals for a single model's score.

The default method is the task-clustered t interval: runs are averaged within
each task, and the interval is built from the per-task values (scores, or paired
differences on the tasks both models attempted). Bootstrap methods resample tasks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from evalstats import validity
from evalstats.intervals import (
    Interval,
    cluster_wilson_interval,
    interval_for_mean,
    mean_se,
    mover_wilson_interval,
    mover_wilson_p_value,
)
from evalstats.schema import EvalResults


@dataclass(frozen=True)
class ScoreInterval:
    """A model's benchmark score with its interval."""

    model: str
    n_tasks: int
    runs_per_task: float
    interval: Interval
    se: float


@dataclass(frozen=True)
class PairedComparison:
    """Difference in benchmark score ``model_a - model_b`` on shared tasks.

    ``score_a`` and ``score_b`` are the scores on the shared tasks only, which can
    differ from each model's overall score when the task sets differ;
    ``unpaired_tasks_a`` and ``unpaired_tasks_b`` count the tasks left out.
    """

    model_a: str
    model_b: str
    n_tasks: int
    score_a: float
    score_b: float
    difference: Interval
    se: float
    t_stat: float
    p_value: float
    unpaired_tasks_a: int
    unpaired_tasks_b: int

    @property
    def supported(self) -> bool:
        """True if the interval excludes zero (the ordering is supported at its level)."""
        return self.difference.lo > 0 or self.difference.hi < 0


def score_interval(
    results: EvalResults,
    model: str,
    *,
    method: str = "cluster_t",
    level: float = 0.95,
    n_boot: int = 10_000,
    seed: int | None = None,
) -> ScoreInterval:
    """Interval for one model's benchmark score (mean over tasks of task means).

    ``method`` is ``cluster_t`` (default), ``bootstrap``, ``bca``, or ``cluster_wilson``
    (a Wilson interval at the design-effect sample size, for success rates near 0 or 1;
    see :func:`evalstats.intervals.cluster_wilson_interval`).
    """
    scores = results.task_scores(model)
    runs = results.run_counts(model)
    values = scores.to_numpy(dtype=np.float64)
    validity.check("score_interval", method, int(values.size))
    if method == "cluster_wilson":
        counts = runs.reindex(scores.index).to_numpy(dtype=np.float64)
        interval = cluster_wilson_interval(values, counts, level)
    else:
        interval = interval_for_mean(values, method=method, level=level, n_boot=n_boot, seed=seed)
    return ScoreInterval(
        model=model,
        n_tasks=int(values.size),
        runs_per_task=float(runs.mean()),
        interval=interval,
        se=mean_se(values),
    )


def paired_difference(
    results: EvalResults,
    model_a: str,
    model_b: str,
    *,
    method: str = "cluster_t",
    level: float = 0.95,
    n_boot: int = 10_000,
    seed: int | None = None,
) -> PairedComparison:
    """Paired comparison on the tasks both models attempted.

    ``method`` is ``cluster_t`` (default), ``bootstrap``, ``bca``, or ``mover_wilson`` (for
    success rates; see :func:`evalstats.intervals.mover_wilson_interval`). The p-value is the
    two-sided paired t-test on per-task differences, except with ``mover_wilson``, whose
    p-value inverts its interval (:func:`evalstats.intervals.mover_wilson_p_value`); ``se``
    and ``t_stat`` are always the paired t quantities.
    """
    if model_a == model_b:
        raise ValueError("compare two different models")
    pair = results.paired(model_a, model_b)
    diffs = pair.differences
    validity.check("paired_difference", method, int(diffs.size))
    mover = method == "mover_wilson"
    if mover:
        args = (pair.a, pair.runs_a.astype(np.float64), pair.b, pair.runs_b.astype(np.float64))
        interval = mover_wilson_interval(*args, level=level)
    else:
        interval = interval_for_mean(diffs, method=method, level=level, n_boot=n_boot, seed=seed)
    se = mean_se(diffs)
    if diffs.size < 2:
        t_stat, p_value = math.nan, math.nan
    elif se == 0.0:
        t_stat = math.copysign(math.inf, float(diffs.mean())) if diffs.mean() != 0 else 0.0
        p_value = 0.0 if diffs.mean() != 0 else 1.0
    else:
        t_stat = float(diffs.mean() / se)
        p_value = float(2 * stats.t.sf(abs(t_stat), df=diffs.size - 1))
    if mover and diffs.size >= 2:
        p_value = mover_wilson_p_value(*args)
    n_a = len(results.task_scores(model_a))
    n_b = len(results.task_scores(model_b))
    return PairedComparison(
        model_a=model_a,
        model_b=model_b,
        n_tasks=len(pair.tasks),
        score_a=float(pair.a.mean()),
        score_b=float(pair.b.mean()),
        difference=interval,
        se=se,
        t_stat=t_stat,
        p_value=p_value,
        unpaired_tasks_a=n_a - len(pair.tasks),
        unpaired_tasks_b=n_b - len(pair.tasks),
    )


@dataclass(frozen=True)
class ContrastDifference:
    """Difference of two paired contrasts, ``(a1 - a0) - (b1 - b0)``, paired by task.

    For example the harness effect of one model minus that of another: does a scaffold
    help one model more than the other? ``n_tasks`` counts the tasks with runs under all
    four labels; ``labels`` is ``(a1, a0, b1, b0)``.
    """

    labels: tuple[str, str, str, str]
    n_tasks: int
    difference: Interval
    se: float
    t_stat: float
    p_value: float

    @property
    def supported(self) -> bool:
        """True if the interval excludes zero (the contrasts differ at its level)."""
        return self.difference.lo > 0 or self.difference.hi < 0


def contrast_difference(
    results: EvalResults,
    a1: str,
    a0: str,
    b1: str,
    b0: str,
    *,
    method: str = "cluster_t",
    level: float = 0.95,
    n_boot: int = 10_000,
    seed: int | None = None,
) -> ContrastDifference:
    """Interval for ``(a1 - a0) - (b1 - b0)``, the difference of two contrasts, by task.

    ``results`` labels each run by one of the four groups in its ``model`` column (for
    example ``"m1:native"``). On every task with runs in all four groups, the per-task
    value is the difference of the two groups' per-task contrasts (each a difference of
    per-task means); the interval is for the mean of these values (``cluster_t`` by
    default, or a bootstrap method of :func:`evalstats.intervals.interval_for_mean`), and
    the p-value is the two-sided one-sample t-test on them.
    """
    labels = (a1, a0, b1, b0)
    if len(set(labels)) != 4:
        raise ValueError("the four groups must be different")
    scores = [results.task_scores(label) for label in labels]
    tasks = sorted(set.intersection(*(set(s.index) for s in scores)))
    if not tasks:
        raise ValueError("no task has runs in all four groups")
    v = [s.reindex(tasks).to_numpy(dtype=np.float64) for s in scores]
    values = (v[0] - v[1]) - (v[2] - v[3])
    validity.check("contrast_difference", method, int(values.size))
    interval = interval_for_mean(values, method=method, level=level, n_boot=n_boot, seed=seed)
    se = mean_se(values)
    if values.size < 2:
        t_stat, p_value = math.nan, math.nan
    elif se == 0.0:
        mean = float(values.mean())
        t_stat = math.copysign(math.inf, mean) if mean != 0 else 0.0
        p_value = 0.0 if mean != 0 else 1.0
    else:
        t_stat = float(values.mean() / se)
        p_value = float(2 * stats.t.sf(abs(t_stat), df=values.size - 1))
    return ContrastDifference(
        labels=labels,
        n_tasks=len(tasks),
        difference=interval,
        se=se,
        t_stat=t_stat,
        p_value=p_value,
    )
