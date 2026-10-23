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

from evalstats.intervals import Interval, interval_for_mean, mean_se
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
    """Interval for one model's benchmark score (mean over tasks of task means)."""
    scores = results.task_scores(model)
    runs = results.run_counts(model)
    values = scores.to_numpy(dtype=np.float64)
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

    The p-value is the two-sided paired t-test on per-task differences, whatever
    interval method is chosen.
    """
    if model_a == model_b:
        raise ValueError("compare two different models")
    pair = results.paired(model_a, model_b)
    diffs = pair.differences
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
