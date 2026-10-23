"""Power and minimum detectable effects (MDE) for paired model comparisons.

Model of a paired comparison on T tasks with k runs per model per task: the
per-task difference of mean outcomes has variance ``between + within / k``, where
``between`` is the variance across tasks of the true per-task difference and
``within`` is the run-level variance of one run of A plus one run of B within a
task. The score difference averages T such tasks. Formulas use the normal
approximation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from evalstats.schema import EvalResults


@dataclass(frozen=True)
class PairedDesign:
    """Variance inputs for a paired comparison (see the module docstring)."""

    between_task_var: float
    within_task_var: float

    def __post_init__(self) -> None:
        """Reject negative or non-finite variances."""
        for name in ("between_task_var", "within_task_var"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}")

    def se(self, n_tasks: int, runs_per_task: int = 1) -> float:
        """Standard error of the score difference for T tasks and k runs per task."""
        if n_tasks < 1 or runs_per_task < 1:
            raise ValueError("n_tasks and runs_per_task must be positive")
        var = (self.between_task_var + self.within_task_var / runs_per_task) / n_tasks
        return math.sqrt(var)


def _z(alpha: float, power: float, two_sided: bool) -> tuple[float, float]:
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha and power must be in (0, 1)")
    z_alpha = float(stats.norm.ppf(1 - alpha / 2 if two_sided else 1 - alpha))
    return z_alpha, float(stats.norm.ppf(power))


def mde(
    design: PairedDesign,
    n_tasks: int,
    runs_per_task: int = 1,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    two_sided: bool = True,
) -> float:
    """Smallest true score difference detected with the given power and level."""
    z_alpha, z_power = _z(alpha, power, two_sided)
    return (z_alpha + z_power) * design.se(n_tasks, runs_per_task)


def power_at(
    effect: float,
    design: PairedDesign,
    n_tasks: int,
    runs_per_task: int = 1,
    *,
    alpha: float = 0.05,
    two_sided: bool = True,
) -> float:
    """Probability of rejecting no difference when the true difference is ``effect``."""
    z_alpha, _ = _z(alpha, 0.5, two_sided)
    se = design.se(n_tasks, runs_per_task)
    if se == 0.0:
        return 1.0 if effect != 0 else alpha
    shift = abs(effect) / se
    if two_sided:
        return float(stats.norm.sf(z_alpha - shift) + stats.norm.cdf(-z_alpha - shift))
    return float(stats.norm.sf(z_alpha - shift))


def tasks_needed(
    effect: float,
    design: PairedDesign,
    runs_per_task: int = 1,
    *,
    alpha: float = 0.05,
    power: float = 0.8,
    two_sided: bool = True,
) -> int:
    """Smallest number of tasks whose MDE is at most ``effect``."""
    if effect <= 0 or not math.isfinite(effect):
        raise ValueError("effect must be positive and finite")
    z_alpha, z_power = _z(alpha, power, two_sided)
    per_task = design.between_task_var + design.within_task_var / runs_per_task
    return max(1, math.ceil(((z_alpha + z_power) / effect) ** 2 * per_task - 1e-12))


def design_from_results(results: EvalResults, model_a: str, model_b: str) -> PairedDesign:
    """Method-of-moments variance inputs estimated from observed results.

    ``within`` is the mean over tasks of the unbiased run-level variances of A and
    B (tasks where both models have at least 2 runs); ``between`` is the variance
    of per-task differences minus their average sampling variance, floored at 0.
    """
    pair = results.paired(model_a, model_b)
    if len(pair.tasks) < 2:
        raise ValueError("need at least 2 shared tasks")
    frame = results.frame
    var_by_task = (
        frame.loc[frame["model"].isin([model_a, model_b])]
        .groupby(["model", "task"])["outcome"]
        .var(ddof=1)
    )
    within_terms = []
    sampling_terms = []
    for i, task in enumerate(pair.tasks):
        ka, kb = int(pair.runs_a[i]), int(pair.runs_b[i])
        if ka < 2 or kb < 2:
            continue
        va = float(var_by_task.loc[(model_a, task)])
        vb = float(var_by_task.loc[(model_b, task)])
        within_terms.append(va + vb)
        sampling_terms.append(va / ka + vb / kb)
    if not within_terms:
        raise ValueError("need at least 2 runs per model on some shared tasks")
    total = float(np.var(pair.differences, ddof=1))
    between = max(0.0, total - float(np.mean(sampling_terms)))
    return PairedDesign(between_task_var=between, within_task_var=float(np.mean(within_terms)))
