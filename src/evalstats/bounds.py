"""Policy sensitivity: scores and orderings under rules for ambiguous runs.

Runs whose ``status`` is one of the ambiguous statuses (by default
``infra_error``, ``timeout``, ``flagged``) can be scored by different rules:

* ``as_scored``: keep the outcome the source recorded;
* ``fail``: count them as failures (0);
* ``success``: count them as successes (1);
* ``exclude``: drop them (a task with no other runs drops out for that model).

Without assumptions about what an ambiguous run would have scored, a model's
score lies between the ``fail`` and ``success`` scores (worst-case bounds in the
style of Manski). Every named policy lands inside these bounds. Intervals for the
bounded score use the Imbens and Manski (2004) construction, which covers the
true score (not the whole set) with the stated probability. For a pair of models,
per-task differences are bounded the same way, and an ordering is called robust
when the lower bound of the difference is positive (identified) and its interval
excludes zero (statistically robust).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy import optimize, stats

from evalstats.schema import OUTCOME, EvalResults, SchemaError

FloatArray = npt.NDArray[np.float64]
AMBIGUOUS = ("infra_error", "timeout", "flagged")
POLICIES = ("as_scored", "fail", "success", "exclude")


@dataclass(frozen=True)
class BoundedScore:
    """A model's score under each policy, its worst-case bounds, and their interval."""

    model: str
    scores: dict[str, float]
    lower: float
    upper: float
    ci_lo: float
    ci_hi: float
    ambiguous_runs: int
    ambiguous_share: float
    excluded_tasks: int


@dataclass(frozen=True)
class BoundedPair:
    """Bounds on ``higher - lower`` over all ways of scoring the ambiguous runs."""

    higher: str
    lower: str
    difference: float
    bound_lo: float
    bound_hi: float
    ci_lo: float
    ci_hi: float

    @property
    def identified(self) -> bool:
        """True if every way of scoring ambiguous runs keeps ``higher`` ahead."""
        return self.bound_lo > 0

    @property
    def robust(self) -> bool:
        """True if the Imbens and Manski interval for the difference excludes zero."""
        return self.ci_lo > 0


@dataclass(frozen=True)
class PolicyReport:
    """Policy-sensitivity report for a set of models."""

    level: float
    ambiguous: tuple[str, ...]
    policy: str
    models: tuple[BoundedScore, ...]
    rankings: dict[str, tuple[str, ...]]
    adjacent: tuple[BoundedPair, ...]
    reference: tuple[str, ...] = ()

    @property
    def reference_order(self) -> tuple[str, ...]:
        """The order adjacent pairs follow (the given order, else the policy's ranking)."""
        return self.reference or self.rankings[self.policy]

    @property
    def ranking_changes(self) -> int:
        """Number of policies whose ranking differs from the reference order."""
        ref = self.reference_order
        return sum(order != ref for order in self.rankings.values())


def imbens_manski_critical(gap: float, se_max: float, level: float) -> float:
    """Critical value C with Phi(C + gap / se_max) - Phi(-C) = level."""
    if se_max <= 0 or not math.isfinite(se_max):
        return float(stats.norm.ppf(0.5 + level / 2))
    ratio = max(gap, 0.0) / se_max

    def eq(c: float) -> float:
        return float(stats.norm.cdf(c + ratio) - stats.norm.cdf(-c) - level)

    hi = float(stats.norm.ppf(0.5 + level / 2))
    lo = float(stats.norm.ppf(level))
    if eq(lo) >= 0:
        return lo
    return float(optimize.brentq(eq, lo, hi))


def imbens_manski_interval(
    lower: FloatArray, upper: FloatArray, level: float
) -> tuple[float, float, float, float]:
    """Point bounds and Imbens and Manski interval from per-task lower and upper values."""
    n = lower.size
    lo_hat, hi_hat = float(lower.mean()), float(upper.mean())
    se_lo = float(lower.std(ddof=1) / math.sqrt(n)) if n > 1 else math.inf
    se_hi = float(upper.std(ddof=1) / math.sqrt(n)) if n > 1 else math.inf
    crit = imbens_manski_critical(hi_hat - lo_hat, max(se_lo, se_hi), level)
    return lo_hat, hi_hat, lo_hat - crit * se_lo, hi_hat + crit * se_hi


def _task_table(frame: pd.DataFrame, status_col: str, ambiguous: Sequence[str]) -> pd.DataFrame:
    amb = frame[status_col].isin(list(ambiguous))
    work = pd.DataFrame(
        {
            "model": frame["model"],
            "task": frame["task"],
            "k": 1,
            "n_amb": amb.astype(int),
            "sum_ok": frame[OUTCOME].where(~amb, 0.0),
            "sum_amb": frame[OUTCOME].where(amb, 0.0),
        }
    )
    table = work.groupby(["model", "task"], sort=True).sum()
    table["n_ok"] = table["k"] - table["n_amb"]
    k = table["k"].astype(float)
    table["as_scored"] = (table["sum_ok"] + table["sum_amb"]) / k
    table["fail"] = table["sum_ok"] / k
    table["success"] = (table["sum_ok"] + table["n_amb"]) / k
    with np.errstate(invalid="ignore", divide="ignore"):
        table["exclude"] = np.where(
            table["n_ok"] > 0, table["sum_ok"] / table["n_ok"].replace(0, 1), np.nan
        )
    return table


def _nanmean(values: FloatArray) -> float:
    """Mean of the non-NaN values (NaN if there are none)."""
    kept = values[~np.isnan(values)]
    return float(kept.mean()) if kept.size else math.nan


def _ranking(models: Sequence[str], scores: dict[str, float]) -> tuple[str, ...]:
    """Models from best to worst score (NaN last), ties broken by name."""

    def key(m: str) -> tuple[bool, float, str]:
        return (math.isnan(scores[m]), -scores[m] if not math.isnan(scores[m]) else 0.0, m)

    return tuple(sorted(models, key=key))


def policy_bounds(
    results: EvalResults,
    *,
    status_col: str = "status",
    ambiguous: Sequence[str] = AMBIGUOUS,
    policy: str = "as_scored",
    level: float = 0.95,
    reference_order: Sequence[str] | None = None,
) -> PolicyReport:
    """Scores under each policy, worst-case bounds, and robustness of adjacent orderings.

    Adjacent pairs follow ``reference_order`` when it is given (for a board whose own
    scoring rule is none of the named policies), else the ranking under ``policy``,
    and use the tasks both models attempted. Each pair's ``difference`` is the
    difference under ``policy``.
    """
    if policy not in POLICIES:
        raise ValueError(f"unknown policy {policy!r}; choose from {', '.join(POLICIES)}")
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    frame = results.frame
    if status_col not in frame.columns:
        raise SchemaError(f"no {status_col!r} column; adapters add it")
    table = _task_table(frame, status_col, ambiguous)
    models = results.models
    bounded: dict[str, BoundedScore] = {}
    for model in models:
        t = table.loc[model]
        scores = {p: _nanmean(t[p].to_numpy(dtype=np.float64)) for p in POLICIES}
        lo, hi, ci_lo, ci_hi = imbens_manski_interval(
            t["fail"].to_numpy(dtype=np.float64), t["success"].to_numpy(dtype=np.float64), level
        )
        n_amb = int(t["n_amb"].sum())
        bounded[model] = BoundedScore(
            model=model,
            scores=scores,
            lower=lo,
            upper=hi,
            ci_lo=ci_lo,
            ci_hi=ci_hi,
            ambiguous_runs=n_amb,
            ambiguous_share=n_amb / int(t["k"].sum()),
            excluded_tasks=int((t["n_ok"] == 0).sum()),
        )
    rankings = {p: _ranking(models, {m: bounded[m].scores[p] for m in models}) for p in POLICIES}
    if reference_order is not None and sorted(reference_order) != sorted(models):
        raise ValueError("reference_order must list every model exactly once")
    order = tuple(reference_order) if reference_order is not None else rankings[policy]
    pairs = []
    for a, b in itertools.pairwise(order):
        ta, tb = table.loc[a], table.loc[b]
        shared = ta.index.intersection(tb.index)
        d_lo = (ta.loc[shared, "fail"] - tb.loc[shared, "success"]).to_numpy(dtype=np.float64)
        d_hi = (ta.loc[shared, "success"] - tb.loc[shared, "fail"]).to_numpy(dtype=np.float64)
        d_ref = (ta.loc[shared, policy] - tb.loc[shared, policy]).to_numpy(dtype=np.float64)
        lo, hi, ci_lo, ci_hi = imbens_manski_interval(d_lo, d_hi, level)
        pairs.append(
            BoundedPair(
                higher=a,
                lower=b,
                difference=_nanmean(d_ref),
                bound_lo=lo,
                bound_hi=hi,
                ci_lo=ci_lo,
                ci_hi=ci_hi,
            )
        )
    return PolicyReport(
        level=level,
        ambiguous=tuple(ambiguous),
        policy=policy,
        models=tuple(bounded[m] for m in order),
        rankings=rankings,
        adjacent=tuple(pairs),
        reference=tuple(reference_order) if reference_order is not None else (),
    )
