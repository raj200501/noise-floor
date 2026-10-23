"""Rank-confidence sets for leaderboards.

For M models scored on the same T tasks, every pairwise score difference has a
task-clustered (paired) standard error. A task bootstrap of the largest
studentized deviation over pairs gives a critical value ``c`` such that, with
probability about ``level``, every pairwise interval ``diff +- c * se`` holds at
once. Model j is then surely better than k when ``diff_jk - c * se_jk > 0``, and
j's rank interval is::

    best rank  = 1 + #{k surely better than j}
    worst rank = M - #{k surely worse than j}

``simultaneous=True`` controls the error over all models jointly (the
intervals hold together); ``simultaneous=False`` gives marginal intervals, one
critical value per model over its own M - 1 comparisons. This follows the
projection approach to rank inference of Mogstad, Romano, Shaikh, and Wilhelm
(2024). Rank 1 is the best score.

Only tasks attempted by every model are used; the number dropped is reported.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import stats

from evalstats.schema import EvalResults, SchemaError

FloatArray = npt.NDArray[np.float64]
BOOT_CHUNK = 500


@dataclass(frozen=True)
class RankInterval:
    """A model's score, point rank, and rank interval."""

    model: str
    score: float
    rank: int
    best_rank: int
    worst_rank: int

    @property
    def could_be_best(self) -> bool:
        """True if rank 1 is inside the interval."""
        return self.best_rank == 1


@dataclass(frozen=True)
class AdjacentPair:
    """A pair of neighbours in score order and whether their order is supported."""

    higher: str
    lower: str
    difference: float
    lo: float
    hi: float
    p_value: float
    supported: bool
    supported_simultaneous: bool


@dataclass(frozen=True)
class RankConfidence:
    """Rank intervals and adjacent-pair support for a leaderboard."""

    level: float
    simultaneous: bool
    critical_value: float
    n_tasks: int
    dropped_tasks: int
    n_boot: int
    intervals: tuple[RankInterval, ...]
    adjacent: tuple[AdjacentPair, ...]

    @property
    def share_adjacent_supported(self) -> float:
        """Share of adjacent pairs whose marginal interval excludes zero."""
        if not self.adjacent:
            return float("nan")
        return sum(p.supported for p in self.adjacent) / len(self.adjacent)

    @property
    def top_set(self) -> tuple[str, ...]:
        """Models whose rank interval includes 1."""
        return tuple(r.model for r in self.intervals if r.could_be_best)


def complete_score_matrix(results: EvalResults) -> tuple[list[str], FloatArray, int]:
    """Models, the tasks-by-models score matrix on tasks every model attempted, and drops."""
    matrix = results.score_matrix()
    complete = matrix.dropna(axis=0, how="any")
    if len(complete) < 2:
        raise SchemaError("need at least 2 tasks attempted by every model")
    return list(matrix.columns), complete.to_numpy(dtype=np.float64), len(matrix) - len(complete)


def _pairwise(scores: FloatArray) -> tuple[FloatArray, FloatArray]:
    """Mean differences ``theta_j - theta_k`` and their paired standard errors."""
    n_tasks = scores.shape[0]
    diffs = scores[:, :, None] - scores[:, None, :]
    mean = diffs.mean(axis=0)
    se = diffs.std(axis=0, ddof=1) / np.sqrt(n_tasks)
    return mean, se


def _max_t(
    scores: FloatArray, mean: FloatArray, se: FloatArray, n_boot: int, rng: np.random.Generator
) -> FloatArray:
    """Bootstrap draws of max |t| per row model (shape n_boot x M) over its comparisons."""
    n_tasks, n_models = scores.shape
    safe = np.where(se > 0, se, np.inf)
    out = np.empty((n_boot, n_models), dtype=np.float64)
    probs = np.full(n_tasks, 1.0 / n_tasks)
    for start in range(0, n_boot, BOOT_CHUNK):
        stop = min(start + BOOT_CHUNK, n_boot)
        weights = rng.multinomial(n_tasks, probs, size=stop - start) / n_tasks
        boot_means = weights @ scores
        boot_diff = boot_means[:, :, None] - boot_means[:, None, :]
        t = np.abs(boot_diff - mean[None]) / safe[None]
        out[start:stop] = t.max(axis=2)
    return out


def rank_confidence(
    results: EvalResults,
    *,
    level: float = 0.95,
    simultaneous: bool = True,
    n_boot: int = 2000,
    seed: int | None = None,
) -> RankConfidence:
    """Rank intervals for every model and support for each adjacent ordering."""
    if not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    models, scores, dropped = complete_score_matrix(results)
    n_tasks, n_models = scores.shape
    if n_models < 2:
        raise SchemaError("need at least 2 models")
    mean, se = _pairwise(scores)
    rng = np.random.default_rng(seed)
    draws = _max_t(scores, mean, se, n_boot, rng)
    if simultaneous:
        crit = np.full(n_models, float(np.quantile(draws.max(axis=1), level)))
    else:
        crit = np.quantile(draws, level, axis=0)
    lower = mean - crit[:, None] * se
    better = lower > 0
    np.fill_diagonal(better, False)
    point = scores.mean(axis=0)
    order = np.argsort(-point, kind="stable")
    rank_of = np.empty(n_models, dtype=np.int64)
    rank_of[order] = np.arange(1, n_models + 1)
    intervals = []
    for i in order:
        surely_better = int(better[:, i].sum())
        surely_worse = int(better[i, :].sum())
        intervals.append(
            RankInterval(
                model=models[i],
                score=float(point[i]),
                rank=int(rank_of[i]),
                best_rank=1 + surely_better,
                worst_rank=n_models - surely_worse,
            )
        )
    t_crit = float(stats.t.ppf(0.5 + level / 2, df=n_tasks - 1))
    adjacent = []
    for hi_idx, lo_idx in itertools.pairwise(order):
        d, s = float(mean[hi_idx, lo_idx]), float(se[hi_idx, lo_idx])
        if s > 0:
            p_value = float(2 * stats.t.sf(abs(d / s), df=n_tasks - 1))
        else:
            p_value = 0.0 if d != 0 else 1.0
        lo, hi = d - t_crit * s, d + t_crit * s
        adjacent.append(
            AdjacentPair(
                higher=models[hi_idx],
                lower=models[lo_idx],
                difference=d,
                lo=lo,
                hi=hi,
                p_value=p_value,
                supported=lo > 0,
                supported_simultaneous=bool(better[hi_idx, lo_idx]),
            )
        )
    return RankConfidence(
        level=level,
        simultaneous=simultaneous,
        critical_value=float(crit.max()),
        n_tasks=n_tasks,
        dropped_tasks=dropped,
        n_boot=n_boot,
        intervals=tuple(intervals),
        adjacent=tuple(adjacent),
    )
