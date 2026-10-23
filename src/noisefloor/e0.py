"""E0: simulation coverage of every evalstats interval (library validity, gate G5).

A cell fixes a method, a data-generating process (DGP), a nominal level, and a
number of replicates. Each replicate draws synthetic data with a known true value,
computes the interval, and records whether it covers the truth. All data here is
synthetic.

Truths are super-population quantities: tasks are draws from a population, and a
model's true score is ``E[sigmoid(mu_m + task effect + interaction)]`` over that
population, computed by Gauss-Hermite quadrature. The DGPs are logit-normal:

    logit p[m, t] = intercept + model[m] + task[t] + model_task[m, t] + factor effects

Methods (``method`` in a cell):

* ``score_interval``: one model's score (``interval`` in cluster_t, bootstrap, bca);
* ``paired_difference``: score difference of two models;
* ``cluster_robust_mean``: observation-weighted mean with CR1 standard errors;
* ``rank_simultaneous`` / ``rank_marginal``: rank intervals cover the true ranks
  (all models jointly / the model at position ``target``);
* ``bounded_score``: Imbens and Manski interval for a score with ambiguous runs;
* ``glmm_component``: lme4 profile interval for a planted variance component.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from evalstats import (
    EvalResults,
    cluster_robust_mean,
    paired_difference,
    policy_bounds,
    rank_confidence,
    score_interval,
)
from evalstats.glmm import decompose

FloatArray = npt.NDArray[np.float64]
_GH_X, _GH_W = np.polynomial.hermite_e.hermegauss(80)
_GH_W = _GH_W / _GH_W.sum()


def expected_sigmoid(mu: float, sd: float) -> float:
    """E[sigmoid(mu + sd * Z)] for standard normal Z (Gauss-Hermite, 80 nodes)."""
    return float(np.sum(_GH_W / (1.0 + np.exp(-(mu + sd * _GH_X)))))


@dataclass(frozen=True)
class Cell:
    """One E0 cell (the shape frozen in the preregistered grid)."""

    cell: str
    method: str
    interval_type: str
    nominal: float
    replicates: int
    params: Mapping[str, Any]

    def seed_for(self, replicate: int) -> int:
        """Deterministic seed for a replicate, from the cell id and the replicate index."""
        digest = hashlib.sha256(f"{self.cell}:{replicate}".encode()).hexdigest()
        return int(digest[:12], 16)


@dataclass(frozen=True)
class Replicate:
    """One replicate's interval and truth (``hit`` overrides for joint coverage)."""

    truth: float
    lo: float
    hi: float
    estimate: float
    hit: bool | None = None

    @property
    def covered(self) -> bool:
        """True if the interval contains the truth (NaN bounds never cover)."""
        if self.hit is not None:
            return self.hit
        return self.lo <= self.truth <= self.hi


def _draw(
    rng: np.random.Generator,
    model_effects: list[float],
    *,
    n_tasks: int,
    runs: int,
    intercept: float,
    task_sd: float,
    model_task_sd: float,
    factor_beta: float | None = None,
) -> pd.DataFrame:
    """Logit-normal draws; with ``factor_beta`` a two-level factor alternates over runs."""
    task = rng.normal(0.0, task_sd, n_tasks)
    inter = rng.normal(0.0, model_task_sd, (len(model_effects), n_tasks))
    level = np.arange(runs) % 2
    shift = (factor_beta or 0.0) * level
    rows = []
    for m, effect in enumerate(model_effects):
        eta = intercept + effect + task + inter[m]
        prob = 1.0 / (1.0 + np.exp(-(eta[:, None] + shift[None, :])))
        draws = rng.random((n_tasks, runs)) < prob
        for t in range(n_tasks):
            for r in range(runs):
                row: tuple[Any, ...] = (f"m{m}", f"t{t}", str(r), float(draws[t, r]))
                if factor_beta is not None:
                    row = (*row, "ab"[level[r]])
                rows.append(row)
    columns = ["model", "task", "run", "outcome"] + (["factor"] if factor_beta is not None else [])
    return pd.DataFrame(rows, columns=columns)


def _true_score(effect: float, p: Mapping[str, Any]) -> float:
    sd = math.sqrt(float(p["task_sd"]) ** 2 + float(p.get("model_task_sd", 0.0)) ** 2)
    return expected_sigmoid(float(p.get("intercept", 0.0)) + effect, sd)


def _true_difference(a: float, b: float, p: Mapping[str, Any]) -> float:
    # E[sigmoid(c + a + T + I_a) - sigmoid(c + b + T + I_b)]; I_a, I_b independent, so each
    # term's expectation only depends on its own total SD.
    return _true_score(a, p) - _true_score(b, p)


def _sim_score(cell: Cell, rng: np.random.Generator, seed: int) -> Replicate:
    p = cell.params
    effects = [float(p.get("effect", 0.0))]
    frame = _draw(
        rng,
        effects,
        n_tasks=int(p["n_tasks"]),
        runs=int(p["runs"]),
        intercept=float(p.get("intercept", 0.0)),
        task_sd=float(p["task_sd"]),
        model_task_sd=float(p.get("model_task_sd", 0.0)),
    )
    res = EvalResults.from_frame(frame)
    iv = score_interval(
        res,
        "m0",
        method=cell.interval_type,
        level=cell.nominal,
        n_boot=int(p.get("n_boot", 2000)),
        seed=seed,
    ).interval
    return Replicate(_true_score(effects[0], p), iv.lo, iv.hi, iv.estimate)


def _sim_paired(cell: Cell, rng: np.random.Generator, seed: int) -> Replicate:
    p = cell.params
    effects = [float(p["effect_a"]), float(p["effect_b"])]
    frame = _draw(
        rng,
        effects,
        n_tasks=int(p["n_tasks"]),
        runs=int(p["runs"]),
        intercept=float(p.get("intercept", 0.0)),
        task_sd=float(p["task_sd"]),
        model_task_sd=float(p.get("model_task_sd", 0.0)),
    )
    res = EvalResults.from_frame(frame)
    d = paired_difference(
        res,
        "m0",
        "m1",
        method=cell.interval_type,
        level=cell.nominal,
        n_boot=int(p.get("n_boot", 2000)),
        seed=seed,
    ).difference
    return Replicate(_true_difference(effects[0], effects[1], p), d.lo, d.hi, d.estimate)


def _sim_cluster_robust(cell: Cell, rng: np.random.Generator, seed: int) -> Replicate:
    p = cell.params
    frame = _draw(
        rng,
        [float(p.get("effect", 0.0))],
        n_tasks=int(p["n_tasks"]),
        runs=int(p["runs"]),
        intercept=float(p.get("intercept", 0.0)),
        task_sd=float(p["task_sd"]),
        model_task_sd=0.0,
    )
    iv, _ = cluster_robust_mean(frame["outcome"].to_numpy(), frame["task"].to_numpy(), cell.nominal)
    return Replicate(_true_score(float(p.get("effect", 0.0)), p), iv.lo, iv.hi, iv.estimate)


def _rank_frame(cell: Cell, rng: np.random.Generator) -> tuple[EvalResults, list[float]]:
    p = cell.params
    effects = [float(e) for e in p["effects"]]
    frame = _draw(
        rng,
        effects,
        n_tasks=int(p["n_tasks"]),
        runs=int(p["runs"]),
        intercept=float(p.get("intercept", 0.0)),
        task_sd=float(p["task_sd"]),
        model_task_sd=float(p.get("model_task_sd", 0.0)),
    )
    return EvalResults.from_frame(frame), effects


def _sim_rank(cell: Cell, rng: np.random.Generator, seed: int) -> Replicate:
    res, effects = _rank_frame(cell, rng)
    simultaneous = cell.method == "rank_simultaneous"
    rc = rank_confidence(
        res,
        level=cell.nominal,
        simultaneous=simultaneous,
        n_boot=int(cell.params.get("n_boot", 1000)),
        seed=seed,
    )
    order = sorted(range(len(effects)), key=lambda i: -effects[i])
    true_rank = {f"m{i}": order.index(i) + 1 for i in range(len(effects))}
    by_model = {r.model: r for r in rc.intervals}
    if simultaneous:
        ok = all(r.best_rank <= true_rank[r.model] <= r.worst_rank for r in rc.intervals)
        width = float(np.mean([r.worst_rank - r.best_rank for r in rc.intervals]))
        return Replicate(math.nan, 0.0, width, math.nan, hit=ok)
    target = f"m{order[int(cell.params['target'])]}"
    r = by_model[target]
    return Replicate(
        float(true_rank[target]), float(r.best_rank), float(r.worst_rank), float(r.rank)
    )


def _sim_bounded(cell: Cell, rng: np.random.Generator, seed: int) -> Replicate:
    p = cell.params
    effect = float(p.get("effect", 0.0))
    frame = _draw(
        rng,
        [effect, 0.0],
        n_tasks=int(p["n_tasks"]),
        runs=int(p["runs"]),
        intercept=float(p.get("intercept", 0.0)),
        task_sd=float(p["task_sd"]),
        model_task_sd=0.0,
    )
    amb = rng.random(len(frame)) < float(p["ambiguous_rate"])
    # The true outcome stays as drawn; the recorded outcome of an ambiguous run is 0.
    frame["status"] = np.where(amb, "infra_error", "ok")
    frame.loc[amb, "outcome"] = 0.0
    rep = policy_bounds(EvalResults.from_frame(frame), level=cell.nominal)
    m0 = next(m for m in rep.models if m.model == "m0")
    return Replicate(_true_score(effect, p), m0.ci_lo, m0.ci_hi, m0.scores["as_scored"])


def _sim_glmm(cell: Cell, rng: np.random.Generator, seed: int) -> Replicate:
    p = cell.params
    n_models = int(p["n_models"])
    model_sd = float(p.get("model_sd", 0.0))
    effects = [float(e) for e in rng.normal(0.0, model_sd, n_models)]
    beta = float(p["factor_beta"]) if "factor_beta" in p else None
    frame = _draw(
        rng,
        effects,
        n_tasks=int(p["n_tasks"]),
        runs=int(p["runs"]),
        intercept=float(p.get("intercept", 0.0)),
        task_sd=float(p["task_sd"]),
        model_task_sd=float(p.get("model_task_sd", 0.0)),
        factor_beta=beta,
    )
    target = str(p["component"])
    random = tuple(p.get("random", ("task", "model", "model:task")))
    fixed = ("factor",) if beta is not None else ()
    d = decompose(
        EvalResults.from_frame(frame),
        random=random,
        fixed=fixed,
        level=cell.nominal,
        backend=str(p.get("backend", "lme4")),
        seed=seed,
    )
    comp = d.component(target)
    planted = {
        "task": float(p["task_sd"]) ** 2,
        "model": model_sd**2,
        "model:task": float(p.get("model_task_sd", 0.0)) ** 2,
        "factor": 0.25 * (beta or 0.0) ** 2,
    }[target]
    lo = math.nan if comp.lo is None else comp.lo
    hi = math.nan if comp.hi is None else comp.hi
    return Replicate(planted, lo, hi, comp.variance)


SIMULATORS: dict[str, Callable[[Cell, np.random.Generator, int], Replicate]] = {
    "score_interval": _sim_score,
    "paired_difference": _sim_paired,
    "cluster_robust_mean": _sim_cluster_robust,
    "rank_simultaneous": _sim_rank,
    "rank_marginal": _sim_rank,
    "bounded_score": _sim_bounded,
    "glmm_component": _sim_glmm,
}


def run_replicate(cell: Cell, replicate: int) -> Replicate:
    """Run one replicate of a cell (deterministic in the cell id and index)."""
    seed = cell.seed_for(replicate)
    rng = np.random.default_rng(seed)
    return SIMULATORS[cell.method](cell, rng, seed)


def summarize(cell: Cell, reps: list[Replicate]) -> dict[str, Any]:
    """Cell summary in the artifacts/e0/coverage.json format."""
    covered = sum(r.covered for r in reps)
    n = len(reps)
    cov = covered / n if n else math.nan
    return {
        "cell": cell.cell,
        "method": cell.method,
        "interval_type": cell.interval_type,
        "nominal": cell.nominal,
        "replicates": n,
        "covered": covered,
        "coverage": cov,
        "mcse": math.sqrt(cov * (1 - cov) / n) if n else math.nan,
        "mean_width": float(np.nanmean([r.hi - r.lo for r in reps])) if n else math.nan,
        "params": dict(cell.params),
    }


def cells_from_grid(grid: list[Mapping[str, Any]]) -> list[Cell]:
    """Cells from a preregistered grid (the ```json e0-grid``` block)."""
    return [
        Cell(
            cell=str(c["cell"]),
            method=str(c["method"]),
            interval_type=str(c["interval_type"]),
            nominal=float(c["nominal"]),
            replicates=int(c["replicates"]),
            params=dict(c.get("params", {})),
        )
        for c in grid
    ]


def canonical(obj: Any) -> str:
    """Canonical JSON used for hashing grids and results."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))
