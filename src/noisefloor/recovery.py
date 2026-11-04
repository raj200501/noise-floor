"""Planted-component recovery for the E1 analysis (G6), on synthetic data only.

Each dataset mirrors the E1 local-arm design: models crossed with tasks, a balanced
two-level fractional factorial of run-level factors inside every (model, task) cell,
runs grouped into batches that share a server configuration (model and concurrency),
and batches spread over days and time-of-day blocks. Outcomes are Bernoulli draws from

    logit p = intercept + model + factors + day + block + task + model:task + batch

with ``task``, ``model:task``, and ``batch`` drawn from normal distributions with the
planted variances. The E1 analysis (:func:`analyze`) fits the logit GLMM with
:func:`evalstats.decompose` and a parametric bootstrap; :func:`recover` reports, per
component, the planted value, the estimate, and the interval.

Planted values: a random component's planted value is its variance parameter; a fixed
factor's is the variance over runs of its planted contribution to the linear predictor
under the design's level shares (for a balanced two-level factor with effect ``b``,
``b**2 / 4``). Every row carries ``synthetic = True``.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import itertools
import json
import math
import multiprocessing as mp
import os
import sys
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from evalstats import EvalResults, paired_difference, score_interval
from evalstats.glmm import (
    BOOT_INTERVALS,
    COEF_INTERVALS,
    LOGISTIC_RESIDUAL_VAR,
    Decomposition,
    boot_coef_interval,
    boot_variance_interval,
    contribution_box,
    decompose,
    level_weights,
)
from noisefloor.analysis import mde as mde_lib
from noisefloor.e0 import expected_sigmoid
from noisefloor.preregs import prereg_block

# Run-level two-level factors of the half fraction 2^(5-1) (defining relation I = ABCDE),
# split into two blocks of 8 runs by the ABC interaction so that every main effect is
# balanced within each block (4 runs at each level).
RUN_FACTORS = ("harness", "timeout", "steps", "temperature")
BATCH_FACTOR = "concurrency"
TWO_LEVEL = (*RUN_FACTORS, BATCH_FACTOR)


@dataclass(frozen=True)
class Design:
    """Shape of one synthetic E1-like dataset."""

    n_models: int = 4
    n_tasks: int = 40
    runs_per_cell: int = 8
    batch_size: int = 10
    days: int = 3
    blocks_per_day: int = 2

    def __post_init__(self) -> None:
        """Reject shapes the factorial cannot fill."""
        if self.runs_per_cell not in (8, 16):
            raise ValueError("runs_per_cell must be 8 (one block) or 16 (the half fraction)")
        if min(self.n_models, self.n_tasks, self.batch_size, self.days, self.blocks_per_day) < 1:
            raise ValueError("every design size must be positive")


@dataclass(frozen=True)
class Planted:
    """Planted effects (logit scale) and variance parameters."""

    intercept: float = 0.0
    model: tuple[float, ...] = (0.6, 0.2, -0.2, -0.6)
    factors: Mapping[str, float] = field(
        default_factory=lambda: {
            "harness": 0.5,
            "timeout": 0.4,
            "steps": 0.4,
            "temperature": 0.1,
            "concurrency": 0.0,
        }
    )
    day: tuple[float, ...] = (0.0, 0.1, -0.1)
    block: tuple[float, ...] = (0.0, 0.05)
    task_var: float = 2.0
    model_task_var: float = 0.5
    batch_var: float = 0.1

    def to_json(self) -> dict[str, Any]:
        """Plain-data form."""
        return {**asdict(self), "factors": dict(self.factors)}


def half_fraction_blocks() -> tuple[list[tuple[int, ...]], list[tuple[int, ...]]]:
    """The 16 runs of 2^(5-1) with I = ABCDE, split into two blocks by ABC (0/1 coding)."""
    runs = [bits for bits in itertools.product((0, 1), repeat=5) if sum(bits) % 2 == 0]
    first = [r for r in runs if (r[0] + r[1] + r[2]) % 2 == 0]
    second = [r for r in runs if (r[0] + r[1] + r[2]) % 2 == 1]
    return first, second


def cell_runs(
    first: list[tuple[int, ...]], second: list[tuple[int, ...]], parity: int, runs: int
) -> list[tuple[int, ...]]:
    """The factor levels of one (model, task) cell: one block by parity, or both."""
    if runs == 16:
        return first + second
    return first if parity % 2 == 0 else second


def fixed_contribution(levels: Sequence[float], shares: Sequence[float]) -> float:
    """Variance over runs of a factor's contribution, given level effects and shares."""
    effects = np.asarray(levels, dtype=np.float64)
    weights = np.asarray(shares, dtype=np.float64)
    return float(weights @ (effects - weights @ effects) ** 2)


def simulate(design: Design, planted: Planted, seed: int) -> pd.DataFrame:
    """One synthetic dataset (deterministic given ``seed``); see the module docstring."""
    if len(planted.model) != design.n_models or len(planted.day) != design.days:
        raise ValueError("planted model and day effects must match the design")
    if len(planted.block) != design.blocks_per_day:
        raise ValueError("planted block effects must match the design")
    rng = np.random.default_rng(seed)
    first, second = half_fraction_blocks()
    task_eff = rng.normal(0.0, math.sqrt(planted.task_var), design.n_tasks)
    cell_eff = rng.normal(0.0, math.sqrt(planted.model_task_var), (design.n_models, design.n_tasks))
    runs = []
    for m in range(design.n_models):
        for t in range(design.n_tasks):
            half = cell_runs(first, second, m + t, design.runs_per_cell)
            for bits in half:
                runs.append((m, t, dict(zip(TWO_LEVEL, bits, strict=True))))
    order = rng.permutation(len(runs))
    # Batches: consecutive runs (in random order) of one model and one concurrency level.
    groups: dict[tuple[int, int], list[int]] = {}
    for index in order:
        m, _, levels = runs[int(index)]
        groups.setdefault((m, levels[BATCH_FACTOR]), []).append(int(index))
    batches: list[list[int]] = []
    for members in groups.values():
        batches += [
            members[i : i + design.batch_size] for i in range(0, len(members), design.batch_size)
        ]
    slots = design.days * design.blocks_per_day
    batch_slot = rng.permutation(np.arange(len(batches)) % slots)
    batch_eff = rng.normal(0.0, math.sqrt(planted.batch_var), len(batches))
    rows = []
    for b, members in enumerate(batches):
        day, tod = divmod(int(batch_slot[b]), design.blocks_per_day)
        for index in members:
            m, t, levels = runs[index]
            eta = (
                planted.intercept
                + planted.model[m]
                + sum(planted.factors[name] * levels[name] for name in TWO_LEVEL)
                + planted.day[day]
                + planted.block[tod]
                + task_eff[t]
                + cell_eff[m, t]
                + batch_eff[b]
            )
            rows.append(
                {
                    "model": f"m{m}",
                    "task": f"t{t:03d}",
                    "run": f"r{index}",
                    "outcome": float(rng.random() < 1.0 / (1.0 + math.exp(-eta))),
                    **{name: str(levels[name]) for name in TWO_LEVEL},
                    "day": f"d{day}",
                    "block": f"b{tod}",
                    "batch": f"B{b:04d}",
                    "synthetic": True,
                }
            )
    return pd.DataFrame(rows)


# MDE reference configuration (simulation coding): minimal harness, the long timeout, the
# larger step budget, temperature 0, concurrency 1; day and block averaged.
MDE_REFERENCE = {
    "harness": "0",
    "timeout": "1",
    "steps": "1",
    "temperature": "0",
    "concurrency": "0",
}
MDE_AVERAGED = ("day", "block")
RANDOM_TERMS = ("task", "model:task", "batch")
FIXED_TERMS = ("model", *TWO_LEVEL, "day", "block")


def planted_mde(
    design: Design, planted: Planted, data: pd.DataFrame, designs: Sequence[tuple[int, int]]
) -> dict[tuple[int, int], float]:
    """The true MDE of each design, from the planted parameters at the reference levels."""
    coefs: dict[str, float] = {"(Intercept)": planted.intercept}
    for m in range(design.n_models):
        coefs[f"modelm{m}"] = planted.model[m] - planted.model[0]
    coefs["(Intercept)"] += planted.model[0]
    for name in TWO_LEVEL:
        coefs[f"{name}1"] = planted.factors[name]
    for d in range(design.days):
        coefs[f"dayd{d}"] = planted.day[d] - planted.day[0]
    coefs["(Intercept)"] += planted.day[0]
    for b in range(design.blocks_per_day):
        coefs[f"blockb{b}"] = planted.block[b] - planted.block[0]
    coefs["(Intercept)"] += planted.block[0]
    models = [f"m{m}" for m in range(design.n_models)]
    etas = mde_lib.reference_eta(coefs, data, models, MDE_REFERENCE, MDE_AVERAGED)
    pooled = mde_lib.pooled_design(etas, planted.task_var, planted.model_task_var)
    return mde_lib.mde_values(pooled, designs)


def mde_components(
    decomp: Decomposition,
    data: pd.DataFrame,
    design: Design,
    designs: Sequence[tuple[int, int]],
    level: float,
) -> dict[str, dict[str, Any]]:
    """Estimated MDE of each design with its replicate values (for intervals)."""
    models = sorted(data["model"].unique())
    variances = {c.name: c.variance for c in decomp.components if c.kind == "random"}
    etas = mde_lib.reference_eta(decomp.fixed_effects, data, models, MDE_REFERENCE, MDE_AVERAGED)
    point = mde_lib.mde_values(
        mde_lib.pooled_design(etas, variances["task"], variances["model:task"]), designs
    )
    reps = mde_lib.replicate_mdes(
        decomp.coefficient_replicates,
        {k: decomp.replicates[k] for k in ("task", "model:task")},
        data,
        models=models,
        reference=MDE_REFERENCE,
        averaged=MDE_AVERAGED,
        designs=designs,
    )
    out = {}
    for key in designs:
        out[f"mde:T{key[0]}:k{key[1]}"] = {"estimate": point[key], "replicates": reps[key]}
    return out


def planted_values(design: Design, planted: Planted, data: pd.DataFrame) -> dict[str, float]:
    """The planted value of every component, with fixed shares taken from ``data``."""
    values = {
        "task": planted.task_var,
        "model:task": planted.model_task_var,
        "batch": planted.batch_var,
    }

    def shares(column: str, names: Sequence[str]) -> list[float]:
        return [float(np.mean(data[column] == name)) for name in names]

    models = [f"m{m}" for m in range(design.n_models)]
    values["model"] = fixed_contribution(planted.model, shares("model", models))
    for name in TWO_LEVEL:
        values[name] = fixed_contribution((0.0, planted.factors[name]), shares(name, ["0", "1"]))
    days = [f"d{d}" for d in range(design.days)]
    values["day"] = fixed_contribution(planted.day, shares("day", days))
    blocks = [f"b{b}" for b in range(design.blocks_per_day)]
    values["block"] = fixed_contribution(planted.block, shares("block", blocks))
    return values


def planted_shares(truth: Mapping[str, float], names: Sequence[str]) -> dict[str, float]:
    """Planted share of each analyzed component, and of the residual, out of their sum.

    The total is over the components the analysis reports (``names``) plus the logistic
    residual ``pi**2 / 3``, the total each estimated share is a fraction of.
    """
    total = sum(truth[n] for n in names) + LOGISTIC_RESIDUAL_VAR
    out = {f"share:{n}": truth[n] / total for n in names}
    out["share:residual"] = LOGISTIC_RESIDUAL_VAR / total
    return out


def replicate_shares(decomp: Decomposition) -> dict[str, np.ndarray]:
    """Bootstrap replicate share of every component, the residual included."""
    names = [c.name for c in decomp.components if c.kind != "residual"]
    reps = np.column_stack([np.asarray(decomp.replicates[n], dtype=np.float64) for n in names])
    totals = reps.sum(axis=1) + LOGISTIC_RESIDUAL_VAR
    out = {f"share:{n}": reps[:, i] / totals for i, n in enumerate(names)}
    out["share:residual"] = LOGISTIC_RESIDUAL_VAR / totals
    return out


@dataclass(frozen=True)
class AnalysisSettings:
    """The E1 decomposition settings (fixed in the preregistration)."""

    n_boot: int = 200
    nagq: int = 0
    boot_interval: str = "basic"
    level: float = 0.95
    random: tuple[str, ...] = RANDOM_TERMS
    box_interval: str = "percentile"
    mde_designs: tuple[tuple[int, int], ...] = ()
    mde_interval: str = "percentile"
    share_interval: str = "percentile"
    score_interval: str = "cluster_t"  # local-arm scores
    contrast_interval: str = "cluster_t"  # local harness contrasts and model pairs


def run_fixed_eta(data: pd.DataFrame, planted: Planted) -> np.ndarray:
    """Each run's planted linear predictor without its task, model:task, or batch effect."""
    eta = np.full(len(data), planted.intercept, dtype=np.float64)
    eta += np.array([planted.model[int(str(m)[1:])] for m in data["model"]])
    for name in TWO_LEVEL:
        eta += planted.factors[name] * data[name].astype(int).to_numpy()
    eta += np.array([planted.day[int(str(d)[1:])] for d in data["day"]])
    eta += np.array([planted.block[int(str(b)[1:])] for b in data["block"]])
    return eta


def planted_success(data: pd.DataFrame, planted: Planted) -> np.ndarray:
    """Each run's success probability averaged over the task population.

    The task, model:task, and batch effects are integrated out (normal, Gauss-Hermite), so a
    mean of these over runs is the super-population quantity a task-clustered interval for
    a score or a paired difference estimates.
    """
    sd = math.sqrt(planted.task_var + planted.model_task_var + planted.batch_var)
    cache: dict[float, float] = {}
    out = np.empty(len(data), dtype=np.float64)
    for i, eta in enumerate(run_fixed_eta(data, planted)):
        key = round(float(eta), 12)
        if key not in cache:
            cache[key] = expected_sigmoid(key, sd)
        out[i] = cache[key]
    return out


def _results(frame: pd.DataFrame, model: pd.Series) -> EvalResults:
    return EvalResults.from_frame(
        pd.DataFrame(
            {
                "model": model.to_numpy(),
                "task": frame["task"].to_numpy(),
                "run": frame["run"].to_numpy(),
                "outcome": frame["outcome"].to_numpy(),
            }
        )
    )


def _entry(truth: float, interval: Any) -> dict[str, Any]:
    return {"planted": truth, "estimate": interval.estimate, "lo": interval.lo, "hi": interval.hi}


def local_intervals(
    data: pd.DataFrame, planted: Planted, settings: AnalysisSettings
) -> dict[str, dict[str, Any]]:
    """Local-arm score and contrast intervals as the E1 analysis computes them.

    ``score:<m>``: model ``m``'s success rate over all its runs; ``harness:<m>``: harness
    level 1 minus level 0 within model ``m`` (paired by task); ``pair:<a>-<b>``: model ``a``
    minus model ``b`` on harness level 0 runs (the minimal-agent runs of E1). Planted values
    are the matching means of :func:`planted_success` over the same runs.
    """
    success = planted_success(data, planted)
    models = sorted(str(m) for m in data["model"].unique())
    level = settings.level
    out: dict[str, dict[str, Any]] = {}
    everyone = _results(data, data["model"])
    harness_one = (data["harness"] == "1").to_numpy()
    for m in models:
        mine = (data["model"] == m).to_numpy()
        si = score_interval(everyone, m, method=settings.score_interval, level=level)
        out[f"score:{m}"] = _entry(float(success[mine].mean()), si.interval)
        sub = data[mine]
        by_harness = _results(sub, "h" + sub["harness"].astype(str))
        pc = paired_difference(
            by_harness, "h1", "h0", method=settings.contrast_interval, level=level
        )
        truth = success[mine & harness_one].mean() - success[mine & ~harness_one].mean()
        out[f"harness:{m}"] = _entry(float(truth), pc.difference)
    minimal = data[~harness_one]
    pairs = _results(minimal, minimal["model"])
    for i, a in enumerate(models):
        for b in models[i + 1 :]:
            pc = paired_difference(pairs, a, b, method=settings.contrast_interval, level=level)
            is_a = (data["model"] == a).to_numpy() & ~harness_one
            is_b = (data["model"] == b).to_numpy() & ~harness_one
            truth = success[is_a].mean() - success[is_b].mean()
            out[f"pair:{a}-{b}"] = _entry(float(truth), pc.difference)
    return out


def analyze(data: pd.DataFrame, settings: AnalysisSettings, seed: int) -> Decomposition:
    """The E1 decomposition of one table of runs (a factor held constant is left out)."""
    fixed = tuple(name for name in FIXED_TERMS if data[name].nunique() > 1)
    return decompose(
        EvalResults(data),
        random=settings.random,
        fixed=fixed,
        level=settings.level,
        backend="lme4",
        method="parametric_bootstrap",
        n_boot=settings.n_boot,
        nagq=settings.nagq,
        boot_interval=settings.boot_interval,
        box_interval=settings.box_interval,
        share_interval=settings.share_interval,
        seed=seed,
    )


def recover(
    design: Design, planted: Planted, settings: AnalysisSettings, seed: int
) -> dict[str, Any]:
    """Simulate one dataset, analyze it, and return each component's row for G6.

    A fit that fails is recorded, not dropped: every component keeps its planted value
    with no estimate and no interval, and ``error`` says what failed.
    """
    data = simulate(design, planted, seed)
    truth = planted_values(design, planted, data)
    try:
        decomp = analyze(data, settings, seed)
    except Exception as exc:  # recorded as a dataset without intervals (prereg 0006)
        names = [n for n in FIXED_TERMS if data[n].nunique() > 1]
        empty = {"estimate": None, "lo": None, "hi": None}
        analyzed = [*settings.random, *names]
        failed: dict[str, dict[str, Any]] = {n: {"planted": truth[n], **empty} for n in analyzed}
        for key, value in planted_shares(truth, analyzed).items():
            failed[key] = {"planted": value, **empty}
        if settings.mde_designs:
            for (t, k), value in planted_mde(design, planted, data, settings.mde_designs).items():
                failed[f"mde:T{t}:k{k}"] = {"planted": value, **empty}
        failed.update(local_intervals(data, planted, settings))
        error = f"{type(exc).__name__}: {exc}"[:300]
        return {"seed": seed, "components": failed, "converged": False, "error": error}
    components = {}
    shares = planted_shares(truth, [c.name for c in decomp.components if c.kind != "residual"])
    for comp in decomp.components:
        components[f"share:{comp.name}"] = {
            "planted": shares[f"share:{comp.name}"],
            "estimate": comp.share,
            "lo": comp.share_lo,
            "hi": comp.share_hi,
        }
        if comp.kind == "residual":
            continue
        components[comp.name] = {
            "planted": truth[comp.name],
            "estimate": comp.variance,
            "lo": comp.lo,
            "hi": comp.hi,
        }
    if settings.mde_designs:
        true_mde = planted_mde(design, planted, data, settings.mde_designs)
        found = mde_components(decomp, data, design, settings.mde_designs, settings.level)
        for (t, k), value in true_mde.items():
            entry = found[f"mde:T{t}:k{k}"]
            ci = boot_coef_interval(
                entry["estimate"],
                np.asarray(entry["replicates"]),
                settings.level,
                settings.mde_interval,
            )
            components[f"mde:T{t}:k{k}"] = {
                "planted": value,
                "estimate": entry["estimate"],
                "lo": None if ci is None else max(0.0, ci[0]),
                "hi": None if ci is None else ci[1],
            }
    components.update(local_intervals(data, planted, settings))
    return {
        "seed": seed,
        "components": components,
        "converged": decomp.converged,
        "notes": list(decomp.messages)[-1:],
    }


def coverage_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    """Per component: coverage, mean bias, and interval width over datasets."""
    names = sorted({name for row in rows for name in row["components"]})
    out = {}
    for name in names:
        comps = [row["components"][name] for row in rows if name in row["components"]]
        ok = [c for c in comps if c["lo"] is not None and c["hi"] is not None]
        covered = sum(1 for c in ok if c["lo"] <= c["planted"] <= c["hi"])
        errors = [c["estimate"] - c["planted"] for c in comps if c["estimate"] is not None]
        out[name] = {
            "n": len(comps),
            "with_interval": len(ok),
            "coverage": covered / len(comps) if comps else math.nan,  # a missing bound misses
            "bias": float(np.mean(errors)) if errors else math.nan,
            "mean_width": float(np.mean([c["hi"] - c["lo"] for c in ok])) if ok else math.nan,
        }
    return out


def dumps(obj: Any) -> str:
    """Canonical JSON (NaN and infinities as null)."""

    def clean(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return [clean(v) for v in value]
        return value

    return json.dumps(clean(obj), sort_keys=True)


PILOT_SEEDS = range(990000, 991000)  # exploratory method pilots; never used by G6


def pilot_row(
    design: Design, planted: Planted, settings: AnalysisSettings, seed: int
) -> dict[str, Any]:
    """One exploratory dataset scored under every random-effect interval rule.

    The bootstrap runs once; ``percentile``, ``basic``, and ``bc`` intervals are all
    computed from its replicates. Fixed factors use the box interval.
    """
    data = simulate(design, planted, seed)
    truth = planted_values(design, planted, data)
    decomp = analyze(data, settings, seed)
    rows: dict[str, Any] = {}
    for comp in decomp.components:
        if comp.kind == "residual":
            continue
        entry: dict[str, Any] = {"planted": truth[comp.name], "estimate": comp.variance}
        if comp.kind == "random":
            reps = np.asarray(decomp.replicates[comp.name], dtype=np.float64)
            for kind in BOOT_INTERVALS:
                entry[kind] = boot_variance_interval(comp.variance, reps, settings.level, kind)
        else:
            levels, weights = level_weights(data, comp.name)
            names = [f"{comp.name}{lvl}" for lvl in levels[1:]]
            est = np.array([decomp.fixed_effects.get(n, 0.0) for n in names])
            draws = np.column_stack(
                [np.asarray(decomp.coefficient_replicates[n], dtype=np.float64) for n in names]
            )
            se = np.array([decomp.coefficient_se.get(n, np.nan) for n in names])
            se_draws = np.column_stack(
                [np.asarray(decomp.coefficient_se_replicates[n], dtype=np.float64) for n in names]
            )
            for kind in COEF_INTERVALS:
                entry[f"box_{kind}"] = contribution_box(
                    weights, est, draws, settings.level, kind, se=se, se_draws=se_draws
                )
        rows[comp.name] = entry
    share_draws = replicate_shares(decomp)
    shares = planted_shares(truth, [c.name for c in decomp.components if c.kind != "residual"])
    for comp in decomp.components:
        key = f"share:{comp.name}"
        entry = {"planted": shares[key], "estimate": comp.share}
        for kind in BOOT_INTERVALS:
            ci = boot_variance_interval(comp.share, share_draws[key], settings.level, kind)
            entry[kind] = None if ci is None else (ci[0], min(ci[1], 1.0))
        rows[key] = entry
    if settings.mde_designs:
        true_mde = planted_mde(design, planted, data, settings.mde_designs)
        found = mde_components(decomp, data, design, settings.mde_designs, settings.level)
        for (t, k), value in true_mde.items():
            item = found[f"mde:T{t}:k{k}"]
            entry = {"planted": value, "estimate": item["estimate"]}
            for kind in BOOT_INTERVALS:
                ci = boot_coef_interval(
                    item["estimate"], np.asarray(item["replicates"]), settings.level, kind
                )
                entry[kind] = None if ci is None else (max(0.0, ci[0]), ci[1])
            rows[f"mde:T{t}:k{k}"] = entry
    return {"seed": seed, "components": rows, "notes": list(decomp.messages)[-1:]}


def pilot_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    """Coverage of each interval rule per component, with bias, over pilot datasets."""
    out: dict[str, dict[str, Any]] = {}
    names = sorted({n for row in rows for n in row["components"]})
    for name in names:
        comps = [row["components"][name] for row in rows if name in row["components"]]
        summary: dict[str, Any] = {
            "n": len(comps),
            "bias": float(np.mean([c["estimate"] - c["planted"] for c in comps])),
            "sd": float(np.std([c["estimate"] for c in comps])),
        }
        kinds = ("percentile", "basic", "bc", "basic_log", *(f"box_{k}" for k in COEF_INTERVALS))
        for kind in kinds:
            ok = [c for c in comps if c.get(kind) and None not in c[kind]]
            if ok:
                hit = sum(1 for c in ok if c[kind][0] <= c["planted"] <= c[kind][1])
                summary[kind] = hit / len(ok)
        out[name] = summary
    return out


# ---------------------------------------------------------------------------
# confirmatory runs, configured only by frozen preregistration blocks

REPO = Path(__file__).resolve().parents[2]
CONFIG_BLOCK = "recovery-config"  # design, planted values, and analysis settings
E0_DIR = REPO / "artifacts" / "e0"
G6_DIR = REPO / "artifacts" / "g6"


def config_from(block: Mapping[str, Any]) -> tuple[Design, Planted, AnalysisSettings]:
    """Design, planted values, and settings from a ``recovery-config`` block."""
    p = block["planted"]
    planted = Planted(
        intercept=float(p["intercept"]),
        model=tuple(float(v) for v in p["model"]),
        factors={str(k): float(v) for k, v in p["factors"].items()},
        day=tuple(float(v) for v in p["day"]),
        block=tuple(float(v) for v in p["block"]),
        task_var=float(p["task_var"]),
        model_task_var=float(p["model_task_var"]),
        batch_var=float(p["batch_var"]),
    )
    st = block["settings"]
    settings = AnalysisSettings(
        n_boot=int(st["n_boot"]),
        nagq=int(st["nagq"]),
        boot_interval=str(st["boot_interval"]),
        level=float(st["level"]),
        random=tuple(str(r) for r in st["random"]),
        box_interval=str(st.get("box_interval", "percentile")),
        mde_designs=tuple((int(t), int(k)) for t, k in st.get("mde_designs", [])),
        mde_interval=str(st.get("mde_interval", "percentile")),
        share_interval=str(st.get("share_interval", "percentile")),
        score_interval=str(st.get("score_interval", "cluster_t")),
        contrast_interval=str(st.get("contrast_interval", "cluster_t")),
    )
    return Design(**block["design"]), planted, settings


def _pilot_job(job: tuple[Design, Planted, AnalysisSettings, int]) -> dict[str, Any]:
    return pilot_row(*job)


def _local_job(job: tuple[dict[str, Any], int]) -> dict[str, Any]:
    """Only the local-arm score and contrast intervals of one dataset (no decomposition)."""
    block, seed = job
    design, planted, settings = config_from(block)
    data = simulate(design, planted, seed)
    return {"seed": seed, "components": local_intervals(data, planted, settings)}


def _recover_job(job: tuple[dict[str, Any], int]) -> dict[str, Any]:
    block, seed = job
    design, planted, settings = config_from(block)
    return recover(design, planted, settings, seed)


def _pool_map(jobs: list[Any], func: Any, workers: int) -> Iterator[Any]:
    if workers <= 1:
        yield from map(func, jobs)
        return
    with mp.get_context("spawn").Pool(workers, initializer=os.nice, initargs=(10,)) as pool:
        yield from pool.imap_unordered(func, jobs)


def run_g6(prereg: Path, raw: Path, workers: int = 1, log: Any = print) -> int:
    """Append a recovery row for every preregistered seed not yet in ``raw`` (resumable)."""
    block = prereg_block(prereg, CONFIG_BLOCK)
    seeds = [int(s) for s in prereg_block(prereg, "g6-seeds")]
    done = set()
    if raw.exists():
        done = {json.loads(ln)["seed"] for ln in raw.read_text().splitlines() if ln.strip()}
    jobs = [(block, s) for s in seeds if s not in done]
    raw.parent.mkdir(parents=True, exist_ok=True)
    added = 0
    with raw.open("a") as fh:
        for row in _pool_map(jobs, _recover_job, workers):
            fh.write(dumps(row) + "\n")
            fh.flush()
            added += 1
            if added % 10 == 0:
                log(f"g6: {len(done) + added}/{len(seeds)} datasets")
    return added


def bank_g6(prereg: Path, raw: Path, out: Path) -> dict[str, Any]:
    """Write the G6 artifact from the raw rows of every preregistered seed."""
    block = prereg_block(prereg, CONFIG_BLOCK)
    seeds = [int(s) for s in prereg_block(prereg, "g6-seeds")]
    rows = {}
    for line in raw.read_text().splitlines():
        if line.strip():
            row = json.loads(line)
            rows[int(row["seed"])] = row
    missing = [s for s in seeds if s not in rows]
    if missing:
        raise ValueError(f"{len(missing)} preregistered datasets are not run yet")
    datasets = [
        {"seed": s, "components": rows[s]["components"], "converged": rows[s].get("converged")}
        for s in seeds
    ]
    names = sorted({n for d in datasets for n in d["components"]})

    def mean_error(name: str) -> float:
        comps = [d["components"][name] for d in datasets]
        errors = [c["estimate"] - c["planted"] for c in comps if c["estimate"] is not None]
        return float(np.mean(errors)) if errors else math.nan

    bias = {n: mean_error(n) for n in names}
    nominal = float(block["settings"]["level"])
    artifact = {
        "prereg": prereg.relative_to(REPO).as_posix()
        if prereg.is_relative_to(REPO)
        else str(prereg),
        "synthetic": True,
        "confirmatory": True,
        "nominal": nominal,
        "config": block,
        "datasets": datasets,
        "bias": bias,
        "coverage": coverage_summary(datasets),
        "generator": "python -m noisefloor.recovery g6-bank " + prereg.name,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dumps(artifact) + "\n")
    return artifact


def _repo_path(path: Path) -> str:
    absolute = path.resolve()
    return absolute.relative_to(REPO).as_posix() if absolute.is_relative_to(REPO) else str(path)


def e0_glmm_seed(group: str, replicate: int) -> int:
    """Dataset seed of one replicate of a group of E0 GLMM cells (all components share it)."""
    digest = hashlib.sha256(f"e0-glmm:{group}:{replicate}".encode()).hexdigest()
    return int(digest[:12], 16)


def run_e0_glmm(amendment: Path, workers: int = 1, root: Path = E0_DIR, log: Any = print) -> int:
    """Fill the raw files of the amendment's grouped cells (those with a ``group``; resumable).

    Each replicate index is one simulated dataset shared by every cell of its group (seed
    :func:`e0_glmm_seed`); a cell's raw line holds that component's planted value as
    ``truth``, its estimate, and its interval. A group whose cells all have
    ``"analysis": "local"`` computes only the local-arm score and contrast intervals
    (:func:`local_intervals`); any other group runs the full E1 analysis (:func:`recover`).
    """
    grid = prereg_block(amendment, "e0-grid")
    block = prereg_block(amendment, CONFIG_BLOCK)
    cells = [c for c in grid if "group" in c["params"]]
    groups: dict[str, list[dict[str, Any]]] = {}
    for c in cells:
        groups.setdefault(str(c["params"]["group"]), []).append(c)
    added = 0
    for group, members in groups.items():
        kinds = {c["params"].get("analysis", "full") for c in members}
        if len(kinds) != 1:
            raise ValueError(f"group {group} mixes analyses {sorted(kinds)}")
        job = _local_job if kinds == {"local"} else _recover_job
        reps = max(int(c["replicates"]) for c in members)
        paths = {
            str(c["params"]["component"]): root / "raw" / f"{c['cell']}.jsonl" for c in members
        }
        done = None
        for path in paths.values():
            have = set()
            if path.exists():
                have = {json.loads(ln)["i"] for ln in path.read_text().splitlines() if ln.strip()}
            done = have if done is None else done & have
        todo = [i for i in range(reps) if i not in (done or set())]
        jobs = [(block, e0_glmm_seed(group, i)) for i in todo]
        index = {e0_glmm_seed(group, i): i for i in todo}
        for path in paths.values():
            path.parent.mkdir(parents=True, exist_ok=True)
        for row in _pool_map(jobs, job, workers):
            i = index[int(row["seed"])]
            for component, path in paths.items():
                comp = row["components"].get(component)
                line = {
                    "i": i,
                    "seed": row["seed"],
                    "truth": None if comp is None else comp["planted"],
                    "estimate": None if comp is None else comp["estimate"],
                    "lo": None if comp is None else comp["lo"],
                    "hi": None if comp is None else comp["hi"],
                    "hit": None,
                }
                with path.open("a") as fh:
                    fh.write(dumps(line) + "\n")
            added += 1
            if added % 10 == 0:
                log(f"e0 {group}: {added}/{len(todo)} datasets")
    return added


def main(argv: Sequence[str] | None = None) -> int:
    """``python -m noisefloor.recovery pilot OUT.jsonl FIRST COUNT [--n-boot B] [--nagq 0|1]``.

    Runs exploratory method-pilot datasets (seeds from :data:`PILOT_SEEDS`) with the
    default design and planted values, appending one JSON line per dataset (resumable;
    ``--workers`` runs datasets in parallel, so lines arrive out of seed order), then
    prints the coverage summary.
    """
    args_list = list(argv) if argv is not None else sys.argv[1:]
    if args_list and args_list[0] in ("g6-run", "g6-bank", "e0-run"):
        return _confirmatory_main(args_list)
    parser = argparse.ArgumentParser(description="planted-component recovery (synthetic)")
    parser.add_argument("command", choices=["pilot"])
    parser.add_argument("out", type=Path)
    parser.add_argument("first", type=int)
    parser.add_argument("count", type=int)
    parser.add_argument("--n-boot", type=int, default=100)
    parser.add_argument(
        "--no-batch", action="store_true", help="plant no batch variance and fit no batch term"
    )
    parser.add_argument("--tasks", type=int, default=40)
    parser.add_argument("--runs-per-cell", type=int, default=8)
    parser.add_argument(
        "--mde", action="store_true", help="also score MDE intervals at three designs"
    )
    parser.add_argument(
        "--nagq", type=int, choices=(0, 1), default=1, help="1 for Laplace fits, 0 for nAGQ = 0"
    )
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(args_list)
    seeds = [s for s in range(args.first, args.first + args.count)]
    if any(s not in PILOT_SEEDS for s in seeds):
        raise SystemExit("pilot seeds must come from PILOT_SEEDS")
    done = set()
    if args.out.exists():
        done = {json.loads(ln)["seed"] for ln in args.out.read_text().splitlines() if ln.strip()}
    settings = AnalysisSettings(n_boot=args.n_boot, nagq=args.nagq)
    planted = Planted()
    if args.no_batch:
        settings = dataclasses.replace(settings, random=("task", "model:task"))
        planted = dataclasses.replace(planted, batch_var=0.0)
    if args.mde:
        settings = dataclasses.replace(settings, mde_designs=((20, 1), (40, 1), (89, 5)))
    design = Design()
    if (args.tasks, args.runs_per_cell) != (40, 8):
        design = Design(n_tasks=args.tasks, runs_per_cell=args.runs_per_cell)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    jobs = [(design, planted, settings, s) for s in seeds if s not in done]
    with args.out.open("a") as fh:
        for row in _pool_map(jobs, _pilot_job, args.workers):
            fh.write(dumps(row) + "\n")
            fh.flush()
    rows = [json.loads(ln) for ln in args.out.read_text().splitlines() if ln.strip()]
    print(json.dumps(pilot_summary(rows), indent=1, sort_keys=True))
    return 0


def _confirmatory_main(argv: list[str]) -> int:
    """``g6-run PREREG [--workers N]``, ``g6-bank PREREG``, ``e0-run AMENDMENT [--workers N]``."""
    parser = argparse.ArgumentParser(description="confirmatory recovery runs from a prereg")
    parser.add_argument("command", choices=["g6-run", "g6-bank", "e0-run"])
    parser.add_argument("prereg", type=Path)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args(argv)
    raw = G6_DIR / "raw.jsonl"
    if args.command == "g6-run":
        print(f"added {run_g6(args.prereg, raw, args.workers)} datasets")
    elif args.command == "g6-bank":
        artifact = bank_g6(args.prereg, raw, G6_DIR / "recovery.json")
        print(json.dumps(artifact["coverage"], indent=1, sort_keys=True))
    else:
        print(f"added {run_e0_glmm(args.prereg, args.workers)} datasets")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
