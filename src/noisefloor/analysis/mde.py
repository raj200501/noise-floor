"""Minimum detectable effects implied by a fitted logit GLMM (E1 research question 2).

At a reference configuration of the non-model factors, model ``m`` has linear predictor
``eta_m``; with the task and model-by-task variances, :func:`evalstats.design_from_logit`
gives the paired-design variance inputs for every pair of models, which are averaged over
pairs and turned into the MDE of a paired comparison on ``T`` tasks with ``k`` runs per
model (two-sided ``alpha``, ``power``). Bootstrap replicates of the fit give replicate
MDEs, hence intervals.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from evalstats import PairedDesign, design_from_logit, mde

DESIGNS: tuple[tuple[int, int], ...] = tuple(
    (t, k) for t in (20, 40, 89, 200, 500) for k in (1, 3, 5)
)


def reference_eta(
    coefs: Mapping[str, float],
    data: pd.DataFrame,
    models: Sequence[str],
    reference: Mapping[str, str],
    averaged: Sequence[str] = (),
) -> dict[str, float]:
    """Linear predictor of each model at the reference levels.

    Treatment coding as in lme4 (``<factor><level>``; the first sorted level is 0).
    Factors in ``averaged`` (such as day and time-of-day block) contribute the mean of
    their level effects weighted by the levels' row shares.
    """
    base = float(coefs.get("(Intercept)", 0.0))
    for factor, level in reference.items():
        base += float(coefs.get(f"{factor}{level}", 0.0))
    for factor in averaged:
        shares = data[factor].value_counts(normalize=True)
        base += float(
            sum(share * coefs.get(f"{factor}{lvl}", 0.0) for lvl, share in shares.items())
        )
    return {m: base + float(coefs.get(f"model{m}", 0.0)) for m in models}


def bernoulli_variance(eta: float, task_var: float, model_task_var: float) -> float:
    """``E[p (1 - p)]`` of one run of a model at linear predictor ``eta`` over the tasks.

    ``p = sigmoid(eta + u + v)`` with task effect ``u`` and model-by-task effect ``v``
    (normal, variances as given), by Gauss-Hermite quadrature: the within-task Bernoulli
    variance a paired MDE assumes for that model, if repeated runs are independent.
    """
    x, w = np.polynomial.hermite_e.hermegauss(40)
    w = w / w.sum()
    p = 1.0 / (1.0 + np.exp(-(eta + math.sqrt(task_var + model_task_var) * x)))
    return float(np.sum(w * p * (1 - p)))


def pooled_design(
    etas: Mapping[str, float], task_var: float, model_task_var: float
) -> PairedDesign:
    """Variance inputs averaged over every pair of models."""
    pairs = list(itertools.combinations(sorted(etas), 2))
    if not pairs:
        raise ValueError("need at least two models")
    designs = [design_from_logit(etas[a], etas[b], task_var, model_task_var) for a, b in pairs]
    return PairedDesign(
        between_task_var=float(np.mean([d.between_task_var for d in designs])),
        within_task_var=float(np.mean([d.within_task_var for d in designs])),
    )


def mde_values(
    design: PairedDesign,
    designs: Sequence[tuple[int, int]] = DESIGNS,
    alpha: float = 0.05,
    power: float = 0.8,
) -> dict[tuple[int, int], float]:
    """MDE of a paired comparison for each (tasks, runs per model) design."""
    return {(t, k): mde(design, t, k, alpha=alpha, power=power) for t, k in designs}


def replicate_mdes(
    coef_replicates: Mapping[str, Sequence[float]],
    variance_replicates: Mapping[str, Sequence[float]],
    data: pd.DataFrame,
    *,
    models: Sequence[str],
    reference: Mapping[str, str],
    averaged: Sequence[str],
    designs: Sequence[tuple[int, int]] = DESIGNS,
    alpha: float = 0.05,
    power: float = 0.8,
) -> dict[tuple[int, int], np.ndarray]:
    """MDE of every bootstrap replicate (NaN where a replicate lacks a parameter)."""
    n = len(next(iter(variance_replicates.values())))
    out = {d: np.full(n, np.nan) for d in designs}
    for b in range(n):
        coefs = {name: float(values[b]) for name, values in coef_replicates.items()}
        task_var = float(variance_replicates["task"][b])
        mt_var = float(variance_replicates["model:task"][b])
        if not all(np.isfinite([task_var, mt_var, *coefs.values()])):
            continue
        etas = reference_eta(coefs, data, models, reference, averaged)
        pooled = pooled_design(etas, task_var, mt_var)
        for key, value in mde_values(pooled, designs, alpha=alpha, power=power).items():
            out[key][b] = value
    return out
