"""Synthetic eval results with planted effects (always labeled synthetic).

Outcomes are Bernoulli draws from a logit model::

    logit p = intercept + model[m] + task[t] + model_task[m, t] + sum of factor effects

with ``task ~ N(0, task_sd^2)`` and ``model_task ~ N(0, model_task_sd^2)``. Every
row carries ``synthetic = True`` so simulated data is never mistaken for model
results. Used for tests, the offline demo fixture, and validity simulations.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class PlantedSpec:
    """Planted effects for a synthetic results table.

    ``factors`` maps a factor name to its levels and their logit effects; every
    (model, task) pair is run ``runs`` times in each combination of factor levels.
    """

    models: dict[str, float]
    n_tasks: int
    runs: int = 1
    intercept: float = 0.0
    task_sd: float = 1.0
    model_task_sd: float = 0.0
    factors: dict[str, dict[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject empty or negative designs."""
        if not self.models:
            raise ValueError("need at least one model")
        if self.n_tasks < 1 or self.runs < 1:
            raise ValueError("n_tasks and runs must be positive")
        if self.task_sd < 0 or self.model_task_sd < 0:
            raise ValueError("standard deviations must be non-negative")
        for name, levels in self.factors.items():
            if not levels:
                raise ValueError(f"factor {name!r} has no levels")


def simulate(spec: PlantedSpec, seed: int) -> pd.DataFrame:
    """Draw one synthetic long-format results table (deterministic given ``seed``)."""
    rng = np.random.default_rng(seed)
    models = list(spec.models)
    tasks = [f"task{t:03d}" for t in range(spec.n_tasks)]
    task_effect = rng.normal(0.0, spec.task_sd, size=spec.n_tasks)
    model_task = rng.normal(0.0, spec.model_task_sd, size=(len(models), spec.n_tasks))
    names = list(spec.factors)
    combos = list(itertools.product(*(spec.factors[n].items() for n in names)))
    rows = []
    for mi, model in enumerate(models):
        for ti, task in enumerate(tasks):
            base = spec.intercept + spec.models[model] + task_effect[ti] + model_task[mi, ti]
            for combo in combos:
                eta = base + sum(effect for _, effect in combo)
                prob = 1.0 / (1.0 + np.exp(-eta))
                draws = rng.random(spec.runs) < prob
                levels = {n: lvl for n, (lvl, _) in zip(names, combo, strict=True)}
                for r in range(spec.runs):
                    rows.append(
                        {
                            "model": model,
                            "task": task,
                            "run": f"{'.'.join(levels.values()) + '.' if levels else ''}{r}",
                            "outcome": float(draws[r]),
                            **levels,
                            "synthetic": True,
                        }
                    )
    return pd.DataFrame(rows)
