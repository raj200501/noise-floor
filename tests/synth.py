"""Synthetic (labeled synthetic) results for tests only."""

import numpy as np
import pandas as pd

from evalstats import EvalResults


def make_results(
    n_tasks: int = 30,
    runs: int = 3,
    models: tuple[str, ...] = ("a", "b"),
    p: tuple[float, ...] = (0.6, 0.5),
    seed: int = 0,
) -> EvalResults:
    """Synthetic binary results (labeled synthetic; for tests only)."""
    rng = np.random.default_rng(seed)
    base = rng.normal(0, 1, size=n_tasks)
    rows = []
    for m, pm in zip(models, p, strict=True):
        logit = np.log(pm / (1 - pm)) + base
        prob = 1 / (1 + np.exp(-logit))
        for t in range(n_tasks):
            for r in range(runs):
                rows.append(
                    {
                        "model": m,
                        "task": f"t{t:03d}",
                        "run": r,
                        "outcome": float(rng.random() < prob[t]),
                    }
                )
    return EvalResults.from_frame(pd.DataFrame(rows))
