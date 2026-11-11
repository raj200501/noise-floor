"""Rank-confidence intervals for a small leaderboard (synthetic data drawn here).

Simultaneous 95% rank intervals from a task bootstrap, and which adjacent orderings
are supported. Usage: python docs/examples/03_rank_confidence.py
"""

import numpy as np
import pandas as pd

from evalstats import EvalResults, rank_confidence

rng = np.random.default_rng(3)
effects = {"m-a": 0.8, "m-b": 0.5, "m-c": 0.45, "m-d": 0.0, "m-e": -0.6}
task = rng.normal(0.0, 1.2, size=80)
rows = []
for model, effect in effects.items():
    p = 1 / (1 + np.exp(-(effect + task + rng.normal(0.0, 0.5, size=80))))
    for t in range(80):
        for run in range(2):
            rows.append((model, f"t{t}", run, float(rng.random() < p[t])))
results = EvalResults.from_frame(pd.DataFrame(rows, columns=["model", "task", "run", "outcome"]))

ranks = rank_confidence(results, n_boot=2000, seed=1)
for r in ranks.intervals:
    interval = f"[{r.best_rank}, {r.worst_rank}]"
    print(f"{r.rank}. {r.model}: score {r.score:.3f}, rank interval {interval}")
for pair in ranks.adjacent:
    print(f"{pair.higher} > {pair.lower}: {'supported' if pair.supported else 'not supported'}")
