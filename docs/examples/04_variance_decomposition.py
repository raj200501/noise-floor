"""Variance decomposition of eval outcomes with a logit GLMM (synthetic data drawn here).

Needs a GLMM backend: lme4 through rpy2 (`pip install evalstats[r]` with R and lme4) or
statsmodels (`pip install evalstats[glmm]`); the backend is named in the output.
Usage: python docs/examples/04_variance_decomposition.py
"""

import numpy as np
import pandas as pd

from evalstats import EvalResults, decompose

rng = np.random.default_rng(11)
models = {"m1": 0.6, "m2": 0.0, "m3": -0.6}
task = rng.normal(0.0, 1.0, size=40)
rows = []
for model, effect in models.items():
    for t in range(40):
        for run, harness in enumerate(("h0", "h1", "h0", "h1")):
            eta = effect + task[t] + (0.8 if harness == "h1" else 0.0)
            outcome = float(rng.random() < 1 / (1 + np.exp(-eta)))
            rows.append((model, f"t{t}", run, outcome, harness))
frame = pd.DataFrame(rows, columns=["model", "task", "run", "outcome", "harness"])
results = EvalResults.from_frame(frame)

decomp = decompose(results, random=("task", "model"), fixed=("harness",))
print(decomp.formula, "|", decomp.backend)
for c in decomp.components:
    interval = "" if c.lo is None else f" [{c.lo:.3f}, {c.hi:.3f}]"
    print(f"{c.name:10s} {c.kind:9s} variance {c.variance:.3f}{interval} share {c.share:.3f}")
