"""Compare two models on the same tasks with a paired, task-clustered interval.

Runs offline on the synthetic demo fixture (planted effects, not model results).
Usage: python docs/examples/01_compare_two_models.py
"""

from pathlib import Path

from evalstats import paired_difference, read_csv

DATA = Path(__file__).resolve().parents[2] / "fixtures" / "demo" / "synthetic_results.csv"

results = read_csv(DATA)
a, b = results.models[0], results.models[1]
cmp = paired_difference(results, a, b)
d = cmp.difference
print(
    f"{a} - {b} on {cmp.n_tasks} shared tasks: {d.estimate:+.3f}, "
    f"95% CI [{d.lo:+.3f}, {d.hi:+.3f}], p = {cmp.p_value:.3f}"
)
print("ordering supported at 95%:", cmp.supported)
