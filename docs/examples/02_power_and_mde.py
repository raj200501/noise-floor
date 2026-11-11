"""How many tasks or runs does a comparison need? Minimum detectable effects.

Estimates the variance inputs from the synthetic demo fixture and prints the MDE for
several designs, then the number of tasks needed to detect a 5-point gap.
Usage: python docs/examples/02_power_and_mde.py
"""

from pathlib import Path

from evalstats import design_from_results, mde, read_csv, tasks_needed

DATA = Path(__file__).resolve().parents[2] / "fixtures" / "demo" / "synthetic_results.csv"

results = read_csv(DATA)
design = design_from_results(results, results.models[0], results.models[1])
for n_tasks in (100, 300, 1000):
    for runs in (1, 5):
        print(f"{n_tasks:>5} tasks, {runs} runs per task: MDE {mde(design, n_tasks, runs):.3f}")
print("tasks needed for a 0.05 gap at 3 runs:", tasks_needed(0.05, design, runs_per_task=3))
