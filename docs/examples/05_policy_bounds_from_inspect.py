"""Read Inspect AI logs and bound scores under rules for errored and limited runs.

Uses the recorded Inspect fixture logs in tests/fixtures/adapters/inspect (written by
Inspect itself with its mock model, so every outcome is mock data).
Usage: python docs/examples/05_policy_bounds_from_inspect.py
"""

from pathlib import Path

from evalstats import policy_bounds
from evalstats.adapters import read_inspect

LOGS = Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "adapters" / "inspect" / "json"

results = read_inspect(LOGS)
print(f"{len(results)} runs, statuses: {results.frame['status'].value_counts().to_dict()}")
report = policy_bounds(results)
for m in report.models:
    recorded = m.scores["as_scored"]
    bounds = f"[{m.lower:.3f}, {m.upper:.3f}]"
    print(f"{m.model}: as recorded {recorded:.3f}, bounds {bounds}, {m.ambiguous_runs} ambiguous")
