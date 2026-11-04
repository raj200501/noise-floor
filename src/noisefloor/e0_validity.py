"""Generate the library's validity data and the headline-cell map from E0 results.

``python -m noisefloor.e0_validity`` reads ``artifacts/e0/coverage.json``, applies the
G5 rule to every cell, and writes:

* ``src/evalstats/data/validity.json``: the failing cells, each with the function,
  interval method, and largest task count it covers, and a supported alternative;
* ``artifacts/headline_cells.json``: the E0 cells behind each headline interval
  (the list frozen in prereg/0003, extended by the ```json headline``` block of each
  amendment the coverage artifact banks, such as the E1 cells of E0 part 2).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

from noisefloor.preregs import prereg_block

REPO = Path(__file__).resolve().parents[2]
COVERAGE = REPO / "artifacts" / "e0" / "coverage.json"
VALIDITY = REPO / "src" / "evalstats" / "data" / "validity.json"
HEADLINE = REPO / "artifacts" / "headline_cells.json"
FUNCTIONS = {
    "score_interval": "score_interval",
    "paired_difference": "paired_difference",
    "cluster_robust_mean": "cluster_robust_mean",
    "rank_simultaneous": "rank_confidence",
    "rank_marginal": "rank_confidence",
    "bounded_score": "policy_bounds",
    "glmm_component": "decompose",
}
ALTERNATIVES = {
    "bootstrap": "method='cluster_t'",
    "bca": "method='cluster_t'",
    "cluster_t": "more tasks",
}
HEADLINE_SOURCES = {
    "e2 score intervals (tb2)": "S-tb2-t",
    "e2 score intervals (epoch-swebench)": "S-ep-t",
    "e2 adjacent pairs (tb2)": "P-tb2-t",
    "e2 adjacent pairs (epoch-swebench)": "P-ep-t",
    "e2 rank intervals (tb2)": "R-tb2-sim",
    "e2 rank intervals (epoch-swebench)": "R-ep-sim",
    "e3 bounds (tb2)": "B-tb2",
    "e3 bounds (epoch-swebench)": "B-ep",
}


def passes(covered: int, reps: int, nominal: float) -> bool:
    """The G5 rule."""
    c = covered / reps
    return c >= nominal - 0.015 - 2 * math.sqrt(c * (1 - c) / reps) - 1e-12


def build(
    coverage: dict[str, Any], extra_headline: dict[str, str] | None = None
) -> tuple[dict[str, Any], dict[str, Any]]:
    """(validity data, headline map) from a coverage artifact."""
    unsupported = {}
    for cell in coverage["cells"]:
        if passes(int(cell["covered"]), int(cell["replicates"]), float(cell["nominal"])):
            continue
        unsupported[cell["cell"]] = {
            "function": FUNCTIONS[cell["method"]],
            "method": cell["interval_type"],
            "max_tasks": int(cell["params"]["n_tasks"]),
            "coverage": cell["covered"] / cell["replicates"],
            "nominal": cell["nominal"],
            "alternative": ALTERNATIVES.get(cell["interval_type"], "another method"),
        }
    validity = {"source": "artifacts/e0/coverage.json", "unsupported": unsupported}
    sources = {**HEADLINE_SOURCES, **(extra_headline or {})}
    headline = {
        "prereg": coverage["prereg"],
        "sources": sources,
        "cells": sorted(set(sources.values())),
    }
    return validity, headline


def amendment_headlines(coverage: dict[str, Any], repo: Path = REPO) -> dict[str, str]:
    """Headline sources added by the banked amendments (their ```json headline``` blocks)."""
    out: dict[str, str] = {}
    for rel in coverage.get("amendments", []):
        try:
            block = prereg_block(repo / rel, "headline")
        except ValueError:  # an amendment without headline intervals
            continue
        out.update({str(k): str(v) for k, v in block.items()})
    return out


def main() -> int:
    """Write both files from the banked coverage artifact."""
    coverage = json.loads(COVERAGE.read_text())
    validity, headline = build(coverage, amendment_headlines(coverage))
    VALIDITY.parent.mkdir(parents=True, exist_ok=True)
    VALIDITY.write_text(json.dumps(validity, indent=1, sort_keys=True) + "\n")
    HEADLINE.write_text(json.dumps(headline, indent=1, sort_keys=True) + "\n")
    print(
        f"unsupported cells: {sorted(validity['unsupported'])}; "
        f"headline cells: {len(headline['cells'])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
