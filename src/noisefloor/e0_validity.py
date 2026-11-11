"""Generate the library's validity data and the headline-cell map from E0 results.

``python -m noisefloor.e0_validity`` reads ``artifacts/e0/coverage.json`` and each
amendment's ``artifacts/e0/coverage-NNNN.json``, applies the G5 rule to every cell, and
writes:

* ``src/evalstats/data/validity.json``: the failing cells, each with the function,
  interval method, and largest task count it covers, and a supported alternative;
* ``artifacts/headline_cells.json``: the E0 cells behind each headline interval
  (the list frozen in prereg/0003, extended by the ```json headline``` block of each
  banked amendment, such as the E1 cells of E0 part 2).
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
# E0 cell method -> the function whose validity check matches the cell's interval type.
FUNCTIONS = {
    "score_interval": "score_interval",
    "paired_difference": "paired_difference",
    "contrast_difference": "contrast_difference",
    "cluster_robust_mean": "cluster_robust_mean",
    "rank_simultaneous": "rank_confidence",
    "rank_marginal": "rank_confidence",
    "bounded_score": "policy_bounds",
    "glmm_component": "decompose",
    "glmm_mde": "replicate_mdes",  # noisefloor.analysis.e1.mde_tables checks it
    "sampling_discordance": "score_interval",  # the sampling table's interval
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


def load_coverage(directory: Path = COVERAGE.parent) -> dict[str, Any]:
    """The base coverage artifact with every amendment artifact's cells merged in."""
    merged: dict[str, Any] = json.loads((directory / "coverage.json").read_text())
    merged["amendments"] = []
    for path in sorted(directory.glob("coverage-*.json")):
        part = json.loads(path.read_text())
        merged["amendments"].append(part["prereg"])
        merged["cells"] = [*merged["cells"], *part["cells"]]
    return merged


def outputs(directory: Path = COVERAGE.parent) -> dict[Path, str]:
    """The text of each output file, built from the banked coverage artifacts."""
    coverage = load_coverage(directory)
    validity, headline = build(coverage, amendment_headlines(coverage))
    return {
        VALIDITY: json.dumps(validity, indent=1, sort_keys=True) + "\n",
        HEADLINE: json.dumps(headline, indent=1, sort_keys=True) + "\n",
    }


def stale(directory: Path = COVERAGE.parent) -> list[Path]:
    """Output files whose text differs from what the banked coverage gives (run main)."""
    return [p for p, text in outputs(directory).items() if not p.exists() or p.read_text() != text]


def main() -> int:
    """Write both files from the banked coverage artifacts."""
    texts = outputs()
    VALIDITY.parent.mkdir(parents=True, exist_ok=True)
    for path, text in texts.items():
        path.write_text(text)
    validity, headline = (json.loads(texts[p]) for p in (VALIDITY, HEADLINE))
    print(
        f"unsupported cells: {sorted(validity['unsupported'])}; "
        f"headline cells: {len(headline['cells'])}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
