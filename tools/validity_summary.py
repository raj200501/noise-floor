#!/usr/bin/env python3
"""Pass or fail for each banked E0 part 2 cell and G6 component, by the gates' own rule.

Derived from banked artifacts only (rule 1), with the coverage rule and hit test of
``tools/ship_check.py`` (G5 and G6), so a verdict here is the gate's verdict for that row:

* ``artifacts/e0/coverage-0005.json`` (E0 part 2, prereg 0005) gives
  ``artifacts/e0/summary-0005.json``: per cell its coverage, the pass point, whether it passes,
  and whether it is a headline cell (``artifacts/headline_cells.json``, written by
  ``python -m noisefloor.e0_validity``).
* ``artifacts/g6/recovery.json`` (G6, prereg 0006) gives ``artifacts/g6/summary.json``: per
  variance component its coverage over the datasets, the pass point, whether it passes, its
  mean bias and interval width, and how often its interval is above zero.

Both outputs are lists that claims can count with ``where`` (docs/INTEGRITY.md). Every row is
synthetic (rule 4): these studies simulate their data. Usage: ``python tools/validity_summary.py``
(``--check`` fails if a banked summary differs from a fresh build; a missing source is skipped).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

REPO = Path(__file__).resolve().parent.parent
E0_SOURCE = "artifacts/e0/coverage-0005.json"
E0_OUT = "artifacts/e0/summary-0005.json"
G6_SOURCE = "artifacts/g6/recovery.json"
G6_OUT = "artifacts/g6/summary.json"
HEADLINE = "artifacts/headline_cells.json"


def _ship_check() -> ModuleType:
    name = "validity_summary_ship_check"
    spec = importlib.util.spec_from_file_location(name, REPO / "tools" / "ship_check.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # its dataclasses look their module up there
    spec.loader.exec_module(module)
    return module


SC = _ship_check()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def e0_summary(source: dict[str, Any], headline: set[str]) -> dict[str, Any]:
    """The E0 part 2 summary: one row per cell with the G5 coverage rule's verdict."""
    rows = []
    for cell in source["cells"]:
        reps, covered = int(cell["replicates"]), int(cell["covered"])
        ok, cov, threshold = SC.coverage_rule(covered, reps, float(cell["nominal"]))
        rows.append(
            {
                "cell": cell["cell"],
                "method": cell["method"],
                "interval_type": cell["interval_type"],
                "replicates": reps,
                "covered": covered,
                "coverage": cov,
                "threshold": threshold,
                "pass": bool(ok),
                "headline": cell["cell"] in headline,
            }
        )
    return {
        "label": "synthetic: coverage cells of E0 part 2 (prereg 0005), not model results",
        "synthetic": True,
        "generator": "tools/validity_summary.py",
        "prereg": source.get("prereg"),
        "cells": rows,
    }


def g6_summary(source: dict[str, Any]) -> dict[str, Any]:
    """The G6 summary: one row per variance component with the G6 coverage rule's verdict."""
    datasets = source["datasets"]
    nominal = float(source["nominal"])
    names = sorted({n for d in datasets for n in d["components"]})
    rows = []
    for name in names:
        comps = [d["components"][name] for d in datasets if name in d["components"]]
        covered = sum(1 for c in comps if SC.g6_hit(c))
        ok, cov, threshold = SC.coverage_rule(covered, len(comps), nominal)
        planted = {float(c["planted"]) for c in comps}
        summary = (source.get("coverage") or {}).get(name, {})
        rows.append(
            {
                "component": name,
                "datasets": len(comps),
                "covered": covered,
                "coverage": cov,
                "threshold": threshold,
                "pass": bool(ok),
                "planted": planted.pop() if len(planted) == 1 else None,
                "missing": summary.get("missing"),
                "bias": summary.get("bias"),
                "mean_width": summary.get("mean_width"),
                "above_zero": summary.get("above_zero"),
            }
        )
    return {
        "label": "synthetic: planted-component recovery of the E1 analysis (prereg 0006)",
        "synthetic": True,
        "generator": "tools/validity_summary.py",
        "prereg": source.get("prereg"),
        "failed_fits": source.get("failed_fits"),
        "components": rows,
    }


def _dump(obj: dict[str, Any]) -> str:
    def clean(value: Any) -> Any:
        if isinstance(value, float) and value != value:  # NaN as null, as the sources do
            return None
        if isinstance(value, dict):
            return {k: clean(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean(v) for v in value]
        return value

    return json.dumps(clean(obj), indent=1, sort_keys=True) + "\n"


def build(repo: Path = REPO) -> dict[str, str]:
    """The text of each summary whose source is banked, by output path."""
    out: dict[str, str] = {}
    e0 = repo / E0_SOURCE
    if e0.exists():
        head = repo / HEADLINE
        cells = set(json.loads(head.read_text())["cells"]) if head.exists() else set()
        summary = e0_summary(json.loads(e0.read_text()), cells)
        summary["source"] = {"path": E0_SOURCE, "sha256": _sha(e0)}
        if head.exists():
            summary["headline_source"] = {"path": HEADLINE, "sha256": _sha(head)}
        out[E0_OUT] = _dump(summary)
    g6 = repo / G6_SOURCE
    if g6.exists():
        summary = g6_summary(json.loads(g6.read_text()))
        summary["source"] = {"path": G6_SOURCE, "sha256": _sha(g6)}
        out[G6_OUT] = _dump(summary)
    return out


def main(argv: list[str]) -> int:
    texts = build()
    if not texts:
        print("VALIDITY-SUMMARY: nothing banked yet")
        return 0
    if "--check" in argv:
        stale = [
            p
            for p, text in texts.items()
            if not (REPO / p).exists() or (REPO / p).read_text() != text
        ]
        print(f"VALIDITY-SUMMARY: {'FAIL ' + ', '.join(stale) if stale else 'PASS'}")
        return 1 if stale else 0
    for path, text in texts.items():
        (REPO / path).write_text(text)
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
