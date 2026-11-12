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

Both outputs are lists that claims can count with ``where`` (docs/INTEGRITY.md). The G6 rows'
coverage, bias with its count of estimates, missing intervals, mean width, and share above
zero are recomputed here from the datasets, with the definitions of
``noisefloor.recovery.coverage_summary``, so G10's rebuild checks them too. Each summary also
gets a fixed-width text table (``artifacts/e0/summary-0005.txt``, ``artifacts/g6/summary.txt``,
one line per row) that the report can show as a ``text from=`` block, which claimcheck holds
equal to the file. Every row is synthetic (rule 4): these studies simulate their data. Usage:
``python tools/validity_summary.py`` (``--check`` fails if a banked summary or table differs
from a fresh build; a missing source is skipped).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from types import ModuleType
from typing import Any

REPO = Path(__file__).resolve().parent.parent
E0_SOURCE = "artifacts/e0/coverage-0005.json"
E0_OUT = "artifacts/e0/summary-0005.json"
G6_SOURCE = "artifacts/g6/recovery.json"
G6_OUT = "artifacts/g6/summary.json"
HEADLINE = "artifacts/headline_cells.json"
E0_TXT = "artifacts/e0/summary-0005.txt"
G6_TXT = "artifacts/g6/summary.txt"


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
        # The definitions of noisefloor.recovery.coverage_summary, recomputed from the rows.
        bounded = [c for c in comps if c["lo"] is not None and c["hi"] is not None]
        errors = [
            float(c["estimate"]) - float(c["planted"]) for c in comps if c["estimate"] is not None
        ]
        widths = [float(c["hi"]) - float(c["lo"]) for c in bounded]
        above = sum(1 for c in bounded if float(c["lo"]) > 0)
        rows.append(
            {
                "component": name,
                "datasets": len(comps),
                "covered": covered,
                "coverage": cov,
                "threshold": threshold,
                "pass": bool(ok),
                "planted": planted.pop() if len(planted) == 1 else None,
                "missing": len(comps) - len(bounded),
                "with_estimate": len(errors),
                "bias": sum(errors) / len(errors) if errors else None,
                "mean_width": sum(widths) / len(widths) if widths else None,
                "above_zero": above / len(comps) if comps else None,
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


def _num(value: Any, fmt: str) -> str:
    if value is None or (isinstance(value, float) and value != value):
        return "-"
    return format(value, fmt)


def _columns(head: Sequence[str], body: Sequence[Sequence[str]]) -> str:
    """Fixed-width columns: the first left-aligned, the others right-aligned."""
    widths = [max(len(row[i]) for row in (head, *body)) for i in range(len(head))]
    lines = [
        "  ".join(
            cell.ljust(width) if i == 0 else cell.rjust(width)
            for i, (cell, width) in enumerate(zip(row, widths, strict=True))
        ).rstrip()
        for row in (head, *body)
    ]
    return "\n".join(lines) + "\n"


def e0_table(summary: dict[str, Any]) -> str:
    """One line per E0 part 2 cell: its counts, the G5 pass point, and the verdict."""
    head = (
        "cell",
        "method",
        "interval",
        "replicates",
        "covered",
        "coverage",
        "pass point",
        "verdict",
        "headline",
    )
    body = [
        (
            r["cell"],
            r["method"],
            r["interval_type"],
            str(r["replicates"]),
            str(r["covered"]),
            _num(r["coverage"], ".3f"),
            _num(r["threshold"], ".3f"),
            "pass" if r["pass"] else "FAIL",
            "yes" if r["headline"] else "no",
        )
        for r in summary["cells"]
    ]
    title = f"Synthetic data: E0 part 2 coverage cells (prereg 0005), from {E0_SOURCE}\n"
    return title + _columns(head, body)


def g6_table(summary: dict[str, Any]) -> str:
    """One line per G6 component: coverage and its pass point, bias, width, share above 0."""
    head = (
        "component",
        "planted",
        "datasets",
        "covered",
        "coverage",
        "pass point",
        "verdict",
        "estimates",
        "bias",
        "mean width",
        "above 0",
    )
    body = [
        (
            r["component"],
            _num(r["planted"], ".4g"),
            str(r["datasets"]),
            str(r["covered"]),
            _num(r["coverage"], ".3f"),
            _num(r["threshold"], ".3f"),
            "pass" if r["pass"] else "FAIL",
            str(r["with_estimate"]),
            _num(r["bias"], "+.4f"),
            _num(r["mean_width"], ".4f"),
            _num(r["above_zero"], ".3f"),
        )
        for r in summary["components"]
    ]
    title = (
        "Synthetic data: G6 planted-component recovery (prereg 0006), "
        f"{summary['failed_fits']} failed fits, from {G6_SOURCE}\n"
    )
    return title + _columns(head, body)


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
        out[E0_TXT] = e0_table(summary)
    g6 = repo / G6_SOURCE
    if g6.exists():
        summary = g6_summary(json.loads(g6.read_text()))
        summary["source"] = {"path": G6_SOURCE, "sha256": _sha(g6)}
        out[G6_OUT] = _dump(summary)
        out[G6_TXT] = g6_table(summary)
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
