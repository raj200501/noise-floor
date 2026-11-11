"""Tests for tools/validity_summary.py: per-row verdicts by the G5 and G6 rules."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

TOOLS = Path(__file__).resolve().parent.parent / "tools"


def _load() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "tools_validity_summary", TOOLS / "validity_summary.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


vs = _load()


def _cell(name: str, covered: int, reps: int = 2000) -> dict[str, object]:
    return {
        "cell": name,
        "method": "m",
        "interval_type": "t",
        "nominal": 0.95,
        "replicates": reps,
        "covered": covered,
        "coverage": covered / reps,
    }


def test_e0_summary_applies_the_g5_rule_and_marks_headline_cells() -> None:
    source = {
        "prereg": "prereg/0005-e0-glmm-cells.md",
        "cells": [_cell("A", 1900), _cell("B", 1840)],
    }
    out = vs.e0_summary(source, {"A"})
    rows = {r["cell"]: r for r in out["cells"]}
    assert rows["A"]["pass"] and rows["A"]["headline"]
    assert not rows["B"]["pass"] and not rows["B"]["headline"]
    assert rows["B"]["threshold"] < 0.95 - 0.015
    assert out["synthetic"] is True and "synthetic" in out["label"]


def _dataset(seed: int, hit: bool, missing: bool = False) -> dict[str, object]:
    lo, hi = (None, None) if missing else ((0.5, 1.5) if hit else (2.0, 3.0))
    return {
        "seed": seed,
        "components": {"task": {"planted": 1.0, "estimate": 1.1, "lo": lo, "hi": hi}},
    }


def test_g6_summary_counts_hits_and_misses_by_the_g6_rule() -> None:
    good = [_dataset(i, True) for i in range(190)] + [_dataset(190 + i, False) for i in range(10)]
    source = {
        "prereg": "prereg/0006-g6-recovery.md",
        "nominal": 0.95,
        "failed_fits": 0,
        "datasets": good,
        "coverage": {"task": {"missing": 0, "bias": 0.1, "mean_width": 1.0, "above_zero": 0.95}},
    }
    row = vs.g6_summary(source)["components"][0]
    assert row["covered"] == 190 and row["pass"] and row["planted"] == 1.0
    bad = [_dataset(i, True) for i in range(170)] + [
        _dataset(170 + i, True, True) for i in range(30)
    ]
    source["datasets"] = bad
    row = vs.g6_summary(source)["components"][0]
    assert row["covered"] == 170 and not row["pass"]  # a missing interval counts as a miss


def test_build_skips_unbanked_sources_and_records_source_hashes(tmp_path: Path) -> None:
    assert vs.build(tmp_path) == {}
    (tmp_path / "artifacts/e0").mkdir(parents=True)
    (tmp_path / "artifacts/e0/coverage-0005.json").write_text(
        json.dumps({"cells": [_cell("A", 1900)]})
    )
    (tmp_path / "artifacts/headline_cells.json").write_text(json.dumps({"cells": ["A"]}))
    texts = vs.build(tmp_path)
    data = json.loads(texts["artifacts/e0/summary-0005.json"])
    assert data["source"]["path"] == "artifacts/e0/coverage-0005.json"
    assert len(data["source"]["sha256"]) == 64 and data["cells"][0]["headline"]
    assert "artifacts/g6/summary.json" not in texts
