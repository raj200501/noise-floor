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


def test_banked_summaries_are_current() -> None:
    # A source not banked yet gives no summary; a banked one must match its committed summary
    # (G10 runs the same comparison through `validity_summary.py --check`).
    for path, text in vs.build().items():
        assert (vs.REPO / path).exists(), f"{path} is not committed beside its banked source"
        assert (vs.REPO / path).read_text() == text, f"{path} differs from a fresh build"


def _g6_source(datasets: list[dict[str, object]]) -> dict[str, object]:
    return {"prereg": "p", "nominal": 0.95, "failed_fits": 1, "datasets": datasets}


def test_g6_rows_are_recomputed_with_the_bank_s_definitions() -> None:
    from noisefloor.recovery import coverage_summary

    rows = []
    for i in range(40):
        lo, hi = (None, None) if i % 9 == 0 else (0.2 + 0.01 * i, 1.4 + 0.02 * i)
        estimate = None if i % 13 == 0 else 0.9 + 0.005 * i
        comp = {"planted": 1.0, "estimate": estimate, "lo": lo, "hi": hi}
        zero = {"planted": 0.0, "estimate": 0.01 * (i % 5), "lo": -0.1 + 0.01 * i, "hi": 0.5}
        rows.append({"seed": i, "components": {"task": comp, "concurrency": zero}})
    bank = coverage_summary(rows)
    for row in vs.g6_summary(_g6_source(rows))["components"]:
        want = bank[row["component"]]
        for key in ("missing", "with_estimate"):
            assert row[key] == want[key], (row["component"], key)
        for key in ("bias", "mean_width", "above_zero"):
            assert abs(row[key] - want[key]) <= 1e-12, (row["component"], key)


def test_text_tables_have_one_line_per_row_and_mark_failures() -> None:
    e0 = vs.e0_summary({"cells": [_cell("A", 1900), _cell("B-long-name", 1840)]}, {"A"})
    table = vs.e0_table(e0).splitlines()
    assert table[0].startswith("Synthetic data: E0 part 2") and len(table) == 4
    assert table[2].split()[-2:] == ["pass", "yes"] and table[3].split()[-2:] == ["FAIL", "no"]
    good = [_dataset(i, True) for i in range(170)] + [_dataset(170 + i, False) for i in range(30)]
    g6 = vs.g6_table(vs.g6_summary(_g6_source(good))).splitlines()
    assert g6[0].startswith("Synthetic data: G6") and "1 failed fits" in g6[0]
    assert len(g6) == 3 and "FAIL" in g6[2].split() and "+0.1000" in g6[2].split()
    for lines in (table[1:], g6[1:]):
        assert all(ln == ln.rstrip() for ln in lines)
        assert len({len(ln) for ln in lines}) == 1  # right-aligned columns end together


def test_build_writes_both_tables_beside_their_summaries(tmp_path: Path) -> None:
    (tmp_path / "artifacts/e0").mkdir(parents=True)
    (tmp_path / "artifacts/g6").mkdir(parents=True)
    (tmp_path / "artifacts/e0/coverage-0005.json").write_text(
        json.dumps({"cells": [_cell("A", 1900)]})
    )
    (tmp_path / "artifacts/g6/recovery.json").write_text(
        json.dumps(_g6_source([_dataset(i, True) for i in range(200)]))
    )
    texts = vs.build(tmp_path)
    assert sorted(texts) == sorted([vs.E0_OUT, vs.E0_TXT, vs.G6_OUT, vs.G6_TXT])
    assert vs.e0_table(json.loads(texts[vs.E0_OUT])) == texts[vs.E0_TXT]
    assert vs.g6_table(json.loads(texts[vs.G6_OUT])) == texts[vs.G6_TXT]
