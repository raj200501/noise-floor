import warnings
from pathlib import Path

import pytest

from evalstats import paired_difference, score_interval, validity
from noisefloor import e0_validity
from synth import make_results


def test_unsupported_cells_come_from_the_data_file() -> None:
    rules = validity.unsupported_rules()
    assert set(rules) == set(validity.UNSUPPORTED_CELLS)
    for rule in rules.values():
        assert {"function", "method", "max_tasks", "coverage", "nominal", "alternative"} <= set(
            rule
        )


def test_warning_for_matching_small_design(monkeypatch: pytest.MonkeyPatch) -> None:
    rule = {
        "function": "paired_difference",
        "method": "bootstrap",
        "max_tasks": 20,
        "coverage": 0.9,
        "nominal": 0.95,
        "alternative": "method='cluster_t'",
    }
    monkeypatch.setattr(validity, "_DATA", {"unsupported": {"X-cell": rule}})
    small = make_results(n_tasks=15)
    with pytest.warns(validity.UnsupportedDesignWarning, match="X-cell"):
        paired_difference(small, "a", "b", method="bootstrap", n_boot=200, seed=0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        paired_difference(small, "a", "b")  # cluster_t: no warning
        paired_difference(
            make_results(n_tasks=40), "a", "b", method="bootstrap", n_boot=200, seed=0
        )  # larger than the failing design
        score_interval(small, "a", method="bootstrap", n_boot=200, seed=0)
    assert validity.check("paired_difference", "bca", 5) == []


def test_generator_applies_the_g5_rule() -> None:
    coverage = {
        "prereg": "prereg/0003-x.md",
        "cells": [
            {
                "cell": "ok",
                "method": "score_interval",
                "interval_type": "cluster_t",
                "nominal": 0.95,
                "replicates": 2000,
                "covered": 1900,
                "params": {"n_tasks": 30},
            },
            {
                "cell": "bad",
                "method": "paired_difference",
                "interval_type": "bootstrap",
                "nominal": 0.95,
                "replicates": 500,
                "covered": 440,
                "params": {"n_tasks": 20},
            },
        ],
    }
    data, headline = e0_validity.build(coverage, {"e1 thing": "G-1"})
    assert list(data["unsupported"]) == ["bad"]
    assert data["unsupported"]["bad"]["function"] == "paired_difference"
    assert "G-1" in headline["cells"] and headline["prereg"] == "prereg/0003-x.md"
    assert e0_validity.passes(1900, 2000, 0.95) and not e0_validity.passes(440, 500, 0.95)


def test_amendment_headlines(tmp_path: Path) -> None:
    amend = tmp_path / "prereg" / "0009-x.md"
    amend.parent.mkdir()
    amend.write_text('# x\n\n```json headline\n{"e1 task variance": "G-e1-task"}\n```\n')
    plain = tmp_path / "prereg" / "0010-y.md"
    plain.write_text("# no headline block\n")
    coverage = {"amendments": ["prereg/0009-x.md", "prereg/0010-y.md"]}
    found = e0_validity.amendment_headlines(coverage, tmp_path)
    assert found == {"e1 task variance": "G-e1-task"}
    assert e0_validity.amendment_headlines({}, tmp_path) == {}


def test_load_coverage_merges_amendment_artifacts(tmp_path: Path) -> None:
    import json

    base = {"prereg": "prereg/0003-x.md", "amendments": [], "cells": [{"cell": "A"}]}
    part = {"prereg": "prereg/0005-y.md", "amends": "prereg/0003-x.md", "cells": [{"cell": "B"}]}
    (tmp_path / "coverage.json").write_text(json.dumps(base))
    (tmp_path / "coverage-0005.json").write_text(json.dumps(part))
    merged = e0_validity.load_coverage(tmp_path)
    assert merged["prereg"] == "prereg/0003-x.md"
    assert merged["amendments"] == ["prereg/0005-y.md"]
    assert [c["cell"] for c in merged["cells"]] == ["A", "B"]
