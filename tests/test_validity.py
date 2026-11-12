import warnings
from collections.abc import Callable
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


def _rule(function: str, method: str) -> dict:
    return {
        "function": function,
        "method": method,
        "max_tasks": 1000,
        "coverage": 0.8,
        "nominal": 0.95,
        "alternative": "another method",
    }


def test_every_e0_function_warns_on_a_matching_failing_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Each function an E0 cell maps to (e0_validity.FUNCTIONS) checks validity (G5)."""
    import numpy as np
    import pandas as pd

    from evalstats import EvalResults, contrast_difference, policy_bounds, rank_confidence
    from evalstats.intervals import cluster_robust_mean

    def expect(function: str, method: str, call: Callable[[], object]) -> None:
        monkeypatch.setattr(validity, "_DATA", {"unsupported": {"X": _rule(function, method)}})
        with pytest.warns(validity.UnsupportedDesignWarning, match="E0 cell X"):
            call()

    data = make_results(n_tasks=12, models=("a", "b", "c"), p=(0.6, 0.5, 0.4))
    expect("score_interval", "cluster_t", lambda: score_interval(data, "a"))
    expect("paired_difference", "cluster_t", lambda: paired_difference(data, "a", "b"))
    expect(
        "contrast_difference",
        "cluster_t",
        lambda: contrast_difference(
            make_results(n_tasks=12, models=("w", "x", "y", "z"), p=(0.6, 0.5, 0.4, 0.3)),
            "w",
            "x",
            "y",
            "z",
        ),
    )
    expect("rank_confidence", "bootstrap_max_t", lambda: rank_confidence(data, n_boot=50, seed=1))
    expect(
        "cluster_robust_mean",
        "cluster_robust",
        lambda: cluster_robust_mean(np.array([0.0, 1.0, 1.0, 0.0]), np.array([1, 1, 2, 2])),
    )
    frame = data.frame.assign(status=["ok", "timeout"] * (len(data.frame) // 2))
    expect(
        "policy_bounds",
        "imbens_manski",
        lambda: policy_bounds(EvalResults.from_frame(pd.DataFrame(frame))),
    )
    assert set(e0_validity.FUNCTIONS.values()) == {
        "score_interval",
        "paired_difference",
        "contrast_difference",
        "cluster_robust_mean",
        "rank_confidence",
        "policy_bounds",
        "decompose",
        "replicate_mdes",
    }


def test_decompose_and_mde_labels_match_the_e0_glmm_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from evalstats import glmm

    seen: list = []
    monkeypatch.setattr(validity, "check", lambda f, m, n: seen.append((f, m, n)) or [])
    frame = make_results(n_tasks=7).frame
    glmm._check_validity(frame, ("harness",), "jackknife", "plugin", "task")
    assert seen == [
        ("decompose", "parametric_bootstrap", 7),
        ("decompose", "glmm_task_jackknife", 7),
        ("decompose", "glmm_plugin_share", 7),
    ]
    seen.clear()
    glmm._check_validity(frame, (), "box", "percentile", "task")
    assert seen == [("decompose", "parametric_bootstrap", 7)]


def test_the_validity_outputs_are_rebuilt_from_the_banked_coverage(
    tmp_path: Path, monkeypatch
) -> None:
    import json

    cell = {
        "cell": "X",
        "method": "score_interval",
        "interval_type": "cluster_t",
        "nominal": 0.95,
        "replicates": 100,
        "covered": 80,
        "params": {"n_tasks": 10},
    }
    (tmp_path / "coverage.json").write_text(json.dumps({"prereg": "p", "cells": [cell]}))
    monkeypatch.setattr(e0_validity, "VALIDITY", tmp_path / "validity.json")
    monkeypatch.setattr(e0_validity, "HEADLINE", tmp_path / "headline_cells.json")
    assert e0_validity.stale(tmp_path) == [
        tmp_path / "validity.json",
        tmp_path / "headline_cells.json",
    ]
    for path, content in e0_validity.outputs(tmp_path).items():
        path.write_text(content)
    assert e0_validity.stale(tmp_path) == []
    assert "X" in json.loads((tmp_path / "validity.json").read_text())["unsupported"]
    # The committed file in the repository is what the banked coverage gives.
    monkeypatch.undo()
    assert e0_validity.stale() == []


def test_banking_outputs_on_the_frozen_0005_grid(tmp_path: Path) -> None:
    """e0_validity on prereg 0005's frozen grid, with made-up counts (no raw rows are read).

    Every cell passing leaves part 1's list of failing cells as it is and adds 0005's
    headline cells; one failing cell of each kind is listed with its function and design.
    """
    import json
    import shutil

    from noisefloor.preregs import prereg_block

    p5 = e0_validity.REPO / "prereg/0005-e0-glmm-cells.md"
    grid = prereg_block(p5, "e0-grid")
    part1 = tmp_path / "part1"
    part1.mkdir()
    shutil.copy(e0_validity.COVERAGE, part1 / "coverage.json")
    shutil.copy(e0_validity.COVERAGE, tmp_path / "coverage.json")

    def outputs(failing: str | None) -> dict[str, dict[str, object]]:
        cells = [
            {**c, "covered": c["replicates"] * (4 if c["cell"] == failing else 5) // 5}
            for c in grid
        ]
        art = {"prereg": "prereg/0005-e0-glmm-cells.md", "cells": cells}
        (tmp_path / "coverage-0005.json").write_text(json.dumps(art))
        return {p.name: json.loads(t) for p, t in e0_validity.outputs(tmp_path).items()}

    base = {p.name: json.loads(t) for p, t in e0_validity.outputs(part1).items()}
    clean = outputs(None)
    assert clean["validity.json"] == base["validity.json"]
    added = set(prereg_block(p5, "headline").values())
    assert (
        set(clean["headline_cells.json"]["cells"])
        == set(base["headline_cells.json"]["cells"]) | added
    )
    kinds = {(c["method"], c["interval_type"]): c for c in grid}
    for cell in kinds.values():
        listed = outputs(cell["cell"])["validity.json"]["unsupported"]
        assert listed[cell["cell"]]["function"] == e0_validity.FUNCTIONS[cell["method"]]
        assert listed[cell["cell"]]["max_tasks"] == cell["params"]["n_tasks"]
        assert set(listed) == set(base["validity.json"]["unsupported"]) | {cell["cell"]}
