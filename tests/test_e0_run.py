import json
from pathlib import Path

import pytest

from noisefloor import e0_crosscheck, e0_run
from noisefloor.e0 import Cell

GRID = [
    {
        "cell": "T-a",
        "method": "score_interval",
        "interval_type": "cluster_t",
        "nominal": 0.9,
        "replicates": 6,
        "params": {"n_tasks": 15, "runs": 1, "task_sd": 1.0},
    },
    {
        "cell": "T-r",
        "method": "rank_simultaneous",
        "interval_type": "bootstrap_max_t",
        "nominal": 0.9,
        "replicates": 3,
        "params": {
            "n_tasks": 15,
            "runs": 1,
            "task_sd": 1.0,
            "effects": [0.5, 0.0, -0.5],
            "n_boot": 100,
        },
    },
]


def write_prereg(path: Path, grid: list[dict[str, object]], extra: str = "") -> Path:
    path.write_text(f"{extra}# grid\n\n```json e0-grid\n{json.dumps(grid)}\n```\n")
    return path


def test_run_bank_and_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(e0_run, "REPO", tmp_path)
    prereg = write_prereg(tmp_path / "0003-grid.md", GRID)
    cells = e0_run.grid_from_prereg(prereg)
    assert [c.cell for c in cells] == ["T-a", "T-r"]
    root = tmp_path / "e0"
    assert e0_run.run_cell(cells[0], 1, root, log=lambda m: None) == 6
    assert e0_run.run_cell(cells[0], 1, root, log=lambda m: None) == 0
    assert e0_run.run_cell(cells[1], 2, root, log=lambda m: None) == 3
    raw = e0_run.read_raw(cells[1], root)
    assert sorted(raw) == [0, 1, 2] and all(r.hit is not None for r in raw.values())
    amend = write_prereg(
        tmp_path / "0004-more.md", [{**GRID[0], "cell": "T-b", "replicates": 2}], "amends: 0003\n"
    )
    e0_run.run_cell(e0_run.grid_from_prereg(amend)[0], 1, root, log=lambda m: None)
    art = e0_run.bank([prereg, amend], root)
    assert art["prereg"] == "0003-grid.md" and art["amendments"] == ["0004-more.md"]
    by_id = {c["cell"]: c for c in art["cells"]}
    assert by_id["T-a"]["replicates"] == 6 and by_id["T-b"]["replicates"] == 2
    assert 0 <= by_id["T-r"]["covered"] <= 3
    text = (root / "coverage.json").read_text()
    assert "NaN" not in text and json.loads(text)["confirmatory"] is True
    with pytest.raises(ValueError, match="repeat"):
        e0_run.bank([prereg, prereg], root)


def test_grid_errors_and_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "none.md").write_text("no grid here")
    with pytest.raises(ValueError, match="e0-grid"):
        e0_run.grid_from_prereg(tmp_path / "none.md")
    monkeypatch.setattr(e0_run, "REPO", tmp_path)
    monkeypatch.setattr(e0_run, "E0_DIR", tmp_path / "e0")
    prereg = write_prereg(tmp_path / "0003-grid.md", GRID[:1])
    (tmp_path / "e0").mkdir()
    real_run = e0_run.run_cell

    def run_here(
        cell: Cell, workers: int = 1, root: Path = tmp_path / "e0", log: object = print
    ) -> int:
        return real_run(cell, 1, tmp_path / "e0", log=lambda m: None)

    monkeypatch.setattr(e0_run, "run_cell", run_here)
    assert e0_run.main(["run", str(prereg), "--cells", "T-a", "--workers", "1"]) == 0
    assert e0_run.main(["run", str(prereg), "--cells", "nothing"]) == 0
    grouped = {
        "cell": "G-x",
        "method": "glmm_component",
        "interval_type": "parametric_bootstrap",
        "nominal": 0.95,
        "replicates": 2,
        "params": {"group": "e1", "component": "task", "n_tasks": 30},
    }
    both = write_prereg(tmp_path / "0004-both.md", [grouped, GRID[0]])
    capsys.readouterr()
    assert e0_run.main(["run", str(both)]) == 0
    assert (
        "G-x: skipped; run it with python -m noisefloor.recovery e0-run" in capsys.readouterr().out
    )
    assert not (tmp_path / "e0" / "raw" / "G-x.jsonl").exists()
    monkeypatch.setattr(e0_run, "bank", lambda paths, root=None: {"cells": [1]})
    assert e0_run.main(["bank", str(prereg)]) == 0
    assert "banked 1 cells" in capsys.readouterr().out
    assert e0_run._clean({"x": [float("nan"), 1.0]}) == {"x": [None, 1.0]}


def test_crosscheck_library_checks_agree() -> None:
    checks = e0_crosscheck.library_checks()
    assert len(checks) >= 10
    assert all(c["abs_diff"] <= c["tolerance"] for c in checks)
    assert any("statsmodels" in c["reference_impl"] for c in checks)


def test_crosscheck_main_writes_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(e0_crosscheck, "OUT", tmp_path / "cc.json")
    monkeypatch.setattr(
        e0_crosscheck,
        "glmm_checks",
        lambda: [e0_crosscheck.check("bad", 1.0, 2.0, 0.1, "x", note="forced")],
    )
    assert e0_crosscheck.main() == 1
    data = json.loads((tmp_path / "cc.json").read_text())
    assert data["checks"][-1]["name"] == "bad" and "OUTSIDE" in capsys.readouterr().out


@pytest.mark.skipif(not e0_crosscheck.lme4_available(), reason="R and lme4 unavailable")
def test_crosscheck_glmm_checks_agree() -> None:
    checks = e0_crosscheck.glmm_checks()
    assert len(checks) == 5 and all(c["abs_diff"] <= c["tolerance"] for c in checks)


def test_lower_priority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(e0_run.os, "nice", lambda inc: calls.append(inc) or 0)
    group = tmp_path / "autogroup"
    e0_run.lower_priority(group)
    assert calls == [10] and group.read_text() == "19\n"
    e0_run.lower_priority(tmp_path / "missing" / "autogroup")  # no such file: ignored
    assert calls == [10, 10]
