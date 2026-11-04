import json
import math
from pathlib import Path

import numpy as np
import pytest

from evalstats.glmm import lme4_available
from noisefloor import recovery as rc

needs_r = pytest.mark.skipif(not lme4_available(), reason="R, lme4, or rpy2 not available")
SMALL = rc.Design(n_models=2, n_tasks=6, batch_size=4, days=2, blocks_per_day=1)
SMALL_PLANTED = rc.Planted(model=(0.3, -0.3), day=(0.0, 0.1), block=(0.0,))


def test_half_fraction_blocks_are_balanced() -> None:
    first, second = rc.half_fraction_blocks()
    assert len(first) == len(second) == 8
    assert len(set(first) | set(second)) == 16
    for half in (first, second):
        assert np.array(half).sum(axis=0).tolist() == [4] * 5


def test_fixed_contribution() -> None:
    assert rc.fixed_contribution((0.0, 0.5), (0.5, 0.5)) == pytest.approx(0.0625)
    assert rc.fixed_contribution((1.0, 1.0, 1.0), (0.2, 0.3, 0.5)) == pytest.approx(0.0)
    assert rc.fixed_contribution((0.6, 0.2, -0.2, -0.6), [0.25] * 4) == pytest.approx(0.2)


def test_simulate_structure_and_determinism() -> None:
    data = rc.simulate(rc.Design(), rc.Planted(), 5)
    assert len(data) == 4 * 40 * 8 and data["synthetic"].all()
    assert data.equals(rc.simulate(rc.Design(), rc.Planted(), 5))
    assert not data.equals(rc.simulate(rc.Design(), rc.Planted(), 6))
    per_batch = data.groupby("batch")
    assert per_batch.size().max() <= 10
    assert (per_batch["model"].nunique() == 1).all()
    assert (per_batch["concurrency"].nunique() == 1).all()
    assert (per_batch["day"].nunique() == 1).all() and (per_batch["block"].nunique() == 1).all()
    cells = data.groupby(["model", "task"])
    for name in rc.TWO_LEVEL:
        assert (cells[name].apply(lambda s: (s == "1").sum()) == 4).all()
    truth = rc.planted_values(rc.Design(), rc.Planted(), data)
    assert truth["harness"] == pytest.approx(0.0625) and truth["model"] == pytest.approx(0.2)
    assert truth["task"] == 2.0 and truth["concurrency"] == 0.0


def test_design_and_planted_validation() -> None:
    with pytest.raises(ValueError, match="runs_per_cell"):
        rc.Design(runs_per_cell=4)
    with pytest.raises(ValueError, match="positive"):
        rc.Design(n_tasks=0)
    with pytest.raises(ValueError, match="model and day"):
        rc.simulate(rc.Design(), rc.Planted(model=(0.0,)), 1)
    with pytest.raises(ValueError, match="block"):
        rc.simulate(rc.Design(), rc.Planted(block=(0.0,)), 1)
    assert rc.Planted().to_json()["factors"]["harness"] == 0.5


@needs_r
def test_recover_and_pilot_rows() -> None:
    settings = rc.AnalysisSettings(n_boot=4)
    row = rc.recover(SMALL, SMALL_PLANTED, settings, 7)
    assert row["seed"] == 7 and set(row["components"]) == {
        "task",
        "model:task",
        "batch",
        "model",
        *rc.TWO_LEVEL,
        "day",
    }  # one time-of-day block: block is held constant and left out
    for comp in row["components"].values():
        assert comp["lo"] is None or comp["lo"] <= comp["hi"]
    pilot = rc.pilot_row(SMALL, SMALL_PLANTED, settings, 7)
    assert set(pilot["components"]["task"]) >= {"percentile", "basic", "bc"}
    assert "box" in pilot["components"]["model"]


def test_summaries() -> None:
    rows = [
        {"components": {"a": {"planted": 1.0, "estimate": 1.2, "lo": 0.5, "hi": 1.5}}},
        {"components": {"a": {"planted": 1.0, "estimate": 0.8, "lo": 1.1, "hi": 1.6}}},
        {"components": {"a": {"planted": 1.0, "estimate": 1.0, "lo": None, "hi": None}}},
    ]
    summary = rc.coverage_summary(rows)["a"]
    assert summary["coverage"] == 0.5 and summary["with_interval"] == 2 and summary["n"] == 3
    assert summary["bias"] == pytest.approx(0.0)
    empty = rc.coverage_summary(
        [{"components": {"b": {"planted": 1, "estimate": 1, "lo": None, "hi": None}}}]
    )
    assert math.isnan(empty["b"]["coverage"])
    pilot = [
        {"components": {"a": {"planted": 1.0, "estimate": 1.0, "basic": [0.5, 1.5], "box": None}}},
        {"components": {"a": {"planted": 1.0, "estimate": 2.0, "basic": [1.5, 2.5]}}},
    ]
    out = rc.pilot_summary(pilot)["a"]
    assert out["basic"] == 0.5 and "box" not in out and out["bias"] == pytest.approx(0.5)
    assert json.loads(rc.dumps({"x": math.nan, "y": [math.inf, 1.0]})) == {
        "x": None,
        "y": [None, 1.0],
    }


@needs_r
def test_pilot_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(rc, "Design", lambda: SMALL)
    monkeypatch.setattr(rc, "Planted", lambda: SMALL_PLANTED)
    out = tmp_path / "pilot.jsonl"
    assert rc.main(["pilot", str(out), "990000", "1", "--n-boot", "3"]) == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary["task"]["n"] == 1
    assert rc.main(["pilot", str(out), "990000", "1", "--n-boot", "3"]) == 0  # resumes
    assert len(out.read_text().splitlines()) == 1
    with pytest.raises(SystemExit, match="PILOT_SEEDS"):
        rc.main(["pilot", str(out), "5", "1"])
