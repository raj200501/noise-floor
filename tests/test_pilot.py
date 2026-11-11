import itertools
from pathlib import Path
from typing import Any

import pytest

from noisefloor import pilot
from noisefloor.pilot import PILOT_SEEDS, PLANS, plan_runs


def test_plan_is_deterministic_crossed_and_disjoint_from_confirmatory() -> None:
    a, b = plan_runs("p1"), plan_runs("p1")
    assert a == b and len({s.run_id for s in a}) == len(a)
    plan = PLANS["p1"]
    assert len(a) == len(plan.models) * len(plan.harnesses) * 5 * len(plan.difficulties)
    assert all(s.task_seed in PILOT_SEEDS and s.phase == "pilot" for s in a)
    models_in_order = [s.model for s in a]
    # each model's runs are contiguous (one server configuration per batch)
    assert sum(1 for x, y in itertools.pairwise(models_in_order) if x != y) == len(plan.models) - 1


def test_main_dry_run_and_execute(
    capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    assert pilot.main(["p1", "--dry-run"]) == 0
    assert "200 runs" in capsys.readouterr().out
    seen: dict[str, Any] = {}

    def fake_execute(specs: list[Any], **kwargs: Any) -> int:
        seen.update(kwargs, n=len(specs))
        return 0

    monkeypatch.setattr(pilot, "execute", fake_execute)
    assert pilot.main(["p1"]) == 0
    assert seen["n"] == 200 and Path(seen["artifact"]).name == "p1.jsonl"
    seen["log"]("message")


def test_summary_rows() -> None:
    rows = [
        {"model": "b", "outcome": 1.0, "status": "ok", "wall_s": 10.0},
        {"model": "b", "outcome": 0.0, "status": "timeout", "wall_s": 300.0},
        {"model": "a", "outcome": 0.0, "status": "ok", "wall_s": 20.0},
    ]
    out = pilot.summary(rows)
    assert [r["model"] for r in out] == ["a", "b"]
    assert out[1] == {
        "model": "b",
        "runs": 2,
        "success": 0.5,
        "timeouts": 1,
        "median_wall_s": 155.0,
    }
