"""Summaries of the exploratory method pilots (synthetic rows only)."""

import json
import math
from pathlib import Path

import pytest

from noisefloor import method_pilots as mp


def row(seed: int, harness_box: list | None, task_ci: list | None) -> dict:
    return {
        "seed": seed,
        "components": {
            "harness": {"planted": 0.0625, "estimate": 0.0625, "box_basic": harness_box},
            "task": {"planted": 2.0, "estimate": 1.9, "basic": task_ci},
        },
    }


ROWS = [
    row(2, [0.04, 0.09], [1.5, 2.5]),
    row(1, [0.07, 0.10], [1.0, 1.8]),
    row(3, None, [1.9, 2.1]),
]


def test_component_summary_counts_missing_as_miss() -> None:
    out = mp.component_summary(ROWS, "harness")
    assert out["n"] == 3 and out["coverage"]["box_basic"] == pytest.approx(1 / 3)
    assert out["missing"]["box_basic"] == 1
    assert out["above_zero"]["box_basic"] == pytest.approx(2 / 3)
    task = mp.component_summary(ROWS, "task")
    assert task["coverage"]["basic"] == pytest.approx(2 / 3)
    assert task["bias"] == pytest.approx(-0.1)


def test_two_level_pool_and_sd_ratio() -> None:
    pooled = mp.two_level_pooled(ROWS)
    assert pooled["factors"] == ["harness"] and pooled["n"] == 3
    ratio = mp.sd_ratio(ROWS, "harness", "box_basic")
    assert ratio is not None and ratio["planted_beta"] == pytest.approx(0.5)
    assert ratio["n_intervals"] == 2 and ratio["empirical_sd"] == 0.0
    half = [
        (2 * math.sqrt(hi) - 2 * math.sqrt(lo)) / (2 * mp.Z975)
        for lo, hi in ((0.04, 0.09), (0.07, 0.10))
    ]
    assert ratio["implied_sd"] == pytest.approx(sum(half) / 2)
    assert mp.sd_ratio(ROWS, "task", "box_basic") is None  # not a two-level factor box
    small = [{"seed": 1, "components": {"harness": {"planted": 0.001, "estimate": 0.0}}}]
    assert mp.sd_ratio(small, "harness", "box_basic") is None


def test_build_and_check(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    (tmp_path / "a.jsonl").write_text("".join(json.dumps(r) + "\n" for r in ROWS[:2]))
    (tmp_path / "b.jsonl").write_text(json.dumps(ROWS[2]) + "\n")
    manifest = {
        "pilots": [{"label": "a", "file": "a.jsonl"}, {"label": "b", "file": "b.jsonl"}],
        "pools": [{"label": "ab", "pilots": ["a", "b"]}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    out = mp.build(tmp_path)
    assert out["synthetic"] and out["exploratory"]
    assert out["pilots"]["a"]["seeds"] == [1, 2] and out["pools"]["ab"]["datasets"] == 3
    assert mp.main(["--check", "--dir", str(tmp_path)]) == 1
    assert mp.main(["--dir", str(tmp_path)]) == 0
    assert mp.main(["--check", "--dir", str(tmp_path)]) == 0
    assert "summary current" in capsys.readouterr().out


def test_banked_summary_is_current() -> None:
    assert mp.main(["--check"]) == 0
