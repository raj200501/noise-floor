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


def test_choose_rule() -> None:
    def comp(planted: float, a: list, b: list) -> dict:
        return {"planted": planted, "estimate": planted, "basic": a, "percentile": b}

    rows = [
        {
            "seed": i,
            "components": {
                "task": comp(2.0, [1.5, 2.5], [1.9, 2.1] if i else [2.1, 2.2]),
                "model:task": comp(0.5, [0.3, 0.7], [0.4, 0.6]),
            },
        }
        for i in range(4)
    ]
    picked = mp.choose_rule(rows, "random")
    assert picked["components"] == ["task", "model:task"]
    # percentile misses task once (min coverage 0.75): basic wins outright
    assert picked["rule"] == "basic" and picked["min_coverage"] == 1.0 and picked["headline"]
    assert picked["min_coverage_by_rule"] == {"basic": 1.0, "percentile": 0.75}
    for r in rows:  # now both rules always cover: the narrower one wins the tie
        r["components"]["task"]["percentile"] = [1.9, 2.1]
    tied = mp.choose_rule(rows, "random")
    assert tied["tied"] == ["basic", "percentile"] and tied["rule"] == "percentile"
    for r in rows:
        r["components"]["task"]["basic"] = [2.1, 2.2]
        r["components"]["task"]["percentile"] = [2.1, 2.2]
    assert not mp.choose_rule(rows, "random")["headline"]
    shares = [{"seed": 1, "components": {"share:task": comp(0.3, [0.2, 0.4], [0.1, 0.2])}}]
    assert mp.choose_rule(shares, "share")["rule"] == "basic"


def test_plugin_shares_and_choices() -> None:
    resid = math.pi**2 / 3
    rows = []
    for i in range(3):
        rows.append(
            {
                "seed": i,
                "components": {
                    "task": {"planted": 2.0, "estimate": 2.0, "basic": [1.0, 3.0]},
                    "model:task": {"planted": 0.5, "estimate": 0.5, "basic": [0.2, 0.9]},
                    "harness": {"planted": 0.0, "estimate": 0.01, "box_basic": [0.0, 0.05]},
                    "share:task": {"planted": 2.0 / (2.51 + resid), "estimate": 0.3},
                    "share:harness": {"planted": 0.0, "estimate": 0.001},
                    "share:residual": {"planted": 0.5, "estimate": 0.5, "percentile": [0.4, 0.6]},
                    "mde:T20:k1": {"planted": 0.3, "estimate": 0.3, "basic": [0.2, 0.4]},
                },
            }
        )
    aug = mp.with_plugin_shares(rows, "basic", "box_basic")
    total = 2.51 + resid
    task = aug[0]["components"]["share:task"]["plugin"]
    assert task == pytest.approx([1.0 / (1.0 + total - 2.0), 3.0 / (3.0 + total - 2.0)])
    assert aug[0]["components"]["share:harness"]["plugin"][0] == 0.0  # covers a planted 0
    assert aug[0]["components"]["share:residual"]["plugin"] == [0.4, 0.6]
    assert "plugin" not in rows[0]["components"]["share:task"]  # input rows untouched
    picked = mp.choices(aug, ["random", "fixed", "share", "mde"])
    assert picked["share"]["rule"] == "plugin" and picked["share"]["min_coverage"] == 1.0
    assert picked["random"]["rule"] == "basic" and picked["mde"]["components"] == ["mde:T20:k1"]
