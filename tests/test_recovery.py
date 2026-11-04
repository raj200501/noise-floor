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
SMALL_PLANTED_2D = rc.Planted(model=(0.3, -0.3))


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
    full = rc.simulate(rc.Design(n_models=2, n_tasks=3, runs_per_cell=16), SMALL_PLANTED_2D, 1)
    assert len(full) == 2 * 3 * 16
    assert (full.groupby(["model", "task"])["harness"].apply(lambda s: (s == "1").sum()) == 8).all()
    with pytest.raises(ValueError, match="positive"):
        rc.Design(n_tasks=0)
    with pytest.raises(ValueError, match="model and day"):
        rc.simulate(rc.Design(), rc.Planted(model=(0.0,)), 1)
    with pytest.raises(ValueError, match="block"):
        rc.simulate(rc.Design(), rc.Planted(block=(0.0,)), 1)
    assert rc.Planted().to_json()["factors"]["harness"] == 0.5


@needs_r
def test_planted_shares() -> None:
    truth = {"task": 2.0, "model:task": 0.5, "harness": 0.0625, "batch": 0.3}
    shares = rc.planted_shares(truth, ["task", "model:task", "harness"])
    assert sum(shares.values()) == pytest.approx(1.0)
    total = 2.5625 + math.pi**2 / 3
    assert shares["share:task"] == pytest.approx(2.0 / total)
    assert shares["share:residual"] == pytest.approx(math.pi**2 / 3 / total)
    assert "share:batch" not in shares  # not analyzed, so not part of the total


def test_recover_and_pilot_rows() -> None:
    settings = rc.AnalysisSettings(n_boot=4)
    row = rc.recover(SMALL, SMALL_PLANTED, settings, 7)
    analyzed = {"task", "model:task", "batch", "model", *rc.TWO_LEVEL, "day"}
    shares = {f"share:{n}" for n in (*analyzed, "residual")}
    # one time-of-day block: block is held constant and left out
    assert row["seed"] == 7 and set(row["components"]) == analyzed | shares
    for comp in row["components"].values():
        assert comp["lo"] is None or comp["lo"] <= comp["hi"]
    assert sum(row["components"][s]["planted"] for s in shares) == pytest.approx(1.0)
    assert sum(row["components"][s]["estimate"] for s in shares) == pytest.approx(1.0)
    pilot = rc.pilot_row(SMALL, SMALL_PLANTED, settings, 7)
    assert set(pilot["components"]["task"]) >= {"percentile", "basic", "bc"}
    assert {"box_percentile", "box_basic", "box_bc"} <= set(pilot["components"]["model"])
    share = pilot["components"]["share:task"]
    assert share["planted"] == row["components"]["share:task"]["planted"]
    assert set(share) >= {"percentile", "basic", "bc", "basic_log"}
    assert all(0.0 <= share[k][0] <= share[k][1] <= 1.0 for k in ("percentile", "basic"))


def test_summaries() -> None:
    rows = [
        {"components": {"a": {"planted": 1.0, "estimate": 1.2, "lo": 0.5, "hi": 1.5}}},
        {"components": {"a": {"planted": 1.0, "estimate": 0.8, "lo": 1.1, "hi": 1.6}}},
        {"components": {"a": {"planted": 1.0, "estimate": 1.0, "lo": None, "hi": None}}},
    ]
    summary = rc.coverage_summary(rows)["a"]
    # the dataset without an interval counts as a miss (prereg 0006)
    assert summary["coverage"] == pytest.approx(1 / 3) and summary["with_interval"] == 2
    assert summary["n"] == 3
    assert summary["bias"] == pytest.approx(0.0)
    empty = rc.coverage_summary(
        [{"components": {"b": {"planted": 1, "estimate": 1, "lo": None, "hi": None}}}]
    )
    assert empty["b"]["coverage"] == 0.0 and math.isnan(empty["b"]["mean_width"])
    pilot = [
        {
            "components": {
                "a": {"planted": 1.0, "estimate": 1.0, "basic": [0.5, 1.5], "box_basic": None}
            }
        },
        {"components": {"a": {"planted": 1.0, "estimate": 2.0, "basic": [1.5, 2.5]}}},
    ]
    out = rc.pilot_summary(pilot)["a"]
    assert out["basic"] == 0.5 and "box_basic" not in out and out["bias"] == pytest.approx(0.5)
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


@needs_r
def test_pilot_command_workers(tmp_path: Path) -> None:
    out = tmp_path / "pilot.jsonl"
    argv = ["pilot", str(out), "990001", "2", "--n-boot", "3", "--nagq", "0", "--tasks", "6"]
    assert rc.main([*argv, "--workers", "2"]) == 0
    assert {json.loads(ln)["seed"] for ln in out.read_text().splitlines()} == {990001, 990002}


def small_config() -> dict:
    planted = rc.Planted(model=(0.3, -0.3), day=(0.0, 0.1), block=(0.0,)).to_json()
    return {
        "design": {"n_models": 2, "n_tasks": 6, "batch_size": 4, "days": 2, "blocks_per_day": 1},
        "planted": planted,
        "settings": {
            "n_boot": 3,
            "nagq": 0,
            "boot_interval": "percentile",
            "level": 0.95,
            "random": ["task", "model:task"],
        },
    }


def write_prereg(path: Path, **blocks: object) -> Path:
    parts = ["# test prereg", ""]
    for name, value in blocks.items():
        parts += [f"```json {name.replace('_', '-')}", json.dumps(value), "```", ""]
    path.write_text("\n".join(parts))
    return path


def test_config_and_blocks(tmp_path: Path) -> None:
    prereg = write_prereg(tmp_path / "p.md", recovery_config=small_config(), g6_seeds=[1, 2])
    design, planted, settings = rc.config_from(rc.prereg_block(prereg, rc.CONFIG_BLOCK))
    assert design.n_tasks == 6 and planted.model == (0.3, -0.3)
    assert settings.random == ("task", "model:task") and settings.n_boot == 3
    with pytest.raises(ValueError, match="no ```json missing``` block"):
        rc.prereg_block(prereg, "missing")
    assert rc.e0_glmm_seed("g", 1) == rc.e0_glmm_seed("g", 1) != rc.e0_glmm_seed("g", 2)


@needs_r
def test_g6_run_and_bank(tmp_path: Path) -> None:
    prereg = write_prereg(tmp_path / "p.md", recovery_config=small_config(), g6_seeds=[11, 12])
    raw = tmp_path / "raw.jsonl"
    assert rc.run_g6(prereg, raw, log=lambda m: None) == 2
    assert rc.run_g6(prereg, raw, log=lambda m: None) == 0  # resumes
    artifact = rc.bank_g6(prereg, raw, tmp_path / "recovery.json")
    assert [d["seed"] for d in artifact["datasets"]] == [11, 12]
    comps = artifact["datasets"][0]["components"]
    assert set(comps) >= {"task", "model:task", "model", "harness"} and "batch" not in comps
    name = "task"
    expected = np.mean(
        [
            d["components"][name]["estimate"] - d["components"][name]["planted"]
            for d in artifact["datasets"]
        ]
    )
    assert artifact["bias"][name] == pytest.approx(expected)
    assert artifact["synthetic"] and artifact["nominal"] == 0.95
    stored = json.loads((tmp_path / "recovery.json").read_text())
    assert stored["prereg"] == str(prereg)
    prereg2 = write_prereg(tmp_path / "q.md", recovery_config=small_config(), g6_seeds=[11, 99])
    with pytest.raises(ValueError, match="not run yet"):
        rc.bank_g6(prereg2, raw, tmp_path / "x.json")


@needs_r
def test_e0_glmm_cells(tmp_path: Path) -> None:
    grid = [
        {
            "cell": f"G-t-{comp}",
            "method": "glmm_component",
            "interval_type": "parametric_bootstrap",
            "nominal": 0.95,
            "replicates": 2,
            "params": {"group": "t", "component": comp, "n_tasks": 6},
        }
        for comp in ("task", "harness", "batch")
    ]
    prereg = write_prereg(tmp_path / "a.md", e0_grid=grid, recovery_config=small_config())
    assert rc.run_e0_glmm(prereg, root=tmp_path, log=lambda m: None) == 2
    assert rc.run_e0_glmm(prereg, root=tmp_path, log=lambda m: None) == 0
    rows = [json.loads(ln) for ln in (tmp_path / "raw" / "G-t-task.jsonl").read_text().splitlines()]
    assert [r["i"] for r in rows] == [0, 1] and rows[0]["truth"] == 2.0
    batch = [
        json.loads(ln) for ln in (tmp_path / "raw" / "G-t-batch.jsonl").read_text().splitlines()
    ]
    assert all(r["lo"] is None and r["truth"] is None for r in batch)  # not fitted


def test_failed_fit_is_recorded(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("lme4 refused")

    monkeypatch.setattr(rc, "analyze", boom)
    settings = rc.AnalysisSettings(n_boot=3, random=("task", "model:task"), mde_designs=((40, 1),))
    row = rc.recover(SMALL, SMALL_PLANTED, settings, 3)
    assert row["converged"] is False and "lme4 refused" in row["error"]
    assert row["components"]["task"] == {"planted": 2.0, "estimate": None, "lo": None, "hi": None}
    assert "mde:T40:k1" in row["components"] and "harness" in row["components"]
