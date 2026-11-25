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


@needs_r
def test_recover_and_pilot_rows() -> None:
    settings = rc.AnalysisSettings(n_boot=4)
    row = rc.recover(SMALL, SMALL_PLANTED, settings, 7)
    analyzed = {"task", "model:task", "batch", "model", *rc.TWO_LEVEL, "day"}
    shares = {f"share:{n}" for n in (*analyzed, "residual")}
    local = {"score:m0", "score:m1", "harness:m0", "harness:m1", "hdiff:m0-m1", "pair:m0-m1"}
    # one time-of-day block: block is held constant and left out
    assert row["seed"] == 7 and set(row["components"]) == analyzed | shares | local
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
    monkeypatch.setattr(rc, "Design", lambda **kw: SMALL)
    monkeypatch.setattr(rc, "Planted", lambda **kw: SMALL_PLANTED)
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


def lenient(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bank rows stamped in a working tree under edit (the dirty check has its own test)."""
    import functools

    from noisefloor import provenance

    lax = functools.partial(provenance.single_environment, allow_dirty=True)
    monkeypatch.setattr(rc, "single_environment", lax)


@needs_r
@pytest.mark.usefixtures("unfrozen_ok")
def test_g6_run_and_bank(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    lenient(monkeypatch)
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
    env = artifact["environment"]
    assert len(env["env"]) == 16 and env["key"]["lme4"] and len(env["commits"]) == 1
    assert artifact["prereg_sha256"] == "0" * 64
    stored = json.loads((tmp_path / "recovery.json").read_text())
    assert stored["prereg"] == str(prereg)
    prereg2 = write_prereg(tmp_path / "q.md", recovery_config=small_config(), g6_seeds=[11, 99])
    with pytest.raises(ValueError, match="not run yet"):
        rc.bank_g6(prereg2, raw, tmp_path / "x.json")


@needs_r
@pytest.mark.usefixtures("unfrozen_ok")
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
    assert rows[0]["stamp"]["env"] in json.loads(
        (tmp_path / "raw" / "environments.json").read_text()
    )
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
    assert row["components"]["score:m0"]["lo"] is not None  # needs no GLMM fit


def test_local_intervals_and_planted_success() -> None:
    data = rc.simulate(SMALL, SMALL_PLANTED, 5)
    success = rc.planted_success(data, SMALL_PLANTED)
    eta = rc.run_fixed_eta(data, SMALL_PLANTED)
    zero = rc.Planted(
        model=(0.0, 0.0), factors=dict.fromkeys(rc.TWO_LEVEL, 0.0), day=(0.0, 0.0), block=(0.0,)
    )
    assert rc.planted_success(data, zero) == pytest.approx(np.full(len(data), 0.5))
    assert np.all((success > 0) & (success < 1)) and np.corrcoef(eta, success)[0, 1] > 0.99
    out = rc.local_intervals(data, SMALL_PLANTED, rc.AnalysisSettings())
    assert set(out) == {
        "score:m0",
        "score:m1",
        "harness:m0",
        "harness:m1",
        "hdiff:m0-m1",
        "pair:m0-m1",
    }
    for entry in out.values():
        assert entry["lo"] <= entry["estimate"] <= entry["hi"]
    m0 = data[data["model"] == "m0"]
    assert out["score:m0"]["estimate"] == pytest.approx(m0.groupby("task")["outcome"].mean().mean())
    assert out["score:m0"]["planted"] == pytest.approx(
        success[(data["model"] == "m0").to_numpy()].mean()
    )
    assert out["harness:m0"]["planted"] > 0  # planted harness effect 0.5 on level 1
    assert out["pair:m0-m1"]["planted"] > 0  # model effects 0.3 and -0.3


@pytest.mark.usefixtures("unfrozen_ok")
def test_e0_local_group(tmp_path: Path) -> None:
    grid = [
        {
            "cell": f"L-t-{comp.replace(':', '-')}",
            "method": "score_interval" if comp.startswith("score") else "paired_difference",
            "interval_type": "cluster_t",
            "nominal": 0.95,
            "replicates": 3,
            "params": {"group": "loc", "analysis": "local", "component": comp, "n_tasks": 6},
        }
        for comp in ("score:m0", "harness:m1", "pair:m0-m1")
    ]
    prereg = write_prereg(tmp_path / "a.md", e0_grid=grid, recovery_config=small_config())
    assert rc.run_e0_glmm(prereg, root=tmp_path, log=lambda m: None) == 3
    rows = [
        json.loads(ln) for ln in (tmp_path / "raw" / "L-t-score-m0.jsonl").read_text().splitlines()
    ]
    assert [r["i"] for r in rows] == [0, 1, 2] and all(r["lo"] <= r["hi"] for r in rows)
    mixed = [grid[0], {**grid[1], "params": {**grid[1]["params"], "analysis": "full"}}]
    bad = write_prereg(tmp_path / "b.md", e0_grid=mixed, recovery_config=small_config())
    with pytest.raises(ValueError, match="mixes analyses"):
        rc.run_e0_glmm(bad, root=tmp_path, log=lambda m: None)


def test_simulated_batches_spread_like_the_plan() -> None:
    import collections

    design = rc.Design(n_models=3, n_tasks=30, runs_per_cell=16)
    data = rc.simulate(design, rc.Planted(model=(0.6, 0.0, -0.6), batch_var=0.0), 11)
    per_batch = data.groupby("batch").agg(
        model=("model", "first"),
        conc=("concurrency", "first"),
        day=("day", "first"),
        block=("block", "first"),
        n_models=("model", "nunique"),
        n_conc=("concurrency", "nunique"),
    )
    assert (per_batch["n_models"] == 1).all() and (per_batch["n_conc"] == 1).all()
    for _, group in per_batch.groupby(["model", "conc"]):
        slots = collections.Counter(zip(group["day"], group["block"], strict=True))
        assert len(slots) == 6 and max(slots.values()) - min(slots.values()) <= 1


@needs_r
def test_planted_interactions_and_jackknife_pilot_row() -> None:
    het = rc.Planted(model=(0.3, -0.3), day=(0.0, 0.1), block=(0.0,), interactions={"harness": 1.0})
    plain = rc.simulate(SMALL, SMALL_PLANTED, 4)
    varied = rc.simulate(SMALL, het, 4)
    assert len(plain) == len(varied) and not plain["outcome"].equals(varied["outcome"])
    sd = rc.planted_success(varied, het)
    assert np.all((sd > 0) & (sd < 1))
    assert het.to_json()["interactions"] == {"harness": 1.0}
    block = small_config()
    block["planted"] = het.to_json()
    block["settings"]["fixed_interval"] = "jackknife"
    _, planted, settings = rc.config_from(block)
    assert planted.interactions == {"harness": 1.0} and settings.fixed_interval == "jackknife"
    row = rc.pilot_row(
        SMALL, SMALL_PLANTED, rc.AnalysisSettings(n_boot=3, fixed_interval="jackknife"), 4
    )
    jk = row["components"]["harness"]["jackknife"]
    assert jk is not None and jk[0] <= row["components"]["harness"]["estimate"] <= jk[1]


def test_local_intervals_carry_harness_differences() -> None:
    design = rc.Design(n_models=3, n_tasks=12, runs_per_cell=16)
    planted = rc.Planted(model=(0.6, 0.0, -0.6))
    data = rc.simulate(design, planted, 11)
    out = rc.local_intervals(data, planted, rc.AnalysisSettings())
    names = sorted(k for k in out if k.startswith("hdiff:"))
    assert names == ["hdiff:m0-m1", "hdiff:m0-m2", "hdiff:m1-m2"]
    for name in names:
        a, b = name.removeprefix("hdiff:").split("-")
        entry = out[name]
        planted_diff = out[f"harness:{a}"]["planted"] - out[f"harness:{b}"]["planted"]
        assert entry["planted"] == pytest.approx(planted_diff)
        assert entry["lo"] <= entry["estimate"] <= entry["hi"]


@pytest.mark.usefixtures("unfrozen_ok")
def test_e0_groups_take_planted_overrides(tmp_path: Path) -> None:
    def cells(group: str, planted: dict[str, object]) -> list[dict[str, object]]:
        return [
            {
                "cell": f"L-{group}-score-m0",
                "method": "score_interval",
                "interval_type": "cluster_t",
                "nominal": 0.95,
                "replicates": 2,
                "params": {
                    "group": group,
                    "analysis": "local",
                    "component": "score:m0",
                    "n_tasks": 6,
                    "planted": planted,
                },
            }
        ]

    grid = cells("low", {}) + cells("high", {"intercept": 1.5})
    prereg = write_prereg(tmp_path / "a.md", e0_grid=grid, recovery_config=small_config())
    assert rc.run_e0_glmm(prereg, root=tmp_path, log=lambda m: None) == 4
    truth = {
        g: json.loads((tmp_path / "raw" / f"L-{g}-score-m0.jsonl").read_text().splitlines()[0])[
            "truth"
        ]
        for g in ("low", "high")
    }
    assert truth["high"] > truth["low"] + 0.1
    split = [*cells("x", {}), {**cells("x", {"intercept": 1.0})[0], "cell": "L-x-other"}]
    bad = write_prereg(tmp_path / "b.md", e0_grid=split, recovery_config=small_config())
    with pytest.raises(ValueError, match="disagree on planted overrides"):
        rc.run_e0_glmm(bad, root=tmp_path, log=lambda m: None)
    odd = write_prereg(
        tmp_path / "c.md", e0_grid=cells("y", {"slope": 1.0}), recovery_config=small_config()
    )
    with pytest.raises(ValueError, match="unknown planted values"):
        rc.run_e0_glmm(odd, root=tmp_path, log=lambda m: None)


REPS = ((0, 1, 1, 0, 0), (0, 1, 1, 1, 1))


def test_simulated_batches_hold_replicate_runs_as_the_plan_does() -> None:
    import collections

    from noisefloor import e1

    design = rc.Design(n_models=3, n_tasks=10, runs_per_cell=16, days=4, replicates=REPS)
    planted = rc.Planted(model=(0.6, 0.0, -0.6), day=(0.0, 0.1, -0.1, 0.05), batch_var=0.0)
    data = rc.simulate(design, planted, 3)
    assert len(data) == 3 * 10 * 16  # replicate runs are dealt into batches, then left out
    sizes = data.groupby("batch").size()
    assert sizes.max() <= 10 and (sizes < 10).sum() > 0  # replicates took some seats
    per_group = data.groupby(["model", "concurrency"])["batch"].nunique()
    assert set(per_group) == {9}  # (10 tasks x 8 runs + 10 replicates) / 10 per batch
    plan_design = e1.E1Design(
        models=("granite-4.0-micro", "qwen3-1.7b", "qwen3-4b"),
        families=("config", "pointer"),
        tasks_per_family=5,
        runs_per_cell=16,
        days=4,
        replicates=REPS,
    )
    batches = e1.planned_batches(e1.local_plan(plan_design))
    planned = collections.Counter((s[0].model, s[0].concurrency) for s in batches.values())
    assert set(planned.values()) == {9}  # the plan deals the same number per group
    with pytest.raises(ValueError, match="half fraction"):
        rc.Design(runs_per_cell=16, replicates=((1, 0, 0, 0, 0),))
    with pytest.raises(ValueError, match="whole half fraction"):
        rc.Design(runs_per_cell=8, replicates=REPS)
    order: list = []
    dealt = rc.deal_batches(
        {"b": [1, 2, 3], "a": [4, 5]}, 2, 2, shuffle=lambda xs: None, randrange=lambda n: 1
    )
    order = [(slot, chunk) for slot, chunk in dealt]
    assert order == [(1, [4, 5]), (1, [1, 2]), (0, [3])]


@needs_r
def test_g6_analysis_equals_the_e1_analysis() -> None:
    """recover's analysis and noisefloor.analysis.e1 agree on one dataset (review brief-14)."""
    from noisefloor.analysis import e1 as analysis

    design = rc.Design(n_models=3, n_tasks=12, runs_per_cell=16, days=2, replicates=REPS)
    planted = rc.Planted(model=(0.6, 0.0, -0.6), day=(0.0, 0.1), task_var=1.0, batch_var=0.0)
    settings = rc.AnalysisSettings(
        n_boot=4,
        nagq=0,
        boot_interval="basic_log",
        random=("task", "model:task"),
        box_interval="basic",
        share_interval="plugin",
        fixed_interval="jackknife",
        mde_designs=((20, 1), (40, 4)),
        mde_interval="bc",
    )
    seed = 31
    data = rc.simulate(design, planted, seed)
    models = ("granite-4.0-micro", "qwen3-1.7b", "qwen3-4b")
    block = {
        "random": list(settings.random),
        "fixed": list(rc.FIXED_TERMS),
        "level": settings.level,
        "n_boot": settings.n_boot,
        "nagq": settings.nagq,
        "boot_interval": settings.boot_interval,
        "box_interval": settings.box_interval,
        "share_interval": settings.share_interval,
        "fixed_interval": settings.fixed_interval,
        "score_interval": {"local": "cluster_t"},
        "contrast_interval": {
            "harness-local": "cluster_t",
            "harness-claude": "mover_wilson",
            "harness-diff-local": "cluster_t",
            "models-local": "cluster_t",
            "tiers-claude": "mover_wilson",
            "models-claude": "mover_wilson",
        },
        "seed": seed,
        "exclude_status": ["infra_error"],
        "block_hours": [[0, 12], [12, 24]],
        "mde": {
            "designs": [list(d) for d in settings.mde_designs],
            "reference": {
                "harness": "minimal",
                "timeout": "300.0",
                "steps": "15",
                "temperature": "0.0",
                "concurrency": "1",
            },
            "averaged": ["day", "block"],
            "interval": settings.mde_interval,
        },
    }
    records = rc.e1_records(data, models)
    planned = {r["run_id"]: "e1" for r in records}
    rows = analysis.runs_table(
        records, prereg="synthetic", planned=planned, block_hours=block["block_hours"]
    )
    frame = analysis.analysis_frame(rows, block["exclude_status"])
    theirs = analysis.decomposition(frame, block)
    ours = rc.analyze(data, settings, seed)
    close = {"rel": 1e-3, "abs": 1e-6}
    for mine in ours.components:
        other = theirs.component(mine.name)
        for key in ("variance", "lo", "hi", "share", "share_lo", "share_hi"):
            a, b = getattr(mine, key), getattr(other, key)
            assert (a is None) == (b is None), (mine.name, key)
            if a is not None:
                assert a == pytest.approx(b, **close), (mine.name, key)
    mde_rows = analysis.mde_tables(theirs, frame, block)["rows"]
    found = rc.mde_components(ours, data, design, settings.mde_designs, settings.level)
    for row in mde_rows:
        entry = found[f"mde:T{row['tasks']}:k{row['runs']}"]
        assert row["mde"] == pytest.approx(entry["estimate"], **close)
        ci = rc.boot_coef_interval(
            entry["estimate"], np.asarray(entry["replicates"]), settings.level, "bc"
        )
        assert ci is not None
        assert (row["lo"], row["hi"]) == pytest.approx((max(0.0, ci[0]), ci[1]), **close)
    local = rc.local_intervals(data, planted, settings)
    names = {f"m{i}": m for i, m in enumerate(models)}
    scores = {r["model"]: r for r in analysis.scores_table(rows, block)}
    contrasts = analysis.contrasts_table(rows, block)
    harness = {r["contrast"]: r for r in contrasts["harness"]}
    diffs = {r["contrast"]: r for r in contrasts["harness_differences"]}
    pairs = {r["contrast"]: r for r in contrasts["models"]}
    for key, entry in local.items():
        kind, who = key.split(":")
        if kind == "score":
            row = scores[names[who]]
            got = (row["score"], row["lo"], row["hi"])
        elif kind == "harness":
            row = harness[f"harness:{names[who]}"]
            got = (row["difference"], row["lo"], row["hi"])
        else:
            a, b = (names[w] for w in who.split("-"))
            table = diffs if kind == "hdiff" else pairs
            label = f"harness-difference:{a}-{b}" if kind == "hdiff" else f"model:{a}-{b}"
            row = table[label]
            got = (row["difference"], row["lo"], row["hi"])
        assert got == pytest.approx((entry["estimate"], entry["lo"], entry["hi"]), **close), key


def test_task_truth_includes_half_of_each_task_by_factor_variance() -> None:
    design = rc.Design(n_models=2, n_tasks=4, runs_per_cell=16, days=2, blocks_per_day=1)
    planted = rc.Planted(
        model=(0.3, -0.3),
        day=(0.0, 0.1),
        block=(0.0,),
        task_var=2.0,
        interactions={"harness": 0.2, "steps": 0.4},
    )
    data = rc.simulate(design, planted, 2)
    assert rc.planted_values(design, planted, data)["task"] == pytest.approx(2.0 + 0.3)
    faster = rc.Planted(model=(0.3, -0.3), model_factors={"harness": (0.0, 1.0)})
    eta = rc.run_fixed_eta(
        rc.simulate(rc.Design(n_models=2, n_tasks=2, runs_per_cell=16), faster, 3), faster
    )
    assert eta.max() > eta.min()
    with pytest.raises(ValueError, match="one value per model"):
        rc.simulate(
            rc.Design(n_models=2),
            rc.Planted(model=(0.0, 0.0), model_factors={"harness": (1.0,)}),
            1,
        )


def test_the_mde_truth_uses_the_task_truth_of_the_other_components() -> None:
    import dataclasses

    design = rc.Design()
    with_terms = rc.Planted(interactions={"harness": 0.25, "steps": 0.25, "timeout": 0.25})
    data = rc.simulate(design, with_terms, 3)
    truth = rc.planted_values(design, with_terms, data)["task"]
    assert truth == pytest.approx(with_terms.task_var + 0.375)
    folded = dataclasses.replace(with_terms, task_var=truth, interactions={})
    designs = [(20, 1), (80, 3)]
    assert rc.planted_mde(design, with_terms, data, designs) == pytest.approx(
        rc.planted_mde(design, folded, data, designs)
    )
    plain = dataclasses.replace(with_terms, interactions={})
    assert rc.planted_mde(design, with_terms, data, designs)[(20, 1)] != pytest.approx(
        rc.planted_mde(design, plain, data, designs)[(20, 1)], rel=1e-4
    )
